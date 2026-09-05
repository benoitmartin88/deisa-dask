# =============================================================================
# Copyright (C) 2026 Commissariat a l'energie atomique et aux energies alternatives (CEA)
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the names of CEA, nor the names of the contributors may be used to
#   endorse or promote products derived from this software without specific
#   prior written  permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# =============================================================================
import asyncio
import logging
import pickle
import sys
import uuid
import zlib
from collections import defaultdict, deque
from numbers import Number
from typing import Any, Deque, Dict, Final, Iterator, List, Optional, Tuple, Union

import numpy as np
from deisa.core import IBridge, ICommunicator, validate_arrays_metadata
from distributed import Client, Event, Queue
from distributed.protocol import to_serialize
from distributed.utils_comm import scatter_to_workers
from tlz import valmap

from dask.tokenize import tokenize
from deisa.dask.branch import BranchSpec
from deisa.dask.constants import CLIENT_KEY, FEEDBACK_QUEUE_PREFIX, KEY_PREFIX, WAIT_FOR_EXECUTE_CB_EVENT
from deisa.dask.handshake import Handshake
from deisa.dask.utils import get_client

logger = logging.getLogger(__name__)

_COMM_NULL: Final[None] = None
try:
    from mpi4py import MPI

    _UNDEFINED = MPI.UNDEFINED
except ImportError:
    _UNDEFINED = 2147483647


def _extract_chunk_axis_from_hint(hint: Optional[Dict]) -> Optional[Tuple[int, ...]]:
    """Pull the chunk_func's reduction-axes tuple out of a task hint.

    Dask stores the axes being reduced in ``chunk_kwargs['axis']`` (either
    a single int for one axis, or a tuple for several). For our precompute
    topic event we need to ship this to the Deisa side so the combine's
    output shape and agg ``axis`` are computed correctly. Returns
    ``None`` when the hint is missing or has no ``axis`` kwarg.
    """
    if hint is None:
        return None
    ck = hint.get("chunk_kwargs") or {}
    ax = ck.get("axis")
    if isinstance(ax, (list, tuple)):
        return tuple(int(a) for a in ax)
    if ax is not None:
        return (int(ax),)
    return None


class Bridge(IBridge):
    def __init__(self, comm: ICommunicator, arrays_metadata: Dict[str, Dict], *args, **kwargs):
        """
        Initializes an instance of the class, setting up communication, metadata validation,
        client connection (for id=0), workers initialization, and handshake configuration for the bridge.

        - ``:param comm:`` An ICommunicator facilitating communication between processes.
            Must provide Get_rank(), Get_size(), gather(), bcast(), barrier(),
            Split(color, key), and Free() — the same API as an MPI communicator.
        - ``:param arrays_metadata:`` Dictionary containing metadata for arrays.
            eg:

            arrays_metadata = {
                    'temperature': {
                        'global_shape': [20, 20],
                        'chunk_shape': [10, 10],
                        'chunk_position': [0, 0],
                    }
                    'pressure': {
                        'global_shape': [20, 20],
                        'chunk_shape': [10, 10],
                        'chunk_position': [0, 0],
                    }
            }

        - ``:type arrays_metadata: Dict[str, Dict]``
        - ``:param args:`` Additional positional arguments for the initialization.
        - ``:param kwargs:`` Additional keyword arguments for the initialization. Can include
            configuration parameters like timeout used during client setup.
        """
        super().__init__(comm, arrays_metadata, *args, **kwargs)
        self.comm: ICommunicator = comm
        self.id = self.comm.Get_rank()
        self.arrays_metadata = validate_arrays_metadata(arrays_metadata)
        self._my_arrays: set[str] = set(arrays_metadata.keys())  # arrays this bridge owns
        self._feedback_queues = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self._has_close_been_called = False
        self.workers = None
        self.handshake = None
        self.client: Optional[Client] = None
        self._array_comms: Dict[str, Any] = {}  # array_name -> sub-comm (from comm.Split)
        self._handshake_metadata = None
        self._task_hints: Dict[str, List[Dict]] = {}  # array_name -> hints for local execution

        if self.id == 0:
            # only id 0 has a real dask client
            self.client = get_client(timeout=kwargs.get("timeout", 10), name=f"bridge-{self.comm.Get_rank()}")
            assert self.client, "client cannot be None for Bridge id 0."
            # get all workers from scheduler
            self.workers = self.client.scheduler_info(n_workers=-1)["workers"]

        # retrieve workers from rank 0 and bcast
        logger.debug(f"[{self.id}] Bridge __init__(): pre-bcast")
        self.workers = self.comm.bcast(self.workers, root=0)
        logger.debug(f"[{self.id}] Bridge __init__(): post-bcast. workers={self.workers}")

        # Gather each bridge's partial metadata → global view
        # Each bridge declares only the arrays it sends; merge into a single dict.
        self._gather_global_metadata()

        # Auto-discover array participation and create sub-communicators
        self._setup_array_comms()

        if self.id == 0:
            # all bridges are ready, tell handshake actor
            assert self.client is not None, "client cannot be None for Bridge id 0."
            self.handshake = Handshake(self.client)
            # Send merged metadata (from all bridges) to the handshake actor
            metadata_for_handshake = self._handshake_metadata if self._handshake_metadata else self.arrays_metadata
            self.handshake.all_bridges_ready(
                nb_bridge=self.comm.Get_size(), arrays_metadata=metadata_for_handshake, **kwargs
            )

            if kwargs.get("wait_for_go", True):
                Event(WAIT_FOR_EXECUTE_CB_EVENT, client=self.client).wait()

    def _gather_global_metadata(self):
        """
        Gather each bridge's partial arrays_metadata to discover the global
        set of array names.

        Each bridge declares only the arrays it actually sends. This method
        collects all array names across all bridges so that every rank can
        call Split() for every array (arrays not owned get color=_UNDEFINED).

        The bridge's own arrays_metadata (with its chunk_position etc.) is
        preserved for use in send(). Only a lightweight set of global array
        names is stored in self._global_array_names.

        Rank 0 also produces a merged metadata dict for the handshake actor
        (so the Deisa analytics side has the full picture).

        When the communicator size is 1 (single-bridge, no MPI peers), this
        is a no-op: the bridge's own metadata is already the full picture.
        """
        if self.comm.Get_size() == 1:
            # Single-bridge case: nothing to run a collective with
            self._global_array_names = set(self._my_arrays)
            return

        all_metadata = self.comm.gather(self.arrays_metadata, root=0)

        if all_metadata:
            merged = {}
            for partial in all_metadata:
                for name, metadata in partial.items():
                    merged.setdefault(name, metadata)

            self._handshake_metadata = merged
            global_names = set(merged)
        else:
            global_names = None

        # Broadcast the global set of array names to all bridges
        global_names = self.comm.bcast(global_names, root=0)
        self._global_array_names = sorted(global_names)

    def _setup_array_comms(self):
        """
        Create per-array sub-communicators using comm.Split().

        The global array list (from _gather_global_metadata) is known to all
        bridges, so every rank can call Split() for every array. Bridges that
        don't own a given array use color=_UNDEFINED and receive _COMM_NULL.

        The color is derived from zlib.crc32 of the array name so
        that all participating ranks get the same sub-communicator group.
        """
        for array_name in self._global_array_names:
            participates = array_name in self._my_arrays
            # Force into a positive 31-bit integer
            # Reserve 0x7FFFFFFF as _UNDEFINED value.
            color = (zlib.crc32(array_name.encode()) & 0x7FFFFFFE) if participates else _UNDEFINED

            # sub_comm is either an instance of: mpi4py.MPI.Comm, mpi4py.MPI.CommNull or None (FakeComm)
            # Split is a collective. All ranks of parent comm must call this.
            sub_comm = self.comm.Split(color, self.id)

            # convert mpi4py.MPI.CommNull to _COMM_NULL
            sub_comm = _COMM_NULL if not participates else sub_comm

            self._array_comms[array_name] = sub_comm

            # create a new client for sub_comm id==0 if needed
            if not self.client and sub_comm is not _COMM_NULL and sub_comm.Get_rank() == 0:
                self.client = get_client(timeout=10, name=f"bridge-{self.comm.Get_rank()}")
                # Connect to existing handshake actor from analytics side (created by Deisa)
                self.handshake = Handshake(self.client)

            # Store empty hints initially - they will be fetched on first send()
            self._task_hints[array_name] = []

            logger.debug(
                f"[{self.id}] _setup_array_comms: "
                f"array={array_name}, "
                f"participates={participates}, "
                f"sub_comm_size={sub_comm.Get_size() if sub_comm is not _COMM_NULL else 'NULL'}"
            )

    def __del__(self):
        """
        Clean up resources before destruction.
        """
        self.close(timestep=sys.maxsize)

    def close(self, timestep: int) -> None:
        """
        Close the bridge: synchronize bridges, free sub-comms, shut down client.

        - ``:param timestep:`` The current timestep associated with the closure action.
        - ``:type timestep:`` int
        - ``:return:`` None
        """
        logger.info(f"[{self.id}] Bridge close()")
        try:
            if not self._has_close_been_called:
                self._has_close_been_called = True
                # Barrier on communicator — all bridges must synchronize
                self.comm.barrier()
                # Free sub-communicators created via Split()
                for array_name, sub_comm in self._array_comms.items():
                    if sub_comm is not None and sub_comm != _COMM_NULL:
                        sub_comm.Free()
                        logger.debug(f"[{self.id}] Freed sub-communicator for array '{array_name}'")
                self._array_comms.clear()
                if self.id == 0:
                    assert self.handshake, "handshake cannot be None for Bridge id 0."
                    assert self.client, "client cannot be None for Bridge id 0."
                    self.handshake.set_bridges_done(timestep=timestep)

                if self.client:
                    self.client.close()
        except Exception as e:
            logger.error(f"[{self.id}] Cloud not cleanly close bridge. exception={e}")

    def send(self, array_name: str, chunk: np.ndarray, timestep: int, *args, **kwargs):
        """
        Distribute a data chunk to a Dask workers.

        Scatters the chunk to the next worker (round-robin), then gathers all
        chunks metadata to bridge rank 0, which updates the Dask scheduler.

        For single-bridge arrays (sub_comm_size == 1), skips the gather entirely
        and updates the scheduler directly.

        - ``:param array_name:`` The name of the data array being sent as a string.
        - ``:param chunk:`` A numpy ndarray containing the data chunk to be sent to the workers.
        - ``:param timestep:`` The current timestep associated to the sent data chunk.
        - ``:param args:`` Additional positional arguments if required by the method implementation.
        - ``:param kwargs:`` Additional keyword arguments for optional configurations.
            Supported kwargs: update_workers (bool), filter_workers (callable).
        - ``:return:`` None
        """
        logger.debug(f"[{self.id}] send() array_name={array_name}, data.shape={chunk.shape}, iteration={timestep}")

        if array_name not in self.arrays_metadata:
            raise ValueError(f"array {array_name} is unknown.")

        assert isinstance(self.workers, dict)
        workers = dict(self.workers)  # make a copy so that the user-defined function does not modify self

        if kwargs.get("update_workers", False):
            # only update worker list if requested
            sub_comm = self._array_comms[array_name]

            if sub_comm.Get_rank() == 0:
                assert self.client is not None, "client cannot be None for Bridge comm id 0."
                # rank 0 retrieve workers and bcast to other bridges
                workers = self.client.scheduler_info(n_workers=-1)["workers"]

            # bcast
            logger.debug(f"[{self.id}] send() pre-bcast workers={workers}")
            self.workers = sub_comm.bcast(workers, root=0)
            logger.debug(f"[{self.id}] send() post-bcast workers={workers}")
            workers = dict(self.workers)

        if kwargs.get("filter_workers", False):
            workers = kwargs["filter_workers"](workers)
            # check return type
            if not isinstance(workers, list):
                raise TypeError(f"worker_filter must return a list, got {type(workers)}")
            if len(workers) == 0:
                raise TypeError("worker_filter must return a non-empty list")
            for w in workers:
                if not isinstance(w, str):
                    raise TypeError(f"worker_filter must return a list of strings, got {type(w)}")
        else:
            workers = list(workers.keys())

        workers = sorted(workers)
        # per bridge id and iteration round-robin over the workers
        index = (timestep + self.id) % len(workers)
        workers = [workers[index]]

        assert len(workers) == 1, "worker list should be of length 1."

        # Fetch task hints and execute reduction operations locally on the
        # bridge-process numpy chunk. The resulting partials are tiny (scalar /
        # 1-d arrays) compared to the full chunk -- the goal of precompute is
        # to ship only the partials to the worker, never the full chunk.
        task_hints = self._get_task_hints(array_name)
        partials = self._execute_operations_on_chunk(array_name, chunk, task_hints)

        # Determine communicator from cached sub-comms (from comm.Split())
        sub_comm = self._array_comms.get(array_name)

        if sub_comm == _COMM_NULL:
            # This rank doesn't participate in this array
            logger.debug(f"[{self.id}] send() rank not in participating set for '{array_name}', skipping")
            return

        # Decide what to ship to workers:
        # - If precompute produced partials for this callback: scatter ONLY the
        #   partials (tiny). The full chunk stays on the bridge process and
        #   never enters worker memory.
        # - Otherwise (no reductions detected): fall back to the legacy path
        #   and scatter the full chunk, preserving backward compatibility for
        #   callbacks that don't return lazy dask reductions.
        precomputed_meta: Dict[str, Dict[str, Any]] = {}
        if partials:
            logger.debug(
                f"[{self.id}] send() precompute-active: scattering {len(partials)} partials "
                f"instead of full chunk shape={chunk.shape}"
            )
            partial_res = self._scatter_partials(partials, task_hints, array_name, workers=workers)
            res = partial_res["future-info"]
            precomputed_meta = partial_res["precomputed"]
        else:
            logger.debug(f"[{self.id}] send() precompute-inactive: scattering full chunk shape={chunk.shape}")
            res = self._better_scatter(chunk, workers=workers, hash=False)

        # Single-bridge fast-path: no collective needed
        if sub_comm.Get_size() == 1:
            self._direct_send(
                array_name,
                res,
                chunk,
                timestep,
                precomputed=precomputed_meta,
                precomputed_meta=precomputed_meta,
                task_hints=task_hints,
                branches=task_hints,  # BranchSpec list; legacy hints also accepted
            )
            return

        to_send = {
            "future-info": res,
            "chunk_position": self.arrays_metadata[array_name]["chunk_position"],
            "precomputed": precomputed_meta,
        }
        logger.debug(f"[{self.id}] send() gather: to_send={to_send}")

        gathered_data = sub_comm.gather(to_send, root=0)

        logger.debug(f"[{self.id}] send() gathered_data={gathered_data}")

        if gathered_data:
            assert self.client is not None, "client cannot be None for Bridge id 0."
            # rank 0 (root=0 in comm.gather): aggregate who_has from all partials/chunks
            who_has = {}
            nbytes = {}
            keys = []
            all_partials_meta: List[Optional[Dict[str, Dict]]] = []
            for d in gathered_data:
                who_has.update(d["future-info"]["who_has"])
                nbytes.update(d["future-info"]["nbytes"])
                future_field = d["future-info"]["future"]
                if isinstance(future_field, list):
                    keys.extend(future_field)
                else:
                    keys.append(future_field)
                if d.get("precomputed"):
                    all_partials_meta.append(d["precomputed"])

            # only update the scheduler with who has what and register the futures once
            self.client.sync(self.client.scheduler.update_data, who_has=who_has, nbytes=nbytes)

            # mimic mechanism from Queue. Keep a reference on keys until reception in topic handler.
            # TODO: id=0 can use a queue
            self.client._send_to_scheduler({"op": "client-desires-keys", "keys": keys, "client": CLIENT_KEY})

            # Build the topic event. When precompute is active, `futures` lists
            # one entry per (bridge, reduction) pair, each entry pointing to the
            # partial's reduced shape and dtype. The Deisa side reconstructs the
            # dask graph from these small partials -- the full chunk never
            # reaches the workers.
            #
            # Build a per-output_key chunk_axis lookup. BranchSpec
            # objects carry ``chunk_axis`` directly; legacy hint dicts
            # require going through ``_extract_chunk_axis_from_hint``.
            chunk_axis_by_key: Dict[str, Optional[Tuple[int, ...]]] = {}
            for b in task_hints:
                if isinstance(b, BranchSpec):
                    chunk_axis_by_key[b.output_key] = b.chunk_axis
                else:
                    # Legacy hint dict.
                    chunk_axis_by_key[b["output_key"]] = _extract_chunk_axis_from_hint(b)
            futures_payload: List[Dict[str, Any]]
            if all_partials_meta:
                # Precompute path: emit one entry per (bridge, reduction).
                # ``chunk_position`` here is the MPI coords of the bridge
                # that contributed this partial, so the Deisa side can
                # rebuild the nested list structure that ``mean_agg`` /
                # ``moment_agg`` expect (matching the chunk-grid layout).
                # ``chunk_axis`` is the chunk_func's reduction axes tuple
                # (the chunk_kwargs ``axis``), used by the topic handler
                # to compute the combine's output shape and pass the
                # correct ``axis`` to ``mean_agg`` / ``moment_agg``.
                futures_payload = []
                for bridge_idx, partial_meta in enumerate(all_partials_meta):
                    for output_key, p_info in partial_meta.items():
                        chunk_axis = chunk_axis_by_key.get(output_key)
                        futures_payload.append(
                            {
                                "future": p_info["future"],
                                "shape": p_info["shape"],
                                "dtype": p_info["dtype"],
                                "kind": p_info.get("kind", "scalar"),
                                "finalize": p_info.get("finalize"),
                                "chunk_position": gathered_data[bridge_idx]["chunk_position"],
                                "chunk_axis": chunk_axis,
                                "output_key": output_key,
                            }
                        )
            else:
                # Legacy path: emit one entry per bridge with the full-chunk
                # shape, same as before the precompute feature.
                futures_payload = [
                    {
                        "future": d["future-info"]["future"][0]
                        if isinstance(d["future-info"]["future"], list)
                        else d["future-info"]["future"],
                        "shape": chunk.shape,
                        "dtype": str(chunk.dtype),
                        "chunk_position": d["chunk_position"],
                    }
                    for d in gathered_data
                ]

            to_send = {
                "array_name": array_name,
                "iteration": timestep,
                "precomputed": True if all_partials_meta else None,
                "futures": futures_payload,
            }
            logger.debug(
                f"[{self.id}] send() log_event: array={array_name}, "
                f"timestep={timestep}, n_futures={len(futures_payload)}"
            )
            self.client.log_event(array_name, to_send)

        # TODO: what to do if error ?

    def _direct_send(
        self,
        array_name: str,
        res: dict,
        chunk: np.ndarray,
        timestep: int,
        precomputed: Optional[Dict] = None,
        precomputed_meta: Optional[Dict[str, Dict]] = None,
        task_hints: Optional[List[Dict]] = None,
        branches: Optional[List[Any]] = None,
    ):
        """
        Handle single-bridge array send without collective.

        For arrays that exist on only one bridge, we skip the gather() entirely
        and directly update the Dask scheduler.

        - ``:param array_name:`` The array name being sent.
        - ``:param res:`` The scatter result (legacy: dict with a single ``future``;
            precompute: dict with a list ``future`` of all partial keys).
        - ``:param chunk:`` The numpy ndarray data chunk (kept for legacy shape/dtype).
        - ``:param timestep:`` The current timestep.
        - ``:param precomputed:`` Optional precomputed values dict (legacy key,
            kept for API stability; prefer ``precomputed_meta``).
        - ``:param precomputed_meta:`` Per-partial scatter metadata
            (``{output_key: {"future", "shape", "dtype"}}``); only set on the
            precompute path. When provided, the topic event's ``futures`` list
            carries the partials' reduced shapes instead of the full-chunk shape.
        """
        assert self.client is not None, "client cannot be None for single-bridge send."

        who_has = res["who_has"]
        nbytes = res["nbytes"]

        # On the precompute path, ``res["future"]`` is a list of partial keys
        # (one per reduction). On the legacy path, it's a single future key.
        future_keys = res["future"] if isinstance(res["future"], list) else [res["future"]]

        self.client.sync(self.client.scheduler.update_data, who_has=who_has, nbytes=nbytes)
        self.client._send_to_scheduler({"op": "client-desires-keys", "keys": future_keys, "client": CLIENT_KEY})

        # Build the topic event. On the precompute path, emit one entry per
        # partial (with its reduced shape); on the legacy path, emit one entry
        # pointing at the full chunk.
        if precomputed_meta:
            # Build a per-output_key lookup for chunk_axis. Prefer
            # BranchSpec objects (the new wire format) over legacy
            # hint dicts; both expose ``output_key`` and a way to get
            # the chunk axis. BranchSpec carries ``chunk_axis``
            # directly; the legacy hint requires going through
            # ``_extract_chunk_axis_from_hint``.
            chunk_axis_by_key: Dict[str, Optional[Tuple[int, ...]]] = {}
            if branches:
                for b in branches:
                    if isinstance(b, BranchSpec):
                        chunk_axis_by_key[b.output_key] = b.chunk_axis
                    else:
                        # Legacy hint dict.
                        chunk_axis_by_key[b["output_key"]] = _extract_chunk_axis_from_hint(b)
            elif task_hints:
                for h in task_hints:
                    chunk_axis_by_key[h["output_key"]] = _extract_chunk_axis_from_hint(h)
            futures_payload = [
                {
                    "future": info["future"],
                    "shape": info["shape"],
                    "dtype": info["dtype"],
                    "kind": info.get("kind", "scalar"),
                    "finalize": info.get("finalize"),
                    "chunk_position": self.arrays_metadata[array_name]["chunk_position"],
                    "chunk_axis": chunk_axis_by_key.get(output_key),
                    "output_key": output_key,
                }
                for output_key, info in precomputed_meta.items()
            ]
        else:
            futures_payload = [
                {
                    "future": future_keys[0],
                    "shape": chunk.shape,
                    "dtype": str(chunk.dtype),
                    "chunk_position": self.arrays_metadata[array_name]["chunk_position"],
                }
            ]

        to_send = {
            "array_name": array_name,
            "iteration": timestep,
            "precomputed": True if precomputed_meta else precomputed,
            "futures": futures_payload,
        }
        self.client.log_event(array_name, to_send)

    def get(self, key: str, timestep: Optional[int] = None, default: Any = None) -> Optional[Union[Deque, Any]]:
        """
        Retrieve an element associated with a specific key and optional timestep from a feedback queue.
        If a queue for the key does not exist, it initializes the queue for the specified key.

        - ``:param key:`` The unique identifier for the feedback queue.
        - ``:type key:`` str
        - ``:param timestep:`` An optional specific timestep to look for. If None, returns the entire deque.
        - ``:type timestep:`` Optional[int]
        - ``:param default:`` The default value to return if the specified timestep is not found.
        - ``:type default:`` Any
        - ``:return:`` The element associated with the specified timestep if found, the entire deque if no
            timestep is specified, or the default value if the timestep is not found.
        - ``:rtype:`` Optional[Union[Deque, Any]]
        """
        logger.debug(f"[{self.id}] get() key={key}, timestep={timestep}, default={default}")
        fb_state: Dict = self._feedback_queues[key]

        if self.id == 0:
            if len(fb_state) == 0:
                feedback_queue_size = self.handshake.get_feedback_queue_size()
                fb_state[key] = {
                    "q": Queue(f"{FEEDBACK_QUEUE_PREFIX}{key}", client=self.client, maxsize=feedback_queue_size),
                    "deque": deque(maxlen=feedback_queue_size),
                }

            q: Queue = fb_state[key]["q"]
            d: deque = fb_state[key]["deque"]

            if q.qsize() != 0:
                # List[(int, Any), ...]
                full_q = q.get(batch=True)  # get all elements. This pops elements from the Dask queue.
                for v in full_q:
                    d.append(v)  # add all elements to deque
            logger.debug(f"[{self.id}] get() fb_state={fb_state}")

        d = self.comm.bcast(fb_state[key]["deque"], root=0)

        if timestep is None:
            return d

        for t, v in d:
            if timestep == t:
                # found the timestep
                return v

        return default

    def _better_scatter(self, data: np.ndarray, workers: List[str] = None, hash=False):
        logger.debug(f"[{self.id}] scatter to {workers}")

        if workers is None:
            workers = self.workers

        if self.client:
            return self.client.sync(self.__scatter, data, workers=workers, hash=hash)
        else:
            return asyncio.run(self.__scatter(data, workers=workers, hash=hash))

    async def __scatter(self, data, workers=None, hash=False):
        if isinstance(workers, (str, Number)):
            workers = [workers]
        if isinstance(data, type(range(0))):
            data = list(data)

        input_type = type(data)
        names = False
        unpack = False
        if isinstance(data, Iterator):
            data = list(data)
        if isinstance(data, (set, frozenset)):
            data = list(data)
        if not isinstance(data, (dict, list, tuple, set, frozenset)):
            unpack = True
            data = [data]
        if isinstance(data, (list, tuple)):
            if hash:
                names = [KEY_PREFIX + "-" + type(x).__name__ + "-" + tokenize(x) for x in data]
            else:
                names = [KEY_PREFIX + "-" + type(x).__name__ + "-" + uuid.uuid4().hex for x in data]
            data = dict(zip(names, data))

        assert isinstance(data, dict)

        data2 = valmap(to_serialize, data)

        _, who_has, nbytes = await scatter_to_workers(workers, data2)

        out = {k: {"future": k, "who_has": who_has, "nbytes": nbytes} for k in data}

        if issubclass(input_type, (list, tuple, set, frozenset)):
            out = input_type(out[k] for k in names)

        if unpack:
            assert len(out) == 1
            out = list(out.values())[0]
        return out

    def _get_task_hints(self, array_name: str) -> List[Dict]:
        """
        Retrieve stored task hints for an array.

        Hints are fetched from HandshakeActor on sub_comm rank 0 and broadcast to all ranks.
        If no hints are available, this method returns an empty list (no precomputation).

        - ``:param array_name:`` The array name to get hints for.
        - ``:return:`` List of reduction hints (each carrying a pickled chunk
            callable, pickled aggregator, and the dask kwargs to apply).
        """
        # Check cache first
        if self._task_hints.get(array_name):
            return self._task_hints[array_name]

        # If not cached, need to fetch (only sub_comm rank 0 has client)
        sub_comm = self._array_comms.get(array_name)
        hints: List[Dict] = []

        if sub_comm is not None and sub_comm is not _COMM_NULL:
            if sub_comm.Get_rank() == 0 and self.handshake is not None:
                hints = self.handshake.get_task_hints(array_name)
                # Broadcast to all ranks in sub_comm
                sub_comm.bcast(hints, root=0)
            else:
                # Receive broadcast
                hints = sub_comm.bcast(None, root=0)

            # Cache the hints
            if hints:
                self._task_hints[array_name] = hints

        return hints

    def _scatter_partials(
        self,
        partials: Dict[str, Any],
        branches: List[Any],  # List[BranchSpec] or List[dict] (legacy hints)
        array_name: str,
        workers: List[str],
    ) -> Dict[str, Any]:
        """
        Scatter precomputed reduction partials to a worker instead of the full chunk.

        Each partial value is the local result of running the branch's
        chunk-stage callable on the bridge's numpy chunk. Two flavors:

        - ``"scalar"`` partials (sum/prod/max/min): plain scalars or numpy
          arrays. Shipped as one future key per reduction.
        - ``"mean"`` partials (mean): a ``{"n": x, "total": y}`` dict from
          dask's ``mean_chunk``. Shipped as one future key whose value is
          the whole dict; the Deisa-side combine resolves the dicts and
          calls ``mean_agg`` over them.
        - ``"moment"`` partials (var/std): a ``{"n": x, "total": y, "M": z}``
          dict from dask's ``moment_chunk``. Same dict-blob handling as
          mean, but the combine calls ``moment_agg`` (and sqrt for std).

        Returns a dict shaped like the legacy ``_better_scatter`` result
        (``{"future": [...], "who_has": {...}, "nbytes": {...}}``) plus a
        ``precomputed`` entry mapping each ``output_key`` to its scatter
        metadata (``{future, kind, shape, dtype, finalize}``) so the topic
        handler can reconstruct the right dask graph.

        - ``:param partials:`` Mapping of ``output_key`` -> partial value
            produced by :meth:`_execute_operations_on_chunk`.
        - ``:param branches:`` The :class:`BranchSpec` objects the
            bridge used to compute the partials. Carry per-reduction
            ``kind``/``finalize``/``partial_shape``/``partial_dtype``
            metadata.
        - ``:param array_name:`` Array name (used for key prefixing).
        - ``:param workers:`` Single-element list of worker names to scatter to.
        - ``:return:`` Dict with ``future-info`` (legacy-shape scatter result
            containing all partials' keys) and ``precomputed`` (per-partial
            metadata for the topic handler).
        """
        assert len(workers) == 1, "_scatter_partials expects a single target worker"
        target_worker = workers[0]

        # Index branches by output_key for fast lookup. BranchSpec
        # carries the per-reduction metadata directly, so the loop
        # body doesn't need to inspect the partial value (legacy code
        # did ``isinstance(value, dict)`` and then peek at ``total``/
        # ``M`` -- the analyzer already recorded that in
        # ``branch.partial_shape``/``branch.partial_dtype``).
        #
        # Backward-compat: ``branches`` may be a list of legacy hint
        # dicts (from the pre-BranchSpec registration path). We
        # dispatch on element type -- BranchSpec items get their
        # fields used directly; dict items get a small dict-lookup
        # shim that mirrors the legacy behaviour.
        if branches and not isinstance(branches[0], BranchSpec):
            # Convert each legacy hint dict to a duck-typed
            # BranchSpec-like accessor. Avoids importing the dataclass
            # machinery just for a couple of fields.
            class _DictBranch:
                __slots__ = ("output_key", "output_kind", "finalize", "partial_shape", "partial_dtype")

                def __init__(self, h):
                    self.output_key = h["output_key"]
                    self.output_kind = h.get("kind", "scalar")
                    self.finalize = h.get("finalize")
                    # Legacy hints don't carry shape/dtype; leave as
                    # None and let the loop fall back to value inspection.
                    self.partial_shape = h.get("shape")
                    self.partial_dtype = h.get("dtype")

            branches = [_DictBranch(h) for h in branches]
        branch_by_key = {b.output_key: b for b in branches}

        payload: Dict[str, Any] = {}
        shape_dtype: Dict[str, Dict[str, Any]] = {}
        for output_key, value in partials.items():
            branch = branch_by_key.get(output_key)
            if branch is None:
                # Backward-compat: legacy hint path produced dicts here.
                # We don't have shape/dtype metadata, so inspect the
                # value as the old code did.
                if isinstance(value, dict):
                    if "total" in value:
                        rep = np.asarray(value["total"])
                    elif "M" in value:
                        rep = np.asarray(value["M"])
                    else:
                        rep = np.asarray(next(iter(value.values())))
                    red_shape = tuple(rep.shape)
                    red_dtype = str(rep.dtype)
                    kind = "mean" if "total" in value else "moment"
                else:
                    arr = np.asarray(value)
                    red_shape = tuple(arr.shape)
                    red_dtype = str(arr.dtype)
                    kind = "scalar"
                finalize = None
            else:
                kind = branch.output_kind
                finalize = branch.finalize
                red_shape = branch.partial_shape
                red_dtype = branch.partial_dtype
                # Backward-compat: legacy hint dicts (and any future
                # branch that didn't record shape/dtype at the
                # analyzer side) have ``partial_shape`` and
                # ``partial_dtype`` set to None. Fall back to
                # inspecting the actual partial value -- the
                # same heuristic the pre-BranchSpec code used.
                if red_shape is None or red_dtype is None:
                    if isinstance(value, dict):
                        if "total" in value:
                            rep = np.asarray(value["total"])
                        elif "M" in value:
                            rep = np.asarray(value["M"])
                        else:
                            rep = np.asarray(next(iter(value.values())))
                        red_shape = tuple(rep.shape)
                        red_dtype = str(rep.dtype)
                        # ``kind`` may also be missing on legacy
                        # hints; default to dict-shape.
                        if branch.output_kind == "scalar":
                            kind = "mean" if "total" in value else "moment"
                    else:
                        arr = np.asarray(value)
                        red_shape = tuple(arr.shape)
                        red_dtype = str(arr.dtype)
            key = f"{KEY_PREFIX}{array_name}-partial-{output_key}-{uuid.uuid4().hex}"
            payload[key] = value
            shape_dtype[output_key] = {
                "future": key,
                "kind": kind,
                "shape": red_shape,
                "dtype": red_dtype,
                "finalize": finalize,
            }

        # Serialize for scatter (handles numpy arrays in dict values).
        payload2 = valmap(to_serialize, payload)

        # Use scatter_to_workers directly so we get the (who_has, nbytes) pair.
        # Mirrors the legacy ``_better_scatter`` pattern: client.sync when a
        # Client is available, asyncio.run otherwise (rank-0 only has the
        # Client; non-rank-0 bridges run the scatter from a fresh event loop).
        if self.client is not None:
            _, who_has, nbytes = self.client.sync(self._scatter_to_workers_async, target_worker, payload2)
        else:
            _, who_has, nbytes = asyncio.run(self._scatter_to_workers_async(target_worker, payload2))

        future_keys = list(payload.keys())
        return {
            "future-info": {
                "future": future_keys,
                "who_has": who_has,
                "nbytes": nbytes,
            },
            "precomputed": shape_dtype,
        }

    async def _scatter_to_workers_async(self, worker: str, data: Dict[str, Any]):
        """Async helper: scatter ``data`` to a single worker. Returns (ok, who_has, nbytes)."""
        _, who_has, nbytes = await scatter_to_workers([worker], data)
        return True, who_has, nbytes

    def _execute_operations_on_chunk(
        self, array_name: str, chunk: np.ndarray, branches: List["BranchSpec"]
    ) -> Dict[str, Any]:
        """
        Execute branch chunk-stage callables locally on the bridge's
        numpy chunk before scattering.

        Each branch in ``branches`` is a :class:`BranchSpec` whose
        ``branch_func`` is a pickle-able Python callable that takes a
        numpy chunk and returns the per-bridge partial (a scalar,
        ndarray, or dict for mean/moment). The closure inside
        ``branch_func`` already binds the analyzer's chunk kwargs
        (axis, keepdims, dtype, ...), so the call here is
        ``branch_func(chunk)`` with no extra arguments.

        - ``:param array_name:`` The array name being processed.
        - ``:param chunk:`` The numpy ndarray data chunk.
        - ``:param branches:`` List of :class:`BranchSpec` from
            ``analyze_branch``.
        - ``:return:`` Dict of partial results keyed by output_key.
        """
        from deisa.dask.branch import BranchSpec

        partials = {}
        for branch in branches:
            if not isinstance(branch, BranchSpec):
                # Backward compat: legacy hint dicts are still produced
                # by ``_analyze_callback_for_operations``. Convert on
                # the fly using the same kwargs logic the legacy path
                # used.
                output_key = branch["output_key"]
                kind = branch.get("kind", "scalar")
                try:
                    chunk_func = pickle.loads(branch["chunk_func_pickle"])
                    chunk_kwargs = dict(branch.get("chunk_kwargs", {}) or {})
                    if kind in ("mean", "moment"):
                        chunk_kwargs["keepdims"] = True
                    partial = chunk_func(chunk, **chunk_kwargs)
                except Exception as e:
                    logger.warning(f"[{self.id}] _execute_operations_on_chunk: could not execute {output_key}: {e}")
                    continue
                partials[output_key] = partial
                continue

            output_key = branch.output_key
            try:
                partial = branch.branch_func(chunk)
            except Exception as e:
                logger.warning(f"[{self.id}] _execute_operations_on_chunk: could not execute {output_key}: {e}")
                continue
            partials[output_key] = partial

        logger.debug(f"[{self.id}] _execute_operations_on_chunk: {partials}")
        return partials

    # NOTE: pre-v2 architecture used a centralized ``_combine_reduction_partials``
    # on rank 0 to combine partials across bridges before scattering. That
    # approach was replaced in PR #4 by the per-bridge partial-scatter: each
    # bridge now ships its own raw partial to a worker, and the second-stage
    # combine (e.g. summing per-bridge scalar partials for ``arr.sum()``) is
    # expressed naturally by the dask graph the Deisa side builds from the
    # partials. Keeping the old combiner here is unnecessary and would defeat
    # the wire/worker-memory savings the optimization is meant to deliver.
