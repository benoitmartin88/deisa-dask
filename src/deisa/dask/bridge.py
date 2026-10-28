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
"""The simulation-side half of DEISA: one ``Bridge`` per MPI rank, shipping chunk-local reductions.

Two delivery modes coexist on this side, and only the analytics host decides which one a callback gets:

- the runtime reduction path (the delivery mode of ``precompute=True``): ``send`` ships an eager scalar menu per
  timestep and answers on-demand plan requests at its ``send`` boundary;
- the LEGACY full-chunk scatter path (``precompute=False``): ``send`` ships the chunk itself, and any branch funcs the
  host still files are executed on the local chunk first.

**The P3 -> P4 coexistence cost, stated once.** While both modes exist, the eager menu's
``sum``/``min``/``max``/``prod`` entries recompute what ``_execute_operations_on_chunk`` just computed on the SAME
chunk. That double work is intentional and priced in: P4 deletes the analyzer path and makes the menu the only fold
site, which removes it. Until then, ``_build_runtime_menu`` is the only place it happens, and it sits after the reduce
step and before the first scatter so the chunk never travels to convey a menu value.
"""

import asyncio
import logging
import sys
import threading
import uuid
import zlib
from collections import defaultdict, deque
from numbers import Number
from typing import Any, Deque, Dict, Final, Iterator, List, Mapping, Optional, Tuple, Union

import numpy as np
from deisa.core import IBridge, ICommunicator, validate_arrays_metadata
from distributed import Client, Event, Queue
from distributed.protocol import to_serialize
from distributed.utils_comm import scatter_to_workers
from tlz import valmap

from dask.tokenize import tokenize
from deisa.dask.branch import BranchSpec
from deisa.dask.collection import DeisaCollection
from deisa.dask.constants import CLIENT_KEY, FEEDBACK_QUEUE_PREFIX, KEY_PREFIX, WAIT_FOR_EXECUTE_CB_EVENT
from deisa.dask.handshake import Handshake
from deisa.dask.precompute_errors import PrecomputeRuntimeError
from deisa.dask.runtime_dispatch import (
    EagerMenu,
    PlanRequestBacklog,
    PlanResponder,
    build_eager_menu,
    plan_report_queue_name,
    plan_request_queue_name,
    serve_plan_queue,
)
from deisa.dask.utils import called_from_event_loop, get_client, run_coro_on_private_loop

logger = logging.getLogger(__name__)

_COMM_NULL: Final[None] = None
try:
    from mpi4py import MPI

    _UNDEFINED = MPI.UNDEFINED
except Exception:
    logger.warning("mpi4py not available.")
    _UNDEFINED = 2147483647


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

        # array_name -> branches for local execution (None = "not fetched yet")
        self._task_branches: Dict[str, Optional[List[BranchSpec]]] = {}
        # Arrays the analytics host declared as runtime-delivery targets (fetched with the branches, one bcast per
        # array). An array outside this set has no callback that can read an eager menu, so ``send`` folds nothing for
        # it -- the menu of an array nobody subscribes to is pure critical-path cost.
        self._runtime_delivery_arrays: set[str] = set()
        # array_name -> output_key -> branch (BranchSpec)
        self._branch_by_key: Dict[str, Dict[str, Any]] = {}
        # Runtime reduction transport: the array's plan queues live only on the rank that owns its dask client, the
        # backlog exists on every rank (the others receive the requests through the sub-comm broadcast), and the
        # retained chunk is the one this bridge last sent -- what an on-demand request for that timestep folds.
        self._runtime_plan_backlog: Dict[str, PlanRequestBacklog] = {}
        self._runtime_plan_report_queues: Dict[str, Any] = {}
        self._runtime_plan_stops: Dict[str, threading.Event] = {}
        self._runtime_plan_threads: Dict[str, threading.Thread] = {}
        self._runtime_last_chunk: Dict[str, Tuple[int, Any]] = {}

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

        # Gather each bridge's partial metadata -> global view: each bridge declares only the arrays it sends; merge
        # into a single dict.
        self._gather_global_metadata()

        # Auto-discover array participation and create sub-communicators
        self._setup_array_comms()

        # The runtime reduction transport is attached here, before the first send: the plan queues must be listening by
        # the time the analytics host dispatches (its first on-demand request follows the first delivered event).
        self._start_runtime_plan_pumps()

        if self.id == 0:
            # all bridges are ready, tell handshake actor
            assert self.client is not None, "client cannot be None for Bridge id 0."
            self.handshake = Handshake(self.client)
            # Send merged metadata (from all bridges) to the handshake actor
            metadata_for_handshake = self._handshake_metadata or self.arrays_metadata
            self.handshake.all_bridges_ready(
                nb_bridge=self.comm.Get_size(), arrays_metadata=metadata_for_handshake, **kwargs
            )

            # The go-event wait is unconditional on the wait_for_go path: ``WAIT_FOR_EXECUTE_CB_EVENT`` is set by
            # ``execute_callbacks()`` only. wait_for_go=False skips the wait and fetches branches lazily on first
            # send().
            if kwargs.get("wait_for_go", True):
                Event(WAIT_FOR_EXECUTE_CB_EVENT, client=self.client).wait()

        # World-wide barrier: rank 0 waited for the go event (all callbacks registered), the other ranks skipped that
        # wait. The barrier aligns every rank past callback registration so the prefetch below reads a consistent
        # handshake state. MUST be unconditional (it is a world collective); deliberately NOT gated on wait_for_go.
        self.comm.barrier()
        logger.debug(f"[{self.id}] Bridge __init__(): post-registration barrier done")

        if kwargs.get("wait_for_go", True) and self.id == 0:
            # Prefetch static branches out of the send() path. Only safe here: the barrier guarantees branches exist in
            # the handshake. On wait_for_go=False they may not be registered yet (--lazy first-send fetch in
            # _get_task_branches).
            self._prefetch_task_branches()

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
            # Force into a positive 31-bit integer Reserve 0x7FFFFFFF as _UNDEFINED value.
            color = (zlib.crc32(array_name.encode()) & 0x7FFFFFFE) if participates else _UNDEFINED

            # sub_comm is either an instance of: mpi4py.MPI.Comm, mpi4py.MPI.CommNull or None (FakeComm) Split is a
            # collective. All ranks of parent comm must call this.
            sub_comm = self.comm.Split(color, self.id)

            # convert mpi4py.MPI.CommNull to _COMM_NULL
            sub_comm = _COMM_NULL if not participates else sub_comm

            self._array_comms[array_name] = sub_comm

            # create a new client for sub_comm id==0 if needed
            if not self.client and sub_comm is not _COMM_NULL and sub_comm.Get_rank() == 0:
                self.client = get_client(timeout=10, name=f"bridge-{self.comm.Get_rank()}")
                # Connect to existing handshake actor from analytics side (created by Deisa)
                self.handshake = Handshake(self.client)

            # Seed a "not fetched yet" sentinel: None != [] (a cached branch-less array). Fetched once and cached, []
            # included.
            self._task_branches[array_name] = None

            logger.debug(
                f"[{self.id}] _setup_array_comms: "
                f"array={array_name}, "
                f"participates={participates}, "
                f"sub_comm_size={sub_comm.Get_size() if sub_comm is not _COMM_NULL else 'NULL'}"
            )

    def __del__(self):
        """Clean up resources before destruction.

        Never runs a blocking collective at interpreter shutdown
        (``sys.is_finalizing()``): ``close()`` synchronizes all bridges with an
        MPI barrier, and at shutdown the peer ranks may already be torn down,
        so that barrier would hang forever -- a hang is worse than an
        exception.

        Never runs it from an event-loop callback either. The collector runs
        ``__del__`` on whichever thread dropped the last reference, and for a
        leaked bridge that is a distributed "IO loop" thread. ``close()``
        synchronizes with the world (MPI barrier) and with the scheduler
        (``Handshake.set_bridges_done`` -> ``Client.sync()``), and ``Client.sync()``
        blocks the calling thread until the client's loop runs the callback: from
        inside that very loop the callback can never run, so the thread sits out the
        whole ``sync()`` timeout (60 s measured) and, with an in-process cluster whose
        scheduler, workers and client share that loop, freezes the whole cluster.

        During normal (non-shutdown) garbage collection from any other thread the
        full ``close()`` still runs.
        """
        if sys.is_finalizing():
            logger.debug(f"[{self.id}] Bridge.__del__ during interpreter shutdown: skipping close()")
            return
        if called_from_event_loop():
            logger.debug(
                f"[{self.id}] Bridge.__del__ from inside an event-loop callback: skipping close(), it would "
                f"block the loop that has to service its collectives"
            )
            return
        try:
            self.close(timestep=sys.maxsize)
        except Exception:
            # Never raise from __del__. close() already logs its own errors.
            pass

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
                if sys.is_finalizing():
                    # Interpreter shutdown: the world barrier and the sub-comm Free() calls are collectives that can
                    # never complete once peer ranks are gone. Skip the coordinated part; the client (non-collective) is
                    # still closed below.
                    logger.debug(f"[{self.id}] Bridge close() during interpreter shutdown: skipping collectives")
                else:
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
                    # The runtime plan pumps poll a dask queue through this client, so they leave the queue before it
                    # goes away. A daemon thread, never a reason to fail a teardown.
                    self._stop_runtime_plan_pumps()
                    self.client.close()
        except Exception as e:
            logger.error(f"[{self.id}] Could not cleanly close bridge. exception={e}")

    def send(self, array_name: str, chunk: np.ndarray, timestep: Union[int, np.ndarray], *args, **kwargs):
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

        # handle different types of timestep. Cast to int if necessary.
        if isinstance(timestep, np.ndarray):
            if timestep.ndim == 0:
                timestep = int(timestep)
            elif timestep.ndim == 1 and timestep.size == 1:
                timestep = timestep.item()
            else:
                raise ValueError("timestep must be a scalar, a 0D array or a 1D array of size 1.")
        assert isinstance(timestep, int)

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

        # Determine communicator from cached sub-comms (from comm.Split())
        sub_comm = self._array_comms.get(array_name)

        if sub_comm == _COMM_NULL:
            # This rank doesn't participate in this array
            logger.debug(f"[{self.id}] send() rank not in participating set for '{array_name}', skipping")
            return

        # Runtime reduction transport: answer the on-demand requests that arrived since the previous send BEFORE the
        # retained chunk is replaced (a request names the timestep that was last sent, so that is the chunk it folds),
        # then retain this timestep's chunk for the next boundary. Both steps only READ or hold a reference to the
        # chunk -- nothing is computed from it here -- so the reduce-before-scatter ordering below is untouched. The
        # cost is one retained chunk reference per array per bridge (replaced, not accumulated) for every array this
        # rank participates in, flag or no flag: a rank whose own flag is off still has to answer a request its root
        # broadcast, so neither the collective nor the retention may depend on the per-process flag.
        if array_name in self._runtime_plan_backlog:
            self._serve_pending_plans(array_name, sub_comm)
            self._runtime_last_chunk[array_name] = (int(timestep), chunk)

        # Execute branch funcs locally on the numpy chunk (running on participating ranks only -- the ``_COMM_NULL``
        # early return above exits first), then ship ONLY the tiny partials; the full chunk never enters worker memory.
        branches = self._get_task_branches(array_name)
        partials = self._execute_operations_on_chunk(chunk, branches)

        # Eager scalar menu (the zero-round-trip half of the transfer strategy): computed HERE, on the local numpy
        # chunk and before ANY scatter, so the menu values are chunk-local reductions and the chunk itself is never
        # shipped to convey them. Nothing below may run before this line without breaking the ordering invariant.
        # Gated on the array being a runtime-delivery target: with no callback reading it, the folds would be pure
        # critical-path cost.
        menu, menu_error = self._build_runtime_menu(array_name, chunk, timestep)

        # With partials: scatter them only. Without: full-chunk scatter (backwards compatibility for non-reduction
        # callbacks).
        precomputed_meta: Dict[str, Dict[str, Any]] = {}
        if partials:
            logger.debug(
                f"[{self.id}] send() precompute-active: scattering {len(partials)} partials "
                f"instead of full chunk shape={chunk.shape}"
            )
            partial_res = self._scatter_partials(partials, branches, array_name, workers=workers)
            res = partial_res["future-info"]
            precomputed_meta = partial_res["precomputed"]
        else:
            logger.debug(f"[{self.id}] send() precompute-inactive: scattering full chunk shape={chunk.shape}")
            res = self._scatter_full(chunk, workers=workers, hash=False)

        # Single-bridge fast-path: no collective needed
        if sub_comm.Get_size() == 1:
            self._direct_send(
                array_name,
                res,
                chunk,
                timestep,
                precomputed_meta=precomputed_meta,
                menu=menu,
                menu_error=menu_error,
            )
            return

        to_send = {
            "future-info": res,
            "chunk_position": self.arrays_metadata[array_name]["chunk_position"],
            "precomputed": precomputed_meta,
            "menu": menu,
        }
        if menu_error is not None:
            to_send["menu_error"] = menu_error
        logger.debug(f"[{self.id}] send() gather: to_send={to_send}")

        gathered_data = sub_comm.gather(to_send, root=0)

        logger.debug(f"[{self.id}] send() gathered_data={gathered_data}")

        if gathered_data:
            assert self.client is not None, "client cannot be None for Bridge id 0."
            # rank 0 (root=0 in comm.gather): aggregate who_has from all partials/chunks
            who_has = {}
            nbytes = {}
            keys = []
            # Each partial set carries its OWN chunk_position: indexing filtered metadata with ``enumerate`` would
            # misalign when a bridge shipped no partials, letting one bridge's partial inherit another bridge's
            # coordinates (or under-count the grid) and corrupt the combine.
            all_partials_meta: List[Dict[str, Any]] = []
            menus: List[EagerMenu] = []
            menu_errors: List[str] = []
            for d in gathered_data:
                if d.get("menu_error") is not None:
                    menu_errors.append(d["menu_error"])
                who_has.update(d["future-info"]["who_has"])
                nbytes.update(d["future-info"]["nbytes"])
                future_field = d["future-info"]["future"]
                if isinstance(future_field, list):
                    keys.extend(future_field)
                else:
                    keys.append(future_field)
                if d.get("precomputed"):
                    all_partials_meta.append({"precomputed": d["precomputed"], "chunk_position": d["chunk_position"]})
                if d.get("menu") is not None:
                    menus.append(d["menu"])

            # only update the scheduler with who has what and register the futures once
            self.client.sync(self.client.scheduler.update_data, who_has=who_has, nbytes=nbytes)

            # mimic mechanism from Queue. Keep a reference on keys until reception in topic handler. TODO: id=0 can use
            # a queue
            self.client._send_to_scheduler({"op": "client-desires-keys", "keys": keys, "client": CLIENT_KEY})

            # Topic event: with precompute, ``futures`` lists one entry per (bridge, reduction) pair with the partial's
            # reduced shape/dtype; the Deisa side rebuilds the graph from these small partials.
            futures_payload: List[Dict[str, Any]]
            if all_partials_meta:
                # One entry per (bridge, reduction). ``chunk_position`` is the contributing bridge's MPI coords
                # (matching the chunk-grid layout ``mean_agg``/``moment_agg`` expect); ``chunk_axis`` is the reduction
                # axis tuple used to compute the combine output shape.
                futures_payload = []
                for partial_meta in all_partials_meta:
                    futures_payload.extend(
                        _build_futures_payload(
                            partial_meta["precomputed"],
                            partial_meta["chunk_position"],
                        )
                    )
            else:
                # Full-chunk path: one entry per bridge with the full-chunk shape.
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
            if menus:
                # Every participating bridge's eager menu, carried with the timestep it was computed for. Absent from
                # the payload on the legacy path, so that payload stays byte-identical there. Encoded with
                # ``EagerMenu.as_payload`` because the event bus (``Client.log_event``) carries msgpack primitives only:
                # a live ``EagerMenu`` (and the numpy scalars/arrays its values hold) would be refused by the client.
                to_send["menu"] = [menu.as_payload() for menu in menus]
            if menu_errors:
                # A bridge could not build its menu for this timestep: the payload carries the cause so the analytics
                # side fails the RUN at its cycle boundary instead of serving a silently menu-less delivery.
                to_send["menu_error"] = " | ".join(menu_errors)
            logger.debug(
                f"[{self.id}] send() log_event: array={array_name}, "
                f"timestep={timestep}, n_futures={len(futures_payload)}"
            )
            self.client.log_event(array_name, to_send)

        # TODO: what to do if error ?

    def _start_runtime_plan_pumps(self) -> None:
        """Attach the runtime reduction transport of every array this bridge roots: one plan-request thread each.

        The transport is a pair of dask queues per array, named by convention (``plan_request_queue_name`` /
        ``plan_report_queue_name``), so the host and the bridges agree without exchanging anything. Only the rank that
        owns the array's dask client can reach them, which is the array sub-comm's root -- and that is enough: the root
        broadcasts a request to its peers at the next send boundary, where every rank of the array is already parked in
        a sub-comm collective.

        Two things are deliberately NOT gated:

        - the per-array :class:`PlanRequestBacklog`, which every participating rank gets unconditionally, because its
          presence is what makes ``send`` enter the boundary collective. Gating it on a per-process setting made the
          ``send`` boundary a ``bcast`` on one rank and a plain scatter on another: two different collectives on the
          same communicator, i.e. a hang instead of a loud refusal. The array's participating set is the same
          everywhere, so the backlog is too.
        - the broadcast itself: every rank enters it every timestep, and a rank with no request simply gets an empty
          list back. The cost is one small collective per array per ``send`` when nothing is on demand.

        The idle cost is one queue-polling thread per array on the rank that owns the dask client. It runs for the
        life of the bridge because the first on-demand request can arrive as soon as the first delivered timestep; the
        thread exists to answer it, not because a callback was registered yet. An array this rank does not participate
        in gets neither a backlog nor a pump.
        """
        for array_name, sub_comm in self._array_comms.items():
            if sub_comm is None or sub_comm is _COMM_NULL:
                continue
            self._runtime_plan_backlog[array_name] = PlanRequestBacklog()
        for array_name, sub_comm in self._array_comms.items():
            if sub_comm is None or sub_comm is _COMM_NULL:
                continue
            if sub_comm.Get_rank() != 0 or self.client is None:
                continue
            self._runtime_plan_report_queues[array_name] = Queue(plan_report_queue_name(array_name), client=self.client)
            request_queue = Queue(plan_request_queue_name(array_name), client=self.client)
            stop = threading.Event()
            self._runtime_plan_stops[array_name] = stop
            thread = threading.Thread(
                target=serve_plan_queue,
                args=(request_queue, self._runtime_plan_backlog[array_name], stop),
                name=f"deisa-plan-{array_name}-{self.id}",
                daemon=True,
            )
            thread.start()
            self._runtime_plan_threads[array_name] = thread
            logger.debug(f"[{self.id}] runtime plan pump started for array '{array_name}'")

    def _stop_runtime_plan_pumps(self) -> None:
        """Stop the plan-request threads and wait briefly for them to leave their queue poll.

        Called from ``close()`` before the client goes away: the threads poll a dask queue through that client, so they
        must be off the queue first. A thread that does not stop within the timeout is left to the daemon flag rather
        than blocking the teardown of the whole bridge.
        """
        stops = list(getattr(self, "_runtime_plan_stops", {}).values())
        for stop in stops:
            stop.set()
        threads = getattr(self, "_runtime_plan_threads", {})
        for thread in threads.values():
            thread.join(timeout=5.0)
        stale = [thread.name for thread in threads.values() if thread.is_alive()]
        if stale:  # pragma: no cover - only reachable on a wedged scheduler
            logger.warning(f"[{self.id}] runtime plan pump(s) {stale} did not stop within the timeout")
        threads.clear()
        getattr(self, "_runtime_plan_stops", {}).clear()
        getattr(self, "_runtime_plan_report_queues", {}).clear()

    def _serve_pending_plans(self, array_name: str, sub_comm: Any) -> None:
        """Answer the on-demand reduction requests that arrived since the previous ``send`` for ``array_name``.

        Called at the send boundary, where every rank of the array is about to enter the same collectives: the root
        broadcasts the requests it collected, every rank folds its OWN chunk with the requested program, and the root
        posts the reports back to the host. The chunk is this rank's last-sent chunk, which is why it is retained --
        the host's request for timestep N necessarily arrives after ``send`` returned for N.

        A request whose timestep this rank no longer holds is answered with an error report by ``PlanResponder``: the
        host refuses the set rather than merging a partial from another timestep.

        - ``:param array_name:`` Registered array the send is for.
        - ``:param sub_comm:`` The array's sub-communicator (this rank is a member).
        """
        backlog = self._runtime_plan_backlog.get(array_name)
        if backlog is None:
            return
        requests = backlog.drain()  # only the root has a pump thread; every other rank has an empty backlog
        if sub_comm.Get_size() > 1:
            requests = sub_comm.bcast(requests, root=0)
        if not requests:
            return
        served_timestep, served_chunk = self._runtime_last_chunk.get(array_name, (None, None))
        rank = sub_comm.Get_rank()
        metadata = self.arrays_metadata.get(array_name, {})
        report_queue = self._runtime_plan_report_queues.get(array_name)
        for plan in requests:
            responder = PlanResponder(
                array_name,
                served_timestep,
                served_chunk,
                rank=rank,
                chunk_position=metadata.get("chunk_position", ()),
            )
            report = responder.respond(plan)
            reports = sub_comm.gather(report, root=0) if sub_comm.Get_size() > 1 else [report]
            if rank == 0 and report_queue is not None:
                for answered in reports or ():
                    report_queue.put(answered)
            logger.debug(
                f"[{self.id}] served plan {plan.request_id} for array '{array_name}' at timestep {served_timestep}"
            )

    def _build_runtime_menu(
        self,
        array_name: str,
        chunk: np.ndarray,
        timestep: int,
    ) -> Tuple[Optional[EagerMenu], Optional[str]]:
        """Build this bridge's eager scalar menu for one timestep: ``(menu, error)``.

        The menu is computed on the bridge's OWN numpy chunk, so it obeys the standing invariant: what leaves this
        method is a handful of chunk-local scalars and the two small ``{n, total[, M]}`` dict partials, never the chunk.
        It is called from ``send()`` before the first scatter.

        One gate, deliberate: the array must be a runtime-delivery target, i.e. the host declared (over the handshake)
        that at least one callback registered with ``precompute=True`` reads it. A deployment whose callbacks are all
        ``precompute=False``, or that registered callbacks on other arrays, would otherwise pay six chunk folds per
        bridge per timestep for a menu nobody could read (measured at 2.8-45.2 ms per bridge per timestep at
        2-32 MB).

        The failure contract is fail-loud but NOT fail-the-send: a menu entry that cannot fold its chunk (an empty
        chunk, a numeric edge) is returned as ``error`` and shipped with the topic event, where the analytics side
        records it as the run failure and raises it at the cycle boundary. Raising here instead would abort the
        simulation's send loop with no diagnostic on the analytics side, and dropping it silently would deliver a
        menu-less timestep that looks healthy.

        - ``:param array_name:`` Registered array the chunk belongs to.
        - ``:param chunk:`` The bridge's own numpy chunk, already available at ``send()``.
        - ``:param timestep:`` Timestep the chunk belongs to; every menu value is stamped with it, which is what lets
          the analytics side refuse a menu value of an earlier timestep.
        """
        if array_name not in self._runtime_delivery_arrays:
            return None, None
        metadata = self.arrays_metadata[array_name]
        collection = DeisaCollection.from_metadata(array_name, timestep, metadata)
        sub_comm = self._array_comms.get(array_name)
        rank = 0 if sub_comm is None or sub_comm is _COMM_NULL else sub_comm.Get_rank()
        try:
            return (
                build_eager_menu(
                    collection,
                    chunk,
                    rank=rank,
                    chunk_position=metadata.get("chunk_position", ()),
                ),
                None,
            )
        except Exception as exc:  # the fail-loud path: ship the cause, abort at the cycle boundary
            logger.error(f"[{self.id}] eager menu failed for array {array_name!r} at timestep {timestep}: {exc!r}")
            return None, f"array {array_name!r} at timestep {timestep}, bridge {rank}: {exc!r}"

    def _direct_send(
        self,
        array_name: str,
        res: dict,
        chunk: np.ndarray,
        timestep: int,
        precomputed_meta: Optional[Dict[str, Dict]] = None,
        menu: Optional[EagerMenu] = None,
        menu_error: Optional[str] = None,
    ) -> None:
        """
        Handle single-bridge array send without collective.

        For arrays that exist on only one bridge, we skip the gather() entirely and directly update the Dask scheduler.

        - ``:param array_name:`` The array name being sent.
        - ``:param res:`` The scatter result (full-chunk: dict with a single ``future``;
            precompute: dict with a list ``future`` of all partial keys).
        - ``:param chunk:`` The numpy ndarray data chunk (kept for full-chunk shape/dtype).
        - ``:param timestep:`` The current timestep.
        - ``:param precomputed_meta:`` Per-partial scatter metadata
            (``{output_key: {"future", "shape", "dtype"}}``); only set on the
            precompute path. When provided, the topic event's ``futures`` list
            carries the partials' reduced shapes instead of the full-chunk shape.
        - ``:param menu:`` This bridge's eager scalar menu for ``timestep``, or ``None`` when the runtime path is off.
            A single-bridge array is its own whole grid, so the one menu is also the complete menu of the array.
        """
        assert self.client is not None, "client cannot be None for single-bridge send."

        # On the precompute path, ``res["future"]`` is a list of partial keys (one per reduction). On the non-precompute
        # path, it's a single future key.
        future_keys = res["future"] if isinstance(res["future"], list) else [res["future"]]
        who_has = res["who_has"]
        nbytes = res["nbytes"]

        self.client.sync(self.client.scheduler.update_data, who_has=who_has, nbytes=nbytes)
        self.client._send_to_scheduler({"op": "client-desires-keys", "keys": future_keys, "client": CLIENT_KEY})

        # Build the topic event. On the precompute path, emit one entry per partial (with its reduced shape). On the
        # non-precompute path, emit one entry pointing at the full chunk.
        if precomputed_meta:
            futures_payload = _build_futures_payload(
                precomputed_meta,
                self.arrays_metadata[array_name]["chunk_position"],
            )
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
            "precomputed": True if precomputed_meta else None,
            "futures": futures_payload,
        }
        if menu is not None:
            to_send["menu"] = [menu.as_payload()]
        if menu_error is not None:
            to_send["menu_error"] = menu_error
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

    def _scatter_full(self, data: np.ndarray, workers: List[str] = None, hash=False):
        """Scatter data to workers and return the legacy ``{"future", "who_has", "nbytes"}`` result."""
        logger.debug(f"[{self.id}] scatter to {workers}")

        if workers is None:
            workers = self.workers

        if self.client:
            return self.client.sync(self._scatter_blocking, workers, data, hash=hash)
        else:
            return self._run_async(self._scatter_blocking(workers, data, hash=hash))

    def _run_async(self, coro):
        """Run ``coro`` to completion from synchronous code.
        ``asyncio.run`` cannot be called when the calling thread already has a running event loop; it raises
        ``RuntimeError: asyncio.run() cannot be called from a running event loop``.
        The no-client path reaches this from bridge teardown, where a loop is frequently still running on this thread.
        Use the existing loop when there is one, and ``asyncio.run`` otherwise.
        """
        try:
            # Probe for a running loop on this thread. We are in synchronous code, so a running loop here cannot make
            # progress while we block on run_until_complete, that would deadlock.
            # Hand the coroutine to a private loop on a worker thread instead, which runs concurrently with the caller's
            # loop.
            asyncio.get_running_loop()
        except RuntimeError:
            # No loop on this thread: asyncio.run is safe and needs no cleanup.
            return asyncio.run(coro)

        return run_coro_on_private_loop(coro)

    async def _scatter_blocking(self, workers, data, hash=False):
        """Scatter ``data`` to ``workers`` and return the legacy per-key result (one future key per element)."""
        if isinstance(workers, (str, Number)):
            workers = [workers]

        input_type = type(data)
        names: List[str] = []
        unpack = False
        if isinstance(data, Iterator) or isinstance(data, type(range(0))):
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

        _, who_has, nbytes = await scatter_to_workers(list(workers), data2)

        out = {k: {"future": k, "who_has": who_has, "nbytes": nbytes} for k in data}

        if issubclass(input_type, (list, tuple, set, frozenset)):
            out = input_type(out[k] for k in names)

        if unpack:
            assert len(out) == 1
            out = list(out.values())[0]
        return out

    async def _scatter_to_workers_async(self, workers: List[str], data: Dict[str, Any]):
        """Scatter an already-keyed payload dict to ``workers``; returns ``(who_has, nbytes)``.

        Used on the precompute path, which builds its own payload keys and only needs the scheduling pair.
        """
        _, who_has, nbytes = await scatter_to_workers(list(workers), data)
        return who_has, nbytes

    def _fetch_task_state_bcast(self, array_name: str) -> Optional[dict]:
        """Mirror of the per-array bcast protocol for an array's scheduling state.

        Sub-comm rank 0 reads ``{"branches", "runtime"}`` from the handshake actor and broadcasts it; every other
        participating rank receives via ``bcast(None)``. Both facts travel in ONE collective on purpose: a second
        broadcast would have to be entered in the same order on every rank of the sub-comm, i.e. one more way to hang
        the array for one dict entry. Returns None when this bridge does not participate in the array's sub-comm.
        Callers decide caching (None is NOT a cached result)."""
        sub_comm = self._array_comms.get(array_name)
        if sub_comm is None or sub_comm is _COMM_NULL:
            return None
        if sub_comm.Get_rank() == 0 and self.handshake is not None:
            state = self.handshake.get_task_state(array_name)
            sub_comm.bcast(state, root=0)
        else:
            state = sub_comm.bcast(None, root=0)
        return state

    def _cache_task_state(self, array_name: str, state: dict) -> None:
        """Store one array's fetched scheduling state in the local caches.

        - ``:param array_name:`` Registered array the state belongs to.
        - ``:param state:`` The ``{"branches", "runtime"}`` dict fetched (or broadcast) for it.
        """
        self._task_branches[array_name] = state.get("branches") or []
        if state.get("runtime"):
            self._runtime_delivery_arrays.add(array_name)
        else:
            self._runtime_delivery_arrays.discard(array_name)

    def _get_task_branches(self, array_name: str) -> List[BranchSpec]:
        """
        Retrieve stored task hints for an array.

        Branches are fetched from HandshakeActor on sub_comm rank 0 and broadcast to all ranks.
        If no hints are available, this method returns an empty list (no precomputation).

        - ``:param array_name:`` The array name to get hints for.
        - ``:return:`` List of reduction hints (each carrying a pickled chunk
            callable and the dask kwargs to apply).
        """
        # Cache hit first: after one fetch, sends never touch the handshake.
        cached = self._task_branches.get(array_name)
        if cached is not None:
            return cached

        # Cache miss -> shared per-array bcast protocol (lazy path: wait_for_go=False or a cache miss).
        state = self._fetch_task_state_bcast(array_name)
        if state is not None:
            # Store unconditionally: [] is a valid cached result meaning "no precompute, full-chunk scatter" -- don't
            # re-fetch per send.
            self._cache_task_state(array_name, state)
        return self._task_branches.get(array_name) or []

    def _prefetch_task_branches(self) -> None:
        """Prefetch static task state for every global array into the local cache.

        On the wait_for_go path the handshake is complete by the time this
        runs (go event + world barrier), so broadcasting per array removes
        branch fetches from the send() critical path. The broadcast mirrors
        ``_get_task_branches``: sub-comm rank 0 reads the handshake actor and
        broadcasts; non-participating ranks leave the None sentinel. A
        branch-less array ([]) is cached as-is.
        """
        for array_name in self._global_array_names:
            state = self._fetch_task_state_bcast(array_name)
            if state is not None:
                self._cache_task_state(array_name, state)

    def _get_branch_by_key(self, array_name: str, branches: List[BranchSpec]) -> Dict[str, Any]:
        """Return the ``output_key -> branch`` lookup for an array, building and caching it on first use.

        Branches are static after registration, so the indexed map is not rebuilt on every ``send()``.
        """
        cached = self._branch_by_key.get(array_name)
        if cached is not None:
            return cached

        indexed = {b.output_key: b for b in branches}
        self._branch_by_key[array_name] = indexed
        return indexed

    def _scatter_partials(
        self,
        partials: Dict[str, Any],
        branches: List[BranchSpec],  # List[BranchSpec]
        array_name: str,
        workers: List[str],
    ) -> Dict[str, Any]:
        """Scatter precomputed reduction partials to a worker instead of the full chunk.

        Partial flavors: ``scalar`` (sum/prod/max/min: plain numbers/arrays,
        one future key per reduction), ``mean`` (``{"n", "total"}`` dict,
        combined with ``mean_agg``) and ``moment`` (``{"n", "total", "M"}``
        dict for var/std, combined with ``moment_agg`` + sqrt). Returns the
        legacy scatter result plus a ``precomputed`` map of ``output_key``
        to ``{future, kind, shape, dtype, finalize}`` metadata so the topic
        handler can reconstruct the right dask graph.
        """
        assert len(workers) == 1, "_scatter_partials expects a single target worker"
        target_worker = workers[0]

        # Index branches by output_key (cached; BranchSpec carries the analyzer's per-reduction metadata directly).
        branch_by_key = self._get_branch_by_key(array_name, branches)

        payload: Dict[str, Any] = {}
        shape_dtype: Dict[str, Dict[str, Any]] = {}
        for output_key, value in partials.items():
            branch = branch_by_key.get(output_key)
            if branch is None:
                # Every partial comes from a branch (they are built together in _execute_operations_on_chunk).
                raise KeyError(f"no branch found for precomputed output_key {output_key!r}")
            key = f"{KEY_PREFIX}{array_name}-partial-{output_key}-{uuid.uuid4().hex}"
            payload[key] = value
            shape_dtype[output_key] = {
                "future": key,
                "kind": branch.output_kind,
                "shape": branch.partial_shape,
                "dtype": branch.partial_dtype,
                "finalize": branch.finalize,
                "chunk_axis": branch.chunk_axis,
                "op_name": branch.op_name,
            }

        # Serialize for scatter (handles numpy arrays in dict values).
        payload2 = valmap(to_serialize, payload)

        # scatter_to_workers directly for the (who_has, nbytes) pair. client.sync when a Client is available,
        # asyncio.run otherwise (only rank 0 has a Client).
        if self.client is not None:
            who_has, nbytes = self.client.sync(self._scatter_to_workers_async, [target_worker], payload2)
        else:
            who_has, nbytes = self._run_async(self._scatter_to_workers_async([target_worker], payload2))

        future_keys = list(payload.keys())
        return {
            "future-info": {
                "future": future_keys,
                "who_has": who_has,
                "nbytes": nbytes,
            },
            "precomputed": shape_dtype,
        }

    def _execute_operations_on_chunk(self, chunk: np.ndarray, branches: List["BranchSpec"]) -> Dict[str, Any]:
        """
        Execute branch chunk-stage callables locally on the bridge's numpy chunk before scattering.

        Each branch in ``branches`` is a :class:`BranchSpec` whose ``branch_func`` is a pickle-able Python callable
        that takes a numpy chunk and returns the per-bridge partial (a scalar, ndarray, or dict for mean/moment).
        The closure inside ``branch_func`` already binds the analyzer's chunk kwargs (axis, keepdims, dtype, ...),
        so the call here is ``branch_func(chunk)`` with no extra arguments.

        - ``:param array_name:`` The array name being processed.
        - ``:param chunk:`` The numpy ndarray data chunk.
        - ``:param branches:`` List of :class:`BranchSpec` from ``analyze_branch``.
        - ``:return:`` Dict of partial results keyed by output_key.
        """
        partials = {}
        for branch in branches:
            output_key = branch.output_key
            try:
                partial = branch.branch_func(chunk)
            except Exception as e:
                # A dropped partial silently corrupts the combined reduction (scalar stacks get smaller sums,
                # mean_agg/moment_agg miss a whole bridge's n). FAIL LOUDLY instead of skipping.
                raise PrecomputeRuntimeError(
                    f"[{self.id}] _execute_operations_on_chunk: branch {output_key!r} failed on the local "
                    f"chunk: {e!r}. Refusing to ship a partial set that would produce a wrong reduction."
                ) from e
            partials[output_key] = partial

        logger.debug(f"[{self.id}] _execute_operations_on_chunk: {partials}")
        return partials


def _build_futures_payload(meta: Mapping[str, Mapping[str, Any]], chunk_position: Any) -> List[Dict[str, Any]]:
    """Build per-reduction ``futures`` entries for a precompute topic event.

    One entry per reduction in ``meta``, carrying the partial's reduced
    shape/dtype, the reduction's ``chunk_axis`` and the caller-provided
    ``chunk_position``; shared by multi-bridge gather and the single-bridge
    fast path so both byte-identically shape the payload.
    """
    return [
        {
            "future": info["future"],
            "shape": info["shape"],
            "dtype": info["dtype"],
            "kind": info.get("kind", "scalar"),
            "finalize": info.get("finalize"),
            "chunk_position": chunk_position,
            "chunk_axis": info.get("chunk_axis"),
            "op_name": info.get("op_name"),
            "output_key": output_key,
        }
        for output_key, info in meta.items()
    ]
