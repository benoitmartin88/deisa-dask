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
import sys
import uuid
from collections import deque, defaultdict
from numbers import Number
from itertools import cycle
from typing import Any, Iterator, List, Dict, Optional, Union, Deque

import numpy as np
from dask.tokenize import tokenize
from deisa.core import IBridge, ICommunicator
from distributed import Queue, Client
from distributed.worker import _global_workers
from distributed.core import rpc
from distributed.utils import All
from distributed.protocol import to_serialize
from tlz import drop, groupby, merge, valmap

from deisa.dask.constants import KEY_PREFIX, FEEDBACK_QUEUE_PREFIX, CLIENT_KEY
from deisa.dask.handshake import Handshake
from deisa.dask.utils import get_client

logger = logging.getLogger(__name__)

_round_robin_counter = [0]

# Sentinel values for MPI.Comm.Split() — used until ICommunicator is updated
# to expose these constants. These mirror mpi4py's MPI.UNDEFINED and MPI.COMM_NULL.
_UNDEFINED = -1
_COMM_NULL = None


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

        - ``:type arrays_metadata:`` Dict[str, Dict]
        - ``:param args:`` Additional positional arguments for the initialization.
        - ``:param kwargs:`` Additional keyword arguments for the initialization. Can include
            configuration parameters like timeout used during client setup.
        """
        super().__init__(comm, arrays_metadata, *args, **kwargs)
        self.comm: ICommunicator = comm
        self.id = self.comm.Get_rank()
        self.arrays_metadata = arrays_metadata
        self._my_arrays: set[str] = set(arrays_metadata.keys())  # arrays this bridge owns
        self._feedback_queues = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self._has_close_been_called = False
        self.workers = None
        self.handshake = None
        self.client: Optional[Client] = None
        self._array_comms: Dict[str, Any] = {}  # array_name -> sub-comm (from comm.Split)

        # Barrier: ensure all bridges have entered __init__ before proceeding to collectives
        self.comm.barrier()

        if self.id == 0:
            # only id 0 has a real dask client
            self.client = get_client(timeout=kwargs.get("timeout", 10), name="bridge")
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

        # Ensure all bridges have finished __init__ before any proceeds to handshake
        self.comm.barrier()

        if self.id == 0:
            # all bridges are ready, tell handshake actor
            assert self.client is not None, "client cannot be None for Bridge id 0."
            self.handshake = Handshake(self.client)
            # Send merged metadata (from all bridges) to the handshake actor
            metadata_for_handshake = getattr(self, '_handshake_metadata', self.arrays_metadata)
            self.handshake.all_bridges_ready(nb_bridge=self.comm.Get_size(),
                                             arrays_metadata=metadata_for_handshake, **kwargs)

            # Fetch task hints from HandshakeActor and broadcast to all bridges
            task_hints = self.handshake.get_task_hints_dict()
        else:
            task_hints = {}

        # Broadcast task hints to all bridges (rank 0 has them, others empty)
        task_hints = self.comm.bcast(task_hints, root=0)
        self._task_hints = task_hints

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
            # Single-bridge case: nothing to collectivate with
            self._global_array_names = set(self._my_arrays)
            return

        all_metadata = self.comm.gather(self.arrays_metadata, root=0)

        if self.id == 0:
            # Build global set of array names
            global_names = set()
            for partial in all_metadata:
                global_names.update(partial.keys())
            # Build merged metadata for handshake/Deisa:
            # For each array, take the metadata from the first bridge that declared it
            merged = {}
            for array_name in global_names:
                for partial in all_metadata:
                    if array_name in partial:
                        merged[array_name] = partial[array_name]
                        break
            self._handshake_metadata = merged
        else:
            global_names = None

        # Broadcast the global set of array names to all bridges
        self.comm.barrier()  # ensure all ranks reach bcast together
        global_names = self.comm.bcast(global_names, root=0)
        self._global_array_names = global_names

    def _setup_array_comms(self):
        """
        Create per-array sub-communicators using comm.Split().

        The global array list (from _gather_global_metadata) is known to all
        bridges, so every rank can call Split() for every array. Bridges that
        don't own a given array use color=_UNDEFINED and receive _COMM_NULL.

        The color is derived from a consistent hash of the array name so
        that all participating ranks get the same sub-communicator group.
        """
        for array_name in self._global_array_names:
            if array_name in self._my_arrays:
                color = hash(array_name) % (2**31)
            else:
                color = _UNDEFINED
            key = self.id  # preserve ordering
            sub_comm = self.comm.Split(color, key)
            self._array_comms[array_name] = sub_comm
            logger.debug(
                f"[{self.id}] _setup_array_comms: array={array_name}, "
                f"participates={array_name in self._my_arrays}, sub_comm_size="
                f"{sub_comm.Get_size() if sub_comm is not None and sub_comm.Get_size() > 0 else 'NULL'}"
            )

    def __del__(self):
        """Clean up resources before destruction."""
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
                    if sub_comm.Get_size() > 0:
                        sub_comm.Free()
                        logger.debug(f"[{self.id}] Freed sub-communicator for array '{array_name}'")
                self._array_comms.clear()
                if self.id == 0:
                    assert self.handshake, "handshake cannot be None for Bridge id 0."
                    assert self.client, "client cannot be None for Bridge id 0."
                    self.handshake.set_bridges_done(timestep=timestep)
                    self.client.close()
        except Exception as e:
            logger.error(f"[{self.id}] Cloud not cleanly close bridge. exception={e}")

    def send(self, array_name: str, chunk: np.ndarray, timestep: int, *args, **kwargs):
        """
        Distribute a data chunk to a Dask worker via scatter + gather.

        Scatters the chunk to the next worker (round-robin), then gathers all
        chunks across bridges to rank 0, which updates the Dask scheduler.

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

        if kwargs.get('update_workers', False):
            # only update worker list if requested
            if self.id == 0:
                assert self.client is not None, "client cannot be None for Bridge id 0."
                # rank 0 retrieve workers and bcast to other bridges
                workers = self.client.scheduler_info(n_workers=-1)["workers"]

            # bcast
            logger.debug(f"[{self.id}] send() pre-bcast workers={workers}")
            self.workers = self.comm.bcast(workers, root=0)
            logger.debug(f"[{self.id}] send() post-bcast workers={workers}")
            workers = dict(self.workers)

        if kwargs.get('filter_workers', False):
            workers = kwargs['filter_workers'](workers)
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

        # === Execute local operations on chunk BEFORE scatter ===
        partials = self._execute_operations_on_chunk(array_name, chunk)

        # Send data to worker
        res = self._better_scatter(chunk, workers=workers, hash=False)  # send data to workers

        # Get per-array metadata
        meta = self.arrays_metadata[array_name]

        # === Determine communicator from cached sub-comms (from comm.Split()) ===
        sub_comm = self._array_comms.get(array_name)

        if sub_comm.Get_size() == 0:
            # This rank doesn't participate in this array (COMM_NULL from Split)
            logger.debug(f"[{self.id}] send() rank not in participating set for '{array_name}', skipping")
            return

        comm_to_use = sub_comm if sub_comm.Get_size() > 0 else self.comm
        sub_comm_size = comm_to_use.Get_size()

        # === Single-bridge fast-path: no collective needed ===
        if sub_comm_size == 1:
            self._direct_send(array_name, res, chunk, timestep, partials)
            return

        to_send = {
            'future-info': res,
            'chunk_position': meta['chunk_position'],
            'partials': partials
        }
        logger.debug(f"[{self.id}] send() gather: to_send={to_send}")

        gathered_data = comm_to_use.gather(to_send, root=0)
        logger.debug(f"[{self.id}] send() gathered_data={gathered_data}")

        if gathered_data is not None:
            assert self.client is not None, "client cannot be None for Bridge id 0."
            # rank 0 (root=0 in comm.gather): aggregate who_has from all chunks
            who_has = {}
            nbytes = {}
            keys = []
            all_partials = []
            for d in gathered_data:
                who_has.update(d['future-info']['who_has'])
                nbytes.update(d['future-info']['nbytes'])
                keys.append(d['future-info']['future'])
                if d.get('partials'):
                    all_partials.append(d['partials'])

            # only update the scheduler with who has what and register the future once
            self.client.sync(self.client.scheduler.update_data, who_has=who_has, nbytes=nbytes)

            # mimic mechanism from Queue. Keep a reference on keys until reception in topic handler.
            # TODO: id=0 can use a queue
            self.client._send_to_scheduler({"op": "client-desires-keys", "keys": keys, "client": CLIENT_KEY})

            # === Combine reduction partials on rank 0 ===
            precomputed = self._combine_reduction_partials(all_partials)

            to_send = {
                'array_name': array_name,
                'iteration': timestep,
                'futures': [{
                    'future': d['future-info']['future'],
                    'shape': chunk.shape,
                    'dtype': str(chunk.dtype),
                    'chunk_position': d['chunk_position']
                } for d in gathered_data],
                'precomputed': precomputed if precomputed else None
            }
            logger.debug(f"[{self.id}] send() log_event: array={array_name}, timestep={timestep}, n_futures={len(gathered_data)}")
            self.client.log_event(array_name, to_send)

        # TODO: what to do if error ?

    def _direct_send(self, array_name: str, res: dict, chunk: np.ndarray, timestep: int, partials: Optional[Dict] = None):
        """
        Handle single-bridge array send without collective.

        For arrays that exist on only one bridge, we skip the gather() entirely
        and directly update the Dask scheduler.

        - ``:param array_name:`` The name of the data array being sent.
        - ``:param res:`` The scatter result dict containing future, who_has, and nbytes.
        - ``:param chunk:`` The numpy ndarray data chunk.
        - ``:param timestep:`` The current timestep.
        - ``:param partials:`` Optional precomputed partial results.
        - ``:return:`` None
        """
        assert self.client is not None, "client cannot be None for single-bridge send."

        future_key = res['future']
        who_has = res['who_has']
        nbytes = res['nbytes']

        self.client.sync(self.client.scheduler.update_data, who_has=who_has, nbytes=nbytes)
        self.client._send_to_scheduler({
            "op": "client-desires-keys",
            "keys": [future_key],
            "client": CLIENT_KEY
        })
        to_send = {
            'array_name': array_name,
            'iteration': timestep,
            'futures': [{
                'future': future_key,
                'shape': chunk.shape,
                'dtype': str(chunk.dtype),
                'chunk_position': self.arrays_metadata[array_name]['chunk_position']
            }],
            'precomputed': partials if partials else None
        }
        self.client.log_event(array_name, to_send)

    def _execute_operations_on_chunk(self, array_name: str, chunk: np.ndarray) -> Optional[Dict]:
        """
        Execute reducible operations on numpy chunk before scatter.
        
        Uses task hints stored during __init__ (broadcast from HandshakeActor)
        and executes them on the local chunk.
        
        - ``:param array_name:`` The name of the array being sent.
        - ``:param chunk:`` The numpy ndarray data chunk.
        - ``:return:`` Dict of partial results or None.
        """
        hints = self._task_hints.get(array_name)
        if not hints:
            return None
            
        partials = {}
        for hint in hints:
            op = hint.get('op')
            output_key = hint.get('output_key', f"{array_name}-{op}")
            
            if op == 'sum':
                partials[output_key] = float(chunk.sum())
            elif op == 'max':
                partials[output_key] = float(chunk.max())
            elif op == 'min':
                partials[output_key] = float(chunk.min())
            elif op == 'mean':
                partials[f"{output_key}-sum"] = float(chunk.sum())
                partials[f"{output_key}-count"] = int(chunk.size)
                
        return partials if partials else None

    def _combine_reduction_partials(self, all_partials: List[Dict]) -> Optional[Dict]:
        """
        Combine partial reduction results from all bridges on rank 0.
        
        - ``:param all_partials:`` List of partial result dicts from each bridge.
        - ``:return:`` Combined results dict or None.
        """
        if not all_partials:
            return None
            
        combined: Dict[str, Any] = {}
        
        for key in set().union(*[p.keys() for p in all_partials if p]):
            values = [p.get(key) for p in all_partials if p.get(key) is not None]
            
            # Detect mean keys (end with -sum or -count)
            if key.endswith('-sum'):
                base_key = key[:-4]  # remove '-sum'
                count_key = f"{base_key}-count"
                count_values = [p.get(count_key) for p in all_partials if p.get(count_key) is not None]
                if values and count_values:
                    combined[base_key] = float(sum(float(v) for v in values if v is not None)) / float(sum(float(v) for v in count_values if v is not None))
            # Skip count keys (they're handled with their sum)
            elif key.endswith('-count'):
                continue
            # Regular reduction keys
            elif values:
                if key == 'sum':
                    combined[key] = float(sum(float(v) for v in values if v is not None))
                elif key == 'max':
                    combined[key] = max(float(v) for v in values if v is not None)
                elif key == 'min':
                    combined[key] = min(float(v) for v in values if v is not None)
                else:
                    try:
                        combined[key] = float(sum(float(v) for v in values if v is not None))
                    except (TypeError, ValueError):
                        pass
            
        return combined if combined else None

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
                    'q': Queue(f'{FEEDBACK_QUEUE_PREFIX}{key}', client=self.client, maxsize=feedback_queue_size),
                    'deque': deque(maxlen=feedback_queue_size)}

            q: Queue = fb_state[key]['q']
            d: deque = fb_state[key]['deque']

            if q.qsize() != 0:
                # List[(int, Any), ...]\n                full_q = q.get(batch=True)  # get all elements. This pops elements from the Dask queue.
                for v in full_q: d.append(v)  # add all elements to deque
            logger.debug(f"[{self.id}] get() fb_state={fb_state}")

        d = self.comm.bcast(fb_state[key]['deque'], root=0)

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

        client = self.client if self.client else get_client(name="bridge")
        return client.sync(
            self.__scatter,
            data,
            workers=workers,
            hash=hash)

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

        _, who_has, nbytes = await self.scatter_to_workers(workers, data)

        out = {
            k: {
                'future': k,
                'who_has': who_has,
                'nbytes': nbytes
            }
            for k in data
        }

        if issubclass(input_type, (list, tuple, set, frozenset)):
            out = input_type(out[k] for k in names)

        if unpack:
            assert len(out) == 1
            out = list(out.values())[0]
        return out

    async def scatter_to_workers(self, workers, data):
        """
        Internal scatter implementation supporting in-process zero-copy.

        - ``:param workers:`` List of worker addresses to scatter to.
        - ``:param data:`` Dictionary mapping keys to serialized data values.
        """
        assert isinstance(data, dict)

        local_worker_map = {w.address: w for w in _global_workers}

        workers = sorted(workers)
        names = list(data.keys())

        worker_iter = drop(_round_robin_counter[0] % len(workers), cycle(workers))
        _round_robin_counter[0] += len(data)

        items = list(zip(worker_iter, names, data.values()))

        who_has: dict = {}
        nbytes: dict = {}

        # In-process, zero-copy
        local_items = [(w, k, v) for w, k, v in items if w in local_worker_map]
        if local_items:
            local_who_has = {}
            local_nbytes = {}
            for addr, key, val in local_items:
                worker = local_worker_map[addr]
                worker.update_data({key: val})
                assert id(worker.data[key]) == id(val), \
                    f"In-process scatter copied data: id(orig)={id(val)}"
                size = int(val.nbytes) if hasattr(val, "nbytes") else sys.getsizeof(val)
                who_has[key] = [addr]
                nbytes[key] = size
                local_who_has[key] = [addr]
                local_nbytes[key] = size

            if self.client:
                await self.client.scheduler.update_data(
                    who_has=local_who_has,
                    nbytes=local_nbytes,
                    client=self.client.id,
                )

        # Remote, serialize and RPC
        remote_items = [(w, k, v) for w, k, v in items if w not in local_worker_map]
        if remote_items:
            d = {
                worker: {key: value for _, key, value in v}
                for worker, v in groupby(0, remote_items).items()
            }
            d_serialized = {
                worker: valmap(to_serialize, chunk)
                for worker, chunk in d.items()
            }
            rpcs = {addr: rpc(addr) for addr in d_serialized}
            try:
                out = await All([
                    rpcs[address].update_data(data=v)
                    for address, v in d_serialized.items()
                ])
            finally:
                for r in rpcs.values():
                    await r.close_rpc()

            nbytes.update(merge(o["nbytes"] for o in out))
            for k, v in groupby(1, remote_items).items():
                who_has[k] = [w for w, _, _ in v]

        return (names, who_has, nbytes)