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
import collections
import logging
import threading
import time
import weakref
from typing import Any, Callable, Collection, Dict, List, Literal, Set

import numpy as np
from deisa.core import CallbackArgs, Window
from deisa.core.interface import IDeisa
from distributed import Client, Event, Future, Queue

import dask.array as da
from deisa.dask.branch import _analyze_callback_for_branches, _combine_array_from_partials
from deisa.dask.constants import (
    CALLBACK_PREFIX,
    CLIENT_KEY,
    DEFAULT_SLIDING_WINDOW_SIZE,
    FEEDBACK_QUEUE_PREFIX,
    KEY_PREFIX,
    WAIT_FOR_EXECUTE_CB_EVENT,
)
from deisa.dask.handshake import Handshake
from deisa.dask.precompute_analyzer import NoPrecomputableReductionError
from deisa.dask.utils import build_deisa_array, get_client

logger = logging.getLogger(__name__)


class Deisa(IDeisa):
    Callback_id = str

    def __init__(self, feedback_queue_size: int = 1024, *args, **kwargs) -> None:
        """
        Initializes a class instance, configuring the client and setting up the necessary
        infrastructure for interactions. This includes setting up the necessary feedback
        queue length, performing handshake operations with the client, and initializing
        various metadata structures.

        - ``:param feedback_queue_size:`` The maximum size of the feedback queue. Defaults to 1024.
        - ``:type feedback_queue_size:`` int
        - ``:param args:`` Additional positional arguments passed to the initializer.
        - ``:type args:`` tuple
        - ``:param kwargs:`` Additional keyword arguments passed to the initializer.
        - ``:type kwargs:`` dict
        """
        # dask.config.set({
        # "distributed.deploy.lost-worker-timeout": 60,
        # "distributed.workers.memory.spill":0.97,
        # "distributed.workers.memory.target":0.95,
        # "distributed.workers.memory.terminate":0.99 })

        super().__init__(feedback_queue_size, *args, **kwargs)
        self.client: Client = get_client(timeout=kwargs.get("timeout", 10), name="deisa")
        self.feedback_queue_size = feedback_queue_size

        # blocking until all bridges are ready
        self.handshake = Handshake(self.client)
        self.handshake.deisa_ready(feedback_queue_size=feedback_queue_size, **kwargs)

        self.mpi_comm_size = self.handshake.get_nb_bridges()
        self.arrays_metadata = self.handshake.get_arrays_metadata()

        self.received_metadata = dict[str, list[dict[str, Any]]]()  # array_name: list[metadata]
        self.current_sliding_windows = {}

        self._callbacks: Dict[Deisa.Callback_id, Dict] = {}
        self._callbacks_by_array: Dict[str, Set[Deisa.Callback_id]] = {}
        self._topic_handlers: Dict[str, Callable] = {}
        self._callback_seq = 0  # unique counter
        self._tasks = set()

    def __del__(self):
        try:
            if self.client:
                self.client.close()
            # delete Futures
            if self._callbacks:
                for cb in self._callbacks.values():
                    for a in cb["state"].values():
                        a["window"].clear()
                del self._callbacks
        except Exception as e:
            logger.error(f"Cloud not cleanly close deisa. exception={e}")

    @staticmethod
    def __default_exception_handler(exception: BaseException):
        logger.error(f"Exception thrown for callback id: {exception}")

    def register(
        self,
        *callback_args: CallbackArgs,
        exception_handler: IDeisa.ExceptionHandler = __default_exception_handler,
        when: Literal["AND", "OR"] = "AND",
        precompute: bool = True,
    ) -> Callable:
        """
        Registers a callback function with specific arguments, exception handling, and conditional execution criteria.

        This function acts as a decorator that allows you to register a callback with
        parameters provided through ``callback_args``. It also handles exceptions using the
        ``exception_handler`` and defines the execution rules with ``when`` parameter.

        Supports:
        Default window size is 1.
        ``@deisa.register("arr1")``                             # default window size
        ``@deisa.register("arr1", "arr2")``                     # two arrays, default window size
        ``@deisa.register(Window("arr1"))``                     # default window size
        ``@deisa.register(Window("arr1", 2))``                  # window size 2
        ``@deisa.register(Window("arr1", 2), Window("arr2", 5))``   # window size 2 for arr1 and 5 for arr2
        ``@deisa.register(Window("arr1", 2), Window("arr2", 5), "arr3")``  # window size 2 for arr1 and 5 for arr2,
                                                                             default window size for arr3

        Every callback is automatically analyzed for dask reduction operations (sum, mean, std, var, max, min, prod)
        which are executed locally on each bridge before scatter to reduce network transfer. Precompute analysis
        is always attempted (the default), and any callback that cannot be precomputed (no reductions, or a reduction
        that depends on another reduction's output) raises at registration time. Use ``precompute=False`` to skip
        the analysis with a warning and fall back to the legacy full-chunk scatter path.

        - ``:param callback_args:`` Variable-length arguments representing callback-specific parameters.
        - ``:param exception_handler:`` Optional exception handler to manage errors during callback execution.
             Defaults to ``__default_exception_handler``.
        - ``:param when:`` Specifies the conditional logic for triggering the callback. Can be 'AND' or 'OR'.
             Defaults to 'AND'.
        - ``:param precompute:`` If False, skip precompute analysis with a warning and use the full-chunk scatter path.
             Defaults to True (analysis is required).
        - ``:return:`` A callable that wraps the provided callback with the configured parameters and logic.
        - ``:rtype:`` Callable
        """

        def decorator(callback: IDeisa.Callback) -> IDeisa.Callback:
            return self.register_callback(
                callback,
                *callback_args,
                exception_handler=exception_handler,
                when=when,
                precompute=precompute,
            )

        return decorator

    def register_callback(
        self,
        callback: IDeisa.Callback,
        *callback_args: CallbackArgs,
        exception_handler: IDeisa.ExceptionHandler = __default_exception_handler,
        when: Literal["AND", "OR"] = "AND",
        precompute: bool = True,
    ) -> Callable:
        """
        Registers a callback function with specific arguments, exception handling, and conditional execution criteria.

        This function allows you to register a callback with parameters provided through ``callback_args``.
        It also handles exceptions using the ``exception_handler`` and defines the execution rules with ``when``
        parameter.

        Supports:
        Default window size is 1.
        ``@deisa.register("arr1")``                             # default window size
        ``@deisa.register("arr1", "arr2")``                     # two arrays, default window size
        ``@deisa.register(Window("arr1")) ``                    # default window size
        ``@deisa.register(Window("arr1", 2))``                  # window size 2
        ``@deisa.register(Window("arr1", 2), Window("arr2", 5))``   # window size 2 for arr1 and 5 for arr2
        ``@deisa.register(Window("arr1", 2), Window("arr2", 5), "arr3")`` # window size 2 for arr1 and 5 for arr2,
                                                                            default window size for arr3

        - ``:param callback:``  Callback function to register.
        - ``:param callback_args:``  Variable-length arguments representing callback-specific parameters.
        - ``:param exception_handler:``  Optional exception handler to manage errors during callback execution.
        - ``:param when:``  Specifies the conditional logic for triggering the callback. Can be 'AND' or 'OR'.
        - ``:param precompute:``  If False, skip precompute analysis with a warning and send the full data to workers.
        - ``:return:``  A callable that wraps the provided callback with the configured parameters and logic.
        """
        logger.debug(f"register_callback: callback={callback}, callback_args={callback_args}")
        if not callback_args:
            raise TypeError("register_callback requires at least one array name or a list of Window(name, window_size)")

        parsed: List[Window] = []

        for arg in callback_args:
            if isinstance(arg, str):
                parsed.append(Window(arg, size=DEFAULT_SLIDING_WINDOW_SIZE))
            elif isinstance(arg, Window):
                parsed.append(arg)
            else:
                raise TypeError("callback_args must be str or tuple")

        callback_id = self._register_callback_impl(
            callback, parsed, exception_handler=exception_handler, when=when, precompute=precompute
        )
        callback.callback_id = callback_id
        return callback

    def _register_callback_impl(
        self,
        callback: IDeisa.Callback,
        parsed: List[Window],
        exception_handler: IDeisa.ExceptionHandler,
        when: Literal["AND", "OR"],
        precompute: bool = True,
    ) -> Callback_id:

        if when not in ("AND", "OR"):
            raise ValueError("when must be 'AND' or 'OR'")

        for array_name, _ in parsed:
            if array_name not in self.arrays_metadata:
                raise ValueError(f"unknown array name: {array_name}")

        array_names = [w.name for w in parsed]
        callback_id = self.__next_callback_id()

        logger.debug(f"_register_callback_impl: register callback_id={callback_id}")

        # per-callback state
        callback_state = {
            arr_name: {"window": collections.deque(maxlen=ws), "changed": False, "last_iteration": None}
            for arr_name, ws in parsed
        }

        self._callbacks[callback_id] = {
            "callback": callback,
            "when": when,
            "exception_handler": exception_handler,
            "array_names": array_names,
            "state": callback_state,
        }

        # Analyze all registered arrays together (single pass, not per-array loop).
        # The method takes the full {name: stub} dict so cross-array callbacks (e.g. cb(temperature, pressure))
        # are handled in one analysis.

        # ``precompute=False`` is the explicit opt-out of precompute: skip the analysis entirely and fall back to the
        # full-chunk scatter path. This is the documented contract (see ``register``) and what the
        # test_no_precompute_worker_sees_full_chunk test expects.

        if not precompute:
            logger.warning(
                f"_register_callback_impl: callback {callback.__name__!r} registered "
                f"with precompute=False -- skipping precompute analysis; the bridge will "
                f"use the legacy full-chunk scatter path."
            )
        else:
            # Single analysis call for all arrays. No loop overhead.
            # The method takes the full registered_arrays dict, and we pass arrays_metadata here.
            # The bridge distributes branches per array using the array names embedded in each BranchSpec.
            branches = _analyze_callback_for_branches(callback, self.arrays_metadata, precompute=True)
            array_name = array_names[0] if array_names else None
            if array_name and branches:
                self.handshake.set_task_branches(array_name, branches)
            elif array_name:
                logger.debug(
                    f"_register_callback_impl: callback {callback.__name__!r} produced no precomputable branches for "
                    f"array '{array_name}'."
                    f"Without branches, the bridge will fall back to scattering the FULL chunk to the dask workers."
                    f"This is the behavior the precompute optimization is designed to avoid. Set precompute=False and "
                    f"catch the exception if the full-chunk path is acceptable."
                )
                raise NoPrecomputableReductionError(
                    f"Callback {callback.__name__!r} produced no precomputable reductions for array '{array_name}'."
                    f"The precompute path requires at least one chunk-local reduction. To run the callback on the "
                    f"full-chunk scatter path, redesign the callback to use a single dask reduction (sum, mean, var, "
                    f"std, min, max, prod) and avoid expressions whose reduction depends on another reduction's output."
                )

        # create the topic handler and subscribe for EVERY array in a callback (both precompute=True and
        # precompute=False paths. With precompute=False the bridge still needs the topic subscription to receive data
        # and fire the callback on the full-chunk path).
        for array_name in array_names:
            self._callbacks_by_array.setdefault(array_name, set()).add(callback_id)
            if array_name not in self._topic_handlers:
                handler = self._make_topic_handler(array_name)
                self._topic_handlers[array_name] = handler
                logger.debug(f"_register_callback_impl: subscribe_topic() {array_name}")
                self.client.subscribe_topic(array_name, handler)

        return callback_id

    def unregister_callback(self, callback_id: Callback_id) -> None:
        # also accept a decorated callback function, which stores its id in .callback_id
        callback_id = getattr(callback_id, "callback_id", callback_id)
        cb_data = self._callbacks.pop(callback_id, None)
        if cb_data is None:
            return

        for array_name in cb_data["array_names"]:
            s = self._callbacks_by_array.get(array_name)
            if s:
                s.discard(callback_id)
                if not s:
                    del self._callbacks_by_array[array_name]

    def set(self, key: str, value: Any, timestep: int) -> None:
        """
        Sets a value in a queue for a given key, associating it with a specific timestep. This action is
        intended to store feedback or other time-specific data for the provided key.

        - ``:param key:`` The identifier for which the value is to be set.
        - ``:type key:`` str
        - ``:param value:`` The value to be stored, associated with the key and timestep.
        - ``:type value:`` Any
        - ``:param timestep:`` The timestamp that corresponds to when the value is set.
        - ``:type timestep:`` int
        - ``:return:`` None
        """
        logger.debug(f"set() key={key}, value={value}, timestep={timestep}")

        q = Queue(f"{FEEDBACK_QUEUE_PREFIX}{key}", client=self.client, maxsize=self.feedback_queue_size)

        # TODO: check for consistency
        # if q.qsize() > 0:
        #     # check for consistency
        #     t, _ = q.get()
        #     if timestep < t:
        #         raise ValueError(f"timestep {timestep} is smaller than previous timestep {t}")
        #     elif timestep == t:
        #         raise ValueError(f"timestep {timestep} has already been set")

        value = (timestep, value)
        q.put(value)

    def execute_callbacks(self) -> None:
        """
        Executes a series of callbacks and waits for necessary processes to finish.

        This method handles the execution of callbacks related to bridges and their completion while also ensuring the
        orchestration of subsequent tasks. It is responsible for unblocking bridges and waiting for dependencies to
        signal completion.

        - ``:param self:`` The instance of the class invoking this method.
        - ``:return:`` None
        """
        logger.info("execute_callbacks()")

        logger.info("Bridges are ready, unblock bridges")
        Event(WAIT_FOR_EXECUTE_CB_EVENT, client=self.client).set()

        logger.info("execute_callbacks() waiting for bridges")
        self.handshake.wait_for_bridges_to_finish()

        # wait for analysis to be finish
        def _check_deisa_tasks(dask_scheduler):
            """
            Analyzes the tasks on the Dask scheduler to determine the number
            of tasks that are strings that do not start with the deisa task prefix.
            """
            tasks = [
                task
                for task in dask_scheduler.tasks
                if isinstance(task, str)
                # Note: This may be an issue if the Dask scheduler is used by multiple users
                if not task.startswith(KEY_PREFIX)
            ]
            return len(tasks)

        while (nb_running_tasks := self.client.run_on_scheduler(_check_deisa_tasks)) > 0:
            logger.info(f"execute_callbacks() waiting for {nb_running_tasks} tasks to finish")
            time.sleep(1)

        # Wait for every callback to finish.
        # The double sleep(0) recheck flushes any topic messages that are still queued on the client loop
        # into handlers -> tasks before confirming the set is stably empty.
        async def _drain_callback_tasks():
            while True:
                await asyncio.sleep(0)
                if not self._tasks:
                    await asyncio.sleep(0)
                    if not self._tasks:
                        break

        self.client.sync(_drain_callback_tasks)

        logger.info("execute_callbacks() done")

    def _make_topic_handler(self, array_name):
        # use a weak reference to avoid circular references.
        weak_self = weakref.ref(self)

        async def topic_handler(event):
            _weak_self = weak_self()
            if _weak_self is None:
                logger.error(f"topic_handler: weak_self is None, array_name={array_name}")
                raise RuntimeError("weak_self is None")
            try:
                _, payload = event

                logger.debug(f"topic_handler: array_name={array_name}, payload={payload}")

                iteration = payload["iteration"]
                futures = payload["futures"]
                futures = tuple({**d, "future": Future(d["future"], client=_weak_self.client)} for d in futures)

                _weak_self.__update_futures_ownership(futures)

                precomputed = payload.get("precomputed")
                if precomputed:
                    # Precompute path: each ``futures`` entry is one (bridge, reduction) pair with the partial's reduced
                    # shape/dtype. Group by ``output_key`` and dispatch on the partial ``kind``:
                    # - ``"scalar"`` (sum/prod/max/min): stack per-bridge partials along a new axis via ``da.stack``.
                    #   The callback's reduction (e.g. ``arr.sum()``) sums the stacked partials via dask's natural graph
                    # - ``"mean"``: each bridge ships a ``{n, total}`` dict blob (per dask's ``mean_chunk``).
                    #   Build a delayed task that resolves the per-bridge dicts and calls
                    #   ``dask.array.reductions.mean_agg`` over them, arranged in a nested list that matches the MPI
                    #   chunk grid. Return the combined value as a scalar (full reduction) or N-d array (axis reduction)
                    #   dask array.
                    # - ``"moment"``: same pattern, but using ``moment_agg``. For ``std`` (finalize="sqrt"), the
                    #   combined result is sqrt-ed before being returned.
                    by_reduction: Dict[str, List[Any]] = {}
                    for f in futures:
                        by_reduction.setdefault(f["output_key"], []).append(f)
                    darr_chunks: List[Any] = []
                    for output_key, partial_futures in by_reduction.items():
                        kind = partial_futures[0].get("kind", "scalar")
                        finalize = partial_futures[0].get("finalize")
                        partial_shape = partial_futures[0]["shape"]
                        partial_dtype = partial_futures[0]["dtype"]
                        if kind == "scalar":
                            # Sum-able scalar/array partials. Stack along a new axis so the callback's reduction
                            # combines them via dask's natural graph.
                            sorted_partials = sorted(partial_futures, key=lambda p: tuple(p["chunk_position"]))
                            blocks = [
                                da.from_delayed(p["future"], shape=partial_shape, dtype=partial_dtype)
                                for p in sorted_partials
                            ]
                            if len(blocks) == 1:
                                darr_chunks.append(blocks[0])
                            else:
                                darr_chunks.append(da.stack(blocks))
                        elif kind in ("mean", "moment"):
                            # Dict-blob partials. Build a nested list of per-bridge dict futures matching the MPI
                            # chunk grid, then call mean_agg / moment_agg in a single delayed task.
                            # ``chunk_axis`` is shipped in the topic event. It tells us the chunk_func's reduction axes
                            # so the combine's output shape and agg axis are computed correctly.
                            hint_axis = partial_futures[0].get("chunk_axis")
                            array_ndim = len(_weak_self.arrays_metadata[array_name]["global_shape"])
                            darr_chunks.append(
                                _combine_array_from_partials(
                                    partial_futures,
                                    kind=kind,
                                    finalize=finalize,
                                    hint_axis=hint_axis,
                                    array_ndim=array_ndim,
                                )
                            )
                        else:
                            logger.warning(
                                f"topic_handler: unknown precompute kind {kind!r} for {output_key}, skipping"
                            )
                    # For precompute, the dask array *is* the partials. There's no global tiling.
                    # The callback consumes these blocks via the registered callback (the callback gets the stacked
                    # partials as its array). Multiple reductions on the same array produce multiple dask chunks.
                    # We dispatch the first one to the callback. If only one reduction is registered, the callback sees
                    # a single dask array.
                    darr = darr_chunks[0]
                    if len(darr_chunks) > 1:
                        # Multiple reductions: wrap as a tuple of dask arrays and stash on the darr for the callback to
                        # iterate. Most realistic callbacks register exactly one reduction, so this branch is uncommon.
                        darr.extra_precomputed_chunks = darr_chunks[1:]
                    logger.debug(
                        f"topic_handler: precompute path produced {len(darr_chunks)} reduction chunk(s) "
                        f"with shapes {[c.shape for c in darr_chunks]}"
                    )
                else:
                    # Full-chunk path: ``futures`` carries one entry per bridge with the full-chunk shape.
                    # Tile them into a single dask array.
                    parts = sorted(futures, key=lambda p: p["chunk_position"])
                    darr_chunks = [da.from_delayed(p["future"], shape=p["shape"], dtype=p["dtype"]) for p in parts]
                    darr = _weak_self.__tile_dask_blocks(
                        darr_chunks, _weak_self.arrays_metadata[array_name]["global_shape"]
                    )

                # tell the scheduler that gc must *not* collect the futures used by this dask array
                _weak_self.client.persist(darr)

                # dispatch to interested callbacks
                for callback_id in list(_weak_self._callbacks_by_array.get(array_name, [])):
                    cb_data = _weak_self._callbacks.get(callback_id)
                    if cb_data is None:
                        continue

                    if array_name not in cb_data["state"]:
                        continue

                    try:
                        _weak_self._process_callback(callback_id, cb_data, array_name, darr, iteration)
                    except Exception as e:
                        _weak_self._handle_callback_exception(callback_id, cb_data, e)

            except Exception as e:
                logger.error(f"topic_handler: topic handler error array_name={array_name}, e={e}")

        return topic_handler

    def _process_callback(self, callback_id, cb_data, array_name: str, darr: da.Array, iteration: int):
        state = cb_data["state"]

        # Update the sliding window for the modified array.
        entry = state[array_name]
        if entry["last_iteration"] is not None and iteration < entry["last_iteration"]:
            raise ValueError(
                f"callback {callback_id}: array {array_name} received iteration "
                f"{iteration} which is before last seen iteration "
                f"{entry['last_iteration']}. Iterations must be monotonically increasing."
            )
        entry["window"].append(build_deisa_array(darr, iteration))
        entry["changed"] = True
        entry["last_iteration"] = iteration

        ordered_array_names = cb_data["array_names"]

        def _call_callback():
            windows = [list(state[name]["window"]) for name in ordered_array_names]

            # Save the current head of each deque so we know whether it can be
            # safely discarded once the callback completes.
            # This must be done *BEFORE* the callback is run as another callback may modify the window.
            pre_cb_exec_info = [
                (window[0], len(window) == state[name]["window"].maxlen)
                for window, name in zip(windows, ordered_array_names)
            ]

            async def _run():
                try:
                    await asyncio.to_thread(cb_data["callback"], *windows)
                except Exception as ex:
                    self._handle_callback_exception(callback_id, cb_data, ex)

            def _free_windows(_):
                for name, (first_elem, was_full) in zip(ordered_array_names, pre_cb_exec_info):
                    dq = state[name]["window"]
                    if was_full and dq and dq[0] is first_elem:
                        dq.popleft()

            task = asyncio.create_task(_run())
            task.add_done_callback(self._tasks.discard)
            task.add_done_callback(_free_windows)
            self._tasks.add(task)

        if cb_data["when"] == "OR":
            _call_callback()
            entry["changed"] = False

        else:  # AND
            # Verify all arrays arrived at the same iteration before calling
            iterations = {state[name]["last_iteration"] for name in ordered_array_names}
            if (
                all(state[name]["changed"] for name in ordered_array_names)
                and len(iterations) == 1
                and None not in iterations
            ):
                _call_callback()

                for name in ordered_array_names:
                    state[name]["changed"] = False

    def _handle_callback_exception(self, callback_id, cb_data, ex):
        try:
            handler = cb_data["exception_handler"]
            if handler:
                handler(ex)
        except BaseException:
            logger.info(f"Exception in exception handler. Unregistering callback_id={callback_id}")
            self.unregister_callback(callback_id)

    def __update_futures_ownership(self, futures: Collection[Any]):
        keys = [p["future"].key for p in futures]
        # tell scheduler that my client is using these futures
        self.client._send_to_scheduler({"op": "client-desires-keys", "keys": keys, "client": self.client.id})
        # tell scheduler that deisa client no longer needs the futures
        self.client._send_to_scheduler({"op": "client-releases-keys", "keys": keys, "client": CLIENT_KEY})

    def __next_callback_id(self):
        self._callback_seq += 1
        return f"{CALLBACK_PREFIX}{self._callback_seq}"

    @staticmethod
    async def __get_all_chunks(q: Queue, mpi_comm_size: int, timeout=None) -> list[tuple[dict, Future]]:
        """This will return a list of tuples (metadata, data_future) for all chunks in the queue."""
        try:
            res = []
            for _ in range(mpi_comm_size):
                res.append(q.get(timeout=timeout))
            return await asyncio.gather(*res)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Timeout reached while waiting for chunks in queue '{q.name}'.")

    @staticmethod
    def __tile_dask_blocks(blocks: list[da.Array], global_shape: tuple[int, ...]) -> da.Array:
        """
        Given a flat list of N-dimensional Dask arrays, tile them into a single Dask array.
        The tiling layout is inferred from the provided global shape.

        Parameters:
            blocks (list of dask.array): Flat list of Dask arrays. All must have the same shape.
            global_shape (tuple of int): Shape of the full array to reconstruct.

        Returns:
            dask.array.Array: Combined tiled Dask array.
        """
        if not blocks:
            raise ValueError("No blocks provided.")

        block_shape = blocks[0].shape
        ndim = len(block_shape)

        if len(global_shape) != ndim:
            raise ValueError("global_shape must have the same number of dimensions as blocks.")

        # Check that all blocks have the same shape
        for b in blocks:
            if b.shape != block_shape:
                raise ValueError("All blocks must have the same shape.")

        # Compute how many blocks are needed per dimension
        tile_counts = tuple(g // b for g, b in zip(global_shape, block_shape))

        if np.prod(tile_counts) != len(blocks):
            raise ValueError(
                f"Mismatch between number of blocks ({len(blocks)}) "
                f"and expected number from global_shape {global_shape} "
                f"with block shape {block_shape} (expected {np.prod(tile_counts)} blocks)."
            )

        # Reshape the flat list into an N-dimensional grid of blocks
        def nest_blocks(flat_blocks, shape):
            """Nest a flat list of blocks into a nested list matching the grid shape."""
            if len(shape) == 1:
                return flat_blocks
            else:
                size = shape[0]
                stride = int(len(flat_blocks) / size)
                return [nest_blocks(flat_blocks[i * stride : (i + 1) * stride], shape[1:]) for i in range(size)]

        nested = nest_blocks(blocks, tile_counts)

        # Use da.block to combine blocks
        return da.block(nested)

    @staticmethod
    def __in_client_loop(client):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False

        return loop is client.loop.asyncio_loop

    @staticmethod
    def run_task_sync(coro, loop):
        container = {}
        done = threading.Event()

        def callback():
            task = asyncio.create_task(coro)

            async def wrapper():
                try:
                    container["result"] = await task
                finally:
                    done.set()

            asyncio.create_task(wrapper())

        loop.call_soon_threadsafe(callback)
        done.wait()
        return container.get("result")

    @staticmethod
    def make_topic(arrays, when) -> str:
        return f"{when}|" + "|".join(sorted(arrays))
