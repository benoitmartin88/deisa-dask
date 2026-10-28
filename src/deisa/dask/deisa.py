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

"""
The analytics-host side of deisa-dask: callback registration, the per-array topic handlers that deliver data to the
callbacks, and the cycle boundary (``execute_callbacks``).

Two delivery modes exist and each callback's ``precompute`` argument selects its own (decision F3):

- **runtime reduction delivery** (``precompute=True``, the default): the callback receives one
  :class:`~deisa.dask.collection.DeisaCollection` per (registered array, timestep) instead of an array. The collection
  is not a dask collection and carries no data -- it exposes the closed reduction surface and dispatches each
  ``.compute()`` to the bridges, which reduce their OWN chunk. The bridges' eager scalar menu and the on-demand plan
  round both arrive through the topic event and the array transport attached to the delivery.
- **legacy full-chunk scatter** (``precompute=False``): unchanged. The chunk itself reaches the workers, which is the
  remedy every refusal message points at.

Delivery mode is per CALLBACK, not per run: two callbacks on the same array may be registered one way each, so the
topic handler dispatches each callback down its own path within a single event.

Window semantics under runtime delivery: the callback's argument is an ordinary Python list with one entry per array,
so ``window[-1]`` is list indexing and delivers the current timestep's collection and ``window[-2]`` the previous
one. A callback written against the old stub model that indexed the ARGUMENT as if it were the array (``def cb(a):
return a[-1].sum()``) therefore keeps working and now reads ``a[-1]`` as "the latest timestep" -- the runtime reading,
which is what the stub model was approximating. Anyone who needs the old array reading uses ``precompute=False``.

Every refusal raised from a callback's ``.compute()`` fails the RUN (see ``_handle_callback_exception``) instead of
degrading into a logged error: a callback that reads raw data must stop the simulation once, not silently read the
wrong thing on every later timestep.
"""

import asyncio
import collections
import logging
import threading
import time
import weakref
from typing import Any, Callable, Collection, Dict, List, Literal, Mapping, Optional, Set, Tuple

import numpy as np
from deisa.core import CallbackArgs, DeisaArray, Window
from deisa.core.interface import IDeisa
from distributed import Client, Event, Future, Queue

import dask.array as da
from deisa.dask import runtime_errors as runtime_errors
from deisa.dask.collection import DeisaCollection
from deisa.dask.constants import (
    CALLBACK_PREFIX,
    CLIENT_KEY,
    DEFAULT_SLIDING_WINDOW_SIZE,
    FEEDBACK_QUEUE_PREFIX,
    KEY_PREFIX,
    WAIT_FOR_EXECUTE_CB_EVENT,
)
from deisa.dask.handshake import Handshake
from deisa.dask.precompute_errors import (
    MaterializationError,
    PrecomputeRuntimeError,
    RawFieldReadError,
    UnsupportedReductionError,
)
from deisa.dask.runtime_dispatch import (
    EagerMenu,
    PlanDispatcher,
    QueuePlanExchange,
    plan_report_queue_name,
    plan_request_queue_name,
)
from deisa.dask.runtime_errors import RuntimeRefusal
from deisa.dask.utils import build_deisa_array, get_client

logger = logging.getLogger(__name__)

# The refusal classes the precompute path raises mid-run: ``UnsupportedReductionError`` from chunk-local planning,
# ``PrecomputeRuntimeError`` from the runtime transfer/merge steps. They are a separate family from
# :mod:`deisa.dask.runtime_errors` (which subclasses them, never the other way around), so the abort predicate has to
# name them explicitly.
_LEGACY_RUN_ABORTING: Tuple[type, ...] = (
    RawFieldReadError,
    MaterializationError,
    UnsupportedReductionError,
    PrecomputeRuntimeError,
)


def _is_run_aborting(ex: BaseException) -> bool:
    """Whether ``ex`` is a refusal, i.e. a callback failure that must FAIL THE RUN rather than be logged.

    Every refusal means the delivery path cannot serve what the callback asked for (a raw read, a numpy
    materialization, an off-surface or non-associative reduction, a bridge-side invariant violation). Continuing would
    deliver a wrong value on every later timestep, so the run stops once instead.

    BOTH taxonomies are covered because two delivery modes raise them:
    :class:`~deisa.dask.runtime_errors.RuntimeRefusal` is the runtime path's family, and
    :data:`_LEGACY_RUN_ABORTING` are the classes the legacy precompute view raises mid-run. The runtime classes
    subclass their legacy namesakes, so a legacy ``except`` clause catches a new refusal too -- the two families are one
    hierarchy, and this predicate walks it from the top.

    The check lives in this module-level predicate -- and not inline -- so a test can disable it and prove the abort
    test can fail.

    - ``:param ex:`` The exception a callback raised.
    """
    return isinstance(ex, (RuntimeRefusal, *_LEGACY_RUN_ABORTING))


class Deisa(IDeisa):
    Callback_id = str

    def __init__(self, feedback_queue_size: int = 1024, *args, **kwargs) -> None:
        """
        Initializes a class instance, configuring the client and setting up the necessary infrastructure for
        interactions. This includes setting up the necessary feedback queue length, performing handshake operations
        with the client, and initializing various metadata structures.

        - ``:param feedback_queue_size:`` The maximum size of the feedback queue. Defaults to 1024. - ``:type
        feedback_queue_size:`` int - ``:param args:`` Additional positional arguments passed to the initializer. -
        ``:type args:`` tuple - ``:param kwargs:`` Additional keyword arguments passed to the initializer. - ``:type
        kwargs:`` dict
        """
        # dask.config.set({ "distributed.deploy.lost-worker-timeout": 60, "distributed.workers.memory.spill":0.97,
        # "distributed.workers.memory.target":0.95, "distributed.workers.memory.terminate":0.99 })

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
        # True once execute_callbacks() has run; no new callbacks may be registered after the cycle has started.
        self._execute_callbacks_called = False
        # array_name -> exchange callable of the runtime reduction path (one request/reports round trip over the
        # array's sub-comm). Internal, never a user parameter: the delivery attaches the collection's compute boundary
        # from here, and an array without an entry still serves its eager menu and refuses the rest.
        self._runtime_transports: Dict[str, Callable[[Any], Any]] = {}
        # Arrays that at least one ``precompute=True`` callback reads. Declared to the bridges over the handshake
        # (``_flush_delivery_state_to_handshake``), because an eager scalar menu for an array no callback subscribes to
        # is six chunk folds per bridge per timestep of pure critical-path cost.
        self._runtime_arrays: Set[str] = set()
        # Arrays that at least one ``precompute=False`` callback reads, i.e. that still need the LEGACY full-chunk
        # scatter. Filed in the same handshake read as the two facts above: a runtime-delivery array WITHOUT a legacy
        # reader needs no data shipped at all (its menu and its on-demand plans answer every callback), while one WITH
        # a legacy reader must keep the chunk -- that is decision F3's escape hatch.
        self._legacy_arrays: Set[str] = set()
        # array_name -> newest timestep the delivery handed to the callbacks. Read by every live dispatcher, including
        # the ones held by OLDER collections in a callback's window: the bridges retain one chunk per array, so an
        # on-demand reduction of an older window element is refused at compose time instead of aborting the run.
        self._runtime_latest_timestep: Dict[str, int] = {}
        # First runtime refusal a callback raised; ``execute_callbacks`` re-raises it so the run fails instead of
        # logging once per timestep. None while the run is healthy.
        self._run_failure: Optional[BaseException] = None
        # Populate the per-array transports now, once: a delivery may ask for an on-demand reduction as soon as the
        # first event arrives, and attaching is a scheduler round trip per array.
        self._attach_runtime_transports()

    def _attach_runtime_transports(self) -> None:
        """Attach the production transport of every registered array: the array's pair of dask queues.

        This is the writer of ``_runtime_transports``. The queues are named by convention and per array, the same way
        the feedback queues are, so the bridges attach to them without the host sending them anything: the host puts a
        request on ``plan_request_queue_name``, the array's root bridge picks it up and answers on
        ``plan_report_queue_name``, and what comes back is merged exactly as a test-injected exchange's reports are.

        Every array gets one: the first on-demand request can arrive as soon as the first delivered timestep, which is
        before any callback has necessarily run, so there is no later point at which attaching could be justified over
        attaching at ``__init__``. An array whose bridge count the metadata does not describe is skipped: without a
        count the host cannot know when a round is complete.
        """
        for array_name, metadata in self.arrays_metadata.items():
            bridge_count = _grid_size_from_metadata(metadata)
            if not bridge_count:
                continue
            self._runtime_transports[array_name] = QueuePlanExchange(
                Queue(plan_request_queue_name(array_name), client=self.client),
                Queue(plan_report_queue_name(array_name), client=self.client),
                bridge_count,
            )
            logger.debug(f"runtime transport attached for array {array_name!r} over {bridge_count} bridge(s)")

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
            logger.error(f"Could not cleanly close deisa. exception={e}")

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

                This function acts as a decorator that allows you to register a callback with parameters provided
                through ``callback_args``. It also handles exceptions using the ``exception_handler`` and defines the
                execution rules with ``when`` parameter.

                Supports: Default window size is 1. ``@deisa.register("arr1")`` # default window size
                ``@deisa.register("arr1", "arr2")`` # two arrays, default window size
                ``@deisa.register(Window("arr1"))`` # default window size ``@deisa.register(Window("arr1", 2))`` #
                window size 2 ``@deisa.register(Window("arr1", 2), Window("arr2", 5))`` # window size 2 for arr1 and 5
                for arr2 ``@deisa.register(Window("arr1", 2), Window("arr2", 5), "arr3")`` # window size 2 for arr1 and
                5 for arr2, default window size for arr3

                Every callback is served by runtime reduction delivery by default: the bridges reduce their own chunk
                and the callback receives one data-free collection per array and timestep, whose reductions answer
                from an eager scalar menu or an on-demand plan round. A callback that needs the whole field -- a raw
                read, a reduction whose input is a derived expression the closed surface cannot express -- registers
                with ``precompute=False`` and receives the legacy full-chunk scatter instead.

                Delivery state filing contract: registration declares its arrays as runtime-delivery targets to the
                handshake actor immediately (required: a bridge on the wait_for_go=False path fetches the state lazily
                on its first send()). ``unregister_callback`` prunes memory only; the pruned set is re-filed at the
                per-cycle boundary by ``execute_callbacks()`` (which always flushes before unblocking the bridges).

        - ``:param callback_args:`` Variable-length arguments representing callback-specific parameters. - ``:param
        exception_handler:`` Optional exception handler to manage errors during callback execution. Defaults to
        ``__default_exception_handler``. - ``:param when:`` Specifies the conditional logic for triggering the
        callback. Can be 'AND' or 'OR'. Defaults to 'AND'. - ``:param precompute:`` Selects the delivery mode of THIS
        callback: True (default) for runtime reduction delivery, False for the legacy full-chunk scatter. -
        ``:return:`` A callable that wraps the provided callback with the configured parameters and logic. -
        ``:rtype:`` Callable
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
        Registers a callback function with specific arguments, exception handling, and conditional execution
        criteria. This function allows you to register a callback with parameters provided through
        ``callback_args``. It also handles exceptions using the ``exception_handler`` and defines the execution
        rules with ``when`` parameter. Supports: Default window size is 1. ``@deisa.register("arr1")`` #
        default window size ``@deisa.register("arr1", "arr2")`` # two arrays, default window size
        ``@deisa.register(Window("arr1")) `` # default window size ``@deisa.register(Window("arr1", 2))`` #
        window size 2 ``@deisa.register(Window("arr1", 2), Window("arr2", 5))`` # window size 2 for arr1 and 5
        for arr2 ``@deisa.register(Window("arr1", 2), Window("arr2", 5), "arr3")`` # window size 2 for arr1 and
        5 for arr2, default window size for arr3

        - ``:param callback:`` Callback function to register. - ``:param callback_args:`` Variable-length
        arguments representing callback-specific parameters. - ``:param exception_handler:`` Optional exception
        handler to manage errors during callback execution. - ``:param when:`` Specifies the conditional logic
        for triggering the callback. Can be 'AND' or 'OR'. - ``:param precompute:`` Selects the delivery mode of
        this callback: True (default) for runtime reduction delivery, False for the legacy full-chunk scatter. -
        ``:return:`` A callable that wraps the provided callback with the configured parameters and logic.
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

        if self._execute_callbacks_called:
            raise PrecomputeRuntimeError(
                "Cannot register new callbacks after execute_callbacks() has been called. "
                "All callbacks must be registered before the computation cycle starts."
            )

        # per-callback state
        callback_state = {
            arr_name: {"window": collections.deque(maxlen=ws), "changed": False, "last_iteration": None}
            for arr_name, ws in parsed
        }

        # NOTE: ``self._callbacks[callback_id]`` is written at the END of this method, only after the delivery state is
        # filed and the topic subscription succeeds. Writing it earlier left a half-registered entry behind when a
        # later step raised: the id stayed unreachable via ``unregister_callback`` and ``callback.callback_id`` was
        # never set.

        # ``precompute`` selects the DELIVERY MODE of this callback and nothing else (decision F3): True hands the
        # callback one data-free collection per array and timestep, whose reductions the bridges answer chunk-locally,
        # and False keeps the legacy full-chunk scatter that every refusal message names as the remedy.
        #
        # Nothing is analyzed at registration any more (plan P4): the reduction plan is composed from the callback's
        # own expression at its compute boundary, so there is no upfront AST pass to fail. A raw read is therefore
        # refused at the FIRST compute boundary -- which fails the run -- instead of at registration.

        if not precompute:
            logger.warning(
                f"_register_callback_impl: callback {callback.__name__!r} registered "
                f"with precompute=False -- the bridge will use the legacy full-chunk scatter path."
            )
            # A legacy reader means this array's chunk really is read: the bridges must keep scattering it even when a
            # ``precompute=True`` callback on the same array needs no data at all.
            self._legacy_arrays.update(array_names)
        else:
            # These arrays are runtime-delivery targets: the bridges need to know, because an eager scalar menu for an
            # array no callback reads is six chunk folds per bridge per timestep of pure critical-path cost. Filed
            # with the handshake actor here and at the per-cycle boundary, like the rest of the delivery state.
            self._runtime_arrays.update(array_names)

        # Both per-array facts travel in ONE handshake read, so they are filed together whatever the mode.
        self._flush_delivery_state_to_handshake()

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

        # Register the callback payload only after every step that can raise (delivery-state filing, subscription) has
        # succeeded. ``precompute`` is carried in the payload because it is read per EVENT: the topic handler dispatches
        # each callback down its own delivery path, and an unregistration recomputes the array's runtime-delivery
        # membership from the surviving payloads.
        self._callbacks[callback_id] = {
            "callback": callback,
            "when": when,
            "exception_handler": exception_handler,
            "array_names": array_names,
            "precompute": precompute,
            "state": callback_state,
        }

        return callback_id

    def _prune_delivery_flags(self, array_name: str) -> None:
        """Recompute both per-array delivery facts from the surviving callbacks.

        - ``runtime``: at least one callback registered on the array delivers with ``precompute=True``;
        - ``legacy``: at least one callback registered on the array delivers with ``precompute=False``, i.e. reads the
          chunk itself, so the bridges must keep scattering it.

        This only touches memory: the changed sets reach the bridges at the next flush point
        (``execute_callbacks``), never from inside the async topic handler where the unregistration may run.

        - ``:param array_name:`` Registered array the removed callback was registered on.
        """
        surviving = [self._callbacks.get(cid, {}) for cid in self._callbacks_by_array.get(array_name, ())]
        if any(payload.get("precompute", True) for payload in surviving):
            self._runtime_arrays.add(array_name)
        else:
            self._runtime_arrays.discard(array_name)
        if any(not payload.get("precompute", True) for payload in surviving):
            self._legacy_arrays.add(array_name)
        else:
            self._legacy_arrays.discard(array_name)

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
            # Prune unconditionally for every array the callback touched: the dead callback's runtime-delivery claim
            # must leave the set even if other callbacks remain registered.
            self._prune_delivery_flags(array_name)

    def set(self, key: str, value: Any, timestep: int) -> None:
        """
        Sets a value in a queue for a given key, associating it with a specific timestep. This action is intended to
        store feedback or other time-specific data for the provided key.

        - ``:param key:`` The identifier for which the value is to be set. - ``:type key:`` str - ``:param value:`` The
        value to be stored, associated with the key and timestep. - ``:type value:`` Any - ``:param timestep:`` The
        timestamp that corresponds to when the value is set. - ``:type timestep:`` int - ``:return:`` None
        """
        logger.debug(f"set() key={key}, value={value}, timestep={timestep}")

        q = Queue(f"{FEEDBACK_QUEUE_PREFIX}{key}", client=self.client, maxsize=self.feedback_queue_size)

        # TODO: consistency check -- reject an out-of-order or duplicate timestep here once feedback ordering semantics
        # are settled.

        value = (timestep, value)
        q.put(value)

    def _flush_delivery_state_to_handshake(self) -> None:
        """File every array's current delivery state with the handshake actor.

        Two per-array facts travel to the bridges, over the ONE broadcast each of them already enters per array (see
        :meth:`deisa.dask.handshake.Handshake.get_task_state`):

        - whether the array is a runtime-delivery target, i.e. whether the eager scalar menu it computes has any
          reader at all, and
        - whether the array still has a legacy (``precompute=False``) reader, i.e. whether its chunk must be scattered
          even though a runtime callback on it needs no data.

        Called by ``register_callback`` (a bridge on the wait_for_go=False path fetches lazily on first send(),
        possibly before the next cycle) and by ``execute_callbacks`` (before the bridges are unblocked).
        ``unregister_callback`` never files: an actor call inside the async topic handler stalls the handler loop. The
        pruned state is filed at the next flush point.
        """
        self.handshake.set_runtime_arrays(sorted(self._runtime_arrays))
        self.handshake.set_legacy_arrays(sorted(self._legacy_arrays))

    def execute_callbacks(self) -> None:
        """
        Executes a series of callbacks and waits for necessary processes to finish.

                This method handles the execution of callbacks related to bridges and their completion while also
                ensuring the orchestration of subsequent tasks. It is responsible for unblocking bridges and waiting
                for dependencies to signal completion.

                This is also the delivery-state filing boundary for unregistration: the pruned state (pruned in memory
                by ``unregister_callback``) is re-filed to the handshake actor here, once per cycle, before the bridges
                see it. - ``:param self:`` The instance of the class invoking this method. - ``:return:`` None
        """
        logger.info("execute_callbacks()")

        # Snapshot the pruned/merged delivery state into the handshake actor before unblocking the bridges: from their
        # point of view it is static for the whole cycle.
        self._flush_delivery_state_to_handshake()

        # From this point on the bridge contract is sealed: no new callbacks may be registered and no new delivery
        # state will be filed.
        self._execute_callbacks_called = True

        logger.info("Bridges are ready, unblock bridges")
        Event(WAIT_FOR_EXECUTE_CB_EVENT, client=self.client).set()

        logger.info("execute_callbacks() waiting for bridges")
        self.handshake.wait_for_bridges_to_finish()

        # wait for analysis to be finish
        def _check_deisa_tasks(dask_scheduler):
            """
            Analyzes the tasks on the Dask scheduler to determine the number of tasks that are strings that do not
            start with the deisa task prefix.
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

        # Wait for every callback to finish. The double sleep(0) recheck flushes any topic messages that are still
        # queued on the client loop into handlers -> tasks before confirming the set is stably empty.
        async def _drain_callback_tasks():
            while True:
                await asyncio.sleep(0)
                if not self._tasks:
                    await asyncio.sleep(0)
                    if not self._tasks:
                        break

        self.client.sync(_drain_callback_tasks)

        logger.info("execute_callbacks() done")

        # A refused callback failed the run: raise it here, at the cycle boundary the caller already blocks on, instead
        # of letting the simulation continue with a callback that can never deliver a correct value. Cleared first so a
        # caller that catches it can drive a fresh cycle (the offending callback is already unregistered).
        failure = getattr(self, "_run_failure", None)
        if failure is not None:
            self._run_failure = None
            raise failure

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

                # Delivery mode is a property of the CALLBACK, not of the event (decision F3): ``precompute=True``
                # callbacks get one data-free collection per array and timestep, ``precompute=False`` callbacks keep
                # the legacy full-chunk scatter. The event's ``precomputed`` field is no longer read -- with nothing
                # filing branches it is empty by construction.
                runtime_views: Dict[Any, Any] = {}
                if _weak_self._has_runtime_callbacks(array_name):
                    # The bridges' eager menu is the zero-round-trip half of the reduction path, so its failure is the
                    # RUN's failure: recorded once per event, before any collection is built on top of it.
                    _weak_self._record_menu_failure(payload)
                    runtime_views = _weak_self._runtime_views(array_name, iteration, payload)

                full_chunk_darr: List[Any] = []

                def _get_full_chunk_darr():
                    """Tile the full-chunk futures into one dask array on first use, and pin it in the scheduler.

                    Built lazily on purpose: an event whose callbacks are ALL served by runtime delivery must not pay
                    for a dask graph over a chunk nobody reads.

                    An EMPTY ``futures`` is a delivery-state bug, not a data condition: the bridges skip the scatter
                    only for an array they were told has no legacy reader, so a ``precompute=False`` callback reaching
                    this point is being handed an event the host should never have routed to it. Refuse loudly, naming
                    the array and the timestep; the alternative is ``ValueError: No blocks provided`` from
                    ``__tile_dask_blocks``, which names neither.
                    """
                    if not full_chunk_darr:
                        if not futures:
                            raise PrecomputeRuntimeError(
                                f"array {array_name!r} was delivered at timestep {iteration} with no data futures, "
                                f"but a precompute=False callback reads it: the bridges skipped the full-chunk "
                                f"scatter for an array they were told has no legacy reader. The delivery state is "
                                f"inconsistent, not the data."
                            )
                        parts = sorted(futures, key=lambda p: p["chunk_position"])
                        blocks = [da.from_delayed(p["future"], shape=p["shape"], dtype=p["dtype"]) for p in parts]
                        darr = _weak_self.__tile_dask_blocks(
                            blocks, _weak_self.arrays_metadata[array_name]["global_shape"]
                        )
                        # tell the scheduler that gc must *not* collect the futures used by this dask array
                        _weak_self.client.persist(darr)
                        full_chunk_darr.append(darr)
                    return full_chunk_darr[0]

                # dispatch to interested callbacks
                for callback_id in list(_weak_self._callbacks_by_array.get(array_name, [])):
                    cb_data = _weak_self._callbacks.get(callback_id)
                    if cb_data is None:
                        continue

                    if array_name not in cb_data["state"]:
                        continue

                    try:
                        # Resolving the operand is inside the try: a delivery that cannot produce one is a refusal of
                        # THIS callback (run-aborting, named after the array), never a logged handler error.
                        if cb_data.get("precompute", True):
                            cb_darr = runtime_views.get(callback_id)
                            if cb_darr is None:
                                # The callback was unregistered while this event was in flight: nothing to deliver.
                                continue
                        else:
                            cb_darr = _get_full_chunk_darr()

                        _weak_self._process_callback(callback_id, cb_data, array_name, cb_darr, iteration)
                    except Exception as e:
                        _weak_self._handle_callback_exception(callback_id, cb_data, e)

            except Exception as e:
                logger.error(f"topic_handler: topic handler error array_name={array_name}, e={e}")

        return topic_handler

    def _record_menu_failure(self, payload: Mapping[str, Any]) -> None:
        """Record a bridge's eager-menu failure as the RUN failure, so the cycle boundary raises it.

        A menu entry that cannot fold a chunk (an empty chunk is enough: ``np.min`` of it raises) used to escape
        ``Bridge.send()`` as an uncaught error and break the simulation's send loop. The bridge now ships the failure
        with the event and keeps the legacy scatter going, and the run fails HERE -- at the boundary the caller already
        blocks on -- rather than degrading into a silently menu-less delivery that looks healthy for the whole run.

        - ``:param payload:`` The topic event; its ``menu_error`` entry is absent unless a bridge failed to build its
          menu for this timestep.
        """
        detail = payload.get("menu_error")
        if not detail:
            return
        failure = runtime_errors.precompute_runtime_invariant("the bridge eager menu", str(detail))
        if getattr(self, "_run_failure", None) is None:
            self._run_failure = failure
        logger.error(f"the bridge eager menu failed for array {payload.get('array_name')!r}: {detail}")

    def _has_runtime_callbacks(self, array_name: str) -> bool:
        """Whether ANY callback registered on ``array_name`` is served by runtime delivery.

        The topic handler reads this once per event: with no runtime callback there is nobody to hand a collection
        to, so the event carries no menu worth recording and no view worth building.

        - ``:param array_name:`` Registered array the topic event is for.
        """
        return any(
            self._callbacks.get(cid, {}).get("precompute", True) for cid in self._callbacks_by_array.get(array_name, ())
        )

    def _runtime_views(self, array_name: str, iteration: int, payload: Mapping[str, Any]) -> Dict[Any, Any]:
        """Build the runtime delivery view of one array for one timestep: a FRESH collection per callback.

        - ``:param array_name:`` Registered array the topic event is for.
        - ``:param iteration:`` Timestep the event delivered.
        - ``:param payload:`` The topic event, read for the bridges' eager menus and for a menu failure.

        One instance per callback, not one per (array, timestep): a collection is mutable metadata (``name``,
        ``timestep``, ``chunk_shape`` ... are plain attributes), so a callback that writes an attribute -- by accident
        or through a helper that "normalizes" its argument -- would poison every sibling callback on that
        array/timestep AND the retained window history, and every later refusal would name the wrong array/timestep.
        Building one per callback is metadata-only work: no data is copied, and the deliveries of two callbacks are
        independent objects.

        The newest timestep of the array is recorded here, before the views are built, so a dispatcher held by an
        OLDER collection in a callback's window can tell that its timestep is no longer the one the bridges retain.

        ``precompute=False`` callbacks are skipped: they read the full-chunk scatter, so a collection for them would
        be a delivery nobody consumes.
        """
        self._runtime_latest_timestep[array_name] = int(iteration)
        menus = None if payload.get("menu_error") else payload.get("menu")
        return {
            callback_id: self._runtime_collection(array_name, iteration, menus)
            for callback_id in list(self._callbacks_by_array.get(array_name, ()))
            if self._callbacks.get(callback_id, {}).get("precompute", True)
        }

    def _runtime_collection(self, array_name: str, iteration: int, menus: Any = None) -> DeisaCollection:
        """Build the collection a callback receives for ``array_name`` at ``iteration``.

        The compute boundary is attached here and nowhere else. It carries the bridges' eager menu, so a scalar full
        fold costs no round trip, and the array's transport, so anything else can dispatch on demand. Without a
        transport the menu still serves, and a menu miss refuses with the no-transport refusal -- never a gather.

        - ``:param array_name:`` Registered array the collection describes.
        - ``:param iteration:`` Timestep the collection is delivered for.
        - ``:param menus:`` The eager menu entries the bridges shipped with this event (one per bridge), if any.
        """
        metadata = self.arrays_metadata.get(array_name, {})
        dispatcher = PlanDispatcher(
            self._runtime_transports.get(array_name),
            bridge_count=_grid_size_from_metadata(metadata),
            menu=_as_eager_menus(menus),
            latest_timestep=lambda: self._runtime_latest_timestep.get(array_name),
        )
        return DeisaCollection.from_metadata(array_name, iteration, metadata, dispatcher=dispatcher)

    def _process_callback(self, callback_id, cb_data, array_name: str, darr: da.Array | DeisaArray, iteration: int):
        state = cb_data["state"]

        # Update the sliding window for the modified array.
        entry = state[array_name]
        if entry["last_iteration"] is not None and iteration < entry["last_iteration"]:
            raise ValueError(
                f"callback {callback_id}: array {array_name} received iteration "
                f"{iteration} which is before last seen iteration "
                f"{entry['last_iteration']}. Iterations must be monotonically increasing."
            )
        entry["window"].append(
            darr if isinstance(darr, (DeisaArray, DeisaCollection)) else build_deisa_array(darr, iteration)
        )
        entry["changed"] = True
        entry["last_iteration"] = iteration

        ordered_array_names = cb_data["array_names"]

        def _call_callback():
            windows = [list(state[name]["window"]) for name in ordered_array_names]

            # Save the current head of each deque so we know whether it can be safely discarded once the callback
            # completes. This must be done *BEFORE* the callback is run as another callback may modify the window.
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
        """Route a callback failure: a runtime refusal FAILS THE RUN, anything else goes to the user's handler.

        The refusal branch is first and is not negotiable by an exception handler. A refusal means the callback asked
        for something the reduction path cannot serve -- a raw read, a numpy materialization, an off-surface or
        non-associative reduction, a bridge/merge invariant violation -- and the old behaviour (log it and keep the
        callback registered) turned one loud refusal into a wrong value per timestep. So the callback is unregistered
        (never retried) and the failure is recorded for ``execute_callbacks`` to raise.

        - ``:param callback_id:`` The callback that raised.
        - ``:param cb_data:`` Its registration payload, read for the user's exception handler.
        - ``:param ex:`` The exception the callback raised.
        """
        if _is_run_aborting(ex):
            self._abort_run(callback_id, ex)
            return
        try:
            handler = cb_data["exception_handler"]
            if handler:
                handler(ex)
        except BaseException:
            logger.info(f"Exception in exception handler. Unregistering callback_id={callback_id}")
            self.unregister_callback(callback_id)

    def _abort_run(self, callback_id, ex: BaseException) -> None:
        """Record ``ex`` as the run failure and detach the offending callback, so the run ends at the cycle boundary.

        The first refusal wins: a later one is almost always the same callback failing again on a later timestep.

        - ``:param callback_id:`` The callback that was refused.
        - ``:param ex:`` The refusal.
        """
        if getattr(self, "_run_failure", None) is None:
            self._run_failure = ex
        logger.error(f"callback_id={callback_id} was refused, failing the run: {ex}")
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
        Given a flat list of N-dimensional Dask arrays, tile them into a single Dask array. The tiling layout is
        inferred from the provided global shape.

            Parameters: blocks (list of dask.array): Flat list of Dask arrays. All must have the same shape.
            global_shape (tuple of int): Shape of the full array to reconstruct. Returns: dask.array.Array:
            Combined tiled Dask array.
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


def _as_eager_menus(menus: Any) -> List[EagerMenu]:
    """Normalize the topic event's menu entries into :class:`EagerMenu` objects.

    The bridges ship ``EagerMenu`` instances encoded as msgpack-safe mappings (``EagerMenu.as_payload``), because the
    topic event bus is ``Client.log_event``; a transport is free to hand over live objects instead, so both shapes are
    accepted here instead of at every call site.

    - ``:param menus:`` The event's ``menu`` entry: ``None``, or a sequence of ``EagerMenu``/mapping entries.
    """
    normalized: List[EagerMenu] = []
    for entry in menus or ():
        normalized.append(entry if isinstance(entry, EagerMenu) else EagerMenu.from_payload(dict(entry)))
    return normalized


def _grid_extent_from_metadata(metadata: Dict[str, Any]) -> Optional[Tuple[int, ...]]:
    """Per-data-axis chunk-grid extent from the array metadata.

    ``global_shape[i] // chunk_shape[i]`` (at least 1) is the number of MPI chunks along data axis ``i`` -- the harness
    invariant that the MPI cart dims map one-to-one onto the array's data axes. ``None`` when metadata is incomplete.
    """
    g = metadata.get("global_shape")
    c = metadata.get("chunk_shape")
    if not g or not c or len(g) != len(c):
        return None
    if any(ch <= 0 for ch in c):
        return None
    return tuple(max(1, int(gl // ch)) for gl, ch in zip(g, c))


def _grid_size_from_metadata(metadata: Dict[str, Any]) -> Optional[int]:
    """Total number of MPI chunks (bridges) the metadata implies for an array."""
    extent = _grid_extent_from_metadata(metadata)
    if extent is None:
        return None
    return int(np.prod(extent))
