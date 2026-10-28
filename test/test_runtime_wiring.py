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
# * Neither the names of CEA, nor the names of the contributors may be used
#   to endorse or promote products derived from this software without specific
#   prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# =============================================================================
"""
The delivery boundary of the runtime reduction path: topic handler, eager menu, run abort and the PCA hook.

Every test here runs the real pieces without a cluster: the callbacks are handed over by the real
``Deisa._make_topic_handler``, their reductions run through the real ``PlanDispatcher`` over a ``FakeComm`` fleet whose
responders own the actual numpy chunks, and the run boundary is the real ``Deisa.execute_callbacks``. What is faked is
only the transport the analytics host would use to reach a scheduler (topic delivery, the go event, the handshake's
completion wait), which is exactly the part no test in this file is about.

Ground truth is always the analytic value on the whole array -- assembled in the test, never on the delivery path.
"""

import asyncio
import collections
import pickle
import threading
import time

import dask.array as da
import msgpack
import numpy as np
import pytest
from deisa.core.types import DeisaArray
from utils import FakeComm

from deisa.dask import deisa as deisa_module
from deisa.dask import runtime_errors as errors
from deisa.dask.bridge import Bridge
from deisa.dask.collection import DeisaCollection
from deisa.dask.deisa import Deisa
from deisa.dask.handshake import Handshake
from deisa.dask.mergeable_pca import MergeablePCA, PCASummary
from deisa.dask.runtime_compose import PCARequest, compose_request
from deisa.dask.runtime_dispatch import (
    EAGER_MENU_OPS,
    EagerMenu,
    PlanRequest,
    PlanRequestBacklog,
    PlanResponder,
    build_eager_menu,
    dispatch_plan_over_comm,
    serve_plan_requests,
)

ARRAY_NAME = "f"
# ``n_samples > d`` is what keeps the merged PCA summary smaller than the data it summarizes, so the estimator's
# ``_check_summary_smaller_than_input`` guard accepts the fit. (8, 8) with (4, 8) chunks is REFUSED by that guard.
GLOBAL_SHAPE = (8, 4)
CHUNK_SHAPE = (4, 4)
GRID_EXTENT = (2, 1)
METADATA = {ARRAY_NAME: {"global_shape": GLOBAL_SHAPE, "chunk_shape": CHUNK_SHAPE}}


# ---------------------------------------------------------------------------------------------------------------------
# Harness: a two-bridge FakeComm fleet, one per timestep
# ---------------------------------------------------------------------------------------------------------------------
def make_field(seed=0):
    """The whole field of one timestep. Only the test ever holds it."""
    return np.random.default_rng(seed).normal(size=GLOBAL_SHAPE)


def split_chunks(data):
    """Cut ``data`` into the per-bridge chunks the metadata describes (stacked on axis 0)."""
    rows = CHUNK_SHAPE[0]
    return [np.array(data[start : start + rows]) for start in range(0, data.shape[0], rows)]


def chunk_position(rank):
    """MPI coordinates of bridge ``rank``'s chunk in the ``GRID_EXTENT`` grid."""
    return tuple(int(value) for value in np.unravel_index(rank, GRID_EXTENT))


class BridgeFleet:
    """The bridges holding ONE timestep of the array, answering plan requests over one ``FakeComm``."""

    def __init__(self, chunks, timestep=0, array_name=ARRAY_NAME):
        self.chunks = list(chunks)
        self.timestep = int(timestep)
        self.array_name = str(array_name)
        self.state = FakeComm.State(len(self.chunks))
        self.comms = [FakeComm(self.state, rank) for rank in range(len(self.chunks))]
        self.responders = [
            PlanResponder(self.array_name, self.timestep, chunk, rank=rank, chunk_position=chunk_position(rank))
            for rank, chunk in enumerate(self.chunks)
        ]
        self.rounds = []
        self._threads = []

    def start(self):
        """Run the non-root bridges in their own threads, as a bridge's request loop would."""
        for rank in range(1, len(self.chunks)):
            thread = threading.Thread(
                target=serve_plan_requests, args=(self.comms[rank], self.responders[rank]), daemon=True
            )
            thread.start()
            self._threads.append(thread)

    def stop(self):
        """Release the bridge threads with the stop sentinel."""
        self.comms[0].bcast(None, root=0)
        for thread in self._threads:
            thread.join(timeout=10)
        assert not any(thread.is_alive() for thread in self._threads), "a bridge thread did not stop"

    def exchange(self, plan):
        """One request/reports round trip, driven from the array communicator's root."""
        reports = dispatch_plan_over_comm(self.comms[0], plan, self.responders[0])
        self.rounds.append((plan, reports))
        return reports

    def menus(self):
        """The eager menu every bridge of this fleet would ship with its timestep event."""
        collection = DeisaCollection.from_metadata(self.array_name, self.timestep, METADATA[self.array_name])
        return [
            build_eager_menu(collection, chunk, rank=rank, chunk_position=chunk_position(rank))
            for rank, chunk in enumerate(self.chunks)
        ]


class Fleet:
    """Every timestep's fleet, with one exchange routing a request to the fleet holding that timestep."""

    def __init__(self, fields):
        self.fields = dict(fields)
        self.fleets = {
            timestep: BridgeFleet(split_chunks(data), timestep=timestep) for timestep, data in fields.items()
        }

    def start(self):
        for fleet in self.fleets.values():
            fleet.start()

    def stop(self):
        for fleet in self.fleets.values():
            fleet.stop()

    def exchange(self, plan):
        """Route one round trip to the fleet holding the plan's timestep (what the array transport does)."""
        return self.fleets[plan.timestep].exchange(plan)

    def menus(self, timestep):
        return self.fleets[timestep].menus()

    def rounds(self, timestep):
        return self.fleets[timestep].rounds


@pytest.fixture
def fleets():
    world = Fleet({0: make_field(seed=0), 1: make_field(seed=1)})
    world.start()
    try:
        yield world
    finally:
        world.stop()


# ---------------------------------------------------------------------------------------------------------------------
# Harness: the analytics host without a cluster
# ---------------------------------------------------------------------------------------------------------------------
class _LoopShim:
    """The slice of the Tornado loop surface ``distributed.Future`` touches when it is released.

    ``Future.release`` calls ``client.loop.add_callback(client._dec_ref, key)``. The test loop is a plain asyncio loop
    with no ``add_callback``, so every future a delivered event created raised ``AttributeError`` out of ``__del__`` at
    teardown -- an unraisable exception warning in the two tests that now carry a full-chunk payload.
    """

    def __init__(self, loop):
        self._loop = loop

    def add_callback(self, fn, *args, **kwargs):
        """- ``:param fn:`` The callback the client wanted scheduled."""
        fn(*args, **kwargs)

    def run_until_complete(self, awaitable):
        """- ``:param awaitable:`` The coroutine ``Client.sync`` would run."""
        return self._loop.run_until_complete(awaitable)

    def close(self):
        self._loop.close()


class _LoopClient:
    """The host-side client calls the topic handler and ``execute_callbacks`` make, on one persistent loop.

    ``sync`` is distributed's ``Client.sync`` analogue and runs on a loop that outlives the call, which is what lets the
    callback tasks created by a topic event be drained afterwards -- the same loop lifetime a real client has.
    """

    def __init__(self):
        self.loop = _LoopShim(asyncio.new_event_loop())
        self.desired_keys = []
        self.persisted = []
        self.subscribed = []
        self.futures = {}
        self.generation = 0
        self.id = "fake-analytics-host"

    def _inc_ref(self, key):  # noqa: D102
        # ``distributed.Future`` registers its key through the client; this one never tracks state.
        pass

    def _dec_ref(self, key):  # noqa: D102
        # Symmetric with ``_inc_ref``: releasing a future of this client is a no-op.
        pass

    def _send_to_scheduler(self, payload):  # noqa: D102
        self.desired_keys.append(payload)

    def run_on_scheduler(self, fn):  # noqa: D102
        return 0

    def persist(self, value):  # noqa: D102
        self.persisted.append(value)
        return value

    def sync(self, fn):  # noqa: D102
        result = fn()
        if asyncio.iscoroutine(result):
            return self.loop.run_until_complete(result)
        return result

    def subscribe_topic(self, name, handler):  # noqa: D102
        self.subscribed.append((name, handler))

    def close(self):  # noqa: D102
        self.loop.close()


class _FakeHandshake:
    """The handshake surface ``execute_callbacks`` touches."""

    def __init__(self):
        self.runtime_arrays = set()
        self.legacy_arrays = set()
        self.finished = 0

    def set_runtime_arrays(self, array_names):  # noqa: D102
        self.runtime_arrays = set(array_names)

    def set_legacy_arrays(self, array_names):  # noqa: D102
        self.legacy_arrays = set(array_names)

    def wait_for_bridges_to_finish(self):  # noqa: D102
        self.finished += 1


class _FakeEvent:
    """The go event: recording that it was set is the whole contract this boundary has."""

    def __init__(self, name, client=None):  # noqa: D102
        self.name = name
        self.client = client
        self.is_set = False

    def set(self):  # noqa: D102
        self.is_set = True


@pytest.fixture
def client():
    fake = _LoopClient()
    try:
        yield fake
    finally:
        fake.close()


@pytest.fixture
def go_event(monkeypatch):
    """Swap the distributed go event for the recording one, for the tests that drive ``execute_callbacks``."""
    monkeypatch.setattr(deisa_module, "Event", _FakeEvent)
    return _FakeEvent


def make_deisa(fleets, client, monkeypatch):
    """A Deisa stub wired for runtime delivery: registration surfaces stubbed, transports attached."""
    deisa = Deisa.__new__(Deisa)
    deisa.client = client
    deisa.handshake = _FakeHandshake()
    deisa.arrays_metadata = METADATA
    deisa._callbacks = {}
    deisa._callbacks_by_array = {}
    deisa._topic_handlers = {}
    deisa._callback_seq = 0
    deisa._runtime_arrays = set()
    deisa._legacy_arrays = set()
    deisa._tasks = set()
    deisa._execute_callbacks_called = False
    deisa._runtime_transports = {ARRAY_NAME: fleets.exchange}
    deisa._runtime_latest_timestep = {}
    deisa._run_failure = None
    deisa._topic_handlers[ARRAY_NAME] = deisa._make_topic_handler(ARRAY_NAME)
    return deisa


def register(deisa, callback, window_size=2, when="OR", exception_handler=None, precompute=True):
    """Register ``callback`` directly: the precompute analysis is not this card's boundary."""
    callback_id = f"{deisa_module.CALLBACK_PREFIX}{len(deisa._callbacks) + 1}"
    deisa._callbacks[callback_id] = {
        "callback": callback,
        "when": when,
        "exception_handler": exception_handler,
        "array_names": [ARRAY_NAME],
        "precompute": precompute,
        "state": {
            ARRAY_NAME: {"window": collections.deque(maxlen=window_size), "changed": False, "last_iteration": None}
        },
    }
    deisa._callbacks_by_array.setdefault(ARRAY_NAME, set()).add(callback_id)
    if precompute:
        deisa._runtime_arrays.add(ARRAY_NAME)
    else:
        deisa._legacy_arrays.add(ARRAY_NAME)
    return callback_id


async def _drain(tasks, timeout=20.0):
    """Wait until no callback task is left: the analogue of the client draining its topic handlers."""
    deadline = time.monotonic() + timeout
    while tasks and time.monotonic() < deadline:
        await asyncio.sleep(0.005)
    assert not tasks, "a callback task never finished"


def deliver(deisa, timestep, menus, menu_error=None, futures=()):
    """Hand the topic event for ``timestep`` to the real handler, then drain the callback tasks it spawned.

    ``futures`` is the event's data payload: empty for a runtime delivery (the menu is the payload), one full-chunk
    entry per bridge for a legacy scatter.
    """
    payload = {
        "iteration": timestep,
        "futures": list(futures),
        "menu": menus,
    }
    if menu_error is not None:
        payload["menu_error"] = menu_error
    handler = deisa._topic_handlers[ARRAY_NAME]
    deisa.client.sync(lambda: handler((None, payload)))
    deisa.client.sync(lambda: _drain(deisa._tasks))


def full_chunk_futures(shape=CHUNK_SHAPE):
    """The data payload a legacy full-chunk scatter produces: one future per bridge, with that bridge's chunk shape.

    - ``:param shape:`` The per-bridge chunk shape the bridge shipped.
    """
    return [
        {"future": f"full-chunk-{rank}", "shape": shape, "dtype": "float64", "chunk_position": chunk_position(rank)}
        for rank in range(2)
    ]


# ---------------------------------------------------------------------------------------------------------------------
# Delivery: the callback receives collections, and the menu answers the scalar folds
# ---------------------------------------------------------------------------------------------------------------------
def test_scalar_fold_is_served_from_the_eager_menu(fleets, client, monkeypatch):
    """``window[-1].sum().compute()`` is answered by the pre-placed menu: no round trip, correct value."""
    deisa = make_deisa(fleets, client, monkeypatch)
    seen = []

    def callback(window):
        assert isinstance(window[-1], DeisaCollection), "the runtime delivery must hand over collections"
        seen.append(window[-1].sum().compute())

    register(deisa, callback)
    deliver(deisa, 0, fleets.menus(0))

    assert seen == [pytest.approx(fleets.fields[0].sum())]
    assert fleets.rounds(0) == [], "a menu-served scalar fold must not dispatch an on-demand request"


def test_axis_fold_dispatches_on_demand(fleets, client, monkeypatch):
    """A partial fold is not in the menu, so it dispatches one round trip and still returns the right value."""
    deisa = make_deisa(fleets, client, monkeypatch)
    seen = []

    def callback(window):
        seen.append(window[-1].sum(axis=0).compute())

    register(deisa, callback)
    deliver(deisa, 0, fleets.menus(0))

    np.testing.assert_allclose(seen[0], fleets.fields[0].sum(axis=0))
    assert len(fleets.rounds(0)) == 1, "an axis fold must dispatch exactly one on-demand request"


def test_derived_reduction_is_not_answered_by_the_bare_array_menu(fleets, client, monkeypatch):
    """A derived reduction misses the menu: the menu holds the BARE array's partials, so serving it would be wrong.

    The menu is built from the bare collection (``build_eager_menu`` composes the identity program), so its ``sum``
    entry is ``sum(field)``. ``(window[-1] * 2).sum()`` is ``2 * sum(field)``; answering it from the menu silently
    returns the wrong number with no round trip and no exception. Only the on-demand path can serve it.
    """
    deisa = make_deisa(fleets, client, monkeypatch)
    seen = []

    def callback(window):
        derived = (window[-1] * 2).sum().compute()
        bare = window[-1].sum().compute()
        seen.append((derived, bare))

    register(deisa, callback)
    deliver(deisa, 0, fleets.menus(0))

    derived, bare = seen[0]
    truth = fleets.fields[0].sum()
    assert bare == pytest.approx(truth), "the bare fold is still the menu's job"
    assert derived == pytest.approx(2 * truth), "a derived reduction must fold the DERIVED operand"
    assert derived != pytest.approx(bare), "the derived value must not be the bare array's partial"
    assert len(fleets.rounds(0)) == 1, "exactly the derived reduction may dispatch; the bare fold stays in the menu"


def test_menu_value_of_another_timestep_cannot_answer(fleets, client, monkeypatch):
    """A menu computed for timestep N never answers a timestep N+1 request: the dispatcher keys the menu by timestep."""
    deisa = make_deisa(fleets, client, monkeypatch)
    seen = []

    def callback(window):
        seen.append(window[-1].sum().compute())

    register(deisa, callback)
    # The event announces timestep 1 but carries timestep 0's menus. The menu lookup must miss and dispatch.
    deliver(deisa, 1, fleets.menus(0))

    assert seen == [pytest.approx(fleets.fields[1].sum())]
    assert len(fleets.rounds(1)) == 1, "a stale menu value must not be consumed"


def test_window_is_a_list_of_collections(fleets, client, monkeypatch):
    """``window[-1]`` is the latest timestep's collection and ``window[-2]`` the previous one."""
    deisa = make_deisa(fleets, client, monkeypatch)
    seen = []

    def callback(window):
        if len(window) < 2:
            return
        seen.append((window[-1].timestep, window[-2].timestep, window[-1].sum().compute()))

    register(deisa, callback, window_size=2)
    deliver(deisa, 0, fleets.menus(0))
    deliver(deisa, 1, fleets.menus(1))

    assert seen == [(1, 0, pytest.approx(fleets.fields[1].sum()))]


def test_each_callback_gets_its_own_collection(fleets, client, monkeypatch):
    """Isolation: a collection one callback received is not the object a sibling callback receives.

    A collection is mutable metadata, so a callback that writes an attribute -- by accident, or through a helper that
    "normalizes" its argument -- must not poison the other callbacks of the same array and timestep.
    """
    deisa = make_deisa(fleets, client, monkeypatch)
    seen = {}

    def first(window):
        window[-1].name = "mutated-by-the-first-callback"
        seen["first"] = window[-1]

    def second(window):
        seen["second"] = window[-1]

    register(deisa, first)
    register(deisa, second)
    deliver(deisa, 0, fleets.menus(0))

    assert seen["first"] is not seen["second"], "each callback must get its own collection object"
    assert seen["second"].name == ARRAY_NAME, "a sibling's write must not reach the other callback's view"


def test_menu_failure_payload_aborts_the_run(fleets, client, monkeypatch):
    """A bridge that could not build its menu ships the cause; the delivery must record it as the run failure."""
    deisa = make_deisa(fleets, client, monkeypatch)
    calls = []

    def callback(window):
        calls.append(window[-1])

    register(deisa, callback)
    deliver(deisa, 0, None, menu_error="array 'f' at timestep 0, bridge 1: ValueError('empty chunk')")

    assert calls, "the delivery still reaches the callback: fail-loud, not fail-the-send"
    assert isinstance(deisa._run_failure, errors.PrecomputeRuntimeError), (
        f"the menu failure must fail the run at the cycle boundary: {deisa._run_failure!r}"
    )
    assert "empty chunk" in str(deisa._run_failure), "the run failure must carry the bridge's cause"


# ---------------------------------------------------------------------------------------------------------------------
# Run abort: a refusal fails the run instead of degrading into the default handler
# ---------------------------------------------------------------------------------------------------------------------
def test_raw_read_refusal_aborts_the_run(fleets, client, monkeypatch, go_event):
    """A raw read from a callback fails the run, at the cycle boundary the caller already blocks on."""
    deisa = make_deisa(fleets, client, monkeypatch)
    failures = []

    def callback(window):
        try:
            window[-1].compute()
        except BaseException as ex:  # recorded so the test can prove the refusal it caused
            failures.append(ex)
            raise

    callback_id = register(deisa, callback)
    deliver(deisa, 0, fleets.menus(0))

    assert isinstance(failures[0], errors.RawFieldReadError)
    assert isinstance(deisa._run_failure, errors.RawFieldReadError), "the refusal must be recorded as the run failure"
    assert deisa._callbacks == {}, "a refused callback must not stay registered for the next cycle"
    assert callback_id not in deisa._callbacks_by_array.get(ARRAY_NAME, set())

    with pytest.raises(errors.RawFieldReadError):
        deisa.execute_callbacks()
    assert deisa._run_failure is None, "the failure is consumed once it has failed the run"


def test_materialization_refusal_aborts_the_run(fleets, client, monkeypatch, go_event):
    """``np.asarray(collection)`` is a refusal too, and it fails the run the same way."""
    deisa = make_deisa(fleets, client, monkeypatch)

    def callback(window):
        np.asarray(window[-1])

    register(deisa, callback)
    deliver(deisa, 0, fleets.menus(0))

    assert isinstance(deisa._run_failure, errors.MaterializationError)
    with pytest.raises(errors.MaterializationError):
        deisa.execute_callbacks()


def test_mixed_timestep_expression_aborts_the_run(fleets, client, monkeypatch, go_event):
    """``window[-1] - window[-2]`` is off-surface, and its refusal fails the run instead of being logged."""
    deisa = make_deisa(fleets, client, monkeypatch)

    def callback(window):
        if len(window) < 2:
            return
        window[-1] - window[-2]

    register(deisa, callback, window_size=2)
    deliver(deisa, 0, fleets.menus(0))
    deliver(deisa, 1, fleets.menus(1))

    assert isinstance(deisa._run_failure, errors.UnsupportedReductionError)
    with pytest.raises(errors.UnsupportedReductionError):
        deisa.execute_callbacks()


def test_precompute_taxonomy_refusal_aborts_the_run(fleets, client, monkeypatch, go_event):
    """A refusal from the PRECOMPUTE taxonomy aborts too, not just the runtime one.

    The runtime classes subclass the precompute namesakes (``runtime_errors``), so a caller that catches
    ``PrecomputeError`` still catches a runtime refusal -- one hierarchy, walked from the top by ``_is_run_aborting``.
    This test raises the BASE class from a callback, which is what the legacy precompute view used to do, and proves
    the precompute half of the predicate is covered on its own rather than only through a subclass.
    """
    deisa = make_deisa(fleets, client, monkeypatch)

    def callback(window):
        raise deisa_module.UnsupportedReductionError("the precompute view refused this reduction")

    register(deisa, callback)
    deliver(deisa, 0, fleets.menus(0))

    assert isinstance(deisa._run_failure, deisa_module.UnsupportedReductionError)
    assert not isinstance(deisa._run_failure, deisa_module.RuntimeRefusal), "the PRECOMPUTE taxonomy, not the runtime"
    assert deisa._callbacks == {}, "a refused callback must not stay registered for the next cycle"

    with pytest.raises(deisa_module.UnsupportedReductionError):
        deisa.execute_callbacks()


def test_mutation_probe_without_the_refusal_branch_logs_and_continues(fleets, client, monkeypatch, go_event):
    """The abort test can fail: with the refusal branch disabled the default handler logs and the run continues.

    This is the mutation probe for the run-abort contract. ``_is_run_aborting`` is the single seam that decides
    whether a callback failure is a run abort, so disabling it reproduces the pre-fix behaviour exactly, and the
    assertions below are the negation of the ones in ``test_raw_read_refusal_aborts_the_run``.
    """
    deisa = make_deisa(fleets, client, monkeypatch)
    monkeypatch.setattr(deisa_module, "_is_run_aborting", lambda ex: False)

    def callback(window):
        window[-1].compute()

    callback_id = register(deisa, callback)
    deliver(deisa, 0, fleets.menus(0))

    assert deisa._run_failure is None, "without the refusal branch nothing records a run failure"
    assert callback_id in deisa._callbacks, "the default handler leaves the callback registered"
    assert deisa.execute_callbacks() is None, "the default handler must not fail the run"


# ---------------------------------------------------------------------------------------------------------------------
# PCA: the estimator hook dispatches summary branches and merges them at the root
# ---------------------------------------------------------------------------------------------------------------------
def subspace_distance(components, reference):
    """Sign-invariant distance between two component subspaces (never a raw error: the signs are arbitrary)."""
    return float(1.0 - np.linalg.svd(components @ reference.T, compute_uv=False).min())


def reference_components(data, n_components):
    """The top ``n_components`` right singular vectors of the centered whole field."""
    centered = data - data.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[:n_components]


def test_pca_fit_dispatches_a_summary_branch(fleets, client, monkeypatch):
    """``MergeablePCA.fit(window[-1])`` dispatches a PCA branch and matches the single-machine subspace."""
    deisa = make_deisa(fleets, client, monkeypatch)
    fitted = []

    def callback(window):
        fitted.append(MergeablePCA(n_components=2).fit(window[-1]))

    register(deisa, callback)
    deliver(deisa, 0, fleets.menus(0))

    estimator = fitted[0]
    reference = reference_components(fleets.fields[0], n_components=2)
    assert subspace_distance(estimator.components_, reference) < 1e-8
    assert estimator.n_components_ == 2
    assert fleets.rounds(0), "the PCA hook must dispatch through the bridge transport"


def test_pca_branch_ships_summaries_not_chunks(fleets, client, monkeypatch):
    """The PCA wire contract: a PCA request, module-level partials, summary-shaped partials smaller than the data."""
    deisa = make_deisa(fleets, client, monkeypatch)

    def callback(window):
        MergeablePCA(n_components=2).fit(window[-1])

    register(deisa, callback)
    deliver(deisa, 0, fleets.menus(0))

    data_bytes = fleets.fields[0].nbytes
    (plan, reports), *_ = fleets.rounds(0)
    assert isinstance(plan.request, PCARequest)
    assert plan.request.kind == "pca"
    # The branch callable crosses a process boundary, so it must survive a pickle round trip.
    revived = pickle.loads(pickle.dumps(plan.request))
    assert isinstance(revived.branch_func(fleets.fleets[0].chunks[0]), PCASummary)
    for report in reports:
        assert isinstance(report.partial, PCASummary), "a bridge may only report a mergeable summary"
        assert report.partial.components.nbytes + report.partial.mean.nbytes < data_bytes


def test_pca_fit_without_a_transport_refuses(client):
    """A collection without a compute boundary refuses the fit instead of gathering the field."""
    collection = DeisaCollection(ARRAY_NAME, 0, GLOBAL_SHAPE, chunk_shape=CHUNK_SHAPE)
    with pytest.raises(errors.PrecomputeRuntimeError) as refusal:
        MergeablePCA(n_components=2).fit(collection)
    assert "precompute=False" in str(refusal.value)


# ---------------------------------------------------------------------------------------------------------------------
# The legacy paths are untouched
# ---------------------------------------------------------------------------------------------------------------------
def test_precompute_false_still_receives_the_full_chunk(fleets, client, monkeypatch):
    """A ``precompute=False`` callback still gets the whole tiled array, not a collection.

    Delivery mode is the CALLBACK's, so this one registers ``precompute=False`` and the sibling tests' default
    ``precompute=True`` callbacks are unaffected: one event may serve both paths.
    """
    deisa = make_deisa(fleets, client, monkeypatch)
    received = []
    monkeypatch.setattr(
        Deisa,
        "_Deisa__tile_dask_blocks",
        lambda self, blocks, global_shape: da.zeros(global_shape, chunks=CHUNK_SHAPE),
    )

    def callback(window):
        received.append(window[-1])

    register(deisa, callback, precompute=False)
    assert deisa._runtime_arrays == set(), "a full-chunk callback declares no runtime-delivery array"
    deliver(deisa, 0, fleets.menus(0), futures=full_chunk_futures())

    assert len(received) == 1
    assert isinstance(received[0], DeisaArray)
    assert not isinstance(received[0], DeisaCollection)
    assert received[0].shape == GLOBAL_SHAPE


def test_one_event_serves_both_delivery_modes(fleets, client, monkeypatch):
    """Two callbacks on ONE array, one per delivery mode: each gets its own payload from the same event (F3)."""
    deisa = make_deisa(fleets, client, monkeypatch)
    collections_seen = []
    arrays_seen = []
    monkeypatch.setattr(
        Deisa,
        "_Deisa__tile_dask_blocks",
        lambda self, blocks, global_shape: da.zeros(global_shape, chunks=CHUNK_SHAPE),
    )

    def runtime_callback(window):
        collections_seen.append(window[-1])

    def full_chunk_callback(window):
        arrays_seen.append(window[-1])

    register(deisa, runtime_callback)
    register(deisa, full_chunk_callback, precompute=False)
    deliver(deisa, 0, fleets.menus(0), futures=full_chunk_futures())

    assert len(collections_seen) == 1 and isinstance(collections_seen[0], DeisaCollection)
    assert len(arrays_seen) == 1 and isinstance(arrays_seen[0], DeisaArray)
    assert not isinstance(arrays_seen[0], DeisaCollection)


# ---------------------------------------------------------------------------------------------------------------------
# K7F-01: what the bridge ships depends on BOTH per-array facts -- a runtime target ships its menu, and it ships the
# chunk only while something on it reads the chunk. The two facts travel in one handshake read (the file's docstring
# explains why a second collective is forbidden).
# ---------------------------------------------------------------------------------------------------------------------
def test_mixed_array_still_gets_the_full_chunk(fleets, client, monkeypatch):
    """One array with a runtime callback AND a legacy one keeps the full-chunk scatter (decision F3's escape hatch).

    This is the case that forces the second handshake fact. The runtime callback alone makes the array a data-free
    delivery -- the bridge would ship the menu and nothing else -- but the ``precompute=False`` callback on the SAME
    array reads the chunk, so skipping the scatter would hand it an event with ``futures: []``. Host, handshake actor
    and bridge are driven end to end here (no cluster): the host files both facts, the actor returns them in ONE dict,
    and the bridge's send must still scatter.
    """
    deisa = make_deisa(fleets, client, monkeypatch)
    register(deisa, lambda window: None)
    register(deisa, lambda window: None, precompute=False)
    deisa._flush_delivery_state_to_handshake()

    assert deisa.handshake.runtime_arrays == {ARRAY_NAME}
    assert deisa.handshake.legacy_arrays == {ARRAY_NAME}, "the legacy reader must be filed next to the runtime flag"

    # The actor's read: the shape the bridge consumes, with both facts in one dict.
    actor = Handshake.HandshakeActor.__new__(Handshake.HandshakeActor)
    actor.runtime_arrays = deisa.handshake.runtime_arrays
    actor.legacy_arrays = deisa.handshake.legacy_arrays
    state = actor.get_task_state(ARRAY_NAME)
    assert state == {"runtime": True, "legacy": True}, state

    bridge, wire_client = _make_send_bridge()
    # Start from an empty cache, like a bridge taking its state from the handshake read.
    bridge._runtime_delivery_arrays = set()
    bridge._legacy_delivery_arrays = set()
    bridge._cache_task_state(ARRAY_NAME, state)
    assert bridge._needs_chunk_delivery(ARRAY_NAME), "a legacy reader on the array keeps the chunk delivery"

    scattered = []

    def recording_scatter(data, **kwargs):
        """- ``:param data:`` The chunk the bridge decided to ship."""
        scattered.append(data.shape)
        return {
            "future": "key-0",
            "who_has": {"worker-0": ["key-0"]},
            "nbytes": {"key-0": int(np.prod(CHUNK_SHAPE))},
        }

    bridge._scatter_full = recording_scatter
    bridge.send(ARRAY_NAME, np.zeros(CHUNK_SHAPE, dtype=np.float32), 0)
    bridge._has_close_been_called = True

    assert scattered == [CHUNK_SHAPE], "the legacy reader's chunk must still cross the boundary"
    assert len(wire_client.events[0][1]["futures"]) == 1, "the event must still name the full chunk for that reader"


def test_runtime_only_array_ships_its_menu_and_no_data(monkeypatch):
    """Runtime target with NO legacy reader: no scatter, ``futures: []`` -- the menu is the payload.

    This is the regression the memory suite measures (``test_runtime_worker_only_sees_partials``): at HEAD every
    ``precompute=True`` callback cost a 32 MB scatter per bridge per timestep for data nothing read. Three things must
    hold here -- nothing is scattered, the menu is still folded (it is the ENTIRE payload), and no key reaches the
    scheduler (an empty ``update_data`` + an empty desire list would be pure round trips).
    """
    bridge, wire_client = _make_send_bridge()
    bridge._legacy_delivery_arrays = set()
    scattered = []

    def recording_scatter(*args, **kwargs):
        """- ``:param args:`` The chunk the bridge would ship.

        Returns a plausible scatter result so a mutation that keeps the scatter on fails on the assertion below (with
        the reason in it) instead of on an ``IndexError`` from a ``None`` payload.
        """
        scattered.append(args)
        return {"future": "key-0", "who_has": {"worker-0": ["key-0"]}, "nbytes": {"key-0": int(np.prod(CHUNK_SHAPE))}}

    bridge._scatter_full = recording_scatter

    bridge.send(ARRAY_NAME, np.zeros(CHUNK_SHAPE, dtype=np.float32), 0)
    bridge._has_close_been_called = True

    assert scattered == [], "no scatter may happen for an array no callback reads"
    _, payload = wire_client.events[0]
    assert payload["futures"] == [], "the event must not name data that was never shipped"
    assert len(payload["menu"]) == 1, "the menu IS the payload on the data-free path"
    assert payload["menu"][0]["values"]["sum"]["data"], "the menu must still be folded, not dropped"
    assert wire_client.updates == [], "an empty key set needs no update_data"
    assert wire_client.desired_keys == [], "an empty key set needs no client-desires-keys"


class _WireHandshake:
    """The one handshake surface ``_ensure_task_state`` reads on sub-comm rank 0: an array's delivery state."""

    def __init__(self, state):
        self.state = state
        self.reads = []

    def get_task_state(self, array_name):
        """- ``:param array_name:`` Registered array the bridge asks its delivery state for."""
        self.reads.append(array_name)
        return dict(self.state)


def test_first_send_on_lazy_path_ships_no_data():
    """K7F-06: the FIRST send on the lazy path (``wait_for_go=False``) must already be data-free.

    ``test_runtime_only_array_ships_its_menu_and_no_data`` starts from a WARM cache -- the state the ``wait_for_go``
    init-time prefetch leaves behind. A deployment that never prefetches is the memory harness's shape
    (``test_precompute_memory.py``, which asserts after exactly ONE send), and there the caches are still empty when
    ``send`` runs. Computing ``data_free_delivery`` before ``_ensure_task_state`` filled them made that first send take
    the full-chunk scatter while still building the menu from the state it had just fetched -- exactly the acceptance
    log's signature (full-chunk scatter and a built ``EagerMenu`` in the SAME event). The fetch has to come first, so
    the decision is taken on the state the send itself read.

    Witness from K7F-06/27f3a68, adapted to the post-K8 surface (no branch machinery left to stub).
    """
    bridge, wire_client = _make_send_bridge()
    # The lazy-path state at the first send: nothing prefetched, so every cache is empty and the handshake is the only
    # source. Runtime target with no legacy reader -- the data-free shape.
    bridge._task_state_fetched = set()
    bridge._runtime_delivery_arrays = set()
    bridge._legacy_delivery_arrays = set()
    handshake = _WireHandshake({"runtime": True, "legacy": False})
    bridge.handshake = handshake

    scattered = []

    def recording_scatter(*args, **kwargs):
        """- ``:param args:`` The chunk the bridge would ship.

        Returns a plausible scatter result so a mutation that keeps the scatter fails on the assertion below (with the
        reason in it) instead of on an ``IndexError`` from a ``None`` payload.
        """
        scattered.append(args)
        return {"future": "key-0", "who_has": {"worker-0": ["key-0"]}, "nbytes": {"key-0": int(np.prod(CHUNK_SHAPE))}}

    bridge._scatter_full = recording_scatter

    bridge.send(ARRAY_NAME, np.zeros(CHUNK_SHAPE, dtype=np.float32), 0)
    bridge._has_close_been_called = True

    assert handshake.reads == [ARRAY_NAME], "the first send must read the array's delivery state"
    assert scattered == [], "the FIRST send of a runtime-only array must not scatter the chunk"
    assert bridge._runtime_delivery_arrays == {ARRAY_NAME}, "the fetch must fill the cache the decision needs"
    _, payload = wire_client.events[0]
    assert payload["futures"] == [], "the event must not name data that was never shipped"
    assert len(payload["menu"]) == 1, "the menu IS the payload on the data-free path"
    assert payload["menu"][0]["values"]["sum"]["data"], "the menu must still be folded, not dropped"
    assert wire_client.updates == [], "an empty key set needs no update_data"
    assert wire_client.desired_keys == [], "an empty key set needs no client-desires-keys"


def test_no_data_event_handed_to_a_legacy_reader_fails_the_run(fleets, client, monkeypatch, go_event):
    """A ``precompute=False`` callback on an event with no futures FAILS THE RUN, naming the array (K7F-05).

    Reaching the full-chunk path with ``futures: []`` means the host and the bridges disagree about the array's
    delivery state. Without the guard the failure surfaces as ``ValueError: No blocks provided`` (or an IndexError)
    from dask internals, with neither the array nor the timestep in it; the guard names both and aborts the run at the
    cycle boundary instead of logging a handler error per timestep.
    """
    from deisa.dask.precompute_errors import PrecomputeRuntimeError

    deisa = make_deisa(fleets, client, monkeypatch)
    received = []
    register(deisa, lambda window: received.append(window[-1]), precompute=False)

    # ``deliver`` publishes an event with ``futures: []``: the shape the data-free path produces.
    deliver(deisa, 0, None)

    assert received == [], "a callback must not run on data it did not receive"
    assert isinstance(deisa._run_failure, PrecomputeRuntimeError), deisa._run_failure
    detail = str(deisa._run_failure)
    assert repr(ARRAY_NAME) in detail and "timestep 0" in detail, f"the refusal must name the delivery: {detail}"
    with pytest.raises(PrecomputeRuntimeError):
        deisa.execute_callbacks()


def test_departed_runtime_callback_flips_the_array_off(fleets, client, monkeypatch, go_event):
    """Unregistering the last ``precompute=True`` callback drops the array from the bridged runtime set."""
    deisa = make_deisa(fleets, client, monkeypatch)

    runtime_id = register(deisa, lambda window: None)
    assert deisa._runtime_arrays == {ARRAY_NAME}
    deisa._flush_delivery_state_to_handshake()
    assert deisa.handshake.runtime_arrays == {ARRAY_NAME}

    register(deisa, lambda window: None, precompute=False)
    deisa.unregister_callback(runtime_id)

    assert deisa._runtime_arrays == set(), "no runtime callback is left on the array"
    deisa.execute_callbacks()
    assert deisa.handshake.runtime_arrays == set(), "the pruned set is re-filed at the cycle boundary"


# ---------------------------------------------------------------------------------------------------------------------
# The harness itself: the menus it ships are the delivered shape
# ---------------------------------------------------------------------------------------------------------------------
def test_harness_ships_one_eager_menu_per_bridge(fleets):
    """The menus the harness hands the handler are real ``EagerMenu`` entries, one per bridge, for that timestep."""
    menus = fleets.menus(0)
    assert len(menus) == 2, "the (8, 4) field in (4, 4) chunks is a two-bridge grid"
    for menu in menus:
        assert isinstance(menu, EagerMenu)
        assert menu.timestep == 0
        assert menu.array_name == ARRAY_NAME
        assert "sum" in menu.values


# ---------------------------------------------------------------------------------------------------------------------
# The wire: the menu the bridge publishes on the topic event (the ``Client.log_event`` msgpack hop)
# ---------------------------------------------------------------------------------------------------------------------
# ``Bridge.send`` ships the eager menu encoded with ``EagerMenu.as_payload`` because the event bus carries msgpack
# primitives only. Before ``1fbb650`` the live objects went on the wire and the FIRST ``send`` of a runtime-mode run
# raised ``TypeError: Message must be msgpack serializable``: the delivery mode was unusable in production. Every test
# above hands a live ``EagerMenu`` straight to the topic handler, so none of them crosses this hop. These two do.
WIRE_METADATA = {
    ARRAY_NAME: {
        "global_shape": GLOBAL_SHAPE,
        "chunk_shape": CHUNK_SHAPE,
        "chunk_position": (0, 0),
        # float32 on purpose: a menu value's dtype is part of the contract (the merge reports it), so a payload that
        # widened it to float64 would be invisible to every value assertion.
        "dtype": np.float32,
    }
}


class _WireClient:
    """The bridge-side client surface ``Bridge.send`` touches, recording the topic event instead of publishing it."""

    def __init__(self):
        self.scheduler = self
        self.updates = []
        self.desired_keys = []
        self.events = []

    def update_data(self, who_has=None, nbytes=None):
        """- ``:param who_has:`` Worker -> keys map the bridge publishes.
        - ``:param nbytes:`` Key -> size map the bridge publishes.
        """
        self.updates.append((who_has, nbytes))

    def sync(self, fn, *args, **kwargs):
        """- ``:param fn:`` The callable the bridge would hand to ``Client.sync``."""
        return fn(*args, **kwargs)

    def _send_to_scheduler(self, payload):
        """- ``:param payload:`` The ``client-desires-keys`` message."""
        self.desired_keys.append(payload)

    def log_event(self, name, payload):
        """- ``:param name:`` Topic of the event, i.e. the registered array name.
        - ``:param payload:`` The topic payload; this is the msgpack-hop boundary the tests below are about.
        """
        self.events.append((name, payload))


def _make_send_bridge():
    """Build the ``Bridge.__new__`` stub the tests below drive: one bridge, one array, no scheduler.

    Only the scatter is stubbed -- it is ``test_bridge.py``'s subject and needs a scheduler. Everything else (``send``'s
    runtime-delivery boundary, the eager menu, the plan backlog seam) is the real code.
    """
    bridge = Bridge.__new__(Bridge)
    wire_client = _WireClient()
    bridge.id = 0
    bridge.comm = FakeComm(FakeComm.State(1), 0)
    bridge.client = wire_client
    bridge.workers = {"worker-0": {}}
    bridge.arrays_metadata = WIRE_METADATA
    bridge._array_comms = {ARRAY_NAME: FakeComm(FakeComm.State(1), 0)}
    # The menu is gated on the array being a runtime-delivery target, so declare it; its chunk stage is not this
    # test's subject.
    bridge._runtime_delivery_arrays = {ARRAY_NAME}
    # ... and declare it a legacy (chunk-reading) array too, so these tests keep measuring the full-chunk payload.
    # The data-free path (runtime target and no legacy reader) has its own test below.
    bridge._legacy_delivery_arrays = {ARRAY_NAME}
    # No handshake: the two facts above are already the cached state, so ``send`` must find the array fetched and
    # never enter the per-array broadcast (``_task_state_fetched`` membership is the fetch marker).
    bridge.handshake = None
    bridge._task_state_fetched = {ARRAY_NAME}
    bridge._runtime_plan_backlog = {}
    bridge._runtime_plan_report_queues = {}
    bridge._runtime_last_chunk = {}
    bridge._scatter_full = lambda *args, **kwargs: {
        "future": "key-0",
        "who_has": {"worker-0": ["key-0"]},
        "nbytes": {"key-0": int(np.prod(CHUNK_SHAPE))},
    }
    return bridge, wire_client


def send_one_chunk(monkeypatch, chunk):
    """Run the REAL ``Bridge.send`` for a single-bridge array and return the topic payload it published.

    The subject is the encode step (``to_send["menu"] = [menu.as_payload() ...]``) and the eager menu that feeds it, so
    both run for real.
    """
    bridge, wire_client = _make_send_bridge()
    bridge.send(ARRAY_NAME, chunk, 0)
    # The stub has no peers and no client to close: mark it closed so its finalizer stays quiet.
    bridge._has_close_been_called = True

    assert wire_client.events, "the bridge must publish one topic event per send"
    name, payload = wire_client.events[0]
    assert name == ARRAY_NAME
    return payload


def test_topic_event_from_the_bridge_is_msgpack_serializable(monkeypatch):
    """Every entry the bridge puts on the wire must survive ``msgpack.packb``; a live ``EagerMenu`` does not.

    Fail-before (the ``1fbb650`` bug, reproduced in ``evidence/probe_msgpack_roundtrip.log`` line 2): with the menu
    shipped as ``EagerMenu`` objects, ``msgpack.packb`` raises ``TypeError: can not serialize 'EagerMenu' object`` --
    which is exactly what ``Client.log_event`` does to the first ``send`` of a runtime-mode run. The line that fails is
    the ``msgpack.packb(entry)`` inside the loop below.
    """
    chunk = np.random.default_rng(0).normal(size=CHUNK_SHAPE).astype(np.float32)
    payload = send_one_chunk(monkeypatch, chunk)

    # The whole event travels, not just its ``menu`` entry: ``Client.log_event`` packs the payload.
    entries = payload["menu"]
    assert len(entries) == 1, "a single-bridge array publishes its one menu"
    for entry in entries:
        msgpack.packb(entry)
        assert set(entry["values"]) == set(EAGER_MENU_OPS)
        assert entry["values"]["sum"]["dtype"] == "float32", "the wire must carry the fold's dtype, not widen it"
        assert entry["array_name"] == ARRAY_NAME and entry["timestep"] == 0
    msgpack.packb(payload)


def _assert_same_menu_value(original, rebuilt):
    """Assert one menu value survived the msgpack round trip: same structure, dtype, shape and numbers.

    - ``:param original:`` The value the bridge computed (a numpy scalar/array, or a mapping of them).
    - ``:param rebuilt:`` The value decoded from the payload.
    """
    if isinstance(original, dict):
        assert isinstance(rebuilt, dict), f"a dict partial came back as a {type(rebuilt).__name__}"
        assert set(rebuilt) == set(original)
        for key, item in original.items():
            _assert_same_menu_value(item, rebuilt[key])
        return
    assert not isinstance(rebuilt, (list, tuple)), "a partial must not come back as a flattened sequence"
    original_array, rebuilt_array = np.asarray(original), np.asarray(rebuilt)
    assert rebuilt_array.dtype == original_array.dtype, f"{original_array.dtype} widened to {rebuilt_array.dtype}"
    assert rebuilt_array.shape == original_array.shape
    assert np.array_equal(rebuilt_array, original_array)


def test_menu_round_trip_preserves_keys_values_dtypes_and_stamps(fleets, client, monkeypatch):
    """``from_payload(unpackb(packb(as_payload())))`` rebuilds the menu EXACTLY, and the delivery serves from it.

    What a lossy encoding would break, in order of subtlety: the dict partial of ``mean``/``var`` (a flattened list
    would make the aggregation algebra concatenate instead of add), a float32 partial widened to float64 (the merge
    reports the dtype), and the ``timestep``/``rank``/``chunk_position`` stamps (which decide WHICH bridge's chunk a
    value belongs to, and whether a stale menu may answer). The delivery half then hands the DECODED entries to the
    real topic handler, so ``_as_eager_menus`` and the merge really consume the rebuilt values.
    """
    chunks = [chunk.astype(np.float32) for chunk in fleets.fleets[0].chunks]
    received = []
    for rank, chunk in enumerate(chunks):
        collection = DeisaCollection.from_metadata(ARRAY_NAME, 0, WIRE_METADATA[ARRAY_NAME])
        menu = build_eager_menu(collection, chunk, rank=rank, chunk_position=chunk_position(rank))

        # The Client.log_event hop: encode, pack, unpack -- and keep the DECODED entry, i.e. what the host receives.
        entry = msgpack.unpackb(msgpack.packb(menu.as_payload()), raw=False)
        assert isinstance(entry["values"]["mean"], dict) and "dict" in entry["values"]["mean"], (
            "a dict partial must travel as a mapping, never as a flattened list the mean/moment algebra "
            "would concatenate"
        )
        received.append(entry)

        rebuilt = EagerMenu.from_payload(entry)
        assert rebuilt.array_name == ARRAY_NAME
        assert set(rebuilt.values) == set(menu.values)
        for op_name, value in menu.values.items():
            _assert_same_menu_value(value, rebuilt.values[op_name])
        assert np.asarray(rebuilt.values["sum"]).dtype == np.float32, "a float32 scalar must stay float32"
        assert (rebuilt.timestep, rebuilt.rank, rebuilt.chunk_position) == (0, rank, chunk_position(rank))

    deisa = make_deisa(fleets, client, monkeypatch)
    seen = []

    def callback(window):
        seen.append(window[-1].sum().compute())

    register(deisa, callback)
    deliver(deisa, 0, received)

    assert len(seen) == 1
    assert seen[0] == pytest.approx(fleets.fields[0].sum(), rel=1e-5, abs=1e-4)
    assert fleets.rounds(0) == [], "a menu-served fold must not dispatch an on-demand request"


# ---------------------------------------------------------------------------------------------------------------------
# ``Bridge.send`` <-> the on-demand transport (K5F-18). Two lines inside one ``if`` in ``send()`` are the ONLY link
# between the plan transport and the simulation loop: the ``_serve_pending_plans`` boundary call and the
# ``_runtime_last_chunk`` retention. Before this pin, deleting either one left this file at 21 passed -- ``std``, every
# axis reduction and every menu miss could be disconnected from the run with a green suite.
# ---------------------------------------------------------------------------------------------------------------------
class _RecordingReportQueue:
    """The array's plan report queue, recording what the bridge posts instead of handing it to the host."""

    def __init__(self):
        self.reports = []

    def put(self, report):
        """- ``:param report:`` One ``PartialReport`` posted by the rank that roots the array."""
        self.reports.append(report)


def test_send_serves_pending_plans_then_retains_the_chunk_by_reference():
    """``send`` enters the plan boundary on every send and retains the chunk it was handed -- the same object.

    Three properties, all on the real ``send``:

    - the boundary is entered on every send, even with an EMPTY backlog: it is a sub-comm collective, so skipping it on
      the rank that happens to have nothing queued is a one-sided collective (a hang), not a saved round trip;
    - a request is served from the PREVIOUS chunk: the boundary runs before the retention line replaces it, so a
      request naming timestep N is still answered with timestep N's chunk while ``send`` is already at N+1;
    - the retained chunk is the object passed in, not a copy: ``_serve_pending_plans`` only reads it, so a copy would
      double the bridge's per-array footprint for nothing.
    """
    first = np.random.default_rng(0).normal(size=CHUNK_SHAPE).astype(np.float32)
    second = np.random.default_rng(1).normal(size=CHUNK_SHAPE).astype(np.float32)
    bridge, _ = _make_send_bridge()
    # The production shape of a participating bridge: a real backlog (its presence is what makes ``send`` enter the
    # boundary) and the array's report queue.
    bridge._runtime_plan_backlog = {ARRAY_NAME: PlanRequestBacklog()}
    reports = _RecordingReportQueue()
    bridge._runtime_plan_report_queues = {ARRAY_NAME: reports}

    boundary_calls = []
    serve_pending_plans = bridge._serve_pending_plans

    def recording_boundary(array_name, sub_comm):
        """Record the boundary call, then run the REAL boundary.

        - ``:param array_name:`` Array the send is for.
        - ``:param sub_comm:`` The array's sub-communicator.
        """
        boundary_calls.append((array_name, sub_comm))
        return serve_pending_plans(array_name, sub_comm)

    bridge._serve_pending_plans = recording_boundary

    # Send 1: nothing on demand yet. The boundary still runs (an empty drain) and the chunk is retained.
    bridge.send(ARRAY_NAME, first, 0)
    assert len(boundary_calls) == 1, "send() must enter the plan boundary on every send, empty backlog or not"
    assert boundary_calls[0] == (ARRAY_NAME, bridge._array_comms[ARRAY_NAME])
    retained_timestep, retained_chunk = bridge._runtime_last_chunk[ARRAY_NAME]
    assert retained_timestep == 0
    assert retained_chunk is first, "the retained chunk must be the object passed in, not a copy"

    # A request for timestep 0 lands between the two sends: the host asks for the chunk the bridges last delivered.
    request = compose_request(DeisaCollection.from_metadata(ARRAY_NAME, 0, WIRE_METADATA[ARRAY_NAME]).sum())
    bridge._runtime_plan_backlog[ARRAY_NAME].put(
        PlanRequest(request_id="req-0", array_name=ARRAY_NAME, timestep=0, request=request)
    )

    # Send 2: the boundary answers from the PREVIOUS chunk, then the retention takes this one.
    bridge.send(ARRAY_NAME, second, 1)
    bridge._has_close_been_called = True

    assert len(boundary_calls) == 2, "the boundary runs once per send"
    assert boundary_calls[1] == (ARRAY_NAME, bridge._array_comms[ARRAY_NAME])
    retained_timestep, retained_chunk = bridge._runtime_last_chunk[ARRAY_NAME]
    assert retained_timestep == 1
    assert retained_chunk is second, "the retention must follow the boundary, keeping THIS timestep's chunk"
    # The answer belongs to timestep 0: ``first`` was still the retained chunk when the plan was served.
    assert [report.timestep for report in reports.reports] == [0]
    assert float(reports.reports[0].partial) == pytest.approx(float(np.sum(first)), rel=1e-5)
