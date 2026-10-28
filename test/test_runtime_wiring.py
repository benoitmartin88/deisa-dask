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
import numpy as np
import pytest
from deisa.core.types import DeisaArray
from utils import FakeComm

from deisa.dask import deisa as deisa_module
from deisa.dask import runtime_errors as errors
from deisa.dask.collection import DeisaCollection
from deisa.dask.deisa import Deisa
from deisa.dask.mergeable_pca import MergeablePCA, PCASummary
from deisa.dask.runtime_compose import PCARequest
from deisa.dask.runtime_dispatch import (
    EagerMenu,
    PlanResponder,
    build_eager_menu,
    dispatch_plan_over_comm,
    runtime_delivery_enabled,
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
class _LoopClient:
    """The host-side client calls the topic handler and ``execute_callbacks`` make, on one persistent loop.

    ``sync`` is distributed's ``Client.sync`` analogue and runs on a loop that outlives the call, which is what lets the
    callback tasks created by a topic event be drained afterwards -- the same loop lifetime a real client has.
    """

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.desired_keys = []
        self.persisted = []
        self.id = "fake-analytics-host"

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

    def close(self):  # noqa: D102
        self.loop.close()


class _FakeHandshake:
    """The handshake surface ``execute_callbacks`` touches."""

    def __init__(self):
        self.branches = {}
        self.finished = 0

    def set_task_branches(self, array_name, hints):  # noqa: D102
        self.branches[array_name] = hints

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
    deisa._callback_reductions = {}
    deisa._callback_seq = 0
    deisa._branch_groups = {}
    deisa._tasks = set()
    deisa._execute_callbacks_called = False
    deisa._runtime_transports = {ARRAY_NAME: fleets.exchange}
    deisa._run_failure = None
    deisa._topic_handlers[ARRAY_NAME] = deisa._make_topic_handler(ARRAY_NAME)
    monkeypatch.setattr(deisa_module, "runtime_delivery_enabled", lambda: True)
    return deisa


def register(deisa, callback, window_size=2, when="OR", exception_handler=None):
    """Register ``callback`` directly: the precompute analysis is not this card's boundary."""
    callback_id = f"{deisa_module.CALLBACK_PREFIX}{len(deisa._callbacks) + 1}"
    deisa._callbacks[callback_id] = {
        "callback": callback,
        "when": when,
        "exception_handler": exception_handler,
        "array_names": [ARRAY_NAME],
        "state": {
            ARRAY_NAME: {"window": collections.deque(maxlen=window_size), "changed": False, "last_iteration": None}
        },
    }
    deisa._callbacks_by_array.setdefault(ARRAY_NAME, set()).add(callback_id)
    return callback_id


async def _drain(tasks, timeout=20.0):
    """Wait until no callback task is left: the analogue of the client draining its topic handlers."""
    deadline = time.monotonic() + timeout
    while tasks and time.monotonic() < deadline:
        await asyncio.sleep(0.005)
    assert not tasks, "a callback task never finished"


_DEFAULT_PRECOMPUTED = {"f-sum": {"kind": "scalar"}}


def deliver(deisa, timestep, menus, precomputed=_DEFAULT_PRECOMPUTED):
    """Hand the topic event for ``timestep`` to the real handler, then drain the callback tasks it spawned."""
    payload = {
        "iteration": timestep,
        "futures": [],
        "precomputed": precomputed,
        "menu": menus,
    }
    handler = deisa._topic_handlers[ARRAY_NAME]
    deisa.client.sync(lambda: handler((None, payload)))
    deisa.client.sync(lambda: _drain(deisa._tasks))


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
    """A ``precompute=False`` callback still gets the whole tiled array, not a collection."""
    deisa = make_deisa(fleets, client, monkeypatch)
    received = []
    monkeypatch.setattr(
        Deisa,
        "_Deisa__tile_dask_blocks",
        lambda self, blocks, global_shape: da.zeros(global_shape, chunks=CHUNK_SHAPE),
    )

    def callback(window):
        received.append(window[-1])

    register(deisa, callback)
    deliver(deisa, 0, fleets.menus(0), precomputed=None)

    assert len(received) == 1
    assert isinstance(received[0], DeisaArray)
    assert not isinstance(received[0], DeisaCollection)
    assert received[0].shape == GLOBAL_SHAPE


def test_flag_off_keeps_the_legacy_delivery(fleets, client, monkeypatch):
    """With the flag off the topic handler never builds a collection: the legacy view is selected."""
    deisa = make_deisa(fleets, client, monkeypatch)
    monkeypatch.setattr(deisa_module, "runtime_delivery_enabled", lambda: False)
    legacy_calls = []

    def fake_legacy_views(self, array_name, iteration, combined_by_key):
        legacy_calls.append((array_name, iteration))
        return {}

    def forbidden(self, array_name, iteration, payload):
        raise AssertionError("the runtime delivery must not be selected while the flag is off")

    monkeypatch.setattr(Deisa, "_build_callback_views", fake_legacy_views)
    monkeypatch.setattr(Deisa, "_runtime_views", forbidden)

    register(deisa, lambda window: None)
    deliver(deisa, 0, fleets.menus(0))

    assert legacy_calls == [(ARRAY_NAME, 0)]


def test_delivery_flag_defaults_to_off():
    """The delivery flag is an internal switch read once at import, off by default."""
    assert runtime_delivery_enabled() is False
    assert isinstance(deisa_module.runtime_delivery_enabled(), bool)


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
