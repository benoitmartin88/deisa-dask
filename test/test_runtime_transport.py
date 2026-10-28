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
#   prior written permission.
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
The PRODUCTION transport of the runtime reduction path: the host's queue exchange and the bridge's plan pump.

Nothing in this file injects a transport. The host half is ``Deisa._attach_runtime_transports`` -- the exact call
``Deisa.__init__`` makes -- over real ``distributed.Queue`` objects, and the bridge half is the production SHAPE of the
array side: exactly ONE queue consumer per array (the root bridge's ``serve_plan_queue`` thread), the requests parked
in the root's ``PlanRequestBacklog``, and ``Bridge._serve_pending_plans`` doing the folding at the send boundary. One
consumer is what production has, and a per-rank pump harness (N consumers) hides the failure mode that matters: one
request handed to one consumer must produce exactly N reports for N bridges, not N per copy of the request.

Only MPI itself is out of scope: the multi-bridge collective runs over ``FakeComm`` (the same stand-in
``test_runtime_dispatch`` uses), because a two-rank mpi4py run is not something a unit suite can host. Everything that
decides whether an axis reduction aborts the run -- the wiring, the queues, the reports, the merge -- is the production
code path, and two SUCCESSIVE reductions are asserted, because the single-request case cannot see a leftover report.

The failure halves are asserted too: an exchange whose bridges never answer refuses the request loudly instead of
merging a partial set with a hole in it, and a reduction of a window element the bridges no longer retain is refused at
compose time instead of being dispatched into an error report.
"""

import asyncio
import collections
import threading
import time
import uuid

import numpy as np
import pytest
from distributed import Client, LocalCluster, Queue
from utils import FakeComm

from deisa.dask import bridge as bridge_module
from deisa.dask import deisa as deisa_module
from deisa.dask import runtime_dispatch as runtime_dispatch_module
from deisa.dask import runtime_errors as errors
from deisa.dask.bridge import Bridge
from deisa.dask.collection import DeisaCollection
from deisa.dask.deisa import Deisa
from deisa.dask.runtime_compose import compose_request
from deisa.dask.runtime_dispatch import (
    PlanRequest,
    PlanRequestBacklog,
    QueuePlanExchange,
    plan_report_queue_name,
    plan_request_queue_name,
    serve_plan_queue,
)

ARRAY_NAME = "f"
GLOBAL_SHAPE = (8, 4)
CHUNK_SHAPE = (4, 4)
GRID_EXTENT = (2, 1)
METADATA = {ARRAY_NAME: {"global_shape": GLOBAL_SHAPE, "chunk_shape": CHUNK_SHAPE}}


@pytest.fixture(scope="module")
def client():
    """A real scheduler: the queues are the production ones, not a stand-in."""
    cluster = LocalCluster(n_workers=1, threads_per_worker=1, processes=False, dashboard_address=None)
    instance = Client(cluster, name="runtime-transport-test")
    try:
        yield instance
    finally:
        instance.close()
        cluster.close()


def make_field(seed=0):
    """The whole field of one timestep; only the test ever holds it."""
    return np.random.default_rng(seed).normal(size=GLOBAL_SHAPE)


def split_chunks(data):
    """Cut ``data`` into the per-bridge chunks the metadata describes."""
    rows = CHUNK_SHAPE[0]
    return [np.array(data[start : start + rows]) for start in range(0, data.shape[0], rows)]


def chunk_position(rank):
    """MPI coordinates of bridge ``rank``'s chunk."""
    return tuple(int(value) for value in np.unravel_index(rank, GRID_EXTENT))


def make_plan(timestep=0, axes=(0,)):
    """A real composed plan for one reduction of the bare collection."""
    collection = DeisaCollection.from_metadata(ARRAY_NAME, timestep, METADATA[ARRAY_NAME])
    request = compose_request(collection.sum(axis=axes))
    return PlanRequest(request_id=uuid.uuid4().hex, array_name=ARRAY_NAME, timestep=timestep, request=request)


def make_bridge_stub(rank, timestep, chunk, report_queue):
    """One bridge, with exactly the transport state ``Bridge.__init__`` gives it, and nothing else.

    ``_runtime_plan_backlog`` is the production state (every participating rank has one, flag or no flag);
    ``_runtime_last_chunk`` is what the last ``send`` retained; only the array root owns a report queue.
    """
    bridge = Bridge.__new__(Bridge)
    bridge.id = rank
    bridge.arrays_metadata = {ARRAY_NAME: {**METADATA[ARRAY_NAME], "chunk_position": chunk_position(rank)}}
    bridge._runtime_plan_backlog = {ARRAY_NAME: PlanRequestBacklog()}
    bridge._runtime_last_chunk = {ARRAY_NAME: (timestep, chunk)}
    bridge._runtime_plan_report_queues = {ARRAY_NAME: report_queue} if rank == 0 else {}
    return bridge


def make_deisa_stub(client, monkeypatch, transports=None):
    """A Deisa stub whose only real wiring is the delivery path under test.

    ``transports=None`` leaves the table EMPTY, so the production attach has to fill it.
    """
    deisa = Deisa.__new__(Deisa)
    deisa.client = client
    deisa.arrays_metadata = METADATA
    deisa._callbacks = {}
    deisa._callbacks_by_array = {}
    deisa._topic_handlers = {}
    deisa._callback_reductions = {}
    deisa._callback_seq = 0
    deisa._branch_groups = {}
    deisa._tasks = set()
    deisa._execute_callbacks_called = False
    deisa._run_failure = None
    deisa._runtime_transports = {} if transports is None else transports
    deisa._runtime_latest_timestep = {}
    monkeypatch.setattr(deisa_module, "runtime_delivery_enabled", lambda: True)
    return deisa


def register_callback(deisa, callback, window_size=1):
    """Register ``callback`` directly: the precompute analysis is not this card's boundary."""
    callback_id = f"{deisa_module.CALLBACK_PREFIX}{len(deisa._callbacks) + 1}"
    deisa._callbacks[callback_id] = {
        "callback": callback,
        "when": "OR",
        "exception_handler": None,
        "array_names": [ARRAY_NAME],
        "state": {
            ARRAY_NAME: {"window": collections.deque(maxlen=window_size), "changed": False, "last_iteration": None}
        },
    }
    deisa._callbacks_by_array.setdefault(ARRAY_NAME, set()).add(callback_id)
    return callback_id


def deliver(deisa, timestep):
    """Push one production topic event through the array's handler and wait for the callback task."""
    payload = {"iteration": timestep, "futures": [], "precomputed": {"f-sum": {"kind": "scalar"}}, "menu": None}
    handler = deisa._topic_handlers[ARRAY_NAME]
    deisa.client.sync(lambda: handler((None, payload)))
    deisa.client.sync(lambda: _drain(deisa._tasks))


def test_serve_plan_queue_moves_requests_into_the_backlog(client):
    """The bridge-side loop is `serve_plan_queue`, unchanged, over a real dask queue."""
    request_queue = Queue(plan_request_queue_name("unit-test-array"), client=client)
    backlog = PlanRequestBacklog()
    stop = threading.Event()
    thread = threading.Thread(target=serve_plan_queue, args=(request_queue, backlog, stop), daemon=True)
    thread.start()
    try:
        plan = make_plan()
        request_queue.put(plan)
        drained = []
        deadline = time.monotonic() + 20.0
        while not drained and time.monotonic() < deadline:
            time.sleep(0.01)
            drained = backlog.drain()
        assert [entry.request_id for entry in drained] == [plan.request_id]
    finally:
        stop.set()
        thread.join(timeout=10)
        assert not thread.is_alive()


def test_two_successive_axis_folds_are_served_by_the_production_transport(client, monkeypatch):
    """``sum(axis=0)`` twice, over the wiring production has: ONE root pump, the send boundary, no injected exchange.

    The production shape is the point. Exactly one queue consumer per array (the root bridge's ``serve_plan_queue``)
    parks requests in the root's backlog, and the array's send boundary broadcasts them to the peers, folds every
    chunk and posts one report per bridge. A harness with one straight-through pump per rank models N consumers, which
    production does not have: it hides the failure this test exists for -- one request answered N times per copy, so
    that the second request collects the first one's leftovers and the run aborts on its second reduction.
    """
    fields = [make_field(seed=0), make_field(seed=1)]
    per_timestep = [split_chunks(field) for field in fields]
    bridge_count = len(per_timestep[0])
    report_queue = Queue(plan_report_queue_name(ARRAY_NAME), client=client)
    request_queue = Queue(plan_request_queue_name(ARRAY_NAME), client=client)

    deisa = make_deisa_stub(client, monkeypatch)
    # A wiring failure must be a failing test in seconds, not a 60 s stall.
    monkeypatch.setattr(runtime_dispatch_module, "plan_report_timeout", lambda: 20.0)

    # The wiring under test: the SAME call Deisa.__init__ makes. Nothing is injected into the table.
    deisa._attach_runtime_transports()
    assert set(deisa._runtime_transports) == {ARRAY_NAME}, "the production attach must populate the transport table"
    assert isinstance(deisa._runtime_transports[ARRAY_NAME], QueuePlanExchange)
    deisa._topic_handlers[ARRAY_NAME] = deisa._make_topic_handler(ARRAY_NAME)

    bridges = [make_bridge_stub(rank, 0, per_timestep[0][rank], report_queue) for rank in range(bridge_count)]
    state = FakeComm.State(bridge_count)
    comms = [FakeComm(state, rank) for rank in range(bridge_count)]
    stop = threading.Event()

    def boundary(rank):
        """One rank's send boundary, re-entered like the simulation re-enters it: in lockstep with every peer."""
        while not stop.is_set():
            bridges[rank]._serve_pending_plans(ARRAY_NAME, comms[rank])
            time.sleep(0.002)

    # ONE consumer for the whole array, exactly as ``Bridge._start_runtime_plan_pumps`` starts it.
    pump_stop = threading.Event()
    pump = threading.Thread(
        target=serve_plan_queue,
        args=(request_queue, bridges[0]._runtime_plan_backlog[ARRAY_NAME], pump_stop),
        name="plan-pump-root",
        daemon=True,
    )
    threads = [
        threading.Thread(target=boundary, args=(rank,), name=f"send-boundary-{rank}", daemon=True)
        for rank in range(bridge_count)
    ]

    seen = []

    def callback(window):
        seen.append(window[-1].sum(axis=0).compute())

    register_callback(deisa, callback)
    pump.start()
    for thread in threads:
        thread.start()
    try:
        for timestep in (0, 1):
            for rank in range(bridge_count):
                # The simulation has just SENT this timestep: the bridge retains it, so that is what the boundary folds.
                bridges[rank]._runtime_last_chunk[ARRAY_NAME] = (timestep, per_timestep[timestep][rank])
            deliver(deisa, timestep)
    finally:
        stop.set()
        pump_stop.set()
        for thread in threads:
            thread.join(timeout=20)
        pump.join(timeout=20)
        # ``Deisa.__del__`` closes ``self.client``; this stub borrows the module-scoped one.
        deisa.client = None

    assert deisa._run_failure is None, f"the run must not abort: {deisa._run_failure!r}"
    assert len(seen) == 2, "both deliveries must reach the callback"
    for timestep, value in enumerate(seen):
        assert np.allclose(value, fields[timestep].sum(axis=0)), (
            f"reduction {timestep} must fold the chunks of ITS timestep"
        )
        assert not isinstance(value, float), "an axis fold is one value per kept column, not a full scalar"


def test_window_reduction_older_than_the_delivery_is_refused_at_compose_time(client, monkeypatch):
    """A reduction of ``window[-2]`` never reaches a bridge: refused at compose time, with the remedy named.

    The bridges retain ONE chunk per array -- the timestep they last sent -- so a plan stamped with an older timestep
    can only come back as an error report and abort the run at the merge. Refusing before the plan is dispatched keeps
    the failure on the host, names what to do instead, and is what makes the window honest: menu operations on an
    older element still work (the menu travels with the delivery), on-demand ones do not.
    """
    dispatched = []

    def forbidden_exchange(plan):
        """Any plan reaching the transport is the bug: the refusal has to fire first."""
        dispatched.append(plan)
        raise AssertionError(f"a stale-timestep plan left the host: {plan!r}")

    deisa = make_deisa_stub(client, monkeypatch, transports={ARRAY_NAME: forbidden_exchange})
    deisa._topic_handlers[ARRAY_NAME] = deisa._make_topic_handler(ARRAY_NAME)
    windows = []

    def callback(window):
        windows.append(len(window))
        if len(window) >= 2:  # window_size=2, so this is window[-2] at the second delivery
            window[-2].sum(axis=0).compute()

    register_callback(deisa, callback, window_size=2)
    for timestep in (0, 1):
        deliver(deisa, timestep)

    deisa.client = None
    assert windows == [1, 2], "the second delivery must reach the callback with a full window"
    assert not dispatched, "no plan may be dispatched for a window element the bridges no longer retain"
    assert isinstance(deisa._run_failure, errors.UnsupportedReductionError), (
        f"the stale-timestep refusal must fail the run by name: {deisa._run_failure!r}"
    )
    message = str(deisa._run_failure)
    assert "timestep 0" in message and "newest timestep delivered" in message
    assert "window[-1]" in message, "the refusal must name the remedy: reduce the latest delivered element"


def test_backlog_belongs_to_every_participating_array_whatever_the_flag(monkeypatch):
    """The send-boundary collective is decided by PARTICIPATION, never by the per-process flag.

    A rank whose flag is off still enters the boundary collective the array root broadcasts in. Gating the backlog on
    the flag (the pre-fix shape) made the root issue a ``bcast`` while its peers went straight to the scatter: two
    different collectives on one communicator, i.e. a hang instead of a refusal. Only the queue pump is gated.
    """
    monkeypatch.setattr(bridge_module, "runtime_delivery_enabled", lambda: False)
    bridge = Bridge.__new__(Bridge)
    bridge.id = 0
    bridge._runtime_plan_backlog = {}
    bridge._runtime_plan_threads = {}
    bridge._runtime_plan_stops = {}
    bridge._runtime_plan_report_queues = {}
    bridge._array_comms = {ARRAY_NAME: FakeComm(FakeComm.State(1), 0), "not-mine": bridge_module._COMM_NULL}

    bridge._start_runtime_plan_pumps()

    assert set(bridge._runtime_plan_backlog) == {ARRAY_NAME}, (
        "every participating array needs a backlog, so the send boundary is the same collective on every rank"
    )
    assert bridge._runtime_plan_threads == {}, "the queue pump is the only thing the flag gates"
    assert bridge._runtime_plan_report_queues == {}, "no queue is touched without the flag"


def test_bridge_send_boundary_serves_the_backlog_over_the_sub_comm(client):
    """``Bridge._serve_pending_plans`` -- the send-boundary half -- broadcasts, folds every chunk and answers."""
    field = make_field(seed=1)
    chunks = split_chunks(field)
    plan = make_plan(axes=(0,))

    state = FakeComm.State(len(chunks))
    report_queue = Queue(plan_report_queue_name(ARRAY_NAME), client=client)
    bridges = []
    backlogs = []
    for rank, chunk in enumerate(chunks):
        bridge = Bridge.__new__(Bridge)
        bridge.id = rank
        bridge.arrays_metadata = {ARRAY_NAME: {**METADATA[ARRAY_NAME], "chunk_position": chunk_position(rank)}}
        backlog = PlanRequestBacklog()
        bridge._runtime_plan_backlog = {ARRAY_NAME: backlog}
        bridge._runtime_last_chunk = {ARRAY_NAME: (0, chunk)}
        bridge._runtime_plan_report_queues = {ARRAY_NAME: report_queue} if rank == 0 else {}
        bridges.append(bridge)
        backlogs.append(backlog)

    # The root's pump thread would have filled its backlog; the peers receive the request through the bcast.
    backlogs[0].put(plan)
    comms = [FakeComm(state, rank) for rank in range(len(chunks))]

    def serve(rank):
        """Drive one rank's send boundary (the collectives need every rank, as they do in ``send``)."""
        bridges[rank]._serve_pending_plans(ARRAY_NAME, comms[rank])

    peer = threading.Thread(target=serve, args=(1,), daemon=True)
    peer.start()
    serve(0)
    peer.join(timeout=20)
    assert not peer.is_alive(), "a peer rank never left the send boundary"

    reports = []
    while len(reports) < len(chunks):
        reports.append(report_queue.get(timeout=20))
    assert all(report.request_id == plan.request_id for report in reports)
    assert all(report.error is None for report in reports), "no bridge may answer with an error report"
    merged = np.sum([report.partial for report in reports], axis=0)
    assert np.allclose(merged, field.sum(axis=0))


def test_exchange_refuses_when_no_bridge_answers(client):
    """A transport nobody serves is a loud refusal, not a silent hang or a short partial set."""
    exchange = QueuePlanExchange(
        Queue(plan_request_queue_name("unserved-array"), client=client),
        Queue(plan_report_queue_name("unserved-array"), client=client),
        bridge_count=2,
        timeout=0.05,
    )
    with pytest.raises(errors.PrecomputeRuntimeError) as excinfo:
        exchange(make_plan(timestep=3))
    assert "0 of 2 report(s)" in str(excinfo.value)


def test_one_request_is_put_once_however_many_bridges(client):
    """One put per request: one consumer per array fans it out, so N puts would be N requests, not a broadcast."""
    request_queue = Queue(plan_request_queue_name("one-put-array"), client=client)
    report_queue = Queue(plan_report_queue_name("one-put-array"), client=client)
    exchange = QueuePlanExchange(request_queue, report_queue, bridge_count=3, timeout=0.05)
    plan = make_plan(timestep=7)

    with pytest.raises(errors.PrecomputeRuntimeError):
        exchange(plan)  # nobody answers, so this refuses after the timeout

    drained = []
    while True:
        try:
            drained.append(request_queue.get(timeout=0.5))
        except TimeoutError:
            break
    assert [entry.request_id for entry in drained] == [plan.request_id], (
        f"exactly one copy of the request may be enqueued, got {len(drained)}"
    )


async def _drain(tasks, timeout=60.0):
    """Wait until no callback task is left: the analogue of the client draining its topic handlers."""
    deadline = time.monotonic() + timeout
    while tasks and time.monotonic() < deadline:
        await asyncio.sleep(0.005)
    assert not tasks, "a callback task never finished"
