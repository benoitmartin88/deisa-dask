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
The PRODUCTION transport of the on-demand reduction path: the host's queue exchange and the bridge's plan pump.

Nothing in this file injects a transport. The host half is ``Deisa._attach_ondemand_transports`` -- the exact call
``Deisa.__init__`` makes -- over real ``distributed.Queue`` objects, and the bridge half is the production SHAPE of the
array side: exactly ONE queue consumer per array (the root bridge's poll inside ``PlanPump.serve``), the requests it
takes parked in the root's ``PlanRequestBacklog``, and ``PlanPump.serve`` doing the folding at the send boundary. One
consumer is what production has, and a per-rank pump harness (N consumers) hides the failure mode that matters: one
request handed to one consumer must produce exactly N reports for N bridges, not N per copy of the request.

Nothing polls the request queue between boundaries either: the pickup is one non-blocking drain at the boundary itself,
so a request is answered at the first send AFTER it reached the queue. One test pins that contract directly (an empty
queue folds nothing; a request queued before the boundary is served by it; nothing is served twice) and one pins its
cost (an idle boundary answers in one round trip, not after a grace window).

Only MPI itself is out of scope: the multi-bridge collective runs over ``FakeComm`` (the same stand-in
``test_ondemand_dispatch`` uses), because a two-rank mpi4py run is not something a unit suite can host. Everything that
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
from utils import FakeComm, wait_for

from deisa.dask import bridge as bridge_module
from deisa.dask import deisa as deisa_module
from deisa.dask import ondemand_dispatch as ondemand_dispatch_module
from deisa.dask import ondemand_errors as errors
from deisa.dask.bridge import Bridge, DeliveryState, PlanPump
from deisa.dask.collection import DeisaCollection
from deisa.dask.deisa import ArrayMode, Deisa
from deisa.dask.ondemand_compose import compose_request
from deisa.dask.ondemand_dispatch import (
    PlanRequest,
    QueuePlanExchange,
    plan_report_queue_name,
    plan_request_queue_name,
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
    instance = Client(cluster, name="on-demand-transport-test")
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


def make_bridge_stub(rank, timestep, chunk, request_queue, report_queue, sub_comm):
    """One bridge, with exactly the transport state ``Bridge.__init__`` gives it, and nothing else.

    The array's ``PlanPump`` is the production state (every participating rank has one, flag or no flag); the root
    holds the array's pair of queues, because it is the array's ONE queue consumer; ``DeliveryState.last_chunk`` is
    what the last ``send`` retained.
    """
    bridge = Bridge.__new__(Bridge)
    bridge.id = rank
    bridge.arrays_metadata = {ARRAY_NAME: {**METADATA[ARRAY_NAME], "chunk_position": chunk_position(rank)}}
    pump = PlanPump(ARRAY_NAME, sub_comm, chunk_position(rank), rank)
    if rank == 0:
        pump.request_queue = request_queue
        pump.report_queue = report_queue
    bridge._plan_pumps = {ARRAY_NAME: pump}
    bridge._delivery = {ARRAY_NAME: DeliveryState(fetched=True, last_chunk=(timestep, chunk))}
    return bridge


def serve_boundary(bridge):
    """Drive one rank's send boundary: the array's pump folds whatever chunk the last ``send`` retained."""
    bridge._plan_pumps[ARRAY_NAME].serve(bridge._delivery[ARRAY_NAME].last_chunk)


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
    deisa._callback_seq = 0
    deisa._tasks = set()
    deisa._execute_callbacks_called = False
    deisa._run_failure = None
    deisa._ondemand_transports = {} if transports is None else transports
    deisa._arrays = collections.defaultdict(ArrayMode)
    return deisa


def register_callback(deisa, callback, window_size=1):
    """Register ``callback`` directly: the precompute analysis is not this card's boundary."""
    callback_id = f"{deisa_module.CALLBACK_PREFIX}{len(deisa._callbacks) + 1}"
    deisa._callbacks[callback_id] = {
        "callback": callback,
        "when": "OR",
        "exception_handler": None,
        "array_names": [ARRAY_NAME],
        "precompute": True,
        "state": {
            ARRAY_NAME: {"window": collections.deque(maxlen=window_size), "changed": False, "last_iteration": None}
        },
    }
    deisa._callbacks_by_array.setdefault(ARRAY_NAME, set()).add(callback_id)
    deisa._arrays[ARRAY_NAME].ondemand = True
    return callback_id


def deliver(deisa, timestep):
    """Push one production topic event through the array's handler and wait for the callback task."""
    payload = {"iteration": timestep, "futures": [], "menu": None}
    handler = deisa._topic_handlers[ARRAY_NAME]
    deisa.client.sync(lambda: handler((None, payload)))
    deisa.client.sync(lambda: _drain(deisa._tasks))


def test_the_boundary_poll_is_the_pickup(client):
    """``PlanPump.serve`` polls the host's queue itself: no thread, no stop flag, and exactly one boundary's work.

    Three phases, the whole contract the removed polling thread used to carry:

    - an empty queue folds nothing and posts nothing;
    - a request queued BEFORE the boundary is answered by that boundary -- and a request that arrives after one
      boundary's poll is therefore answered by the NEXT one, because it simply stays in the queue;
    - a request is served ONCE: the boundary after that has nothing left to drain.
    """
    field = make_field(seed=2)
    plan = make_plan(axes=(0,))
    pump = PlanPump(ARRAY_NAME, FakeComm(FakeComm.State(1), 0), chunk_position(0), 0)
    pump.open_queues(client)  # the production root: it derives its own pair of queues from the array name
    host_requests = Queue(plan_request_queue_name(ARRAY_NAME), client=client)
    host_reports = Queue(plan_report_queue_name(ARRAY_NAME), client=client)

    pump.serve((0, field))  # nothing queued: this boundary folds nothing
    with pytest.raises(TimeoutError):
        host_reports.get(timeout=0.2)

    host_requests.put(plan)  # the host asks between two boundaries
    pump.serve((0, field))  # the NEXT boundary serves it
    report = host_reports.get(timeout=20)
    assert report.request_id == plan.request_id
    assert report.error is None, f"no bridge may answer with an error report: {report.error!r}"
    assert np.allclose(report.partial, field.sum(axis=0))

    pump.serve((0, field))  # and only once
    with pytest.raises(TimeoutError):
        host_reports.get(timeout=0.2)


def test_an_idle_boundary_answers_in_a_round_trip_not_a_grace_window(client):
    """The pickup is a non-blocking drain: an empty queue answers at once, it does not wait out a timeout.

    ``distributed.Queue.get(timeout=t)`` is NOT a probe -- the scheduler handler is
    ``wait_for(queue.get(), timeout=t)``, so an EMPTY queue answers only when the timeout expires (~25 ms measured,
    against ~0.2 ms for the batch drain, i.e. ~100x). A ``get(timeout=0.02)`` at this boundary would stall EVERY
    participating array's ``send`` by that much, with the array's peers parked in the boundary ``broadcast`` -- and
    would hand a 20 ms grace window nobody asked for. The bound is deliberately loose (median of 20 idle boundaries,
    10 ms) so a busy box cannot make it flake, while a return to the timeout poll fails it by a wide margin.
    """
    field = make_field(seed=4)
    pump = PlanPump(ARRAY_NAME, FakeComm(FakeComm.State(1), 0), chunk_position(0), 0)
    pump.open_queues(client)  # the production root: it derives its own pair of queues from the array name
    pump.serve((0, field))  # warm-up: the first call pays the queue's first round trip, which is not the cost asserted

    samples = []
    for _ in range(20):
        start = time.perf_counter()
        pump.serve((0, field))  # nothing queued: the idle shape production pays once per array per timestep
        samples.append(time.perf_counter() - start)

    median_ms = 1000.0 * sorted(samples)[len(samples) // 2]
    assert median_ms < 10.0, f"an idle boundary took {median_ms:.2f} ms: the drain is waiting out a timeout"


def test_two_successive_axis_folds_are_served_by_the_production_transport(client, monkeypatch):
    """``sum(axis=0)`` twice, over the wiring production has: ONE root poll, the send boundary, no injected exchange.

    The production shape is the point. Exactly one queue consumer per array (the root bridge's poll inside ``serve``)
    takes the requests at the boundary and broadcasts them to the peers, and every rank folds its chunk and posts one
    report per bridge. A harness with one straight-through pump per rank models N consumers, which production does not
    have: it hides the failure this test exists for -- one request answered N times per copy, so that the second
    request collects the first one's leftovers and the run aborts on its second reduction.
    """
    fields = [make_field(seed=0), make_field(seed=1)]
    per_timestep = [split_chunks(field) for field in fields]
    bridge_count = len(per_timestep[0])
    report_queue = Queue(plan_report_queue_name(ARRAY_NAME), client=client)
    request_queue = Queue(plan_request_queue_name(ARRAY_NAME), client=client)

    deisa = make_deisa_stub(client, monkeypatch)
    # A wiring failure must be a failing test in seconds, not a 60 s stall.
    monkeypatch.setattr(ondemand_dispatch_module, "plan_report_timeout", lambda: 20.0)

    # The wiring under test: the SAME call Deisa.__init__ makes. Nothing is injected into the table.
    deisa._attach_ondemand_transports()
    assert set(deisa._ondemand_transports) == {ARRAY_NAME}, "the production attach must populate the transport table"
    assert isinstance(deisa._ondemand_transports[ARRAY_NAME], QueuePlanExchange)
    deisa._topic_handlers[ARRAY_NAME] = deisa._make_topic_handler(ARRAY_NAME)

    state = FakeComm.State(bridge_count)
    comms = [FakeComm(state, rank) for rank in range(bridge_count)]
    bridges = [
        make_bridge_stub(rank, 0, per_timestep[0][rank], request_queue, report_queue, comms[rank])
        for rank in range(bridge_count)
    ]
    stop = threading.Event()
    # One counter per rank: completed boundary entries. The test retains a timestep's chunk and then waits until every
    # rank has been through the boundary once WITH it, so no rank can still be parked in a collective holding the
    # PREVIOUS timestep's chunk when the host asks -- the production invariant that ``send`` retains the chunk before a
    # request can name that timestep.
    served = [0] * bridge_count

    def serve_rank(rank):
        """Drive one rank's send boundary once.

        - ``:param rank:`` The bridge rank entering the boundary.
        """
        serve_boundary(bridges[rank])
        served[rank] += 1

    def one_round():
        """One full send boundary, every rank in lockstep -- so no rank is ever left inside a collective."""
        peers = [
            threading.Thread(target=serve_rank, args=(rank,), name=f"send-boundary-{rank}", daemon=True)
            for rank in range(1, bridge_count)
        ]
        for peer in peers:
            peer.start()
        serve_rank(0)  # the root polls the host's queue here, exactly as production does
        for peer in peers:
            peer.join(timeout=20)
        assert not any(peer.is_alive() for peer in peers), "a peer rank never left the send boundary"

    def boundary():
        """Re-enter the boundary the way the simulation does, one lockstep round at a time, until the test is done."""
        while not stop.is_set():
            one_round()
            time.sleep(0.002)

    def retain(timestep):
        """Retain the timestep's chunk on every rank, then wait until every boundary has run with it.

        - ``:param timestep:`` The timestep the simulation has just sent.
        """
        for rank in range(bridge_count):
            bridges[rank]._delivery[ARRAY_NAME].last_chunk = (timestep, per_timestep[timestep][rank])
        before = list(served)
        assert wait_for(lambda: all(served[rank] > before[rank] for rank in range(bridge_count)), timeout=20.0), (
            "a rank never re-entered its send boundary after the chunk was retained"
        )

    driver = threading.Thread(target=boundary, name="send-boundary", daemon=True)

    seen = []

    def callback(window):
        seen.append(window[-1].sum(axis=0).compute())

    register_callback(deisa, callback)
    driver.start()
    try:
        for timestep in (0, 1):
            # The simulation has just SENT this timestep: the bridge retains it, so that is what the boundary folds.
            retain(timestep)
            deliver(deisa, timestep)
    finally:
        stop.set()
        driver.join(timeout=20)
        assert not driver.is_alive(), "the send-boundary driver never left its round"
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


def test_backlog_belongs_to_every_participating_array(monkeypatch):
    """The send-boundary collective is decided by PARTICIPATION, and only the QUEUE PAIR needs a dask client.

    A rank still enters the boundary collective the array root broadcasts in. Gating the backlog on a per-process
    setting (the pre-fix shape) made the root issue a ``bcast`` while its peers went straight to the scatter: two
    different collectives on one communicator, i.e. a hang instead of a refusal. The queues, by contrast, are
    reachable only through a dask client, so a rank without one cannot open them.
    """
    bridge = Bridge.__new__(Bridge)
    bridge.id = 0
    bridge.client = None  # this rank owns no dask client: nothing to poll a queue through
    bridge.arrays_metadata = METADATA
    bridge._plan_pumps = {}
    bridge._array_comms = {ARRAY_NAME: FakeComm(FakeComm.State(1), 0), "not-mine": bridge_module._COMM_NULL}

    bridge._start_ondemand_plan_pumps()

    assert set(bridge._plan_pumps) == {ARRAY_NAME}, (
        "every participating array needs a pump, so the send boundary is the same collective on every rank"
    )
    pump = bridge._plan_pumps[ARRAY_NAME]
    assert pump.request_queue is None, "the queue poll is what the client gate stops"
    assert pump.report_queue is None, "no queue is touched without a client"


def test_bridge_send_boundary_serves_the_queued_request_over_the_sub_comm(client):
    """``PlanPump.serve`` -- the send-boundary half -- polls, broadcasts, folds every chunk and answers."""
    field = make_field(seed=1)
    chunks = split_chunks(field)
    plan = make_plan(axes=(0,))

    state = FakeComm.State(len(chunks))
    report_queue = Queue(plan_report_queue_name(ARRAY_NAME), client=client)
    host_requests = Queue(plan_request_queue_name(ARRAY_NAME), client=client)
    comms = [FakeComm(state, rank) for rank in range(len(chunks))]
    bridges = []
    for rank, chunk in enumerate(chunks):
        bridge = Bridge.__new__(Bridge)
        bridge.id = rank
        bridge.arrays_metadata = {ARRAY_NAME: {**METADATA[ARRAY_NAME], "chunk_position": chunk_position(rank)}}
        pump = PlanPump(ARRAY_NAME, comms[rank], chunk_position(rank), rank)
        if rank == 0:
            pump.request_queue = host_requests
            pump.report_queue = report_queue
        bridge._plan_pumps = {ARRAY_NAME: pump}
        bridge._delivery = {ARRAY_NAME: DeliveryState(fetched=True, last_chunk=(0, chunk))}
        bridges.append(bridge)

    # The host puts the request ONCE; the root's boundary poll takes it, the peers receive it through the bcast.
    host_requests.put(plan)

    def serve(rank):
        """Drive one rank's send boundary (the collectives need every rank, as they do in ``send``)."""
        serve_boundary(bridges[rank])

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
