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
The PRODUCTION transport of the on-demand reduction path: the handshake actor and the topic event.

Nothing in this file injects a transport. The host half is ``Deisa._attach_ondemand_transports`` -- the exact call
``Deisa.__init__`` makes -- over a REAL ``Handshake`` (the actor the run already starts), and the bridge half is the
production SHAPE of the array side: exactly ONE actor consumer per array (the root bridge's take inside
``PlanPump.serve``), the requests it takes parked in the root's ``PlanRequestBacklog``, and ``PlanPump.serve`` doing the
folding at the send boundary and RETURNING the gathered reports, which the send publishes on its event. One consumer is
what production has, and a per-rank pump harness (N consumers) hides the failure mode that matters: one request handed
to one consumer must produce exactly N reports for N bridges, not N per copy of the request.

Nothing polls the actor between boundaries either: the pickup is one actor call at the boundary itself, so a request is
answered at the first send AFTER it reached the actor and its reports ride that send's event. One test pins that
contract directly (an idle boundary takes nothing; a request put before the boundary is served by it; nothing is served
twice) and one pins its cost (an idle boundary answers in one round trip, not after a grace window).

Only MPI itself is out of scope: the multi-bridge collective runs over ``FakeComm`` (the same stand-in
``test_ondemand_dispatch`` uses), because a two-rank mpi4py run is not something a unit suite can host. Everything that
decides whether an axis reduction aborts the run -- the wiring, the actor, the events, the reports, the merge -- is the
production code path, and two SUCCESSIVE reductions are asserted, because the single-request case cannot see a leftover
report.

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
from distributed import Client, LocalCluster
from utils import FakeComm

from deisa.dask import bridge as bridge_module
from deisa.dask import deisa as deisa_module
from deisa.dask import ondemand_dispatch as ondemand_dispatch_module
from deisa.dask import ondemand_errors as errors
from deisa.dask.bridge import Bridge, DeliveryState, PlanPump
from deisa.dask.deisa import ArrayMode, Deisa
from deisa.dask.deisa_array import OndemandDeisaArray
from deisa.dask.handshake import Handshake
from deisa.dask.ondemand_compose import compose_request
from deisa.dask.ondemand_dispatch import PlanEventExchange, PlanReportCollector, PlanRequest

ARRAY_NAME = "f"
GLOBAL_SHAPE = (8, 4)
CHUNK_SHAPE = (4, 4)
GRID_EXTENT = (2, 1)
METADATA = {ARRAY_NAME: {"global_shape": GLOBAL_SHAPE, "chunk_shape": CHUNK_SHAPE}}


@pytest.fixture(scope="module")
def client():
    """A real scheduler: the handshake actor is the production one, not a stand-in."""
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


def make_plan(timestep=0, axes=(0,), array_name=ARRAY_NAME):
    """A real composed plan for one reduction of the bare collection."""
    collection = OndemandDeisaArray.from_metadata(array_name, timestep, METADATA[ARRAY_NAME])
    request = compose_request(collection.sum(axis=axes))
    return PlanRequest(request_id=uuid.uuid4().hex, array_name=array_name, timestep=timestep, request=request)


def make_bridge_stub(rank, timestep, chunk, handshake, sub_comm):
    """One bridge, with exactly the transport state ``Bridge.__init__`` gives it, and nothing else.

    The array's ``PlanPump`` is the production state (every participating rank has one); the root holds the array's
    handshake handle, because it is the array's ONE actor consumer; ``DeliveryState.last_chunk`` is what the last
    ``send`` retained.
    """
    bridge = Bridge.__new__(Bridge)
    bridge.id = rank
    bridge.arrays_metadata = {ARRAY_NAME: {**METADATA[ARRAY_NAME], "chunk_position": chunk_position(rank)}}
    pump = PlanPump(ARRAY_NAME, sub_comm, chunk_position(rank), rank)
    pump.attach_handshake(handshake)
    bridge._plan_pumps = {ARRAY_NAME: pump}
    bridge._delivery = {ARRAY_NAME: DeliveryState(fetched=True, last_chunk=(timestep, chunk))}
    return bridge


def serve_boundary(bridge):
    """Drive one rank's send boundary: the pump folds whatever chunk the last ``send`` retained.

    Returns the reports the boundary gathered -- on the root, the set the send publishes on its event.
    """
    return bridge._plan_pumps[ARRAY_NAME].serve(bridge._delivery[ARRAY_NAME].last_chunk)


def make_deisa_stub(client, monkeypatch, transports=None):
    """A Deisa stub whose only real wiring is the delivery path under test.

    ``transports=None`` leaves the table EMPTY, so the production attach has to fill it.
    """
    deisa = Deisa.__new__(Deisa)
    deisa.client = client
    deisa.handshake = Handshake(client)
    deisa.arrays_metadata = METADATA
    deisa._callbacks = {}
    deisa._callbacks_by_array = {}
    deisa._topic_handlers = {}
    deisa._callback_seq = 0
    deisa._tasks = set()
    deisa._execute_callbacks_called = False
    deisa._run_failure = None
    deisa._plan_reports = PlanReportCollector()
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
    deisa._arrays[ARRAY_NAME].analytic = True
    return callback_id


def deliver(deisa, timestep, plan_reports=()):
    """Push one production topic event through the array's handler.

    ``plan_reports`` are the reports the event carries, exactly as the bridge's ``send`` attaches them: the payload
    entries are the wire shape (``PartialReport.as_payload``), so the host-side rebuild runs for real. The callback
    tasks are NOT drained here -- a callback that dispatched a plan is blocked until a LATER event carries its reports,
    which the caller delivers.
    """
    payload = {
        "iteration": timestep,
        "futures": [],
        "aggregates": None,
        "plan_reports": [report.as_payload() for report in plan_reports],
    }
    handler = deisa._topic_handlers[ARRAY_NAME]
    deisa.client.sync(lambda: handler((None, payload)))


def drain(deisa):
    """Wait until no callback task is left: the analogue of the client draining its topic handlers."""
    deisa.client.sync(lambda: _drain(deisa._tasks))


def test_the_boundary_take_is_the_pickup(client):
    """``PlanPump.serve`` takes the host's request out of the actor: no thread, no stop flag, one boundary's work.

    Three phases, the whole contract the removed polling thread used to carry:

    - an idle actor slot folds nothing and returns no reports;
    - a request put BEFORE the boundary is answered by that boundary -- and a request that arrives after one
      boundary's take is therefore answered by the NEXT one, because it simply stays parked;
    - a request is served ONCE: the boundary after that finds the slot empty.
    """
    field = make_field(seed=2)
    plan = make_plan(axes=(0,))
    handshake = Handshake(client)
    pump = PlanPump(ARRAY_NAME, FakeComm(FakeComm.State(1), 0), chunk_position(0), 0)
    pump.attach_handshake(handshake)  # the production root: it takes the host's request from the actor

    assert pump.serve((0, field)) == [], "an idle boundary must not invent a report"

    handshake.put_plan(ARRAY_NAME, plan)  # the host asks between two boundaries
    reports = pump.serve((0, field))  # the NEXT boundary serves it and hands the reports back
    assert [report.request_id for report in reports] == [plan.request_id]
    assert reports[0].error is None, f"no bridge may answer with an error report: {reports[0].error!r}"
    assert np.allclose(reports[0].partial, field.sum(axis=0))

    assert pump.serve((0, field)) == [], "and only once"


def test_the_actor_keeps_one_slot_per_array(client):
    """One slot per array, latest wins: a request nobody waits for is replaced, never served late.

    The host's exchange BLOCKS until the reports of the request it put have arrived, so it never has two requests for
    one array in flight; the replacement rule is what keeps a request whose host side already timed out from being
    served at a later boundary, where its reports would be dropped.
    """
    handshake = Handshake(client)
    first = make_plan(timestep=1)
    second = make_plan(timestep=1)

    handshake.put_plan(ARRAY_NAME, first)
    handshake.put_plan(ARRAY_NAME, second)

    assert [plan.request_id for plan in handshake.take_plans(ARRAY_NAME)] == [second.request_id], (
        "the slot keeps the LATEST request: an older, unserved one is replaced"
    )
    assert handshake.take_plans(ARRAY_NAME) == [], "and taking it empties the slot"


def test_an_idle_boundary_answers_in_a_round_trip_not_a_grace_window(client):
    """The pickup is a non-blocking actor call: an idle boundary answers at once, it does not wait out a timeout.

    The bound is deliberately loose (median of 20 idle boundaries, 10 ms) so a busy box cannot make it flake, while a
    return to a deadline poll -- the retired ``get(timeout=...)`` cost ~25 ms per idle boundary, with the array's peers
    parked in the boundary broadcast -- fails it by a wide margin.
    """
    field = make_field(seed=4)
    pump = PlanPump(ARRAY_NAME, FakeComm(FakeComm.State(1), 0), chunk_position(0), 0)
    pump.attach_handshake(Handshake(client))
    pump.serve((0, field))  # warm-up: the first call pays the actor's first round trip, which is not the cost asserted

    samples = []
    for _ in range(20):
        start = time.perf_counter()
        pump.serve((0, field))  # nothing parked: the idle shape production pays once per array per timestep
        samples.append(time.perf_counter() - start)

    median_ms = 1000.0 * sorted(samples)[len(samples) // 2]
    assert median_ms < 10.0, f"an idle boundary took {median_ms:.2f} ms: the take is waiting on something"


def test_two_successive_axis_folds_are_served_by_the_production_transport(client, monkeypatch):
    """``sum(axis=0)`` twice, over the wiring production has: ONE actor slot, the send boundary, no injected exchange.

    The production shape is the point. Exactly one actor consumer per array (the root bridge's take inside ``serve``)
    picks the requests up at the boundary and broadcasts them to the peers, and every rank folds its chunk; the root
    returns one report per bridge, which travels on the event the SAME send publishes. A harness with one
    straight-through pump per rank models N consumers, which production does not have: it hides the failure this test
    exists for -- one request answered N times per copy, so that the second request collects the first one's
    leftovers and the run aborts on its second reduction.

    The clock is the production one: the event carrying a plan's reports is the array's NEXT send, so the third send
    exists only to deliver the second reduction's reports.
    """
    fields = [make_field(seed=0), make_field(seed=1), make_field(seed=2)]
    per_timestep = [split_chunks(field) for field in fields]
    bridge_count = len(per_timestep[0])
    handshake = Handshake(client)

    deisa = make_deisa_stub(client, monkeypatch)
    # A wiring failure must be a failing test in seconds, not a 60 s stall.
    monkeypatch.setattr(ondemand_dispatch_module, "plan_report_timeout", lambda: 20.0)

    # The wiring under test: the SAME call Deisa.__init__ makes. Nothing is injected into the table.
    deisa._attach_ondemand_transports()
    assert set(deisa._ondemand_transports) == {ARRAY_NAME}, "the production attach must populate the transport table"
    assert isinstance(deisa._ondemand_transports[ARRAY_NAME], PlanEventExchange)
    deisa._topic_handlers[ARRAY_NAME] = deisa._make_topic_handler(ARRAY_NAME)

    state = FakeComm.State(bridge_count)
    comms = [FakeComm(state, rank) for rank in range(bridge_count)]
    bridges = [make_bridge_stub(rank, 0, per_timestep[0][rank], handshake, comms[rank]) for rank in range(bridge_count)]

    def one_round():
        """One full send boundary, every rank in lockstep -- so no rank is ever left inside a collective.

        Returns the reports the ROOT gathered: exactly what the send attaches to the event it publishes.
        """
        peers = [
            threading.Thread(target=serve_boundary, args=(bridges[rank],), name=f"send-boundary-{rank}", daemon=True)
            for rank in range(1, bridge_count)
        ]
        for peer in peers:
            peer.start()
        reports = serve_boundary(bridges[0])  # the root takes the host's request here, exactly as production does
        for peer in peers:
            peer.join(timeout=20)
        assert not any(peer.is_alive() for peer in peers), "a peer rank never left the send boundary"
        return reports

    def retain(timestep):
        """Retain the timestep's chunk on every rank, as ``send`` does AFTER the boundary.

        - ``:param timestep:`` The timestep the simulation has just sent.
        """
        for rank in range(bridge_count):
            bridges[rank]._delivery[ARRAY_NAME].last_chunk = (timestep, per_timestep[timestep][rank])

    seen = []

    def callback(window):
        if window[-1].timestep < 2:  # the third send only carries the second reduction's reports
            seen.append(window[-1].sum(axis=0).compute())

    register_callback(deisa, callback)
    for timestep in (0, 1, 2):
        # The simulation's send for this timestep: the boundary runs FIRST (folding the PREVIOUS chunk), then the chunk
        # is retained, then the event publishes whatever the boundary gathered.
        reports = one_round()
        retain(timestep)
        deliver(deisa, timestep, reports)
    drain(deisa)
    # ``Deisa.__del__`` closes ``self.client``; this stub borrows the module-scoped one.
    deisa.client = None

    assert deisa._run_failure is None, f"the run must not abort: {deisa._run_failure!r}"
    assert len(seen) == 2, "both reductions must reach the callback"
    for timestep, value in enumerate(seen):
        assert np.allclose(value, fields[timestep].sum(axis=0)), (
            f"reduction {timestep} must fold the chunks of ITS timestep"
        )
        assert not isinstance(value, float), "an axis fold is one value per kept column, not a full scalar"


def test_window_reduction_older_than_the_delivery_is_refused_at_compose_time(client, monkeypatch):
    """A reduction of ``window[-2]`` never reaches a bridge: refused at compose time, with the remedy named.

    The bridges retain ONE chunk per array -- the timestep they last sent -- so a plan stamped with an older timestep
    can only come back as an error report and abort the run at the merge. Refusing before the plan is dispatched keeps
    the failure on the host, names what to do instead, and is what makes the window honest: aggregates operations on an
    older element still work (the aggregates travel with the delivery), on-demand ones do not.
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
    drain(deisa)

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
    """The send-boundary collective is decided by PARTICIPATION, and only the TRANSPORT needs a dask client.

    A rank still enters the boundary collective the array root broadcasts in. Gating the backlog on a per-process
    setting (the pre-fix shape) made the root issue a ``bcast`` while its peers went straight to the scatter: two
    different collectives on one communicator, i.e. a hang instead of a refusal. The actor, by contrast, is reachable
    only through a dask client, so a rank without one cannot attach a transport.
    """
    bridge = Bridge.__new__(Bridge)
    bridge.id = 0
    bridge.client = None  # this rank owns no dask client: nothing to take a request through
    bridge.handshake = None
    bridge.arrays_metadata = METADATA
    bridge._plan_pumps = {}
    bridge._array_comms = {ARRAY_NAME: FakeComm(FakeComm.State(1), 0), "not-mine": bridge_module._COMM_NULL}

    bridge._start_ondemand_plan_pumps()
    bridge._attach_plan_transports()

    assert set(bridge._plan_pumps) == {ARRAY_NAME}, (
        "every participating array needs a pump, so the send boundary is the same collective on every rank"
    )
    pump = bridge._plan_pumps[ARRAY_NAME]
    assert pump.handshake is None, "the actor take is what the client gate stops"


def test_bridge_send_boundary_serves_the_actor_request_over_the_sub_comm(client):
    """``PlanPump.serve`` -- the send-boundary half -- takes, broadcasts, folds every chunk and returns the reports."""
    field = make_field(seed=1)
    chunks = split_chunks(field)
    plan = make_plan(axes=(0,))
    handshake = Handshake(client)

    state = FakeComm.State(len(chunks))
    comms = [FakeComm(state, rank) for rank in range(len(chunks))]
    bridges = [make_bridge_stub(rank, 0, chunk, handshake, comms[rank]) for rank, chunk in enumerate(chunks)]

    # The host puts the request ONCE; the root's boundary take gets it, the peers receive it through the bcast.
    handshake.put_plan(ARRAY_NAME, plan)

    def serve(rank):
        """Drive one rank's send boundary (the collectives need every rank, as they do in ``send``)."""
        return serve_boundary(bridges[rank])

    peer = threading.Thread(target=serve, args=(1,), daemon=True)
    peer.start()
    reports = serve(0)
    peer.join(timeout=20)
    assert not peer.is_alive(), "a peer rank never left the send boundary"

    assert len(reports) == len(chunks), "one report per bridge, gathered at the root"
    assert all(report.request_id == plan.request_id for report in reports)
    assert all(report.error is None for report in reports), "no bridge may answer with an error report"
    merged = np.sum([report.partial for report in reports], axis=0)
    assert np.allclose(merged, field.sum(axis=0))


def test_exchange_refuses_when_no_bridge_answers(client):
    """A transport nobody serves is a loud refusal, not a silent hang or a short partial set."""
    exchange = PlanEventExchange(
        Handshake(client),
        PlanReportCollector(),
        bridge_count=2,
        timeout=0.05,
    )
    with pytest.raises(errors.PrecomputeRuntimeError) as excinfo:
        exchange(make_plan(timestep=3, array_name="unserved-array"))
    assert "0 of 2 report(s)" in str(excinfo.value)
    assert "unserved-array" in str(excinfo.value), "the refusal must name the array the request was rooted in"


def test_one_request_is_put_once_however_many_bridges(client):
    """One actor put per request: the root serves it for all N bridges, so N puts would be N rounds, not a broadcast."""
    handshake = Handshake(client)
    exchange = PlanEventExchange(handshake, PlanReportCollector(), bridge_count=3, timeout=0.05)
    plan = make_plan(timestep=7, array_name="one-put-array")

    with pytest.raises(errors.PrecomputeRuntimeError):
        exchange(plan)  # nobody takes, so this refuses after the timeout

    parked = handshake.take_plans("one-put-array")
    assert [entry.request_id for entry in parked] == [plan.request_id], (
        f"exactly one copy of the request may be parked, got {len(parked)}"
    )


async def _drain(tasks, timeout=60.0):
    """Wait until no callback task is left: the analogue of the client draining its topic handlers."""
    deadline = time.monotonic() + timeout
    while tasks and time.monotonic() < deadline:
        await asyncio.sleep(0.005)
    assert not tasks, "a callback task never finished"
