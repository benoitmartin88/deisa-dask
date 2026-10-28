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
``Deisa.__init__`` makes -- over real ``distributed.Queue`` objects, and the bridge half is the real
``serve_plan_queue`` thread plus the real ``Bridge._serve_pending_plans`` send boundary. Only MPI itself is out of
scope: the multi-bridge collective runs over ``FakeComm`` (the same stand-in ``test_runtime_dispatch`` uses), because
a two-rank mpi4py run is not something a unit suite can host. Everything that decides whether an axis reduction
aborts the run -- the wiring, the queues, the reports, the merge -- is the production code path.

The failure half is asserted too: an exchange whose bridges never answer refuses the request loudly instead of merging
a partial set with a hole in it.
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

from deisa.dask import deisa as deisa_module
from deisa.dask import runtime_errors as errors
from deisa.dask.bridge import Bridge
from deisa.dask.collection import DeisaCollection
from deisa.dask.deisa import Deisa
from deisa.dask.runtime_compose import compose_request
from deisa.dask.runtime_dispatch import (
    PlanRequest,
    PlanRequestBacklog,
    PlanResponder,
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


def start_bridge_pump(client, rank, chunk, timestep=0):
    """The bridge's production half: one queue poll thread per bridge, over the real queues.

    Returns ``(stop, thread)``. The thread is ``serve_plan_queue`` -- the same function
    ``Bridge._start_runtime_plan_pumps`` starts -- answering straight from a ``PlanResponder`` so this test does not
    need a Bridge instance for the queue half.
    """
    responder = PlanResponder(ARRAY_NAME, timestep, chunk, rank=rank, chunk_position=chunk_position(rank))
    report_queue = Queue(plan_report_queue_name(ARRAY_NAME), client=client)
    request_queue = Queue(plan_request_queue_name(ARRAY_NAME), client=client)
    stop = threading.Event()

    def pump():
        """Move requests from the queue into a local backlog, then answer them."""
        while not stop.is_set():
            try:
                plan = request_queue.get(timeout=0.05)
            except TimeoutError:
                continue
            report_queue.put(responder.respond(plan))

    thread = threading.Thread(target=pump, daemon=True, name=f"plan-pump-{rank}")
    thread.start()
    return stop, thread


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


def test_axis_fold_is_served_by_the_production_transport(client, monkeypatch):
    """``sum(axis=0)`` goes over the queues the production code attaches: no injected exchange, no abort."""
    field = make_field(seed=0)
    chunks = split_chunks(field)
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
    deisa._runtime_transports = {}
    monkeypatch.setattr(deisa_module, "runtime_delivery_enabled", lambda: True)

    # The wiring under test: the SAME call Deisa.__init__ makes. Nothing is injected into the table.
    deisa._attach_runtime_transports()
    assert set(deisa._runtime_transports) == {ARRAY_NAME}, "the production attach must populate the transport table"
    assert isinstance(deisa._runtime_transports[ARRAY_NAME], QueuePlanExchange)

    stops = []
    for rank, chunk in enumerate(chunks):
        stop, _thread = start_bridge_pump(client, rank, chunk)
        stops.append(stop)
    try:
        deisa._topic_handlers[ARRAY_NAME] = deisa._make_topic_handler(ARRAY_NAME)
        seen = []

        def callback(window):
            seen.append(window[-1].sum(axis=0).compute())

        callback_id = f"{deisa_module.CALLBACK_PREFIX}1"
        deisa._callbacks[callback_id] = {
            "callback": callback,
            "when": "OR",
            "exception_handler": None,
            "array_names": [ARRAY_NAME],
            "state": {
                ARRAY_NAME: {
                    "window": collections.deque(maxlen=1),
                    "changed": False,
                    "last_iteration": None,
                }
            },
        }
        deisa._callbacks_by_array[ARRAY_NAME] = {callback_id}

        payload = {"iteration": 0, "futures": [], "precomputed": {"f-sum": {"kind": "scalar"}}, "menu": None}
        handler = deisa._topic_handlers[ARRAY_NAME]
        deisa.client.sync(lambda: handler((None, payload)))
        deisa.client.sync(lambda: _drain(deisa._tasks))
    finally:
        for stop in stops:
            stop.set()
        # ``Deisa.__del__`` closes ``self.client``; this stub borrows the module-scoped one.
        deisa.client = None

    assert deisa._run_failure is None, f"the run must not abort: {deisa._run_failure}"
    assert len(seen) == 1
    assert np.allclose(seen[0], field.sum(axis=0)), "the axis fold must equal the analytic per-column sum"
    assert not isinstance(seen[0], float), "an axis fold is one value per kept column, not a full scalar"


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


async def _drain(tasks, timeout=60.0):
    """Wait until no callback task is left: the analogue of the client draining its topic handlers."""
    deadline = time.monotonic() + timeout
    while tasks and time.monotonic() < deadline:
        await asyncio.sleep(0.005)
    assert not tasks, "a callback task never finished"
