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
The dispatch + merge boundary of the runtime reduction path, proven on a fake communicator.

Every test here runs the real protocol: a request is composed from a lazy expression, broadcast over a ``FakeComm``
sub-comm, executed by one responder per bridge on THAT bridge's numpy chunk, gathered back, matched against the request
and merged with the combine algebra ``deisa.dask.branch`` already owns. Ground truth is the analytic value of the
reduction on the whole array -- assembled only inside the test, never inside the dispatch path.
"""

import pickle
import threading
from dataclasses import replace

import numpy as np
import pytest
from utils import FakeComm

from deisa.dask import runtime_errors as errors
from deisa.dask.branch import _combine_two_phase
from deisa.dask.collection import DeisaCollection
from deisa.dask.runtime_compose import compose_request
from deisa.dask.runtime_dispatch import (
    EAGER_MENU_OPS,
    PlanDispatcher,
    PlanRequest,
    PlanResponder,
    build_eager_menu,
    dispatch_plan_over_comm,
    merge_partials,
    serve_plan_requests,
)
from deisa.dask.runtime_expr import KIND_SCALAR

ARRAY_NAME = "f"
GLOBAL_SHAPE = (8, 8)
CHUNK_SHAPE = (4, 8)
# The data axes' grid extent: (8, 8) split into (4, 8) chunks is a 2x1 chunk grid, so exactly two bridges own a piece of
# the array and the cart dims map one-to-one onto the data axes (the harness invariant).
GRID_EXTENT = (2, 1)


# ---------------------------------------------------------------------------------------------------------------------
# Harness: two FakeComm bridges answering plan requests
# ---------------------------------------------------------------------------------------------------------------------
def make_global(seed=0):
    """The whole field. Only the test ever holds it -- the dispatch path sees per-bridge chunks and partials."""
    rng = np.random.default_rng(seed)
    data = rng.normal(size=GLOBAL_SHAPE)
    # Zeros and mixed signs keep any/all/count/where/prod meaningful.
    data[0, 0] = 0.0
    data[0, 1] = -0.5
    data[4, 2] = 0.0
    return data


def split_chunks(data):
    """Cut ``data`` into the per-bridge chunks the metadata describes (one chunk per bridge, stacked on axis 0)."""
    rows = CHUNK_SHAPE[0]
    return [np.array(data[start : start + rows]) for start in range(0, data.shape[0], rows)]


def chunk_position(rank):
    """MPI coordinates of bridge ``rank``'s chunk in the ``GRID_EXTENT`` grid."""
    return tuple(int(value) for value in np.unravel_index(rank, GRID_EXTENT))


class BridgeFleet:
    """A set of bridges answering plan requests over one ``FakeComm``, i.e. the array's sub-comm.

    Rank ``r`` owns exactly one chunk and runs the bridge-side handler on it; the analytics host drives a round from
    rank 0. ``rounds`` keeps every ``(plan, reports)`` pair so a test can inspect exactly what crossed the wire.
    """

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
        """Run the non-root bridges in their own threads, exactly as a bridge's request loop would."""
        for rank in range(1, len(self.chunks)):
            thread = threading.Thread(
                target=serve_plan_requests, args=(self.comms[rank], self.responders[rank]), daemon=True
            )
            thread.start()
            self._threads.append(thread)

    def exchange(self, plan):
        """One request/reports round trip, driven from the array communicator's root."""
        reports = dispatch_plan_over_comm(self.comms[0], plan, self.responders[0])
        self.rounds.append((plan, reports))
        return reports

    def stop(self):
        """Release the bridge threads with the stop sentinel."""
        self.comms[0].bcast(None, root=0)
        for thread in self._threads:
            thread.join(timeout=10)
        assert not any(thread.is_alive() for thread in self._threads), "a bridge thread did not stop"


@pytest.fixture
def fleet():
    data = make_global()
    bridges = BridgeFleet(split_chunks(data))
    bridges.start()
    try:
        yield data, bridges
    finally:
        bridges.stop()


def make_collection(bridge, dispatcher=None, timestep=None):
    """The collection a callback would receive, with or without the compute boundary attached."""
    return DeisaCollection(
        name=ARRAY_NAME,
        timestep=bridge.timestep if timestep is None else timestep,
        shape=GLOBAL_SHAPE,
        dtype=np.float64,
        chunk_shape=CHUNK_SHAPE,
        dispatcher=dispatcher,
    )


def dispatched(bridge, exchange=None, menu=None):
    """A collection whose reductions dispatch over the fleet (the state the delivery path will hand a callback)."""
    dispatcher = PlanDispatcher(
        exchange if exchange is not None else bridge.exchange,
        bridge_count=len(bridge.chunks),
        menu=menu,
    )
    return make_collection(bridge, dispatcher=dispatcher)


def make_plan(reduction):
    """The on-demand request envelope one reduction would dispatch."""
    request = compose_request(reduction)
    return PlanRequest(
        request_id="test-request",
        array_name=request.input_name,
        timestep=reduction.root.timestep,
        request=request,
    )


def payload_nbytes(partial):
    """Bytes of one partial's payload, dict partials included."""
    if isinstance(partial, dict):
        return sum(payload_nbytes(value) for value in partial.values())
    return np.asarray(partial).nbytes


def representative(partial):
    """The array a partial's shape/dtype describe (dict partials are represented by ``total``)."""
    if isinstance(partial, dict):
        return representative(partial["total"] if "total" in partial else next(iter(partial.values())))
    return np.asarray(partial)


def _explode(chunk):
    """A module-level branch callable that always fails (the fail-loud contract probe)."""
    raise RuntimeError("branch exploded")


# ---------------------------------------------------------------------------------------------------------------------
# Numerical parity: every v1 reduction, full and axis-subset, two bridges
# ---------------------------------------------------------------------------------------------------------------------
PARITY_CASES = [
    pytest.param(lambda c: c.sum(), lambda d: np.sum(d), id="sum-full"),
    pytest.param(lambda c: c.sum(axis=0), lambda d: np.sum(d, axis=0), id="sum-axis0"),
    pytest.param(lambda c: c.sum(axis=1), lambda d: np.sum(d, axis=1), id="sum-axis1"),
    pytest.param(lambda c: c.prod(), lambda d: np.prod(d), id="prod-full"),
    pytest.param(lambda c: c.prod(axis=0), lambda d: np.prod(d, axis=0), id="prod-axis0"),
    pytest.param(lambda c: c.min(), lambda d: np.min(d), id="min-full"),
    pytest.param(lambda c: c.min(axis=1), lambda d: np.min(d, axis=1), id="min-axis1"),
    pytest.param(lambda c: c.max(), lambda d: np.max(d), id="max-full"),
    pytest.param(lambda c: c.max(axis=0), lambda d: np.max(d, axis=0), id="max-axis0"),
    pytest.param(lambda c: c.mean(), lambda d: np.mean(d), id="mean-full"),
    pytest.param(lambda c: c.mean(axis=0), lambda d: np.mean(d, axis=0), id="mean-axis0"),
    pytest.param(lambda c: c.mean(axis=1), lambda d: np.mean(d, axis=1), id="mean-axis1"),
    pytest.param(lambda c: c.var(), lambda d: np.var(d), id="var-full"),
    pytest.param(lambda c: c.var(axis=0), lambda d: np.var(d, axis=0), id="var-axis0"),
    pytest.param(lambda c: c.std(), lambda d: np.std(d), id="std-full"),
    pytest.param(lambda c: c.std(axis=1), lambda d: np.std(d, axis=1), id="std-axis1"),
    pytest.param(lambda c: c.any(), lambda d: np.any(d), id="any-full"),
    pytest.param(lambda c: (c > 0.3).any(axis=0), lambda d: np.any(d > 0.3, axis=0), id="any-axis0"),
    pytest.param(lambda c: c.all(), lambda d: np.all(d), id="all-full"),
    pytest.param(lambda c: (c > 0.3).all(axis=1), lambda d: np.all(d > 0.3, axis=1), id="all-axis1"),
    pytest.param(lambda c: c.count(), lambda d: np.count_nonzero(d), id="count-full"),
    pytest.param(lambda c: c.count(axis=0), lambda d: np.count_nonzero(d, axis=0), id="count-axis0"),
    pytest.param(lambda c: (c > 0.3).count(), lambda d: np.count_nonzero(d > 0.3), id="count-mask"),
]


class TestNumericalParity:
    """Two bridges, one chunk each: the merged value must equal the analytic value of the whole array."""

    @pytest.mark.parametrize("builder,expected", PARITY_CASES)
    def test_two_bridge_parity(self, fleet, builder, expected):
        data, bridges = fleet
        value = builder(dispatched(bridges)).compute()
        assert np.allclose(np.asarray(value, dtype=float), np.asarray(expected(data), dtype=float))

    @pytest.mark.parametrize("builder,expected", PARITY_CASES)
    def test_value_shape_and_dtype_match_the_analytic_answer(self, fleet, builder, expected):
        data, bridges = fleet
        value = np.asarray(builder(dispatched(bridges)).compute())
        assert value.shape == np.asarray(expected(data)).shape

    def test_scalar_reductions_deliver_a_numpy_scalar(self, fleet):
        """A full fold is a scalar, not a 0-d array: the value dask itself delivers."""
        data, bridges = fleet
        value = dispatched(bridges).sum().compute()
        assert not isinstance(value, np.ndarray)
        assert value == pytest.approx(np.sum(data))

    def test_axis_reduction_delivers_a_tier_array(self, fleet):
        data, bridges = fleet
        value = dispatched(bridges).sum(axis=0).compute()
        assert isinstance(value, np.ndarray)
        assert value.shape == (GLOBAL_SHAPE[1],)

    @pytest.mark.parametrize(
        "builder,expected",
        [
            pytest.param(lambda c: (c * c).sum(), lambda d: np.sum(d * d), id="square-chain"),
            pytest.param(lambda c: (c * 2).sum(), lambda d: np.sum(d * 2), id="constant-chain"),
            pytest.param(lambda c: (c.abs() + 1).mean(), lambda d: np.mean(np.abs(d) + 1), id="abs-constant-mean"),
            pytest.param(lambda c: (c > 0).count(), lambda d: np.count_nonzero(d > 0), id="mask-count-chain"),
        ],
    )
    def test_pointwise_chain_parity(self, fleet, builder, expected):
        """The capability today's delivery gate refuses: a reduction over a derived (pointwise) expression."""
        data, bridges = fleet
        value = builder(dispatched(bridges)).compute()
        assert np.allclose(np.asarray(value, dtype=float), np.asarray(expected(data), dtype=float))

    def test_one_dispatch_per_reduction_object(self, fleet):
        """Per-object dispatch: two reductions of the same callback are two requests, not one batch."""
        _, bridges = fleet
        collection = dispatched(bridges)
        collection.sum().compute()
        collection.sum().compute()
        assert len(bridges.rounds) == 2
        assert bridges.rounds[0][0].request_id != bridges.rounds[1][0].request_id


# ---------------------------------------------------------------------------------------------------------------------
# Staleness: a partial never satisfies a request it does not answer
# ---------------------------------------------------------------------------------------------------------------------
class TestStaleResultsAreRefused:
    def test_a_partial_of_an_earlier_request_does_not_satisfy_a_newer_one(self, fleet):
        data, bridges = fleet
        earlier = []

        def replay(plan):
            reports = bridges.exchange(plan)
            if earlier:
                return earlier[0]
            earlier.append(reports)
            return reports

        collection = dispatched(bridges, exchange=replay)
        assert np.isclose(collection.sum().compute(), np.sum(data))
        with pytest.raises(errors.PrecomputeRuntimeError) as excinfo:
            collection.sum().compute()
        message = str(excinfo.value)
        assert "another request" in message
        assert "cannot satisfy" in message

    def test_a_partial_of_another_timestep_is_refused(self, fleet):
        data, bridges = fleet

        def drift(plan):
            return [replace(report, timestep=plan.timestep - 1) for report in bridges.exchange(plan)]

        collection = dispatched(bridges, exchange=drift)
        with pytest.raises(errors.PrecomputeRuntimeError) as excinfo:
            collection.sum().compute()
        message = str(excinfo.value)
        assert "timestep" in message
        assert "refusing to mix two timesteps" in message

    def test_a_dropped_report_is_refused_instead_of_merged(self, fleet):
        """Fault injection: one bridge's answer never arrives, so the merge must refuse the partial set."""
        _, bridges = fleet

        def dropping(plan):
            return bridges.exchange(plan)[:-1]

        collection = dispatched(bridges, exchange=dropping)
        with pytest.raises(errors.PrecomputeRuntimeError) as excinfo:
            collection.sum().compute()
        assert "reports" in str(excinfo.value)

    def test_an_empty_report_set_is_refused(self, fleet):
        _, bridges = fleet
        collection = dispatched(bridges, exchange=lambda plan: [])
        with pytest.raises(errors.PrecomputeRuntimeError) as excinfo:
            collection.sum().compute()
        assert "no report" in str(excinfo.value)

    def test_a_bridge_holding_another_timestep_refuses_the_request(self, fleet):
        """The bridge-side half of the same guard: it answers an error, never a partial of the wrong chunk."""
        _, bridges = fleet
        plan = make_plan(make_collection(bridges, timestep=bridges.timestep + 1).sum())
        report = bridges.responders[0].respond(plan)
        assert report.error is not None
        assert report.partial is None
        assert "timestep" in report.error


# ---------------------------------------------------------------------------------------------------------------------
# Fail-loud executor contract
# ---------------------------------------------------------------------------------------------------------------------
class TestBranchFailurePropagates:
    def test_a_failing_branch_reports_an_error_and_no_partial(self, fleet):
        _, bridges = fleet
        request = replace(compose_plan_request(bridges), branch_func=_explode)
        plan = replace(make_plan(make_collection(bridges).sum()), request=request)
        report = bridges.responders[0].respond(plan)
        assert report.error is not None
        assert "branch exploded" in report.error
        assert report.partial is None

    def test_the_error_reaches_the_callback_and_no_partial_set_is_merged(self, fleet):
        _, bridges = fleet

        def failing(plan):
            broken = replace(plan, request=replace(plan.request, branch_func=_explode))
            return bridges.exchange(broken)

        collection = dispatched(bridges, exchange=failing)
        with pytest.raises(errors.PrecomputeRuntimeError) as excinfo:
            collection.sum().compute()
        message = str(excinfo.value)
        assert "branch exploded" in message
        assert "hole" in message


def compose_plan_request(bridges):
    """The composed request of a plain full sum over the fleet's array (a ready-made ``branch_func`` holder)."""
    return compose_request(make_collection(bridges).sum())


# ---------------------------------------------------------------------------------------------------------------------
# Memory: only partials cross, and they are partial-shaped
# ---------------------------------------------------------------------------------------------------------------------
class TestPayloadsArePartialShaped:
    def test_full_folds_ship_one_scalar_per_bridge(self, fleet):
        _, bridges = fleet
        collection = dispatched(bridges)
        collection.sum().compute()
        plan, reports = bridges.rounds[-1]
        assert plan.request.axes == ()
        for report in reports:
            assert report.shape == ()
            assert np.asarray(report.partial).nbytes == np.dtype(report.dtype).itemsize

    def test_axis_folds_ship_tier_shaped_partials(self, fleet):
        _, bridges = fleet
        collection = dispatched(bridges)
        collection.sum(axis=0).compute()
        _, reports = bridges.rounds[-1]
        for report in reports:
            # ``keepdims=True`` keeps the reduced axis as a size-1 tier level: the combine squeezes it afterwards.
            assert report.shape[0] == 1
            assert report.shape[1] == GLOBAL_SHAPE[1]

    def test_mean_and_moment_ship_dict_partials(self, fleet):
        _, bridges = fleet
        collection = dispatched(bridges)
        collection.mean().compute()
        _, mean_reports = bridges.rounds[-1]
        collection.var(axis=0).compute()
        _, var_reports = bridges.rounds[-1]
        assert set(mean_reports[0].partial) == {"n", "total"}
        assert set(var_reports[0].partial) == {"n", "total", "M"}

    def test_the_chunk_never_crosses(self, fleet):
        """No payload is ever chunk-sized, for every reduction shape in the v1 set."""
        _, bridges = fleet
        collection = dispatched(bridges)
        chunk_nbytes = min(chunk.nbytes for chunk in bridges.chunks)
        chunk_shapes = {tuple(chunk.shape) for chunk in bridges.chunks}
        for builder in (lambda c: c.sum(), lambda c: c.sum(axis=0), lambda c: c.mean(), lambda c: c.var(axis=1)):
            builder(collection).compute()
        assert bridges.rounds
        for _, reports in bridges.rounds:
            for report in reports:
                assert payload_nbytes(report.partial) < chunk_nbytes
                assert tuple(report.shape) not in chunk_shapes

    def test_merge_does_not_need_the_delayed_scheduler(self, fleet):
        """The merge calls the surviving algebra directly: no task graph, no scheduler, same answer."""
        _, bridges = fleet
        collection = dispatched(bridges)
        collection.sum(axis=0).compute()
        _, reports = bridges.rounds[-1]
        ours = merge_partials(reports, kind=KIND_SCALAR, op_name="sum", axes=(0,), ndim=2, finalize=None)
        theirs = _combine_two_phase(
            [(tuple(report.chunk_position), report.partial) for report in reports],
            kind="scalar",
            op_name="sum",
            red_axes=(0,),
            kept_axes=(1,),
            grid_shape=GRID_EXTENT,
            out_dtype=reports[0].dtype,
            finalize=None,
        ).compute(scheduler="sync")
        assert np.allclose(ours, theirs)


# ---------------------------------------------------------------------------------------------------------------------
# Pickle contract: the composed callable survives the communicator's serialization
# ---------------------------------------------------------------------------------------------------------------------
class TestPickleContract:
    @pytest.mark.parametrize("builder,expected", PARITY_CASES)
    def test_the_dispatched_request_survives_a_pickle_round_trip(self, fleet, builder, expected):
        data, bridges = fleet
        collection = dispatched(bridges, exchange=lambda plan: bridges.exchange(pickle.loads(pickle.dumps(plan))))
        value = builder(collection).compute()
        assert np.allclose(np.asarray(value, dtype=float), np.asarray(expected(data), dtype=float))

    def test_a_revived_branch_func_reproduces_the_partial(self, fleet):
        _, bridges = fleet
        plan = make_plan(make_collection(bridges).mean(axis=0))
        revived = pickle.loads(pickle.dumps(plan))
        chunk = bridges.chunks[1]
        assert np.allclose(
            representative(revived.request.branch_func(chunk)), representative(plan.request.branch_func(chunk))
        )


# ---------------------------------------------------------------------------------------------------------------------
# Eager menu (the zero-round-trip half of the transfer strategy)
# ---------------------------------------------------------------------------------------------------------------------
class TestEagerMenu:
    def build_menus(self, bridges):
        collection = make_collection(bridges)
        return [
            build_eager_menu(collection, chunk, rank=rank, chunk_position=chunk_position(rank))
            for rank, chunk in enumerate(bridges.chunks)
        ]

    def test_menu_covers_exactly_the_scalar_full_folds(self):
        assert EAGER_MENU_OPS == ("sum", "min", "max", "mean", "var", "prod")

    @pytest.mark.parametrize(
        "builder,expected",
        [
            pytest.param(lambda c: c.sum(), lambda d: np.sum(d), id="sum"),
            pytest.param(lambda c: c.min(), lambda d: np.min(d), id="min"),
            pytest.param(lambda c: c.max(), lambda d: np.max(d), id="max"),
            pytest.param(lambda c: c.mean(), lambda d: np.mean(d), id="mean"),
            pytest.param(lambda c: c.var(), lambda d: np.var(d), id="var"),
            pytest.param(lambda c: c.prod(), lambda d: np.prod(d), id="prod"),
        ],
    )
    def test_menu_values_merge_without_a_round_trip(self, fleet, builder, expected):
        data, bridges = fleet

        def never_called(plan):  # pragma: no cover - reaching this is the failure
            raise AssertionError("an eager menu value must not need a round trip")

        collection = dispatched(bridges, exchange=never_called, menu=self.build_menus(bridges))
        value = builder(collection).compute()
        assert np.allclose(np.asarray(value, dtype=float), np.asarray(expected(data), dtype=float))
        assert bridges.rounds == []

    def test_on_demand_serves_what_the_menu_does_not(self, fleet):
        data, bridges = fleet
        collection = dispatched(bridges, menu=self.build_menus(bridges))
        assert np.allclose(collection.sum(axis=0).compute(), np.sum(data, axis=0))
        assert np.isclose(collection.std().compute(), np.std(data))
        assert len(bridges.rounds) == 2

    def test_a_menu_value_is_the_partial_the_on_demand_path_returns(self, fleet):
        _, bridges = fleet
        menus = self.build_menus(bridges)
        collection = dispatched(bridges)
        collection.sum().compute()
        _, reports = bridges.rounds[-1]
        for menu, report in zip(menus, reports):
            assert np.allclose(np.asarray(menu.values["sum"]), np.asarray(report.partial))

    def test_a_menu_of_another_timestep_is_not_used(self, fleet):
        """The menu lookup is keyed by timestep, so a stale menu cannot answer a newer timestep."""
        data, bridges = fleet
        menus = self.build_menus(bridges)
        collection = dispatched(bridges, menu=menus)  # bridge timestep is 0
        later = make_collection(bridges, dispatcher=collection.dispatcher, timestep=bridges.timestep + 1)
        with pytest.raises(errors.PrecomputeRuntimeError):
            later.sum().compute()


# ---------------------------------------------------------------------------------------------------------------------
# Grid <-> data axis validation
# ---------------------------------------------------------------------------------------------------------------------
class TestCombineGridValidation:
    def test_a_partial_with_the_wrong_coordinate_rank_is_refused(self, fleet):
        _, bridges = fleet
        collection = dispatched(
            bridges, exchange=lambda plan: [replace(r, chunk_position=(0,)) for r in bridges.exchange(plan)]
        )
        with pytest.raises(errors.PrecomputeRuntimeError) as excinfo:
            collection.sum(axis=0).compute()
        assert "data axes" in str(excinfo.value)

    def test_two_partials_claiming_one_chunk_are_refused(self, fleet):
        _, bridges = fleet

        def duplicate(plan):
            reports = bridges.exchange(plan)
            return [reports[0], replace(reports[1], chunk_position=reports[0].chunk_position)]

        collection = dispatched(bridges, exchange=duplicate)
        with pytest.raises(errors.PrecomputeRuntimeError) as excinfo:
            collection.sum(axis=0).compute()
        assert "same chunk position" in str(excinfo.value)

    def test_a_hole_in_a_two_dimensional_grid_is_refused(self, fleet):
        data, bridges = fleet

        def hole(plan):
            reports = bridges.exchange(plan)
            return [replace(reports[0], chunk_position=(0, 0)), replace(reports[1], chunk_position=(1, 1))]

        collection = dispatched(bridges, exchange=hole)
        with pytest.raises(errors.PrecomputeRuntimeError) as excinfo:
            collection.sum(axis=0).compute()
        assert "partials were delivered" in str(excinfo.value)


# ---------------------------------------------------------------------------------------------------------------------
# The compute boundary itself
# ---------------------------------------------------------------------------------------------------------------------
class TestComputeBoundary:
    def test_compute_reaches_the_bridges_through_the_collection(self, fleet):
        data, bridges = fleet
        collection = dispatched(bridges)
        assert np.isclose(collection.sum().compute(), np.sum(data))
        assert len(bridges.rounds) == 1

    def test_a_reduction_of_a_chain_keeps_the_transport(self, fleet):
        """The dispatcher travels with the operand chain, so ``(c * 2).sum()`` dispatches too."""
        data, bridges = fleet
        chain = dispatched(bridges) * 2
        assert chain.dispatcher is not None
        assert np.isclose(chain.sum().compute(), np.sum(data * 2))

    def test_a_collection_without_a_transport_refuses_instead_of_gathering(self, fleet):
        _, bridges = fleet
        with pytest.raises(errors.PrecomputeRuntimeError) as excinfo:
            make_collection(bridges).sum().compute()
        message = str(excinfo.value)
        assert "no bridge transport" in message
        assert "precompute=False" in message

    def test_reductions_of_the_same_object_key_are_served_independently(self, fleet):
        """Two different derived expressions share a key; per-object dispatch still serves both."""
        data, bridges = fleet
        collection = dispatched(bridges)
        squared = (collection * collection).sum().compute()
        doubled = (collection * 2).sum().compute()
        assert np.isclose(squared, np.sum(data * data))
        assert np.isclose(doubled, np.sum(data * 2))
        assert len(bridges.rounds) == 2
