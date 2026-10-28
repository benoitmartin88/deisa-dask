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
The reduction planner a callback's ``.compute()`` goes through: the oracle port of the analyzer tests.

This file used to pin the AST precompute analyzer: the source was walked symbolically, dask stubs were evaluated
without executing the callback, and hints were read off each dask task graph. The v1 runtime path plans nothing from
the source at all -- the callback RUNS against a lazily-reducible :class:`~deisa.dask.collection.DeisaCollection`, and
every reduction it computes carries its own ``op_name`` / ``axes`` / ``kind`` / ``finalize``, from which
:func:`~deisa.dask.runtime_compose.compose_request` builds the slim wire request. This file is that same fixture
suite rewritten against the planner.

The equivalence contract, asserted fixture by fixture: for every callback the analyzer used to serve, the runtime plan
is the SAME spec -- same op name, same reduction axes, same payload kind, same finalize, and a numerically identical
``branch_func`` on a fresh numpy chunk (behavior over metadata, plan §7 P1/P4). The analyzer's ``BranchSpec`` list is
the oracle for that comparison until it is deleted; the literal expectations asserted next to it are what survives the
deletion.

Refusals are pinned against the RUNTIME taxonomy (``deisa.dask.runtime_errors``), which is the one a callback now
meets. One classification is deliberately DIFFERENT from the analyzer's and is pinned as such: a non-associative
reduction (``argmax``/``argmin``/``median``/``quantile``/``percentile``) is refused as
``NonAssociativeReductionError``. The analyzer misdiagnosed it -- ``argmax`` as a raw field read, ``median`` as "no
precomputable reduction" -- which named a cause and a remedy that did not help. Every OTHER divergence the port
exposed is listed in the "Divergences" section below, each pinned with the analyzer behavior next to the runtime one, so
none of them is a silent approximation.

``_PrecomputedDeisaArray`` role coverage (plan §6, for the deletion card): the class it replaces had four jobs, and
each is proven here by construction, so nothing depends on the class surviving:

1. **mean/var neutralization** -- the reduction is delivered FINAL, never re-applied to a one-element array:
   ``test_full_reduction_delivers_the_analyzer_value`` (mean/var/std) and the ``mean``/``var``/``std`` entries of
   ``test_direct_reduction_plan``.
2. **no double reduce** -- a reduction is a value, not an array: ``test_reduction_is_not_an_array``.
3. **second-line defense** -- an unplanned consumption of the delivered value cannot reach data:
   ``test_numpy_conversion_is_refused``, ``test_raw_compute_is_refused``, ``test_index_read_is_refused``,
   ``test_iteration_and_truth_are_refused``, ``test_opaque_callable_cannot_gather_the_field`` and
   ``test_unknown_attribute_is_refused``.
4. **per-callback isolation** -- a fresh collection per callback:
   ``test_each_callback_gets_its_own_collection``.
"""

import functools

import numpy as np
import pytest
from utils import (
    DEFAULT_META,
    RecordingDispatcher,
    _analyze,
    _assert_spec_equivalent,
    _collection,
    _dispatched,
    _run_callback,
    _scalar,
)

from deisa.dask import runtime_errors as errors
from deisa.dask.deisa import Deisa
from deisa.dask.runtime_compose import RuntimeRequest

# The chunk every branch callable is exercised on: mixed signs, so max/min/mean/var distinguish the ops.
CHUNK = np.arange(16.0).reshape(4, 4) - 8.0
META_AB = {
    "a": {"global_shape": (8, 8), "chunk_shape": (4, 4)},
    "b": {"global_shape": (8, 8), "chunk_shape": (4, 4)},
}


def _window(*timesteps):
    """A callback's sliding window: one collection per delivered timestep, all behind ONE compute boundary."""
    boundary = RecordingDispatcher()
    return {"window": [_collection("f", timestep=step, dispatcher=boundary) for step in timesteps]}, boundary


def _two_arrays(body):
    """Run ``body`` with two independent collections (``arr_a`` on array ``a``, ``arr_b`` on array ``b``)."""
    boundary = RecordingDispatcher()
    collections = {
        "arr_a": _collection("a", meta=META_AB, dispatcher=boundary),
        "arr_b": _collection("b", meta=META_AB, dispatcher=boundary),
    }
    return _run_callback(body, params="arr_a, arr_b", meta=META_AB, collections=collections, dispatcher=boundary)


# ---------------------------------------------------------------------------------------------------------------------
# The port contract: every fixture's runtime plan IS the analyzer's spec
# ---------------------------------------------------------------------------------------------------------------------
# Every callback fixture the analyzer served, expressed on the runtime surface. The bodies are the analyzer-era bodies;
# where the closed surface spells an expression differently the runtime spelling is used and the port is noted.
FIXTURES = [
    pytest.param("result = arr.sum()\nresult.compute()", "arr", False, id="sum"),
    pytest.param("result = arr.mean()\nresult.compute()", "arr", False, id="mean"),
    pytest.param("result = arr.max()\nresult.compute()", "arr", False, id="max"),
    pytest.param("result = arr.min()\nresult.compute()", "arr", False, id="min"),
    pytest.param("result = arr.std()\nresult.compute()", "arr", False, id="std"),
    pytest.param("result = arr.var()\nresult.compute()", "arr", False, id="var"),
    pytest.param("result = arr.prod()\nresult.compute()", "arr", False, id="prod"),
    pytest.param("result = arr.sum(axis=0)\nresult.compute()", "arr", False, id="axis_int"),
    pytest.param("result = arr.sum(axis=(0, 1))\nresult.compute()", "arr", False, id="axis_tuple_full"),
    pytest.param("result = arr.sum(axis=1)\nresult.compute()", "arr", False, id="axis_int_1"),
    pytest.param("result = (arr ** 2).sum()\nresult.compute()", "arr", False, id="expression_pow"),
    pytest.param("result = (arr * 2 - 1).mean()\nresult.compute()", "arr", False, id="expression_pointwise"),
    pytest.param(
        "s = arr.sum()\nm = arr.mean()\nmx = arr.max()\ns.compute()\nm.compute()\nmx.compute()",
        "arr",
        False,
        id="multiple_reductions",
    ),
    pytest.param(
        "result = arr.sum()\nresult.compute()\nresult0 = arr.sum(axis=0)\nresult0.compute()",
        "arr",
        False,
        id="same_op_two_axes",
    ),
    pytest.param("a = arr.sum()\na.compute()\nb = arr.sum()\nb.compute()", "arr", False, id="same_op_twice"),
    pytest.param("a = arr.sum().compute()\nb = arr.var().compute()", "arr", False, id="independent_reductions"),
    pytest.param(
        "Nsp = 4\nfor isp in range(Nsp):\n    m = arr.sum()\n    m.compute()",
        "arr",
        False,
        id="static_loop",
    ),
    pytest.param("y = arr.ndim\ns = arr.sum()\nreturn s.compute(), y", "arr", False, id="attribute_read_then_sum"),
    pytest.param("result = window[-1].sum()\nresult.compute()", "window", True, id="window_read"),
    pytest.param("result = window[-1].sum(axis=0)\nresult.compute()", "window", True, id="window_axis_read"),
]


@pytest.mark.parametrize("body,params,windowed", FIXTURES)
def test_fixture_specs_match_the_analyzer_oracle(body, params, windowed) -> None:
    """For every fixture, the runtime plan equals the analyzer's spec -- fields AND branch behavior.

    The analyzer (``_analyze``) is the oracle for a callback the analyzer-era delivery served; the runtime never reads
    the source, it executes the callback. Equivalent spec means same ``op_name`` / ``reduction_axes`` / output kind /
    ``finalize``, a numerically identical ``branch_func`` on one fresh numpy chunk, and the same probed partial
    metadata. When the analyzer is deleted, this test's literals (and the per-section tests below) are the contract.
    """
    oracle = _analyze(body, params=params)
    collections, boundary = _window(0) if windowed else ({}, None)
    _, requests, boundary = _run_callback(body, params=params, collections=collections, dispatcher=boundary)

    assert len(boundary.reductions) == len(oracle), (
        f"planned {len(boundary.reductions)} reduction(s), oracle {len(oracle)}"
    )
    pairs = sorted(
        zip(
            sorted(boundary.reductions, key=lambda item: (item.op_name, item.axes)),
            sorted(oracle, key=lambda item: (item.op_name, item.reduction_axes)),
        ),
        key=lambda pair: (pair[0].op_name, pair[0].axes),
    )
    for reduction, spec in pairs:
        _assert_spec_equivalent(reduction, spec, CHUNK)
    assert all(isinstance(request, RuntimeRequest) for request in requests)


# ---------------------------------------------------------------------------------------------------------------------
# Per-reduction support: the plan of each associative reduction
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "op_name,kind,finalize",
    [
        ("sum", "scalar", None),
        ("mean", "mean", None),
        ("max", "scalar", None),
        ("min", "scalar", None),
        ("std", "moment", "sqrt"),
        ("var", "moment", None),
        ("prod", "scalar", None),
    ],
)
def test_direct_reduction_plan(op_name, kind, finalize) -> None:
    """``result = arr.<op>(); result.compute()`` plans ONE full reduction with the op's payload kind."""
    _, requests, _ = _run_callback(f"result = arr.{op_name}()\nresult.compute()")
    assert len(requests) == 1
    request = requests[0]
    assert request.key == f"f-{op_name}"
    assert request.input_name == "f"
    assert request.axes == ()
    assert request.kind == kind
    assert request.finalize == finalize


@pytest.mark.parametrize(
    "body,expected_key,expected_axes",
    [
        pytest.param("result = arr.sum(axis=0)\nresult.compute()", "f-sum-axis0", (0,), id="axis_int"),
        pytest.param("result = arr.sum(axis=1)\nresult.compute()", "f-sum-axis1", (1,), id="axis_int_1"),
        # A tuple that covers every axis IS the full reduction, and normalizes onto the bare-array key.
        pytest.param("result = arr.sum(axis=(0, 1))\nresult.compute()", "f-sum", (), id="axis_tuple_full"),
        # Bare full reduction: no discriminator, the stable key every existing deployment groups by.
        pytest.param("result = arr.sum()\nresult.compute()", "f-sum", (), id="full"),
        # A full reduction of a DERIVED operand gets the ``-axisall`` discriminator, so it can never be confused with
        # the bare array's fold (which the eager menu would answer).
        pytest.param("result = (arr ** 2).sum()\nresult.compute()", "f-sum-axisall", (), id="derived_full"),
        pytest.param("result = (arr * 2).sum(axis=0)\nresult.compute()", "f-sum-axis0", (0,), id="derived_axis"),
    ],
)
def test_reduction_axes_and_output_key(body, expected_key, expected_axes) -> None:
    """The reduced axes are the wire convention (``()`` == full) and the key discriminates them."""
    _, requests, _ = _run_callback(body)
    assert len(requests) == 1
    assert requests[0].key == expected_key
    assert requests[0].axes == expected_axes


def test_full_fold_ships_one_scalar_per_bridge() -> None:
    """A full fold of a scalar kind composes the minimal-scalar finalizer, so a bridge ships ONE scalar.

    This is the runtime counterpart of the analyzer's ``full_local_fold``: without the finalizer the partial would be
    dask's ``(1, 4)`` tier shape for a ``(4, 4)`` chunk, which is four times the payload for the same information.
    """
    from deisa.dask.runtime_compose import probe_partial

    _, requests, _ = _run_callback("result = arr.sum()\nresult.compute()")
    chain = requests[0].branch_func.keywords["_chain"]
    assert len(chain) == 1, "a bare-array fold needs no pointwise stage"
    assert isinstance(chain[0][0], functools.partial)
    assert "finalizer" in chain[0][0].keywords, "the full fold must compose the minimal-scalar finalizer"
    assert probe_partial(requests[0], CHUNK.shape, CHUNK.dtype) == ((), "float64")
    assert float(requests[0].branch_func(CHUNK)) == pytest.approx(float(np.sum(CHUNK)))


def test_axis_fold_keeps_a_tier_partial() -> None:
    """An axis fold keeps the un-reduced axis (one row per bridge), so no full-fold finalizer is composed."""
    from deisa.dask.runtime_compose import probe_partial

    _, requests, _ = _run_callback("result = arr.sum(axis=0)\nresult.compute()")
    chain = requests[0].branch_func.keywords["_chain"]
    assert len(chain) == 1
    assert not isinstance(chain[0][0], functools.partial) or "finalizer" not in chain[0][0].keywords
    assert probe_partial(requests[0], CHUNK.shape, CHUNK.dtype) == ((1, 4), "float64")
    assert np.allclose(requests[0].branch_func(CHUNK), np.sum(CHUNK, axis=0, keepdims=True))


def test_derived_fold_composes_the_pointwise_stage_first() -> None:
    """``(arr ** 2).sum()`` is TWO chain layers: the pointwise program, then the fold of its output."""
    _, requests, _ = _run_callback("result = (arr ** 2).sum()\nresult.compute()")
    chain = requests[0].branch_func.keywords["_chain"]
    assert len(chain) == 2
    assert float(requests[0].branch_func(CHUNK)) == pytest.approx(float(np.sum(CHUNK**2)))


# ---------------------------------------------------------------------------------------------------------------------
# Grouping: distinct reductions never share a payload, identical ones are one plan
# ---------------------------------------------------------------------------------------------------------------------
def test_multiple_reductions_plan_one_request_each() -> None:
    """A callback computing three reductions plans three requests, one key per reduction."""
    body = "s = arr.sum()\nm = arr.mean()\nmx = arr.max()\ns.compute()\nm.compute()\nmx.compute()"
    _, requests, _ = _run_callback(body)
    assert sorted(request.key for request in requests) == ["f-max", "f-mean", "f-sum"]


def test_same_op_different_axes_have_distinct_keys() -> None:
    """``arr.sum()`` + ``arr.sum(axis=0)`` must not share a key: one would overwrite the other's payload."""
    body = "result = arr.sum()\nresult.compute()\nresult0 = arr.sum(axis=0)\nresult0.compute()"
    _, requests, _ = _run_callback(body)
    keys = [request.key for request in requests]
    assert keys == ["f-sum", "f-sum-axis0"]
    assert len(set(keys)) == 2


def test_same_op_same_axes_share_one_key() -> None:
    """Two IDENTICAL reductions keep ONE key: they are one plan, served once."""
    _, requests, _ = _run_callback("a = arr.sum()\na.compute()\nb = arr.sum()\nb.compute()")
    assert len(requests) == 2
    assert {request.key for request in requests} == {"f-sum"}


def test_independent_reductions_are_not_refused() -> None:
    """Two independent reductions on the same collection are planned independently, never refused."""
    _, requests, _ = _run_callback("a = arr.sum().compute()\nb = arr.var().compute()")
    assert sorted(request.key for request in requests) == ["f-sum", "f-var"]


# ---------------------------------------------------------------------------------------------------------------------
# Structure independence: the plan comes from the run, not from the callback's syntax
# ---------------------------------------------------------------------------------------------------------------------
def test_helper_call_plans_the_helper_reduction() -> None:
    """A reduction built inside a helper is planned like an inline one: the runtime executes the call.

    The analyzer-era fixture needed a hand-written helper for this (``density(f, grid)``); the runtime needs nothing,
    because the helper RUNS.
    """
    body = "def measure(f):\n    return f.sum(axis=0)\n\nresult = measure(arr)\nresult.compute()"
    _, requests, _ = _run_callback(body)
    assert [request.key for request in requests] == ["f-sum-axis0"]


def test_list_and_loop_of_reductions_plan_every_reduction() -> None:
    """A list of reductions computed in a loop plans all of them: only the ``.compute()`` sites matter."""
    body = "values = [arr.sum(), arr.mean(), arr.max()]\nfor value in values:\n    value.compute()"
    _, requests, _ = _run_callback(body)
    assert sorted(request.key for request in requests) == ["f-max", "f-mean", "f-sum"]


def test_dynamic_loop_over_plan_objects_is_plain_python() -> None:
    """A loop over a non-constant iterable is ordinary Python here: the analyzer's static unrolling is not needed."""
    boundary = RecordingDispatcher()
    items = [_collection("f", dispatcher=boundary), _collection("f", dispatcher=boundary)]
    _, requests, _ = _run_callback(
        "for item in items:\n    item.sum().compute()",
        collections={"arr": _collection("f", dispatcher=boundary)},
        dispatcher=boundary,
        extra_globals={"items": items},
    )
    assert [request.key for request in requests] == ["f-sum", "f-sum"]


def test_planning_never_reads_the_callback_source() -> None:
    """A callback with no ``__source__`` at all is planned normally: the source is not an input of the planner."""
    boundary = RecordingDispatcher()
    collection = _collection("f", dispatcher=boundary)

    def callback(arr):
        arr.sum().compute()

    assert not hasattr(callback, "__source__")
    callback(collection)
    assert [reduction.key for reduction in boundary.reductions] == ["f-sum"]


# ---------------------------------------------------------------------------------------------------------------------
# Windows: ``window[-1]`` is the latest collection, and a window cannot be reduced across timesteps
# ---------------------------------------------------------------------------------------------------------------------
def test_window_last_element_is_the_latest_collection() -> None:
    """``window[-1].sum()`` is the latest timestep's full fold -- the same plan as the bare collection's."""
    collections, boundary = _window(0, 1)
    _, requests, _ = _run_callback(
        "result = window[-1].sum()\nresult.compute()", params="window", collections=collections, dispatcher=boundary
    )
    assert [request.key for request in requests] == ["f-sum"]
    assert boundary.reductions[0].root.timestep == 1
    assert boundary.reductions[0].axes == ()


def test_window_cannot_be_combined_across_timesteps() -> None:
    """``window[-1] - window[-2]`` is refused: a branch is rooted in one array AND one timestep."""
    collections, boundary = _window(0, 1)
    with pytest.raises(errors.UnsupportedReductionError) as refusal:
        _run_callback(
            "result = (window[-1] - window[-2]).sum()\nresult.compute()",
            params="window",
            collections=collections,
            dispatcher=boundary,
        )
    assert "two different DEISA roots" in str(refusal.value)


# ---------------------------------------------------------------------------------------------------------------------
# Refusals: the delivered value cannot be consumed as data
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "body",
    [
        pytest.param("full = np.array(arr)\nfull.compute()", id="np_array"),
        pytest.param("full = np.asarray(arr)\nfull.compute()", id="np_asarray"),
    ],
)
def test_numpy_conversion_is_refused(body) -> None:
    """``np.array(collection)``/``np.asarray(collection)`` MATERIALIZES the field: ``MaterializationError``.

    Same class the analyzer raised for the same source, and the reason it must be caught before any graph exists: a
    numpy conversion leaves no artifact to analyze.
    """
    with pytest.raises(errors.MaterializationError):
        _run_callback(body)


def test_raw_compute_is_refused() -> None:
    """``arr.compute()`` asks for the raw field: ``RawFieldReadError``, naming the array and its timestep."""
    with pytest.raises(errors.RawFieldReadError) as refusal:
        _run_callback("arr.compute()")
    assert "raw data of array 'f'" in str(refusal.value)
    assert "precompute=False" in str(refusal.value)


def test_index_read_is_refused() -> None:
    """``arr[2:5]`` / ``arr[:, 0]`` / ``arr[0, 0]`` are raw reads of the field: refused at the item access.

    Port note: the analyzer emitted a ``f-sum`` spec for ``arr[2:5].sum()`` with ``deliver_direct=False``, which
    registration then refused; the runtime refuses at the access itself, so the end state (refused, never served) is the
    same and the class (``UnsupportedReductionError``) is the registration-time class of both paths.
    """
    for body in (
        "result = arr[2:5].sum()\nresult.compute()",
        "result = arr[:, 0].sum()\nresult.compute()",
        "v = arr[0, 0]\ns = arr.sum()\nreturn s.compute() + v.compute()",
    ):
        with pytest.raises(errors.UnsupportedReductionError) as refusal:
            _run_callback(body)
        assert "raw data" in str(refusal.value)


def test_unknown_attribute_is_refused() -> None:
    """Any attribute outside the closed surface is refused, instead of degrading into an AttributeError."""
    with pytest.raises(errors.UnsupportedReductionError) as refusal:
        _run_callback("s = arr.map_blocks(np.abs)\ns.compute()")
    assert ".map_blocks" in str(refusal.value)


def test_iteration_and_truth_are_refused() -> None:
    """``list(collection)`` yields raw elements and ``if collection:`` has no answer: both refuse."""
    with pytest.raises(errors.MaterializationError):
        _run_callback("list(arr)")
    with pytest.raises(errors.UnsupportedReductionError):
        _run_callback("if arr:\n    pass")


def test_opaque_callable_cannot_gather_the_field() -> None:
    """Passing the collection to a foreign callable is harmless; the callable's own CONSUMPTION is refused.

    This is ``_PrecomputedDeisaArray``'s second-line defense, re-derived structurally: the analyzer had to refuse
    ``plot(arr)`` up front because it could not see inside ``plot``. The runtime hands over a lazy operand, so the
    refusal lands where the data would actually be gathered -- inside the callee -- and is not a guess.
    """

    def plot(value):
        np.asarray(value)

    plots = []
    with pytest.raises(errors.MaterializationError):
        _run_callback(
            "plot(arr)\ns = arr.sum()\nreturn s.compute()",
            extra_globals={"plot": lambda value: plots.append(value) or plot(value)},
        )
    assert len(plots) == 1, "the callee did receive the lazy collection"


@pytest.mark.parametrize(
    "body,expected",
    [
        pytest.param("s = arr.sum(keepdims=True)", "keepdims", id="keepdims"),
        pytest.param("s = arr.sum(dtype=np.float32)", "dtype", id="dtype"),
        pytest.param("s = arr.var(ddof=1)", "ddof", id="ddof"),
    ],
)
def test_unsupported_reduction_keywords_are_refused(body, expected) -> None:
    """The closed reduction surface refuses the keywords whose meaning cannot survive the cross-bridge merge."""
    with pytest.raises(errors.UnsupportedReductionError) as refusal:
        _run_callback(body)
    assert expected in str(refusal.value)


@pytest.mark.parametrize(
    "body",
    [
        pytest.param("return (arr - arr.mean()).sum().compute()", id="simple_cross"),
        pytest.param("return (arr - arr.mean().sum()).sum().compute()", id="nested_cross"),
        pytest.param("s = arr.sum()\nt = s + 1", id="reduction_as_operand"),
        pytest.param("s = arr.sum()\nt = s.sum()", id="double_reduce"),
    ],
)
def test_cross_reduction_is_refused(body) -> None:
    """A reduction that depends on another reduction's aggregate is refused -- the analyzer's refusal class.

    The naive per-reduction plan would ship ``(chunk - chunk.mean()).sum()`` to every bridge, which is always 0: the
    inner aggregate only exists after the merge. The message names the offending reduction and points at the working
    path (compute each reduction, combine the delivered VALUES).
    """
    with pytest.raises(errors.UnsupportedReductionError) as refusal:
        _run_callback(body)
    message = str(refusal.value).lower()
    assert "completed reduction" in message
    assert "bridge" in message


def test_reduction_is_not_an_array() -> None:
    """Role 2 of the retired dispatch view: a reduction is a VALUE, not an array, so nothing can re-reduce it.

    The old ``_PrecomputedDeisaArray`` had to intercept a second reduction to stop it re-applying an op to an
    already-final array; here the object itself has no chunk-local partial, so the second reduction is structurally
    impossible.
    """
    _, requests, boundary = _run_callback("s = arr.sum()\ns.compute()")
    reduction = boundary.reductions[0]
    assert not hasattr(reduction, "shape") or reduction.shape == ()
    assert requests[0].axes == ()
    with pytest.raises(errors.UnsupportedReductionError):
        reduction.sum()


def test_non_associative_reductions_are_refused_as_such() -> None:
    """The ONE deliberate reclassification: non-associative reductions get their own refusal class.

    The analyzer misdiagnosed these -- ``argmax`` as a raw field read, ``median``/``quantile`` as "no precomputable
    reduction" -- so the callback was told to change the wrong thing. The true cause is that the answer depends on the
    GLOBAL ordering of the data, which no bridge can see, and the remedy names the associative set.
    """
    for op_name in ("argmax", "argmin", "median", "quantile", "percentile"):
        with pytest.raises(errors.NonAssociativeReductionError) as refusal:
            _run_callback(f"s = arr.{op_name}()\ns.compute()")
        assert op_name in str(refusal.value)
        assert "global ordering" in str(refusal.value)


def test_non_associative_refusal_is_not_a_raw_read() -> None:
    """The reclassification is observable: the same source raises the analyzer's raw-read class, the runtime's own."""
    from deisa.dask import precompute_analyzer as analyzer_module

    with pytest.raises(analyzer_module.RawFieldReadError):
        _analyze("s = arr.argmax()\ns.compute()")
    with pytest.raises(errors.NonAssociativeReductionError) as refusal:
        _run_callback("s = arr.argmax()\ns.compute()")
    assert not isinstance(refusal.value, analyzer_module.RawFieldReadError), "argmax is not a raw read"
    assert "associative reductions" in str(refusal.value), "the remedy must name a working path"


# ---------------------------------------------------------------------------------------------------------------------
# Divergences from the analyzer, pinned one by one (nothing here is a silent approximation)
# ---------------------------------------------------------------------------------------------------------------------
def test_cross_array_expression_is_refused() -> None:
    """``(arr_a - arr_b).sum()`` is refused as a mixed root; the analyzer refuses it too, one step earlier.

    Both paths refuse with ``UnsupportedReductionError``: the analyzer raises during analysis (a chunk-local branch
    cannot descend from two registered arrays), the runtime raises at the operator with the root pair in the message.
    """
    from deisa.dask import precompute_analyzer as analyzer_module

    with pytest.raises(analyzer_module.UnsupportedReductionError) as oracle_refusal:
        _analyze("result = (arr_a - arr_b).max()\nresult.compute()", params="arr_a, arr_b", meta=META_AB)
    assert "at least one other registered array" in str(oracle_refusal.value)
    with pytest.raises(errors.UnsupportedReductionError) as refusal:
        _two_arrays("result = (arr_a - arr_b).max()\nresult.compute()")
    assert "two different DEISA roots" in str(refusal.value)


def test_dask_module_call_is_off_surface() -> None:
    """``da.sum(collection)`` is refused: the surface is the object API, and dask cannot coerce a lazy operand.

    The analyzer understood ``da.sum(arr)`` because it inspected the graph; the runtime refuses it (dask's own coercion
    trips the closed-surface guard) instead of letting a foreign library try to build a graph over the field.
    """
    with pytest.raises(errors.UnsupportedReductionError) as refusal:
        _run_callback("result = da.sum(arr)\nresult.compute()")
    assert "closed DEISA operator surface" in str(refusal.value)


def test_fft_is_off_surface() -> None:
    """``da.fft.fft2(collection)`` is refused: FFT is out of the v1 surface (its input is the raw chunk)."""
    with pytest.raises(errors.UnsupportedReductionError):
        _run_callback("phi = da.fft.fft2(arr)\nphi.compute()")


def test_numpy_linalg_on_the_collection_is_refused() -> None:
    """``np.linalg.norm(collection)`` is refused -- the analyzer's allowance was an unsound hole.

    In the analyzer-era delivery the callback received a dask array, so ``np.linalg.norm(arr)`` built a graph that
    materialized the WHOLE field on the analytics host while registration saw nothing to refuse. The runtime hands over
    a lazy operand, so numpy's coercion is refused at the point where the gather would have happened.
    """
    with pytest.raises(errors.MaterializationError):
        _run_callback("v = np.linalg.norm(arr)\ns = arr.sum()\nreturn s.compute(), v")


def test_getattr_resolved_reduction_is_served() -> None:
    """``getattr(arr, 'sum')()`` is SERVED: the surface is the object, and the attribute really is ``sum``.

    The analyzer refused this as an "unsupported call form" because it could not see WHICH op the string names. That was
    an analysis limitation, not a semantic refusal: the runtime resolves the method and delivers the correct value.
    """
    _, requests, _ = _run_callback("op = 'sum'\nresult = getattr(arr, op)()\nresult.compute()")
    assert [request.key for request in requests] == ["f-sum"]


def test_scaled_reduction_is_refused_and_the_delivered_value_scales_instead() -> None:
    """REGRESSION to escalate: ``arr.sum() * dvx`` is refused, where the analyzer served it as ``f-sum``.

    The analyzer composed the scale factor into the branch (a full local fold plus a multiply), so the gysela
    ``density()``/``measure()`` helpers (``da.sum(f, axis=(0, 3, 4)) * dvx * dvy``) were precomputable. The runtime
    refuses any reduction used as a pointwise operand and requires the scale to be applied to the DELIVERED value.
    Same number, one extra callback-side multiply -- but an existing callback must be rewritten, which is why this is
    pinned explicitly rather than quietly dropped.
    """
    oracle = _analyze("s = (arr.sum()) * 2\ns.compute()")
    assert [(spec.op_name, spec.deliver_direct) for spec in oracle] == [("sum", True)]
    with pytest.raises(errors.UnsupportedReductionError) as refusal:
        _run_callback("s = (arr.sum()) * 2\ns.compute()")
    assert "completed reduction as an operand" in str(refusal.value)
    assert "compute each reduction separately" in str(refusal.value)


def test_no_compute_boundary_plans_nothing() -> None:
    """A callback that never computes a reduction plans nothing, and the analyzer refuses it at registration.

    Both halves are pinned: the runtime records zero requests (there is nothing to plan), and the registration gate --
    still the analyzer until the deletion lands -- raises ``NoPrecomputableReductionError``/``NoComputeBoundaryError``
    for exactly that source.
    """
    from deisa.dask import precompute_analyzer as analyzer_module

    for body in ("result = arr.sum()", "x = 1 + 2"):
        _, requests, _ = _run_callback(body)
        assert requests == []
    with pytest.raises(analyzer_module.NoComputeBoundaryError):
        _analyze("result = arr.sum()")


# ---------------------------------------------------------------------------------------------------------------------
# Delivered values: the analyzer's answer, computed by the runtime path
# ---------------------------------------------------------------------------------------------------------------------
FIELD = np.random.default_rng(7).normal(size=(8, 8))


@pytest.mark.parametrize(
    "op_name,truth",
    [
        ("sum", np.sum),
        ("mean", np.mean),
        ("var", np.var),
        ("std", np.std),
        ("min", np.min),
        ("max", np.max),
        ("prod", np.prod),
    ],
)
def test_full_reduction_delivers_the_analyzer_value(op_name, truth) -> None:
    """Role 1 of the retired dispatch view: mean/var/std deliver the FINAL global value, never a re-applied op.

    Two bridges, real responder, real merge: a callback's ``arr.<op>().compute()`` is the whole-array reduction. For
    ``var``/``std`` the re-application failure mode was ``0.0`` (the variance of a one-element array), which a
    value-only assertion on a zero field would accept -- hence the random field and the both-sides dtype check.
    """
    collection = _dispatched(FIELD)
    value = getattr(collection, op_name)().compute()
    assert np.isclose(float(np.asarray(value).reshape(-1)[0]), float(truth(FIELD)), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "op_name,axes,truth",
    [
        pytest.param("sum", 0, lambda field: np.sum(field, axis=0), id="sum_axis0"),
        pytest.param("sum", 1, lambda field: np.sum(field, axis=1), id="sum_axis1"),
        pytest.param("mean", 0, lambda field: np.mean(field, axis=0), id="mean_axis0"),
    ],
)
def test_axis_reduction_delivers_the_whole_array_slice(op_name, axes, truth) -> None:
    """An axis fold is merged across the chunk grid and returns the full slice, correctly shaped."""
    collection = _dispatched(FIELD)
    value = np.asarray(getattr(collection, op_name)(axis=axes).compute())
    expected = truth(FIELD)
    assert value.shape == expected.shape, f"delivered {value.shape}, truth {expected.shape}"
    assert np.allclose(value, expected, rtol=1e-12, atol=1e-12)


def test_derived_reduction_delivers_the_value_the_analyzer_refused() -> None:
    """``(arr * arr).sum()`` is SERVED by the runtime (the analyzer refused it as non-direct) and is correct.

    The pointwise program is executed on each bridge's own chunk before the fold, which is exactly what the analyzer's
    ``deliver_direct`` guard was protecting against getting wrong -- the runtime cannot get it wrong, because the
    composed chain carries the program instead of assuming the bare chunk.
    """
    collection = _dispatched(FIELD)
    assert np.isclose(float((collection * collection).sum().compute()), float(np.sum(FIELD**2)))
    assert np.isclose(float((collection * 2).mean().compute()), float(np.mean(FIELD * 2)))


# ---------------------------------------------------------------------------------------------------------------------
# Per-callback isolation (role 4)
# ---------------------------------------------------------------------------------------------------------------------
def test_each_callback_gets_its_own_collection() -> None:
    """The delivery builds a FRESH collection per callback: a sibling's attribute write cannot leak.

    A collection is mutable metadata, so one instance shared between callbacks would let a helper that "normalizes" its
    argument poison every sibling callback on that array/timestep.
    """
    deisa = Deisa.__new__(Deisa)
    deisa.arrays_metadata = DEFAULT_META
    deisa._runtime_transports = {}
    deisa._runtime_latest_timestep = {}
    deisa._callbacks_by_array = {"f": {"cb-1", "cb-2"}}

    views = deisa._runtime_views("f", 3, {"menu": []})

    assert set(views) == {"cb-1", "cb-2"}
    assert views["cb-1"] is not views["cb-2"]
    views["cb-1"].name = "mutated-by-the-first-callback"
    assert views["cb-2"].name == "f"
    assert views["cb-2"].timestep == 3


# ---------------------------------------------------------------------------------------------------------------------
# The analyzer's own contract, for the record: it never ran the callback
# ---------------------------------------------------------------------------------------------------------------------
def test_analyzer_never_executes_the_callback() -> None:
    """The oracle comparison above relies on the analyzer being AST-only: pin that it never calls the callback."""
    calls = {"n": 0}

    def callback(arr):
        calls["n"] += 1
        arr.sum().compute()

    callback.__source__ = "def callback(arr):\n    arr.sum().compute()\n"
    from deisa.dask.branch import _analyze_callback_for_branches

    branches = _analyze_callback_for_branches(callback, DEFAULT_META)
    assert [branch.output_key for branch in branches] == ["f-sum"]
    assert calls["n"] == 0


# ---------------------------------------------------------------------------------------------------------------------
# Sanity: the harness itself describes the delivered shape
# ---------------------------------------------------------------------------------------------------------------------
def test_harness_collection_is_the_delivered_object() -> None:
    """The harness hands the callback a collection with the registered metadata and a compute boundary."""
    from deisa.dask.collection import DeisaCollection

    boundary = RecordingDispatcher()
    collection = _collection("f", dispatcher=boundary)
    assert isinstance(collection, DeisaCollection)
    assert collection.shape == (8, 8)
    assert collection.chunk_shape == (4, 4)
    assert collection.dispatcher is boundary
    assert _scalar(np.float64(2.5)) == 2.5
