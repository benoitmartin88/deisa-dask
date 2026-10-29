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
The reduction planner a callback's ``.compute()`` goes through.

The retired AST precompute analyzer walked the callback source symbolically, evaluated dask stubs without executing the
callback, and read hints off each dask task graph. The v1 runtime path plans nothing from the source at all -- the
callback RUNS against a lazily-reducible :class:`~deisa.dask.collection.DeisaCollection`, and every reduction it
computes carries its own ``op_name`` / ``axes`` / ``kind`` / ``finalize``, from which
:func:`~deisa.dask.runtime_compose.compose_request` builds the slim wire request. This file is the fixture suite for
that planner.

The suite keeps the dimensionality of the gysela workload that motivated the design
(``test_gysela_density_helper_plans_the_5d_axis_tuple_and_delivers_the_value`` and
``test_gysela_measure_helper_loop_reduces_five_expressions_per_species``): those are the only fixtures that reduce an
array with ``ndim > 2``, which is why they cannot be dropped as redundant with the 2-D axis fixtures.

Refusals are pinned against the RUNTIME taxonomy (``deisa.dask.runtime_errors``), which is the one a callback meets.
A non-associative reduction (``argmax``/``argmin``/``median``/``quantile``/``percentile``) is refused as
``NonAssociativeReductionError``, and the refusal names the cause (the answer depends on the GLOBAL ordering of the
data) and the remedy (the associative set). The retired analyzer misdiagnosed it -- ``argmax`` as a raw field read,
``median`` as "no precomputable reduction" -- which named a cause and a remedy that did not help.

``_PrecomputedDeisaArray`` role coverage (plan §6, for the deletion card): the class it replaces had four jobs, and
each is proven here by construction, so nothing depends on the class surviving:

1. **mean/var neutralization** -- the reduction is delivered FINAL, never re-applied to a one-element array:
   ``test_full_reduction_delivers_the_final_value`` (mean/var/std) and the ``mean``/``var``/``std`` entries of
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
    _collection,
    _dispatched,
    _run_callback,
    _scalar,
)

from deisa.dask import runtime_errors as errors
from deisa.dask.collection import DeisaCollection
from deisa.dask.deisa import Deisa
from deisa.dask.runtime_dispatch import PlanDispatcher, PlanResponder

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
# The published contract: one literal per fixture body
# ---------------------------------------------------------------------------------------------------------------------
# Every published fixture with the plan it must produce, one tuple per reduction: (key, op, reduced axes, payload kind).
# The list is compared as a SET, so it pins what is planned without pinning the callback's compute order.
#
# ``truth`` is set only for the two pointwise-expression fixtures: their plan folds a composed program into the
# reduction, which is the one place a silent change would still produce a plausible-looking spec. The plain ops are
# pinned by value in the per-section tests below and are deliberately NOT re-tested here.
FIXTURES = [
    pytest.param(
        "result = arr.sum()\nresult.compute()", "arr", False, (("f-sum", "sum", (), "scalar"),), None, id="sum"
    ),
    pytest.param(
        "result = arr.mean()\nresult.compute()", "arr", False, (("f-mean", "mean", (), "mean"),), None, id="mean"
    ),
    pytest.param(
        "result = arr.max()\nresult.compute()", "arr", False, (("f-max", "max", (), "scalar"),), None, id="max"
    ),
    pytest.param(
        "result = arr.min()\nresult.compute()", "arr", False, (("f-min", "min", (), "scalar"),), None, id="min"
    ),
    pytest.param(
        "result = arr.std()\nresult.compute()", "arr", False, (("f-std", "std", (), "moment"),), None, id="std"
    ),
    pytest.param(
        "result = arr.var()\nresult.compute()", "arr", False, (("f-var", "var", (), "moment"),), None, id="var"
    ),
    pytest.param(
        "result = arr.prod()\nresult.compute()", "arr", False, (("f-prod", "prod", (), "scalar"),), None, id="prod"
    ),
    pytest.param(
        "result = arr.sum(axis=0)\nresult.compute()",
        "arr",
        False,
        (("f-sum-axis0", "sum", (0,), "scalar"),),
        None,
        id="axis_int",
    ),
    # A tuple covering every axis IS the full reduction, and normalizes onto the bare-array key.
    pytest.param(
        "result = arr.sum(axis=(0, 1))\nresult.compute()",
        "arr",
        False,
        (("f-sum", "sum", (), "scalar"),),
        None,
        id="axis_tuple_full",
    ),
    pytest.param(
        "result = arr.sum(axis=1)\nresult.compute()",
        "arr",
        False,
        (("f-sum-axis1", "sum", (1,), "scalar"),),
        None,
        id="axis_int_1",
    ),
    # A derived operand's full fold carries the ``-axisall`` discriminator, so it can never collide with the bare
    # array's fold.
    pytest.param(
        "result = (arr ** 2).sum()\nresult.compute()",
        "arr",
        False,
        (("f-sum-axisall", "sum", (), "scalar"),),
        lambda chunk: float(np.sum(chunk**2)),
        id="expression_pow",
    ),
    pytest.param(
        "result = (arr * 2 - 1).mean()\nresult.compute()",
        "arr",
        False,
        (("f-mean-axisall", "mean", (), "mean"),),
        lambda chunk: float(np.mean(chunk * 2 - 1)),
        id="expression_pointwise",
    ),
    pytest.param(
        "s = arr.sum()\nm = arr.mean()\nmx = arr.max()\ns.compute()\nm.compute()\nmx.compute()",
        "arr",
        False,
        (("f-sum", "sum", (), "scalar"), ("f-mean", "mean", (), "mean"), ("f-max", "max", (), "scalar")),
        None,
        id="multiple_reductions",
    ),
    pytest.param(
        "result = arr.sum()\nresult.compute()\nresult0 = arr.sum(axis=0)\nresult0.compute()",
        "arr",
        False,
        (("f-sum", "sum", (), "scalar"), ("f-sum-axis0", "sum", (0,), "scalar")),
        None,
        id="same_op_two_axes",
    ),
    pytest.param(
        "a = arr.sum()\na.compute()\nb = arr.sum()\nb.compute()",
        "arr",
        False,
        (("f-sum", "sum", (), "scalar"), ("f-sum", "sum", (), "scalar")),
        None,
        id="same_op_twice",
    ),
    pytest.param(
        "a = arr.sum().compute()\nb = arr.var().compute()",
        "arr",
        False,
        (("f-sum", "sum", (), "scalar"), ("f-var", "var", (), "moment")),
        None,
        id="independent_reductions",
    ),
    pytest.param(
        "Nsp = 4\nfor isp in range(Nsp):\n    m = arr.sum()\n    m.compute()",
        "arr",
        False,
        (("f-sum", "sum", (), "scalar"),) * 4,
        None,
        id="static_loop",
    ),
    pytest.param(
        "y = arr.ndim\ns = arr.sum()\nreturn s.compute(), y",
        "arr",
        False,
        (("f-sum", "sum", (), "scalar"),),
        None,
        id="attribute_read_then_sum",
    ),
    pytest.param(
        "result = window[-1].sum()\nresult.compute()",
        "window",
        True,
        (("f-sum", "sum", (), "scalar"),),
        None,
        id="window_read",
    ),
    pytest.param(
        "result = window[-1].sum(axis=0)\nresult.compute()",
        "window",
        True,
        (("f-sum-axis0", "sum", (0,), "scalar"),),
        None,
        id="window_axis_read",
    ),
]


def _partial_value(partial) -> float:
    """The number a per-bridge partial stands for: a bare scalar, or a ``mean``/``moment`` ``{n, total}`` dict."""
    if isinstance(partial, dict):
        return float(np.asarray(partial["total"]).reshape(-1)[0]) / float(np.asarray(partial["n"]).reshape(-1)[0])
    return float(np.asarray(partial).reshape(-1)[0])


@pytest.mark.parametrize("body,params,windowed,expected,truth", FIXTURES)
def test_planned_spec_matches_the_published_contract(body, params, windowed, expected, truth) -> None:
    """A published callback plans a fixed literal set: key, op, reduced axes and payload kind per reduction.

    This is the wire contract, so a change here is a wire-format change: the bridge answers the key, the combine
    consumes the kind, and the axes drive the merge. It has to be a deliberate edit of this table, not a side effect of
    a refactor.
    """
    collections, boundary = _window(0) if windowed else ({}, None)
    _, requests, boundary = _run_callback(body, params=params, collections=collections, dispatcher=boundary)
    planned = sorted(
        (reduction.key, reduction.op_name, reduction.axes, reduction.kind) for reduction in boundary.reductions
    )
    assert planned == sorted(expected)
    if truth is not None:
        assert _partial_value(requests[0].branch_func(CHUNK)) == pytest.approx(truth(CHUNK))


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

    Without the finalizer the partial would be dask's ``(1, 4)`` tier shape for a ``(4, 4)`` chunk, which is four times
    the payload for the same information.
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
    """A reduction built inside a helper is planned like an inline one: the helper RUNS, so its reduction crosses a
    compute boundary like any other.
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
    """A loop over a non-constant iterable is ordinary Python: nothing has to be unrolled statically."""

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

    A numpy conversion leaves nothing to plan from, so the gather has to be refused at the coercion itself.
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

    Indexing is not on the closed operator surface, so the access itself is refused
    (``UnsupportedReductionError``) and the data is never served.
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

    This is ``_PrecomputedDeisaArray``'s second-line defense, re-derived structurally: the operand is lazy, so the
    refusal lands where the data would actually be gathered -- inside the callee -- and is not a guess about what the
    callee might do.
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
    """A reduction that depends on another reduction's aggregate is refused.

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
    """Non-associative reductions get their own refusal class, distinct from every data-access refusal.

    The answer depends on the GLOBAL ordering of the data, which no bridge can see, so the refusal names that cause and
    the remedy names the associative set.
    """
    for op_name in ("argmax", "argmin", "median", "quantile", "percentile"):
        with pytest.raises(errors.NonAssociativeReductionError) as refusal:
            _run_callback(f"s = arr.{op_name}()\ns.compute()")
        assert op_name in str(refusal.value)
        assert "global ordering" in str(refusal.value)


def test_non_associative_refusal_is_not_a_raw_read() -> None:
    """The refusal class is ``NonAssociativeReductionError``, never a raw-read class: the cause is the merge, not data.

    ``argmax`` is not a read of the field -- the surface refuses it because the answer needs the GLOBAL ordering -- so
    the callback is pointed at the associative set instead of at a data-access fix.
    """
    with pytest.raises(errors.NonAssociativeReductionError) as refusal:
        _run_callback("s = arr.argmax()\ns.compute()")
    assert not isinstance(refusal.value, errors.RawFieldReadError), "argmax is not a raw read"
    assert "associative reductions" in str(refusal.value), "the remedy must name a working path"


# ---------------------------------------------------------------------------------------------------------------------
# Off-surface spellings: what the closed surface refuses, and why
# ---------------------------------------------------------------------------------------------------------------------
def test_cross_array_expression_is_refused() -> None:
    """``(arr_a - arr_b).sum()`` is refused as a mixed root: a chunk-local branch cannot descend from two arrays.

    The refusal names the root pair, because that is the thing the callback has to change -- the two reductions have to
    be computed separately and combined on the delivered VALUES.
    """
    with pytest.raises(errors.UnsupportedReductionError) as refusal:
        _two_arrays("result = (arr_a - arr_b).max()\nresult.compute()")
    assert "two different DEISA roots" in str(refusal.value)


def test_dask_module_call_is_off_surface() -> None:
    """``da.sum(collection)`` is refused: the surface is the object API, and dask cannot coerce a lazy operand.

    Dask's own coercion trips the closed-surface guard, so a foreign library never gets to build a graph over the field.
    """
    with pytest.raises(errors.UnsupportedReductionError) as refusal:
        _run_callback("result = da.sum(arr)\nresult.compute()")
    assert "closed DEISA operator surface" in str(refusal.value)


def test_fft_is_off_surface() -> None:
    """``da.fft.fft2(collection)`` is refused: FFT is out of the v1 surface (its input is the raw chunk)."""
    with pytest.raises(errors.UnsupportedReductionError):
        _run_callback("phi = da.fft.fft2(arr)\nphi.compute()")


def test_numpy_linalg_on_the_collection_is_refused() -> None:
    """``np.linalg.norm(collection)`` is refused: on the analytics host there is no chunk to consume.

    Numpy's coercion is refused at the point where the gather would have happened, so the whole field never
    materializes on the analytics host. This closes an unsound hole; it is not a capability the runtime should acquire.
    """
    with pytest.raises(errors.MaterializationError):
        _run_callback("v = np.linalg.norm(arr)\ns = arr.sum()\nreturn s.compute(), v")


def test_getattr_resolved_reduction_is_served() -> None:
    """``getattr(arr, 'sum')()`` is SERVED: the surface is the object, and the attribute really is ``sum``.

    The method is resolved at call time like any other attribute of the collection, so a computed operator name is not
    an obstacle -- the reduction it produces is planned normally.
    """
    _, requests, _ = _run_callback("op = 'sum'\nresult = getattr(arr, op)()\nresult.compute()")
    assert [request.key for request in requests] == ["f-sum"]


def test_scaled_reduction_is_refused_and_the_delivered_value_scales_instead() -> None:
    """``arr.sum() * dvx`` is refused: a reduction is a VALUE, so the scale is applied AFTER delivery.

    A reduction used as a pointwise operand would need the aggregate merged before the multiply, which no bridge can
    do. The closed surface therefore refuses the operand spelling and the callback scales the delivered value instead
    (``arr.sum().compute() * dvx``) -- the same number with one callback-side multiply. This is the migration rule for
    the gysela ``density()``/``measure()`` helpers (``da.sum(f, axis=(0, 3, 4)) * dvx * dvy``).
    """
    with pytest.raises(errors.UnsupportedReductionError) as refusal:
        _run_callback("s = (arr.sum()) * 2\ns.compute()")
    assert "completed reduction as an operand" in str(refusal.value)
    assert "compute each reduction separately" in str(refusal.value)


def test_no_compute_boundary_plans_nothing() -> None:
    """A callback that never crosses a compute boundary plans nothing: ``.compute()`` is the only planning site."""
    for body in ("result = arr.sum()", "x = 1 + 2"):
        _, requests, _ = _run_callback(body)
        assert requests == []


# ---------------------------------------------------------------------------------------------------------------------
# Delivered values: the whole-array answer, merged across two bridges
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
def test_full_reduction_delivers_the_final_value(op_name, truth) -> None:
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


def test_derived_reduction_delivers_the_value() -> None:
    """``(arr * arr).sum()`` / ``(arr * 2).mean()`` are served and correct: the pointwise program runs per chunk.

    The chain carries the program instead of assuming the bare chunk, so the fold applies to the program's output on
    every bridge and the merge is over the program's partials.
    """
    collection = _dispatched(FIELD)
    assert np.isclose(float((collection * collection).sum().compute()), float(np.sum(FIELD**2)))
    assert np.isclose(float((collection * 2).mean().compute()), float(np.mean(FIELD * 2)))


# ---------------------------------------------------------------------------------------------------------------------
# The gysela 5-D surface: the workload the redesign was motivated by, on the runtime closed surface
# ---------------------------------------------------------------------------------------------------------------------
# ``fdistribu`` is a 5-D distribution function, and ``density()``/``measure()`` reduce velocity and space axes out of
# it. These two fixtures keep that dimensionality (the axis fixtures above are 2-D) on the closed surface: no
# ``da.sum``, no ``client.compute``, and the grid constants (``dvx * dvy``, the velocity weights, ``dV_4D``) scale the
# DELIVERED value instead of being folded into the branch -- the same rewrite
# ``test_scaled_reduction_is_refused_and_the_delivered_value_scales_instead`` pins as the migration rule for these very
# helpers.
GYSELA_META = {"f": {"global_shape": (4, 5, 10, 10, 10), "chunk_shape": (1, 5, 5, 5, 5)}}
GYSELA_CHUNK_SHAPE = (1, 5, 5, 5, 5)
# One chunk per tile: ``global // chunk`` per data axis, i.e. 32 bridges -- the workload's real grid, not a shortcut.
GYSELA_GRID = (4, 1, 2, 2, 2)


def _gysela_tiles(field, chunk_shape=GYSELA_CHUNK_SHAPE):
    """``(chunk_position, tile)`` for every chunk of the tiled 5-D field."""
    grid = tuple(size // chunk for size, chunk in zip(field.shape, chunk_shape))
    assert grid == GYSELA_GRID, f"the fixture's grid is {grid}, not {GYSELA_GRID}"
    tiles = []
    for position in np.ndindex(*grid):
        slices = tuple(slice(index * chunk, (index + 1) * chunk) for index, chunk in zip(position, chunk_shape))
        tiles.append((position, field[slices]))
    return tiles


class _RecordingPlanDispatcher:
    """The production dispatcher, recording every reduction a helper computes.

    A multi-reduction helper reduces to plain values before it returns, so the reductions are only enumerable at the
    compute boundary; this records them while delegating the delivery itself (value, merge, refusals) to the real
    :class:`~deisa.dask.runtime_dispatch.PlanDispatcher`.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.reductions = []

    def compute(self, reduction):
        """- ``:param reduction:`` The reduction the callback computed."""
        self.reductions.append(reduction)
        return self._inner.compute(reduction)


def _gysela_delivery(field, name="f"):
    """A 5-D collection delivered from ``field``: every tile answered by the production responder and merge.

    The returned boundary is the recording wrapper, so a caller can enumerate what a multi-reduction helper computed.
    """
    tiles = _gysela_tiles(field)
    meta = {name: {"global_shape": field.shape, "chunk_shape": GYSELA_CHUNK_SHAPE}}

    def exchange(plan):
        return [
            PlanResponder(name, 0, tile, rank=rank, chunk_position=position).respond(plan)
            for rank, (position, tile) in enumerate(tiles)
        ]

    dispatcher = _RecordingPlanDispatcher(PlanDispatcher(exchange, bridge_count=len(tiles)))
    return DeisaCollection.from_metadata(name, 0, meta[name], dispatcher=dispatcher), dispatcher


def test_gysela_density_helper_plans_the_5d_axis_tuple_and_delivers_the_value() -> None:
    """``density(f, grid) = sum(f, axis=(0, 3, 4)) * dvx * dvy`` on the 5-D gysela grid.

    The published contract of this body: the axis tuple ``(0, 3, 4)`` and its key, the 200 B partial against the
    5000 B chunk, and the delivered value against numpy. The literals are the contract, so a drift in the axis tuple,
    the wire key or the partial shape is a test failure rather than a deployment surprise.
    """
    from deisa.dask.runtime_compose import probe_partial

    body = "s = arr.sum(axis=(0, 3, 4))\ns.compute()"
    _, requests, _ = _run_callback(body, meta=GYSELA_META)
    assert len(requests) == 1
    request = requests[0]
    assert request.key == "f-sum-axis0x3x4"
    assert request.axes == (0, 3, 4)
    assert request.kind == "scalar"
    assert request.finalize is None

    chunk = np.random.default_rng(11).normal(size=GYSELA_CHUNK_SHAPE)
    assert probe_partial(request, chunk.shape, chunk.dtype) == ((1, 5, 5, 1, 1), "float64")
    # The partial keeps the two velocity/space axes the tuple does NOT reduce: 25 numbers instead of the chunk's 125.
    assert int(np.prod((1, 5, 5, 1, 1))) * np.dtype(np.float64).itemsize == 200
    assert chunk.nbytes == 5000
    assert np.allclose(np.asarray(request.branch_func(chunk)), np.sum(chunk, axis=(0, 3, 4), keepdims=True))

    field = np.random.default_rng(13).normal(size=GYSELA_META["f"]["global_shape"])
    collection, _ = _gysela_delivery(field)
    delivered = collection.sum(axis=(0, 3, 4)).compute()
    truth = np.sum(field, axis=(0, 3, 4))
    assert np.shape(delivered) == truth.shape, f"delivered {np.shape(delivered)}, truth {truth.shape}"
    assert np.allclose(np.asarray(delivered), truth, rtol=1e-12, atol=1e-12)


def test_gysela_measure_helper_loop_reduces_five_expressions_per_species() -> None:
    """``measure()`` in the runtime spelling: five full reductions per species, on the same 5-D grid.

    The fixture asserts the VALUE of each quantity, so a program attached to the wrong reduction cannot hide behind a
    count: ``f``, ``f**2``, ``f*vx``, ``f*vy`` and ``f*v2`` are five separate requests, each folding its own program on
    every bridge's chunk. The grid constants (``vx``/``vy``/``v2`` here, ``dvx``/``dvy``/``dV_4D`` in gysela) are
    applied to the DELIVERED value, which is the closed-surface rewrite of ``da.sum(f * vx) * dV``.

    The wire keys do NOT distinguish the five: a key names the array, the operator and the reduced axes, never the
    pointwise program (``ReductionExpr.key``), so the four derived folds share ``-axisall`` and one pass has TWO
    distinct keys. What is five-fold is the number of requests and of programs -- the value assertions below prove none
    of them was collapsed into another (``f*v2`` alone is 13x the identity's value).

    The constants are 0-d on purpose: a pointwise constant is validated against the GLOBAL shape but evaluated on each
    bridge's CHUNK, so a per-axis velocity grid cannot be broadcast on a distributed axis (a global ``(10,)`` does not
    fit a five-wide chunk). 0-d weights are the chunk-safe spelling of the same rewrite.
    """
    vx, vy = 2.0, 3.0
    v2 = vx**2 + vy**2
    dvx, dvy = 0.5, 0.5
    dV = dvx * dvy

    def measure(collection, vx, vy, v2, dV):
        """The gysela ``measure()`` helper, closed-surface spelling: one full fold per quantity."""
        return {
            "ms": collection.sum().compute() * dV,
            "l2": (collection**2).sum().compute() * dV,
            "mx": (collection * vx).sum().compute() * dV,
            "my": (collection * vy).sum().compute() * dV,
            "ek": 0.5 * (collection * v2).sum().compute() * dV,
        }

    for species, seed in (("f1", 21), ("f2", 22)):
        field = np.random.default_rng(seed).normal(size=GYSELA_META["f"]["global_shape"])
        collection, dispatcher = _gysela_delivery(field, name=species)
        values = measure(collection, vx, vy, v2, dV)

        assert len(dispatcher.reductions) == 5, "one request per quantity, five quantities per species"
        assert [reduction.op_name for reduction in dispatcher.reductions] == ["sum"] * 5
        assert [reduction.key for reduction in dispatcher.reductions] == [f"{species}-sum"] + [
            f"{species}-sum-axisall"
        ] * 4
        assert len({reduction.key for reduction in dispatcher.reductions}) == 2

        truth = {
            "ms": np.sum(field) * dV,
            "l2": np.sum(field**2) * dV,
            "mx": np.sum(field * vx) * dV,
            "my": np.sum(field * vy) * dV,
            "ek": 0.5 * np.sum(field * v2) * dV,
        }
        for quantity, expected in truth.items():
            assert np.isclose(float(values[quantity]), float(expected), rtol=1e-12, atol=1e-12), (
                f"species {species}: {quantity} = {values[quantity]}, expected {expected}"
            )


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
    # Delivery mode is per CALLBACK (decision F3): the view builder reads each payload's ``precompute`` flag, so the
    # stubbed registrations must carry one like a real one does.
    deisa._callbacks = {"cb-1": {"precompute": True}, "cb-2": {"precompute": True}}

    views = deisa._runtime_views("f", 3, {"menu": []})

    assert set(views) == {"cb-1", "cb-2"}
    assert views["cb-1"] is not views["cb-2"]
    views["cb-1"].name = "mutated-by-the-first-callback"
    assert views["cb-2"].name == "f"
    assert views["cb-2"].timestep == 3


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
