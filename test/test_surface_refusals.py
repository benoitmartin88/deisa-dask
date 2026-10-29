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
"""The closed surface a callback meets, and the refusals outside it.

A callback RUNS against a lazily-reducible :class:`~deisa.dask.collection.DeisaCollection`: the reductions it computes
are planned, and everything else is ordinary Python over metadata. This file pins the other half of that contract --
the spellings that are OUTSIDE the surface, the class each refusal raises, and the boundary cases where refusing would
be wrong (sibling reductions, predicate builtins, metadata reads, plain Python control flow).

Every refusal here is a property of the runtime surface, stated together with the spelling that works instead. The
notes below collect the calls that look surprising in either direction.

- **``map_blocks``**: refused, opaque or resolvable, because the surface never inspects a callable. The intent of
  ``arr.map_blocks(np.abs).sum()`` is served by the pointwise spelling ``abs(arr).sum()`` -- same plan, same value.
- **branch selection**: unknown values are EXECUTED. The plan is exactly the arm the callback takes, so a
  control-flow-dependent reduction is planned correctly for every state, not guessed from the source.
- **opaque callee on raw data**: handing the operand to a foreign callable is free; the refusal lands where the data
  would be gathered (inside the callee), as ``MaterializationError``.
- **resolvable module call on raw data**: ``np.linalg.norm(arr)`` is refused with ``MaterializationError``. On the
  analytics host there is no chunk to consume, so letting numpy coerce the operand would gather the whole field.
- **compute boundary**: ``arr.compute()`` is ``RawFieldReadError``, naming the array and the timestep.
- **index read**: ``arr[0, 0]`` is off-surface (``UnsupportedReductionError``) -- indexing is not an operator.
- **``hasattr`` on an off-surface attribute**: ``__getattr__`` raises a refusal instead of ``AttributeError``, so
  ``hasattr(arr, "graph")`` RAISES rather than returning ``False``. Loud instead of wrong.
- **``len(arr)``**: refused, because ``__len__`` is the protocol every raw-data consumer reaches for. The shape IS
  exposed as metadata, so a callback that sized the field must read ``arr.shape``.

``_PrecomputedDeisaArray`` role coverage (plan §6, for the deletion card). The retired dispatch view had four jobs;
this file proves two of them by construction:

- **no double reduce** (role 2) -- a reduction is a VALUE, so nothing can re-apply an op to it:
  ``test_cross_reduction_is_refused`` (``s.sum()`` / ``s + 1`` are refused structurally).
- **second-line defense** (role 3) -- an unplanned consumption of the delivered operand cannot reach data:
  ``test_raw_field_read_via_opaque_call_refused_at_consumption``, ``test_raw_field_read_list_unwrapping``,
  ``test_resolvable_module_call_on_raw_data_is_refused``, ``test_raw_field_compute_boundary_without_reduction_refused``,
  ``test_raw_field_index_read_refused``, ``test_len_on_raw_data_is_refused`` and
  ``test_hasattr_on_an_off_surface_attribute_raises``.

(The other two roles -- final mean/var delivery and per-callback isolation -- are covered by ``test_precompute.py``.)
"""

import numpy as np
import pytest
from utils import RecordingDispatcher, _collection, _run_callback, _scalar

from deisa.dask import runtime_errors as errors
from deisa.dask.runtime_compose import probe_partial

META_AB = {
    "a": {"global_shape": (8, 8), "chunk_shape": (4, 4)},
    "b": {"global_shape": (8, 8), "chunk_shape": (4, 4)},
}


def _two_arrays(body: str):
    """Run ``body`` with two independent collections (``arr_a`` on ``a``, ``arr_b`` on ``b``)."""
    boundary = RecordingDispatcher()
    collections = {
        "arr_a": _collection("a", meta=META_AB, dispatcher=boundary),
        "arr_b": _collection("b", meta=META_AB, dispatcher=boundary),
    }
    return _run_callback(body, params="arr_a, arr_b", meta=META_AB, collections=collections, dispatcher=boundary)


# ---------------------------------------------------------------------------------------------------------------------
# map_blocks: off-surface; the pointwise spelling carries the same reduction
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "body",
    [
        pytest.param("y = arr.map_blocks(lambda b: b * 2)\ns = y.sum()\nreturn s.compute()", id="opaque_lambda"),
        pytest.param("y = arr.map_blocks(np.abs)\ns = y.sum()\nreturn s.compute()", id="resolvable_func"),
    ],
)
def test_map_blocks_is_off_surface(body) -> None:
    """``arr.map_blocks(...)`` is refused, opaque or resolvable -- the surface is closed and inspects no callable.

    Folding ``np.abs`` into a branch would mean reasoning about a callable, which the runtime never does, so both forms
    are refused at the same gate and the message names the spellings that work: the pointwise operator, or
    ``precompute=False``.
    """
    with pytest.raises(errors.UnsupportedReductionError) as refusal:
        _run_callback(body)
    assert ".map_blocks" in str(refusal.value)
    assert "closed DEISA operator surface" in str(refusal.value)
    assert "precompute=False" in str(refusal.value), "the remedy must name a working path"


def test_map_blocks_resolvable_function_refused_and_pointwise_spelling_serves_it() -> None:
    """The intent of ``arr.map_blocks(np.abs).sum()`` is served as ``abs(arr).sum()`` -- same plan, same value.

    The pointwise spelling plans one full fold of the derived operand, and its branch callable applies ``abs`` to each
    chunk, which is checked on real values here.
    """
    _, requests, _ = _run_callback("s = abs(arr).sum()\nreturn s.compute()")
    assert len(requests) == 1
    request = requests[0]
    assert (request.key, request.axes, request.kind, request.finalize) == ("f-sum-axisall", (), "scalar", None)
    assert request.input_name == "f"

    chunk = np.full((4, 4), -2.0)
    assert np.isclose(_scalar(request.branch_func(chunk)), abs(chunk).sum()), "the branch must fold the MAPPED chunk"


# ---------------------------------------------------------------------------------------------------------------------
# cross-reduction guard must be ordering-independent
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "body,expected",
    [
        pytest.param("s = (arr - arr.mean()).sum()\nreturn s.compute()", "simple", id="simple"),
        pytest.param("s = arr.mean() + (arr - arr.mean()).sum()\nreturn s.compute()", "added_mean", id="added_mean"),
        pytest.param("s = arr.sum() + (arr - arr.mean()).sum()\nreturn s.compute()", "bypass", id="bypass"),
    ],
)
def test_cross_reduction_is_refused(body, expected) -> None:
    """A reduction feeding another expression is refused, whatever the layer ordering.

    The refusal names the completed reduction and points at the working path (compute each reduction, combine the
    delivered VALUES). The ordering-dependent bypass is the regression this guard exists for: ``arr.sum()`` first in
    the source must not hide the second ``sum``'s dependence on ``mean``.
    """
    with pytest.raises(errors.UnsupportedReductionError) as refusal:
        _run_callback(body)
    message = str(refusal.value).lower()
    assert "completed reduction" in message, expected
    assert "compute each reduction separately" in message, "the remedy must be usable"


def test_reduction_value_cannot_be_reduced_again() -> None:
    """Role 2 (no double reduce): ``s = arr.sum()`` is a delivered value, so ``s.sum()``/``s + 1`` are refused.

    ``_PrecomputedDeisaArray`` had to intercept a second reduction to stop it re-applying the op to an already-final
    array. Here the composed reduction has no chunk-local partial at all, so the second reduction is structurally
    impossible rather than caught.
    """
    _, requests, boundary = _run_callback("s = arr.sum()\nreturn s.compute()")
    reduction = boundary.reductions[0]
    assert (reduction.op_name, reduction.axes) == ("sum", ())
    with pytest.raises(errors.UnsupportedReductionError):
        reduction.sum()
    with pytest.raises(errors.UnsupportedReductionError):
        _run_callback("s = arr.sum()\nt = s + 1\nreturn t.compute()")


def test_sibling_reductions_still_emitted() -> None:
    """The guard must not over-refuse: two independent reductions are both delivered."""
    _, requests, _ = _run_callback("s = arr.sum()\nm = arr.mean()\nreturn s.compute(), m.compute()")
    assert sorted(request.key for request in requests) == ["f-mean", "f-sum"]


# ---------------------------------------------------------------------------------------------------------------------
# unknown values are EXECUTED, not walked on both branches
# ---------------------------------------------------------------------------------------------------------------------
def test_branch_selection_plans_the_executed_arm() -> None:
    """A control-flow-dependent callback plans exactly the arm it EXECUTES, for every state.

    The plan is what the callback did, not what its source might do, so both arms are correct -- asserted by exercising
    each one.
    """
    body = "if state > 3:\n    result = arr.sum()\nelse:\n    result = arr.mean()\nresult.compute()"
    for state, expected_key in ((5, "f-sum"), (1, "f-mean")):
        boundary = RecordingDispatcher()
        _, requests, _ = _run_callback(
            body,
            params="arr, state",
            collections={"arr": _collection("f", dispatcher=boundary), "state": state},
            dispatcher=boundary,
        )
        assert [request.key for request in requests] == [expected_key], f"state={state}"


def test_unbound_param_subscript_of_a_collection_is_refused_not_crashing() -> None:
    """Subscripting a delivered collection is a refused item access, never a ``TypeError``.

    A plain list/array parameter subscripts normally (so the callback keeps working), while a registered array's
    collection refuses the access with a named cause instead of crashing.
    """
    boundary = RecordingDispatcher()
    _, requests, _ = _run_callback(
        "x = p[0]\ns = arr.sum()\nreturn s.compute(), x",
        params="arr, p",
        collections={"arr": _collection("f", dispatcher=boundary), "p": [7]},
        dispatcher=boundary,
    )
    assert [request.key for request in requests] == ["f-sum"]

    boundary = RecordingDispatcher()
    with pytest.raises(errors.UnsupportedReductionError) as refusal:
        _run_callback(
            "x = p[0]\ns = arr.sum()\nreturn s.compute(), x",
            params="arr, p",
            collections={
                "arr": _collection("a", meta=META_AB, dispatcher=boundary),
                "p": _collection("b", meta=META_AB, dispatcher=boundary),
            },
            dispatcher=boundary,
        )
    assert not isinstance(refusal.value, TypeError)
    assert "item access" in str(refusal.value)


# ---------------------------------------------------------------------------------------------------------------------
# partial metadata describes ONE chunk's partial, not the whole array
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "body,axes,expected_shape",
    [
        pytest.param("s = arr.sum(axis=0)\nreturn s.compute()", (0,), (1, 4), id="sum_axis0"),
        pytest.param("s = arr.mean(axis=1)\nreturn s.compute()", (1,), (4, 1), id="mean_axis1"),
        pytest.param("s = arr.sum()\nreturn s.compute()", (), (), id="full_sum"),
    ],
)
def test_partial_metadata_is_probed_from_the_runtime_plan(body, axes, expected_shape) -> None:
    """The shipped partial is the CHUNK's partial (8x8 array, 4x4 chunks), probed off the composed callable.

    The value travels in the topic event and drives the combine ``out_shape``, so it is pinned as a literal here: a
    drift between the callable and the hand-written shape would be a silent wire-format bug.
    """
    _, requests, _ = _run_callback(body)
    request = requests[0]
    shape, dtype = probe_partial(request, (4, 4), np.dtype(np.float64))
    assert request.axes == axes
    assert shape == expected_shape, f"runtime partial {shape}, expected {expected_shape}"
    assert dtype == np.dtype(np.float64)


# ---------------------------------------------------------------------------------------------------------------------
# multi-array callbacks: per-array attribution and distinct keys
# ---------------------------------------------------------------------------------------------------------------------
def test_multiarray_one_reduction_per_array_with_distinct_keys() -> None:
    """Two arrays give two reductions, each attributed to its OWN root, with distinct keys.

    ``arr_a.sum()``/``arr_b.sum()`` must produce ``a-sum`` rooted in ``a`` and ``b-sum`` rooted in ``b``. The attempt is
    keyed per (root, op, axes), so the attribution is what keeps the two partials from being merged together.
    """
    _, requests, _ = _two_arrays("sa = arr_a.sum()\nsb = arr_b.sum()\nreturn sa.compute(), sb.compute()")
    by_key = {request.key: request for request in requests}
    assert sorted(by_key) == ["a-sum", "b-sum"]
    assert by_key["a-sum"].input_name == "a"
    assert by_key["b-sum"].input_name == "b"


def test_multiarray_branch_func_computes_own_reduction() -> None:
    """Each reduction's ``branch_func`` folds its OWN array's chunk, verified on real numbers.

    Regression guard for per-array attribution: before it, both branches folded the same array's data.
    """
    _, requests, _ = _two_arrays("sa = arr_a.sum()\nsb = arr_b.sum()\nreturn sa.compute(), sb.compute()")
    by_key = {request.key: request for request in requests}
    chunk_a = np.arange(16.0).reshape(4, 4)
    chunk_b = chunk_a + 16.0
    assert np.isclose(_scalar(by_key["a-sum"].branch_func(chunk_a)), float(chunk_a.sum()))
    assert np.isclose(_scalar(by_key["b-sum"].branch_func(chunk_b)), float(chunk_b.sum()))


# ---------------------------------------------------------------------------------------------------------------------
# raw-field reads: the refusal lands where the data would be gathered
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("callee", ["plot", "custom"])
def test_raw_field_read_via_opaque_call_refused_at_consumption(callee) -> None:
    """``plot(arr)``/``custom(arr)``: handing the operand over is free, CONSUMING it is refused -- role 3.

    The runtime hands over a lazy operand and refuses at the point where the whole field would have been gathered, with
    ``MaterializationError``. The data never reaches the host, and the refusal names the remedy.
    """

    def consumer(value):
        return np.asarray(value)

    with pytest.raises(errors.MaterializationError) as refusal:
        _run_callback(f"{callee}(arr)\ns = arr.sum()\nreturn s.compute()", extra_globals={callee: consumer})
    assert "materializes the whole registered field" in str(refusal.value)
    assert "precompute=False" in str(refusal.value)


def test_resolvable_module_call_on_raw_data_is_refused() -> None:
    """``np.linalg.norm(collection)`` is refused: on the analytics host there is no chunk to consume.

    Letting numpy coerce the operand would build a graph that materialized the WHOLE field on the analytics host with
    nothing to refuse it, so the coercion itself is what the surface refuses.
    """
    norm_body = "import numpy as npy\nv = npy.linalg.norm(arr)\ns = arr.sum()\nreturn s.compute(), v"
    with pytest.raises(errors.MaterializationError):
        _run_callback(norm_body)


def test_raw_field_read_list_unwrapping() -> None:
    """A raw collection nested in a list is harmless to build and refused when the callee consumes it -- role 3.

    The list holds the lazy operand, so no static unwrapping is needed: the gather is refused at the moment
    ``np.asarray`` reaches it.
    """

    def plot(values):
        return np.asarray(values[0])

    _, requests, _ = _run_callback(
        "plot([arr])\ns = arr.sum()\nreturn s.compute()", extra_globals={"plot": lambda values: values}
    )
    assert [request.key for request in requests] == ["f-sum"], "passing the list alone is not a read"
    with pytest.raises(errors.MaterializationError):
        _run_callback("plot([arr])\ns = arr.sum()\nreturn s.compute()", extra_globals={"plot": plot})


def test_raw_field_compute_boundary_without_reduction_refused() -> None:
    """``arr.compute()`` is a raw read even when the callback also reduces.

    The refusal lands at the compute boundary, names the array and the timestep, and points at ``precompute=False``.
    """
    body = "v = arr.compute()\ns = arr.sum()\nreturn s.compute(), v"
    with pytest.raises(errors.RawFieldReadError) as refusal:
        _run_callback(body)
    assert "raw data of array 'f'" in str(refusal.value)
    assert "precompute=False" in str(refusal.value)


def test_raw_field_index_read_refused() -> None:
    """``arr[0, 0]`` read alongside a reduction is refused at the item access -- role 3.

    Indexing is not on the closed operator surface, so the item access is refused and the data is never served.
    """
    with pytest.raises(errors.UnsupportedReductionError) as refusal:
        _run_callback("v = arr[0, 0]\ns = arr.sum()\nreturn s.compute() + v.compute()")
    assert "raw data" in str(refusal.value)


def test_predicate_builtins_on_raw_data_allowed() -> None:
    """``isinstance(collection, ...)`` / ``hasattr(collection, "shape")`` do NOT break the contract.

    The gysela MPI callback registers under precompute with ``assert isinstance(arr, DeisaArray)``, so predicate
    builtins must keep working: they inspect type and declared metadata, never data.
    """
    _, requests, _ = _run_callback(
        "assert isinstance(arr, object)\nassert hasattr(arr, 'shape')\ns = arr.sum()\nreturn s.compute()"
    )
    assert [request.key for request in requests] == ["f-sum"]


def test_hasattr_on_an_off_surface_attribute_raises() -> None:
    """``hasattr(arr, "graph")`` RAISES instead of returning ``False``, pinned so it is not discovered in a run.

    ``__getattr__`` turns an off-surface attribute into a *refusal*, not an ``AttributeError``, so a probing callback
    cannot silently conclude "no such attribute". Underscore-prefixed names are exempt (protocol probing), which is why
    ``hasattr(collection, "__dask_graph__")`` still works.
    """
    with pytest.raises(errors.UnsupportedReductionError) as refusal:
        _run_callback("h = hasattr(arr, 'graph')\ns = arr.sum()\nreturn s.compute(), h")
    assert ".graph" in str(refusal.value)
    _, requests, _ = _run_callback("h = hasattr(arr, '__dask_graph__')\ns = arr.sum()\nreturn s.compute(), h")
    assert [request.key for request in requests] == ["f-sum"]


def test_len_on_raw_data_is_refused() -> None:
    """``len(collection)`` is refused: ``__len__`` is the protocol every raw-data consumer reaches for.

    The shape IS exposed as metadata, so a callback that sizes the field must read ``arr.shape`` instead.
    """
    with pytest.raises(errors.UnsupportedReductionError) as refusal:
        _run_callback("n = len(arr)\ns = arr.sum()\nreturn s.compute(), n")
    assert "len()" in str(refusal.value)
    _, requests, _ = _run_callback("n = arr.shape[0]\ns = arr.sum()\nreturn s.compute(), n")
    assert [request.key for request in requests] == ["f-sum"], "the metadata path must stay open"


def test_attribute_reads_on_raw_data_allowed() -> None:
    """Attribute (metadata) reads -- ``arr.ndim``, an f-string in a log -- do not taint the callback."""
    result, requests, _ = _run_callback("y = arr.ndim\ns = arr.sum()\nlog = f'{y} dims'\nreturn s.compute(), log")
    assert [request.key for request in requests] == ["f-sum"]
    assert result[1] == "2 dims"
