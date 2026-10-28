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
"""Regression tests for the reduction planner -- the oracle port of the analyzer regressions file.

This file used to pin the AST precompute analyzer's edge cases: how it treated ``map_blocks``, its cross-reduction
guard, its ``_Missing`` degradation, its per-array attribution, and the raw-field reads it refused at registration.
Those are exactly the behaviors K6b deletes, so they are re-pinned here against the runtime planner: the callback RUNS
against a lazily-reducible :class:`~deisa.dask.collection.DeisaCollection` and every reduction it computes composes its
own wire request. Where the analyzer is still the oracle, its ``BranchSpec`` is compared field by field; where the
runtime deliberately does otherwise, the divergence is pinned with both behaviors side by side.

``_PrecomputedDeisaArray`` role coverage (plan §6, for the deletion card). The retired dispatch view had four jobs;
the two this file proves by construction are:

- **no double reduce** (role 2) -- a reduction is a VALUE, so nothing can re-apply an op to it:
  ``test_cross_reduction_is_refused`` (``s.sum()`` / ``s + 1`` are refused structurally).
- **second-line defense** (role 3) -- an unplanned consumption of the delivered operand cannot reach data:
  ``test_raw_field_read_via_opaque_call_refused_at_consumption``, ``test_raw_field_read_list_unwrapping``,
  ``test_resolvable_module_call_on_raw_data_is_refused``, ``test_raw_field_compute_boundary_without_reduction_refused``,
  ``test_raw_field_index_read_refused``, ``test_len_on_raw_data_is_refused`` and
  ``test_hasattr_on_an_off_surface_attribute_raises``.

(The other two roles -- final mean/var delivery and per-callback isolation -- are covered by the ported
``test_precompute.py``.)

Divergences from the analyzer, each pinned by a test below (nothing here is a silent approximation):

- **D1 ``map_blocks``**: the analyzer served ``arr.map_blocks(np.abs).sum()`` and refused the opaque-lambda form with
  ``NoPrecomputableReductionError``. The runtime refuses BOTH as ``UnsupportedReductionError`` -- ``.map_blocks`` is
  not on the closed surface. The intent is served by the pointwise spelling ``abs(arr).sum()``, which is proven to be
  an equivalent spec AND an equivalent value (``test_map_blocks_resolvable_function_refused_and_pointwise_
  spelling_serves_it``). Closed-surface refusal, not a loss of capability.
- **D2 branch selection**: the analyzer had no static answer for ``if state > 3:``. It OVER-planned when the callback
  computes both arms at the end (``f-sum`` and ``f-mean`` for one callback) and UNDER-planned when the result depends on
  the branch -- there it named ``f-mean`` only, i.e. the WRONG arm whenever ``state > 3``. The runtime executes the
  callback, so the plan is exactly the executed arm
  (``test_branch_selection_is_real_not_walked``).
- **D3 opaque callee on raw data**: the analyzer refused ``plot(arr)`` at registration with ``RawFieldReadError``,
  naming the callee and the line. The runtime hands the lazy operand over and refuses where the data would actually be
  gathered -- inside the callee -- with ``MaterializationError``. Same end state, no callee name or line number.
- **D4 resolvable module call on raw data**: the analyzer ALLOWED ``np.linalg.norm(arr)`` (it reasoned about a
  bridge-local chunk). The runtime refuses it with ``MaterializationError``: on the analytics host there is no chunk to
  consume. This closes an unsound hole of the analyzer-era delivery, so the refusal is deliberate and correct.
- **D5 compute boundary**: ``arr.compute()`` is ``RawFieldReadError`` on both paths (one ``isinstance`` graph); the
  runtime message carries the array and timestep instead of a source line.
- **D6 index read**: the analyzer raised ``RawFieldReadError`` for ``arr[0, 0]``; the runtime refuses the item access
  as off-surface (``UnsupportedReductionError``). Same end state (never served); the class is the registration-time
  class of both paths, as pinned in the ported ``test_precompute.py``.
- **D7 ``hasattr`` on an off-surface attribute**: ``__getattr__`` raises a refusal instead of ``AttributeError``, so
  ``hasattr(arr, "graph")`` RAISES rather than returning ``False``. Loud instead of wrong; a callback doing
  ``hasattr`` on a name outside the surface must switch to the surface it actually uses.
- **D8 ``len(arr)``**: the analyzer allowed it as a metadata read. The runtime refuses ``len()`` off-surface, because
  ``__len__`` is the protocol every raw-data consumer reaches for. Escalation candidate: a callback that reads
  ``len(arr)`` (instead of ``arr.shape``) now has to be rewritten.
"""

import numpy as np
import pytest
from utils import RecordingDispatcher, _analyze, _collection, _run_callback, _scalar

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
# map_blocks: off-surface now; the pointwise spelling carries the same reduction
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "body,analyzer_served",
    [
        pytest.param("y = arr.map_blocks(lambda b: b * 2)\ns = y.sum()\nreturn s.compute()", False, id="opaque_lambda"),
        pytest.param("y = arr.map_blocks(np.abs)\ns = y.sum()\nreturn s.compute()", True, id="resolvable_func"),
    ],
)
def test_map_blocks_is_off_surface(body, analyzer_served) -> None:
    """``arr.map_blocks(...)`` is refused, opaque or resolvable -- D1.

    The analyzer refused the opaque form (``NoPrecomputableReductionError``: it could not see inside the lambda) and
    SERVED the resolvable one by folding ``np.abs`` into the branch. The runtime never inspects a callable, so both
    forms leave the closed surface and are refused there -- the same gate, named precisely.
    """
    from deisa.dask import precompute_analyzer as analyzer_module

    if analyzer_served:
        assert [spec.output_key for spec in _analyze(body)] == ["f-sum"], "the analyzer served the resolvable form"
    else:
        with pytest.raises(analyzer_module.PrecomputeError):
            _analyze(body)
    with pytest.raises(errors.UnsupportedReductionError) as refusal:
        _run_callback(body)
    assert ".map_blocks" in str(refusal.value)
    assert "closed DEISA operator surface" in str(refusal.value)
    assert "precompute=False" in str(refusal.value), "the remedy must name a working path"


def test_map_blocks_resolvable_function_refused_and_pointwise_spelling_serves_it() -> None:
    """The intent of ``arr.map_blocks(np.abs).sum()`` is served as ``abs(arr).sum()`` -- same spec, same value -- D1.

    This is the port of ``test_map_blocks_resolvable_function_uses_mapped_array``: the analyzer's answer was "sum of
    the mapped chunks". The pointwise spelling produces the analyzer's spec (op/axes/kind/finalize) for that intent and
    the runtime's branch callable folds ``abs`` on each chunk, which is checked on real values here.
    """
    oracle = _analyze("s = abs(arr).sum()\nreturn s.compute()")
    assert len(oracle) == 1
    _, requests, boundary = _run_callback("s = abs(arr).sum()\nreturn s.compute()")
    assert len(requests) == 1
    reduction, request = boundary.reductions[0], requests[0]
    assert (reduction.op_name, reduction.axes, reduction.kind) == (
        oracle[0].op_name,
        oracle[0].reduction_axes,
        oracle[0].output_kind,
    )
    assert request.input_name == oracle[0].input_name

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
    """A reduction feeding another expression is refused, whatever the layer ordering -- analyzer class, runtime class.

    All three shapes used to be refused by the analyzer's guard as ``UnsupportedReductionError``; the runtime refuses
    them at the operator, naming the completed reduction. The ordering-dependent bypass is the regression this guard
    exists for: ``arr.sum()`` first in the source must not hide the second ``sum``'s dependence on ``mean``.
    """
    from deisa.dask import precompute_analyzer as analyzer_module

    with pytest.raises(analyzer_module.UnsupportedReductionError):
        _analyze(body)
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
    """The guard must not over-refuse: two independent reductions are both delivered, as the analyzer emitted them."""
    oracle = _analyze("s = arr.sum()\nm = arr.mean()\nreturn s.compute(), m.compute()")
    assert sorted(spec.output_key for spec in oracle) == ["f-mean", "f-sum"]
    _, requests, _ = _run_callback("s = arr.sum()\nm = arr.mean()\nreturn s.compute(), m.compute()")
    assert sorted(request.key for request in requests) == ["f-mean", "f-sum"]


# ---------------------------------------------------------------------------------------------------------------------
# unknown values are EXECUTED, not walked on both branches
# ---------------------------------------------------------------------------------------------------------------------
def test_branch_selection_is_real_not_walked() -> None:
    """A control-flow-dependent callback plans exactly the arm it EXECUTES -- the analyzer walked both -- D2.

    The analyzer had no static answer for ``if state > 3:``. On the case where the callback computes both arms at the
    end it over-planned (``f-sum`` AND ``f-mean``); on the case where the result depends on the branch it named ONE arm
    and got it WRONG for ``state > 3`` (``f-mean``). The runtime executes the callback, so the plan is the executed arm,
    and both arms are correct -- asserted by exercising each.
    """
    both_arms = "if state > 3:\n    s = arr.sum()\nelse:\n    m = arr.mean()\nreturn s.compute(), m.compute()"
    assert sorted(spec.output_key for spec in _analyze(both_arms, params="arr, state")) == ["f-mean", "f-sum"]

    body = "if state > 3:\n    result = arr.sum()\nelse:\n    result = arr.mean()\nresult.compute()"
    assert [spec.output_key for spec in _analyze(body, params="arr, state")] == ["f-mean"], (
        "the analyzer's static answer names the else arm even when state > 3"
    )

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
    """Subscripting a delivered collection is a refused item access, never a ``TypeError`` -- the analyzer's downgrade.

    The analyzer modelled an unbound parameter as an opaque ``_Missing`` and degraded. The runtime has a real object:
    a plain list/array parameter subscripts normally (so the callback keeps working), while a registered array's
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
    """The shipped partial shape is the CHUNK's partial (8x8 array, 4x4 chunks), as the analyzer recorded it.

    The value travels in the topic event and drives the combine ``out_shape``, so it is asserted against the runtime
    plan (``probe_partial`` executes the branch on a placeholder chunk) AND against the analyzer's ``partial_shape``
    for the same source -- a drift between the two would be a silent wire-format bug.
    """
    oracle = _analyze(body)
    _, requests, _ = _run_callback(body)
    request = requests[0]
    shape, dtype = probe_partial(request, (4, 4), np.dtype(np.float64))
    assert request.axes == axes
    assert shape == expected_shape, f"runtime partial {shape}, expected {expected_shape}"
    assert shape == tuple(oracle[0].partial_shape), f"runtime {shape} vs analyzer {oracle[0].partial_shape}"
    assert dtype == np.dtype(oracle[0].partial_dtype)


# ---------------------------------------------------------------------------------------------------------------------
# multi-array callbacks: per-array attribution and distinct keys
# ---------------------------------------------------------------------------------------------------------------------
def test_multiarray_one_reduction_per_array_with_distinct_keys() -> None:
    """Two arrays give two reductions, each attributed to its OWN root, with distinct keys.

    ``arr_a.sum()``/``arr_b.sum()`` must produce ``a-sum`` rooted in ``a`` and ``b-sum`` rooted in ``b``. The attempt is
    keyed per (root, op, axes), so the attribution is what keeps the two partials from being merged together.
    """
    oracle = _analyze(
        "sa = arr_a.sum()\nsb = arr_b.sum()\nreturn sa.compute(), sb.compute()", params="arr_a, arr_b", meta=META_AB
    )
    assert sorted(spec.output_key for spec in oracle) == ["a-sum", "b-sum"]
    _, requests, _ = _two_arrays("sa = arr_a.sum()\nsb = arr_b.sum()\nreturn sa.compute(), sb.compute()")
    by_key = {request.key: request for request in requests}
    assert sorted(by_key) == ["a-sum", "b-sum"]
    assert by_key["a-sum"].input_name == "a"
    assert by_key["b-sum"].input_name == "b"


def test_multiarray_branch_func_computes_own_reduction() -> None:
    """Each reduction's ``branch_func`` folds its OWN array's chunk, verified on real numbers.

    Regression the analyzer needed: before per-array attribution both branches folded the same array's data.
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
    """``plot(arr)``/``custom(arr)``: handing the operand over is free, CONSUMING it is refused -- role 3, D3.

    The analyzer had to refuse at registration (``RawFieldReadError``, naming the callee and the line) because it
    could not see inside the callee. The runtime hands over a lazy operand and refuses at the point where the whole
    field would have been gathered, with ``MaterializationError``. Same end state: the data never reaches the host.
    """

    def consumer(value):
        return np.asarray(value)

    with pytest.raises(errors.MaterializationError) as refusal:
        _run_callback(f"{callee}(arr)\ns = arr.sum()\nreturn s.compute()", extra_globals={callee: consumer})
    assert "materializes the whole registered field" in str(refusal.value)
    assert "precompute=False" in str(refusal.value)


def test_resolvable_module_call_on_raw_data_is_refused() -> None:
    """``np.linalg.norm(collection)`` is refused, where the analyzer ALLOWED it -- D4.

    The analyzer reasoned about a bridge-local chunk and let the call through, so in the analyzer-era delivery the
    callback received a dask array and ``np.linalg.norm`` built a graph that materialized the WHOLE field on the
    analytics host with nothing to refuse it. The runtime hands over a lazy operand, so numpy's coercion is refused.
    This closes an unsound hole; it is not a capability the runtime should acquire.
    """

    norm_body = "import numpy as npy\nv = npy.linalg.norm(arr)\ns = arr.sum()\nreturn s.compute(), v"
    assert [spec.output_key for spec in _analyze(norm_body)] == ["f-sum"], "the analyzer allowed the raw module call"
    with pytest.raises(errors.MaterializationError):
        _run_callback(norm_body)


def test_raw_field_read_list_unwrapping() -> None:
    """A raw collection nested in a list is harmless to build and refused when the callee consumes it -- role 3.

    The analyzer had to unwrap ``plot([arr])`` statically to catch the nesting. The runtime does not need to: the list
    holds the lazy operand, and the gather is refused at the moment ``np.asarray`` reaches it.
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
    """``arr.compute()`` is a raw read even when the callback also reduces -- same class on both paths -- D5.

    The analyzer reported it per boundary with a source line; the runtime refuses at the boundary and names the array
    and the timestep instead. Both are ``RawFieldReadError`` in one ``isinstance`` graph.
    """
    from deisa.dask import precompute_analyzer as analyzer_module

    body = "v = arr.compute()\ns = arr.sum()\nreturn s.compute(), v"
    with pytest.raises(analyzer_module.RawFieldReadError):
        _analyze(body)
    with pytest.raises(errors.RawFieldReadError) as refusal:
        _run_callback(body)
    assert "raw data of array 'f'" in str(refusal.value)
    assert "precompute=False" in str(refusal.value)


def test_raw_field_index_read_refused() -> None:
    """``arr[0, 0]`` read alongside a reduction is refused at the item access -- role 3, D6.

    The analyzer called it ``RawFieldReadError``; the runtime refuses it as off-surface
    (``UnsupportedReductionError``) because indexing is not on the closed operator surface. The end state is the same
    -- never served -- and the class matches the registration-time class pinned in ``test_precompute.py``.
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
    """``hasattr(arr, "graph")`` RAISES instead of returning ``False`` -- D7, pinned so it is not discovered in a run.

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
    """``len(collection)`` is refused, where the analyzer allowed it as a metadata read -- D8 (escalation candidate).

    ``__len__`` is the protocol every raw-data consumer reaches for, so the closed surface refuses it rather than
    letting a callback size the field. A callback that reads ``len(arr)`` must use ``arr.shape`` -- the shape IS
    exposed as metadata.
    """

    assert [spec.output_key for spec in _analyze("n = len(arr)\ns = arr.sum()\nreturn s.compute(), n")] == ["f-sum"]
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
