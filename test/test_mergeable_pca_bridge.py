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
End-to-end tests for the BRIDGE-SIDE MergeablePCA path on RUNTIME DELIVERY: registration, the dispatched PCA branch,
and the delivered fit.

This file exists because two execution blockers once shipped unnoticed: the analyzer None-filled every constructor
parameter the caller did not write (the estimator rejects a ``None`` ``whiten``, so ``MergeablePCA(n_components=3)`` was
refused at registration), and the delivered fit cross-checked the summary's feature count against the PCA-only
CARRIER's ``shape[-1]``, which is 1 for every registered array, so every legitimate fit with ``d != 1`` was refused.

Both blockers were pinned through the analyzer and through ``make_precomputed_view``. Neither surface exists any more
(plan P4 deletes the discovery machinery and the legacy precomputed view), so the contracts were PORTED to what replaced
them rather than dropped -- the four below are the acceptance, and each still fails on the unfixed tree:

1. ``MergeablePCA(n_components=3)`` on a registered array is ACCEPTED end to end (``Deisa._register_callback_impl``).
   Registration analyzes nothing now, so acceptance is pinned by what registration actually produces -- the callback
   payload plus the array filed as a runtime-delivery target -- and by the bridge policy the estimator dispatches,
   which carries no root-only field and therefore cannot invent an omitted parameter. That is blocker 1's fix
   expressed structurally.
2. A fit on a delivered collection completes through ``MergeablePCA._fit_runtime_collection`` -> the composed PCA
   request -> per-bridge leaves -> the merge tree -> ``_fit_delivered``, and yields the requested number of components
   with the right ``n_features_in_``, against a numpy batch-PCA reference.
3. An EXPLICITLY invalid configuration is still refused -- by the estimator's own constructor, which is the only
   validator left. Registration accepts it, because with no AST pass a callback that builds an invalid estimator fails
   at its first compute boundary; the test pins that shift so it cannot come back silently as a registration-time pass.
4. A non-trivial ``d >= 2`` works -- the exact ``shape[-1] == 1`` case blocker 2 refused -- and a summary over a
   DIFFERENT feature basis is still refused, so the fix is not \"delete the check\".

Everything asserted here is real behaviour: the branch's ``branch_func`` runs on real numpy chunks, the merge tree
really executes, and the numbers are compared to a numpy reference. Nothing is a mock -- the only stand-in is the
dispatcher, which replaces the QUEUE ROUND TRIP and runs the very same composed request over the very same chunks.
"""

from __future__ import annotations

import textwrap
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import pytest
from deisa.core.types import Window
from utils import _make_callback

from deisa.dask.collection import DeisaCollection
from deisa.dask.deisa import Deisa
from deisa.dask.mergeable_pca import MergeablePCA, PCASummary, local_pca, merge_tree
from deisa.dask.runtime_compose import compose_pca_request

# The registered array every test here works against: 12 samples over 6 features, split into 4 chunks of 3 rows.
# ``d = 6`` is deliberately greater than 1, which is the case the carrier's ``shape[-1] == 1`` refused.
GLOBAL_SHAPE = (12, 6)
CHUNK_SHAPE = (3, 6)
META = {"f": {"global_shape": GLOBAL_SHAPE, "chunk_shape": CHUNK_SHAPE}}


def _field(n_rows: int = GLOBAL_SHAPE[0], n_cols: int = GLOBAL_SHAPE[1], seed: int = 7) -> np.ndarray:
    """The field the bridge would hold, in the same row/feature order the registered metadata declares."""
    return np.random.RandomState(seed).randn(n_rows, n_cols)


def _batch_reference(X: np.ndarray) -> PCASummary:
    """Exact batch PCA of the whole array, computed directly: center, then one SVD. numpy only."""
    Xc = np.asarray(X, dtype=np.float64)
    mean = Xc.mean(axis=0)
    _, singular_values, components = np.linalg.svd(Xc - mean, full_matrices=False)
    return PCASummary(
        n_samples=int(Xc.shape[0]),
        mean=mean,
        components=components,
        singular_values=singular_values,
    )


# ------------------------------------------------------------------------------ the bridge side, without the queues
# -------------------------------------------------------------------------------------------------------
class _FieldDispatcher:
    """The queue round trip, replaced by the work it does: run the composed request per chunk, merge the leaves.

    ``PlanDispatcher.compute_pca`` puts the request on the plan queues and merges what the bridges report back. The
    ROUND TRIP is all this stands in for: the request, the per-bridge leaf and the merge tree are the real ones, so a
    fit driven through here covers the same code path the runtime delivery takes.
    """

    def __init__(self, field: np.ndarray, chunk_shape: Sequence[int], summary: Optional[PCASummary] = None) -> None:
        self.field = field
        self.chunk_shape = tuple(int(size) for size in chunk_shape)
        self.summary = summary
        self.requests: list = []

    def compute_pca(self, request) -> PCASummary:
        self.requests.append(request)
        if self.summary is not None:
            return self.summary
        rows = self.chunk_shape[0]
        leaves = [request.branch_func(self.field[start : start + rows]) for start in range(0, len(self.field), rows)]
        return merge_tree(leaves)


def _collection(
    shape: Sequence[int] = GLOBAL_SHAPE,
    chunk_shape: Sequence[int] = CHUNK_SHAPE,
    dispatcher: Any = None,
    name: str = "f",
    timestep: int = 0,
) -> DeisaCollection:
    """The collection the topic handler delivers for one array and timestep."""
    return DeisaCollection(name, timestep, shape, chunk_shape=chunk_shape, dispatcher=dispatcher)


# ------------------------------------------------------------------------------ a Deisa stub that accepts registrations
# -------------------------------------------------------------------------------------------------------
class _FakeClient:
    """The only client surface registration touches: topic subscription and close."""

    def __init__(self) -> None:
        self.subscribed: list[str] = []

    def subscribe_topic(self, name, handler) -> None:
        self.subscribed.append(name)

    def close(self) -> None:
        pass


class _FakeHandshake:
    """The handshake surface registration touches: filing the per-array delivery state."""

    def __init__(self) -> None:
        self.branches: Dict[str, Any] = {}
        self.runtime_arrays: list = []
        self.legacy_arrays: list = []

    def set_task_branches(self, array_name: str, hints: list) -> None:
        self.branches[array_name] = hints

    def set_runtime_arrays(self, array_names: list) -> None:
        self.runtime_arrays = list(array_names)

    def set_legacy_arrays(self, array_names: list) -> None:
        self.legacy_arrays = list(array_names)


def _deisa_stub() -> Deisa:
    """A ``Deisa`` with the registration surfaces stubbed, so registration runs for real without a cluster."""
    d = Deisa.__new__(Deisa)
    d.client = _FakeClient()
    d.handshake = _FakeHandshake()
    d.arrays_metadata = META
    d._callbacks = {}
    d._callbacks_by_array = {}
    d._runtime_arrays = set()
    d._legacy_arrays = set()
    d._topic_handlers = {}
    d._callback_reductions = {}
    d._callback_seq = 0
    d._branch_groups = {}
    d._tasks = set()
    d._execute_callbacks_called = False
    return d


def _pca_callback(constructor: str = "MergeablePCA(n_components=3)") -> Callable:
    """A callback that fits the registered array and returns its singular values, compiled from real source.

    ``constructor`` is spliced into the body so each test can vary the estimator configuration without hand-writing
    and re-lexing a snippet.

    - ``:param constructor:`` The ``MergeablePCA(...)`` expression to splice in.
    """
    body = textwrap.dedent(
        f"""
        pca = {constructor}
        fitted = pca.fit(arr)
        return fitted.singular_values_
        """
    )
    return _make_callback("pca_callback", body, params="arr")


# ------------------------------------------------------------------------------ 1. registration is accepted end to end
# -------------------------------------------------------------------------------------------------------
def test_registration_accepts_the_documented_motivating_example():
    """``MergeablePCA(n_components=3).fit(<registered array>)`` registers.

    The motivating example of the feature, and the case that was refused: the analyzer used to None-fill ``whiten``,
    the estimator rejects a ``None`` ``whiten``, so EVERY PCA callback died at registration. Asserted through the real
    registration entry point, not through the estimator, because registration is where users hit it.

    Registration analyzes nothing any more (plan P4), so acceptance is pinned by what it produces: the callback payload
    carrying the delivery mode, and the array filed as a runtime-delivery target with the handshake actor. Nothing is
    filed as branches -- nothing files branches any more.
    """
    d = _deisa_stub()
    cid = d._register_callback_impl(
        _pca_callback(), [Window("f", size=1)], exception_handler=None, when="AND", precompute=True
    )

    assert cid in d._callbacks, "the callback must be registered, not refused"
    assert d._callbacks[cid]["precompute"] is True, "the payload must carry the delivery mode selected per callback"
    assert d._runtime_arrays == {"f"}, f"the array must be filed as a runtime-delivery target, got {d._runtime_arrays}"
    assert d.handshake.runtime_arrays == ["f"], "registration files the runtime arrays with the handshake actor"
    assert d._branch_groups == {}, "no analyzer runs, so no branch spec may be filed at registration"


def test_the_bridge_policy_carries_only_the_leaf_relevant_fields():
    """Blocker 1's fix, structurally: nothing re-invents a parameter the caller did not write.

    The estimator dispatches the bridge-side axis policy and NOTHING else. ``n_components`` and ``whiten`` are
    root-only, so they cannot reach the leaf even when the caller writes them, and an omitted parameter therefore
    behaves exactly as it does on the standalone path (the estimator's own default). That is what makes the documented
    motivating example work now that the analyzer -- which fabricated a ``whiten=None`` -- is gone.
    """
    policy = MergeablePCA(n_components=3)._bridge_policy()

    assert set(policy) == {"local_rank", "axis_names", "feature_axes", "sample_axes"}, (
        f"the bridge policy must carry only leaf-relevant fields, got {sorted(policy)}"
    )
    assert "n_components" not in policy and "whiten" not in policy, "root-only parameters must never be dispatched"
    assert policy["axis_names"] is None, "an omitted axis policy stays omitted instead of being filled in"


def test_registration_records_the_axis_policy_the_bridge_leaf_needs():
    """The axis policy reaches the bridge leaf as concrete values, and the leaf summarizes under it."""
    policy = MergeablePCA(n_components=3, axis_names=("tor1", "tor2"))._bridge_policy()
    assert policy["axis_names"] == ("tor1", "tor2"), f"the written axis policy must survive, got {policy['axis_names']}"

    request = compose_pca_request(_collection(), policy)
    assert request.key == "f-pca", f"the wire key must not fork from the analyzer-era name, got {request.key!r}"
    assert request.input_name == "f"

    # The bridge branch runs the local decomposition under that policy: two axes named, the last one the feature axis.
    chunk = np.arange(6, dtype=np.float64).reshape(3, 2)
    summary = request.branch_func(chunk)
    assert isinstance(summary, PCASummary), f"a pca branch must ship a summary, got {type(summary).__name__}"
    assert summary.mean.shape == (2,), f"the feature axis is tor2 (2 cells), got mean shape {summary.mean.shape}"


# ------------------------------------------------------------------------------ 2. the delivered fit completes
# -------------------------------------------------------------------------------------------------------
def test_delivered_fit_yields_the_requested_components_for_a_non_trivial_d():
    """A fit on a delivered collection completes and returns ``n_components`` components over the real ``d``.

    This is the blocker-2 case end to end: the deleted carrier had ``shape[-1] == 1``, while the delivered collection
    declares the registered shape, so the feature-dimension guard refused every such fit. The declared count now comes
    from the registered metadata, so ``d = 6`` fits and reports ``n_features_in_ == 6``.

    The numbers are checked against a batch PCA of the whole array: per-bridge leaves are merged exactly, so the
    truncated singular values must match the reference's leading ones.
    """
    field = _field()
    dispatcher = _FieldDispatcher(field, CHUNK_SHAPE)

    fitted = MergeablePCA(n_components=3).fit(_collection(dispatcher=dispatcher))

    assert len(dispatcher.requests) == 1, "the fit must dispatch exactly one PCA request to the bridges"
    request = dispatcher.requests[0]
    assert request.input_name == "f" and request.timestep == 0, "the request must name the delivered array/timestep"
    assert request.key == "f-pca"

    assert fitted.n_components_ == 3, f"requested 3 components, got n_components_={fitted.n_components_}"
    assert fitted.components_.shape == (3, GLOBAL_SHAPE[1]), (
        f"components must be (n_components, d={GLOBAL_SHAPE[1]}), got {fitted.components_.shape}"
    )
    assert fitted.n_features_in_ == GLOBAL_SHAPE[1], (
        f"the real feature count must be reported, got n_features_in_={fitted.n_features_in_}"
    )
    reference = _batch_reference(field)
    assert np.max(np.abs(fitted.singular_values_ - reference.singular_values[:3])) < 1e-10, (
        "the merged per-bridge summaries must reproduce the batch PCA's leading singular values"
    )
    # n_components_ and components_ can never disagree: a caller sizing buffers off n_components_ gets exactly that.
    assert fitted.n_components_ == fitted.components_.shape[0]


def test_delivered_fit_ships_summaries_not_the_truncated_public_form():
    """The bridge payload stays the MERGEABLE summary, which is what makes the merge exact.

    Truncating at the bridge to ``n_components`` would drop the directions the cross-bridge correction needs, so this
    asserts the shape property that catches it: a leaf over a WIDE short block keeps its full local rank ``min(n, d)``,
    never ``n_components``.
    """
    request = compose_pca_request(_collection(), MergeablePCA(n_components=2)._bridge_policy())

    # 4 samples over d = 6: the leaf's rank is 4 > n_components = 2, so a truncated payload would be visibly smaller.
    wide = _field(n_rows=4, n_cols=GLOBAL_SHAPE[1], seed=11)
    leaf = request.branch_func(wide)

    assert leaf.rank == 4, f"the leaf must keep its full local rank min(n_block, d) = 4, got {leaf.rank}"
    assert leaf.components.shape == (4, GLOBAL_SHAPE[1]), (
        f"the leaf summary must be (rank, d), not truncated to n_components=2, got {leaf.components.shape}"
    )
    # And the payload is the summary type the estimator's delivered path requires.
    assert local_pca(wide).rank == leaf.rank, "the branch must use the same leaf primitive the estimator does"


# ------------------------------------------------------------------------------ 3. explicit invalid is still refused
# -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs, constructor, expected_fragment",
    [
        ({"n_components": 3, "whiten": 1}, "MergeablePCA(n_components=3, whiten=1)", "whiten must be a bool"),
        ({"n_components": 3, "whiten": None}, "MergeablePCA(n_components=3, whiten=None)", "whiten must be a bool"),
        ({"n_components": 3, "whiten": "yes"}, "MergeablePCA(n_components=3, whiten='yes')", "whiten must be a bool"),
        ({"n_components": 0}, "MergeablePCA(n_components=0)", "n_components must be None or a positive integer"),
        ({"n_components": -2}, "MergeablePCA(n_components=-2)", "n_components must be None or a positive integer"),
    ],
)
def test_explicit_invalid_configuration_is_refused_by_the_estimator(kwargs, constructor, expected_fragment):
    """Blocker 1's fix omitted UNSET parameters; it must not have made an EXPLICIT invalid value acceptable.

    The distinction is the whole point: omitting is how "use the default" is spelled, so an explicit value still goes
    through the estimator's own validation and is still refused. ``whiten=None`` is the sharpest case, because it is
    exactly what the analyzer used to inject -- refusing it here is what proves the injection is gone rather than
    merely tolerated.

    What CHANGED is WHERE. Registration parses nothing now, so an invalid constructor registers and fails at the
    callback's first compute boundary instead of at registration. The second half pins that shift.
    """
    with pytest.raises(ValueError, match=expected_fragment):
        MergeablePCA(**kwargs)

    d = _deisa_stub()
    cid = d._register_callback_impl(
        _pca_callback(constructor), [Window("f", size=1)], exception_handler=None, when="AND", precompute=True
    )
    assert cid in d._callbacks, "with no AST pass nothing is left to refuse an invalid constructor at registration"
    assert d._runtime_arrays == {"f"}


def test_constructor_rejects_non_bool_whiten_outside_the_bridge_path():
    """The estimator's own guard is untouched by the analyzer-side fix, checked directly.

    The analyzer fix is the omission of unset parameters, NOT a relaxation of the constructor. If someone later "fixes"
    registration by letting the constructor accept ``None`` for ``whiten``, this test is what fails.
    """
    for bad in (1, None, "yes", 1.5):
        with pytest.raises(ValueError, match="whiten must be a bool"):
            MergeablePCA(n_components=3, whiten=bad)


# ------------------------------------------------------------------------------ 4. d >= 2 is the case that was refused
# -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n_features", [2, 3, 6, 11])
def test_declared_feature_count_multiplies_the_feature_axes(n_features):
    """The declared ``d`` is read from the DELIVERED COLLECTION's registered shape, so any ``d >= 2`` works.

    The delivered path's guard used to compare the summary's ``d`` against the carrier's ``shape[-1]``, which is 1 for
    every registered array, so no ``d`` other than 1 could ever match. These are exactly the ``d`` values that were
    refused before; each must now resolve to its true feature count and cross-check cleanly.

    24 samples rather than the module's 12, so every ``d`` here also clears the estimator's own
    summary-smaller-than-input guard (``n_samples > d``). That guard is a separate contract, covered by
    ``test_mergeable_pca_estimator.py``; raising the row count here keeps this test about the DECLARED count alone.
    """
    n_rows = 24
    shape = (n_rows, n_features)
    chunk_shape = (3, n_features)
    field = _field(n_rows=n_rows, n_cols=n_features, seed=13)

    collection = _collection(shape=shape, chunk_shape=chunk_shape, dispatcher=_FieldDispatcher(field, chunk_shape))
    assert tuple(collection.shape) == shape, "the collection carries the REGISTERED shape, which is what declares d"

    fitted = MergeablePCA(n_components=2).fit(collection)
    assert fitted.n_features_in_ == n_features, f"expected d={n_features}, got n_features_in_={fitted.n_features_in_}"
    reference = _batch_reference(field)
    assert np.max(np.abs(fitted.singular_values_ - reference.singular_values[:2])) < 1e-10


def test_a_real_feature_basis_mismatch_is_still_refused():
    """Moving the declared count off the carrier must not make the guard useless.

    The guard exists to catch a summary describing a DIFFERENT feature basis than the array it was requested for. With
    the declared count taken from the delivered collection's registered shape, that mismatch is still detected -- this
    is the test that keeps the blocker-2 fix from being "delete the check".
    """
    field = _field(n_cols=6, seed=17)
    # A summary over a 4-feature basis, delivered for a 6-feature registered array.
    wrong = local_pca(field[:, :4])
    dispatcher = _FieldDispatcher(field, CHUNK_SHAPE, summary=wrong)

    with pytest.raises(ValueError) as caught:
        MergeablePCA(n_components=2).fit(_collection(dispatcher=dispatcher))

    message = str(caught.value)
    assert "spans 4 features" in message, f"the message must name the summary's feature count, got {message!r}"
    assert "has 6" in message, f"the message must name the declared feature count, got {message!r}"
    assert "rechunk" in message, "the message must name the remedy"


def test_multidim_registered_shape_declares_the_product_of_its_feature_axes():
    """A multi-axis feature dimension multiplies out through the same policy ``_as_2d`` applies.

    A 3-D registered array with ``axis_names`` and the default last-axis feature policy declares ``d`` = that axis's
    extent. Using the delivered collection's registered shape rather than a carrier's own shape is what makes this
    agree with the bridge, which summarized under the same policy.
    """
    shape = (4, 5, 6)
    chunk_shape = (2, 5, 6)
    estimator = MergeablePCA(n_components=2, axis_names=("tor1", "tor2", "vpar"))
    assert estimator._declared_feature_count(shape) == 6, "the default feature axis is the last one"

    field = np.random.RandomState(19).randn(*shape)
    dispatcher = _FieldDispatcher(field, chunk_shape)
    fitted = estimator.fit(_collection(shape=shape, chunk_shape=chunk_shape, dispatcher=dispatcher))

    assert fitted.n_features_in_ == 6, f"the last axis is the feature axis, got d={fitted.n_features_in_}"
