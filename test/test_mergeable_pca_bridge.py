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
End-to-end tests for the BRIDGE-SIDE MergeablePCA path: registration, delivery, and the delivered fit.

This file exists because every stage of the bridge path was implemented and NONE of it was exercised. The full suite
was green with zero tests touching registration of a ``MergeablePCA`` callback, the per-callback dispatch view, or the
delivered fit, which is how two execution blockers shipped unnoticed:

- the analyzer None-filled every constructor parameter the caller did not write, and the estimator rejects a
  ``None`` ``whiten``, so ``MergeablePCA(n_components=3).fit(arr)`` -- the documented motivating example -- was refused
  at registration;
- the delivered fit cross-checked the summary's feature count against the PCA-only CARRIER's ``shape[-1]``, which is 1
  for every registered array, so every legitimate fit with ``d != 1`` was refused.

The four contracts below are the acceptance for those two fixes, and each was written to fail on the unfixed tree:

1. ``MergeablePCA(n_components=3)`` on a registered array is ACCEPTED end to end (registration through the real
   ``Deisa._register_callback_impl``).
2. A fit through the delivered path completes and yields the requested number of components, with the right
   ``n_features_in_``, against a numpy batch-PCA reference.
3. A non-bool ``whiten`` passed EXPLICITLY is still refused, so fix 1 did not weaken validation.
4. A non-trivial ``d >= 2`` works -- the exact ``shape[-1] == 1`` case blocker 2 refused.

Everything asserted here is real behaviour: the analyzer runs on real callback source, the branch's ``branch_func``
runs on real numpy chunks, the merge tree really executes, and the numbers are compared to a numpy reference. Nothing
is a mock.

Test 2 also proves the invariant the whole feature rests on: the raw chunk never reaches the callback side. The
per-bridge payload is the summary, and the fit's public attributes come out of that summary alone.
"""

from __future__ import annotations

import textwrap
from typing import Any, Callable, Dict

import numpy as np
import pytest
from deisa.core.types import Window
from utils import _make_callback

from deisa.dask.deisa import Deisa
from deisa.dask.mergeable_pca import MergeablePCA, PCASummary, local_pca, merge_pca, merge_tree
from deisa.dask.precompute_analyzer import IncompatibleCallbackError

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
    """The handshake surface registration touches: filing branches per array."""

    def __init__(self) -> None:
        self.branches: Dict[str, Any] = {}

    def set_task_branches(self, array_name: str, hints: list) -> None:
        self.branches[array_name] = hints


def _deisa_stub() -> Deisa:
    """A ``Deisa`` with the registration surfaces stubbed, so registration runs for real without a cluster."""
    d = Deisa.__new__(Deisa)
    d.client = _FakeClient()
    d.handshake = _FakeHandshake()
    d.arrays_metadata = META
    d._callbacks = {}
    d._callbacks_by_array = {}
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
    and re-lexing a snippet. The analyzer reads this source, so the snippet is the code under test, not a stand-in.

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
    registration entry point, not through the analyzer, because registration is where users hit this.
    """
    d = _deisa_stub()
    cid = d._register_callback_impl(
        _pca_callback(), [Window("f", size=1)], exception_handler=None, when="AND", precompute=True
    )

    assert cid in d._callbacks, "the callback must be registered, not refused"
    specs = d._branch_groups["f"]
    assert [b.output_key for b in specs] == ["f-pca"], f"expected one pca branch, got {[b.output_key for b in specs]}"
    # The recorded config carries ONLY what the caller wrote: an omitted parameter is absent, so the estimator's own
    # default applies when the config is replayed. This is the rule that makes registration work.
    assert specs[0].summary_config == {"n_components": 3}, (
        f"only the written parameters may be recorded, got {specs[0].summary_config}"
    )


def test_registration_records_the_config_the_bridge_and_root_both_need():
    """The recorded config carries the axis policy through to the bridge branch as concrete values."""
    d = _deisa_stub()
    callback = _pca_callback("MergeablePCA(n_components=3, axis_names=('tor1', 'tor2'))")
    d._register_callback_impl(callback, [Window("f", size=1)], exception_handler=None, when="AND", precompute=True)

    spec = d._branch_groups["f"][0]
    assert spec.summary_config == {"n_components": 3, "axis_names": ("tor1", "tor2")}
    # The bridge branch runs the local decomposition under that policy: two axes named, the last one the feature axis.
    chunk = np.arange(6, dtype=np.float64).reshape(3, 2)
    summary = spec.branch_func(chunk)
    assert isinstance(summary, PCASummary), f"a pca branch must ship a summary, got {type(summary).__name__}"
    assert summary.mean.shape == (2,), f"the feature axis is tor2 (2 cells), got mean shape {summary.mean.shape}"


# ------------------------------------------------------------------------------ 2. the delivered fit completes
# -------------------------------------------------------------------------------------------------------
def test_delivered_fit_yields_the_requested_components_for_a_non_trivial_d():
    """A fit through the delivered path completes and returns ``n_components`` components over the real ``d``.

    This is the blocker-2 case end to end: the delivered view's carrier has ``shape[-1] == 1`` while the summary spans
    ``d = 6``, so the feature-dimension guard refused every such fit. The declared count now comes from the
    REGISTERED array's metadata, so ``d = 6`` fits and reports ``n_features_in_ == 6``.

    The numbers are checked against a batch PCA of the whole array: per-bridge leaves are merged exactly, so the
    truncated singular values must match the reference's leading ones.
    """
    field = _field()
    d = _deisa_stub()
    d._register_callback_impl(
        _pca_callback("MergeablePCA(n_components=3)"),
        [Window("f", size=1)],
        exception_handler=None,
        when="AND",
        precompute=True,
    )
    spec = d._branch_groups["f"][0]

    # What the bridges actually do: each summarizes its own chunk, and the summaries merge on the Dask side.
    per_bridge = [
        spec.branch_func(field[start : start + CHUNK_SHAPE[0]]) for start in range(0, len(field), CHUNK_SHAPE[0])
    ]
    delivered = merge_pca(merge_pca(per_bridge[0], per_bridge[1]), merge_pca(per_bridge[2], per_bridge[3]))

    # The dispatch view a PCA-only callback receives, built exactly as the topic handler builds it.
    from deisa.dask.utils import make_precomputed_view

    view = make_precomputed_view(
        d._pca_carrier_array("f"),
        t=0,
        signatures={},
        reapply=set(),
        registered_ndim=len(GLOBAL_SHAPE),
        registered_shape=GLOBAL_SHAPE,
        pca_summary=delivered,
    )

    fitted = MergeablePCA(n_components=3).fit(view)

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
    asserts the shape property that catches it: a per-bridge leaf over 3 samples of ``d = 6`` keeps rank
    ``min(n_block, d) = 3`` here, and a leaf over a WIDE short block keeps the full ``d``, never ``n_components``.
    """
    d = _deisa_stub()
    d._register_callback_impl(
        _pca_callback("MergeablePCA(n_components=2)"),
        [Window("f", size=1)],
        exception_handler=None,
        when="AND",
        precompute=True,
    )
    spec = d._branch_groups["f"][0]

    # A wide, short block: 2 samples over d = 6, so the leaf's rank is min(2, 6) = 2 and n_components = 2 could not
    # distinguish truncation. Use 4 samples instead: min(4, 6) = 4 > n_components = 2, so a truncated payload would
    # be visibly smaller than the mergeable one.
    wide = _field(n_rows=4, n_cols=GLOBAL_SHAPE[1], seed=11)
    leaf = spec.branch_func(wide)

    assert leaf.rank == 4, f"the leaf must keep its full local rank min(n_block, d) = 4, got {leaf.rank}"
    assert leaf.components.shape == (4, GLOBAL_SHAPE[1]), (
        f"the leaf summary must be (rank, d), not truncated to n_components=2, got {leaf.components.shape}"
    )
    # And the payload is the summary type the estimator's delivered path requires.
    assert local_pca(wide).rank == leaf.rank, "the branch must use the same leaf primitive the estimator does"


# ------------------------------------------------------------------------------ 3. explicit invalid is refused
# -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "constructor, expected_fragment",
    [
        ("MergeablePCA(n_components=3, whiten=1)", "whiten must be a bool"),
        ("MergeablePCA(n_components=3, whiten=None)", "whiten must be a bool"),
        ("MergeablePCA(n_components=3, whiten='yes')", "whiten must be a bool"),
        ("MergeablePCA(n_components=0)", "n_components must be None or a positive integer"),
        ("MergeablePCA(n_components=-2)", "n_components must be None or a positive integer"),
    ],
)
def test_explicit_invalid_configuration_is_still_refused_at_registration(constructor, expected_fragment):
    """Blocker 1's fix omitted UNSET parameters; it must not have made an EXPLICIT invalid value acceptable.

    The distinction is the whole point: omitting is how "use the default" is spelled, so an explicit value still goes
    through the estimator's own validation and is still refused. ``whiten=None`` is the sharpest case, because it is
    exactly what the analyzer used to inject -- refusing it here is what proves the injection is gone rather than
    merely tolerated.
    """
    d = _deisa_stub()
    with pytest.raises(IncompatibleCallbackError, match=expected_fragment):
        d._register_callback_impl(
            _pca_callback(constructor), [Window("f", size=1)], exception_handler=None, when="AND", precompute=True
        )
    assert d._callbacks == {}, "a refused registration must leave no half-registered entry behind"
    assert d._branch_groups == {}, "a refused registration must leave no branch state behind"


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
    """The declared ``d`` is read from the REGISTERED shape under the axis policy, so any ``d >= 2`` works.

    The delivered path's guard used to compare the summary's ``d`` against the carrier's ``shape[-1]``, which is 1 for
    every registered array, so no ``d`` other than 1 could ever match. These are exactly the ``d`` values that were
    refused before; each must now resolve to its true feature count and cross-check cleanly.

    24 samples rather than the module's 12, so every ``d`` here also clears the estimator's own
    summary-smaller-than-input guard (``n_samples > d``). That guard is a separate contract, covered by
    ``test_mergeable_pca_estimator.py``; raising the row count here keeps this test about the DECLARED count alone.
    """
    n_rows = 24
    meta = {"f": {"global_shape": (n_rows, n_features), "chunk_shape": (3, n_features)}}
    d = _deisa_stub()
    d.arrays_metadata = meta
    d._register_callback_impl(
        _pca_callback("MergeablePCA(n_components=2)"),
        [Window("f", size=1)],
        exception_handler=None,
        when="AND",
        precompute=True,
    )
    spec = d._branch_groups["f"][0]

    from deisa.dask.utils import make_precomputed_view

    field = _field(n_rows=n_rows, n_cols=n_features, seed=13)
    leaves = [spec.branch_func(field[start : start + 3]) for start in range(0, n_rows, 3)]
    delivered = merge_tree(leaves)

    view = make_precomputed_view(
        d._pca_carrier_array("f"),
        t=0,
        signatures={},
        reapply=set(),
        registered_ndim=2,
        registered_shape=meta["f"]["global_shape"],
        pca_summary=delivered,
    )
    # The carrier is the one-element placeholder by construction; asserting it here documents why the guard had to move.
    assert tuple(view.shape) == (1, 1), f"the carrier must stay a one-element placeholder, got {tuple(view.shape)}"
    assert view.shape[-1] == 1, "this is the value the old guard compared against"

    fitted = MergeablePCA(n_components=2).fit(view)
    assert fitted.n_features_in_ == n_features, f"expected d={n_features}, got n_features_in_={fitted.n_features_in_}"
    reference = _batch_reference(field)
    assert np.max(np.abs(fitted.singular_values_ - reference.singular_values[:2])) < 1e-10


def test_a_real_feature_basis_mismatch_is_still_refused():
    """Moving the declared count off the carrier must not make the guard useless.

    The guard exists to catch a summary describing a DIFFERENT feature basis than the array it was requested for. With
    the declared count taken from the registered metadata, that mismatch is still detected -- this is the test that
    keeps the blocker-2 fix from being "delete the check".
    """
    from deisa.dask.utils import make_precomputed_view

    d = _deisa_stub()
    field = _field(n_cols=6, seed=17)
    # A summary over a 4-feature basis, delivered for a 6-feature registered array.
    wrong = local_pca(field[:, :4])

    view = make_precomputed_view(
        d._pca_carrier_array("f"),
        t=0,
        signatures={},
        reapply=set(),
        registered_ndim=2,
        registered_shape=GLOBAL_SHAPE,
        pca_summary=wrong,
    )

    with pytest.raises(ValueError) as caught:
        MergeablePCA(n_components=2).fit(view)
    message = str(caught.value)
    assert "spans 4 features" in message, f"the message must name the summary's feature count, got {message!r}"
    assert "has 6" in message, f"the message must name the declared feature count, got {message!r}"
    assert "rechunk" in message, "the message must name the remedy"


def test_multidim_registered_shape_declares_the_product_of_its_feature_axes():
    """A multi-axis feature dimension multiplies out through the same policy ``_as_2d`` applies.

    A 3-D registered array with ``axis_names`` and the default last-axis feature policy declares ``d`` = that axis's
    extent. Using the registered shape rather than the carrier's own shape is what makes this agree with the bridge,
    which summarized under the same policy.
    """
    from deisa.dask.utils import make_precomputed_view

    meta = {"f": {"global_shape": (4, 5, 6), "chunk_shape": (4, 5, 6)}}
    d = _deisa_stub()
    d.arrays_metadata = meta

    estimator = MergeablePCA(n_components=2, axis_names=("tor1", "tor2", "vpar"))
    assert estimator._declared_feature_count((4, 5, 6)) == 6, "the default feature axis is the last one"

    field = np.random.RandomState(19).randn(4, 5, 6)
    leaves = [local_pca(np.ascontiguousarray(part)) for part in np.split(field.reshape(-1, 6), 2)]
    delivered = merge_pca(leaves[0], leaves[1])

    view = make_precomputed_view(
        d._pca_carrier_array("f"),
        t=0,
        signatures={},
        reapply=set(),
        registered_ndim=3,
        registered_shape=meta["f"]["global_shape"],
        pca_summary=delivered,
    )
    assert tuple(view.shape) == (1, 1, 1), "the carrier is a one-element placeholder in every dimension"

    fitted = estimator.fit(view)
    assert fitted.n_features_in_ == 6, f"the last axis is the feature axis, got d={fitted.n_features_in_}"
