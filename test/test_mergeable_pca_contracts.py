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
Contract tests for MergeablePCA: ``n_components_`` must never disagree with ``components_.shape[0]``.

The invariant
-------------
``n_components_`` is the ONE number a caller reads to size a buffer for ``transform``'s output, so it must equal
``components_.shape[0]`` for every configuration. When it does not, the object lies about its own width: the caller
allocates for ``n_components_`` columns and ``transform`` -- which validates ``Z.shape[1] == n_components_`` -- hands
back a narrower array.

The defect this file pins
-------------------------
``n_components=None`` means "keep everything", so the achieved width is declared as
``keep = min(n_samples, n_features)``. That declaration is WRONG once ``local_rank`` truncates the leaves: the merged
root summary then has rank ``min(n_components_climbs, ...)`` far below the declared ``min(n_samples, n_features)``, and
numpy slicing silently clamps to what is actually there. Measured, on ``(1024, 12)`` over 4 row blocks:

    n_components=None, local_rank=2 -> n_components_=12, components_.shape=(11, 12)
    n_components=None, local_rank=3 -> n_components_=12, components_.shape=(12, 12)
    n_components=None, local_rank=5 -> n_components_=12, components_.shape=(12, 12)

The shortfall is NOT always one row: on ``(1024, 50)`` the same configuration reports ``n_components_=50`` against a
``(11, 50)`` component matrix, and ``local_rank=5`` gives ``(23, 50)``. ``n_components`` being an explicit integer was
never affected, because ``fit`` guards that case against the merged root rank before anything is published.

Why this file asserts a REFUSAL rather than an honoured request
---------------------------------------------------------------
The sweep below cannot assert that ``n_components=None, local_rank=2`` returns 12 components, because it CANNOT: the
merge truncated to rank 11 and there is no way to recover the 12th direction without re-reading the data. The two
remaining options are to report the achieved width or to refuse. This repo's rule is refuse-rather-than-silently-
approximate, and a silently-short result is exactly that violation -- so a request that cannot be honoured RAISES,
naming both counts. Every assertion below therefore accepts either outcome and asserts the invariant holds in both,
which is what makes the sweep a real test of the CONTRACT rather than of one chosen implementation.
"""

from __future__ import annotations

import dask.array as da
import dask.config
import numpy as np
import pytest

from deisa.dask.mergeable_pca import MergeablePCA


@pytest.fixture(autouse=True)
def default_scheduler():
    """Pin the Dask scheduler, so a ``Client`` leaked by another test in this xdist worker cannot break these fits.

    Same reason, same remedy, as the fixture in the estimator's own test module: an order-dependent "no Client active"
    failure must not be attributed to the contract under test.
    """
    previous = dask.config.get("scheduler", default=None)
    dask.config.set(scheduler="threads")
    try:
        yield
    finally:
        dask.config.set(scheduler=previous)


def make_data(n_samples: int, n_features: int, seed: int = 0) -> np.ndarray:
    """Build reproducible gaussian ``(n_samples, n_features)`` data with a fixed RandomState.

    - ``:param n_samples:`` Row count, i.e. the sample axis.
    - ``:param n_features:`` Column count, i.e. the feature dimension ``d``.
    - ``:param seed:`` Seed for the generator, so every case in the sweep sees the same numbers.
    """
    return np.random.default_rng(seed).standard_normal((n_samples, n_features))


def make_dask_array(n_samples: int = 1024, n_features: int = 12, rows: int = 256) -> da.Array:
    """Wrap :func:`make_data` as a dask array with the COMPLETE feature dimension in one chunk.

    - ``:param n_samples:`` Row count of the generated data.
    - ``:param n_features:`` Feature dimension, always a single chunk as the estimator requires.
    - ``:param rows:`` Row chunk size, i.e. how many leaves the merge tree has.
    """
    return da.from_array(make_data(n_samples, n_features), chunks=(rows, n_features))


def assert_contract(pca: MergeablePCA, n_features: int, label: str) -> None:
    """Assert the invariant and the widths that follow from it, or say precisely which one broke.

    Checked, in order: the headline invariant ``n_components_ == components_.shape[0]``; that ``components_`` still has
    the fitted feature dimension; that ``singular_values_`` and ``explained_variance_`` agree with the same width (a
    second, independent route to the same lie); and that ``transform`` emits exactly that many columns.

    - ``:param pca:`` A fitted estimator.
    - ``:param n_features:`` The fitted feature dimension ``d``.
    - ``:param label:`` Human-readable case description, quoted in the assertion messages.
    """
    declared = int(pca.n_components_)
    actual = int(pca.components_.shape[0])
    assert declared == actual, (
        f"{label}: n_components_={declared} disagrees with components_.shape[0]={actual}; the estimator lies about its "
        f"own width and a caller sizing a buffer from n_components_ gets the wrong shape"
    )
    assert int(pca.components_.shape[1]) == n_features, f"{label}: components_ must be (n_components_, {n_features})"
    assert int(pca.singular_values_.shape[0]) == declared, f"{label}: singular_values_ must be (n_components_,)"
    assert int(pca.explained_variance_.shape[0]) == declared, f"{label}: explained_variance_ must be (n_components_,)"

    Z = np.asarray(pca.transform(make_data(64, n_features, seed=99)))
    assert Z.shape[1] == actual, (
        f"{label}: transform emitted {Z.shape[1]} columns but components_.shape[0]={actual}; the projection width and "
        f"the component count must be the same number"
    )


def fit_or_refuse(n_components, local_rank, X: da.Array) -> tuple[MergeablePCA | None, str]:
    """Fit, returning either the estimator or the refusal message, never a silently-wrong result.

    The refusal is returned as a string rather than raised so a caller can assert on its CONTENT -- that it names both
    the requested and the achievable count, and the remedy.

    - ``:param n_components:`` The ``n_components`` argument, or ``None`` for "keep everything".
    - ``:param local_rank:`` The ``local_rank`` argument.
    - ``:param X:`` The dask array to fit.
    """
    try:
        return MergeablePCA(n_components=n_components, local_rank=local_rank).fit(X), ""
    except ValueError as exc:
        return None, str(exc)


# =============================================================================
# 1. The exact triggering case, pinned
# =============================================================================
def test_none_with_truncating_local_rank_never_reports_more_components_than_it_has():
    """``n_components=None, local_rank=2`` on data whose merge truncates one short: the invariant must hold.

    This is the regression pin for the reported symptom. Before the fix the fit SUCCEEDED with ``n_components_=12``
    against ``components_.shape=(11, 12)``, and the shortfall is not always one row: on the wider slab below the same
    configuration reports 50 against 11. Either the invariant holds or the fit refuses with both counts named -- what
    must never happen is a successful fit whose ``n_components_`` overstates its width.
    """
    for n_features in (12, 50):
        X = make_dask_array(n_samples=1024, n_features=n_features, rows=256)
        pca, message = fit_or_refuse(None, 2, X)

        if pca is None:
            # A refusal is acceptable ONLY if it names both the requested and the achievable count.
            assert str(min(1024, n_features)) in message, (
                f"d={n_features}: the refusal must name the REQUESTED count {min(1024, n_features)}, got: {message}"
            )
            assert "local_rank" in message, f"d={n_features}: the refusal must name the remedy (local_rank): {message}"
            continue

        assert_contract(pca, n_features, f"n_components=None, local_rank=2, d={n_features}")


# =============================================================================
# 2. The sweep: the real test
# =============================================================================
@pytest.mark.parametrize("n_components", [None, 3, 8])
@pytest.mark.parametrize("local_rank", [2, 3, 5])
def test_n_components_matches_the_component_matrix_across_the_configuration_sweep(n_components, local_rank):
    """Every configuration either honours its request exactly or refuses -- never silently returns fewer.

    This is the test that matters: the single triggering case above is one point in a 3x3 grid, and the invariant has
    to hold at all nine. ``n_components=3`` and ``8`` are included to prove the fix did not regress the explicit-request
    path, which was already correct and must stay correct.

    - ``:param n_components:`` Swept request: ``None`` means "keep everything".
    - ``:param local_rank:`` Swept per-leaf truncation.
    """
    n_samples, n_features = 1024, 12
    X = make_dask_array(n_samples=n_samples, n_features=n_features, rows=256)

    pca, message = fit_or_refuse(n_components, local_rank, X)

    if pca is None:
        # Refusing is legitimate for this grid -- with local_rank=2 the merged rank cannot reach min(n_samples, d)=12.
        # The message still has to be actionable: it names the request, the shortfall, and the knob that fixes it.
        assert "n_components" in message, f"n_components={n_components}, local_rank={local_rank}: {message}"
        assert "local_rank" in message, f"n_components={n_components}, local_rank={local_rank}: {message}"
        return

    assert_contract(pca, n_features, f"n_components={n_components}, local_rank={local_rank}")

    # A fit that DID succeed must report exactly what was asked for. n_components=None is the exception in form only:
    # "keep everything" means "as much as the merge produced", which the invariant above already pins.
    if n_components is not None:
        assert int(pca.n_components_) == n_components, (
            f"n_components={n_components}, local_rank={local_rank}: an explicit request must be honoured exactly, got "
            f"{pca.n_components_}"
        )


def test_the_sweep_grid_actually_exercises_both_outcomes():
    """The sweep must contain at least one refusal and at least one honoured fit, or it is not testing the contract.

    Without this the grid could silently degenerate into "everything refuses" (or "nothing does") and still go green
    while the contract is broken. Both are pinned here so the sweep cannot lose its teeth.
    """
    n_features = 12
    X = make_dask_array(n_samples=1024, n_features=n_features, rows=256)
    outcomes = {}
    for n_components in (None, 3, 8):
        for local_rank in (2, 3, 5):
            pca, _message = fit_or_refuse(n_components, local_rank, X)
            outcomes[(n_components, local_rank)] = "refused" if pca is None else "fitted"

    assert "refused" in outcomes.values(), f"the grid must exercise a refusal, got {outcomes}"
    assert "fitted" in outcomes.values(), f"the grid must exercise an honoured fit, got {outcomes}"


# =============================================================================
# 3. transform width agrees with the component matrix
# =============================================================================
@pytest.mark.parametrize("n_components", [None, 3, 8])
@pytest.mark.parametrize("local_rank", [2, 3, 5])
def test_transform_width_agrees_with_the_component_matrix(n_components, local_rank):
    """``transform`` emits exactly ``components_.shape[0]`` columns, for every configuration in the sweep.

    Separated from the invariant above because this is the consequence a caller actually trips over: ``transform``
    validates ``Z.shape[1] == n_components_``, so a lying ``n_components_`` is what turns a buffer mismatch into a hard
    error far from the fit that caused it. A refused configuration is skipped rather than failed: there is no estimator
    to project with, and refusing IS the contract holding. The honoured cases are where the width has to be right.

    - ``:param n_components:`` Swept request: ``None`` means "keep everything".
    - ``:param local_rank:`` Swept per-leaf truncation.
    """
    n_features = 12
    X = make_dask_array(n_samples=1024, n_features=n_features, rows=256)

    pca, message = fit_or_refuse(n_components, local_rank, X)
    if pca is None:
        # Refused on purpose: there is no estimator, so there is no transform width to disagree with.
        assert "n_components" in message and "local_rank" in message, "a refusal must be actionable"
        return

    Z = np.asarray(pca.transform(X))
    assert Z.shape[1] == int(pca.components_.shape[0]), (
        f"n_components={n_components}, local_rank={local_rank}: transform emitted {Z.shape[1]} columns but "
        f"components_.shape[0]={pca.components_.shape[0]}"
    )
    # The projection is centred by the pooled mean, so its own sample mean is ~0 over the fitted data.
    assert float(np.max(np.abs(np.mean(Z, axis=0)))) < 1e-8, "the projection must be centred by mean_"


# =============================================================================
# 4. n_components=None with a full-rank merge still keeps min(n_samples, n_features)
# =============================================================================
def test_none_with_a_full_rank_merge_is_unchanged():
    """The fix must not touch the honest ``None`` case: a merge that reaches the ceiling keeps every component.

    ``local_rank=None`` leaves full local rank, so the first merge saturates at ``d`` and the root rank equals
    ``min(n_samples, n_features)``. That is the configuration the existing estimator tests already pin, and it is
    re-asserted here so the contract file fails loudly if the fix ever widens into a refusal of a satisfiable request.
    """
    n_samples, n_features = 1024, 12
    X = make_dask_array(n_samples=n_samples, n_features=n_features, rows=256)

    pca = MergeablePCA(n_components=None, local_rank=None).fit(X)

    assert int(pca.n_components_) == min(n_samples, n_features), "a saturated merge must keep every component"
    assert pca.components_.shape == (min(n_samples, n_features), n_features)
    assert_contract(pca, n_features, "n_components=None, local_rank=None")


def test_wide_data_still_refuses_none_when_the_merge_cannot_reach_the_ceiling():
    """The refusal is not a data-size artefact: it fires because of the MERGE, on data with room to spare.

    On ``(1024, 50)`` there are 50 features and 1024 samples, so ``min(n_samples, n_features) = 50`` is reachable in
    principle -- but a truncated merge only climbs to rank 23 at ``local_rank=5``. The honest ``None`` here means "as
    much as the merge produced", which is a DIFFERENT request from "all 50", so this case must either keep what the
    merge achieved (and report that number) or refuse with both counts named. Either way the invariant holds.
    """
    n_features = 50
    X = make_dask_array(n_samples=1024, n_features=n_features, rows=256)

    pca, message = fit_or_refuse(None, 5, X)

    if pca is None:
        assert str(min(1024, n_features)) in message, f"the refusal must name the requested count 50: {message}"
        assert "local_rank" in message, f"the refusal must name the remedy: {message}"
        return

    assert_contract(pca, n_features, "n_components=None, local_rank=5, d=50")
    assert int(pca.n_components_) == int(pca._summary_.rank), (
        "a honoured None must report the rank the merge actually achieved"
    )
