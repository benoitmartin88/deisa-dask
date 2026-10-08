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
Tests for the n-dimensional axis policy of MergeablePCA, and for the two gyrokinetic layouts it distinguishes.

The application these tests target
---------------------------------
A gyrokinetic distribution is indexed ``(species, tor1, tor2, tor3, vpar, mu)`` and split across MPI ranks over the
SPATIAL axes, so every rank owns a contiguous ``(tor1, tor2, tor3)`` box with the COMPLETE velocity space. Two axis
policies follow, and they behave OPPOSITELY (all figures below are from the module docstring's measurement of real
mesh sizes; a summary is ``(rank, d) + d`` float64 elements against ``n_samples * d`` for the data):

- **Layout A**, features ``(vpar, mu)``: ``d = Nvpar * Nmu``, fixed by the physics and independent of the rank count. On
  ``(tor1, tor2, tor3, vpar, mu) = (512, 128, 64, 128, 8)``, ``d = 1024`` and a full-rank leaf summary is 8.0 MiB
  against a 4096 MiB slab at 8 ranks (511x) and a 512 MiB slab at 64 ranks (64x). It compresses at every rank count
  measured, so full local rank is viable and the merge is EXACT.
- **Layout B**, features ``(tor1, tor2, tor3)``: ``d`` is the local box and GROWS AS RANKS DECREASE. At 8 ranks
  ``d = 524288`` with ``n_samples = 1024`` velocity cells, so the merged full-rank summary needs 32800 MiB against a
  distributed slab of 32768 MiB: 1.00x, NO compression. (Sizing a leaf at the naive ``r = d`` rather than the true
  ``min(n_block, d) = 1024`` gives the 2097156 MiB figure quoted elsewhere; either way it does not compress.)

So Layout A is the default and must be exact, and Layout B's full-rank blow-up must be REFUSED with the remedy named.

Metrics: sign-invariant only
-----------------------------
Every accuracy assertion uses one of the metrics already defined in the estimator's test module, re-imported here so
this file stands alone: **M1** ``subspace_distance`` (``1 - min singular value`` of the subspace overlap, 0 = same
subspace) and **M2** ``rel_var_error`` (relative error of retained total variance against the FULL-rank batch
reference). Raw component equality is NEVER asserted: eigenvector sign is arbitrary, so such a test is flaky by
construction. M2 also supplies the monotonicity sweep, because it falls monotonically as ``local_rank`` grows.

Every test here fails without the axis policy: before it, the estimator raised "X must be 2-dimensional" on all of
these arrays, and the Layout B refusal did not exist.
"""

from __future__ import annotations

import dask.array as da
import dask.config
import numpy as np
import pytest
from test_mergeable_pca_estimator import rel_var_error, subspace_distance

from deisa.dask.mergeable_pca import VELOCITY_AXES, MergeablePCA

# ------------------------------------------------------------------------------ the structured_mesh shape
AXIS_NAMES_5D = ("species", "tor1", "tor2", "tor3", "vpar", "mu")
"""The distribution's axes, in order, with ``species`` leading as the app declares them."""

SPATIAL_AXES = ("tor1", "tor2", "tor3")
"""The axes an MPI spatial split distributes, so each rank holds a contiguous box of these."""


@pytest.fixture(autouse=True)
def default_scheduler():
    """Pin the Dask scheduler for this module, so a leaked global Client from another xdist worker cannot break it.

    Same reason and same fix as in the estimator's own test module: a bare ``.compute()`` raises "Requested
    dask.distributed scheduler but no Client active" once another test's Client is gone, and under
    ``-n 16 --dist loadgroup`` that lands on whichever test this worker runs next.

    Restored afterwards: a fix for the TEST environment, not a behaviour change, and it must not leak back.
    """
    previous = dask.config.get("scheduler", default=None)
    dask.config.set(scheduler="threads")
    try:
        yield
    finally:
        dask.config.set(scheduler=previous)


def structured_mesh_field(seed: int = 0, shape: tuple[int, ...] = (2, 8, 4, 4, 6, 2)) -> np.ndarray:
    """A reproducible array shaped like the distribution, with a low-rank structure in the velocity axes.

    The structure is deliberate, not decoration: the assertion that Layout A is EXACT is only meaningful if the data
    actually has a principal subspace. A ``(vpar, mu)``-wise linear field plus noise gives a clear spectrum, so a
    truncated ``local_rank`` is measurably worse and the full-rank fit is measurably exact. A fixed RandomState keeps
    every number in this file stable.

    - ``:param seed:`` RandomState seed.
    - ``:param shape:`` Shape of the field; defaults to ``(species, tor1, tor2, tor3, vpar, mu)``.
    """
    state = np.random.RandomState(seed)
    # Three spatial modes, each with its own weight per velocity point: a rank-3 field in the velocity feature space.
    basis = state.randn(*shape[:-2], 3)  # one coefficient per spatial cell and mode
    amplitudes = state.randn(3, int(np.prod(shape[-2:])))  # one weight per (mode, velocity point)
    signal = np.einsum("...i,ij->...j", basis, amplitudes).reshape(*shape)
    return signal + 0.35 * state.randn(*shape)


def batch_reference(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batch PCA of an already-flattened array: return ``(singular_values, components)``, numpy only.

    - ``:param data:`` 2-D ``(n_samples, n_features)`` array.
    """
    centered = np.asarray(data, dtype=np.float64)
    centered = centered - centered.mean(axis=0)
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    return singular_values, components


def _spatial_block_count(shape: tuple[int, ...], chunks: tuple[int, ...]) -> int:
    """How many blocks the three spatial axes of a ``(species, tor1..3, vpar, mu)`` array are cut into.

    One block per "rank": an MPI layout over the spatial axes distributes them in order and maximally, so the block
    count is the product of the per-axis block counts. Computed from the chunk sizes rather than from ``da`` so the
    test can plan a chunking before building the array.

    - ``:param shape:`` Shape of the array.
    - ``:param chunks:`` Chunk size along every axis.
    """
    return int(np.prod([-(-shape[axis] // chunks[axis]) for axis in (1, 2, 3)]))


# =============================================================================
# Layout A: velocity-space features per spatial cell (the default)
# =============================================================================
def test_layout_a_defaults_to_the_velocity_axes_and_is_exact():
    """Layout A needs no configuration at all: naming ``(vpar, mu)`` in ``axis_names`` selects them as features.

    This is the measured-exact configuration. On the real mesh family, a full-rank summary of the velocity space is 8x
    to 511x smaller than the rank's slab (8.0 MiB against 4096 MiB at 8 ranks for ``d = 1024``), so full local rank is
    viable and the merge reproduces the batch SVD to roundoff. Metric: M1 subspace distance at ``k = 5 < d = 12``,
    plus the sign-free singular values and mean.
    """
    data = structured_mesh_field()
    X = da.from_array(data, chunks=(2, 4, 4, 4, 6, 2))  # 4 spatial blocks = 4 ranks
    expected = data.reshape(-1, int(np.prod(data.shape[-2:])))

    pca = MergeablePCA(n_components=5, axis_names=AXIS_NAMES_5D).fit(X)
    singular_values, components = batch_reference(expected)

    # The axis policy did what it claims: features are the velocity space, samples are the spatial cells.
    assert pca.n_features_in_ == 12, "features must be vpar * mu"
    assert pca.n_samples_ == 2 * 8 * 4 * 4, "samples must be species * tor1 * tor2 * tor3"
    # Sign-free quantities compare directly.
    assert np.max(np.abs(pca.singular_values_ - singular_values[:5])) < 1e-10
    assert np.max(np.abs(pca.mean_ - expected.mean(axis=0))) < 1e-12
    # M1 at k = 5 < d = 12: the principal subspace IS the batch one, so the full-rank merge is exact.
    assert subspace_distance(pca.components_, components, k=5) < 1e-10


def test_layout_a_is_exact_for_every_rank_count():
    """Exactness is a property of the merge, not of a particular number of ranks, so sweep the spatial split.

    One block per "rank": the three spatial axes are split into 1, 2, 4, 8 and 16 blocks (that is what an MPI layout
    over ``(tor1, tor2, tor3)`` produces as the rank count grows). At every count the Layout A fit must reproduce the
    batch PCA, because ``d`` is fixed by the physics and does not depend on how the space is split.
    """
    data = structured_mesh_field(seed=1)
    expected = data.reshape(-1, 12)
    singular_values, components = batch_reference(expected)

    for blocks in (1, 2, 4, 8, 16):
        # Halve tor2, then tor3, then tor2 again, then tor3 again: that reaches 1, 2, 4, 8 and 16 spatial blocks,
        # which is how an MPI layout over (tor1, tor2, tor3) grows the rank count. Halves round UP so no axis empties.
        chunks = [2, 8, 4, 4, 6, 2]
        for axis in (2, 3, 2, 3):
            if _spatial_block_count(data.shape, tuple(chunks)) >= blocks:
                break
            chunks[axis] = max(1, -(-chunks[axis] // 2))
        X = da.from_array(data, chunks=tuple(chunks))
        assert _spatial_block_count(data.shape, tuple(chunks)) == blocks, f"wrong split, got {X.chunks}"

        pca = MergeablePCA(n_components=4, axis_names=AXIS_NAMES_5D).fit(X)

        assert np.max(np.abs(pca.singular_values_ - singular_values[:4])) < 1e-10
        # M1 at k = 4 < d = 12.
        assert subspace_distance(pca.components_, components, k=4) < 1e-10


def test_layout_a_compresses_its_own_input_by_a_measured_factor():
    """Layout A's summary is smaller than the data it summarizes, by the factor the measurement predicts.

    The invariant the guard rests on is ``n_samples > d``. Here ``n_samples = 256``, ``d = 12``, so the summary is at
    most ``12 * 12 + 12 = 156`` elements against ``256 * 12 = 3072``: at least 19x smaller. Asserting the bound rather
    than the exact count keeps the test honest under any BLAS (it is an inequality on a payload size, not a number).
    """
    data = structured_mesh_field(seed=2)
    X = da.from_array(data, chunks=(2, 4, 4, 4, 6, 2))

    pca = MergeablePCA(n_components=5, axis_names=AXIS_NAMES_5D).fit(X)

    payload = pca._summary_.components.size + pca._summary_.mean.size
    assert payload < pca.n_samples_ * pca.n_features_in_
    assert pca.n_samples_ * pca.n_features_in_ / payload > 10.0


def test_layout_a_survives_a_5d_field_without_a_species_axis():
    """The species axis is optional: the real 5-D field is ``(tor1, tor2, tor3, vpar, mu)``.

    The estimator must key off the names the caller supplies, not off a hard-coded six-axis expectation.
    """
    names = ("tor1", "tor2", "tor3", "vpar", "mu")
    data = structured_mesh_field(seed=3, shape=(8, 4, 4, 6, 2))
    X = da.from_array(data, chunks=(4, 4, 4, 6, 2))

    pca = MergeablePCA(n_components=3, axis_names=names).fit(X)

    assert pca.n_features_in_ == 12
    assert pca.n_samples_ == 8 * 4 * 4
    assert pca.n_components_ == 3


def test_layout_a_transform_projects_the_5d_field():
    """``transform`` accepts the same axis policy as ``fit``, so a 5-D field projects without reshaping by hand.

    The projection is compared to the batch projection of the explicitly flattened array, which is the same computation
    written out longhand. It is compared through the sign-invariant Gram matrix ``Z.T @ Z``, which equals
    ``diag(singular_values)**2`` for a centered batch PCA and is invariant to any orthogonal change of basis inside the
    retained subspace -- so this needs no sign alignment and no per-column comparison.
    """
    names = ("tor1", "tor2", "tor3", "vpar", "mu")
    data = structured_mesh_field(seed=4, shape=(8, 4, 4, 6, 2))
    X = da.from_array(data, chunks=(4, 4, 4, 6, 2))

    pca = MergeablePCA(n_components=4, axis_names=names).fit(X)
    projected = np.asarray(pca.transform(X))

    singular_values = batch_reference(data.reshape(-1, 12))[0][:4]
    assert projected.shape == (8 * 4 * 4, 4)
    # Sign-free and rotation-free: the Gram matrix of the projection is the squared spectrum.
    expected_gram = np.diag(singular_values**2)
    assert np.max(np.abs(projected.T @ projected - expected_gram)) < 1e-10 * float(np.sum(singular_values**2))


# =============================================================================
# Layout B: spatial features per velocity cell -- the full-rank blow-up is REFUSED
# =============================================================================
def test_layout_b_refuses_the_full_rank_blowup_and_names_the_remedy():
    """Layout B with a full-rank local summary must be REFUSED, not silently produced bigger than its input.

    Why: the feature dimension is the LOCAL spatial box, which grows as ranks decrease, so the summary's size is
    quadratic in ``d`` while the data is linear. Measured on ``(512, 128, 64, 128, 8)`` at 8 ranks: ``d = 524288``
    against ``n_samples = 1024`` velocity cells, a merged full-rank summary of 32800 MiB for a 32768 MiB distributed
    slab -- 1.00x, i.e. no compression at all, and 32800 MiB with the naive ``r = d`` leaf accounting that some
    write-ups quote. Under Layout A the same field gives ``d = 1024`` and an 8.0 MiB summary, 511x smaller than the
    slab.

    The synthetic case below reproduces the same regime at a size a test can hold: a rank's box is ``d = 32`` spatial
    features against ``n_samples = 16`` velocity samples, so the invariant ``n_samples > d`` fails exactly as it does
    in the measurement. The refusal must name the remedy (``local_rank``) and the alternative (velocity features).
    """
    # One rank's slab: species=1, a (2, 4, 4) spatial box held complete, and the full (8, 2) velocity space as samples.
    data = structured_mesh_field(seed=5, shape=(1, 2, 4, 4, 8, 2))
    X = da.from_array(data, chunks=data.shape)  # one rank owns the whole slab

    with pytest.raises(ValueError) as caught:
        MergeablePCA(n_components=4, axis_names=AXIS_NAMES_5D, feature_axes=SPATIAL_AXES).fit(X)

    message = str(caught.value)
    assert "would not compress" in message, f"the refusal must state the reason: {message}"
    assert "local_rank" in message, f"the refusal must name the remedy: {message}"
    assert "velocity" in message, f"the refusal must name the better axis choice: {message}"
    assert "d=524288" in message, "the refusal must carry the measured reason, not a vague warning"


def test_layout_b_still_works_once_local_rank_truncates_the_leaves():
    """The refusal must be actionable: ``local_rank=R`` below ``n_samples`` makes Layout B run, at a known cost.

    Same slab as the refusal above (``d = 32`` features, ``n_samples = 16`` samples). With ``local_rank = 4`` the root
    summary is ``4 * 32 + 32 = 160`` elements against ``16 * 32 = 512`` for the data: 3.2x smaller, so the guard is
    satisfied. The price is accuracy, and the test says so with M2 rather than hiding it: four retained directions per
    leaf is a drastic truncation of a 32-dimensional feature space, so the retained variance falls well short of the
    batch reference.
    """
    data = structured_mesh_field(seed=5, shape=(1, 2, 4, 4, 8, 2))
    X = da.from_array(data, chunks=data.shape)  # one rank owns the whole slab
    # The axis policy transposes the sample axes in front of the feature axes, so the flattened reference has the
    # same row order: (species, vpar, mu) rows by (tor1, tor2, tor3) features.
    sample_first = (0, 4, 5, 1, 2, 3)
    expected = np.moveaxis(data, sample_first, tuple(range(6))).reshape(-1, 32)

    pca = MergeablePCA(n_components=4, axis_names=AXIS_NAMES_5D, feature_axes=SPATIAL_AXES, local_rank=4).fit(X)

    assert pca.n_features_in_ == 32, "features must be the spatial box"
    assert pca.n_samples_ == 1 * 8 * 2, "samples must be species * vpar * mu"
    assert pca._summary_.rank == 4, "local_rank caps the summary"
    payload = pca._summary_.components.size + pca._summary_.mean.size
    assert payload < pca.n_samples_ * pca.n_features_in_, "the truncation must restore compression"
    # M2 against the FULL-rank batch reference: lossy, deliberately asserted rather than papered over.
    assert rel_var_error(pca._summary_.singular_values, batch_reference(expected)[0]) > 1e-3


def test_layout_b_is_a_different_answer_not_a_rearrangement():
    """Layout A and Layout B answer genuinely different questions, which is why the axis choice must be explicit.

    Both fit without error (Layout B with a ``local_rank`` small enough to compress), and their feature dimensions
    differ by construction -- ``Nvpar * Nmu`` versus the local box. If they somehow produced the same ``d``, the axis
    policy would be silently ignoring its own input, so the dimensions are asserted to differ.
    """
    data = structured_mesh_field(seed=6, shape=(1, 2, 4, 4, 8, 2))
    X = da.from_array(data, chunks=data.shape)

    layout_a = MergeablePCA(n_components=2, axis_names=AXIS_NAMES_5D).fit(X)
    layout_b = MergeablePCA(n_components=2, axis_names=AXIS_NAMES_5D, feature_axes=SPATIAL_AXES, local_rank=2).fit(X)

    assert layout_a.n_features_in_ == 16  # vpar * mu = 8 * 2
    assert layout_b.n_features_in_ == 32  # tor1 * tor2 * tor3 = 2 * 4 * 4
    assert layout_a.n_features_in_ != layout_b.n_features_in_


# =============================================================================
# The axis policy is explicit: a wrong or ambiguous spec must raise
# =============================================================================
@pytest.mark.parametrize(
    ("kwargs", "expected_fragment"),
    [
        ({"feature_axes": ("vr", "mu")}, "not among the axis_names declared by the caller"),
        ({"feature_axes": ("vpar", "mu", "mu")}, "names the axis mu twice"),
        ({"feature_axes": 9}, "out of range for an array with 6 axes"),
        ({"feature_axes": 6}, "out of range for an array with 6 axes"),
        ({"feature_axes": ("vpar", "mu"), "sample_axes": ("species", "tor1")}, "leaves tor2, tor3 unclassified"),
        ({"feature_axes": ("vpar",), "sample_axes": ("vpar", "mu")}, "both claim vpar"),
        ({"feature_axes": AXIS_NAMES_5D}, "every axis was declared as a feature"),
        ({"feature_axes": ()}, "feature_axes is empty"),
    ],
)
def test_ambiguous_or_wrong_axis_specs_are_refused_not_guessed(kwargs, expected_fragment):
    """Every malformed axis specification is a refusal that names the problem, never a guess.

    The alternative in each case is a silently wrong PCA: a misclassified axis changes which cells are samples and
    which are features, and nothing downstream would notice.
    """
    data = structured_mesh_field(seed=7)
    X = da.from_array(data, chunks=(2, 4, 4, 4, 6, 2))

    with pytest.raises(ValueError, match=expected_fragment.replace("(", r"\(").replace(", ", r",\s*")):
        MergeablePCA(n_components=2, axis_names=AXIS_NAMES_5D, **kwargs).fit(X)


def test_axis_names_must_describe_the_array_actually_given():
    """A name list of the wrong length must be refused: names would index past ``ndim`` and mis-resolve.

    The same field is fitted twice, once with the six names it has and once with the five names of the species-free
    variant. The second must raise rather than classify the axes against names that do not line up.
    """
    data = structured_mesh_field(seed=8)
    X = da.from_array(data, chunks=(2, 4, 4, 4, 6, 2))

    assert MergeablePCA(n_components=2, axis_names=AXIS_NAMES_5D).fit(X).n_features_in_ == 12

    with pytest.raises(ValueError, match="axis_names has 5 names but X has 6 axes"):
        MergeablePCA(n_components=2, axis_names=("tor1", "tor2", "tor3", "vpar", "mu")).fit(X)


def test_duplicate_axis_names_are_refused_at_construction():
    """A repeated name makes an axis ambiguous, so it is refused before any work happens."""
    with pytest.raises(ValueError, match="must not repeat a name"):
        MergeablePCA(axis_names=("species", "tor1", "tor1", "tor3", "vpar", "mu"))


@pytest.mark.parametrize("shape", [(), (20,)])
def test_fewer_than_two_axes_is_refused(shape):
    """Zero or one axis cannot be partitioned into samples and features, so it is refused with the remedy.

    A 1-D array is the case that used to report "must be 2-dimensional"; the message now says what is missing and how
    to express it. Both refusals fire before any indexing, so neither can raise an ``IndexError`` instead.
    """
    with pytest.raises(ValueError, match="2-dimensional"):
        MergeablePCA(n_components=1).fit(da.from_array(np.ones(shape)))


def test_axis_names_may_be_used_without_affecting_a_two_dimensional_fit():
    """For 2-D input the axis names are cosmetic: they must not change the answer, only the messages.

    This is what keeps the existing 2-D call sites valid. A 2-D array's last axis is the feature axis either way, so
    naming its axes cannot move a boundary -- and the fitted attributes prove it did not.
    """
    data = structured_mesh_field(seed=9, shape=(200, 12))
    X = da.from_array(data, chunks=(50, 12))

    unnamed = MergeablePCA(n_components=4).fit(X)
    named = MergeablePCA(n_components=4, axis_names=("species", "vpar")).fit(X)

    assert named.n_features_in_ == unnamed.n_features_in_ == 12
    assert np.max(np.abs(named.singular_values_ - unnamed.singular_values_)) < 1e-12
    assert subspace_distance(named.components_, unnamed.components_, k=4) < 1e-12


# =============================================================================
# The feature-chunk guard: fires only where it applies, and names the structured_mesh remedy
# =============================================================================
@pytest.mark.parametrize("split_axis,label", [(4, "vpar"), (5, "mu")])
def test_a_split_velocity_axis_is_refused_and_names_the_remedy(split_axis, label):
    """A VELOCITY axis split across chunks is refused, naming that axis and the velocity-axis rechunk.

    Under Layout A the feature dimension must be complete on the rank that holds it, because a merge stacks summaries
    by feature position. An MPI split over the spatial axes satisfies this for free -- the whole velocity space stays
    on one rank -- so this guard fires only for a caller who rechunked across velocity, which is exactly when the
    summaries really would be unstackerable. The message names the offending axis, the rechunk that fixes it, and the
    structured_mesh reason it usually is unnecessary.
    """
    data = structured_mesh_field(seed=10)
    blocks = (2, 4, 4, 4, 6, 2)  # the axis extent per block; a halved entry splits that axis in two
    halved = data.shape[split_axis] // 2
    chunks = tuple(halved if i == split_axis else blocks[i] for i in range(6))
    X = da.from_array(data, chunks=chunks)
    assert len(X.chunks[split_axis]) == 2, "the test must actually split the axis"

    with pytest.raises(ValueError) as caught:
        MergeablePCA(n_components=2, axis_names=AXIS_NAMES_5D).fit(X)

    message = str(caught.value)
    assert label in message, f"the message must name the split axis, not a position: {message}"
    assert "split across Dask chunks" in message
    assert "rechunk" in message, f"the message must name the remedy: {message}"
    assert "velocity space on one rank" in message, (
        "the message must give the structured_mesh reason the rechunk is unneeded"
    )


def test_the_split_velocity_axis_remedy_actually_works():
    """The named remedy is not decorative: applying it produces an array this estimator accepts.

    Otherwise the message would be advice that does not fix the problem, which is the failure mode this whole
    refuse-not-approximate convention exists to prevent.
    """
    data = structured_mesh_field(seed=11)
    X = da.from_array(data, chunks=(2, 4, 4, 3, 3, 2))  # vpar split in two

    with pytest.raises(ValueError) as caught:
        MergeablePCA(n_components=2, axis_names=AXIS_NAMES_5D).fit(X)

    remedy = str(caught.value).split("X = X.rechunk(")[1].split(")")[0]
    rechunked = X.rechunk({4: -1})
    assert remedy == "{4: -1}", f"the message must name the exact rechunk, got {remedy!r}"
    assert len(rechunked.chunks[4]) == 1
    assert MergeablePCA(n_components=2, axis_names=AXIS_NAMES_5D).fit(rechunked).n_features_in_ == 12


def test_a_split_spatial_axis_is_fine_because_it_is_a_sample_axis():
    """Splitting a SAMPLE axis is the normal case and must never trip the guard.

    One leaf is built per block of the sample axes -- that is the entire parallelism of the reduction, and it is what
    makes one leaf per MPI rank. So the guard must read the feature axes only, and this pins that.
    """
    data = structured_mesh_field(seed=12)
    X = da.from_array(data, chunks=(2, 1, 4, 4, 6, 2))  # tor1 split into 8 blocks
    assert len(X.chunks[1]) == 8

    pca = MergeablePCA(n_components=3, axis_names=AXIS_NAMES_5D).fit(X)

    assert pca.n_features_in_ == 12
    # One leaf per rank: the graph must hold 8 local PCAs.
    graph = MergeablePCA(n_components=3, axis_names=AXIS_NAMES_5D)._fit_dask_delayed(X)
    counts: dict[str, int] = {}
    for task in graph.dask.values():
        name = getattr(getattr(task, "func", None), "__name__", None) or type(task).__name__
        counts[name] = counts.get(name, 0) + 1
    assert counts.get("local_pca", 0) == 8, f"expected one leaf per spatial block, got {counts}"


# =============================================================================
# local_rank sweep under Layout A: monotone, exact at full rank
# =============================================================================
@pytest.mark.parametrize("local_rank", [2, 4, 8, 12, None])
def test_local_rank_sweep_under_layout_a(local_rank):
    """Every Layout A rank keeps the requested public components, and full local rank is exact.

    The accuracy assertions are M1 and M2, both sign-invariant. The monotone-across-the-sweep assertion lives in
    :func:`test_layout_a_local_rank_degrades_monotonically`; this test pins the per-rank contract and the exact
    endpoint.
    """
    data = structured_mesh_field(seed=13)
    X = da.from_array(data, chunks=(2, 4, 4, 4, 6, 2))
    singular_values, components = batch_reference(data.reshape(-1, 12))

    pca = MergeablePCA(n_components=5, axis_names=AXIS_NAMES_5D, local_rank=local_rank).fit(X)

    assert pca.n_components_ == 5, "the request is satisfied in full, not silently short-changed"
    if local_rank is None:
        # Full local rank under Layout A is EXACT, which is the whole reason Layout A is the default.
        assert np.max(np.abs(pca.singular_values_ - singular_values[:5])) < 1e-10
        assert subspace_distance(pca.components_, components, k=5) < 1e-10
        # M2: no variance lost at full rank.
        assert rel_var_error(pca._summary_.singular_values, singular_values) < 1e-12


def test_layout_a_local_rank_degrades_monotonically():
    """Under Layout A the sign-invariant metric degrades MONOTONICALLY as ``local_rank`` shrinks.

    This is the assertion a raw per-component error cannot support: that metric is non-monotonic and reads ~2.0
    (maximally different) at exact reconstruction. The slack absorbs float noise where two consecutive ranks produce
    the same summary; it is far below the smallest measured step, so it cannot hide a regression.
    """
    data = structured_mesh_field(seed=14)
    X = da.from_array(data, chunks=(2, 4, 4, 4, 6, 2))
    singular_values, components = batch_reference(data.reshape(-1, 12))

    ranks = (2, 3, 4, 8, 12, None)
    subspace_distances = []
    variance_errors = []
    for local_rank in ranks:
        pca = MergeablePCA(n_components=5, axis_names=AXIS_NAMES_5D, local_rank=local_rank).fit(X)
        # M1: both sides cut to the same k = 5 < d = 12.
        subspace_distances.append(subspace_distance(pca.components_, components, k=5))
        # M2: retained variance of the summary against the FULL-rank batch reference.
        variance_errors.append(rel_var_error(pca._summary_.singular_values, singular_values))

    slack = 1e-12
    for previous, current in zip(subspace_distances, subspace_distances[1:]):
        assert current <= previous + slack, f"M1 not monotone at ranks={ranks}: {subspace_distances}"
    for previous, current in zip(variance_errors, variance_errors[1:]):
        assert current <= previous + slack, f"M2 not monotone at ranks={ranks}: {variance_errors}"

    # Full local rank is exact on both, and the smallest rank is measurably worse than the largest truncated one.
    assert variance_errors[-1] < 1e-10
    assert subspace_distances[-1] < 1e-10
    assert variance_errors[0] > variance_errors[-2]
    assert subspace_distances[0] > subspace_distances[-2]


# =============================================================================
# The axis names are a public constant, so the default cannot drift from the docs
# =============================================================================
def test_velocity_axes_constant_matches_the_axis_names_the_tests_use():
    """``VELOCITY_AXES`` is the documented default feature set, so it must stay ``(vpar, mu)`` in that order.

    Every message, docstring and test above refers to the velocity axes by these names; if the constant drifted, the
    documented default would silently become something else.
    """
    assert VELOCITY_AXES == ("vpar", "mu")
    assert VELOCITY_AXES[-2:] == AXIS_NAMES_5D[-2:]


def test_the_default_velocity_feature_selection_is_derived_not_hard_coded():
    """The default follows from the NAMES the caller supplies, so it works for any axis naming.

    A caller who calls their velocity axes ``(vpar, mu)`` gets them as features automatically; a caller who names them
    differently gets the last axis, because the estimator cannot know their intent. That difference is the point of
    passing names at all, and it is asserted here so the default cannot become a hard-coded axis position.
    """
    data = structured_mesh_field(seed=15, shape=(8, 4, 4, 6, 2))

    with_velocity_names = MergeablePCA(n_components=2, axis_names=("tor1", "tor2", "tor3", "vpar", "mu")).fit(
        da.from_array(data, chunks=(4, 4, 4, 6, 2))
    )
    other_names = MergeablePCA(n_components=2, axis_names=("r", "theta", "phi", "u", "w")).fit(
        da.from_array(data, chunks=(4, 4, 4, 6, 2))
    )

    # Named (vpar, mu) -> velocity features, d = 12. Named otherwise -> last axis (w, size 2) as the feature axis.
    assert with_velocity_names.n_features_in_ == 12
    assert other_names.n_features_in_ == 2
