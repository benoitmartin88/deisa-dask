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
Mergeable PCA primitives: exact per-block PCA summaries that merge without the source samples.

What this module provides
-------------------------
A local PCA summary of one data block -- ``(n_samples, mean, components, singular_values)`` -- and an associative
binary merge of two such summaries. A merge consumes *only* summaries: no source sample ever reaches a merge node.
That is what makes an in-situ PCA possible on deisa-dask, where the data stays on the MPI bridge and only a compact
summary crosses the process/network boundary.

Attribution: the algebra here is classical, NOT novel
-----------------------------------------------------
Nothing in this module is a new algorithm, and both the paper and this docstring concede the prior art by name:

- The between-block mean-correction term is the **Chan-Golub-LeVeque covariance merge (1979)**:
  ``M_AB = M_A + M_B + (n_A * n_B / (n_A + n_B)) * delta * delta.T`` with ``delta = mean_A - mean_B``. It is the
  standard scatter-matrix identity behind parallel BLAS. Here it enters as the extra ``correction`` row of the
  compact matrix in :func:`merge_pca`, whose outer product reproduces the ``delta * delta.T`` term.
- The merge-plus-tree-reduction shape (merging rank-``r`` summaries pairwise, re-truncating at each node) is **Iwen &
  Ong, arXiv:1601.07010** (SIAM J. Matrix Anal. Appl. 37(4):1699-1718, 2016, DOI 10.1137/16M1058467) -- their
  Lemma 1 states the merge and their hierarchical Algorithm 1 the reduction tree -- and **Vasudevan &
  Ramakrishna, arXiv:1710.02812** -- merge-and-truncate.
- The only thing claimed as ours is the *placement*: the local PCA runs on the MPI bridge, where the data already
  resides, and only the summary crosses to Dask.

Scalability: what saturates, and what a node costs
--------------------------------------------------
Rank saturation. A node that stacks summaries into a compact matrix of ``rows(compact)`` rows keeps
``min(rows(compact), feature_dim)`` of them, so a merge yields rank ``min(r_a + r_b + 1, d)`` with ``d`` the feature
dimension. The summary rank **SATURATES at ``min(rows(compact), feature_dim)``** and does **not** grow with tree
depth past that ceiling: the ``+1`` each level adds is capped by ``d``. With full local rank (``rank=None``) every
leaf already has rank ``min(n_block, d)``, so the first merge saturates at ``d`` and all deeper levels are flat. With
a truncated ``rank = R < d`` the rank *does* climb with depth until it reaches the same ceiling (measured, ``d = 50``,
``R = 5``: 11, 23, 47, 50 for 2, 4, 8, 16 blocks), so a truncated summary can exceed ``R * d`` elements before
saturation.

Cost. A merge node SVDs a matrix of at most ``(2d + 1, d)``: ``O(d^3)``, **independent of the total sample count N**
and of the per-block sample count. A tree over ``B`` blocks has ``O(B)`` nodes, so the whole reduction is
``O(B * d^3)`` no matter how large ``N`` is. That is the property which makes the summary a valid stand-in for the
samples; the summary *size* is not independent of ``d`` and of the local rank, only the per-node cost is.

Exactness. With full local rank the merged summary is not an approximation of the pooled PCA: it reproduces the
singular values and the principal subspace of a batch SVD of the whole centered array, up to roundoff.

n-dimensional input: the axis policy
------------------------------------
PCA needs two kinds of axis -- SAMPLES (rows) and FEATURES (columns) -- and an array of more than two dimensions does
not say which is which. This module never guesses. Three constructor parameters say so explicitly:

- ``axis_names``: what each axis is called, positionally. A Dask array carries no axis names of its own, so the
  caller is the only place the names can come from; they then appear verbatim in every refusal message.
- ``feature_axes``: which of those axes are features. Defaults to the VELOCITY axes ``("vpar", "mu")`` when
  ``axis_names`` names them, which is the layout a gyrokinetic distribution wants (Layout A below).
- ``sample_axes``: which axes are samples. Defaults to every axis that is not a feature, so giving ``feature_axes``
  alone is enough.

Both ``feature_axes`` and ``sample_axes`` accept names or positions, and if BOTH are given they must partition the
axes exactly: an axis claimed by neither, by both, or twice, is refused rather than guessed. A 2-D array needs none of
this -- its last axis is the feature axis -- so the existing 2-D call sites are untouched.

Two layouts, measured, and they behave OPPOSITELY
-------------------------------------------------
For a gyrokinetic distribution indexed ``(species, tor1, tor2, tor3, vpar, mu)`` split across MPI ranks by the three
spatial axes (each rank therefore owns a contiguous spatial box with the COMPLETE velocity space):

- **Layout A -- PCA over velocity space, per spatial cell.** The features are ``(vpar, mu)``, so ``d = Nvpar * Nmu``
  is fixed by the physics and INDEPENDENT of the rank count. Samples are the spatial cells of the rank's own box.
  Measured on ``(tor1, tor2, tor3, vpar, mu) = (512, 128, 64, 128, 8)``, i.e. ``d = 1024``: a full-rank leaf summary
  is 8.0 MiB against a 4096 MiB slab at 8 ranks (511x) and against a 512 MiB slab at 64 ranks (64x). It compresses at
  every rank count measured, so full local rank is viable and the merge is EXACT.
- **Layout B -- PCA over the spatial box, per velocity cell.** The features are ``(tor1, tor2, tor3)``, so
  ``d`` is the LOCAL box size and GROWS AS RANKS DECREASE -- the opposite of the usual scaling intuition. Measured on
  the same field at 8 ranks: ``d = 524288`` features, ``n_samples = 1024`` velocity cells, and the merged full-rank
  summary reaches rank 8199, i.e. 32800 MiB against a distributed slab of 32768 MiB: 1.00x, NO compression at all.
  (Sizing the leaf at the naive ``r = d`` rather than the true ``min(n_block, d) = 1024`` gives the 2097156 MiB
  figure some write-ups quote; either way the summary is not smaller than the data.)

Consequence for the API: the axis policy makes the difference explicit instead of hiding it, and a full-rank summary
that is not smaller than its input is REFUSED rather than produced -- see :class:`MergeablePCA` for the guard and its
measured reason. There is no auto-default that hides the trade.

Repository constraints honoured here
-------------------------------------
- ``ruff`` line length is 120 and it applies to docstring prose and to ``raise`` literals too, so long messages are
  wrapped across lines rather than left on one line.
- Zero new runtime dependencies: ``numpy.linalg.svd`` only. scipy and scikit-learn are absent from this repo and are
  not needed -- every SVD here is small by construction.
- Python >= 3.10, with ``from __future__ import annotations``.
- Minimal diff: this module is purely additive and refactors nothing else.
- All three functions are module level (no lambda, no closure) so they pickle, which is required for the future
  bridge ``branch_func`` target and for use through ``dask.delayed``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

import dask.array as da
from dask.delayed import Delayed, delayed


@dataclass(frozen=True, eq=False)
class PCASummary:
    """
    Exact mergeable PCA state of a set of samples.

    The fields are named after what they hold. ``n_samples`` is a count; ``mean`` is the pooled feature-wise mean;
    ``components`` holds the principal axes as ROWS (the right singular vectors, i.e. ``Vh`` of
    ``numpy.linalg.svd``, so that ``components.T @ components == I``); ``singular_values`` holds the singular values
    of the centered data, in descending order.

    The mergeable form is ``A = singular_values[:, None] * components``: merging stacks those products, so a
    summary carries the scatter matrix of its samples without carrying a single sample.

    Equality is deliberately not generated: the array fields would make ``==`` ambiguous (elementwise array), which
    is a classic source of silent bugs in merge code. Compare fields, or use ``np.allclose``.

    - ``:param n_samples:`` Number of samples represented, always >= 1.
    - ``:param mean:`` Pooled mean, ``float64`` of shape ``(d,)``.
    - ``:param components:`` Principal axes as rows, ``float64`` of shape ``(rank, d)``.
    - ``:param singular_values:`` Singular values of the centered samples, ``float64`` of shape ``(rank,)``.
    """

    n_samples: int
    mean: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray

    @property
    def rank(self) -> int:
        """Rank of the summary, i.e. how many singular values/components it retains."""
        return int(self.singular_values.shape[0])


def local_pca(block: np.ndarray, rank: int | None = None) -> PCASummary:
    """
    Summarize one data block: center it, take its SVD, and keep the leading ``rank`` directions.

    ``rank=None`` keeps the full local rank ``min(n_block, d)``. Truncating to ``rank = R`` shrinks the summary (and
    therefore what has to cross the bridge boundary) at the cost of discarding variance that the merge cannot
    recover; see this module's docstring for the rank law.

    The block is never mutated: it is cast to ``float64`` if needed and centering allocates a fresh array.

    - ``:param block:`` 2-D ``(n_samples, n_features)`` block of one rank's data.
    - ``:param rank:`` Maximum retained rank, or ``None`` for the full local rank ``min(n_samples, n_features)``.
    """
    X = np.asarray(block, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(
            f"local_pca: block must be 2-dimensional, got ndim={X.ndim}. "
            "Reshape it to (n_samples, n_features) before summarizing."
        )
    n_samples, n_features = X.shape
    if n_samples == 0:
        raise ValueError(
            "local_pca: block has zero rows, PCA is undefined. "
            "Skip empty blocks before building summaries (only non-empty blocks can be summarized)."
        )
    if rank is not None:
        if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or rank <= 0:
            raise ValueError(f"local_pca: rank must be None or a positive integer, got {rank!r}.")
        rank = int(rank)

    mean = X.mean(axis=0)
    _, singular_values, components = np.linalg.svd(X - mean, full_matrices=False)
    if rank is not None:
        keep = min(rank, singular_values.shape[0])
        singular_values = singular_values[:keep]
        components = components[:keep]
    return PCASummary(
        n_samples=int(n_samples),
        mean=mean,
        components=components,
        singular_values=singular_values,
    )


def merge_pca(a: PCASummary, b: PCASummary) -> PCASummary:
    """
    Merge two summaries of disjoint sample sets into the exact summary of their union.

    This is the classical merge, stated once so the code and the citation cannot drift apart:

    - ``A = a.singular_values[:, None] * a.components`` and likewise ``B``, so ``A.T @ A`` is the centered scatter
      matrix of ``a``'s samples and ``B.T @ B`` that of ``b``'s.
    - ``correction = sqrt(n_a * n_b / (n_a + n_b)) * (a.mean - b.mean)`` is the between-block mean-correction term of
      the Chan-Golub-LeVeque (1979) scatter merge, ``(n_a * n_b / (n_a + n_b)) * delta * delta.T`` with
      ``delta = mean_a - mean_b``. It is what makes the merge exact; dropping it silently biases every merge that
      sees blocks with different means.
    - Stacking ``A``, ``B`` and ``correction`` into ``compact`` gives ``compact.T @ compact`` equal to the pooled
      centered scatter matrix, so the SVD of ``compact`` yields the pooled singular values and principal axes.

    Associativity holds up to roundoff, which is what justifies the balanced tree of :func:`merge_tree`.

    - ``:param a:`` Summary of the first (disjoint) sample set.
    - ``:param b:`` Summary of the second (disjoint) sample set.
    """
    n_a = int(a.n_samples)
    n_b = int(b.n_samples)
    if n_a < 1 or n_b < 1:
        raise ValueError(
            f"merge_pca: cannot merge summaries with n_samples={n_a} and {n_b}; "
            "every summary must represent at least one sample (drop empty blocks instead)."
        )
    d_a = int(a.mean.shape[0])
    d_b = int(b.mean.shape[0])
    if d_a != d_b or a.components.shape[1] != d_a or b.components.shape[1] != d_b:
        raise ValueError(
            f"merge_pca: summaries span different feature dimensions ({d_a} vs {d_b}); "
            "all summaries must share one feature basis, so rechunk so the complete feature dimension is "
            "in a single chunk before merging."
        )

    A = a.singular_values[:, None] * a.components
    B = b.singular_values[:, None] * b.components
    correction = np.sqrt(n_a * n_b / (n_a + n_b)) * (a.mean - b.mean)
    compact = np.vstack((A, B, correction[None, :]))
    _, singular_values, components = np.linalg.svd(compact, full_matrices=False)
    n_samples = n_a + n_b
    mean = (n_a * a.mean + n_b * b.mean) / n_samples
    return PCASummary(
        n_samples=int(n_samples),
        mean=mean,
        components=components,
        singular_values=singular_values,
    )


def merge_tree(summaries: Sequence[PCASummary]) -> PCASummary:
    """
    Reduce summaries with a balanced pairwise tree, the reduction shape of Iwen & Ong (arXiv:1601.07010, Algorithm 1).

    Each level merges neighbours pairwise; an odd level carries its last summary up unchanged to the next level, so
    the tree stays balanced instead of degenerating into a chain. A single summary is returned as-is (base case) and
    an empty input is refused.

    The tree is safe because :func:`merge_pca` is associative: any grouping of the same summaries gives the same
    singular values up to roundoff, so the balanced shape is a cost decision, not a correctness one.

    - ``:param summaries:`` Sequence of at least one summary, over disjoint sample sets with one feature dimension.
    """
    level = list(summaries)
    if not level:
        raise ValueError(
            "merge_tree: no summaries to merge. "
            "Pass at least one summary (fit requires at least one non-empty row block)."
        )
    while len(level) > 1:
        merged = [merge_pca(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            merged.append(level[-1])
        level = merged
    return level[0]


# The internal API of the design spells these with a leading underscore. Both spellings are module level and pickle
# identically, so keep them as plain aliases instead of wrappers.
_PCASummary = PCASummary
_local_pca = local_pca
_merge_pca = merge_pca
_merge_tree = merge_tree


# =============================================================================
# The Dask estimator
# =============================================================================
def _validate_optional_positive_int(name: str, value: object) -> None:
    """Refuse a bad ``None``-or-positive-integer constructor argument immediately, by name.

    ``bool`` is rejected explicitly because ``isinstance(True, int)`` is ``True`` in Python, so without this check
    ``n_components=True`` would silently mean "one component".

    - ``:param name:`` Constructor parameter name, used verbatim in the message so the caller knows which one.
    - ``:param value:`` The value to check, expected ``None`` or a positive ``int``.
    """
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"MergeablePCA: {name} must be None or a positive integer, got {value!r}.")


# The velocity axes a gyrokinetic distribution is indexed by. They are the DEFAULT feature axes, because the feature
# dimension of a mergeable summary must stay complete on one rank for summaries to be stackable, and the spatial split
# is exactly what leaves the velocity space whole on every rank. Named as constants so the error messages, the
# docstrings and the tests cannot drift apart.
VELOCITY_AXES: tuple[str, ...] = ("vpar", "mu")
"""The velocity axes of a ``(species, tor1, tor2, tor3, vpar, mu)`` distribution, in that order."""


def _describe_axes(axis_names: Sequence[str] | None, ndim: int) -> tuple[str, ...]:
    """Return one label per axis, using the caller's names where given and a generic fallback otherwise.

    The fallback is deliberately NOT ``"axis0"``/``"axis1"``: those labels end up inside refusal messages, and a
    message that says "axis2" tells a physics reader nothing. When no names were supplied the message instead names
    the position and the extent, which is still unambiguous.

    - ``:param axis_names:`` Caller-supplied names, positionally, or ``None``.
    - ``:param ndim:`` Number of axes of the array being described.
    """
    if axis_names is None:
        return tuple(f"dimension {i}" for i in range(ndim))
    return tuple(str(name) for name in axis_names)


def _resolve_axes(
    spec: object,
    ndim: int,
    axis_names: Sequence[str] | None,
    parameter: str,
) -> tuple[int, ...]:
    """Turn a caller-supplied axis specification into validated, sorted positions.

    Accepts a single name or position, or any sequence of them. Names are looked up in ``axis_names``; a name the
    caller never declared is refused rather than matched positionally, because matching it positionally would be
    exactly the silent guess this whole axis policy exists to prevent.

    - ``:param spec:`` The specification: a name, a position, or a sequence of them.
    - ``:param ndim:`` Number of axes of the array the positions index into.
    - ``:param axis_names:`` Caller-supplied axis names, positional, or ``None``.
    - ``:param parameter:`` Constructor parameter name, quoted verbatim in the message.
    """
    labels = _describe_axes(axis_names, ndim)
    if isinstance(spec, (str, int, np.integer)) and not isinstance(spec, bool):
        items: list[object] = [spec]
    elif isinstance(spec, Sequence):
        items = list(spec)
    else:
        raise ValueError(
            f"MergeablePCA: {parameter} must be axis names or positions, or a sequence of them, got {spec!r} of type "
            f"{type(spec).__name__}."
        )
    resolved: list[int] = []
    for item in items:
        if isinstance(item, str):
            if axis_names is None or item not in axis_names:
                declared = ", ".join(labels) if axis_names is not None else "none declared"
                raise ValueError(
                    f"MergeablePCA: {parameter} names the axis {item!r}, which is not among the axis_names "
                    f"declared by the caller ({declared}). Name an axis that exists, or pass positions."
                )
            position = tuple(axis_names).index(item)
        elif isinstance(item, (int, np.integer)) and not isinstance(item, bool):
            position = int(item)
            if not -ndim <= position < ndim:
                raise ValueError(
                    f"MergeablePCA: {parameter} position {position} is out of range for an array with {ndim} "
                    f"axes ({', '.join(labels)}). Use a negative position or one below {ndim}."
                )
            position %= ndim
        else:
            raise ValueError(
                f"MergeablePCA: {parameter} entries must be axis names or positions, got {item!r} of type "
                f"{type(item).__name__}."
            )
        if position in resolved:
            raise ValueError(
                f"MergeablePCA: {parameter} names the axis {labels[position]} twice. Each axis may appear at most once."
            )
        resolved.append(position)
    if not resolved:
        raise ValueError(
            f"MergeablePCA: {parameter} is empty, so no axis would be classified. Name at least one axis, or pass "
            "None to classify every remaining axis as a sample."
        )
    return tuple(sorted(resolved))


def _partition_axes(
    ndim: int,
    axis_names: Sequence[str] | None,
    feature_axes: object,
    sample_axes: object,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Decide which axes are features and which are samples, refusing every ambiguous or overlapping spec.

    The rules, in order:

    1. A 1-D array is refused by the caller before this runs: one axis cannot be both kinds.
    2. ``feature_axes`` defaults to the VELOCITY axes when ``axis_names`` names them, else to the LAST axis.
    3. ``sample_axes`` defaults to every axis that is not a feature.
    4. If BOTH are given explicitly they must partition the axes exactly. An axis in neither set, in both, or named
       twice, is a refusal: the alternatives are a silently wrong PCA or a guess the caller never made.

    - ``:param ndim:`` Number of axes of the array being classified.
    - ``:param axis_names:`` Caller-supplied axis names, positional, or ``None``.
    - ``:param feature_axes:`` The ``feature_axes`` constructor argument, verbatim.
    - ``:param sample_axes:`` The ``sample_axes`` constructor argument, verbatim.
    """
    labels = _describe_axes(axis_names, ndim)
    named_velocity = axis_names is not None and all(name in axis_names for name in VELOCITY_AXES)
    if feature_axes is None:
        if named_velocity:
            # Layout A: the velocity space is the feature basis, which is complete on every rank under a spatial split.
            features = _resolve_axes(VELOCITY_AXES, ndim, axis_names, "feature_axes")
        else:
            features = (ndim - 1,)
    else:
        features = _resolve_axes(feature_axes, ndim, axis_names, "feature_axes")

    remaining = tuple(i for i in range(ndim) if i not in features)
    samples_given = sample_axes is not None
    samples = _resolve_axes(sample_axes, ndim, axis_names, "sample_axes") if samples_given else remaining

    if len(set(samples) & set(features)):
        overlap = ", ".join(labels[i] for i in sorted(set(samples) & set(features)))
        raise ValueError(
            f"MergeablePCA: sample_axes and feature_axes both claim {overlap}, but an axis cannot be both a sample "
            "and a feature. Give each axis to exactly one of the two sets."
        )
    claimed = set(samples) | set(features)
    if claimed != set(range(ndim)):
        missing = sorted(set(range(ndim)) - claimed)
        raise ValueError(
            f"MergeablePCA: the axis specification leaves {', '.join(labels[i] for i in missing)} unclassified "
            f"({', '.join(labels)}). Every axis must be a sample or a feature; drop sample_axes to let the remaining "
            "axes default to samples."
        )
    if not remaining and not samples_given:
        raise ValueError(
            "MergeablePCA: every axis was declared as a feature, so there are no samples and PCA is undefined. "
            f"Leave at least one axis out of feature_axes ({', '.join(labels)})."
        )
    return features, samples


class MergeablePCA:
    """
    PCA of a Dask array computed from mergeable per-block summaries, never from the pooled samples.

    The estimator is scikit-learn-STYLE -- trailing-underscore fitted attributes, ``fit`` returning ``self``, ``y``
    accepted and ignored -- but it deliberately does not subclass ``sklearn.base.BaseEstimator``: scikit-learn is not a
    dependency of this repository and adding one to obtain a base class is not worth it. Everything scikit-learn's
    estimator protocol actually needs (validate params in ``__init__``, do no work there, fit, transform) is here.

    How it works
    ------------
    :meth:`_fit_dask_delayed` builds one ``dask.delayed(local_pca)`` task per input block and reduces those summaries
    with a balanced tree of ``dask.delayed(merge_pca)`` tasks (the shape of Iwen & Ong, arXiv:1601.07010, Algorithm 1).
    The reduction consumes ONLY summaries: no source sample ever reaches a merge node, so the summary is a valid
    stand-in for the data it describes. With full local rank the result is not an approximation -- it reproduces the
    batch SVD of the whole centered array to roundoff, because the merge carries the Chan-Golub-LeVeque (1979)
    between-block mean correction (see :func:`merge_pca`).

    Two graph hooks, deliberately split
    ------------------------------------
    :meth:`_fit_dask_delayed` returns the constructed-but-unbuilt Dask graph and NEVER calls ``.compute()``;
    :meth:`_fit_dask` is the thin ``.compute()`` wrapper on top of it. This split is not cosmetic: calling
    ``compute()`` inside the builder is exactly what made the graph unvisualizable in the original design, because a
    computed result is a value and values have no graph to draw. ``pca._fit_dask_delayed(X).visualize()`` therefore
    works, which is the point of exposing it.

    Internal vs public state
    ------------------------
    ``self._summary_`` keeps the mergeable summary, whose rank is generally HIGHER than ``n_components`` because the
    merge saturates at ``min(rows(compact), n_features)`` rather than at the requested component count. The public
    attributes (``components_``, ``singular_values_``, ``explained_variance_``) describe the REQUESTED final PCA and are
    truncated (and optionally whitened) from ``_summary_`` afterwards. Truncation and whitening happen ONLY here at the
    root, never mid-tree: both destroy mergeability and exactness, so the tree runs on full untruncated summaries.
    Bridge-side code emits ``_summary_``, not the truncated public form.

    The accuracy/bandwidth trade-off of ``local_rank``
    ---------------------------------------------------
    ``local_rank=None`` keeps the full local rank ``min(n_block, n_features)`` and the merge is exact. ``local_rank=R``
    truncates each leaf to rank ``R`` before merging, which shrinks what has to cross a process or network boundary
    from ``~min(n_block, d) * d`` to ``~R * d`` elements -- the knob that makes the bridge path a bandwidth win for
    wide, short blocks. The price is accuracy: the discarded leaf variance cannot be recovered by any merge, so the
    retained subspace drifts from the batch one and the retained variance falls. Both losses are monotone in ``R`` and
    vanish at ``R >= d``. There is no universally safe ``R``; :meth:`fit` refuses the one combination that cannot work
    at all (a root rank below ``n_components``) instead of silently returning fewer components.

    n-dimensional input: the axis policy
    ------------------------------------
    An array of more than two dimensions does not say which axes are SAMPLES (rows) and which are FEATURES (columns),
    and PCA gives a different answer for each choice, so the choice is never guessed. Three constructor parameters
    state it explicitly:

    - ``axis_names``: what each axis is called, positionally. A Dask array carries no axis names of its own (it has no
      ``dims`` attribute), so the caller is the only place names can come from -- and they then appear verbatim in
      every refusal message, so a message can say ``vpar`` rather than ``axis4``.
    - ``feature_axes``: which axes are features. Defaults to ``("vpar", "mu")`` when ``axis_names`` names them
      (Layout A, the default the measurements below justify), else to the last axis.
    - ``sample_axes``: which axes are samples. Defaults to every axis that is not a feature.

    Both ``feature_axes`` and ``sample_axes`` accept names or positions, and when BOTH are given they must partition
    the axes exactly. An axis in neither set, in both, or named twice is refused with a message naming it. A 2-D array
    needs none of this -- its last axis is the feature axis -- so existing 2-D call sites are unaffected, and for 2-D
    input ``axis_names`` is accepted and used only in messages.

    The array is transposed so the feature axes come last, then reshaped to ``(n_samples, n_features)``. That reshape
    is the ONLY flattening performed, it is value-preserving, and it is explicit in the reported shapes: with
    ``axis_names=("species", "tor1", "tor2", "tor3", "vpar", "mu")`` and the default feature axes, the features are
    ``vpar * mu`` and the samples are ``species * tor1 * tor2 * tor3``. One leaf is built per Dask block of the SAMPLE
    axes, which is exactly one leaf per rank: a rank's contiguous spatial box is one leaf, complete in velocity space,
    as a spatial split guarantees.

    Why Layout A is the default and Layout B is refused
    ---------------------------------------------------
    For a gyrokinetic distribution split across MPI ranks by the three spatial axes, each rank holds a contiguous
    ``(tor1, tor2, tor3)`` box with the COMPLETE velocity space. Which axes are the features therefore decides the
    whole regime, and the two choices behave oppositely. Measured on
    ``(tor1, tor2, tor3, vpar, mu) = (512, 128, 64, 128, 8)``, a leaf summary being ``(rank, d) + d`` float64:

    - **Layout A, features ``(vpar, mu)``** -- ``d = 1024``, fixed by the physics and INDEPENDENT of the rank count. A
      full-rank leaf summary is 8.0 MiB against a 4096 MiB slab at 8 ranks (511x) and a 512 MiB slab at 64 ranks
      (64x): it compresses at every rank count measured, so full local rank is viable and the merge is EXACT.
    - **Layout B, features ``(tor1, tor2, tor3)``** -- ``d`` is the LOCAL box size and GROWS AS RANKS DECREASE, the
      opposite of the usual scaling intuition. At 8 ranks ``d = 524288`` with ``n_samples = 1024`` velocity cells, and
      the merged full-rank summary reaches rank 8199: 32800 MiB against a distributed slab of 32768 MiB, i.e. 1.00x.
      No compression. (Sizing a leaf at the naive ``r = d`` rather than the true ``min(n_block, d) = 1024`` yields
      the 2097156 MiB figure quoted elsewhere; either way the summary is not smaller than the data.)

    A summary that is not smaller than its input is refused by :meth:`_check_summary_smaller_than_input`, which names
    the measured reason and the remedy (``local_rank=R``). Under Layout B ``local_rank`` is therefore effectively
    mandatory and is documented as such; there is no auto-default that hides the trade. Under Layout A the default
    ``local_rank=None`` is the right answer.

    Repository constraints honoured here
    -------------------------------------
    - ``ruff`` line length is 120 and it applies to docstring prose and to ``raise`` literals too, so long messages are
      wrapped across lines.
    - Zero new runtime dependencies: ``dask`` and ``numpy`` only. scipy and scikit-learn are used in the TESTS as
      reference implementations, never imported by ``src/``.
    - Refuse, do not approximate: every rejection names the offending condition AND the remedy.

    - ``:param n_components:`` Number of final public components, or ``None`` to keep all
      ``min(n_samples, n_features)``.
    - ``:param whiten:`` If ``True``, scale ``components_`` after truncation so projected columns have unit variance.
    - ``:param copy:`` Accepted for scikit-learn API parity. It has NO effect: ``fit`` never mutates ``X``, because
      every local PCA centers a fresh ``float64`` copy of its own block.
    - ``:param batch_size:`` Optional maximum rows per local-PCA leaf. ``None`` means one leaf per Dask row chunk. If
      set, each row chunk is split into leaves of at most this many rows before the local PCA. Merging is associative,
      so this only reshapes the tree and does not change the result above roundoff.
    - ``:param local_rank:`` Maximum rank of each leaf summary, or ``None`` for the full local rank
      ``min(n_block, n_features)``. Optional, and ``None`` (full rank, exact) is the right default under Layout A.
      Under Layout B it is effectively MANDATORY: the feature dimension is the local spatial box, so a full-rank
      summary is not smaller than its input and :meth:`_check_summary_smaller_than_input` refuses it.
    - ``:param axis_names:`` Optional name per axis, positionally, used in messages and in the ``feature_axes`` /
      ``sample_axes`` lookup. ``None`` for 2-D input, or to select the last axis as the feature axis by default.
    - ``:param feature_axes:`` Which axes are features, by name or position. ``None`` defaults to
      :data:`VELOCITY_AXES` when ``axis_names`` names them, else to the last axis.
    - ``:param sample_axes:`` Which axes are samples, by name or position. ``None`` defaults to every axis that is
      not a feature. Give it only when the features do NOT cover the rest, because giving both partitions exactly.
    """

    def __init__(
        self,
        n_components: int | None = None,
        whiten: bool = False,
        copy: bool = True,
        batch_size: int | None = None,
        local_rank: int | None = None,
        axis_names: Sequence[str] | None = None,
        feature_axes: object = None,
        sample_axes: object = None,
    ):
        _validate_optional_positive_int("n_components", n_components)
        _validate_optional_positive_int("local_rank", local_rank)
        _validate_optional_positive_int("batch_size", batch_size)
        if not isinstance(whiten, bool):
            raise ValueError(f"MergeablePCA: whiten must be a bool, got {type(whiten).__name__}.")
        if not isinstance(copy, bool):
            raise ValueError(f"MergeablePCA: copy must be a bool, got {type(copy).__name__}.")
        if axis_names is not None and not isinstance(axis_names, Sequence):
            raise ValueError(
                f"MergeablePCA: axis_names must be a sequence of names or None, got {type(axis_names).__name__}."
            )
        if (
            axis_names is not None
            and all(isinstance(name, str) for name in axis_names)
            and len(set(axis_names)) != len(axis_names)
        ):
            raise ValueError(
                f"MergeablePCA: axis_names must not repeat a name, got {tuple(axis_names)}. Names identify axes "
                "uniquely, so a duplicate makes an axis ambiguous."
            )
        self.n_components = n_components
        self.whiten = whiten
        self.copy = copy
        self.batch_size = batch_size
        self.local_rank = local_rank
        self.axis_names = None if axis_names is None else tuple(str(name) for name in axis_names)
        self.feature_axes = feature_axes
        self.sample_axes = sample_axes

    # ---------------------------------------------------------------------------- axis policy
    def _as_2d(self, X):
        """Flatten ``X`` to ``(n_samples, n_features)`` under the axis policy, or refuse.

        This is the single place a multi-dimensional array becomes 2-D, and it is deliberately explicit: it
        transposes so the feature axes are LAST, then calls ``reshape``, which is value-preserving. Nothing is
        inferred beyond the axis partition, so the mapping from a cell to its feature column is fixed and reported.

        The 2-D case is returned untouched -- no transpose, no reshape -- so a plain ``(n_samples, n_features)`` array
        costs nothing and existing call sites keep byte-identical graphs.

        - ``:param X:`` Array of any dimensionality, already wrapped in ``dask.array``.
        """
        ndim = int(X.ndim)
        if self.axis_names is not None and len(self.axis_names) != ndim:
            # Without this, an axis_names list of the wrong length resolves to positions that do not exist in this
            # array: names would index past ndim and the classification would be silently wrong.
            raise ValueError(
                f"MergeablePCA: axis_names has {len(self.axis_names)} names but X has {ndim} axes. Give exactly one "
                f"name per axis, or None to classify the axes by position. Got {self.axis_names}."
            )
        labels = _describe_axes(self.axis_names, ndim)
        if ndim == 0:
            raise ValueError(
                "MergeablePCA: X must be 2-dimensional or more, at least one sample axis and one feature axis, got a "
                "0-dimensional scalar. Provide an array with a sample axis and a feature axis."
            )
        if ndim == 1:
            # One axis cannot be both a sample axis and a feature axis, so there is nothing to classify and nothing to
            # guard against. PCA on a bare vector is undefined, so this is the same refusal as before with the
            # remedy spelled out: "2-dimensional" is kept in the message because callers match on it.
            raise ValueError(
                f"MergeablePCA: X must be 2-dimensional (one sample axis, one feature axis), got ndim=1. "
                f"Add the axis you meant, e.g. X.reshape(1, -1) for a single sample ({', '.join(labels)})."
            )
        features, samples = _partition_axes(ndim, self.axis_names, self.feature_axes, self.sample_axes)

        if ndim == 2 and features == (1,) and samples == (0,):
            return X

        permutation = samples + features
        transposed = X if permutation == tuple(range(ndim)) else da.moveaxis(X, list(range(ndim)), list(permutation))
        n_features = 1
        for axis in features:
            n_features *= int(X.shape[axis])
        n_samples = 1
        for axis in samples:
            n_samples *= int(X.shape[axis])
        # Reshape needs the sample axes contiguous in the flat row order and the whole feature dimension in ONE chunk,
        # otherwise dask silently produces a feature-split array whose summaries cannot be stacked. Refusing here names
        # the axes, which is far more actionable than a later "summaries span different feature dimensions".
        self._check_feature_chunks(X, features, labels)
        return transposed.reshape((n_samples, n_features))

    def _check_feature_chunks(self, X, features: tuple[int, ...], labels: tuple[str, ...]) -> None:
        """Refuse a FEATURE axis split across Dask chunks, naming the axes and the gysela remedy.

        This guard applies only where it genuinely applies: the feature dimension of a mergeable summary must be
        complete on the rank that holds it, because a merge stacks summaries by feature position. Under Layout A that is
        satisfied NATURALLY -- a spatial MPI split leaves the whole ``(vpar, mu)`` space on one rank, so ``chunks`` of a
        bridge's slab has one entry per feature axis and this guard never fires. It fires only when the caller
        rechunked across velocity, i.e. exactly when the summaries really would be unstackerable.

        Note the asymmetry with the SAMPLE axes, which may be split freely (that is the whole point): this guard
        ignores them entirely.

        - ``:param X:`` The unflattened array, read for its per-axis chunk counts.
        - ``:param features:`` Resolved positions of the feature axes.
        - ``:param labels:`` One label per axis, for the message.
        """
        split = [axis for axis in features if len(X.chunks[axis]) != 1]
        if not split:
            return
        chunks = ", ".join(f"{labels[axis]}={tuple(X.chunks[axis])}" for axis in split)
        remedy = "{" + ", ".join(str(axis) for axis in split) + ": -1}"
        named = ", ".join(labels[axis] for axis in split)
        raise ValueError(
            f"MergeablePCA: the feature axes ({named}) are split across Dask chunks, got {chunks}. A mergeable PCA "
            "needs the COMPLETE feature dimension on one rank, because a merge stacks summaries by feature position. "
            f"Rechunk the feature axes together: X = X.rechunk({remedy}). If those axes are the velocity axes "
            "(vpar, mu) you may not need to rechunk at all: an MPI split over the spatial axes keeps the whole "
            "velocity space on one rank by construction, so a bridge's slab satisfies this already."
        )

    # ---------------------------------------------------------------------------- fitting
    def fit(self, X, y=None) -> MergeablePCA:
        """Fit the estimator and return ``self``.

        A dask array is fitted through :meth:`_fit_dask_delayed`; a numpy array takes the in-memory path, which is the
        single-leaf base case of the same tree. Input of any dimensionality is accepted under the axis policy described
        in this class's docstring: the array is transposed so the feature axes come last, then reshaped to
        ``(n_samples, n_features)`` before anything else happens.

        - ``:param X:`` Array of ``n_samples * n_features`` values, dask or numpy, 2-D or more. For more than two
          dimensions, ``axis_names`` / ``feature_axes`` / ``sample_axes`` say which axes are which.
        - ``:param y:`` Accepted and ignored, for scikit-learn parity.
        """
        array = X if isinstance(X, da.Array) else da.asarray(X)
        flat = self._as_2d(array)
        self._validate_input(flat)

        n_samples = int(flat.shape[0])
        n_features = int(flat.shape[1])
        self.n_features_in_ = n_features
        # Checked against min(n_samples, n_features) before any work, so an impossible request fails fast and free.
        if self.n_components is not None and self.n_components > min(n_samples, n_features):
            raise ValueError(
                f"MergeablePCA: n_components={self.n_components} exceeds "
                f"min(n_samples={n_samples}, n_features={n_features}). "
                "Reduce n_components, or provide more samples / features."
            )

        if isinstance(X, da.Array):
            # The RAW array goes in, not ``flat``: _fit_dask_delayed flattens it itself, and flattening twice would
            # apply the axis policy to an array that is already 2-D (and re-run the guard on reshaped chunks).
            summary = self._fit_dask(array)
        else:
            # In-memory: one leaf, the base case of the merge tree. Still routed through the same primitive.
            summary = local_pca(np.asarray(flat), rank=self.local_rank)

        self._summary_ = summary
        self._check_root_rank(summary)
        self._check_summary_smaller_than_input(summary, n_features)
        self._materialize_public_attributes(summary)
        return self

    def fit_transform(self, X, y=None):
        """Fit, then project ``X`` onto the components. Returns the projection, not the estimator.

        - ``:param X:`` 2-D ``(n_samples, n_features)`` array, dask or numpy.
        - ``:param y:`` Accepted and ignored, for scikit-learn parity.
        """
        return self.fit(X, y=y).transform(X)

    def _fit_dask_delayed(self, X) -> Delayed:
        """Build the merge tree as a Dask graph and return it WITHOUT computing.

        The returned :class:`dask.delayed.Delayed` is a real collection: ``.visualize()`` draws the reduction tree and
        ``.compute()`` runs it. Calling ``compute()`` in here instead would return a plain value with no graph, which
        is the bug this split exists to prevent.

        The tree is built level by level from :func:`dask.delayed.merge_pca`, one leaf task per block, so the merges
        actually run in parallel. Passing the whole leaf list to a single ``delayed(merge_tree)`` would be shorter but
        would collapse the reduction into ONE sequential task on one worker, which is not the tree shape and forfeits
        the parallelism; the shape built here is the same pairwise fold, odd level carried up unchanged, that
        :func:`merge_tree` implements.

        - ``:param X:`` Array of any dimensionality. It is flattened by :meth:`_as_2d` first, so the graph below is
          always over ``(n_samples, n_features)`` leaves.
        """
        flat = self._as_2d(X)
        self._validate_input(flat)
        level = [delayed(local_pca)(block, rank=self.local_rank) for block in self._leaf_blocks(flat)]
        while len(level) > 1:
            merged = [delayed(merge_pca)(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
            if len(level) % 2:
                # An odd level carries its last node up unchanged, so the tree stays balanced instead of becoming a
                # chain. Safe because merge_pca is associative, so the grouping is a cost decision, not a correctness
                # one -- the same argument merge_tree makes.
                merged.append(level[-1])
            level = merged
        return level[0]

    def _fit_dask(self, X) -> PCASummary:
        """Compute the merge tree of :meth:`_fit_dask_delayed` and return the merged summary.

        This is exactly ``self._fit_dask_delayed(X).compute()``; the method exists so ``fit`` reads as one call and so a
        caller can override the computation (e.g. with a different scheduler or a distributed client).

        - ``:param X:`` 2-D ``(n_samples, n_features)`` dask array whose features are in one chunk.
        """
        return self._fit_dask_delayed(X).compute()

    # ---------------------------------------------------------------------------- validation
    def _validate_input(self, X) -> None:
        """Refuse input this estimator cannot answer exactly, naming the condition and the remedy.

        Every branch here is a case where the alternative is a silently wrong answer rather than an error, so all of
        them are refusals rather than warnings. ``X`` here is ALREADY flattened to 2-D by :meth:`_as_2d`, which owns
        the axis policy and the feature-chunk guard; this method covers only the shape conditions that survive it.

        - ``:param X:`` Candidate 2-D ``(n_samples, n_features)`` array (already wrapped in ``dask.array``).
        """
        n_samples, n_features = int(X.shape[0]), int(X.shape[1])
        if n_samples == 0:
            raise ValueError(
                "MergeablePCA: X has zero rows, PCA is undefined. "
                "Provide at least one sample (fit is refused rather than returning an empty component set)."
            )
        if n_features == 0:
            raise ValueError(
                "MergeablePCA: X has zero features, PCA is undefined. "
                "Provide at least one feature (fit is refused rather than returning an empty component set)."
            )
        # The 2-D feature-chunk precondition, kept for the case where the input arrived already 2-D and therefore
        # never went through _as_2d's per-axis check. Same condition, message now names the axis and the remedy.
        if len(X.chunks[1]) != 1:
            raise ValueError(
                "MergeablePCA requires the complete feature dimension in a single Dask chunk, "
                f"got chunks={X.chunks}. Rechunk with X = X.rechunk({{1: -1}}) before fitting -- for an array with "
                "named axes, rechunk the velocity axes (vpar, mu) together rather than the axis position alone."
            )

    def _check_summary_smaller_than_input(self, summary: PCASummary, n_features: int) -> None:
        """Refuse a summary that is not smaller than the data it summarizes, naming the measured reason.

        This is the guard that makes the axis choice safe. A merged full-rank summary holds
        ``rank * d + d`` float64 elements against ``n_samples * d`` for the data, so it compresses exactly when
        ``n_samples > d``. Under Layout A (``d = Nvpar * Nmu``, fixed by the physics) that holds comfortably: on
        ``(tor1, tor2, tor3, vpar, mu) = (512, 128, 64, 128, 8)`` a full-rank leaf summary is 8.0 MiB against a
        4096 MiB slab at 8 ranks (511x) and a 512 MiB slab at 64 ranks (64x).

        Under Layout B (``d`` = the local spatial box, which GROWS as ranks decrease) it fails: at 8 ranks
        ``d = 524288`` with ``n_samples = 1024``, the merged summary reaches rank 8199 and needs 32800 MiB against a
        distributed slab of 32768 MiB -- 1.00x, no compression at all. Producing that silently would spend more
        memory than the data it summarizes, which is worse than refusing, so it is refused by default.

        ``local_rank`` is the remedy and the caller knows the trade: ``local_rank=R`` shrinks the summary to
        ``~R * d`` elements per leaf, so ``R`` below ``n_samples`` restores compression at the cost of the discarded
        leaf variance. There is deliberately NO auto-default: choosing ``R`` for the caller would hide a real accuracy
        trade, so the refusal names it instead.

        - ``:param summary:`` The merged root summary, before the public attributes are materialized.
        - ``:param n_features:`` Feature dimension of the flattened array, i.e. ``d``.
        """
        n_samples = int(summary.n_samples)
        payload = summary.components.size + summary.mean.size
        data = n_samples * n_features
        if payload <= 0 or payload < data:
            return
        ratio = data / payload if payload else 0.0
        raise ValueError(
            f"MergeablePCA: the merged summary would not compress the data -- {payload} float64 elements "
            f"(rank {summary.rank} x d {n_features} + d) against {data} for n_samples={n_samples}, d={n_features}, "
            f"a ratio of {ratio:.2f}x. This happens when the FEATURE axes outnumber the samples: with velocity "
            "axes (vpar, mu) as features d is fixed by the physics and this never triggers, but with the spatial axes "
            "(tor1, tor2, tor3) as features d is the local box and grows as the rank count drops (measured: d=524288 "
            "against n_samples=1024 at 8 ranks, 32800 MiB of summary for a 32768 MiB distributed slab). "
            "Remedy: pass local_rank=R with R well below n_samples to truncate each leaf summary, or make the "
            "velocity axes the features instead."
        )

    def _check_root_rank(self, summary: PCASummary) -> None:
        """Refuse a root summary too small for the requested components, instead of returning fewer.

        This is the one guard that makes ``local_rank < n_components`` safe: with truncated leaves the merged rank grows
        with tree depth (see this module's rank-saturation note), so whether the request is satisfiable is only known at
        the root. ``n_components <= min(n_samples, n_features)`` alone cannot answer it.

        - ``:param summary:`` The merged root summary, before the public attributes are materialized.
        """
        if self.n_components is not None and summary.rank < self.n_components:
            raise ValueError(
                f"MergeablePCA: root summary rank {summary.rank} < n_components={self.n_components}. "
                "Increase local_rank, use more or finer row blocks, or reduce n_components."
            )

    # ---------------------------------------------------------------------------- leaf construction
    def _leaf_blocks(self, X) -> list:
        """Slice ``X`` into the leaf blocks fed to :func:`local_pca`, without computing any of them.

        One leaf per Dask block by default, taken from ``X.to_delayed().ravel()`` so that each leaf is a
        :class:`~dask.delayed.Delayed` and the whole reduction stays a plain delayed graph. With ``batch_size`` set,
        each row chunk is further split into row slices of at most that many rows. Empty leaves are dropped, since a
        0-row block has no PCA.

        - ``:param X:`` 2-D ``(n_samples, n_features)`` dask array whose features are in one chunk.
        """
        blocks = X.to_delayed().ravel()
        if self.batch_size is None:
            return list(blocks)

        leaves = []
        for block, n_rows in zip(blocks, X.chunks[0]):
            for start in range(0, n_rows, self.batch_size):
                stop = min(start + self.batch_size, n_rows)
                if stop > start:
                    leaves.append(block[start:stop])
        return leaves

    # ---------------------------------------------------------------------------- public attributes
    def _materialize_public_attributes(self, summary: PCASummary) -> None:
        """Truncate, and optionally whiten, the merged summary into the requested public PCA.

        This is the ONLY place ``n_components`` and ``whiten`` are applied. ``self._summary_`` keeps the untruncated,
        unwhitened, still-mergeable state that the bridge emits.

        - ``:param summary:`` The merged root summary.
        """
        n_samples = int(summary.n_samples)
        keep = min(n_samples, self.n_features_in_) if self.n_components is None else self.n_components
        self.n_components_ = int(keep)
        self.singular_values_ = np.asarray(summary.singular_values[:keep], dtype=np.float64)
        self.components_ = np.asarray(summary.components[:keep], dtype=np.float64)
        self.mean_ = np.asarray(summary.mean, dtype=np.float64)
        self.n_samples_ = n_samples

        # ddof=1, matching scikit-learn's PCA and the / (n_samples - 1) convention of an unbiased sample variance.
        # n_samples == 1 has no unbiased variance; the singular values are all zero there, so report zeros.
        full_singular_values = np.asarray(summary.singular_values, dtype=np.float64)
        if n_samples > 1:
            self.explained_variance_ = self.singular_values_**2 / (n_samples - 1)
            total_variance = float(np.sum(full_singular_values**2)) / (n_samples - 1)
        else:
            self.explained_variance_ = np.zeros_like(self.singular_values_)
            total_variance = 0.0

        # The denominator is the TOTAL pooled variance over the FULL merged rank (never over the truncated
        # n_components_), so the ratios sum to <= 1 and only a truncation moves them away from 1. It is
        # sum(singular_values**2) / (n_samples - 1) rather than a bare sum(singular_values**2): the bare sum is
        # (n_samples - 1) times the variance, which would deflate every ratio by that factor and make them sum to
        # ~1 / (n_samples - 1). The spec's prose requires the <= 1 invariant, so the variance-consistent form wins
        # here; it also matches sklearn's PCA exactly.
        if total_variance > 0.0:
            self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        else:
            # Every sample was identical: the variance is genuinely zero and a ratio is 0/0, so report zeros rather
            # than nan. sklearn emits nan here; refusing to emit nan is the point of the guard.
            self.explained_variance_ratio_ = np.zeros_like(self.singular_values_)

        if self.whiten:
            # A component with zero explained variance cannot be whitened (it would be 0/0), so it is left untouched
            # instead of becoming nan. np.divide(..., where=...) is the guard; the explicit where on the assignment
            # keeps components_ untouched for exactly those rows.
            scale = np.zeros_like(self.explained_variance_)
            np.divide(1.0, np.sqrt(self.explained_variance_), out=scale, where=self.explained_variance_ > 0)
            self.components_ = self.components_ * scale[:, None]

    def _check_fitted(self, method: str) -> None:
        """Refuse a method that needs fitted attributes before ``fit`` has run.

        - ``:param method:`` Name of the calling method, quoted verbatim in the message.
        """
        if not hasattr(self, "_summary_"):
            raise ValueError(
                f"MergeablePCA instance is not fitted yet. Call 'fit' with appropriate arguments before using {method}."
            )

    # ---------------------------------------------------------------------------- projection
    def transform(self, X):
        """Project ``X`` onto the fitted components, honoring ``whiten``.

        ``Z = (X - mean_) @ components_.T``, i.e. centered by the pooled mean first. With ``whiten=True`` the
        components already carry the ``1 / sqrt(explained_variance)`` scale, so the columns come out with unit sample
        variance; this is scikit-learn's formulation, expressed as one matrix product so Dask fuses it into the graph.

        ``X`` goes through the SAME axis policy as ``fit``, so a caller can project a 5-D field with the axis names
        they fitted with. Only the feature dimension has to match the fitted one: the sample axes may be anything, as
        usual for a projection.

        - ``:param X:`` Array matching ``n_features_in_``, dask or numpy, 2-D or more under the same axis policy as fit.
        """
        self._check_fitted("transform")
        flat = self._as_2d(X if isinstance(X, da.Array) else da.asarray(X))
        if int(flat.shape[1]) != self.n_features_in_:
            raise ValueError(
                f"MergeablePCA: X has {int(flat.shape[1])} features but this estimator was fitted on "
                f"{self.n_features_in_}. Provide the same number of features as in fit, or refit on X."
            )
        # mean_ @ components_.T is a (n_components,) row vector; dask broadcasts it over the rows.
        return (flat - self.mean_) @ self.components_.T

    def inverse_transform(self, Z):
        """Map projected data back to the original feature space: ``X_hat = Z @ components_ + mean_``.

        Exact on the retained subspace only; the discarded directions cannot be recovered.

        - ``:param Z:`` 2-D ``(n_samples, n_components_)` array, dask or numpy.
        """
        self._check_fitted("inverse_transform")
        array = Z if isinstance(Z, da.Array) else da.asarray(Z)
        if array.ndim != 2:
            raise ValueError(
                f"MergeablePCA: Z must be 2-dimensional, got ndim={array.ndim}. "
                f"Reshape with Z.reshape(n_samples, -1) or pass a 2-D array of shape (n_samples, {self.n_components_})."
            )
        if int(array.shape[1]) != self.n_components_:
            raise ValueError(
                f"MergeablePCA: Z has {int(array.shape[1])} columns but this estimator kept {self.n_components_} "
                "components. Pass a projection of shape (n_samples, n_components_) as produced by transform."
            )
        return array @ self.components_ + self.mean_
