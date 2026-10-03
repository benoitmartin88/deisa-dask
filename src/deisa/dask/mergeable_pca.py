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
- The merge-plus-tree-reduction shape (merging rank-``r`` summaries pairwise, re-truncating at each node) is **Qin &
  Yan, arXiv:1601.07010** -- their Lemma 1 states the merge and their hierarchical Algorithm 1 the reduction tree --
  and **Kjolstad, Demmel et al., arXiv:1710.02812** -- merge-and-truncate.
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
    Reduce summaries with a balanced pairwise tree, the reduction shape of Qin & Yan (arXiv:1601.07010, Algorithm 1).

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


# The technical spec (MERGEABLE_PCA_SPEC.md section 5) spells these with a leading underscore. Both spellings are
# module level and pickle identically, so keep them as plain aliases instead of wrappers.
_PCASummary = PCASummary
_local_pca = local_pca
_merge_pca = merge_pca
_merge_tree = merge_tree
