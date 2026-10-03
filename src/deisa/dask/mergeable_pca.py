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
    with a balanced tree of ``dask.delayed(merge_pca)`` tasks (the shape of Qin & Yan, arXiv:1601.07010, Algorithm 1).
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

    Repository constraints honoured here
    -------------------------------------
    - ``ruff`` line length is 120 and it applies to docstring prose and ``raise`` literals too, so long messages are
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
      ``min(n_block, n_features)``.
    """

    def __init__(
        self,
        n_components: int | None = None,
        whiten: bool = False,
        copy: bool = True,
        batch_size: int | None = None,
        local_rank: int | None = None,
    ):
        _validate_optional_positive_int("n_components", n_components)
        _validate_optional_positive_int("local_rank", local_rank)
        _validate_optional_positive_int("batch_size", batch_size)
        if not isinstance(whiten, bool):
            raise ValueError(f"MergeablePCA: whiten must be a bool, got {type(whiten).__name__}.")
        if not isinstance(copy, bool):
            raise ValueError(f"MergeablePCA: copy must be a bool, got {type(copy).__name__}.")
        self.n_components = n_components
        self.whiten = whiten
        self.copy = copy
        self.batch_size = batch_size
        self.local_rank = local_rank

    # ---------------------------------------------------------------------------- fitting
    def fit(self, X, y=None) -> MergeablePCA:
        """Fit the estimator and return ``self``.

        A dask array is fitted through :meth:`_fit_dask_delayed`; a numpy array takes the in-memory path, which is the
        single-leaf base case of the same tree.

        - ``:param X:`` 2-D ``(n_samples, n_features)`` dask array, or a numpy array of the same shape.
        - ``:param y:`` Accepted and ignored, for scikit-learn parity.
        """
        array = X if isinstance(X, da.Array) else da.asarray(X)
        self._validate_input(array)

        n_samples = int(array.shape[0])
        n_features = int(array.shape[1])
        self.n_features_in_ = n_features
        # Checked against min(n_samples, n_features) before any work, so an impossible request fails fast and free.
        if self.n_components is not None and self.n_components > min(n_samples, n_features):
            raise ValueError(
                f"MergeablePCA: n_components={self.n_components} exceeds "
                f"min(n_samples={n_samples}, n_features={n_features}). "
                "Reduce n_components, or provide more samples / features."
            )

        if isinstance(X, da.Array):
            summary = self._fit_dask(array)
        else:
            # In-memory: one leaf, the base case of the merge tree. Still routed through the same primitive.
            summary = local_pca(np.asarray(X), rank=self.local_rank)

        self._summary_ = summary
        self._check_root_rank(summary)
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

        - ``:param X:`` 2-D ``(n_samples, n_features)`` dask array whose features are in one chunk.
        """
        self._validate_input(X)
        level = [delayed(local_pca)(block, rank=self.local_rank) for block in self._leaf_blocks(X)]
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
        them are refusals rather than warnings.

        - ``:param X:`` Candidate input array (already wrapped in ``dask.array`` by the caller).
        """
        if X.ndim != 2:
            raise ValueError(
                f"MergeablePCA: X must be 2-dimensional, got ndim={X.ndim}. "
                "Reshape with X.reshape(n_samples, -1) or pass a 2-D array."
            )
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
        # Checked after ndim, so a 1-D input reports the ndim error rather than an IndexError on chunks[1].
        if len(X.chunks[1]) != 1:
            raise ValueError(
                "MergeablePCA requires the complete feature dimension in a single Dask chunk, "
                f"got chunks={X.chunks}. Rechunk with X = X.rechunk({{1: -1}}) before fitting."
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

        - ``:param X:`` 2-D dask array whose features are in one chunk.
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

        - ``:param X:`` 2-D ``(n_samples, n_features)`` array, dask or numpy, matching ``n_features_in_``.
        """
        self._check_fitted("transform")
        array = X if isinstance(X, da.Array) else da.asarray(X)
        if array.ndim != 2:
            raise ValueError(
                f"MergeablePCA: X must be 2-dimensional, got ndim={array.ndim}. "
                "Reshape with X.reshape(n_samples, -1) or pass a 2-D array."
            )
        if int(array.shape[1]) != self.n_features_in_:
            raise ValueError(
                f"MergeablePCA: X has {int(array.shape[1])} features but this estimator was fitted on "
                f"{self.n_features_in_}. Provide the same number of features as in fit, or refit on X."
            )
        # mean_ @ components_.T is a (n_components,) row vector; dask broadcasts it over the rows.
        return (array - self.mean_) @ self.components_.T

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
