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
Branch-level local compute on the bridge.

A :class:`BranchSpec` describes one chunk-local sub-expression of the user's callback that the bridge can execute on
its local numpy chunk and whose result the Deisa-side topic handler combines across bridges.

Branch of **length-1** case: each BranchSpec corresponds to one detected reduction (``arr.sum()``,
``arr.mean(axis=0)``, ...). The branch is a single chunk callable followed by the reduction's combine aggregator. For
multi-layer chains (length->=2 branches); the data structure below is designed to support that without further changes.

The structure mirrors the prior per-reduction branch metadata (``kind`` / ``finalize`` / ``shape`` / ``dtype`` /
``chunk_axis``) so the bridge and Deisa-side combine code paths can be refactored to consume :class:`BranchSpec`
directly without changing semantics.
"""

from __future__ import annotations

import functools
import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import dask.array as da
from dask import delayed
from dask.array.reductions import mean_agg, moment_agg
from deisa.dask.mergeable_pca import PCASummary
from deisa.dask.precompute_errors import (
    PrecomputeRuntimeError,
)
from deisa.dask.task_branches import (
    _blockwise_indices_inputs,
)
from deisa.dask.task_branches import (
    # Re-exported for the test suite (test_chain.py imports it from here); the ``as`` alias marks the re-export for ruff
    # (F401).
    _find_chunk_layer as _find_chunk_layer,
)

logger = logging.getLogger(__name__)


# Output kind values. "scalar" / "mean" / "moment" cover the per-reduction cases. Later, may add "scalar-array" for
# chained pointwise + reduction outputs that are 1-d (e.g. ``arr.mean(axis=0)``).
_BRANCH_KIND_SCALAR = "scalar"
_BRANCH_KIND_MEAN = "mean"
_BRANCH_KIND_MOMENT = "moment"
# "pca" is not a reduction but a per-bridge DECOMPOSITION: the bridge runs a local PCA on its own chunk and ships
# the MERGEABLE SUMMARY (:class:`deisa.dask.mergeable_pca.PCASummary`), never the truncated public representation.
# Truncating at the bridge would silently destroy the mergeability that is the whole point of the branch, so the
# Deisa side merges summaries with the tree of :func:`deisa.dask.mergeable_pca.merge_pca` and only the ROOT applies
# ``n_components``/``whiten``.
_BRANCH_KIND_PCA = "pca"

# The op name recorded for a PCA branch. It is NOT a dask reduction op: it names the decomposition the bridge runs, so
# it takes the same path through ``output_key`` (``{array}-pca``), ``op_name`` and the dispatch signatures as every
# other op without pretending to be one of ``SUPPORTED_OPS`` in :mod:`deisa.dask.task_branches`.
_PCA_OP_NAME = "pca"


@dataclass
class BranchSpec:
    """One chunk-local sub-expression the bridge can execute + combine.

        Attributes
        ----------
        output_key : str
            Stable identifier for this branch (e.g. ``"f-mean"``). The bridge uses it to namespace its scatter key and
            the Deisa topic handler uses it to route the per-bridge partials back to the same branch.
        input_name : str
            Registered array name the branch is rooted at (e.g. ``"a"``). The Deisa side groups branches per array using
            this field (never by parsing ``output_key``) and files each group with ``set_task_branches`` under its own
            array name via ``execute_callbacks`` -> ``_flush_branches_to_handshake``.
        output_kind : str
            One of ``"scalar"`` / ``"mean"`` / ``"moment"``. Drives the Deisa-side combine graph: ``scalar`` ->
            ``da.stack`` + dask sum, ``mean`` -> ``mean_agg`` over nested list of dicts, ``moment`` -> ``moment_agg`` (+
            ``np.sqrt`` for ``finalize == "sqrt"``).
        branch_func : Callable
            Python callable that, given a numpy chunk, returns the branch's per-bridge partial value (a scalar /
            ndarray / dict). Pickled across the bridge process boundary. Currently a length-1 callable (``chunk_func``
            from the prior branch); This may produce multi-callable composites.
        chunk_axis : Optional[Tuple[int, ...]]
            For reductions, the tuple of axes being reduced in the chunk (e.g. ``(0, 1)`` for full reduction on a 2-D
            chunk). ``None`` for pointwise-only branches.
        finalize : Optional[str]
            ``"sqrt"`` for std (apply ``np.sqrt`` after combining), else ``None``.
        partial_shape : Tuple[int, ...]
            Shape of the **per-bridge partial** (what the branch_func returns). For ``scalar`` reductions on a 2-D chunk
            with ``keepdims=False`` this is ``()``; with ``keepdims=True`` it is ``(1, 1)``. For axis reductions the
            partial keeps the un-reduced axes' full size (e.g. ``mean(axis=0)`` on ``(M, N)`` partial has shape ``(1,
            N)``). The bridge records this on the topic event so the Deisa side knows what each bridge shipped.
        partial_dtype : str
    NumPy dtype string of the per-bridge partial.
    """

    output_key: str
    input_name: str
    output_kind: str
    branch_func: Callable[[Any], Any]
    chunk_axis: Optional[Tuple[int, ...]]
    finalize: Optional[str]
    partial_shape: Tuple[int, ...]
    partial_dtype: str
    op_name: str = ""
    deliver_direct: bool = True
    window_read: bool = False
    reduction_axes: Tuple[int, ...] = ()
    # For a "pca" branch: the recorded MergeablePCA configuration, applied at the ROOT of the merge tree on the Dask
    # side and deliberately NOT on the bridge. Empty for every reduction branch. The bridge's payload must stay the
    # mergeable summary, so ``n_components``/``whiten`` cannot travel inside ``branch_func`` -- truncating per bridge
    # would destroy exactly the mergeability the branch exists to provide.
    summary_config: Dict[str, Any] = field(default_factory=dict)


def merge_branches(existing: List[BranchSpec], new: List[BranchSpec]) -> List[BranchSpec]:
    """Merge two branch lists, deduping by ``output_key``.

    Identical signatures (same op, same reduction axes) share one bridge execution; same key with a DIFFERENT
    reduction-axes signature (e.g. a window read vs. a true axis reduction over the same axis) always raises
    :class:`PrecomputeRuntimeError` -- the two deliver different shapes and sharing them silently delivers one callback
    the other one's result.
    """
    merged = list(existing)
    seen: Dict[str, BranchSpec] = {b.output_key: b for b in merged}
    for b in new:
        prev = seen.get(b.output_key)
        if prev is None:
            merged.append(b)
            seen[b.output_key] = b
            continue
        if prev.reduction_axes != b.reduction_axes or prev.op_name != b.op_name or prev.output_kind != b.output_kind:
            raise PrecomputeRuntimeError(
                f"merge_branches: array {b.input_name!r} has two reductions sharing output_key "
                f"{b.output_key!r} but different runtime signatures: "
                f"(op={prev.op_name!r}, axes={prev.reduction_axes!r}, kind={prev.output_kind!r}) vs "
                f"(op={b.op_name!r}, axes={b.reduction_axes!r}, kind={b.output_kind!r}). The precompute "
                f"delivery path cannot serve both callbacks from one branch; rename one of the reductions "
                f"or register them on separate arrays."
            )
        # Identical signatures: keep the first branch (dedup).
    return merged




# Scalar-kind (``sum``/``prod``/``max``/``min``) elementwise fold for the first combine stage: scalar-axis partials
# are plain arrays (reduced axes already dropped), combined by binary-ufunc fold over identical shapes.
#
# The boolean/count kinds fold with the same signature and join the table here, because their cross-bridge step IS a
# scalar fold: ``any`` is a max-like OR, ``all`` a min-like AND, and ``count`` is additive (each bridge ships the count
# of its own non-zero elements, so the global count is their sum). Keying them here keeps one table for every fold --
# no build path special-cases an operator.
_SCALAR_FOLDS = {
    "sum": lambda entries: np.sum(entries, axis=0),
    "prod": lambda entries: np.prod(entries, axis=0),
    "max": lambda entries: np.max(entries, axis=0),
    "min": lambda entries: np.min(entries, axis=0),
    "any": lambda entries: np.any(entries, axis=0),
    "all": lambda entries: np.all(entries, axis=0),
    "count": lambda entries: np.sum(entries, axis=0),
}

# Bridge-local finishing step for a FULL local fold (see :func:`_apply_full_local_fold`): reduce dask's tier-shaped
# chunk-stage output down to the minimal per-bridge partial the operator's combine can consume. Generic by design --
# the table is keyed by op name and no build path special-cases an operator. ``mean``/``moment`` never reach here:
# their tier-shaped dict partials are REQUIRED by ``mean_agg`` / ``moment_agg`` (a bare scalar cannot carry
# ``(n, total[, M])``), which is a mathematical property, not a chunk-func signature accident. The finalizers are
# module-level functions (NOT lambdas): the composed ``branch_func`` carries them across the bridge process
# boundary via pickle, and lambdas cannot be pickled by reference.


def _final_full_sum(x):
    return np.sum(x)


def _final_full_prod(x):
    return np.prod(x)


def _final_full_max(x):
    return np.max(x)


def _final_full_min(x):
    return np.min(x)


_FULL_FOLD_FINALIZERS = {
    "sum": _final_full_sum,
    "prod": _final_full_prod,
    "max": _final_full_max,
    "min": _final_full_min,
}


def _full_fold_layer(x, tier_func, tier_kwargs, finalizer):
    """Second chain stage of a full local fold: finish the tier output down to the minimal partial.

    Defined at module level so the composed ``branch_func`` pickles cleanly across the bridge process boundary (the
    same constraint that shaped :func:`_chain_branch_func`).
    """
    return finalizer(tier_func(x, **tier_kwargs))


def _apply_full_local_fold(
    op_name: Optional[str],
    full_local_fold: bool,
    chain: List[Tuple[Callable, dict, int]],
) -> List[Tuple[Callable, dict, int]]:
    """Append the op's full-fold finalizer to ``chain`` (the last layer) when ``full_local_fold`` holds.

    The single site of the minimal-scalar-partial rule: the bridge-local ``branch_func`` becomes
    ``tier -> finalizer`` so the bridge folds its whole chunk before sending (smallest payload, shortest critical
    path), while the recorded signature stays the tier's own ``chunk_axis``-derived value. No builder rebinds dask's
    chunk kwargs and no operator is special-cased outside this table.
    """
    if not full_local_fold:
        return chain
    finalizer = _FULL_FOLD_FINALIZERS.get(op_name or "")
    if finalizer is None:
        # A kind that is scalar-flagged but has no finalizer cannot fold eagerly: keep the tier partial (the delivery
        # contract still holds -- only the payload is larger than minimal).
        return chain
    tier_func, tier_kwargs, input_count = chain[-1]
    layer = (
        functools.partial(_full_fold_layer, tier_func=tier_func, tier_kwargs=tier_kwargs, finalizer=finalizer),
        {},
        input_count,
    )
    return list(chain[:-1]) + [layer]


def _nest_partial_dicts_by_grid(
    partials: List[Dict[str, Any]], grid_extent: Optional[Tuple[int, ...]] = None
) -> Tuple[Any, Tuple[int, ...]]:
    """Arrange per-bridge dict-blob partials into a nested list that mirrors
    the MPI chunk grid, so that ``mean_agg`` / ``moment_agg`` (which walk the nested list with ``_concatenate2``) can
    combine them.

    ``partials`` is a list of dicts each carrying a ``chunk_position`` -- the bridge's MPI coords. Returns
    ``(nested_list, grid_shape)`` where ``grid_shape`` is the MPI grid shape (``(N, M, ...)``) and
    ``nested_list[i_0][i_1]...`` is the dict (or future-of-dict) at MPI coords ``(i_0, i_1, ...)``.

    ``grid_extent`` (per data axis, from ``global_shape // chunk_shape``) is validated against the coords when
    provided: a mismatch means the grid layout the partials describe contradicts the array metadata, which would
    silently corrupt any axis combine -- raise instead (F5).

    For a 1-D MPI grid (e.g. ``(2,)`` or ``(4,)``) this returns a flat list of length N. For higher-D grids the list is
    nested.
    """
    # Determine grid shape from the unique coords across all partials.
    coords = [tuple(p["chunk_position"]) for p in partials]
    if not coords:
        raise ValueError("_nest_partial_dicts_by_grid: no partials provided")
    ndim = len(coords[0])
    # Validate uniform ndim
    for c in coords:
        if len(c) != ndim:
            raise ValueError(f"_nest_partial_dicts_by_grid: partials have inconsistent coord dimensions: {coords}")
    # Per-axis sizes
    axis_sizes: Dict[int, set] = {ax: set() for ax in range(ndim)}
    for c in coords:
        for ax, v in enumerate(c):
            axis_sizes[ax].add(v)
    grid_shape = tuple(len(axis_sizes[ax]) for ax in range(ndim))
    if grid_extent is not None and tuple(grid_extent) != grid_shape:
        raise PrecomputeRuntimeError(
            f"_nest_partial_dicts_by_grid: partials' chunk grid {grid_shape} contradicts the array metadata "
            f"grid {tuple(grid_extent)}. The grid<->data axis mapping is ambiguous; refusing to combine "
            f"rather than silently producing a wrong reduction."
        )
    # Validate full grid is filled
    expected = set(itertools.product(*(range(g) for g in grid_shape)))
    actual = set(coords)
    if actual != expected:
        raise ValueError(
            f"_nest_partial_dicts_by_grid: partials do not cover the full grid (expected {expected}, got {actual})"
        )

    # Build a map from coords -> dict (or future)
    by_coord: Dict[Tuple[int, ...], int] = {c: i for i, c in enumerate(coords)}

    def _build_nested(dims_remaining: Tuple[int, ...], prefix: Tuple[int, ...]) -> Any:
        if not dims_remaining:
            return partials[by_coord[prefix]]["future"]
        head, *tail = dims_remaining
        return [_build_nested(tuple(tail), (*prefix, i)) for i in range(head)]

    nested = _build_nested(grid_shape, ())
    return nested, grid_shape


def _combine_pca_summaries(
    partial_futures: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
) -> Any:
    """Reduce the per-bridge PCA summaries into ONE delivered in-situ PCA, without a worker ever seeing a sample.

    This is the Deisa-side half of the in-situ placement. Each entry of ``partial_futures`` is a scattered
    :class:`~deisa.dask.mergeable_pca.PCASummary` produced by a bridge running
    :func:`~deisa.dask.mergeable_pca.local_pca_from_chunk` over its OWN chunk; the returned object merges them with the
    balanced pairwise tree of :func:`~deisa.dask.mergeable_pca.merge_pca` (the reduction shape of Qin & Yan,
    arXiv:1601.07010, Algorithm 1, and Kjolstad/Demmel et al., arXiv:1710.02812, merge-and-truncate).

    Three properties this is built to have, each of which the acceptance tests assert:

    - **No sample crosses.** The tree's leaves are the scattered summaries, so a merge node consumes summaries only. No
      task in the graph ever holds a raw sample, which is what makes "the chunk never reaches a worker" provable rather
      than asserted: there is no task that COULD return one.
    - **Not truncated at the bridge.** ``n_components`` and ``whiten`` in ``config`` are applied at the ROOT only,
      inside :meth:`MergeablePCA._fit_delivered`. Truncating per bridge would drop directions before the cross-bridge
      Chan-Golub-LeVeque (1979) correction term ever runs and silently destroy exactness.
    - **A real Dask graph, returned unbuilt.** The result is a ``Delayed`` carrying the merged ``PCASummary``, so each
      merge node costs ``O(d^3)`` -- independent of the sample count -- and the graph is inspectable before it runs.

    The merge algebra is classical (Chan-Golub-LeVeque 1979; Qin & Yan; Kjolstad/Demmel); what is claimed as ours is
    only that the leaves are evaluated inside the simulation's MPI bridge rather than on the analytics workers.

    - ``:param partial_futures:`` One ``{"future": Future, ...}`` entry per bridge, each holding a scattered
      ``PCASummary``.
    - ``:param config:`` The recorded root-only estimator configuration (``n_components`` / ``whiten`` / ...), used to
      build the same estimator the user wrote. ``None`` or empty means the defaults.
    """
    from deisa.dask.mergeable_pca import merge_pca

    # Sort by chunk_position for determinism: merge_pca is associative, so the ORDER cannot change the result, but a
    # stable order makes the graph reproducible and the tests comparable run to run.
    ordered = sorted(partial_futures, key=lambda p: tuple(p["chunk_position"]))
    level = [delayed(p["future"]) for p in ordered]
    while len(level) > 1:
        merged = [delayed(merge_pca)(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            # An odd level carries its last node up unchanged, keeping the tree balanced instead of degenerating into
            # a chain -- the same reasoning merge_tree makes, and safe because merge_pca is associative.
            merged.append(level[-1])
        level = merged
    return DeliveredPCA(level[0], dict(config or {}))


@dataclass(frozen=True)
class DeliveredPCA:
    """A merged in-situ PCA summary plus the estimator configuration that should consume it.

    Carried as an attribute (``_deisa_pca_summary``) on the per-callback dispatch view, which is what
    :meth:`MergeablePCA._fit_delivered` looks for. A frozen dataclass pickles by value: no reconstruction callback, no
    globals to resolve on a worker, and the ``Delayed`` inside it rebuilds its own graph when it does.

    - ``:param merged:`` ``Delayed`` (or any object with ``.compute()``) yielding the merged ``PCASummary``.
    - ``:param config:`` Root-only estimator configuration: ``n_components``, ``whiten``, and the axis policy.
    """

    merged: Any
    config: Dict[str, Any]

    def compute(self) -> Any:
        """Materialize the merged summary. Equivalent to ``self.merged.compute()``.

        Named explicitly so the estimator's ``hasattr(delivered, "compute")`` branch treats this wrapper and a bare
        ``Delayed`` identically, instead of a wrapper silently reading ``.merged`` as if it were the summary.
        """
        return self.merged.compute()


def _flatten_grid_entries(nested: Any) -> List[Any]:
    """Flatten a (possibly nested) grid structure of partial values into a flat list."""
    out: List[Any] = []
    stack = [nested]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(item))
        else:
            out.append(item)
    return out


def _concat_kept_grid(grid: Any, depth: int) -> np.ndarray:
    """np.concatenate a kept-grid (nested over kept axes) along the result axes.

    The first-stage result for kept-coordinate ``(i0, i1, ...)`` has axes ``(kept data axes in ascending data order)``;
    nesting level ``depth`` of the grid corresponds to result axis ``depth``, so concatenating level by level
    reproduces the full-kept-extent array.
    """
    if isinstance(grid[0], list):
        return np.concatenate([_concat_kept_grid(g, depth + 1) for g in grid], axis=depth)
    return np.concatenate([np.asarray(g) for g in grid], axis=depth)


@delayed
def _combine_two_phase(
    flat_entries: List[Tuple[Tuple[int, ...], Any]],
    kind: str,
    op_name: Optional[str],
    red_axes: Tuple[int, ...],
    kept_axes: Tuple[int, ...],
    grid_shape: Tuple[int, ...],
    out_dtype: str,
    finalize: Optional[str],
) -> np.ndarray:
    """Combine per-bridge partials into the final, correctly-shaped reduction."""
    by_coord = {tuple(c): v for c, v in flat_entries}

    def _full_coord(kept_coord: Tuple[int, ...], red_coord: Tuple[int, ...]) -> Tuple[int, ...]:
        full = [0] * len(grid_shape)
        for pos, ax in enumerate(kept_axes):
            full[ax] = kept_coord[pos]
        for pos, ax in enumerate(red_axes):
            full[ax] = red_coord[pos]
        return tuple(full)

    def _build_sub(idx: int, red_prefix: Tuple[int, ...], kept_coord: Tuple[int, ...]) -> Any:
        if idx == len(red_axes):
            return by_coord[_full_coord(kept_coord, red_prefix)]
        ax = red_axes[idx]
        return [_build_sub(idx + 1, red_prefix + (i,), kept_coord) for i in range(grid_shape[ax])]

    per_kept_coord: Dict[Tuple[int, ...], np.ndarray] = {}
    for kept_coord in itertools.product(*(range(grid_shape[ax]) for ax in kept_axes)):
        sub = _build_sub(0, (), kept_coord)
        if kind in ("mean", "moment"):
            if kind == "mean":
                res = mean_agg(sub, dtype=np.dtype(out_dtype), axis=red_axes)
            else:
                res = moment_agg(sub, order=2, ddof=0, dtype=np.dtype(out_dtype), axis=red_axes)
            per_kept_coord[kept_coord] = np.asarray(res)
        elif kind == "scalar":
            fold = _SCALAR_FOLDS.get(op_name or "")
            if fold is None:
                raise PrecomputeRuntimeError(f"_combine_two_phase: no scalar fold for op {op_name!r}")
            entries = np.stack([np.asarray(v) for v in _flatten_grid_entries(sub)])
            res = np.asarray(fold(entries))
            # Scalar chunk funcs run with keepdims=True (dask's chunk stage), so the RED axes survive as size-1 dims at
            # their data positions. Drop them so the combined result's shape matches the declared kept-extent out_shape
            # (and concatenates cleanly).
            for ax in sorted(red_axes, reverse=True):
                res = np.squeeze(res, axis=ax)
            per_kept_coord[kept_coord] = res
        else:
            raise PrecomputeRuntimeError(f"_combine_two_phase: unknown kind {kind!r}")

    # Second stage: concatenate the per-kept-coordinate results along the kept levels; no kept axes -> single result.
    if not kept_axes:
        result = per_kept_coord[()]
    else:
        kept_grid_shape = tuple(grid_shape[ax] for ax in kept_axes)

        def _build_kept_grid(idx: int, prefix: Tuple[int, ...]) -> Any:
            if idx == len(kept_axes):
                return per_kept_coord[prefix]
            return [_build_kept_grid(idx + 1, prefix + (i,)) for i in range(kept_grid_shape[idx])]

        result = _concat_kept_grid(_build_kept_grid(0, ()), 0)

    if finalize == "sqrt":
        result = np.sqrt(result)
    return result


def _combine_array_from_partials(
    partials: List[Dict[str, Any]],
    kind: str,
    finalize: Optional[str],
    reduction_axes_hint: Optional[Tuple[int, ...]],
    array_ndim: int,
    op_name: Optional[str] = None,
    global_shape: Optional[Tuple[int, ...]] = None,
    grid_extent: Optional[Tuple[int, ...]] = None,
) -> Any:
    """Build a single-block dask array that combines per-bridge partials."""
    if not partials:
        raise ValueError("_combine_array_from_partials: no partials provided")
    nested, grid_shape = _nest_partial_dicts_by_grid(partials, grid_extent=grid_extent)
    out_dtype = str(partials[0]["dtype"])

    # Grid <-> data axis mapping must be explicit: each grid level is one data axis (the harness's cart dims == array
    # ndim invariant).
    if len(grid_shape) != array_ndim:
        raise PrecomputeRuntimeError(
            f"_combine_array_from_partials: chunk grid {grid_shape} has {len(grid_shape)} levels but the "
            f"array has {array_ndim} data axes; cannot map grid axes to data axes. Refusing to combine."
        )

    # Red axes = the data axes being reduced. ``reduction_axes_hint is None`` is the legacy no-axis form of a
    # full reduction (reduce all data axes).
    if reduction_axes_hint is None:
        red_axes = tuple(range(array_ndim))
    else:
        red_axes = tuple(int(a) for a in reduction_axes_hint)
        for ax in red_axes:
            if not 0 <= ax < array_ndim:
                raise PrecomputeRuntimeError(
                    f"_combine_array_from_partials: reduction axis {ax} out of range for {array_ndim}-D array"
                )
    kept_axes = tuple(ax for ax in range(array_ndim) if ax not in red_axes)

    # Compute the combined output shape.
    if global_shape is not None:
        out_shape = tuple(int(global_shape[ax]) for ax in kept_axes)
    else:
        # Legacy fallback (no metadata): derive from the per-bridge partial.
        if reduction_axes_hint is None:
            out_shape = tuple(partials[0]["shape"])
        else:
            out_shape = tuple(partials[0]["shape"][ax] for ax in kept_axes)

    flat_entries = [(tuple(p["chunk_position"]), p["future"]) for p in partials]
    return da.from_delayed(
        _combine_two_phase(
            flat_entries,
            kind=kind,
            op_name=op_name,
            red_axes=red_axes,
            kept_axes=kept_axes,
            grid_shape=grid_shape,
            out_dtype=out_dtype,
            finalize=finalize,
        ),
        shape=out_shape,
        dtype=out_dtype,
    )


def _discover_partial_metadata(
    branch_func: Callable[[Any], Any],
    placeholder: Optional[Any],
    branch: Dict[str, Any],
) -> Tuple[Tuple[int, ...], str]:
    """Run ``branch_func`` on the placeholder and return ``(partial_shape, partial_dtype)``."""
    if placeholder is None:
        partial_shape: Tuple[int, ...] = tuple(branch.get("shape") or ())
        partial_dtype = str(branch.get("dtype", "float64"))
    else:
        sample = branch_func(placeholder)
        if isinstance(sample, PCASummary):
            # A "pca" branch ships the mergeable summary, whose largest array is ``components``. Read the shape off the
            # summary instead of ``np.asarray``-ing the dataclass, which would build a 0-d object array and record
            # ``()``/``object`` -- metadata the Deisa side cannot use to reason about the payload at all.
            partial_shape = tuple(sample.components.shape)
            partial_dtype = str(sample.components.dtype)
        elif isinstance(sample, dict):
            # mean / moment : the per-bridge partial is a dict with per-key shape. Pick ``total`` as the representative
            # (it always has the reduction-output shape).
            if "total" in sample:
                rep = np.asarray(sample["total"])
            elif "M" in sample:
                rep = np.asarray(sample["M"])
            else:
                rep = np.asarray(next(iter(sample.values())))
            partial_shape = tuple(rep.shape)
            partial_dtype = str(rep.dtype)
        else:
            arr = np.asarray(sample)
            partial_shape = tuple(arr.shape)
            partial_dtype = str(arr.dtype)
    return partial_shape, partial_dtype
















def _find_single_upstream(layer) -> Optional[Tuple[str, int]]:
    """Return ``(upstream_layer_name, array_input_count)`` if the layer reads from a single upstream Blockwise
    (one or more times. e.g. ``arr * arr`` reads from ``arr`` twice and is still chunk-local). Returns ``None`` if the
    layer reads from multiple distinct array upstreams (cross-array, can't fold) or contains scalar constants (deferred
    to a later commit).

    Uses the shared Blockwise index-walking primitive :func:`deisa.dask.task_branches._blockwise_indices_inputs` so the
    ``layer.indices`` parsing lives in one place.
    """
    parsed = _blockwise_indices_inputs(layer)
    if parsed is None:
        # Not a new-style Blockwise (no indices) / empty indices.
        return None
    names, array_input_count, has_non_array_input = parsed
    upstream_names = set(names)
    if not upstream_names or len(upstream_names) > 1:
        return None
    if has_non_array_input:
        # Refuse to fold chains with constants. The constant would need to be embedded into the branch_func call (e.g.
        # ``add(chunk, 1)``) which requires extracting the constant value from the Blockwise task's args.
        return None
    return next(iter(upstream_names)), array_input_count




def _chain_branch_func(chunk, _chain=None):
    """Module-level branch callable: apply each (func, kwargs, input_count) in the chain to the chunk,
    threading the result through.

    Pair with :func:`_build_chain_branch_func` which binds ``_chain`` via :func:`functools.partial`. Defined at module
    level so pickle can find it across the bridge process boundary.
    """
    if _chain is None:
        raise RuntimeError("_chain_branch_func called without bound _chain")
    x = chunk
    for func, kwargs, input_count in _chain:
        if input_count == 1:
            x = func(x, **kwargs)
        elif input_count == 2:
            x = func(x, x, **kwargs)
        else:
            # For now refuse; can be extended for N-ary pointwise.
            raise ValueError(f"chain has layer with {input_count} inputs; only 1 or 2 supported")
    return x
