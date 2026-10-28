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

Each dispatch unit describes one chunk-local sub-expression of the user's callback that the bridge executes on its local
numpy chunk and whose result the Deisa-side runtime path combines across bridges. The request types live in
:mod:`deisa.dask.runtime_expr` (the planner's product-side contract); this module holds the bridge-side and combine-side
callables that consume them.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from dask import delayed
from dask.array.reductions import mean_agg, moment_agg
from deisa.dask.mergeable_pca import PCASummary
from deisa.dask.precompute_errors import (
    PrecomputeRuntimeError,
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
# other op without pretending to be one of the analyzer's ``SUPPORTED_OPS`` (now in ``test/_oracle/task_branches.py``).
_PCA_OP_NAME = "pca"


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

# Bridge-local finishing step for a FULL local fold (the analyzer that composes it lives in
# ``test/_oracle/branch_specs.py::_apply_full_local_fold``): reduce dask's tier-shaped
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
