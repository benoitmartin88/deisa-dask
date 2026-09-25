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

A :class:`BranchSpec` describes one chunk-local sub-expression of the
user's callback that the bridge can execute on its local numpy chunk
and whose result the Deisa-side topic handler combines across bridges.

Branch of **length-1** case: each BranchSpec corresponds to one detected reduction (``arr.sum()``,
``arr.mean(axis=0)``, ...). The branch is a single chunk callable followed by the reduction's combine aggregator.
For multi-layer chains (length->=2 branches); the data structure below is designed to support that without further
changes.

The structure mirrors the prior per-reduction branch metadata
(``kind`` / ``finalize`` / ``shape`` / ``dtype`` / ``chunk_axis``) so the bridge and Deisa-side combine code paths can
be refactored to consume :class:`BranchSpec` directly without changing semantics.
"""

from __future__ import annotations

import functools
import itertools
import logging
import pickle
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import dask.array as da
from dask import delayed
from dask.array.reductions import mean_agg, moment_agg
from deisa.dask.precompute_analyzer import PrecomputeError, analyze_callback_with_dask_arrays
from deisa.dask.task_branches import _blockwise_indices_inputs, _find_chunk_layer, _is_aggregate_layer
from deisa.dask.utils import build_deisa_array

logger = logging.getLogger(__name__)


# Output kind values. "scalar" / "mean" / "moment" cover the per-reduction cases.
# Later, may add "scalar-array" for chained pointwise + reduction outputs that are 1-d (e.g. ``arr.mean(axis=0)``).
_BRANCH_KIND_SCALAR = "scalar"
_BRANCH_KIND_MEAN = "mean"
_BRANCH_KIND_MOMENT = "moment"


@dataclass
class BranchSpec:
    """One chunk-local sub-expression the bridge can execute + combine.

    Attributes
    ----------
    input_name : str
        Registered array name the branch is rooted at (e.g. ``"f"``).
    output_key : str
        Stable identifier for this branch (e.g. ``"f-mean"``). The
        bridge uses it to namespace its scatter key and the Deisa
        topic handler uses it to route the per-bridge partials back to
        the same branch.
    output_kind : str
        One of ``"scalar"`` / ``"mean"`` / ``"moment"``. Drives the
        Deisa-side combine graph: ``scalar`` -> ``da.stack`` + dask sum,
        ``mean`` -> ``mean_agg`` over nested list of dicts,
        ``moment`` -> ``moment_agg`` (+ ``np.sqrt`` for ``finalize ==
        "sqrt"``).
    branch_func : Callable
        Python callable that, given a numpy chunk, returns the branch's
        per-bridge partial value (a scalar / ndarray / dict). Pickled
        across the bridge process boundary. Currently a length-1
        callable (``chunk_func`` from the prior branch); This may
        produce multi-callable composites.
    chunk_axis : Optional[Tuple[int, ...]]
        For reductions, the tuple of axes being reduced in the chunk
        (e.g. ``(0, 1)`` for full reduction on a 2-D chunk). ``None``
        for pointwise-only branches.
    finalize : Optional[str]
        ``"sqrt"`` for std (apply ``np.sqrt`` after combining), else
        ``None``.
    partial_shape : Tuple[int, ...]
        Shape of the **per-bridge partial** (what the branch_func
        returns). For ``scalar`` reductions on a 2-D chunk with
        ``keepdims=False`` this is ``()``; with ``keepdims=True`` it is
        ``(1, 1)``. For axis reductions the partial keeps the un-reduced
        axes' full size (e.g. ``mean(axis=0)`` on ``(M, N)`` partial
        has shape ``(1, N)``). The bridge records this on the topic
        event so the Deisa side knows what each bridge shipped.
    partial_dtype : str
        NumPy dtype string of the per-bridge partial.
    output_shape : Tuple[int, ...]
        Shape of the **combined** reduction output (after the Deisa
        topic handler runs ``mean_agg`` / ``moment_agg`` / dask sum).
        For ``mean()`` on a 2-D array this is ``()``; for
        ``mean(axis=0)`` on ``(M, N)`` this is ``(N,)``. Used as the
        ``shape=`` argument to ``da.from_delayed`` on the Deisa side.
    output_dtype : str
        NumPy dtype string of the combined output.
    """

    input_name: str
    output_key: str
    output_kind: str
    branch_func: Callable[[Any], Any]
    chunk_axis: Optional[Tuple[int, ...]]
    finalize: Optional[str]
    partial_shape: Tuple[int, ...]
    partial_dtype: str
    output_shape: Tuple[int, ...]
    output_dtype: str


def _analyze_callback_for_branches(callback: Callable, registered_arrays: Dict[str, Any], precompute: bool = True):
    """
    Analyze the callback's source and build a list of :class:`BranchSpec` objects describing the chunk-local
    sub-expressions the bridge can execute.

    The callback is NOT executed: the AST is parsed and walked symbolically to find compute boundaries
    (.compute(), client.compute(), etc.) and the dask arrays they reference.

    - ``:param callback:`` The callback function to analyze.
    - ``:param registered_arrays:`` Mapping of array name -> dask array (or placeholder) for all registered arrays.
    - ``:param precompute:`` If False, log warnings instead of raising on
         analysis errors. Defaults to True.
    - ``:return:`` List of BranchSpec objects (empty if analysis fails
         or no reductions are detected).
    """

    # Build a dask array stub matching the registered array's shape/chunks so the symbolic AST walker has something
    # concrete to operate on. The chunking does not matter for hint extraction. We only read the task graph
    # structure, not the data.
    stubs = {}
    for arr_name in registered_arrays:
        # Get metadata for this array from registered_arrays (which is arrays_metadata), used for shape/chunks of
        # the placeholder stub.
        meta = registered_arrays.get(arr_name, {})
        global_shape = meta.get("global_shape")
        # The metadata exposes ``chunk_shape`` (per-bridge chunk), not a ``chunks`` tuple. ``da.zeros`` tiles the
        # global shape into uniform chunks of ``chunk_shape``.
        chunk_shape = meta.get("chunk_shape")
        if global_shape is not None and chunk_shape is not None:
            array_stub = da.zeros(global_shape, chunks=chunk_shape, dtype=np.float64)
        else:
            # Fallback for arrays without full metadata (e.g. opaque helpers or dynamically constructed arrays).
            # The stub's exact size doesn't matter. Only the graph structure matters for precompute analysis.
            # Shape and chunks must be consistent (same dimensionality).
            array_stub = da.zeros((10, 10), chunks=(5, 5), dtype=np.float64)
        # Wrap in DeisaArray so the analyzer sees the full attribute surface (.t, .timestep, dask Array methods)
        # that the callback uses at runtime.
        stubs[arr_name] = build_deisa_array(array_stub, timestep=0)

    try:
        return _analyze_branch(callback, registered_arrays=stubs, precompute=precompute)
    except PrecomputeError:
        # Cross-reduction, opaque parameter, etc. -- propagate so the caller learns the analysis was unable to
        # deliver a hint. precompute=False cases are handled inside analyze_branch (which logs warnings instead of
        # raising), so by the time we get here precompute=True was set, and we must propagate.
        raise
    except Exception as e:
        if not precompute:
            logger.debug(f"_analyze_callback_for_branches: Analysis failed: {e}")
            return []
        raise


def _nest_partial_dicts_by_grid(partials: List[Dict[str, Any]]) -> Tuple[Any, Tuple[int, ...]]:
    """Arrange per-bridge dict-blob partials into a nested list that mirrors
    the MPI chunk grid, so that ``mean_agg`` / ``moment_agg`` (which walk the
    nested list with ``_concatenate2``) can combine them.

    ``partials`` is a list of dicts each carrying a ``chunk_position`` --
    the bridge's MPI coords. Returns ``(nested_list, grid_shape)`` where
    ``grid_shape`` is the MPI grid shape (``(N, M, ...)``) and
    ``nested_list[i_0][i_1]...`` is the dict (or future-of-dict) at MPI
    coords ``(i_0, i_1, ...)``.

    For a 1-D MPI grid (e.g. ``(2,)`` or ``(4,)``) this returns a flat
    list of length N. For higher-D grids the list is nested.
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


def _combine_array_from_partials(
    partials: List[Dict[str, Any]],
    kind: str,
    finalize: Optional[str],
    hint_axis: Optional[Tuple[int, ...]],
    array_ndim: int,
) -> Any:
    """Build a single-block dask array that combines per-bridge partials via ``mean_agg`` or ``moment_agg``.

    For each output_key (``sum``, ``mean``, ...) the bridge shipped a single future whose value is a dict
    ``{n: ..., total: ...[, M: ...]}`` (per dask's ``mean_chunk`` / ``moment_chunk``).
    The Deisa side has one such future per bridge (per MPI rank).
    To combine them we:

    1. Arrange the futures in a nested list matching the MPI grid.
    2. Build a delayed task that, when run, resolves the futures, calls ``mean_agg`` / ``moment_agg`` over the nested
       list, and applies ``sqrt`` if ``finalize == "sqrt"`` (std).
    3. Wrap that single delayed task as ``da.from_delayed`` so the callback sees a normal dask array.

    The output shape comes from the chunk's ``axis`` hint: removing the reduced axes from the array's ndim gives the
    reduction output's shape. For ``mean()`` on a 2-D array (``axis=(0, 1)``), the output is scalar ``()``.
    For ``mean(axis=0)``, the output is 1-d ``(N,)``.
    """
    if not partials:
        raise ValueError("_build_dict_blob_combine_array: no partials provided")
    nested, grid_shape = _nest_partial_dicts_by_grid(partials)
    out_dtype = partials[0]["dtype"]
    # Compute the output shape from the reduced axes.
    if hint_axis is None:
        # Fallback: rely on the bridge-recorded shape (keepdims=True result).
        out_shape = partials[0]["shape"]
    else:
        kept_axes = tuple(ax for ax in range(array_ndim) if ax not in hint_axis)
        out_shape = tuple(partials[0]["shape"][ax] for ax in kept_axes)
    agg_kind = kind  # "mean" or "moment"
    # The agg's axis is the chunk's reduction axis. mean_agg / moment_agg walk the nested list with one axis per nesting
    # level, and the chunk's reduction axis happens to be the same as the MPI-grid axis when each bridge owns exactly
    # one chunk along the reduced axes (our precompute invariant). For higher-D reductions that span multiple bridge
    # axes, ``hint_axis`` carries the full tuple.
    agg_axis = hint_axis if hint_axis is not None else tuple(range(len(grid_shape)))

    @delayed
    def _combine(pairs):
        if agg_kind == "mean":
            result = mean_agg(pairs, dtype=np.dtype(out_dtype), axis=agg_axis)
        else:
            result = moment_agg(pairs, order=2, ddof=0, dtype=np.dtype(out_dtype), axis=agg_axis)
        if finalize == "sqrt":
            result = np.sqrt(result)
        return result

    return da.from_delayed(_combine(nested), shape=out_shape, dtype=out_dtype)


def _derive_combined_output_shape(
    chunk_axis: Optional[Tuple[int, ...]],
    array_ndim: int,
    partial_shape: Tuple[int, ...],
) -> Tuple[int, ...]:
    """Compute the shape of the combined reduction output.

    For full reductions (``chunk_axis == (0, 1, ...)`` matching all axes), the output is scalar ``()``.

    For axis reductions, the output keeps the un-reduced axes' sizes from the partial (each bridge's chunk has the full
    size along non-reduced axes, so the partial shape encodes the combined shape).
    """
    if chunk_axis is None:
        # output shape == partial shape.
        return partial_shape
    kept_axes = tuple(ax for ax in range(array_ndim) if ax not in chunk_axis)
    if not kept_axes:
        return ()
    return tuple(partial_shape[ax] for ax in kept_axes)


def _branch_func_with_kwargs(chunk, _cf=None, _kw=None):
    """Top-level branch callable: ``chunk_func(chunk, **chunk_kwargs)``.

    Defined at module level so pickle can find it across processes.
    The default-arg trick (``_cf=chunk_func``, ``_kw=chunk_kwargs``) binds the closure cells at function-definition
    time, which pickle serializes correctly.
    """
    if _cf is None or _kw is None:
        # Defensive: a stray call with no closure should never happen (BranchSpec.branch_func is always built via the
        # helper), but be loud rather than silent.
        raise RuntimeError("_branch_func_with_kwargs called without bound chunk_func / chunk_kwargs")
    return _cf(chunk, **_kw)


def _make_branch_func(chunk_func: Callable, chunk_kwargs: Dict[str, Any]) -> Callable[[Any], Any]:
    """Build a pickle-safe branch callable that closes over ``chunk_func`` and ``chunk_kwargs``.

    Returns a :func:`_branch_func_with_kwargs` partial with the chunk_func and chunk_kwargs bound via the default-arg
    trick. Returns a ``functools.partial`` so pickle can find the closure contents at unpickle time.
    """
    import functools as _functools

    return _functools.partial(_branch_func_with_kwargs, _cf=chunk_func, _kw=chunk_kwargs)


def _build_length1_branch(
    branch: Dict[str, Any],
    chunk_func: Callable[[Any], Any],
    input_name: str,
    array_ndim: int,
    placeholder: Optional[Any] = None,
) -> BranchSpec:
    """Build a :class:`BranchSpec` from a per-reduction branch dict.

    The branch_func is the same length-1 chunk_func the branch already carries.
    The BranchSpec re-exposes the branch's ``kind``/``finalize``/``shape``/``dtype`` metadata in the structured form
    the bridge and Deisa side will consume.

    If ``placeholder`` is provided, we run the chunk_func on it to discover the partial's actual shape/dtype.
    The placeholder is a dask array or numpy array representing the registered chunk (typically a zero-filled dask array
    of the right shape). This is purely a structural inspection, the chunk_func is a numpy operation, not a callback,
    so it has no side effects.

    For ``mean``/``moment`` partials the chunk_func returns a dict (``{n, total}`` or ``{n, total, M}``), we record the
    per-bridge partial's shape as the shape of the ``total`` value (the representative per-key shape, since all keys
    share the same reduced-axes pattern).
    """

    kind = branch.get("kind", _BRANCH_KIND_SCALAR)
    finalize = branch.get("finalize")

    # Reconstruct chunk_axis from chunk_kwargs['axis']. The branch carries
    # ``axis`` as a tuple/list/int; normalise to a tuple of ints.
    chunk_kwargs = branch.get("chunk_kwargs") or {}
    ax = chunk_kwargs.get("axis")
    if isinstance(ax, (list, tuple)):
        chunk_axis = tuple(int(a) for a in ax)
    elif ax is not None:
        chunk_axis = (int(ax),)
    else:
        chunk_axis = None

    # Discover the partial's shape and dtype by running the chunk_func on the placeholder. This is structural,
    # chunk_func is a numpy op, not the user's callback, so no side effects.
    # For ``mean`` and ``moment`` the bridge overrides ``keepdims=True``
    # (see :meth:`Bridge._execute_operations_on_chunk`) so the per-bridge  dict values are at least 1-D for
    # ``mean_agg`` / ``moment_agg`` to walk with ``_concatenate2``. Mirror that here.
    effective_kwargs = dict(chunk_kwargs)
    if kind in (_BRANCH_KIND_MEAN, _BRANCH_KIND_MOMENT):
        effective_kwargs["keepdims"] = True

    # Wrap the raw chunk_func in a closure that binds the analyzer's chunk_kwargs (axis, keepdims, dtype, ...).
    # The bridge calls ``branch.branch_func(chunk)`` with no extra kwargs, this closure carries them.
    # Pickle-friendly: a function with a closure of two picklable objects (a functools.partial and a dict). Uses the
    # top-level ``_make_branch_func`` helper so the closure is picklable across processes.
    branch_func = _make_branch_func(chunk_func, effective_kwargs)

    if placeholder is None:
        # Fallback: best-effort shape from branch metadata. The branch's ``shape`` field is not populated by the
        # analyzer. Only the bridge records it after running the chunk_func. We try to recover it from
        # ``keepdims`` + ``chunk_axis`` + the chunk's shape (which the branch doesn't carry either). Without a
        # placeholder we can't compute shape reliably, so we leave it as ``()`` and let the bridge's run-time
        # inspection overwrite it.
        partial_shape: Tuple[int, ...] = tuple(branch.get("shape") or ())
        partial_dtype = str(branch.get("dtype", "float64"))
    else:
        sample = branch_func(placeholder)
        if isinstance(sample, dict):
            # mean / moment : the per-bridge partial is a dict with per-key shape.
            # Pick ``total`` as the representative (it always has the reduction-output shape).
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

    output_shape = _derive_combined_output_shape(chunk_axis, array_ndim, partial_shape)
    output_dtype = partial_dtype  # mean/moment keep dtype through the agg

    return BranchSpec(
        input_name=input_name,
        output_key=branch["output_key"],
        output_kind=kind,
        branch_func=branch_func,
        chunk_axis=chunk_axis,
        finalize=finalize,
        partial_shape=partial_shape,
        partial_dtype=partial_dtype,
        output_shape=output_shape,
        output_dtype=output_dtype,
    )


def _analyze_branch(callback: Callable, registered_arrays: Dict[str, Any], precompute: bool = True) -> List[BranchSpec]:
    """Walk the callback's dask graph and emit a :class:`BranchSpec` per branch.

    Every reduction branch becomes one :class:`BranchSpec`. For each branch, the chain walker (``_walk_chain``)
    inspects the registered dask array's task graph and, if the reduction's chunk has a single-input upstream chain of
    pointwise layers, **folds the chain into one branch**. The branch_func applies every chain layer to the chunk on the
    bridge side and ships a single (small) partial instead of relying on dask workers to re-run the chunk pointwise
    chain.

    Folding is opportunistic. If ``_walk_chain`` returns ``None`` for a branch (cross-array upstream, scalar constant,
    non-Blockwise upstream, etc.) the branch degrades to the length-1 path: just the reduction's chunk_func.
    The chain walker only adds coverage, it never removes it.

    Parameters
    ----------
    callback : Callable
        The user's callback function. Not invoked.
    registered_arrays : Dict[str, Any]
        Mapping of name -> dask array (or other placeholder) for each registered array. Dask arrays become the roots of
        the task graph walk. Non-dask values are treated as opaque helpers (their attributes may be read for
        opaque-resolvable branches).
    precompute : bool
        If False, skip unresolvable reductions with a warning instead of raising.
        Matches :func:`analyze_callback`'s ``precompute`` semantics.

    Returns
    -------
    List[BranchSpec]
        One BranchSpec per detected reduction. Empty list if the callback contains no chunk-local precomputable
        operations.
    """
    # Single AST walk: analyze_callback_with_dask_arrays returns BOTH the reduction hints AND the walker's dask_arrays
    # in one pass. (Calling analyze_callback and analyze_callback_with_dask_arrays separately would parse + walk the
    # callback's AST twice.) The dask_arrays are the dask expressions the walker built at each compute boundary (e.g.
    # ``(arr*arr).sum()``). The registered placeholders' graphs only have the root layer, not the chain, so the chain
    # walker needs these.

    try:
        hints, walker_dask_arrays = analyze_callback_with_dask_arrays(
            callback, registered_arrays, precompute=precompute
        )
    except Exception:
        if precompute:
            raise
        # precompute=False: fall through with empty hints/dask_arrays. Length-1 fallback still works.
        hints, walker_dask_arrays = [], []

    if not hints:
        return []

    # Pick the registered-array name and ndim to attach to branches.
    primary = next(iter(registered_arrays)) if registered_arrays else "f"
    primary_arr: Optional[Any] = None
    array_ndim = 0
    try:
        primary_arr = next(iter(registered_arrays.values()))
        array_ndim = int(getattr(primary_arr, "ndim", 0))
    except Exception:
        pass
    # The chain walker needs at least one dask expression to walk. The AST walker builds one dask_arrays entry per
    # compute boundary. If there's nothing in the list, the registered placeholder is the only thing available,
    # but its graph is the root layer (no chain). We fall back to the length-1 path in that case.
    # We use the FIRST walker dask_array as the graph to walk. This matches the analyzer's behavior of treating the
    # first compute boundary as the primary one.
    dask_arr_for_chain: Optional[Any] = None
    if walker_dask_arrays:
        candidate = walker_dask_arrays[0].get("array")
        if hasattr(candidate, "__dask_graph__"):
            dask_arr_for_chain = candidate

    # The chain walker folds multi-layer pointwise chains into one branch_func. We dedupe chains per-branch: a chain
    # is unique by its (agg-layer-name, length). Multiple hints can share the same chain
    # (e.g. ``(arr**2).sum()`` and ``(arr**2).max()``). The walker builds the same branch_func either way.
    # ``_seen_chains`` keeps a memo to avoid rebuilding identical functools.partial objects.
    _seen_chains: Dict[Tuple[str, int], Any] = {}

    branches: List[BranchSpec] = []
    for branch_dict in hints:
        try:
            chunk_func = pickle.loads(branch_dict["chunk_func_pickle"])
        except Exception as e:  # pragma: no cover - safety net
            if not precompute:
                logger.warning("analyze_branch: unpickle failed for %s: %s", branch_dict.get("output_key"), e)
                continue
            raise

        # Discover the partial's shape and dtype by running the branch_func on a numpy placeholder. The placeholder is
        # the dask array's first chunk, materialized via .compute() so the numpy ops return numpy values (chunk_funcs
        # are numpy ops, not callback code).
        placeholder = primary_arr
        if hasattr(placeholder, "compute"):
            try:
                placeholder = placeholder.compute()
            except Exception:
                # If .compute() fails (e.g. no dask client in the CI worker process), fall back to a synthetic numpy
                # array that matches the dask array's shape/dtype/chunks so the chunk_func still produces a
                # representative partial.
                try:
                    placeholder = np.zeros(
                        getattr(placeholder, "shape", (4, 4)),
                        dtype=getattr(placeholder, "dtype", np.float64),
                    )
                except Exception:
                    placeholder = np.zeros((4, 4), dtype=np.float64)

        branch = _try_chain_branch(
            branch=branch_dict,
            primary=primary,
            array_ndim=array_ndim,
            placeholder=placeholder,
            dask_arr_for_chain=dask_arr_for_chain,
            seen_chains=_seen_chains,
        )
        if branch is None:
            # Chain walker refused; fall back to the length-1 path, passing the ORIGINAL branch dict
            # (not the None result).
            branch = _try_length1_branch(
                branch=branch_dict,
                chunk_func=chunk_func,
                primary=primary,
                array_ndim=array_ndim,
                placeholder=placeholder,
            )
        if branch is None:
            if not precompute:
                logger.debug(
                    "analyze_branch: build failed for %s. The length-1 path's build_length1_branch raised "
                    "(most likely the placeholder couldn't be computed or the chunk_func rejected the chunk shape).",
                    branch_dict.get("output_key"),
                )
                continue
            # Both paths returned None. This shouldn't happen for hints that came out of the analyzer (the length-1
            # path is supposed to always succeed). Raise defensively.
            raise RuntimeError(
                f"analyze_branch: cannot build branch for branch {branch_dict.get('output_key')!r}. "
                f"The chain walker refused (likely cross-array or constant upstream) AND the length-1 fallback's "
                f"build_length1_branch raised. This usually means the chunk_func rejected the placeholder. "
                f"Inspect with the failing branch's chunk_kwargs."
            )
        branches.append(branch)
    return branches


def _try_chain_branch(
    branch: Dict[str, Any],
    primary: str,
    array_ndim: int,
    placeholder: Optional[Any],
    dask_arr_for_chain: Optional[Any],
    seen_chains: Dict[Tuple[str, int], Any],
) -> Optional[BranchSpec]:
    """Try to fold the branch's reduction into a chain-folded BranchSpec.

    Returns ``None`` if the registered array has no dask graph (no
    chain to walk) or if ``_walk_chain`` refuses to fold (cross-array,
    constant, etc.). The caller falls back to the length-1 path on
    ``None``.
    """
    if dask_arr_for_chain is None:
        return None
    graph = dask_arr_for_chain.__dask_graph__()
    # Find the aggregate layer name from the branch's chunk_kwargs.
    # ``extract_reduction_hints`` stores the agg-layer name implicitly via the chunk_func's identity.
    # For chain walking we need the explicit aggregate layer name. Walk the graph looking for any aggregate layer
    # reachable from the primary array.
    agg_name = _find_primary_aggregate(graph)
    if agg_name is None:
        return None
    chain = _walk_chain(graph, agg_name)
    if chain is None:
        return None
    # The walker returns layers from root-to-chunk. The chain already covers the pointwise steps.
    # The reduction's chunk_func is the last step. We build a single branch_func via ``_build_chain_branch_func``.
    # If a memoized branch exists for this chain, reuse it.
    chain_key = (agg_name, len(chain))
    chain_branch_func = seen_chains.get(chain_key)
    if chain_branch_func is None:
        chain_branch_func = _build_chain_branch_func(chain)
        seen_chains[chain_key] = chain_branch_func
    try:
        return _build_chain_branch(
            branch=branch,
            chain=chain,
            input_name=primary,
            array_ndim=array_ndim,
            placeholder=placeholder,
            chain_branch_func=chain_branch_func,
        )
    except Exception:
        return None


def _try_length1_branch(
    branch: Dict[str, Any],
    chunk_func: Callable,
    primary: str,
    array_ndim: int,
    placeholder: Optional[Any],
) -> Optional[BranchSpec]:
    """Build a length-1 BranchSpec from a per-reduction branch.

    The branch_func is the branch's chunk_func wrapped in a vpickle-friendly closure (``_make_branch_func``)
    that binds the chunk_kwargs (axis, keepdims, dtype, ...). The bridge calls this branch_func with just the chunk
    and no extra kwargs.
    """
    try:
        return _build_length1_branch(
            branch=branch,
            chunk_func=chunk_func,
            input_name=primary,
            array_ndim=array_ndim,
            placeholder=placeholder,
        )
    except Exception as e:
        # The length-1 path is the last-resort fallback. If it
        # raises, log enough context to diagnose the failure --
        # usually the chunk_func rejected the placeholder's shape or
        # dtype, or the placeholder is itself a dask array (because
        # .compute() silently failed upstream).
        logger.debug(
            "_try_length1_branch: build_length1_branch raised for %s "
            "with chunk_kwargs=%r, array_ndim=%d, placeholder=%r: %s",
            branch.get("output_key"),
            branch.get("chunk_kwargs"),
            array_ndim,
            type(placeholder).__name__ if placeholder is not None else None,
            e,
        )
        return None


def _find_primary_aggregate(graph) -> Optional[str]:
    """Return the first ``-aggregate-`` layer name in the graph.

    Today every detected reduction is rooted at one array. The analyzer emits hints per-array-info.
    The chain walker only needs ONE aggregate layer per branch to start walking back from, and when a callback has
    multiple independent reductions (e.g. ``energy = (arr**2).sum(); drift = arr.mean(axis=0)``), the walker will refuse
    chains it can't fold and the caller falls back to the length-1 path for the rest.
    """
    for layer_name in graph.layers:
        if _is_aggregate_layer(layer_name):
            return layer_name
    return None


def _walk_chain(graph, agg_name: str) -> Optional[List[Tuple[Callable, dict, int]]]:
    """Walk from a chunk layer back to the placeholder root.

    Returns a list of ``(func, kwargs, input_count)`` triples in root-to-chunk order, or ``None`` if the chain can't be
    folded (cross-array, scalar constants, non-Blockwise upstream, etc.).

    The returned chain is built entirely from the dask task graph. No task is ever executed.
    ``input_count`` records how many upstream references the layer has: 1 for a normal single-input pointwise op,
    2 for a self-referential op like ``arr * arr``.
    """
    if "-aggregate-" not in agg_name:
        return None
    chunk_layer_name = _find_chunk_layer(graph, agg_name.split("-aggregate-", 1)[0], exact=True)
    if chunk_layer_name is None:
        return None
    chain: List[Tuple[Callable, dict, int]] = []
    current = chunk_layer_name
    seen = set()
    while current is not None and current not in seen:
        seen.add(current)
        layer = graph.layers[current]
        if "-aggregate-" in current:
            break
        func, kwargs = _extract_layer_func(layer)
        if func is None:
            break
        upstream = _find_single_upstream(layer)
        if upstream is None:
            return None
        upstream_name, upstream_input_count = upstream
        # If the upstream isn't a layer in the graph, we've hit the root (placeholder DataNode).
        # The branch_func receives the actual chunk from the bridge, so we don't fold the root.
        if upstream_name not in graph.layers:
            break
        chain.append((func, kwargs, upstream_input_count))
        current = upstream_name
    chain.reverse()
    return chain


def _extract_layer_func(layer) -> Tuple[Optional[Callable], dict]:
    """Pull the first task's ``func`` and ``kwargs`` out of a Blockwise layer.
    Returns ``(None, {})`` if the layer has no task-shaped values.
    """
    for value in layer.values():
        if hasattr(value, "func") and callable(value.func):
            kwargs = dict(value.kwargs) if value.kwargs else {}
            return value.func, kwargs
    return None, {}


def _find_single_upstream(layer) -> Optional[Tuple[str, int]]:
    """Return ``(upstream_layer_name, array_input_count)`` if the layer reads from a single upstream Blockwise
    (one or more times. e.g. ``arr * arr`` reads from ``arr`` twice and is still chunk-local).
    Returns ``None`` if the layer reads from multiple distinct array upstreams (cross-array, can't fold) or contains
    scalar constants (deferred to a later commit).

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
        # Refuse to fold chains with constants. The constant would need to be embedded into the branch_func call
        # (e.g. ``add(chunk, 1)``) which requires extracting the constant value from the Blockwise task's args.
        return None
    return next(iter(upstream_names)), array_input_count


def _build_chain_branch_func(chain: List[Tuple[Callable, dict, int]]) -> Callable[[Any], Any]:
    """Compose a list of ``(func, kwargs, input_count)`` into a single branch_func(chunk).
    Layers reading from a single upstream twice (e.g. ``arr * arr``) get the chunk passed twice.

    Returns a module-level callable (``_chain_branch_func``) bound to the chain tuple via :func:`functools.partial`.
    The closure is on a top-level function so pickle can find it across processes. Building a fresh
    ``def branch_func(chunk, _chain=...)`` inside this helper would produce an unpicklable local function
    (AttributeError: Can't get local object). The ``functools.partial`` + module-level target recipe is the only shape
    that pickles cleanly.
    """
    chain_tuple = tuple(chain)
    return functools.partial(_chain_branch_func, _chain=chain_tuple)


def _chain_branch_func(chunk, _chain=None):
    """Module-level branch callable: apply each (func, kwargs, input_count) in the chain to the chunk,
    threading the result through.

    Pair with :func:`_build_chain_branch_func` which binds ``_chain`` via :func:`functools.partial`.
    Defined at module level so pickle can find it across the bridge process boundary.
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


def _build_chain_branch(
    branch: Dict[str, Any],
    chain: List[Tuple[Callable, dict, int]],
    input_name: str,
    array_ndim: int,
    placeholder: Optional[Any] = None,
    chain_branch_func: Optional[Callable] = None,
) -> BranchSpec:
    """Build a chain-folded :class:`BranchSpec` from a branch and a layer chain.

    The chain's ``branch_func`` is the composition of the layer funcs (root-to-chunk).
    The branch provides the reduction's ``kind``/``finalize``/``chunk_axis`` metadata; ``keepdims=True`` is forced for
    mean/moment (same as the length-1 path).

    If ``chain_branch_func`` is provided (the memoized version), use it directly instead of rebuilding.
    Otherwise build a fresh callable from ``chain``.
    """
    kind = branch.get("kind", _BRANCH_KIND_SCALAR)
    finalize = branch.get("finalize")

    ck = branch.get("chunk_kwargs") or {}
    ax = ck.get("axis")
    if isinstance(ax, (list, tuple)):
        chunk_axis = tuple(int(a) for a in ax)
    elif ax is not None:
        chunk_axis = (int(ax),)
    else:
        chunk_axis = None

    effective_kwargs = dict(ck)
    if kind in (_BRANCH_KIND_MEAN, _BRANCH_KIND_MOMENT):
        effective_kwargs["keepdims"] = True

    if chain_branch_func is None:
        chain_branch_func = _build_chain_branch_func(chain)

    if placeholder is None:
        partial_shape: Tuple[int, ...] = tuple(branch.get("shape") or ())
        partial_dtype = str(branch.get("dtype", "float64"))
    else:
        sample = chain_branch_func(placeholder)
        if isinstance(sample, dict):
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

    output_shape = _derive_combined_output_shape(chunk_axis, array_ndim, partial_shape)
    output_dtype = partial_dtype

    return BranchSpec(
        input_name=input_name,
        output_key=branch["output_key"],
        output_kind=kind,
        branch_func=chain_branch_func,
        chunk_axis=chunk_axis,
        finalize=finalize,
        partial_shape=partial_shape,
        partial_dtype=partial_dtype,
        output_shape=output_shape,
        output_dtype=output_dtype,
    )
