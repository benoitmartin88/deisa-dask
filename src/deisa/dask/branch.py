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

In Stage 2A this is the **length-1** case: each BranchSpec corresponds
to one detected reduction (``arr.sum()``, ``arr.mean(axis=0)``, ...).
The branch is a single chunk-stage callable followed by the
reduction's combine aggregator. Stage 3 will fold multi-layer chains
into length->=2 branches; the data structure below is designed to
support that without further changes.

The structure mirrors the prior per-reduction hint metadata
(``kind`` / ``finalize`` / ``shape`` / ``dtype`` / ``chunk_axis``) so
the bridge and Deisa-side combine code paths can be refactored to
consume :class:`BranchSpec` directly without changing semantics.

See ``references/branch-level-precompute-design.md`` for the full design.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Output kind values. "scalar" / "mean" / "moment" cover the per-reduction
# cases (Stage 2A). Stage 3 may add "scalar-array" for chained
# pointwise + reduction outputs that are 1-d (e.g.
# ``arr.mean(axis=0)``), but that's not implemented yet.
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
        callable (``chunk_func`` from the prior hint); Stage 3 may
        produce multi-callable composites.
    chunk_axis : Optional[Tuple[int, ...]]
        For reductions, the tuple of axes being reduced in the chunk
        (e.g. ``(0, 1)`` for full reduction on a 2-D chunk). ``None``
        for pointwise-only branches (Stage 3).
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


def _derive_combined_output_shape(
    chunk_axis: Optional[Tuple[int, ...]],
    array_ndim: int,
    partial_shape: Tuple[int, ...],
) -> Tuple[int, ...]:
    """Compute the shape of the combined reduction output.

    For full reductions (``chunk_axis == (0, 1, ...)`` matching all
    axes), the output is scalar ``()``.

    For axis reductions, the output keeps the un-reduced axes' sizes
    from the partial (each bridge's chunk has the full size along
    non-reduced axes, so the partial shape encodes the combined shape).
    """
    if chunk_axis is None:
        # Pointwise branch (Stage 3) — output shape == partial shape.
        return partial_shape
    kept_axes = tuple(ax for ax in range(array_ndim) if ax not in chunk_axis)
    if not kept_axes:
        return ()
    return tuple(partial_shape[ax] for ax in kept_axes)


def _branch_func_with_kwargs(chunk, _cf=None, _kw=None):
    """Top-level branch callable: ``chunk_func(chunk, **chunk_kwargs)``.

    Defined at module level so pickle can find it across processes.
    The default-arg trick (``_cf=chunk_func``, ``_kw=chunk_kwargs``)
    binds the closure cells at function-definition time, which pickle
    serializes correctly.
    """
    if _cf is None or _kw is None:
        # Defensive: a stray call with no closure should never happen
        # (BranchSpec.branch_func is always built via the helper), but
        # be loud rather than silent.
        raise RuntimeError("_branch_func_with_kwargs called without bound chunk_func / chunk_kwargs")
    return _cf(chunk, **_kw)


def _make_branch_func(chunk_func: Callable, chunk_kwargs: Dict[str, Any]) -> Callable[[Any], Any]:
    """Build a pickle-safe branch callable that closes over ``chunk_func``
    and ``chunk_kwargs``.

    Returns a :func:`_branch_func_with_kwargs` partial with the
    chunk_func and chunk_kwargs bound via the default-arg trick.
    Returns a ``functools.partial`` so pickle can find the closure
    contents at unpickle time.
    """
    import functools as _functools

    return _functools.partial(_branch_func_with_kwargs, _cf=chunk_func, _kw=chunk_kwargs)


def build_branch_from_hint(
    hint: Dict[str, Any],
    chunk_func: Callable[[Any], Any],
    input_name: str,
    array_ndim: int,
    placeholder: Optional[Any] = None,
) -> BranchSpec:
    """Build a :class:`BranchSpec` from a per-reduction hint dict.

    For Stage 2A, the branch_func is the same length-1 chunk_func the
    hint already carries. The BranchSpec re-exposes the hint's
    ``kind``/``finalize``/``shape``/``dtype`` metadata in the
    structured form the bridge and Deisa side will consume.

    If ``placeholder`` is provided, we run the chunk_func on it to
    discover the partial's actual shape/dtype. The placeholder is a
    dask array or numpy array representing the registered chunk
    (typically a zero-filled dask array of the right shape). This
    is purely a structural inspection -- the chunk_func is a numpy
    operation, not a callback, so it has no side effects.

    For ``mean``/``moment`` partials the chunk_func returns a dict
    (``{n, total}`` or ``{n, total, M}``); we record the per-bridge
    partial's shape as the shape of the ``total`` value (the
    representative per-key shape, since all keys share the same
    reduced-axes pattern).
    """

    kind = hint.get("kind", _BRANCH_KIND_SCALAR)
    finalize = hint.get("finalize")

    # Reconstruct chunk_axis from chunk_kwargs['axis']. The hint carries
    # ``axis`` as a tuple/list/int; normalise to a tuple of ints.
    chunk_kwargs = hint.get("chunk_kwargs") or {}
    ax = chunk_kwargs.get("axis")
    if isinstance(ax, (list, tuple)):
        chunk_axis = tuple(int(a) for a in ax)
    elif ax is not None:
        chunk_axis = (int(ax),)
    else:
        chunk_axis = None

    # Discover the partial's shape and dtype by running the chunk_func
    # on the placeholder. This is structural -- chunk_func is a numpy
    # op, not the user's callback, so no side effects.
    # For ``mean`` and ``moment`` the bridge overrides ``keepdims=True``
    # (see :meth:`Bridge._execute_operations_on_chunk`) so the per-bridge
    # dict values are at least 1-D for ``mean_agg`` / ``moment_agg`` to
    # walk with ``_concatenate2``. Mirror that here.
    effective_kwargs = dict(chunk_kwargs)
    if kind in (_BRANCH_KIND_MEAN, _BRANCH_KIND_MOMENT):
        effective_kwargs["keepdims"] = True

    # Wrap the raw chunk_func in a closure that binds the analyzer's
    # chunk_kwargs (axis, keepdims, dtype, ...). The bridge calls
    # ``branch.branch_func(chunk)`` with no extra kwargs; this closure
    # carries them. Pickle-friendly: a function with a closure of two
    # picklable objects (a functools.partial and a dict). Uses the
    # top-level ``_make_branch_func`` helper so the closure is
    # picklable across processes.
    branch_func = _make_branch_func(chunk_func, effective_kwargs)

    if placeholder is None:
        # Fallback: best-effort shape from hint metadata. The hint's
        # ``shape`` field is not populated by the analyzer; only the
        # bridge records it after running the chunk_func. We try to
        # recover it from ``keepdims`` + ``chunk_axis`` + the chunk's
        # shape (which the hint doesn't carry either). Without a
        # placeholder we can't compute shape reliably, so we leave it
        # as ``()`` and let the bridge's run-time inspection overwrite
        # it.
        partial_shape: Tuple[int, ...] = tuple(hint.get("shape") or ())
        partial_dtype = str(hint.get("dtype", "float64"))
    else:
        sample = branch_func(placeholder)
        if isinstance(sample, dict):
            # mean / moment -- the per-bridge partial is a dict with
            # per-key shape. Pick ``total`` as the representative
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

    output_shape = _derive_combined_output_shape(chunk_axis, array_ndim, partial_shape)
    output_dtype = partial_dtype  # mean/moment keep dtype through the agg

    return BranchSpec(
        input_name=input_name,
        output_key=hint["output_key"],
        output_kind=kind,
        branch_func=branch_func,
        chunk_axis=chunk_axis,
        finalize=finalize,
        partial_shape=partial_shape,
        partial_dtype=partial_dtype,
        output_shape=output_shape,
        output_dtype=output_dtype,
    )


def branch_to_hint(branch: BranchSpec) -> Dict[str, Any]:
    """Convert a BranchSpec back to the legacy per-reduction hint dict.

    Used as a backward-compat shim for callers that haven't migrated
    to BranchSpec yet. The returned dict has the same schema as
    :func:`deisa.dask.task_hints.extract_reduction_hints` (modulo the
    addition of ``kind``, ``finalize``, ``chunk_axis``, ``shape``,
    ``dtype``).

    For Stage 2A the conversion is lossless because BranchSpec is a
    superset of the hint fields. Stage 3 will lose information when
    folding multi-layer chains.
    """
    import pickle as _pickle

    return {
        "output_key": branch.output_key,
        "op_name": branch.output_key.split("-")[-1] if "-" in branch.output_key else branch.output_key,
        "kind": branch.output_kind,
        "chunk_func_pickle": _pickle.dumps(branch.branch_func),
        "agg_pickle": b"",  # unused in the precompute path; placeholder for back-compat
        "agg_dep_structures": [],
        "agg_keys": [],
        "finalize": branch.finalize,
        "chunk_kwargs": {"axis": branch.chunk_axis} if branch.chunk_axis else {},
        "shape": branch.partial_shape,
        "dtype": branch.partial_dtype,
        "keywords": {},
    }


def hint_to_branch(
    hint: Dict[str, Any], input_name: str, array_ndim: int, placeholder: Optional[Any] = None
) -> BranchSpec:
    """Convert a legacy per-reduction hint dict into a BranchSpec.

    Inverse of :func:`branch_to_hint`. Used by the bridge to convert
    legacy stored hints on the HandshakeActor (kept for backward compat
    with the prior hint-based registration path) into BranchSpec
    instances.
    """
    import pickle as _pickle

    chunk_func = _pickle.loads(hint["chunk_func_pickle"])
    return build_branch_from_hint(
        hint=hint,
        chunk_func=chunk_func,
        input_name=input_name,
        array_ndim=array_ndim,
        placeholder=placeholder,
    )


def analyze_branch(
    callback: Callable,
    registered_arrays: Dict[str, Any],
    force: bool = False,
) -> List[BranchSpec]:
    """Walk the callback's dask graph and emit a :class:`BranchSpec` per
    branch.

    Stage 2B implementation: every reduction hint becomes one
    :class:`BranchSpec`. For each hint, the chain walker (``_walk_chain``)
    inspects the registered dask array's task graph and, if the
    reduction's chunk-stage has a single-input upstream chain of
    pointwise layers, **folds the chain into one branch** -- the
    branch_func applies every chain layer to the chunk on the bridge
    side and ships a single (small) partial instead of relying on dask
    workers to re-run the chunk-stage pointwise chain.

    Folding is opportunistic. If ``_walk_chain`` returns ``None`` for a
    hint (cross-array upstream, scalar constant, non-Blockwise
    upstream, etc.) the branch degrades to the length-1 path: just the
    reduction's chunk_func. That's still a correct optimization -- the
    chain walker only adds coverage, it never removes it.

    Parameters
    ----------
    callback : Callable
        The user's callback function. Not invoked.
    registered_arrays : Dict[str, Any]
        Mapping of name -> dask array (or other placeholder) for each
        registered array. Dask arrays become the roots of the task
        graph walk; non-dask values are treated as opaque helpers
        (their attributes may be read for opaque-resolvable branches).
    force : bool
        If True, skip unresolvable reductions with a warning instead of
        raising. Matches :func:`analyze_callback`'s ``force`` semantics.

    Returns
    -------
    List[BranchSpec]
        One BranchSpec per detected reduction. Empty list if the
        callback contains no chunk-local precomputable operations.
    """
    import pickle as _pickle

    hints = extract_reduction_hints_from_callback(callback, registered_arrays, force=force)
    if not hints:
        return []

    # The chain walker needs the **AST walker's** dask_arrays -- the
    # dask expressions the walker built during symbolic evaluation
    # (e.g. ``(arr*arr).sum()``). The registered placeholders' graphs
    # only contain the root ``zeros_like`` layer; the chain layer(s)
    # are added when the walker composes the expression. Run the AST
    # walker explicitly so we have both the hints AND the dask_arrays
    # from the same walk.
    walker_dask_arrays: List[Dict[str, Any]] = []
    try:
        walker_dask_arrays = extract_dask_arrays_from_callback(callback, registered_arrays, force=force)
    except Exception:
        if not force:
            raise
        # force=True: fall through with empty walker_dask_arrays; the
        # length-1 fallback still works.

    # Pick the registered-array name and ndim to attach to branches.
    primary = next(iter(registered_arrays)) if registered_arrays else "f"
    primary_arr: Optional[Any] = None
    array_ndim = 0
    try:
        primary_arr = next(iter(registered_arrays.values()))
        array_ndim = int(getattr(primary_arr, "ndim", 0))
    except Exception:
        pass
    # The chain walker needs at least one dask expression to walk. The
    # AST walker builds one dask_arrays entry per compute boundary;
    # if there's nothing in the list, the registered placeholder is
    # the only thing available, but its graph is the root layer (no
    # chain). We fall back to the length-1 path in that case.
    # We use the FIRST walker dask_array as the graph to walk; this
    # matches the analyzer's behavior of treating the first compute
    # boundary as the primary one.
    dask_arr_for_chain: Optional[Any] = None
    if walker_dask_arrays:
        candidate = walker_dask_arrays[0].get("array")
        if hasattr(candidate, "__dask_graph__"):
            dask_arr_for_chain = candidate

    # The chain walker folds multi-layer pointwise chains into one
    # branch_func. We dedupe chains per-hint: a chain is unique by its
    # (agg-layer-name, length). Multiple hints can share the same chain
    # (e.g. ``(arr**2).sum()`` and ``(arr**2).max()``); the walker
    # builds the same branch_func either way. ``_seen_chains`` keeps a
    # memo to avoid rebuilding identical functools.partial objects.
    _seen_chains: Dict[Tuple[str, int], Any] = {}

    branches: List[BranchSpec] = []
    for hint in hints:
        try:
            chunk_func = _pickle.loads(hint["chunk_func_pickle"])
        except Exception as e:  # pragma: no cover - safety net
            if force:
                logger.warning("analyze_branch: unpickle failed for %s: %s", hint.get("output_key"), e)
                continue
            raise

        # Discover the partial's shape and dtype by running the
        # branch_func on a numpy placeholder. The placeholder is the
        # dask array's first chunk, materialized via .compute() so the
        # numpy ops return numpy values (chunk_funcs are numpy ops,
        # not callback code).
        placeholder = primary_arr
        if hasattr(placeholder, "compute"):
            try:
                placeholder = placeholder.compute()
            except Exception:
                pass

        branch = _try_chain_branch(
            hint=hint,
            chunk_func=chunk_func,
            primary=primary,
            array_ndim=array_ndim,
            placeholder=placeholder,
            dask_arr_for_chain=dask_arr_for_chain,
            seen_chains=_seen_chains,
        )
        if branch is None:
            # Chain walker refused; fall back to the length-1 path.
            branch = _try_length1_branch(
                hint=hint,
                chunk_func=chunk_func,
                primary=primary,
                array_ndim=array_ndim,
                placeholder=placeholder,
            )
        if branch is None:
            if force:
                logger.warning("analyze_branch: build failed for %s", hint.get("output_key"))
                continue
            # Both paths returned None -- this shouldn't happen for
            # hints that came out of the analyzer (the length-1 path is
            # supposed to always succeed). Raise defensively.
            raise RuntimeError(f"analyze_branch: cannot build branch for hint {hint.get('output_key')!r}")
        branches.append(branch)
    return branches


def _try_chain_branch(
    hint: Dict[str, Any],
    chunk_func: Callable,
    primary: str,
    array_ndim: int,
    placeholder: Optional[Any],
    dask_arr_for_chain: Optional[Any],
    seen_chains: Dict[Tuple[str, int], Any],
) -> Optional[BranchSpec]:
    """Try to fold the hint's reduction into a chain-folded BranchSpec.

    Returns ``None`` if the registered array has no dask graph (no
    chain to walk) or if ``_walk_chain`` refuses to fold (cross-array,
    constant, etc.). The caller falls back to the length-1 path on
    ``None``.
    """
    if dask_arr_for_chain is None:
        return None
    graph = dask_arr_for_chain.__dask_graph__()
    # Find the aggregate layer name from the hint's chunk_kwargs.
    # ``extract_reduction_hints`` stores the agg-layer name implicitly
    # via the chunk_func's identity; for chain walking we need the
    # explicit aggregate layer name. Walk the graph looking for any
    # aggregate layer reachable from the primary array.
    agg_name = _find_primary_aggregate(graph)
    if agg_name is None:
        return None
    chain = _walk_chain(graph, agg_name)
    if chain is None:
        return None
    # The walker returns layers from root-to-chunk-stage. The chain
    # already covers the pointwise steps; the reduction's chunk_func
    # IS the last step. We build a single branch_func via
    # ``_build_chain_branch_func``. If a memoized branch exists for
    # this chain, reuse it.
    chain_key = (agg_name, len(chain))
    chain_branch_func = seen_chains.get(chain_key)
    if chain_branch_func is None:
        chain_branch_func = _build_chain_branch_func(chain)
        seen_chains[chain_key] = chain_branch_func
    try:
        return _build_chain_branch(
            hint=hint,
            chain=chain,
            input_name=primary,
            array_ndim=array_ndim,
            placeholder=placeholder,
            chain_branch_func=chain_branch_func,
        )
    except Exception:
        return None


def _try_length1_branch(
    hint: Dict[str, Any],
    chunk_func: Callable,
    primary: str,
    array_ndim: int,
    placeholder: Optional[Any],
) -> Optional[BranchSpec]:
    """Build a length-1 BranchSpec from a per-reduction hint.

    The branch_func is the hint's chunk_func wrapped in a
    pickle-friendly closure (``_make_branch_func``) that binds the
    chunk_kwargs (axis, keepdims, dtype, ...). The bridge calls this
    branch_func with just the chunk and no extra kwargs.
    """
    try:
        return build_branch_from_hint(
            hint=hint,
            chunk_func=chunk_func,
            input_name=primary,
            array_ndim=array_ndim,
            placeholder=placeholder,
        )
    except Exception:
        return None


def _find_primary_aggregate(graph) -> Optional[str]:
    """Return the first ``-aggregate-`` layer name in the graph.

    Today every detected reduction is rooted at one array; the
    analyzer emits hints per-array-info. The chain walker only needs
    ONE aggregate layer per branch to start walking back from -- and
    when a callback has multiple independent reductions (e.g.
    ``energy = (arr**2).sum(); drift = arr.mean(axis=0)``), the
    walker will refuse chains it can't fold and the caller falls back
    to the length-1 path for the rest.
    """
    for layer_name in graph.layers:
        if "-aggregate-" in layer_name:
            return layer_name
    return None


# ---------------------------------------------------------------------------
# Chain-folding (Stage 2B)
# ---------------------------------------------------------------------------
def _walk_chain(graph, agg_name: str) -> Optional[List[Tuple[Callable, dict, int]]]:
    """Walk from a chunk-stage layer back to the placeholder root.

    Returns a list of ``(func, kwargs, input_count)`` triples in
    root-to-chunk-stage order, or ``None`` if the chain can't be folded
    (cross-array, scalar constants, non-Blockwise upstream, etc.).

    The returned chain is built entirely from the dask task graph;
    no task is ever executed. ``input_count`` records how many upstream
    references the layer has -- 1 for a normal single-input pointwise
    op, 2 for a self-referential op like ``arr * arr``.
    """
    if "-aggregate-" not in agg_name:
        return None
    chunk_layer_name = _find_chunk_layer(graph, agg_name.split("-aggregate-", 1)[0])
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
        # If the upstream isn't a layer in the graph, we've hit the
        # root (placeholder DataNode). The branch_func receives the
        # actual chunk from the bridge, so we don't fold the root.
        if upstream_name not in graph.layers:
            break
        chain.append((func, kwargs, upstream_input_count))
        current = upstream_name
    chain.reverse()
    return chain


def _find_chunk_layer(graph, agg_base: str) -> Optional[str]:
    """Find the chunk-stage layer whose name matches ``agg_base``.

    The existing ``extract_reduction_hints`` has a richer version of
    this that also handles ``-`` suffix variants; for chain folding we
    only need the exact-base match.
    """
    for layer_name in graph.layers:
        if "-aggregate-" in layer_name:
            continue
        layer_base = layer_name.rsplit("-", 1)[0] if "-" in layer_name else layer_name
        if layer_base == agg_base:
            return layer_name
    return None


def _extract_layer_func(layer) -> Tuple[Optional[Callable], dict]:
    """Pull the first task's ``func`` and ``kwargs`` out of a Blockwise
    layer. Returns ``(None, {})`` if the layer has no task-shaped values.
    """
    for value in layer.values():
        if hasattr(value, "func") and callable(value.func):
            kwargs = dict(value.kwargs) if value.kwargs else {}
            return value.func, kwargs
    return None, {}


def _find_single_upstream(layer) -> Optional[Tuple[str, int]]:
    """Return ``(upstream_layer_name, array_input_count)`` if the
    layer reads from a single upstream Blockwise (one or more times --
    e.g. ``arr * arr`` reads from ``arr`` twice and is still
    chunk-local). Returns ``None`` if the layer reads from multiple
    distinct array upstreams (cross-array, can't fold) or contains
    scalar constants (deferred to a later commit).
    """
    if not hasattr(layer, "indices") or not layer.indices:
        return None
    in_keys = list(layer.indices)
    if len(in_keys) == 0:
        return None
    upstream_names = set()
    array_input_count = 0
    has_non_array_input = False
    for in_key in in_keys:
        # An ``in_key`` is an "array input" only if it has a string
        # first element (dask layer names are strings; constants
        # like ``(2, None)`` have a non-string first element).
        if isinstance(in_key, (list, tuple)) and len(in_key) >= 1 and isinstance(in_key[0], str):
            upstream_names.add(in_key[0])
            array_input_count += 1
        else:
            # Scalar constant or other non-array input.
            has_non_array_input = True
    if not upstream_names or len(upstream_names) > 1:
        return None
    if has_non_array_input:
        # Stage 2B: refuse to fold chains with constants. The
        # constant would need to be embedded into the branch_func
        # call (e.g. ``add(chunk, 1)``) which requires extracting the
        # constant value from the Blockwise task's args. Defer to a
        # later commit.
        return None
    return (next(iter(upstream_names)), array_input_count)


def _build_chain_branch_func(chain: List[Tuple[Callable, dict, int]]) -> Callable[[Any], Any]:
    """Compose a list of ``(func, kwargs, input_count)`` into a single
    branch_func(chunk). Layers reading from a single upstream twice
    (e.g. ``arr * arr``) get the chunk passed twice.

    Returns a module-level callable (``_chain_branch_func``) bound to
    the chain tuple via :func:`functools.partial`. The closure is on a
    top-level function so pickle can find it across processes. Building
    a fresh ``def branch_func(chunk, _chain=...)`` inside this helper
    would produce an unpicklable local function (AttributeError: Can't
    get local object). The ``functools.partial`` + module-level target
    recipe is the only shape that pickles cleanly.
    """
    chain_tuple = tuple(chain)
    return functools.partial(_chain_branch_func, _chain=chain_tuple)


def _chain_branch_func(chunk, _chain=None):
    """Module-level branch callable: apply each (func, kwargs, input_count)
    in the chain to the chunk, threading the result through.

    Pair with :func:`_build_chain_branch_func` which binds ``_chain``
    via :func:`functools.partial`. Defined at module level so pickle
    can find it across the bridge process boundary.
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
    hint: Dict[str, Any],
    chain: List[Tuple[Callable, dict, int]],
    input_name: str,
    array_ndim: int,
    placeholder: Optional[Any] = None,
    chain_branch_func: Optional[Callable] = None,
) -> BranchSpec:
    """Build a chain-folded :class:`BranchSpec` from a hint and a
    layer chain.

    The chain's ``branch_func`` is the composition of the layer
    funcs (root-to-chunk-stage). The hint provides the reduction's
    ``kind``/``finalize``/``chunk_axis`` metadata; ``keepdims=True``
    is forced for mean/moment (same as the length-1 path).

    If ``chain_branch_func`` is provided (the memoized version), use
    it directly instead of rebuilding. Otherwise build a fresh
    callable from ``chain``.
    """
    kind = hint.get("kind", _BRANCH_KIND_SCALAR)
    finalize = hint.get("finalize")

    ck = hint.get("chunk_kwargs") or {}
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
        partial_shape: Tuple[int, ...] = tuple(hint.get("shape") or ())
        partial_dtype = str(hint.get("dtype", "float64"))
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
        output_key=hint["output_key"],
        output_kind=kind,
        branch_func=chain_branch_func,
        chunk_axis=chunk_axis,
        finalize=finalize,
        partial_shape=partial_shape,
        partial_dtype=partial_dtype,
        output_shape=output_shape,
        output_dtype=output_dtype,
    )


def extract_reduction_hints_from_callback(
    callback: Callable,
    registered_arrays: Dict[str, Any],
    force: bool = False,
) -> List[Dict[str, Any]]:
    """Wrapper around the legacy analyzer that returns the raw hint list.

    The legacy ``analyze_callback`` does two things: walks the AST to
    detect compute boundaries, then walks the resulting dask graphs to
    extract reduction hints. We only need the hint list here, but the
    AST walk is required to find the dask arrays in the first place.
    For Stage 2A we delegate to the legacy entry point and discard the
    AST-walk side effects (it has no externally observable effects
    beyond emitting the hints).
    """
    from deisa.dask.precompute_analyzer import analyze_callback

    try:
        hints = analyze_callback(callback, registered_arrays, force=force)
    except Exception:
        if force:
            logger.warning("analyze_branch: analyze_callback failed; force=True, returning empty")
            return []
        raise
    return hints


def extract_dask_arrays_from_callback(
    callback: Callable,
    registered_arrays: Dict[str, Any],
    force: bool = False,
) -> List[Dict[str, Any]]:
    """Return the AST walker's dask_arrays list for a callback.

    Each entry is ``{"array": darr, "kind": ..., "lineno": ...}`` where
    ``darr`` is the dask expression the walker built at a compute
    boundary (e.g. ``(arr*arr).sum()``). The chain walker in
    :func:`_walk_chain` needs these expressions' graphs (not the
    registered placeholders' graphs) because the placeholders only
    have the root layer, not the chain.
    """
    from deisa.dask.precompute_analyzer import analyze_callback_with_dask_arrays

    try:
        _hints, dask_arrays = analyze_callback_with_dask_arrays(callback, registered_arrays, force=force)
    except Exception:
        if force:
            logger.warning("analyze_branch: AST walk failed; force=True, returning empty dask_arrays")
            return []
        raise
    return dask_arrays
