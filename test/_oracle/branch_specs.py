"""The precompute analyzer's branch-spec builders, kept alive for the equivalence harness ONLY.

Nothing under ``src/`` may import this module.

These functions were moved VERBATIM out of ``deisa/dask/branch.py`` when plan P4 deleted the discovery machinery from
the shipped package. They exist for one reason: the K6a equivalence harness (``test/utils.py::_analyze``) needs a real
oracle -- the analyzer's own branch specs, computed by the analyzer's own code -- against which the runtime planner's
plan is compared, on metadata AND on the partial a branch callable produces from a fresh chunk. Frozen golden specs
were considered and rejected: they would replace a mutation-probed oracle with expectations nobody has validated, and
the product intent (no discovery machinery in the shipped package) is met by a test-tree copy either way.

Deleting this module means deleting the oracle; do not do that without replacing the equivalence harness.
"""

from __future__ import annotations

import functools
import logging
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple

import dask.array as da
import numpy as np

from _oracle.branch_spec import BranchSpec
from _oracle.precompute_analyzer import _match_source_arrays, analyze_callback_full
from _oracle.task_branches import (
    _aggregate_output_feeds_other_reduction,
    _chunk_func_and_kwargs,
    _chunk_layer_for_aggregate,
    _classify_chain,
    _find_single_upstream,
    _is_aggregate_layer,
    _normalize_reduction_axis,
    _op_for_aggregate_layer,
)
from deisa.dask.branch import (
    _BRANCH_KIND_MEAN,
    _BRANCH_KIND_MOMENT,
    _BRANCH_KIND_PCA,
    _BRANCH_KIND_SCALAR,
    _FULL_FOLD_FINALIZERS,
    _PCA_OP_NAME,
    _chain_branch_func,
    _discover_partial_metadata,
    _full_fold_layer,
)
from deisa.dask.mergeable_pca import _build_pca_branch_func
from deisa.dask.precompute_errors import UnsupportedReductionError
from deisa.dask.utils import build_deisa_array

logger = logging.getLogger(__name__)


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

    Lives in the oracle: the recorded branch contract is what this builds, and the run time that carries the same
    guarantee is ``runtime_compose._build_chain`` -> :func:`deisa.dask.branch._full_fold_layer` over
    :data:`deisa.dask.branch._FULL_FOLD_FINALIZERS` (both still in ``src/``, live via ``runtime_compose``).
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


def _analyze_callback_for_branches(callback: Callable, registered_arrays: Dict[str, Any]):
    """Analyze the callback's source and build a list of :class:`BranchSpec` objects.

    Each spec describes a chunk-local sub-expression the bridge can execute. The callback is NOT executed: the AST is
    parsed and walked symbolically to find compute boundaries (``.compute()``, ``client.compute()``, ...) and the dask
    arrays they reference.

    Analysis is strict: any analysis failure (a ``PrecomputeError`` subclass or an unexpected exception) propagates to
    the caller. The caller decides the fallback policy (skip analysis entirely for ``precompute=False`` callbacks,
    raise otherwise).
    """

    # Build a dask array stub matching the registered array's shape/chunks so the symbolic AST walker has something
    # concrete to operate on. The chunking does not matter for hint extraction. We only read the task graph structure,
    # not the data.
    stubs = {}
    for arr_name in registered_arrays:
        # Get metadata for this array from registered_arrays (which is arrays_metadata), used for shape/chunks of the
        # placeholder stub.
        meta = registered_arrays.get(arr_name, {})
        global_shape = meta.get("global_shape")
        # The metadata exposes ``chunk_shape`` (per-bridge chunk), not a ``chunks`` tuple. ``da.zeros`` tiles the global
        # shape into uniform chunks of ``chunk_shape``.
        chunk_shape = meta.get("chunk_shape")
        if global_shape is not None and chunk_shape is not None:
            array_stub = da.zeros(global_shape, chunks=chunk_shape, dtype=np.float64, name=f"deisa-stub-{arr_name}")
        else:
            # Fallback for arrays without full metadata (e.g. opaque helpers or dynamically constructed arrays). The
            # stub's exact size doesn't matter. Only the graph structure matters for precompute analysis. Shape and
            # chunks must be consistent (same dimensionality).
            array_stub = da.zeros((10, 10), chunks=(5, 5), dtype=np.float64, name=f"deisa-stub-{arr_name}")
        # Wrap in DeisaArray so the analyzer sees the full attribute surface (.t, .timestep, dask Array methods) that
        # the callback uses at runtime.
        stubs[arr_name] = build_deisa_array(array_stub, timestep=0)

    return _analyze_branch(callback, registered_arrays=stubs)


def _analyze_branch(callback: Callable, registered_arrays: Dict[str, Any]) -> List[BranchSpec]:
    """Walk the callback's dask graph and emit a :class:`BranchSpec` per branch.

    Two kinds of branch come out of one analysis pass, in the same list, because they share everything downstream
    (the bridge call site, the scatter, the merge tree, the dispatch view):

    - reduction branches, from the hint/graph walk below;
    - ``pca`` branches, one per recorded ``MergeablePCA(...).fit(<registered array>)``. Those are built by
      :func:`_build_pca_branch` from the recorded estimator configuration rather than from a graph, because the
      decomposition runs on the BRIDGE and therefore has no Dask graph to walk.
    """
    # Single AST walk: analyze_callback_full returns reduction hints, the walker's dask_arrays AND the recorded PCA
    # requests in one pass. The dask_arrays are the walker's expressions at each compute boundary (e.g.
    # ``(arr*arr).sum()``); the registered placeholders only have the root layer, so the chain walker needs these.
    hints, walker_dask_arrays, pca_requests = analyze_callback_full(callback, registered_arrays)

    if not hints and not pca_requests:
        return []

    # Candidate map keyed by (array_name, op_name): from ALL walker graphs, each aggregate layer maps its canonical op
    # name to its (layer, graph). Keying by array avoids collisions when two arrays run the same reduction (each folds
    # its own chain).
    aggregate_candidates: Dict[Tuple[str, str], List[Tuple[str, Any]]] = {}
    for arr_info in walker_dask_arrays:
        candidate = arr_info.get("array")
        if not hasattr(candidate, "__dask_graph__"):
            continue
        graph = candidate.__dask_graph__()
        matched = _match_source_arrays(candidate, registered_arrays)
        candidates_array_name = matched[0] if matched else next(iter(registered_arrays), "f")
        for layer_name in graph.layers:
            if not _is_aggregate_layer(layer_name):
                continue
            op_name = _op_for_aggregate_layer(graph, layer_name)
            if op_name is None:
                continue
            aggregate_candidates.setdefault((candidates_array_name, op_name), []).append((layer_name, graph))

    # Fold each hint with its OWN aggregate layer (matched by op_name via the candidate map) -- folding a shared chain
    # into every hint corrupts multi-reduction callbacks. ``_seen_chains`` memoizes identical (agg-name, length) pairs
    # so identical expressions reuse one partial.
    _seen_chains: Dict[Tuple[str, int], Any] = {}

    branches: List[BranchSpec] = []
    for branch_dict in hints:
        # Cross-array expressions cannot be rebuilt from one array's chunk-local partials (the bridge only owns its
        # array's chunk): refuse them.
        if branch_dict.get("multi_source"):
            raise UnsupportedReductionError(
                f"Cannot precompute reduction {branch_dict.get('output_key')!r} "
                f"(op {branch_dict.get('op_name')!r}): the expression descends from registered array "
                f"{branch_dict.get('array_name')!r} AND at least one other registered array. A chunk-local branch "
                f"cannot compute a cross-array expression because each bridge only owns its own array's chunk. "
                f"Redesign the callback so every reduction descends from exactly one registered array, or register "
                f"with precompute=False to use the legacy full-chunk scatter path."
            )

        try:
            chunk_func = pickle.loads(branch_dict["chunk_func_pickle"])
        except Exception:  # pragma: no cover - safety net
            raise

        # Resolve THIS hint's own registered-array stub (placeholder and ndim are per-array in multi-array callbacks).
        # The placeholder is the dask array's first chunk, materialized via .compute() so the numpy ops return numpy
        # values.
        hint_arr = branch_dict.get("array_name") or next(iter(registered_arrays), None)
        array_stub = registered_arrays.get(hint_arr)
        array_ndim = int(getattr(array_stub, "ndim", 0)) if array_stub is not None else 0
        placeholder = array_stub
        if hasattr(placeholder, "compute") and array_stub is not None:
            try:
                # Materialize the FIRST CHUNK of the stub, not the whole array: the registered partial metadata must
                # describe the real per-bridge chunk partial (a whole-stub run would record ``(1, 8)`` where bridges
                # ship ``(1, 4)``).
                first_chunk_shape = tuple(int(c[0]) for c in array_stub.chunks)
                placeholder = array_stub[tuple(slice(0, s) for s in first_chunk_shape)]
                # Compute on the sync scheduler -- analysis must never touch an ambient distributed client (a stale
                # config would block in its reconnect loop instead of failing fast).
                placeholder = placeholder.compute(scheduler="sync")
            except Exception:
                # If .compute() fails (e.g. no dask client in the CI worker process), fall back to a synthetic numpy
                # array with the dask array's shape/dtype when retrievable, else a (4, 4) float64 stand-in, so the
                # chunk_func still produces a representative partial.
                shape = getattr(array_stub, "shape", None)
                dtype = getattr(array_stub, "dtype", np.float64)
                placeholder = np.zeros(shape if shape else (4, 4), dtype=dtype)

        branch = _try_chain_branch(
            branch=branch_dict,
            array_ndim=array_ndim,
            placeholder=placeholder,
            aggregate_candidates=aggregate_candidates,
            seen_chains=_seen_chains,
        )
        if branch is None:
            # Chain walker refused; fall back to the length-1 path with the ORIGINAL branch dict. The fallback's
            # ``deliver_direct`` is conservative: direct ONLY if every candidate's chunk stage reads from the registered
            # root (see the gate in deisa.py).
            direct, window_read = _candidate_chain_classify(branch_dict, aggregate_candidates)
            branch = _try_length1_branch(
                branch=branch_dict,
                chunk_func=chunk_func,
                array_ndim=array_ndim,
                placeholder=placeholder,
                deliver_direct=direct,
                window_read=window_read,
            )
        if branch is None:
            # Both paths returned None. This shouldn't happen for hints that came out of the analyzer (the length-1 path
            # is supposed to always succeed). Raise defensively.
            raise RuntimeError(
                f"analyze_branch: cannot build branch for branch {branch_dict.get('output_key')!r}. "
                f"The chain walker refused (likely cross-array or constant upstream) AND the length-1 fallback's "
                f"build raised. This usually means the chunk_func rejected the placeholder. "
                f"Inspect with the failing branch's chunk_kwargs."
            )
        branches.append(branch)

    # PCA branches: one per recorded request, built from the estimator configuration rather than from a graph walk.
    # They are appended AFTER the reductions so a callback that asks for both gets its reductions first, and both share
    # the same ``output_key`` namespace (``{array}-pca`` vs ``{array}-{op}``) through the same dedup.
    for pca_request in pca_requests:
        branches.append(_build_pca_branch(pca_request))
    return branches


def _build_pca_branch(pca_request: Dict[str, Any]) -> BranchSpec:
    """Build the ``pca`` :class:`BranchSpec` for one recorded ``MergeablePCA(...).fit(<registered array>)``.

    The branch is deliberately a LENGTH-1 branch over the module-level :func:`_pca_branch_func` partial, so it reuses
    the existing bridge call site (:meth:`Bridge._execute_operations_on_chunk`, which is just
    ``branch.branch_func(chunk)``) unchanged. There is no chain to fold: the bridge's whole job is to decompose its own
    chunk, so there is nothing upstream of ``local_pca``.

    The payload is the MERGEABLE summary, never the truncated public representation. That is the whole point: a summary
    truncated at the bridge to ``n_components`` cannot be merged, because the discarded directions are gone before the
    cross-bridge correction term ever runs. So ``n_components`` and ``whiten`` are recorded here as ROOT-ONLY
    configuration (``_summary_``-time work on the Dask side) and are deliberately absent from ``branch_func``.

    ``partial_shape`` / ``partial_dtype`` describe the summary's components array, the largest array it carries. The
    Deisa side re-derives the root shape from the merged summary itself, so these only inform the topic-event metadata.

    - ``:param pca_request:`` One entry of ``analyze_callback_full``'s ``pca_requests``: ``array_name``, ``request``
      and ``lineno``.
    """
    config: Dict[str, Any] = dict(pca_request["request"].config)
    array_name = str(pca_request["array_name"])
    branch_func = _build_pca_branch_func(
        config.get("local_rank"),
        config.get("axis_names"),
        config.get("feature_axes"),
        config.get("sample_axes"),
    )
    # ``deliver_direct=True`` and ``reduction_axes=()``: the PCA branch reads the registered chunk ITSELF and consumes
    # the WHOLE feature dimension, so it passes the same "the callback's own op can consume this delivery" gate every
    # reduction branch passes. The runtime dispatch is NOT by ``op_name``: there is no per-branch ``.pca`` method to
    # route to. At run time the bridge's ``PCASummary`` comes back over the runtime PCA request channel and the root
    # merges it (``runtime_dispatch`` PCA request -> ``merge_tree``) before ``MergeablePCA._fit_delivered`` adopts it;
    # no object ever carries the summary on a ``_deisa_pca_summary`` attribute.
    return BranchSpec(
        output_key=f"{array_name}-{_PCA_OP_NAME}",
        input_name=array_name,
        output_kind=_BRANCH_KIND_PCA,
        branch_func=branch_func,
        # No reduction axis: the local PCA needs every row of the chunk as samples, and every column as features.
        chunk_axis=None,
        finalize=None,
        partial_shape=(0, 0),
        partial_dtype="float64",
        op_name=_PCA_OP_NAME,
        deliver_direct=True,
        window_read=False,
        reduction_axes=(),
        # ROOT-ONLY estimator configuration. Not applied on the bridge; consumed by the Deisa-side combine to build the
        # same estimator the user wrote, so the merge result and the public attributes agree with a standalone fit.
        summary_config=config,
    )


def _candidate_chain_classify(branch: Dict[str, Any], aggregate_candidates: Dict) -> Tuple[bool, bool]:
    """Conservative ``(deliver_direct, window_read)`` for the length-1 fallback.

    The chain walker can't tell which candidate aggregate belongs to THIS hint when several reductions share
    ``(array_name, op_name)`` (e.g. ``arr.sum()`` + ``arr.sum(axis=0)``). The reduction is deemed DIRECT only when
    EVERY candidate's chain from the chunk stage to the registered root contains just the reduction chunk stage and/or
    window-read getitem layers (``root[-1].op()``). Zero candidates, an unwalkable chain, a pointwise chain
    (``arr*arr``), or a real slice (``arr[2:5]`` / ``arr[:, 0]``) -> not direct (refused at registration) so a
    chained reduction can never sneak past the gate as "direct".

    ``window_read`` is True when every candidate is a window read (the callback's runtime reduction runs on the WHOLE
    delivered array, so its ``reduction_axes`` is the FULL reduction (``()``) regardless of the stub-side chunk axis).
    Delegates the per-chain classification to the shared walker
    :func:`_oracle.task_branches._classify_chain`.
    """
    op_name = branch.get("op_name")
    array_name = branch.get("array_name")
    if op_name is None:
        return False, False
    candidates = aggregate_candidates.get((array_name, op_name), [])
    if not candidates:
        return False, False
    direct = True
    window_read = True
    for agg_name, graph in candidates:
        chunk_layer = _chunk_layer_for_aggregate(graph, agg_name)
        if chunk_layer is None:
            # An aggregate without a resolvable chunk layer can't be classified: treat as not-direct.
            direct = False
            continue
        d, w = _classify_chain(graph, chunk_layer)
        if not d:
            direct = False
        if not w:
            window_read = False
    return direct, window_read


def _try_chain_branch(
    branch: Dict[str, Any],
    array_ndim: int,
    placeholder: Optional[Any],
    aggregate_candidates: Dict[Tuple[str, str], List[Tuple[str, Any]]],
    seen_chains: Dict[Tuple[str, int], Any],
) -> Optional[BranchSpec]:
    """Try to fold the branch's reduction into a chain-folded BranchSpec."""
    op_name = branch.get("op_name")
    array_name = branch.get("array_name")
    if op_name is None:
        return None
    candidates = aggregate_candidates.get((array_name, op_name), [])
    if len(candidates) != 1:
        return None
    agg_name, graph = candidates[0]
    chain = _walk_chain(graph, agg_name)
    if chain is None:
        return None
    # MINIMAL-SCALAR-PARTIAL (see :func:`_oracle.task_branches.extract_reduction_hints`): a scalar-op reduction
    # computed as a FULL fold over the bridge's whole chunk ships ONE scalar (``tier -> finalizer`` chain layer).
    # Dask's tier kwargs are never rebound; ``max``/``min`` come along for free via the finalizer table. Mean/moment
    # are excluded by ``full_local_fold``'s kind gate (their dict partials must stay tier-shaped).
    chain = _apply_full_local_fold(
        branch.get("op_name"),
        bool(branch.get("full_local_fold")),
        chain,
    )
    # The walker returns layers from root-to-chunk. The chain already covers the pointwise steps. The reduction's
    # chunk_func is the last step. We build a single branch_func via ``_build_chain_branch_func``. If a memoized branch
    # exists for this chain, reuse it.
    chain_key = (agg_name, len(chain))
    chain_branch_func = seen_chains.get(chain_key)
    if chain_branch_func is None:
        chain_branch_func = _build_chain_branch_func(chain)
        seen_chains[chain_key] = chain_branch_func
    try:
        return _build_branch(
            branch=branch,
            array_ndim=array_ndim,
            placeholder=placeholder,
            chain_branch_func=chain_branch_func,
            deliver_direct=(len(chain) == 1),
            window_read=False,
        )
    except Exception:
        return None


def _try_length1_branch(
    branch: Dict[str, Any],
    chunk_func: Callable,
    array_ndim: int,
    placeholder: Optional[Any],
    deliver_direct: bool = False,
    window_read: bool = False,
) -> Optional[BranchSpec]:
    """Build a length-1 BranchSpec from a per-reduction branch.

    A length-1 branch is a chain of length 1: represented as ``[(chunk_func, effective_kwargs, 1)]`` and built via the
    unified :func:`_build_branch`. The chain machinery binds the chunk_kwargs (axis, keepdims, dtype, ...) to the
    chunk_func, so the bridge calls ``branch_func(chunk)`` with just the chunk and no extra kwargs.

    ``deliver_direct`` defaults to False because the length-1 path is reached exactly when the chain walker could not
    PROVE the reduction reads the registered root directly (no/ambiguous candidates, unwalkable chain); the caller
    computes the conservative value via :func:`_candidate_chain_classify`. ``window_read`` marks a whole-row-plane
    ``root[-1]`` read (see :mod:`_oracle.task_branches`): the callback's runtime reduction runs on the WHOLE
    delivered array, so the branch's ``reduction_axes`` is the full reduction (``()``) even though the stub-side
    chunk axis is partial.
    """
    try:
        # For ``mean`` and ``moment`` the bridge overrides ``keepdims=True`` (see
        # :meth:`Bridge._execute_operations_on_chunk`) so the per-bridge dict values are at least 1-D for ``mean_agg`` /
        # ``moment_agg`` to walk with ``_concatenate2``. Mirror that here.
        chunk_kwargs = branch.get("chunk_kwargs") or {}
        effective_kwargs = dict(chunk_kwargs)
        if branch.get("kind") in (_BRANCH_KIND_MEAN, _BRANCH_KIND_MOMENT):
            effective_kwargs["keepdims"] = True
        chain = [(chunk_func, effective_kwargs, 1)]
        # MINIMAL-SCALAR-PARTIAL (mirrors :func:`_try_chain_branch` via the shared :func:`_apply_full_local_fold`):
        # a full local fold ships one scalar (``tier -> finalizer``); the tier kwargs stay dask's own.
        chain = _apply_full_local_fold(branch.get("op_name"), bool(branch.get("full_local_fold")), chain)
        return _build_branch(
            branch=branch,
            chain=chain,
            array_ndim=array_ndim,
            placeholder=placeholder,
            deliver_direct=deliver_direct,
            window_read=window_read,
        )
    except Exception as e:
        # Length-1 is the last-resort fallback; log context for diagnosis (usually a chunk_func/placeholder shape-dtype
        # mismatch).
        logger.debug(
            "_try_length1_branch: build_branch raised for %s with chunk_kwargs=%r, array_ndim=%d, placeholder=%r: %s",
            branch.get("output_key"),
            branch.get("chunk_kwargs"),
            array_ndim,
            type(placeholder).__name__ if placeholder is not None else None,
            e,
        )
        return None


def _build_branch(
    branch: Dict[str, Any],
    chain: Optional[List[Tuple[Callable, dict, int]]] = None,
    array_ndim: int = 0,
    placeholder: Optional[Any] = None,
    chain_branch_func: Optional[Callable] = None,
    deliver_direct: Optional[bool] = None,
    window_read: bool = False,
) -> BranchSpec:
    """Build a :class:`BranchSpec` from a branch and a layer chain.

    The chain (root-to-chunk ``(func, kwargs, input_count)`` triples) is composed into a single ``branch_func``; the
    branch provides the reduction's ``kind``/``finalize``/``chunk_axis`` metadata. A length-1 branch is simply a chain
    of length 1 (``[(chunk_func, effective_kwargs, 1)]``).

    If ``chain_branch_func`` is provided (the memoized / pre-built version), use it directly instead of rebuilding.

    ``deliver_direct`` records whether the reduction's chunk stage reads DIRECTLY from the registered array root (a
    plain ``arr.<op>()`` call, possibly via ``window[-1]``) rather than from a pointwise chain or slice
    (``(arr*arr).sum()``, ``arr[2:5].sum()``). The registration gate in :mod:`deisa.dask.deisa` refuses non-direct
    reductions because the precompute delivery path cannot reconstruct a chain on the callback side. ``None`` defaults
    to ``len(chain) == 1`` (a single layer means the chunk stage is the only layer between root and aggregate).

    ``window_read`` marks a whole-row-plane ``root[-1]`` read: the callback's runtime reduction runs on the WHOLE
    delivered array, so the branch's ``reduction_axes`` is the FULL reduction (``()``) even though the stub-side
    chunk axis is partial.
    """
    kind = branch.get("kind", _BRANCH_KIND_SCALAR)
    finalize = branch.get("finalize")

    chunk_kwargs = branch.get("chunk_kwargs") or {}
    ax = chunk_kwargs.get("axis")
    if isinstance(ax, (list, tuple)):
        chunk_axis = tuple(int(a) for a in ax)
    elif ax is not None:
        chunk_axis = (int(ax),)
    else:
        chunk_axis = None

    if chain_branch_func is None:
        chain_branch_func = _build_chain_branch_func(chain or [])

    partial_shape, partial_dtype = _discover_partial_metadata(chain_branch_func, placeholder, branch)

    # Reduction axes the callback's delivered view will be asked for: what the callback's reduction call on the
    # delivered view must match. A window read runs on the whole array (full reduction, ``()``); any other direct
    # reduction normalizes its chunk axis against the registered array's ndim.
    if window_read:
        reduction_axes: Tuple[int, ...] = ()
    elif chunk_axis is None:
        reduction_axes = ()
    else:
        reduction_axes = _normalize_reduction_axis(chunk_axis, array_ndim)

    return BranchSpec(
        output_key=branch["output_key"],
        input_name=branch["array_name"],
        output_kind=kind,
        branch_func=chain_branch_func,
        chunk_axis=chunk_axis,
        finalize=finalize,
        partial_shape=partial_shape,
        partial_dtype=partial_dtype,
        op_name=branch.get("op_name", ""),
        deliver_direct=(len(chain or []) == 1) if deliver_direct is None else deliver_direct,
        window_read=window_read,
        reduction_axes=reduction_axes,
    )


def _walk_chain(graph, agg_name: str) -> Optional[List[Tuple[Callable, dict, int]]]:
    """Walk from a chunk layer back to the placeholder root."""
    if not _is_aggregate_layer(agg_name):
        return None
    # Match the chunk layer via the dedicated chunk/aggregate pairs (mean_chunk / mean_agg, chunk_max / max, chunk_min /
    # min, moment_agg / var) instead of requiring an exact base name.
    chunk_layer_name = _chunk_layer_for_aggregate(graph, agg_name)
    if chunk_layer_name is None:
        return None
    # Refuse to fold an aggregate whose output feeds ANOTHER reduction's chunk stage (cross-reduction expression like
    # ``(arr - arr.mean()).sum()``). Such an aggregate's global value is required downstream; folding it alone would
    # produce a wrong local partial.
    if _aggregate_output_feeds_other_reduction(graph, agg_name):
        return None
    chain: List[Tuple[Callable, dict, int]] = []
    current = chunk_layer_name
    seen = set()
    while current is not None and current not in seen:
        seen.add(current)
        layer = graph.layers[current]
        if "-aggregate-" in current:
            break
        chunk_info = _chunk_func_and_kwargs(layer)
        if chunk_info is None:
            break
        func, kwargs = chunk_info
        upstream = _find_single_upstream(layer)
        if upstream is None:
            return None
        upstream_name, upstream_input_count = upstream
        # If the upstream isn't a layer in the graph, we've hit the root (placeholder DataNode). The branch_func
        # receives the actual chunk from the bridge, so we don't fold the root.
        if upstream_name not in graph.layers:
            break
        chain.append((func, kwargs, upstream_input_count))
        current = upstream_name
    chain.reverse()
    return chain


def _build_chain_branch_func(chain: List[Tuple[Callable, dict, int]]) -> Callable[[Any], Any]:
    """Compose a list of ``(func, kwargs, input_count)`` into a single branch_func(chunk).
    Layers reading from a single upstream twice (e.g. ``arr * arr``) get the chunk passed twice.

    Returns a module-level callable (``_chain_branch_func``) bound to the chain tuple via :func:`functools.partial`.
    The closure is on a top-level function so pickle can find it across processes. Building a fresh ``def
    branch_func(chunk, _chain=...)`` inside this helper would produce an unpicklable local function (AttributeError:
    Can't get local object). The ``functools.partial`` + module-level target recipe is the only shape that pickles
    cleanly.
    """
    chain_tuple = tuple(chain)
    return functools.partial(_chain_branch_func, _chain=chain_tuple)
