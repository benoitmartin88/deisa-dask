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
        sample = chunk_func(placeholder, **effective_kwargs)
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
        branch_func=chunk_func,
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

    Stage 2A implementation: thin wrapper over
    :func:`deisa.dask.task_hints.extract_reduction_hints`. Each
    detected reduction becomes a length-1 BranchSpec. Stage 3 will
    extend this to fold multi-layer chains.

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
    # Stage 2A: defer to the existing reduction hint extractor. The hint
    # dicts already carry kind/finalize/chunk_kwargs metadata that maps
    # 1:1 onto BranchSpec fields. The bridge_func for each branch is
    # the same callable the hint's chunk_func_pickle encodes.
    import pickle as _pickle

    hints = extract_reduction_hints_from_callback(callback, registered_arrays, force=force)
    branches: List[BranchSpec] = []
    for hint in hints:
        try:
            chunk_func = _pickle.loads(hint["chunk_func_pickle"])
        except Exception as e:  # pragma: no cover - safety net
            if force:
                logger.warning("analyze_branch: unpickle failed for %s: %s", hint.get("output_key"), e)
                continue
            raise
        # ``input_name`` for a length-1 branch is the first registered
        # array (the analyzer's primary_name). Today every detected
        # reduction is rooted at one array; the helper exposes the
        # same assumption.
        primary = next(iter(registered_arrays)) if registered_arrays else "f"
        # ``array_ndim`` is the ndim of the first registered array's
        # placeholder. The analyzer uses this to compute the combined
        # output shape (see ``_derive_combined_output_shape``).
        primary_arr: Optional[Any] = None
        try:
            primary_arr = next(iter(registered_arrays.values()))
            array_ndim = int(getattr(primary_arr, "ndim", 0))
        except Exception:
            array_ndim = 0
        # ``placeholder`` is the dask array passed as the first
        # registered array. We use it to discover the partial's shape
        # by running chunk_func on it (structural inspection; the
        # chunk_func is a numpy op, no callback side effects).
        placeholder = primary_arr
        # The chunk_func expects a numpy chunk, not a dask array.
        # Materialize a numpy placeholder so the chunk_func produces the
        # per-bridge partial shape (with ``keepdims=True`` semantics
        # intact). Calling chunk_func on the raw dask placeholder
        # would produce a dask-shaped result whose ``shape`` is the
        # agg output, not the chunk-stage partial.
        if hasattr(placeholder, "compute"):
            try:
                placeholder = placeholder.compute()
            except Exception:
                # Best-effort fallback -- chunk_func may still produce
                # something usable on the dask array.
                pass
        try:
            branch = build_branch_from_hint(
                hint=hint,
                chunk_func=chunk_func,
                input_name=primary,
                array_ndim=array_ndim,
                placeholder=placeholder,
            )
        except Exception as e:
            if force:
                logger.warning("analyze_branch: build failed for %s: %s", hint.get("output_key"), e)
                continue
            raise
        branches.append(branch)
    return branches


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
