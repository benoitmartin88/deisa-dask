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
Compose the slim wire request a DEISA reduction ships to the bridges.

The request is the v1 replacement for the retired ``BranchSpec``: six fields instead of thirteen, and every field
survives for a reason --

``key``          dedup/namespace for the payload (``{array}-{op}`` plus an axis discriminator),
``input_name``   the registered array the branch is rooted in (the Deisa side groups by it, never by parsing the key),
``kind``         the combine algebra that consumes the partial (``scalar`` / ``mean`` / ``moment`` / ``boolean`` /
                 ``count``),
``branch_func``  ``functools.partial`` over MODULE-LEVEL functions, the surviving pickle contract,
``axes``         reduced data axes, ``()`` meaning the full reduction,
``finalize``     post-combine step (``"sqrt"`` for std).

What the old dataclass carried and this one does not: ``partial_shape``/``partial_dtype`` are PROBED off the composed
callable (:func:`probe_partial`) instead of being hand-maintained, ``chunk_axis`` collapsed into ``axes``,
``deliver_direct`` and ``window_read`` died with the AST walker, and ``summary_config`` belongs to the PCA request type,
not to a reduction.

The composed callable is ``functools.partial(_chain_branch_func, _chain=<layers>)`` where each layer is a
``(func, kwargs, input_count)`` triple -- the same shape the bridge already executes, so
``_execute_operations_on_chunk`` stays untouched. The layers are, in order:

1. the pointwise program (skipped when the reduction is taken straight off the collection),
2. the reduction's chunk stage, and
3. for a FULL reduction of a scalar-like kind, the minimal-scalar-partial finalizer, so a bridge ships one scalar
   instead of a tier-shaped array.

Every callable in the chain is a module-level function bound with ``functools.partial``: no lambdas, no closures, no
bound methods. That is not style -- the request is pickled to the bridge process, where a lambda has no name to
resolve.
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np

from dask.array.reductions import mean_chunk, moment_chunk
from deisa.dask import runtime_errors as errors
from deisa.dask.branch import _FULL_FOLD_FINALIZERS as _LEGACY_FULL_FOLD_FINALIZERS
from deisa.dask.branch import _chain_branch_func, _full_fold_layer
from deisa.dask.runtime_expr import _CHUNK_PROGRAM, ReductionExpr, eval_pointwise

__all__ = ["RuntimeRequest", "compose_request", "probe_partial"]


# -- Bridge-side callables (module level, because the request is pickled) ---------------------------------------------
def _pointwise_layer(chunk: Any, _program: Optional[Tuple[Any, ...]] = None) -> Any:
    """First chain stage: evaluate the expression program on the bridge's own chunk.

    - ``:param chunk:`` The bridge's numpy chunk (the composition never sends it anywhere).
    - ``:param _program:`` The picklable program tuple bound by :func:`compose_request`.
    """
    if _program is None:
        raise errors.precompute_runtime_invariant(
            "_pointwise_layer", "the layer was executed without its bound program"
        )
    return eval_pointwise(_program, chunk)


def _final_full_any(x: Any) -> Any:
    """Minimal-scalar-partial finalizer for a full ``any`` fold."""
    return np.any(x)


def _final_full_all(x: Any) -> Any:
    """Minimal-scalar-partial finalizer for a full ``all`` fold."""
    return np.all(x)


def _final_full_count(x: Any) -> Any:
    """Minimal-scalar-partial finalizer for a full ``count`` fold.

    ``count`` is additive (a chunk's count is its own non-zero element count), so the fold is a SUM of the chunk-stage
    counts -- not another ``count_nonzero``, which would ask how many COUNTING entries are non-zero.
    """
    return np.sum(x)


# The minimal-scalar-partial table, extended with the three F4 operators. The four entries branch.py already owns are
# reused verbatim rather than redefined: they ARE the surviving contract (``_final_full_{sum,prod,max,min}``).
FULL_FOLD_FINALIZERS = {
    **_LEGACY_FULL_FOLD_FINALIZERS,
    "any": _final_full_any,
    "all": _final_full_all,
    "count": _final_full_count,
}

# Chunk stages of the kinds whose combine can consume a bare tier-shaped array. mean/var/std are absent by design: their
# partials are the ``{n, total[, M]}`` dicts ``mean_agg``/``moment_agg`` require, which is a mathematical property of
# the combine, not an omission.
_TIER_FUNCS = {
    "sum": np.sum,
    "prod": np.prod,
    "min": np.min,
    "max": np.max,
    "any": np.any,
    "all": np.all,
    "count": np.count_nonzero,
}


def _accumulation_dtype(dtype: Any) -> np.dtype:
    """The accumulation dtype ``mean``/``moment`` chunk stages use for ``dtype`` inputs.

    Reads numpy's own promotion rule off a one-element array instead of guessing it, so ``mean(float32)`` stays float32
    and ``mean(int64)`` widens to float64 exactly as ``np.mean`` does.
    """
    return np.mean(np.zeros(1, dtype=dtype)).dtype


def _tier_func(op_name: str, axes: Tuple[int, ...], dtype: Any) -> Callable[[Any], Any]:
    """Build the reduction's chunk stage: a module-level function bound to its axis/dtype kwargs.

    ``axes == ()`` is the full reduction, which numpy spells ``axis=None``. ``keepdims=True`` is used everywhere because
    the combine squeezes the reduced axes itself (a 0-d intermediate would lose the axis mapping).

    - ``:param op_name:`` Canonical reduction name.
    - ``:param axes:`` Reduced data axes, ``()`` for the full reduction.
    - ``:param dtype:`` dtype of the reduced array, used for the ``mean``/``moment`` accumulation dtype only.
    """
    axis = None if not axes else tuple(axes)
    if op_name == "mean":
        return functools.partial(mean_chunk, axis=axis, keepdims=True, dtype=str(_accumulation_dtype(dtype)))
    if op_name in ("var", "std"):
        return functools.partial(moment_chunk, order=2, axis=axis, keepdims=True, dtype=str(_accumulation_dtype(dtype)))
    func = _TIER_FUNCS.get(op_name)
    if func is None:
        raise errors.precompute_runtime_invariant("compose_request", f"no chunk stage for the reduction {op_name!r}")
    return functools.partial(func, axis=axis, keepdims=True)


def _build_chain(reduction: ReductionExpr) -> Tuple[Tuple[Callable, dict, int], ...]:
    """Build the ``(func, kwargs, input_count)`` chain a bridge executes for ``reduction``.

    The pointwise stage is skipped for a reduction of the bare collection: there is nothing to evaluate, and a program
    of just ``("chunk",)`` would add a redundant pickled layer to every request.
    """
    layers: list = []
    if reduction.operand.program != _CHUNK_PROGRAM:
        program = reduction.operand.program
        layers.append((functools.partial(_pointwise_layer, _program=program), {}, 1))
    tier_func = _tier_func(reduction.op_name, reduction.axes, reduction.dtype)
    finalizer = FULL_FOLD_FINALIZERS.get(reduction.op_name) if not reduction.axes else None
    if finalizer is None:
        layers.append((tier_func, {}, 1))
    else:
        layers.append(
            (
                functools.partial(_full_fold_layer, tier_func=tier_func, tier_kwargs={}, finalizer=finalizer),
                {},
                1,
            )
        )
    return tuple(layers)


@dataclass(frozen=True)
class RuntimeRequest:
    """One dispatchable reduction request (the slim v1 wire format, decision F5).

    - ``:param key:`` Dedup/namespace key of the payload, e.g. ``"f-sum"`` or ``"f-sum-axis0"``.
    - ``:param input_name:`` Registered array the branch is rooted in.
    - ``:param kind:`` Payload kind the combine step dispatches on.
    - ``:param branch_func:`` ``functools.partial`` over module-level functions: ``branch_func(chunk)`` returns the
      chunk-local partial.
    - ``:param axes:`` Reduced data axes; ``()`` means the full reduction.
    - ``:param finalize:`` Post-combine step (``"sqrt"`` for std), else ``None``.
    """

    key: str
    input_name: str
    kind: str
    branch_func: Callable[[Any], Any]
    axes: Tuple[int, ...]
    finalize: Optional[str]


def compose_request(reduction: ReductionExpr) -> RuntimeRequest:
    """Compose the wire request for one reduction.

    Pure object-model work: no graph, no analysis, nothing is sent. P2 hands the result to the bridge communicator and
    reuses the unchanged combine algebra on the partials that come back.

    - ``:param reduction:`` The :class:`~deisa.dask.runtime_expr.ReductionExpr` a callback computed.
    - ``:return:`` The :class:`RuntimeRequest` for it.
    """
    if not isinstance(reduction, ReductionExpr):
        raise errors.precompute_runtime_invariant(
            "compose_request",
            f"expected a reduction expression, got {type(reduction).__name__}; only reductions are dispatchable",
        )
    return RuntimeRequest(
        key=reduction.key,
        input_name=reduction.root.array_name,
        kind=reduction.kind,
        branch_func=functools.partial(_chain_branch_func, _chain=_build_chain(reduction)),
        axes=reduction.axes,
        finalize=reduction.finalize,
    )


def probe_partial(
    request: RuntimeRequest, chunk_shape: Sequence[int], dtype: Any = np.float64
) -> Tuple[Tuple[int, ...], str]:
    """Derive ``(partial_shape, partial_dtype)`` by running the composed callable on a placeholder chunk.

    These two values are deliberately NOT request fields: a hand-maintained copy drifts from the callable, while a probe
    cannot lie. The placeholder is the bridge's own chunk shape, so the probed metadata describes the real per-bridge
    payload.

    - ``:param request:`` The composed request.
    - ``:param chunk_shape:`` Shape of one bridge's chunk.
    - ``:param dtype:`` dtype of the array, used for the placeholder.
    """
    placeholder = np.ones(tuple(int(size) for size in chunk_shape), dtype=dtype)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        # A placeholder chunk is all ones: refusals are about SHAPES, so arithmetic warnings (log, divide) must not
        # pollute the caller's log stream.
        warnings.simplefilter("ignore")
        sample = request.branch_func(placeholder)
    if isinstance(sample, dict):
        representative = sample.get("total", sample.get("M", next(iter(sample.values()))))
    else:
        representative = sample
    array = np.asarray(representative)
    return tuple(int(size) for size in array.shape), str(array.dtype)
