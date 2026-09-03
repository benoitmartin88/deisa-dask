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
#   prior written  permission.
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
Extract reduction hints from a dask array's task graph.

Given a dask array whose graph contains reductions (sum, mean, std, var, max,
min, prod), walk the graph layers to find the aggregate layer and the matching
chunk layer, then build a hint dict the bridge can execute on a local numpy
chunk before scattering.

Hint schema (the contract between the analytics side (precompute analyzer) and
the bridge side (local chunk execution)):

.. code-block:: python

    {
        "output_key": "fdistribu-sum",  # unique key for this reduction
        "op_name": "sum",  # canonical op name
        "chunk_func_pickle": ...,  # pickle of the chunk callable
        "chunk_kwargs": {...},  # kwargs for the chunk callable
        "agg_pickle": ...,  # pickle of the aggregator callable
        "agg_dep_structures": [...],  # structure of chunk deps for the aggregator
        "agg_keys": [...],  # chunk key paths the aggregator expects
        "finalize": "sqrt" | None,  # post-step (sqrt for std)
    }

This module is purely about reading the dask graph; it never executes user
callbacks.
"""

from __future__ import annotations

import logging
import pickle
from typing import Any, Dict, List, Optional

import dask.array as da

logger = logging.getLogger(__name__)

# Map an aggregator function name to a canonical op name.
# sum/prod/max/min/amax/amin are simple; mean_agg and moment_agg are the
# two-stage reducers used by mean/var/std.
_OP_FROM_FUNC_NAME = {
    "sum": "sum",
    "prod": "prod",
    "max": "max",
    "min": "min",
    "amax": "max",
    "amin": "min",
    "mean_agg": "mean",
    "moment_agg": "moment",
}

# Operations supported by ``_combine_reduction_partials`` on the bridge side.
SUPPORTED_OPS = {"sum", "mean", "std", "var", "max", "min", "prod"}

# Reduction kind classification -- controls how the bridge scatters the
# partial and how the Deisa-side combine graph is built.
#
# - ``"scalar"``: the chunk_func returns a plain scalar (or numpy array for
#   axis reductions). Per-bridge partials are summed via dask's natural
#   ``da.stack`` + ``.sum(axis=0)`` -- no custom agg logic needed.
#
# - ``"mean"``: the chunk_func returns ``{"n": ..., "total": ...}`` (per
#   dask's ``mean_chunk``). Per-bridge dicts are scattered as single
#   pickled blobs; the Deisa-side dask graph builds a custom delayed task
#   that resolves the dicts and calls ``dask.array.reductions.mean_agg``
#   over them.
#
# - ``"moment"``: the chunk_func returns ``{"n": ..., "total": ..., "M": ...}``
#   (per dask's ``moment_chunk``). Same scatter-then-delayed-agg pattern
#   as ``"mean"`` but calls ``moment_agg``; ``std`` differs from ``var``
#   only in the trailing ``sqrt`` (``finalize`` is set accordingly).
_REDUCTION_KIND = {
    "sum": "scalar",
    "prod": "scalar",
    "max": "scalar",
    "min": "scalar",
    "mean": "mean",
    "var": "moment",
    "std": "moment",
}


# ---------------------------------------------------------------------------
# Layer name helpers
# ---------------------------------------------------------------------------
def _strip_hash(layer_name: str) -> str:
    """Drop the trailing ``-hash`` from a dask layer name."""
    return layer_name.rsplit("-", 1)[0] if "-" in layer_name else layer_name


def _is_aggregate_layer(layer_name: str) -> bool:
    """An aggregate layer has ``-aggregate-`` in its name."""
    return "-aggregate-" in layer_name


def _is_sqrt_layer(layer_name: str) -> bool:
    """``std`` adds a ``_sqrt-<hash>`` post-step on top of ``var``."""
    return layer_name.startswith("_sqrt-")


def _base_for_aggregate(layer_name: str) -> str:
    """Return the chunk-layer base name for a given aggregate layer.

    For ``<base>-aggregate-<hash>`` returns ``<base>``. The chunk layer may
    use a related base (e.g. ``mean_agg`` aggregate -> ``mean_chunk`` chunk
    layer; ``max`` aggregate -> ``chunk_max`` chunk layer; ``min`` aggregate
    -> ``chunk_min`` chunk layer).
    """
    base = layer_name.split("-aggregate-", 1)[0]
    return base


# Map aggregate base -> possible chunk base names.
_CHUNK_BASE_FOR_AGG = {
    "mean_agg": "mean_chunk",
    "max": "chunk_max",
    "min": "chunk_min",
}


def _chunk_base_for_aggregate_base(agg_base: str) -> List[str]:
    """Return candidate chunk base names for an aggregate base.

    The chunk layer and aggregate layer usually share a base (``sum``,
    ``prod``, ``var``). The exceptions are reductions that use a
    dedicated chunk/aggregate pair: ``mean`` (mean_chunk / mean_agg),
    ``max`` (chunk_max / max), ``min`` (chunk_min / min).
    """
    direct = agg_base
    special = _CHUNK_BASE_FOR_AGG.get(agg_base)
    return [direct, special] if special else [direct]


# ---------------------------------------------------------------------------
# Aggregate/Chunk layer introspection
# ---------------------------------------------------------------------------
def _is_task(value: Any) -> bool:
    """True if ``value`` is a dask ``Task`` (new task spec)."""
    if isinstance(value, tuple):
        return False
    return hasattr(value, "func") and hasattr(value, "args") and hasattr(value, "kwargs")


def _layer_first_task(layer) -> Optional[Any]:
    """Return the first dask ``Task`` in a layer, or None."""
    for value in layer.values():
        if _is_task(value):
            return value
    return None


def _layer_first_tuple(layer) -> Optional[Any]:
    """Return the first tuple-form task in a layer, or None."""
    for value in layer.values():
        if isinstance(value, tuple) and len(value) >= 2:
            return value
    return None


def _unwrap_partial(func: Any) -> Optional[Any]:
    """Return ``func.func`` if ``func`` is a ``functools.partial``."""
    if hasattr(func, "func") and callable(getattr(func, "func", None)):
        return func.func
    return None


def _is_compose(func: Any) -> bool:
    return hasattr(func, "funcs") and isinstance(func.funcs, tuple)


def _op_from_func(func: Any) -> Optional[str]:
    """Identify the canonical op name from an aggregator callable."""
    if _is_compose(func):
        # dask reductions like sum/prod/max/min are wrapped in
        # ``Compose(partial(np_op), partial(_concatenate2))``.
        for f in func.funcs:
            inner = _unwrap_partial(f)
            if inner is not None:
                name = getattr(inner, "__name__", None)
                if name in _OP_FROM_FUNC_NAME:
                    return _OP_FROM_FUNC_NAME[name]
        return None
    inner = _unwrap_partial(func)
    if inner is None:
        return None
    name = getattr(inner, "__name__", None)
    if name is None:
        return None
    return _OP_FROM_FUNC_NAME.get(name)


def _collect_chunk_keys_from_aggregate(layer) -> List[str]:
    """Return the chunk-layer base names referenced by the aggregate layer.

    The bridge only needs to know which chunk keys to expect, not the
    nested positional structure. This walks the aggregate's deps and
    returns the unique chunk-layer base names (strings).
    """
    keys: set = set()
    # legacy form: (func, deps, ...) where deps is a nested list of key tuples
    for value in layer.values():
        if isinstance(value, tuple) and len(value) >= 2:
            deps = value[1]
            stack: List[Any] = [deps]
            while stack:
                item = stack.pop()
                if isinstance(item, (list, tuple)):
                    stack.extend(item)
                elif isinstance(item, str):
                    keys.add(item)
            return sorted(keys)
    # new form: a Task/GraphNode in the aggregate layer references the
    # chunk layer keys directly in its args.
    for value in layer.values():
        if _is_task(value):
            for arg in value.args:
                if isinstance(arg, str):
                    keys.add(arg)
                else:
                    # TaskRef or similar - extract the key
                    name = getattr(arg, "key", None) or getattr(arg, "__str__", lambda: str(arg))()
                    if isinstance(name, str):
                        # take the layer name part (before the first "(")
                        base = name.split("(", 1)[0]
                        if base:
                            keys.add(base)
            return sorted(keys)
    return []


def _agg_dep_structures(layer) -> Optional[Any]:
    """Return the deps structure of the first tuple-form task in ``layer``."""
    for value in layer.values():
        if isinstance(value, tuple) and len(value) >= 2:
            return value[1]
    return None


def _chunk_func_and_kwargs(chunk_layer) -> Optional[tuple]:
    """Return ``(func, kwargs)`` for a chunk layer.

    Supports:
    - the new dask task spec (Task objects): ``func`` and ``kwargs`` come
      from the Task directly.
    - the legacy tuple form: ``(func, args, kwargs)`` where ``func`` may be
      a partial with extra keywords baked in.
    """
    for value in chunk_layer.values():
        if _is_task(value):
            return value.func, dict(value.kwargs or {})
        if isinstance(value, tuple) and len(value) >= 1:
            func = value[0]
            if hasattr(func, "keywords"):
                return func, dict(func.keywords or {})
            return func, {}
    return None


# ---------------------------------------------------------------------------
# Hint extraction
# ---------------------------------------------------------------------------
def _find_chunk_layer(graph, agg_base: str) -> Optional[str]:
    """Locate the chunk layer that feeds the aggregate layer with the given base.

    Returns ``None`` if no matching chunk layer exists.
    """
    candidates = _chunk_base_for_aggregate_base(agg_base)
    for layer_name in graph.layers:
        if _is_aggregate_layer(layer_name) or _is_sqrt_layer(layer_name):
            continue
        layer_base = _strip_hash(layer_name)
        if layer_base in candidates:
            return layer_name
        # Also accept names that start with the base (e.g. ``sum-``)
        if any(layer_name.startswith(c + "-") for c in candidates) and not _is_aggregate_layer(layer_name):
            return layer_name
    return None


def _has_sqrt_poststep(graph) -> bool:
    for layer_name in graph.layers:
        if _is_sqrt_layer(layer_name):
            return True
    return False


def _serialize_func(func: Any) -> bytes:
    return pickle.dumps(func)


def extract_reduction_hints(darr: da.Array, array_name: str = "f") -> List[Dict[str, Any]]:
    """Inspect ``darr``'s task graph and return a hint dict per reduction.

    - ``:param darr:`` A dask array whose graph contains at least one reduction
      (typically built symbolically by ``deisa.dask.precompute_analyzer``).
    - ``:param array_name:`` Base name for the reduction output keys.
    - ``:return:`` List of hint dicts matching the schema above.

    Note: this walks the graph but never executes any task; the dask arrays
    used at analysis time are zero-filled placeholders, and we don't run them.
    """
    hints: List[Dict[str, Any]] = []
    try:
        graph = darr.__dask_graph__()
    except Exception as e:  # pragma: no cover - safety net
        logger.debug("extract_reduction_hints: failed to get graph: %s", e)
        return hints

    has_sqrt = _has_sqrt_poststep(graph)

    for layer_name, layer in graph.layers.items():
        if not _is_aggregate_layer(layer_name):
            continue

        # Find the aggregator function: legacy tuple form preferred, then Task
        agg_func: Optional[Any] = None
        tup = _layer_first_tuple(layer)
        if tup is not None:
            agg_func = tup[0]
        else:
            task = _layer_first_task(layer)
            if task is not None:
                agg_func = task.func

        if agg_func is None:
            continue

        op_name = _op_from_func(agg_func)
        if op_name is None:
            continue

        # ``moment_agg`` is shared by var and std; disambiguate using the
        # _sqrt poststep (std = var followed by sqrt).
        finalize: Optional[str] = None
        if op_name == "moment":
            if has_sqrt:
                op_name = "std"
                finalize = "sqrt"
            else:
                op_name = "var"
                finalize = None

        if op_name not in SUPPORTED_OPS:
            logger.debug("extract_reduction_hints: unsupported op %s, skipping", op_name)
            continue

        # Find the matching chunk layer
        agg_base = _base_for_aggregate(layer_name)
        chunk_layer_name = _find_chunk_layer(graph, agg_base)
        if chunk_layer_name is None:
            logger.debug("extract_reduction_hints: no chunk layer for %s (base=%s)", layer_name, agg_base)
            continue

        chunk_layer = graph.layers[chunk_layer_name]
        chunk_info = _chunk_func_and_kwargs(chunk_layer)
        if chunk_info is None:
            continue
        chunk_func, chunk_kwargs = chunk_info

        chunk_keys = _collect_chunk_keys_from_aggregate(layer)
        agg_deps = _agg_dep_structures(layer)

        try:
            chunk_func_pickle = _serialize_func(chunk_func)
        except Exception as e:
            logger.debug("extract_reduction_hints: failed to pickle chunk func: %s", e)
            continue
        try:
            agg_pickle = _serialize_func(agg_func)
        except Exception as e:
            logger.debug("extract_reduction_hints: failed to pickle agg func: %s", e)
            continue

        output_key = f"{array_name}-{op_name}"
        # Backwards-compatible alias for callers that index hints by
        # ``keywords`` rather than ``chunk_kwargs``. Also unwrap single-
        # element axis tuples (dask normalizes ``axis=0`` to ``axis=(0,)``).
        chunk_kwargs = dict(chunk_kwargs) if chunk_kwargs else {}
        if isinstance(chunk_kwargs.get("axis"), tuple) and len(chunk_kwargs["axis"]) == 1:
            chunk_kwargs["axis"] = chunk_kwargs["axis"][0]
        hints.append(
            {
                "output_key": output_key,
                "op_name": op_name,
                "kind": _REDUCTION_KIND.get(op_name, "scalar"),
                "chunk_func_pickle": chunk_func_pickle,
                "chunk_kwargs": chunk_kwargs,
                "keywords": chunk_kwargs,
                "agg_pickle": agg_pickle,
                "agg_dep_structures": agg_deps,
                "agg_keys": chunk_keys,
                "finalize": finalize,
            }
        )

    return hints
