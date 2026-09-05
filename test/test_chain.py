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
Unit tests for the Stage 2B chain-folding walker.

These tests exercise the chain walker helpers directly on synthetic
dask graphs. They do NOT go through ``analyze_branch`` end-to-end
(see the design doc -- wiring is Stage 2B follow-up). They are
sanity checks that the walker correctly identifies single-input
chains, refuses constants and cross-array chains, and produces
branch_func callables that match naive numpy computations.
"""

import textwrap
from typing import Any, Callable, Dict

import dask.array as da
import numpy as np

from deisa.dask.branch import (
    _build_chain_branch_func,
    _find_chunk_layer,
    _find_single_upstream,
    _walk_chain,
)


def _make_callback(name: str, body: str) -> Callable:
    """Compile a small snippet ``def <name>(arr): <body>`` and return it.

    Mirrors the helper used in test_precompute.py so ``analyze_callback``
    can walk the source if needed.
    """
    src = textwrap.dedent(f"def {name}(arr):\n{textwrap.indent(body, '    ')}")
    scope: Dict[str, Any] = {}
    code = compile(src, f"<test_chain:{name}>", "exec")
    exec(code, scope)
    fn = scope[name]
    fn.__source__ = src  # type: ignore[attr-defined]
    return fn


def _find_agg_layer(graph) -> str:
    """Return the first ``*-aggregate-*`` layer name in a graph."""
    for ln in graph.layers:
        if "-aggregate-" in ln:
            return ln
    raise AssertionError("no aggregate layer in graph")


# ---------------------------------------------------------------------------
# _walk_chain
# ---------------------------------------------------------------------------
class TestWalkChain:
    def test_single_input_chain_with_self_ref(self):
        """``(arr * arr).sum()`` chain has 2 layers: mul, sum."""
        arr = da.zeros((4, 4), chunks=2)
        expr = (arr * arr).sum()
        g = expr.__dask_graph__()
        chain = _walk_chain(g, _find_agg_layer(g))
        assert chain is not None
        assert len(chain) == 2
        # First layer: mul with 2-input (self-ref)
        func, kwargs, input_count = chain[0]
        # dask stores the numpy wrapper as `mul`; the ufunc is `multiply`.
        # Either name is fine; what matters is it's a binary pointwise op
        # that consumes the upstream array twice.
        func_name = getattr(func, "__name__", "")
        assert func_name in {"mul", "multiply"}, f"unexpected func name {func_name!r}"
        assert input_count == 2  # arr * arr references arr twice
        # Second layer: sum chunk-stage
        func, kwargs, input_count = chain[1]
        assert input_count == 1

    def test_ufunc_chain(self):
        """``np.log(np.exp(arr)).sum()`` chain has 3 layers."""
        arr = da.zeros((4, 4), chunks=2)
        expr = np.log(np.exp(arr)).sum()
        g = expr.__dask_graph__()
        chain = _walk_chain(g, _find_agg_layer(g))
        assert chain is not None
        assert len(chain) == 3

    def test_self_ref_pow(self):
        """``(arr ** arr).sum()`` chain has 2 layers: pow, sum."""
        arr = da.zeros((4, 4), chunks=2)
        expr = (arr**arr).sum()
        g = expr.__dask_graph__()
        chain = _walk_chain(g, _find_agg_layer(g))
        assert chain is not None
        assert len(chain) == 2

    def test_chain_with_axis_reduction(self):
        """``np.sin(arr).sum(axis=0)`` chain has 2 layers and a chunk-stage axis reduction."""
        arr = da.zeros((4, 4), chunks=2)
        expr = np.sin(arr).sum(axis=0)
        g = expr.__dask_graph__()
        chain = _walk_chain(g, _find_agg_layer(g))
        assert chain is not None
        assert len(chain) == 2

    def test_cross_array_chain_refused(self):
        """``(arr - arr.mean()).sum()`` reads from two distinct arrays
        (the placeholder and the mean's output). The walker refuses.
        """
        arr = da.zeros((4, 4), chunks=2)
        expr = (arr - arr.mean()).sum()
        g = expr.__dask_graph__()
        chain = _walk_chain(g, _find_agg_layer(g))
        assert chain is None

    def test_chain_with_scalar_constant_refused(self):
        """``(arr + 1).sum()`` has a scalar constant in the add layer.
        The walker refuses because constants aren't currently supported
        (deferred to a follow-up).
        """
        arr = da.zeros((4, 4), chunks=2)
        expr = (arr + 1).sum()
        g = expr.__dask_graph__()
        chain = _walk_chain(g, _find_agg_layer(g))
        assert chain is None


# ---------------------------------------------------------------------------
# _build_chain_branch_func -- numerical correctness
# ---------------------------------------------------------------------------
class TestChainBranchFunc:
    """The composed branch_func must match the dask-computed value for
    every foldable chain. This is the core correctness property.
    """

    def test_squared_sum(self):
        arr = da.zeros((4, 4), chunks=2)
        expr = (arr * arr).sum()
        g = expr.__dask_graph__()
        chain = _walk_chain(g, _find_agg_layer(g))
        assert chain is not None
        branch_func = _build_chain_branch_func(chain)

        np.random.seed(0)
        real = np.random.random((4, 4))
        result = float(branch_func(real).sum())
        expected = float((real * real).sum())
        assert np.isclose(result, expected)

    def test_ufunc_chain(self):
        arr = da.zeros((4, 4), chunks=2)
        expr = np.log(np.exp(arr)).sum()
        g = expr.__dask_graph__()
        chain = _walk_chain(g, _find_agg_layer(g))
        assert chain is not None
        branch_func = _build_chain_branch_func(chain)

        np.random.seed(1)
        real = np.random.random((4, 4))
        result = float(branch_func(real).sum())
        # log(exp(x)) = x, so result == sum(real)
        expected = float(real.sum())
        assert np.isclose(result, expected, rtol=1e-6)

    def test_pow_self(self):
        arr = da.zeros((4, 4), chunks=2)
        expr = (arr**arr).sum()
        g = expr.__dask_graph__()
        chain = _walk_chain(g, _find_agg_layer(g))
        assert chain is not None
        branch_func = _build_chain_branch_func(chain)

        np.random.seed(2)
        # Use small positive values to avoid overflow with arr ** arr
        real = np.random.random((4, 4)) * 0.5
        result = float(branch_func(real).sum())
        expected = float((real**real).sum())
        assert np.isclose(result, expected)

    def test_picklable(self):
        """The composed branch_func must be picklable so it can cross
        the bridge process boundary.
        """
        import pickle

        arr = da.zeros((4, 4), chunks=2)
        expr = (arr * arr).sum()
        g = expr.__dask_graph__()
        chain = _walk_chain(g, _find_agg_layer(g))
        assert chain is not None
        branch_func = _build_chain_branch_func(chain)
        # Round-trip pickle
        restored = pickle.loads(pickle.dumps(branch_func))
        real = np.arange(16, dtype=np.float64).reshape(4, 4)
        assert np.allclose(restored(real), branch_func(real))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class TestFindChunkLayer:
    def test_finds_chunk_stage_layer(self):
        arr = da.zeros((4, 4), chunks=2)
        g = (arr * arr).sum().__dask_graph__()
        agg_name = _find_agg_layer(g)
        agg_base = agg_name.split("-aggregate-", 1)[0]  # "sum"
        chunk_layer = _find_chunk_layer(g, agg_base)
        assert chunk_layer is not None
        assert chunk_layer.startswith("sum-") and "aggregate" not in chunk_layer


class TestFindSingleUpstream:
    def test_single_input_returns_name_and_count(self):
        arr = da.zeros((4, 4), chunks=2)
        g = (arr * arr).sum().__dask_graph__()
        # Find the chunk-stage (sum) layer
        sum_layer = next(ln for ln in g.layers if ln.startswith("sum-") and "aggregate" not in ln)
        result = _find_single_upstream(g.layers[sum_layer])
        assert result is not None
        name, count = result
        assert name.startswith("mul-")
        assert count == 1  # sum reads from mul, single-input

    def test_self_ref_returns_count_2(self):
        arr = da.zeros((4, 4), chunks=2)
        g = (arr * arr).sum().__dask_graph__()
        # Find the mul layer (the chain's pointwise step)
        mul_layer = next(ln for ln in g.layers if ln.startswith("mul-"))
        result = _find_single_upstream(g.layers[mul_layer])
        assert result is not None
        name, count = result
        # mul reads zeros_like twice (arr * arr) -> count == 2
        assert name.startswith("zeros_like-")
        assert count == 2

    def test_constant_input_refused(self):
        arr = da.zeros((4, 4), chunks=2)
        g = (arr + 1).sum().__dask_graph__()
        add_layer = next(ln for ln in g.layers if ln.startswith("add-"))
        # add has 2 inputs: zeros_like (string) and (1, None) (constant tuple)
        # The walker should refuse.
        result = _find_single_upstream(g.layers[add_layer])
        assert result is None


# ---------------------------------------------------------------------------
# analyze_branch -- not wired in (Stage 2B follow-up), but the
# length-1 path still works and emits BranchSpec objects.
# ---------------------------------------------------------------------------
class TestAnalyzeBranchLength1:
    def test_analyze_branch_emits_branches(self):
        """analyze_branch produces length-1 branches for the
        per-reduction path. The chain walker folds multi-layer chains
        into a single branch_func.
        """
        from deisa.dask.branch import analyze_branch

        def cb(arr):
            return arr.sum().compute()

        arrs = {"f": da.zeros((4, 4), chunks=2)}
        branches = analyze_branch(cb, arrs)
        assert len(branches) == 1
        assert branches[0].output_key == "f-sum"
        assert branches[0].output_kind == "scalar"
