"""
Unit tests for the Stage 2B chain-folding walker.

These tests exercise the chain walker helpers directly on synthetic
dask graphs. They do NOT go through ``analyze_branch`` end-to-end
(see the design doc -- wiring is Stage 2B follow-up). They are
sanity checks that the walker correctly identifies single-input
chains, refuses constants and cross-array chains, and produces
branch_func callables that match naive numpy computations.
"""

import functools
import textwrap
from typing import Any, Callable, Dict

import dask.array as da
import numpy as np
import pytest

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
    @pytest.mark.parametrize(
        "expr,expected_length,should_refuse",
        [
            pytest.param(
                lambda arr: (arr * arr).sum(),
                2,
                False,
                id="self-ref-mul-sum",
            ),
            pytest.param(
                lambda arr: np.log(np.exp(arr)).sum(),
                3,
                False,
                id="ufunc-exp-log-sum",
            ),
            pytest.param(
                lambda arr: (arr**arr).sum(),
                2,
                False,
                id="self-ref-pow-sum",
            ),
            pytest.param(
                lambda arr: np.sin(arr).sum(axis=0),
                2,
                False,
                id="sin-axis0-sum",
            ),
            pytest.param(
                lambda arr: (arr - arr.mean()).sum(),
                None,
                True,
                id="cross-array-refused",
            ),
            pytest.param(
                lambda arr: (arr + 1).sum(),
                None,
                True,
                id="scalar-constant-refused",
            ),
        ],
    )
    def test_walk_chain_parametrized(self, expr, expected_length, should_refuse):
        arr = da.zeros((4, 4), chunks=2)
        g = expr(arr).__dask_graph__()
        chain = _walk_chain(g, _find_agg_layer(g))
        if should_refuse:
            assert chain is None
        else:
            assert chain is not None
            assert len(chain) == expected_length


# ---------------------------------------------------------------------------
# _build_chain_branch_func -- numerical correctness
# ---------------------------------------------------------------------------
class TestChainBranchFunc:
    """The composed branch_func must match the dask-computed value for
    every foldable chain. This is the core correctness property.
    """

    @pytest.mark.parametrize(
        "expr,seed,data_generator,expected_expr,rtol",
        [
            pytest.param(
                lambda arr: (arr * arr).sum(),
                0,
                lambda: np.random.random((4, 4)),
                lambda real: (real * real).sum(),
                None,
                id="squared-sum",
            ),
            pytest.param(
                lambda arr: np.log(np.exp(arr)).sum(),
                1,
                lambda: np.random.random((4, 4)),
                lambda real: real.sum(),
                1e-6,
                id="ufunc-chain",
            ),
            pytest.param(
                lambda arr: (arr**arr).sum(),
                2,
                lambda: np.random.random((4, 4)) * 0.5,
                lambda real: (real**real).sum(),
                None,
                id="pow-self",
            ),
        ],
    )
    def test_chain_branch_func_parametrized(self, expr, seed, data_generator, expected_expr, rtol):
        arr = da.zeros((4, 4), chunks=2)
        g = expr(arr).__dask_graph__()
        chain = _walk_chain(g, _find_agg_layer(g))
        assert chain is not None
        branch_func = _build_chain_branch_func(chain)

        np.random.seed(seed)
        real = data_generator()
        result = float(branch_func(real).sum())
        expected = float(expected_expr(real))

        if rtol is not None:
            assert np.isclose(result, expected, rtol=rtol)
        else:
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
    @pytest.mark.parametrize(
        "expr,layer_selector,expected_name_start,expected_count",
        [
            pytest.param(
                lambda arr: (arr * arr).sum(),
                lambda g: next(ln for ln in g.layers if ln.startswith("sum-") and "aggregate" not in ln),
                "mul-",
                1,
                id="sum-layer-single-input",
            ),
            pytest.param(
                lambda arr: (arr * arr).sum(),
                lambda g: next(ln for ln in g.layers if ln.startswith("mul-")),
                "zeros_like-",
                2,
                id="mul-layer-self-ref",
            ),
            pytest.param(
                lambda arr: (arr + 1).sum(),
                lambda g: next(ln for ln in g.layers if ln.startswith("add-")),
                None,
                None,
                id="add-layer-constant-refused",
            ),
        ],
    )
    def test_find_single_upstream_parametrized(self, expr, layer_selector, expected_name_start, expected_count):
        arr = da.zeros((4, 4), chunks=2)
        g = expr(arr).__dask_graph__()
        layer = layer_selector(g)
        result = _find_single_upstream(g.layers[layer])
        if expected_count is None:
            # Refused case (scalar constant): walker returns None.
            assert result is None
        else:
            assert result is not None
            name, count = result
            assert name.startswith(expected_name_start)
            assert count == expected_count


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

        # A multi-layer chain callback: (arr * arr).sum(). The chain
        # walker must fold {mul, sum} into a single branch_func; the
        # composed branch's _chain exposes the folded layers.
        cb = _make_callback("test_analyze_branch_cb", "return (arr * arr).sum().compute()")

        arrs = {"f": da.zeros((4, 4), chunks=2)}
        branches = analyze_branch(cb, arrs)
        assert len(branches) == 1
        assert branches[0].output_key == "f-sum"
        assert branches[0].output_kind == "scalar"

        # The branch_func is a functools.partial over _chain_branch_func;
        # the folded layers live under its keywords["_chain"].
        branch_func = branches[0].branch_func
        chain = branch_func.keywords["_chain"]
        assert len(chain) == 2  # mul + sum genuinely folded into one branch

        def _layer_name(layer):
            func = layer[0]
            # The reduction's chunk-stage layer is itself a partial
            # wrapping numpy.sum (carrying dtype); unwrap it for the name.
            if isinstance(func, functools.partial):
                return getattr(func.func, "__name__", repr(func))
            return getattr(func, "__name__", repr(func))

        names = [_layer_name(layer) for layer in chain]
        assert any(n in {"mul", "multiply"} for n in names)
        assert "sum" in names
