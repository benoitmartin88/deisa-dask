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
Tests for the compute-boundary precompute analyzer.

The analyzer finds compute boundaries (.compute(), client.compute(...),
client.submit(...), etc.) in the user's callback source, symbolically
evaluates the dask arrays being computed (no execution), and extracts
reduction hints from each dask array's task graph.

Crucial contract: the user's callback is NEVER called during analysis.
"""

import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import dask.array as da
import numpy as np
import pytest

from deisa.dask.precompute_analyzer import (
    IncompatibleCallbackError,
    MaterializationError,
    NoComputeBoundaryError,
    NoPrecomputableReductionError,
    analyze_callback,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
def _simple_stub() -> da.Array:
    return da.zeros((10, 10), chunks=(5, 5), dtype=np.float64)


def _hint_keys(hints: List[Dict[str, Any]]) -> List[str]:
    return sorted(h["output_key"] for h in hints)


def _op_names(hints: List[Dict[str, Any]]) -> List[str]:
    return sorted(h["op_name"] for h in hints)


# ---------------------------------------------------------------------------
# Per-reduction support: each reduction found in a graph that crosses a
# compute boundary should be detected.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "op_name",
    ["sum", "mean", "max", "min", "std", "var", "prod"],
)
def test_compute_direct_reduction(op_name: str) -> None:
    """``result = arr.op(); result.compute()`` should produce a hint for op_name."""
    arr = _simple_stub()
    src = f"""
        def callback(arr):
            result = arr.{op_name}()
            result.compute()
        """
    cb = _make_function("callback", src)
    hints = analyze_callback(cb, {"f": arr})
    assert len(hints) == 1
    assert hints[0]["output_key"] == f"f-{op_name}"
    assert hints[0]["op_name"] == op_name


def test_compute_with_axis_kwarg_int() -> None:
    arr = _simple_stub()
    src = """
        def callback(arr):
            result = arr.sum(axis=0)
            result.compute()
        """
    cb = _make_function("callback", src)
    hints = analyze_callback(cb, {"f": arr})
    assert len(hints) == 1
    assert hints[0]["keywords"].get("axis") == 0


def test_compute_with_axis_kwarg_tuple() -> None:
    arr = _simple_stub()
    src = """
        def callback(arr):
            result = arr.sum(axis=(0, 1))
            result.compute()
        """
    cb = _make_function("callback", src)
    hints = analyze_callback(cb, {"f": arr})
    assert len(hints) == 1
    assert hints[0]["keywords"].get("axis") == (0, 1)


def test_compute_multiple_reductions() -> None:
    arr = _simple_stub()
    src = """
        def callback(arr):
            s = arr.sum()
            m = arr.mean()
            mx = arr.max()
            s.compute()
            m.compute()
            mx.compute()
        """
    cb = _make_function("callback", src)
    hints = analyze_callback(cb, {"f": arr})
    assert _hint_keys(hints) == ["f-max", "f-mean", "f-sum"]


def test_compute_sliced_reduction() -> None:
    arr = _simple_stub()
    src = """
        def callback(arr):
            result = arr[2:5].sum()
            result.compute()
        """
    cb = _make_function("callback", src)
    hints = analyze_callback(cb, {"f": arr})
    assert _hint_keys(hints) == ["f-sum"]


def test_compute_column_slice_reduction() -> None:
    arr = _simple_stub()
    src = """
        def callback(arr):
            result = arr[:, 0].sum()
            result.compute()
        """
    cb = _make_function("callback", src)
    hints = analyze_callback(cb, {"f": arr})
    assert _hint_keys(hints) == ["f-sum"]


def test_compute_expression_sub() -> None:
    a = _simple_stub()
    b = _simple_stub()
    src = """
        def callback(arr_a, arr_b):
            result = (arr_a - arr_b).max()
            result.compute()
        """
    cb = _make_function("callback", src)
    hints = analyze_callback(cb, {"a": a, "b": b})
    # The first registered array is the primary name
    assert _hint_keys(hints) == ["a-max"]


def test_compute_expression_pow() -> None:
    arr = _simple_stub()
    src = """
        def callback(arr):
            result = (arr ** 2).sum()
            result.compute()
        """
    cb = _make_function("callback", src)
    hints = analyze_callback(cb, {"f": arr})
    assert _hint_keys(hints) == ["f-sum"]


def test_compute_expression_mul() -> None:
    a = _simple_stub()
    b = _simple_stub()
    src = """
        def callback(arr_a, arr_b):
            result = (arr_a * arr_b).sum()
            result.compute()
        """
    cb = _make_function("callback", src)
    hints = analyze_callback(cb, {"a": a, "b": b})
    assert _hint_keys(hints) == ["a-sum"]


def test_compute_dask_module_sum() -> None:
    """``da.sum(arr)`` module-style reduction should be detected."""
    arr = _simple_stub()
    src = """
        def callback(arr):
            result = da.sum(arr)
            result.compute()
        """
    cb = _make_function("callback", src)
    hints = analyze_callback(cb, {"f": arr})
    assert _hint_keys(hints) == ["f-sum"]


def test_compute_helper_same_file() -> None:
    arr = _simple_stub()
    src = """
        def callback(arr, grid_dvx):
            result = density(arr, grid_dvx)
            result.compute()

        def density(f, dvx):
            return da.sum(f, axis=(0, 1)) * dvx
        """
    cb = _make_function("callback", src)
    helpers = {"density": _make_function("density", src)}
    hints = analyze_callback(cb, {"f": arr}, helpers=helpers)
    assert _hint_keys(hints) == ["f-sum"]


def test_compute_loop_static_range() -> None:
    arr = da.zeros((4, 10, 10), chunks=(1, 5, 5), dtype=np.float64)
    src = """
        def callback(arr):
            Nsp = 4
            for isp in range(Nsp):
                sp = arr[isp]
                m = sp.sum()
                m.compute()
        """
    cb = _make_function("callback", src)
    hints = analyze_callback(cb, {"f": arr})
    # Loop unrolls statically; we get at least one reduction hint
    assert any(h["op_name"] == "sum" for h in hints)


def test_compute_window_subscript_negative_one() -> None:
    """``window[-1].sum().compute()`` should map to the last registered array."""
    arr = _simple_stub()
    src = """
        def callback(window):
            result = window[-1].sum()
            result.compute()
        """
    cb = _make_function("callback", src)
    hints = analyze_callback(cb, {"f": arr})
    assert _hint_keys(hints) == ["f-sum"]


# ---------------------------------------------------------------------------
# client.compute / client.submit boundaries
# ---------------------------------------------------------------------------
def test_client_compute_single_array() -> None:
    """``client.compute(arr)`` should register the dask array as a boundary."""
    arr = _simple_stub()
    client_stub = _FakeClient()
    src = """
        def callback(arr, client):
            client.compute(da.sum(arr))
        """
    cb = _make_function("callback", src)
    hints = analyze_callback(cb, {"f": arr, "client": client_stub})
    assert _hint_keys(hints) == ["f-sum"]


def test_client_compute_list_of_arrays() -> None:
    """``client.compute([a, b, c])`` should register each array as a boundary."""
    arr = _simple_stub()
    client_stub = _FakeClient()
    src = """
        def callback(arr, client):
            client.compute([
                da.sum(arr),
                da.mean(arr),
                da.max(arr),
            ])
        """
    cb = _make_function("callback", src)
    hints = analyze_callback(cb, {"f": arr, "client": client_stub})
    assert _hint_keys(hints) == ["f-max", "f-mean", "f-sum"]


def test_client_submit_with_dask_array() -> None:
    """``client.submit(func, arr)`` should register the dask array as a boundary."""
    arr = _simple_stub()
    client_stub = _FakeClient()
    src = """
        def callback(arr, client):
            client.submit(float, da.sum(arr))
        """
    cb = _make_function("callback", src)
    hints = analyze_callback(cb, {"f": arr, "client": client_stub})
    assert _hint_keys(hints) == ["f-sum"]


def test_client_compute_inside_helper() -> None:
    """The compute boundary can live inside a helper function."""
    arr = _simple_stub()
    client_stub = _FakeClient()
    src = """
        def callback(arr, client):
            measure(client, arr)

        def measure(client, f):
            client.compute(da.sum(f))
        """
    cb = _make_function("callback", src)
    helpers = {"measure": _make_function("measure", src)}
    hints = analyze_callback(cb, {"f": arr, "client": client_stub}, helpers=helpers)
    assert _hint_keys(hints) == ["f-sum"]


# ---------------------------------------------------------------------------
# Materialization (errors)
# ---------------------------------------------------------------------------
def test_np_array_raises_materialization() -> None:
    arr = _simple_stub()
    src = """
        def callback(arr):
            full = np.array(arr)
            full.compute()
        """
    cb = _make_function("callback", src)
    with pytest.raises(MaterializationError):
        analyze_callback(cb, {"f": arr})


def test_np_asarray_raises_materialization() -> None:
    arr = _simple_stub()
    src = """
        def callback(arr):
            full = np.asarray(arr)
            full.compute()
        """
    cb = _make_function("callback", src)
    with pytest.raises(MaterializationError):
        analyze_callback(cb, {"f": arr})


# ---------------------------------------------------------------------------
# Non-reduction compute boundaries
# ---------------------------------------------------------------------------
def test_compute_fft_only_raises_no_precomputable_reduction() -> None:
    """A callback with .compute() but no reductions should raise NoPrecomputableReductionError."""
    arr = da.zeros((10, 10), chunks=(10, 10), dtype=np.float64)  # single chunk so FFT works
    src = """
        def callback(arr):
            phi = da.fft.fft2(arr)
            phi.compute()
        """
    cb = _make_function("callback", src)
    with pytest.raises(NoPrecomputableReductionError):
        analyze_callback(cb, {"f": arr})


def test_compute_no_boundary_raises_no_compute_boundary() -> None:
    """A callback with dask operations but no compute boundary should raise NoComputeBoundaryError."""
    arr = _simple_stub()
    src = """
        def callback(arr):
            result = arr.sum()
            # result not computed - no boundary
        """
    cb = _make_function("callback", src)
    with pytest.raises(NoComputeBoundaryError):
        analyze_callback(cb, {"f": arr})


def test_callback_with_no_dask_raises_no_compute_boundary() -> None:
    """A callback with no dask at all should raise NoComputeBoundaryError."""
    arr = _simple_stub()
    src = """
        def callback(arr):
            x = 1 + 2
        """
    cb = _make_function("callback", src)
    with pytest.raises(NoComputeBoundaryError):
        analyze_callback(cb, {"f": arr})


# ---------------------------------------------------------------------------
# Incompatible patterns
# ---------------------------------------------------------------------------
def test_getattr_raises_incompatible_callback() -> None:
    arr = _simple_stub()
    src = """
        def callback(arr):
            op = 'sum'
            result = getattr(arr, op)()
            result.compute()
        """
    cb = _make_function("callback", src)
    with pytest.raises(IncompatibleCallbackError):
        analyze_callback(cb, {"f": arr})


def test_dynamic_loop_raises_incompatible_callback() -> None:
    arr = _simple_stub()
    src = """
        def callback(arr):
            for x in get_items():
                s = x.sum()
                s.compute()
        """
    cb = _make_function("callback", src)
    with pytest.raises(IncompatibleCallbackError):
        analyze_callback(cb, {"f": arr})


# ---------------------------------------------------------------------------
# force=True
# ---------------------------------------------------------------------------
def test_force_skips_with_warning(caplog) -> None:
    """``force=True`` returns [] and logs a warning instead of raising."""
    arr = da.zeros((10, 10), chunks=(10, 10), dtype=np.float64)
    src = """
        def callback(arr):
            phi = da.fft.fft2(arr)
            phi.compute()
        """
    cb = _make_function("callback", src)
    with caplog.at_level("WARNING"):
        hints = analyze_callback(cb, {"f": arr}, force=True)
    # No reductions to precompute
    assert hints == []
    # At least one warning emitted
    assert any(
        "precompute" in str(rec.message).lower() or "reduc" in str(rec.message).lower() for rec in caplog.records
    )


def test_force_swallows_incompatible_callback(caplog) -> None:
    """``force=True`` swallows IncompatibleCallbackError too."""
    arr = _simple_stub()
    src = """
        def callback(arr):
            op = 'sum'
            result = getattr(arr, op)()
            result.compute()
        """
    cb = _make_function("callback", src)
    with caplog.at_level("WARNING"):
        hints = analyze_callback(cb, {"f": arr}, force=True)
    assert hints == []


# ---------------------------------------------------------------------------
# Gysela-style patterns
# ---------------------------------------------------------------------------
def test_gysela_density_helper() -> None:
    """``density(f, grid) = da.sum(f, axis=(0, 3, 4)) * grid.dvx * grid.dvy``."""
    arr = da.zeros((4, 5, 10, 10, 10), chunks=(1, 5, 5, 5, 5), dtype=np.float64)
    client_stub = _FakeClient()
    grid_obj = _FakeGrid(dvx=0.5, dvy=0.5)
    src = """
        def callback(arr, grid, client):
            n = density(arr, grid)
            client.compute(n)

        def density(f, grid):
            return da.sum(f, axis=(0, 3, 4)) * grid.dvx * grid.dvy
        """
    cb = _make_function("callback", src)
    helpers = {"density": _make_function("density", src)}
    hints = analyze_callback(cb, {"f": arr, "grid": grid_obj, "client": client_stub}, helpers=helpers)
    assert _hint_keys(hints) == ["f-sum"]


def test_gysela_measure_helper_loop() -> None:
    """The measure() helper has 5 reductions and is called in a loop over species.

    The compute boundary is the ``client.compute([...])`` inside measure.
    """
    fdistribu = da.zeros((4, 5, 10, 10, 10), chunks=(1, 5, 5, 5, 5), dtype=np.float64)
    client_stub = _FakeClient()
    grid_obj = _FakeGrid(
        dvx=0.5,
        dvy=0.5,
        vx=da.zeros((10,), chunks=(10,)),
        vy=da.zeros((10,), chunks=(10,)),
        dV_4D=0.25,
    )
    src = """
        def callback(fdistribu, grid, Nsp, client):
            for isp in range(Nsp):
                measure(client, grid, fdistribu[isp])

        def measure(client, cfg, f):
            vx_bc = cfg.vx[(None, None, slice(None), None)]
            vy_bc = cfg.vy[(None, None, None, slice(None))]
            v2 = vx_bc ** 2 + vy_bc ** 2
            ek = 0.5 * da.sum(f * v2) * cfg.dV_4D
            l2 = da.sum(f ** 2) * cfg.dV_4D
            ms = da.sum(f) * cfg.dV_4D
            mx = da.sum(f * vx_bc) * cfg.dV_4D
            my = da.sum(f * vy_bc) * cfg.dV_4D
            client.compute([ek, l2, ms, mx, my])
        """
    cb = _make_function("callback", src)
    helpers = {"measure": _make_function("measure", src)}
    hints = analyze_callback(
        cb,
        {"fdistribu": fdistribu, "grid": grid_obj, "Nsp": 4, "client": client_stub},
        helpers=helpers,
    )
    op_names = sorted(h["op_name"] for h in hints)
    # We expect 5 unique reductions per call site, possibly repeated per species loop iteration
    assert "sum" in op_names
    # At minimum we have at least 5 sum hints (one per reduction in measure)
    sum_count = op_names.count("sum")
    assert sum_count >= 5


def test_offline_compression_raises_materialization() -> None:
    """Gysela's compute_offline_compression uses np.array(...)."""
    arr = _simple_stub()
    src = """
        def callback(arr):
            full = np.array(arr[0])
        """
    cb = _make_function("callback", src)
    with pytest.raises(MaterializationError):
        analyze_callback(cb, {"fdistribu_offline": arr})


# ---------------------------------------------------------------------------
# Cross-reduction refusal (multi-reduction branches)
# ---------------------------------------------------------------------------
def test_cross_reduction_raises_unsupported_reduction_error() -> None:
    """``(arr - arr.mean()).sum()`` must raise, NOT silently produce wrong hints.

    The naive per-reduction hint extraction emits two hints:
    - f-mean: arr.mean() -> per-bridge {n, total}
    - f-sum:  (arr - mean).sum() -> per-bridge (chunk - chunk.mean()).sum()

    The f-sum hint is WRONG in multi-bridge setups because the bridge
    computes (chunk - chunk.mean()).sum() locally -- but the correct
    expression requires the GLOBAL mean, which lives across bridges.
    Per-bridge, (chunk - chunk.mean()).sum() is always 0, so the
    global answer is always 0 regardless of input data.

    We refuse the whole expression. With force=True the analyzer
    swallows the error and returns no hints (so the user gets the
    legacy full-chunk scatter path if they explicitly opt out).
    """
    arr = _simple_stub()
    src = """
        def callback(arr):
            return (arr - arr.mean()).sum().compute()
        """
    cb = _make_function("callback", src)
    from deisa.dask.precompute_analyzer import UnsupportedReductionError

    with pytest.raises(UnsupportedReductionError) as exc_info:
        analyze_callback(cb, {"f": arr})
    # The error message must name the offending reduction and the
    # cross-reduction dependency so the user can fix the callback.
    msg = str(exc_info.value).lower()
    assert "sum" in msg  # the outer reduction
    assert "mean" in msg  # the inner reduction it depends on
    assert "bridge" in msg or "global" in msg or "all bridges" in msg


def test_cross_reduction_nested() -> None:
    """Three-deep nested reductions must also be caught.

    ``(arr - arr.mean().sum()).sum()``: the outer sum depends on
    ``arr.mean().sum()`` (a sum of a mean), which itself depends on
    arr. The walker follows the chain all the way back to the inner
    aggregate and refuses.
    """
    arr = _simple_stub()
    src = """
        def callback(arr):
            return (arr - arr.mean().sum()).sum().compute()
        """
    cb = _make_function("callback", src)
    from deisa.dask.precompute_analyzer import UnsupportedReductionError

    with pytest.raises(UnsupportedReductionError):
        analyze_callback(cb, {"f": arr})


def test_cross_reduction_in_helper() -> None:
    """Cross-reduction inside a same-file helper is caught the same way.

    Mirrors the gysela diagnostic patterns where ``measure()`` and
    ``density()`` helpers may compose reductions. The analyzer walks
    helper bodies, so a cross-reduction dependency in a helper also
    triggers the refusal.
    """
    arr = _simple_stub()
    src = """
        def callback(arr):
            return drift(arr).compute()

        def drift(arr):
            return (arr - arr.mean()).sum()
        """
    cb = _make_function("callback", src)
    helpers = {"drift": _make_function("drift", src)}
    from deisa.dask.precompute_analyzer import UnsupportedReductionError

    with pytest.raises(UnsupportedReductionError):
        analyze_callback(cb, {"f": arr}, helpers=helpers)


def test_cross_reduction_force_true_returns_no_hints(caplog) -> None:
    """``force=True`` swallows the refusal with a warning.

    With force=True the user is explicitly opting out of the
    precompute safety net. The analyzer logs a warning and returns
    zero hints, which the registration layer turns into a
    legacy full-chunk scatter (or raises -- depending on the
    registration-time policy).
    """
    arr = _simple_stub()
    src = """
        def callback(arr):
            return (arr - arr.mean()).sum().compute()
        """
    cb = _make_function("callback", src)
    with caplog.at_level("WARNING"):
        hints = analyze_callback(cb, {"f": arr}, force=True)
    assert hints == []
    # A warning was emitted about the refusal.
    assert any(
        "unsupported" in str(rec.message).lower() or "reduc" in str(rec.message).lower() for rec in caplog.records
    )


def test_independent_reductions_not_refused() -> None:
    """Two INDEPENDENT reductions on the same array must NOT be refused.

    The walker only refuses when one reduction's chunk-stage depends
    on ANOTHER reduction's aggregate layer. Two independent
    reductions (no shared sub-expression) are emitted as separate
    hints and precomputed independently.
    """
    arr = _simple_stub()
    src = """
        def callback(arr):
            a = arr.sum().compute()
            b = arr.mean().compute()
        """
    cb = _make_function("callback", src)
    hints = analyze_callback(cb, {"f": arr})
    # Both reductions detected -- the walker does not refuse.
    op_names = _op_names(hints)
    assert "sum" in op_names
    assert "mean" in op_names


# ---------------------------------------------------------------------------
# Safety: callback is NEVER executed
# ---------------------------------------------------------------------------
def test_callback_never_executed_during_analysis() -> None:
    """The user's callback must never be invoked by analyze_callback."""
    call_count = {"n": 0}

    def cb(arr):
        call_count["n"] += 1
        result = arr.sum()
        result.compute()

    arr = _simple_stub()
    # Even if the analyzer is broken and calls the callback, we want to detect it.
    analyze_callback(cb, {"f": arr})
    assert call_count["n"] == 0, "analyze_callback invoked the user's callback — AST-only contract broken"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@dataclass
class _FakeGrid:
    """A grid config stub with dask-array attributes."""

    dvx: float = 0.5
    dvy: float = 0.5
    dV_4D: float = 0.25
    vx: Any = None
    vy: Any = None


class _FakeClient:
    """Stub for a distributed.Client.

    The analyzer treats ``client.compute(...)`` and ``client.submit(...)`` as
    compute boundaries; the receiver's actual class doesn't matter, just its
    attributes. Returning ``None`` from these methods lets the analysis
    continue without raising.
    """

    def compute(self, *args: Any, **kwargs: Any) -> None:
        return None

    def submit(self, *args: Any, **kwargs: Any) -> None:
        return None


def _make_function(name: str, src: str) -> Callable:
    """Compile a small Python snippet and return the named function."""
    import linecache

    src = textwrap.dedent(src)
    scope: Dict[str, Any] = {}
    code = compile(src, f"<test_precompute:{name}>", "exec")
    exec(code, scope)
    fn = scope[name]
    # Register the source so ``inspect.getsource`` can find it.
    fn.__source__ = src  # type: ignore[attr-defined]
    # Also pin linecache so tools like inspect that rely on file/line lookup
    # can resolve the source.
    linecache.cache[fn.__code__.co_filename] = (
        len(src),
        None,
        [line + "\n" for line in src.splitlines()],
        fn.__code__.co_filename,
    )
    return fn
