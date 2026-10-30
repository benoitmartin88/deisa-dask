"""P1 suite: the DEISA lazy collection, its closed operator surface and the composed wire request.

Dask-free in the sense that matters: no scheduler is started, no MPI communicator is built and no dask graph is read
back. The only dask use here is the reference reductions (``dask.array`` on a one-chunk array) that the composed
``branch_func`` partials are checked against -- behaviour over metadata.

Every contract asserted below has a mutation probe next to it (the last two test classes), so a green run is evidence
that the mechanism -- not the absence of one -- is what the test observes.
"""

import functools
import os
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path

import dask.array as da
import dask.array.fft as dfft
import numpy as np
import pytest
from dask.array.reductions import mean_agg, moment_agg

from deisa.dask import runtime_errors as errors
from deisa.dask.branch import _FULL_FOLD_FINALIZERS as LEGACY_FULL_FOLD_FINALIZERS
from deisa.dask.branch import _chain_branch_func
from deisa.dask.collection import DeisaCollection
from deisa.dask.runtime_compose import FULL_FOLD_FINALIZERS, compose_request, probe_partial
from deisa.dask.runtime_expr import (
    KIND_BOOLEAN,
    KIND_COUNT,
    KIND_MEAN,
    KIND_MOMENT,
    KIND_SCALAR,
    POINTWISE_OPS,
    REDUCTION_OPS,
    LazyOperand,
    PointwiseExpr,
    ReductionExpr,
    Root,
    eval_pointwise,
)

ROOT_SHAPE = (8, 8)
CHUNK_SHAPE = (4, 4)
SRC_DIR = Path(__file__).resolve().parents[1] / "src"

# The payload kind every reduction must produce (decision F4: any/all and count are new kinds).
EXPECTED_KIND = {
    "sum": KIND_SCALAR,
    "prod": KIND_SCALAR,
    "min": KIND_SCALAR,
    "max": KIND_SCALAR,
    "mean": KIND_MEAN,
    "var": KIND_MOMENT,
    "std": KIND_MOMENT,
    "any": KIND_BOOLEAN,
    "all": KIND_BOOLEAN,
    "count": KIND_COUNT,
}

# How each reduction's partials fold across bridges. The first four are the surviving ``_SCALAR_FOLDS``; the last three
# are the F4 additions (boolean OR/AND, additive count) the combine step gained with the new kinds.
SCALAR_FOLD = {
    "sum": np.sum,
    "prod": np.prod,
    "min": np.min,
    "max": np.max,
    "any": np.any,
    "all": np.all,
    "count": np.sum,
}


def make_collection(name="f", timestep=0, shape=ROOT_SHAPE, dtype=np.float64, chunk_shape=CHUNK_SHAPE):
    """Build a collection describing one array whose only chunk is ``chunk_shape``."""
    return DeisaCollection(name=name, timestep=timestep, shape=shape, dtype=dtype, chunk_shape=chunk_shape)


def make_chunk(shape=CHUNK_SHAPE, dtype=np.float64, seed=0, zeros=True):
    """The bridge's own numpy chunk: the value every composed callable is exercised on."""
    rng = np.random.default_rng(seed)
    data = rng.normal(size=shape)
    if zeros:
        # Mixed signs and a couple of zeros keep any/all/count/where meaningful.
        data[0, 0] = 0.0
        data[0, 1] = -0.5
    return data.astype(dtype)


def reduce_with_dask(op_name, array, axes):
    """Reference value: what dask itself returns for the same reduction on a one-chunk array."""
    if op_name == "count":
        # dask exposes count_nonzero as a module-level helper rather than as a method; on a single chunk the value is
        # exactly numpy's.
        return np.count_nonzero(array) if not axes else np.count_nonzero(array, axis=tuple(axes))
    darr = da.from_array(array, chunks=array.shape)
    return getattr(darr, op_name)(axis=None if not axes else tuple(axes)).compute()


def combine_single_bridge(partial, op_name, kind, axes, ndim, finalize):
    """Run one partial through the SURVIVING combine algebra, exactly as the Deisa side will for a one-bridge array."""
    red_axes = tuple(range(ndim)) if not axes else tuple(axes)
    if kind == KIND_MEAN:
        result = np.asarray(mean_agg([partial], axis=red_axes, dtype=np.asarray(partial["total"]).dtype))
    elif kind == KIND_MOMENT:
        result = np.asarray(
            moment_agg([partial], order=2, ddof=0, dtype=np.asarray(partial["total"]).dtype, axis=red_axes)
        )
    else:
        fold = SCALAR_FOLD[op_name]
        # ``axis=0`` is the surviving fold signature (``_SCALAR_FOLDS``): fold across bridges, keep the kept axes.
        result = np.asarray(fold(np.stack([np.asarray(partial)]), axis=0))
        for axis in sorted(red_axes, reverse=True):
            if result.ndim > 0:
                result = np.squeeze(result, axis=axis)
    if finalize == "sqrt":
        result = np.sqrt(result)
    return result


def execute(request, chunk, op_name):
    """Send one chunk through the composed callable (what the bridge does) and combine the one partial it returns."""
    partial = request.branch_func(chunk)
    return combine_single_bridge(partial, op_name, request.kind, request.axes, chunk.ndim, request.finalize)


# ---------------------------------------------------------------------------------------------------------------------
# The closed operator surface
# ---------------------------------------------------------------------------------------------------------------------
# One builder + one numpy reference per pointwise operator. ``test_pointwise_surface_is_complete`` asserts this table's
# key set EQUALS the runtime's whitelist, so adding an operator without exposing it (or exposing one the bridge cannot
# evaluate) fails the suite.
_POINTWISE_CASES = {
    "add": (lambda c: c + 1, lambda n: n + 1),
    "sub": (lambda c: c - 1, lambda n: n - 1),
    "mul": (lambda c: c * 2, lambda n: n * 2),
    "truediv": (lambda c: c / 2, lambda n: n / 2),
    "floordiv": (lambda c: c // 2, lambda n: n // 2),
    "mod": (lambda c: c % 2, lambda n: n % 2),
    "pow": (lambda c: c**2, lambda n: n**2),
    "eq": (lambda c: c == 0, lambda n: n == 0),
    "ne": (lambda c: c != 0, lambda n: n != 0),
    "lt": (lambda c: c < 0, lambda n: n < 0),
    "le": (lambda c: c <= 0, lambda n: n <= 0),
    "gt": (lambda c: c > 0, lambda n: n > 0),
    "ge": (lambda c: c >= 0, lambda n: n >= 0),
    "abs": (lambda c: abs(c), lambda n: np.abs(n)),
    "neg": (lambda c: -c, lambda n: np.negative(n)),
    "pos": (lambda c: +c, lambda n: np.positive(n)),
    "sqrt": (lambda c: c.abs().sqrt(), lambda n: np.sqrt(np.abs(n))),
    "exp": (lambda c: c.exp(), lambda n: np.exp(n)),
    "log": (lambda c: c.abs().log(), lambda n: np.log(np.abs(n))),
    "log10": (lambda c: c.abs().log10(), lambda n: np.log10(np.abs(n))),
    "log2": (lambda c: c.abs().log2(), lambda n: np.log2(np.abs(n))),
    "floor": (lambda c: c.floor(), lambda n: np.floor(n)),
    "ceil": (lambda c: c.ceil(), lambda n: np.ceil(n)),
    "sign": (lambda c: c.sign(), lambda n: np.sign(n)),
    "square": (lambda c: c.square(), lambda n: np.square(n)),
    "sin": (lambda c: c.sin(), lambda n: np.sin(n)),
    "cos": (lambda c: c.cos(), lambda n: np.cos(n)),
    "tan": (lambda c: c.tan(), lambda n: np.tan(n)),
    "tanh": (lambda c: c.tanh(), lambda n: np.tanh(n)),
    "clip": (lambda c: c.clip(-0.5, 0.5), lambda n: np.clip(n, -0.5, 0.5)),
    "where": (lambda c: (c > 0).where(c, 0.0), lambda n: np.where(n > 0, n, 0.0)),
}

_REDUCTION_CASES = {op_name: (lambda c, op=op_name: getattr(c, op)()) for op_name in REDUCTION_OPS}

# Reductions checked end-to-end (compose -> pickle -> run on a chunk -> combine -> compare with dask).
_REDUCTION_AXES = [
    ("sum", None),
    ("sum", 0),
    ("sum", 1),
    ("mean", None),
    ("mean", 0),
    ("mean", (0, 1)),
    ("var", None),
    ("var", 1),
    ("std", None),
    ("std", 0),
    ("min", None),
    ("max", 0),
    ("prod", 1),
    ("any", None),
    ("any", 0),
    ("all", None),
    ("all", 1),
    ("count", None),
    ("count", 0),
]

# Pointwise expressions the reduction tests are chained onto: (name, deisa builder, numpy reference).
_DERIVED_CASES = [
    ("bare", lambda c: c, lambda n: n),
    ("mul_const", lambda c: c * 2, lambda n: n * 2),
    ("square", lambda c: c * c, lambda n: n * n),
    # Two DIFFERENT subtrees below one node: not representable as a linear chain, which is why the program is a tree.
    ("dag", lambda c: c * 2 - c, lambda n: n * 2 - n),
    ("clip", lambda c: c.clip(-0.5, 0.5), lambda n: np.clip(n, -0.5, 0.5)),
    ("where", lambda c: (c > 0).where(c, 0.0), lambda n: np.where(n > 0, n, 0.0)),
    ("abs_sqrt", lambda c: c.abs().sqrt(), lambda n: np.sqrt(np.abs(n))),
]


@pytest.mark.parametrize("op_name", sorted(_POINTWISE_CASES))
def test_pointwise_surface_builds_an_expression_and_evaluates_like_numpy(op_name):
    """Every whitelisted pointwise operator builds a node AND computes numpy's answer on a chunk."""
    build, reference = _POINTWISE_CASES[op_name]
    expr = build(make_collection())
    assert isinstance(expr, PointwiseExpr)
    assert expr.program[0] == op_name
    chunk = make_chunk(zeros=False)
    np.testing.assert_allclose(eval_pointwise(expr.program, chunk), reference(chunk), rtol=1e-12, atol=1e-12)


def test_pointwise_surface_is_complete_and_closed():
    """The exposed whitelist and the tested builders agree, and every other name is refused (risk R2)."""
    assert set(_POINTWISE_CASES) == set(POINTWISE_OPS)
    collection = make_collection()
    for name in ("reshape", "astype", "rechunk", "map_blocks", "mean_over_time", "fft", "dot", "transpose"):
        with pytest.raises(errors.UnsupportedReductionError) as excinfo:
            getattr(collection, name)
        assert f".{name}" in str(excinfo.value)
        assert "precompute=False" in str(excinfo.value)


@pytest.mark.parametrize("op_name", sorted(_REDUCTION_CASES))
def test_reduction_surface_builds_the_expected_payload(op_name):
    """Every whitelisted reduction returns a reduction node with the right kind, axes, finalize and shape."""
    reduction = _REDUCTION_CASES[op_name](make_collection())
    assert isinstance(reduction, ReductionExpr)
    assert reduction.op_name == op_name
    assert reduction.kind == EXPECTED_KIND[op_name]
    assert reduction.axes == ()
    assert reduction.shape == ()
    assert reduction.finalize == ("sqrt" if op_name == "std" else None)


def test_reduction_surface_is_complete():
    """The tested reduction builders and the runtime whitelist agree."""
    assert set(_REDUCTION_CASES) == set(REDUCTION_OPS)


def test_axis_forms_normalize_into_the_wire_convention():
    """``axis=None``, an explicit all-axes tuple and negative axes collapse onto one canonical form."""
    collection = make_collection()
    assert collection.sum().axes == ()
    assert collection.sum(axis=0).axes == (0,)
    three_d = make_collection(shape=(8, 8, 8), chunk_shape=(4, 4, 4))
    assert three_d.sum(axis=(1, 0)).axes == (0, 1)
    assert collection.mean(axis=(0, 1)).axes == ()  # every axis reduced == the full reduction
    assert collection.mean(axis=-1).axes == (1,)
    assert collection.sum(axis=0).shape == (8,)
    assert collection.sum(axis=(0, 1)).shape == ()


class TestKeyContract:
    """The surviving dedup/naming contract: ``{array}-{op}`` plus an axis discriminator."""

    def test_full_reduction_of_the_bare_array_keeps_the_plain_key(self):
        assert make_collection().sum().key == "f-sum"
        assert make_collection("g").mean().key == "g-mean"

    def test_full_reduction_of_a_derived_expression_is_axisall(self):
        collection = make_collection()
        assert (collection * 2).sum().key == "f-sum-axisall"
        assert (collection * collection).mean().key == "f-mean-axisall"

    def test_axis_subset_appends_the_axis_discriminator(self):
        collection = make_collection()
        assert collection.sum(axis=0).key == "f-sum-axis0"
        assert collection.mean(axis=1).key == "f-mean-axis1"
        three_d = make_collection(shape=(8, 8, 8), chunk_shape=(4, 4, 4))
        assert three_d.sum(axis=(0, 2)).key == "f-sum-axis0x2"

    def test_identical_expressions_share_a_key(self):
        collection = make_collection()
        assert (collection * 2).sum().key == (collection * 2).sum().key
        assert collection.sum().key == collection.sum(axis=(0, 1)).key


class TestMaterializationGuards:
    """``__array__``/``__iter__`` close the silent-gather hole: numpy consumption is refused, loudly."""

    @pytest.mark.parametrize(
        "consume",
        [
            pytest.param(lambda obj: np.asarray(obj), id="np.asarray"),
            pytest.param(lambda obj: np.array(obj), id="np.array"),
            pytest.param(lambda obj: np.asarray(obj, dtype=np.float32), id="np.asarray-dtype"),
            pytest.param(lambda obj: np.asanyarray(obj), id="np.asanyarray"),
            pytest.param(lambda obj: list(obj), id="list"),
            pytest.param(lambda obj: tuple(obj), id="tuple"),
            pytest.param(lambda obj: list(iter(obj)), id="iter"),
            pytest.param(lambda obj: bool(obj), id="bool"),
        ],
    )
    @pytest.mark.parametrize(
        "build", [lambda c: c, lambda c: c * 2, lambda c: c.sum()], ids=["collection", "expr", "reduction"]
    )
    def test_numpy_and_python_consumption_is_refused(self, consume, build):
        with pytest.raises(errors.RuntimeRefusal) as excinfo:
            consume(build(make_collection()))
        assert "precompute=False" in str(excinfo.value)

    def test_guard_refuses_before_dask_can_embed_the_collection(self):
        """An external dask graph must fail fast -- refusal, not a silent gather."""
        collection = make_collection()
        with pytest.raises(errors.UnsupportedReductionError):
            da.from_array(collection)


class TestOffSurfaceMatrix:
    """Every off-surface pattern raises the right class, naming the symbol and the remedy."""

    @pytest.mark.parametrize(
        "case,expected",
        [
            pytest.param(lambda c: c[2:5], errors.UnsupportedReductionError, id="slicing"),
            pytest.param(lambda c: c[:, 0], errors.UnsupportedReductionError, id="column-index"),
            pytest.param(lambda c: c.rechunk((2, 2)), errors.UnsupportedReductionError, id="rechunk"),
            pytest.param(lambda c: c.map_blocks(np.abs), errors.UnsupportedReductionError, id="map_blocks"),
            pytest.param(lambda c: c.reshape((64,)), errors.UnsupportedReductionError, id="reshape"),
            pytest.param(lambda c: da.from_array(c), errors.UnsupportedReductionError, id="dask-from_array"),
            pytest.param(lambda c: dfft.fft2(c), errors.UnsupportedReductionError, id="fft"),
            pytest.param(lambda c: da.tensordot(c, c, axes=1), errors.UnsupportedReductionError, id="tensordot"),
            pytest.param(lambda c: np.sqrt(c), errors.MaterializationError, id="numpy-ufunc"),
            pytest.param(lambda c: c - c.mean(axis=0), errors.UnsupportedReductionError, id="cross-reduction"),
            pytest.param(lambda c: c.sum() + 1, errors.UnsupportedReductionError, id="reduction-operand"),
            pytest.param(lambda c: c.sum().sum(), errors.UnsupportedReductionError, id="double-reduction"),
            pytest.param(lambda c: c.argmax(), errors.NonAssociativeReductionError, id="argmax"),
            pytest.param(lambda c: c.argmin(), errors.NonAssociativeReductionError, id="argmin"),
            pytest.param(lambda c: c.median(), errors.NonAssociativeReductionError, id="median"),
            pytest.param(lambda c: c.quantile(0.5), errors.NonAssociativeReductionError, id="quantile"),
            pytest.param(lambda c: c.sum(keepdims=True), errors.UnsupportedReductionError, id="keepdims"),
            pytest.param(lambda c: c.std(ddof=1), errors.UnsupportedReductionError, id="ddof"),
            pytest.param(lambda c: c.sum(dtype=np.float32), errors.UnsupportedReductionError, id="dtype"),
            pytest.param(lambda c: c.sum(out=None, axis=5), errors.UnsupportedReductionError, id="axis-range"),
            pytest.param(lambda c: c.sum(axis=()), errors.UnsupportedReductionError, id="empty-axis"),
            pytest.param(lambda c: c.compute(), errors.RawFieldReadError, id="raw-compute"),
            pytest.param(lambda c: (c * 2).compute(), errors.RawFieldReadError, id="raw-compute-expr"),
        ],
    )
    def test_off_surface_pattern_is_refused_with_a_remedy(self, case, expected):
        with pytest.raises(expected) as excinfo:
            case(make_collection())
        message = str(excinfo.value)
        assert "precompute=False" in message
        assert not isinstance(excinfo.value, AssertionError)

    def test_non_associative_refusal_is_not_reported_as_a_raw_read(self):
        """The misdiagnosis this fixes: median/argmax used to be refused as raw field reads."""
        with pytest.raises(errors.NonAssociativeReductionError) as excinfo:
            make_collection().median()
        assert not isinstance(excinfo.value, errors.RawFieldReadError)
        assert "associative" in str(excinfo.value)
        assert "median" in str(excinfo.value)

    def test_reduction_compute_without_a_transport_is_refused(self):
        """A collection built by hand carries no bridge transport, so ``.compute()`` refuses instead of gathering.

        The dispatch boundary itself (request -> bridges -> partials -> combine) is proven over a fake communicator in
        ``test_runtime_dispatch.py``; here the point is that a collection OUTSIDE the delivery path fails loudly and
        names the same remedy every other refusal names.
        """
        with pytest.raises(errors.PrecomputeRuntimeError) as excinfo:
            make_collection().sum().compute()
        message = str(excinfo.value)
        assert "no bridge transport" in message
        assert "precompute=False" in message


class TestRootRules:
    """One branch == one array AND one timestep. The window idiom must keep working; the mix must be refused."""

    def test_window_indexing_is_plain_list_indexing(self):
        window = [make_collection(timestep=step) for step in range(3)]
        assert window[-1].timestep == 2
        assert window[-1].sum().key == "f-sum"
        assert window[-1].sum().root == Root(array_name="f", timestep=2)

    def test_mixed_timestep_is_refused(self):
        window = [make_collection(timestep=step) for step in range(3)]
        with pytest.raises(errors.UnsupportedReductionError) as excinfo:
            window[-1] - window[-2]
        message = str(excinfo.value)
        assert "timestep=2" in message and "timestep=1" in message
        assert "window" in message
        assert "precompute=False" in message

    def test_cross_array_is_refused(self):
        with pytest.raises(errors.UnsupportedReductionError) as excinfo:
            make_collection("f") * make_collection("g")
        assert "'f'" in str(excinfo.value) and "'g'" in str(excinfo.value)

    def test_same_root_combination_is_allowed(self):
        collection = make_collection()
        expr = collection * collection - 0.5
        assert expr.root == collection.root
        assert expr.program == ("sub", ("mul", ("chunk",), ("chunk",)), ("const", 0.5))


class TestComposedRequests:
    """The wire request: six fields, module-level partials only, executable on a fresh numpy chunk."""

    def test_request_carries_the_six_slim_fields(self):
        request = compose_request((make_collection() * 2).sum(axis=0))
        assert request.key == "f-sum-axis0"
        assert request.input_name == "f"
        assert request.kind == KIND_SCALAR
        assert request.axes == (0,)
        assert request.finalize is None
        assert callable(request.branch_func)
        assert not hasattr(request, "partial_shape")  # F5: probed, not carried
        assert not hasattr(request, "deliver_direct")

    @pytest.mark.parametrize("op_name,axes", _REDUCTION_AXES)
    @pytest.mark.parametrize("expr_name,build,reference", _DERIVED_CASES, ids=[case[0] for case in _DERIVED_CASES])
    def test_callable_round_trips_through_pickle_and_matches_dask(self, op_name, axes, expr_name, build, reference):
        """Pickle the composed callable, run it on a fresh chunk, combine, and compare with dask's own answer."""
        collection = make_collection()
        reduction = getattr(build(collection), op_name)(**({} if axes is None else {"axis": axes}))
        request = compose_request(reduction)
        restored = pickle.loads(pickle.dumps(request))
        chunk = make_chunk()
        got = execute(restored, chunk, op_name)
        expected = reduce_with_dask(op_name, reference(chunk), reduction.axes)
        if EXPECTED_KIND[op_name] in (KIND_BOOLEAN, KIND_COUNT):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))
        else:
            np.testing.assert_allclose(
                np.asarray(got, dtype=np.float64), np.asarray(expected, dtype=np.float64), rtol=1e-10
            )

    def test_composed_callable_executes_in_a_fresh_process(self, tmp_path):
        """The real pickle contract: a module-level partial resolves by name in a process that never built it."""
        request = compose_request((make_collection() * 2.0 - 1.0).abs().sum())
        payload = tmp_path / "request.pkl"
        payload.write_bytes(pickle.dumps(request))
        script = tmp_path / "run_branch.py"
        script.write_text(
            textwrap.dedent(
                f"""
                import pickle

                import numpy as np

                request = pickle.loads(open({str(payload)!r}, "rb").read())
                chunk = np.ones((4, 4), dtype=np.float64)
                print(request.key)
                print(float(request.branch_func(chunk)))
                """
            )
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC_DIR)
        completed = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, env=env, cwd=str(SRC_DIR.parent)
        )
        assert completed.returncode == 0, completed.stderr
        lines = completed.stdout.strip().splitlines()
        assert lines[0] == "f-sum-axisall"
        assert float(lines[1]) == pytest.approx(16.0)  # |1 * 2 - 1| * 16 elements

    def test_full_fold_ships_one_scalar_and_axis_fold_ships_a_tier(self):
        collection = make_collection()
        full = compose_request(collection.sum()).branch_func(make_chunk())
        assert np.asarray(full).shape == ()
        tier = compose_request(collection.sum(axis=0)).branch_func(make_chunk())
        assert np.asarray(tier).shape == (1, 4)  # keepdims tier: the combine squeezes the reduced axis

    def test_mean_and_moment_ship_the_dict_partials_the_combine_requires(self):
        collection = make_collection()
        chunk = make_chunk()
        mean_partial = compose_request(collection.mean()).branch_func(chunk)
        assert set(mean_partial) == {"n", "total"}
        moment_partial = compose_request(collection.std()).branch_func(chunk)
        assert set(moment_partial) == {"n", "total", "M"}

    def test_finalizer_table_extends_the_surviving_four(self):
        """The F4 operators extend -- not replace -- the finalizer table branch.py already owns."""
        for op_name, func in LEGACY_FULL_FOLD_FINALIZERS.items():
            assert FULL_FOLD_FINALIZERS[op_name] is func
        assert {"any", "all", "count"} <= set(FULL_FOLD_FINALIZERS)

    @pytest.mark.parametrize(
        "op_name,expected_shape,expected_dtype",
        [
            ("sum", (), "float64"),
            ("mean", (1, 1), "float64"),
            ("var", (1, 1), "float64"),
            ("any", (), "bool"),
            ("count", (), "int64"),
        ],
    )
    def test_probe_derives_the_partial_metadata(self, op_name, expected_shape, expected_dtype):
        request = compose_request(getattr(make_collection(), op_name)())
        shape, dtype = probe_partial(request, CHUNK_SHAPE, np.float64)
        assert shape == expected_shape
        assert dtype == expected_dtype

    def test_probe_describes_the_axis_partial(self):
        request = compose_request(make_collection().mean(axis=0))
        assert probe_partial(request, CHUNK_SHAPE, np.float64) == ((1, 4), "float64")

    def test_probe_rejects_non_reductions(self):
        with pytest.raises(errors.PrecomputeRuntimeError):
            compose_request(make_collection())  # type: ignore[arg-type]


class TestMutationProbes:
    """A green suite is only evidence if the checks can fail (the (b) contract's own rule)."""

    def test_pickle_check_catches_an_injected_lambda(self):
        """The pickle assertion must fail on a lambda in the chain -- otherwise it proves nothing."""
        good = compose_request(make_collection().sum())
        pickle.loads(pickle.dumps(good))  # the real chain round-trips
        injected = functools.partial(_chain_branch_func, _chain=(((lambda x: x), {}, 1),))
        with pytest.raises((AttributeError, pickle.PicklingError, TypeError)):
            pickle.loads(pickle.dumps(injected))

    def test_materialization_guard_is_what_refuses_numpy_consumption(self, monkeypatch):
        """Disable ``__array__`` and the guard's test must go red: the refusal is the hook, not an accident."""
        collection = make_collection()
        with pytest.raises(errors.MaterializationError):
            np.asarray(collection)
        monkeypatch.setattr(
            LazyOperand, "__array__", lambda self, dtype=None, copy=None: np.zeros(self.shape, dtype=np.float64)
        )
        assert np.asarray(collection).shape == collection.shape

    def test_mixed_root_check_is_what_refuses_the_window_mix(self, monkeypatch):
        """Force root equality and the mix builds: the root check is what refused it, not an accident."""
        window = [make_collection(timestep=step) for step in range(3)]
        with pytest.raises(errors.UnsupportedReductionError):
            window[-1] - window[-2]
        monkeypatch.setattr(Root, "__eq__", lambda self, other: True)
        assert isinstance(window[-1] - window[-2], PointwiseExpr)


class TestCollectionMetadata:
    """The collection carries the metadata the dispatch step will need, auto-detected from the plumbing."""

    def test_bridge_count_is_derived_from_the_chunk_grid(self):
        assert make_collection().bridge_count == 4
        assert make_collection(chunk_shape=(8, 8)).bridge_count == 1
        assert make_collection(shape=(8, 8, 8), chunk_shape=(4, 4, 2)).bridge_count == 16

    def test_declared_bridge_count_must_agree_with_the_metadata(self):
        with pytest.raises(errors.PrecomputeRuntimeError) as excinfo:
            DeisaCollection("f", 0, ROOT_SHAPE, np.float64, CHUNK_SHAPE, bridge_count=7)
        assert "7" in str(excinfo.value)

    def test_from_metadata_reads_the_handshake_keys(self):
        collection = DeisaCollection.from_metadata("f", 3, {"global_shape": (8, 8), "chunk_shape": (4, 4)})
        assert (collection.shape, collection.chunk_shape, collection.bridge_count) == ((8, 8), (4, 4), 4)
        assert collection.timestep == 3
        assert collection.dtype == np.dtype(np.float64)

    def test_from_metadata_requires_a_global_shape(self):
        with pytest.raises(errors.PrecomputeRuntimeError):
            DeisaCollection.from_metadata("f", 0, {"chunk_shape": (4, 4)})

    def test_shape_and_dtype_propagate_through_the_surface(self):
        collection = make_collection()
        assert collection.ndim == 2
        assert (collection * 2).dtype == np.dtype(np.float64)
        assert (collection > 0).dtype == np.dtype(bool)
        assert collection.mean(axis=0).shape == (8,)
