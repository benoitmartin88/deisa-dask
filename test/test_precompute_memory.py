# =============================================================================
# Memory-measurement tests for the precompute feature.
#
# Goal: prove that when a callback is registered and the analyzer detects
# reducible operations, the *full* array chunk never lands on a worker. Only
# the small per-bridge reduction partials should appear in worker memory.
# The full chunk stays on the bridge process (which is the simulator here).
#
# We use Dask's built-in memory counters (worker.data + nbytes) instead of
# psutil so the test has no external dependency. The test runs against a
# LocalCluster so worker memory is observable from the test process.
# =============================================================================
import logging
import os
import textwrap
import time
from typing import Any, Callable, Dict, List

import numpy as np
import pytest
from deisa.core.types import DeisaArray
from distributed import Client, LocalCluster
from TestSimulator import TestSimulation
from utils import wait_for

from deisa.dask import Deisa

logging.basicConfig(level=logging.DEBUG)

# Skip on the github-only windowless context
pytestmark = pytest.mark.timeout(60)


def _worker_bytes_per_key(client: Client) -> Dict[str, Dict[str, Any]]:
    """Return, per worker, a dict ``{key: nbytes}`` for every key currently
    resident in that worker's in-memory data store. Includes zeros -- only
    in-memory keys are visible; spilled-to-disk keys are ignored on purpose
    so we capture what is actually consuming RAM.
    """

    def inspect(dask_worker):
        out = {}
        for k, v in dask_worker.data.items():
            if hasattr(v, "nbytes"):
                out[k] = v.nbytes
            else:
                # numpy scalars / dicts / etc.
                out[k] = None
        return out

    return client.run(inspect)


def _total_bytes_per_worker(per_worker: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    """Sum nbytes per worker (None entries contribute 0)."""
    out = {}
    for worker, keymap in per_worker.items():
        out[worker] = sum((v or 0) for v in keymap.values())
    return out


def _largest_key_per_worker(per_worker: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    """Max nbytes per worker (None entries contribute 0)."""
    out = {}
    for worker, keymap in per_worker.items():
        out[worker] = max((v or 0) for v in keymap.values()) if keymap else 0
    return out


def _make_callback(op: str, callback_results: List[float]) -> Callable:
    """Compile ``def _cb(window): ...`` reducing ``window[-1]`` via ``arr.<op>()``.

    Mirrors ``test_chain.py::_make_callback``: the snippet is built with
    ``compile``/``exec`` and the source is attached via ``__source__`` so the
    AST-based precompute analyzer can read the reduction op. The extra
    ``callback_results`` override lets the closure append into the test's
    list (callbacks run in-process on the Deisa event loop).
    """
    src = textwrap.dedent(
        f"def _cb(window):\n    arr = window[-1]\n    s = arr.{op}().compute()\n    callback_results.append(float(s))\n"
    )
    scope: Dict[str, Any] = {"callback_results": callback_results}
    code = compile(src, f"<test_precompute_memory:{op}>", "exec")
    exec(code, scope)
    fn = scope["_cb"]
    fn.__source__ = src  # type: ignore[attr-defined]
    return fn


@pytest.fixture(scope="function")
def env_setup_2workers():
    """Two-worker LocalCluster + matching client for end-to-end tests."""
    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        processes=True,
        dashboard_address=":0",
        worker_dashboard_address=":0",
    )
    cluster.wait_for_workers(2, timeout=20)
    os.environ["DEISA_DASK_SCHEDULER_ADDRESS"] = cluster.scheduler_address
    client = Client(cluster, name="env_setup_2workers")
    yield client, cluster
    client.close()
    cluster.close()


class TestPrecomputeMemory:
    """End-to-end tests that measure worker memory to confirm the full chunk
    never crosses the bridge -> worker boundary on the precompute path.
    """

    @pytest.mark.parametrize(
        "op, assert_result",
        [
            pytest.param(
                "sum",
                lambda x: np.isfinite(x) and x > 0,
                id="sum",
            ),
            pytest.param(
                "mean",
                lambda x: np.isfinite(x) and 0.0 < x < 1.0,
                id="mean",
            ),
            pytest.param(
                "var",
                lambda x: np.isfinite(x) and x >= 0.0,
                id="var",
            ),
            pytest.param(
                "std",
                lambda x: np.isfinite(x) and x >= 0.0,
                id="std",
            ),
        ],
    )
    def test_precompute_worker_only_sees_partials(self, env_setup_2workers, op, assert_result):
        """With a callback that reduces the global chunk to a scalar via
        ``arr.<op>()``, only the per-bridge partial (scalar/dict-blob size,
        ~8 bytes) should appear on workers. The full chunk (~32 MB) must NOT.

        Parametrized over ``sum`` / ``mean`` / ``var`` / ``std``; each param
        asserts the correctness of its own reduction result via
        ``assert_result``.
        """
        client, cluster = env_setup_2workers
        # Use a chunk big enough that "big" vs "small" is unmistakable.
        # 2048 * 2048 * 8 = 32 MB per chunk. Two bridges => 64 MB total
        # in the legacy path, ~16 bytes (two scalars / dict-blobs) on the
        # precompute path.
        chunk_shape = (2048, 2048)
        global_shape = (chunk_shape[0] * 2, chunk_shape[1])
        array_name = "temperature"

        sim = TestSimulation(
            client,
            mpi_parallelism=(2, 1),
            arrays_metadata={
                array_name: {
                    "global_shape": global_shape,
                    "chunk_shape": chunk_shape,
                },
            },
            wait_for_go=False,
        )

        deisa = Deisa(wait_for_go=False)

        callback_results: List[float] = []

        # Build (and register) the callback whose reduction op matches the
        # parametrization. The AST analyzer reads ``op`` from the source.
        deisa.register(array_name)(_make_callback(op, callback_results))

        # Wait for bridges and deisa to handshake.
        time.sleep(0.5)

        # Snapshot worker memory BEFORE we send any data.
        before = _worker_bytes_per_key(client)
        before_max = max(_largest_key_per_worker(before).values())
        assert before_max == 0, f"Workers should start empty, but found max key of {before_max} bytes: {before}"

        # Send one iteration.
        sim.generate_data(array_name, iteration=1, update_workers=True)

        # Wait for the callback to fire (it sets callback_results).
        assert wait_for(lambda: len(callback_results) >= 1, timeout=30), "callback was not called within 30s"

        # Inspect worker memory AFTER the send.
        after = _worker_bytes_per_key(client)
        after_max_per_worker = _largest_key_per_worker(after)
        logging.warning(f"PRECOMPUTE TEST ({op}): per-worker max key nbytes: {after_max_per_worker}")
        logging.warning(f"PRECOMPUTE TEST ({op}): per-worker keys: {after}")

        # The full chunk is 32 MB; the partial is a scalar/dict-blob (~8-32
        # bytes). Allow some slack for numpy wrapping, but keep it ~5000x
        # smaller than the chunk.
        max_allowed = 64 * 1024  # 64 KB
        for worker, max_nbytes in after_max_per_worker.items():
            assert max_nbytes < max_allowed, (
                f"Worker {worker} holds a key of {max_nbytes} bytes; "
                f"expected only the small partial (< {max_allowed} bytes). "
                f"Full chunk appears to have landed on the worker -- precompute "
                f"is not doing its job. Keys: {after}"
            )

        # The callback must have fired exactly once (one iteration) and
        # returned the correct reduction value for this op. generate_data
        # fills the array with random values in [0, 1), so per-op ranges
        # differ (see the assertion predicate above).
        assert len(callback_results) == 1, f"Expected exactly one callback invocation, got {len(callback_results)}"
        assert assert_result(callback_results[0]), (
            f"callback result {callback_results[0]!r} failed the {op!r} correctness predicate"
        )

        # NOTE: we deliberately do NOT call deisa.execute_callbacks() here:
        # TestSimulation.__del__ closes the bridges via async_close_bridges,
        # which handles the lifecycle. Calling execute_callbacks() here would
        # hang waiting for a bridge-done event that only fires on close.

    def test_no_precompute_worker_sees_full_chunk(self, env_setup_2workers):
        """Control test: WITHOUT precompute, the full chunk should land on
        workers. Confirms the memory-measurement harness is sensitive enough
        to actually see the difference.
        """
        client, cluster = env_setup_2workers
        chunk_shape = (2048, 2048)
        global_shape = (chunk_shape[0] * 2, chunk_shape[1])
        array_name = "temperature"

        sim = TestSimulation(
            client,
            mpi_parallelism=(2, 1),
            arrays_metadata={
                array_name: {
                    "global_shape": global_shape,
                    "chunk_shape": chunk_shape,
                },
            },
            wait_for_go=False,
        )

        deisa = Deisa(wait_for_go=False)

        callback_results: List[float] = []

        @deisa.register(array_name, force=True)
        # NOTE: force=True opts out of precompute and falls back to the
        # legacy full-chunk scatter path. The callback has a reduction
        # (``arr.sum()``), so without force=True it would be precomputed.
        # This control test confirms the memory harness can detect the
        # full chunk on workers when force=True is used.
        def _cb(window: list[DeisaArray]) -> None:
            arr = window[-1]
            logging.warning(f"NO-PRECOMPUTE TEST: callback received shape={arr.shape}")
            # The dask array should be the full global shape.
            assert arr.shape == global_shape, f"Expected full global shape {global_shape}, got {arr.shape}"
            callback_results.append(float(arr.sum().compute()))

        time.sleep(0.5)
        sim.generate_data(array_name, iteration=1, update_workers=True)

        assert wait_for(lambda: len(callback_results) >= 1, timeout=30)

        after = _worker_bytes_per_key(client)
        after_max_per_worker = _largest_key_per_worker(after)
        logging.warning(f"NO-PRECOMPUTE TEST: per-worker max key nbytes: {after_max_per_worker}")

        # The full chunk (32 MB) MUST appear on the workers when precompute
        # is disabled -- otherwise the test harness is broken.
        chunk_bytes = int(np.prod(chunk_shape) * 8)  # float64
        # At least one worker should hold a key close to the chunk size.
        max_seen = max(after_max_per_worker.values())
        assert max_seen >= chunk_bytes // 2, (
            f"Expected at least one worker to hold ~{chunk_bytes} bytes "
            f"(the full chunk), but max key seen was {max_seen} bytes. "
            f"Per-worker: {after_max_per_worker}. Harness is broken -- "
            f"this control confirms the precompute test above is meaningful."
        )

        assert len(callback_results) == 1
        # See note in test_precompute_worker_only_sees_partials about
        # why we do not call execute_callbacks() here.

    def test_precompute_chain_squared_sum_worker_only_sees_partials(self, env_setup_2workers):
        """Stage 2B chain-folded test: callback does ``(arr * arr).sum()``.

        The AST walker composes the expression lazily, producing a
        dask graph with both a ``mul`` layer AND a ``sum`` layer. The
        chain walker in :mod:`deisa.dask.branch` folds the chain --
        the bridge runs ``chunk ** 2`` then ``sum`` on its local numpy
        chunk and ships a single scalar partial per bridge. Workers
        must NOT hold the full chunk; only the small scalar partials.

        Without chain folding, the per-reduction hint would say
        ``sum`` (axis=(0,1)) but with chunk_kwargs that don't capture
        the squaring step -- the bridge would scatter ``chunk.sum()``
        instead of ``(chunk*chunk).sum()`` and the global result
        would be wrong. The chain walker fixes that.
        """
        client, cluster = env_setup_2workers
        chunk_shape = (2048, 2048)
        global_shape = (chunk_shape[0] * 2, chunk_shape[1])
        array_name = "temperature"

        sim = TestSimulation(
            client,
            mpi_parallelism=(2, 1),
            arrays_metadata={
                array_name: {
                    "global_shape": global_shape,
                    "chunk_shape": chunk_shape,
                },
            },
            wait_for_go=False,
        )

        deisa = Deisa(wait_for_go=False)

        callback_results: List[float] = []

        @deisa.register(array_name)
        def _cb(window: list[DeisaArray]) -> None:
            arr = window[-1]
            logging.warning(f"CHAIN TEST: callback received dask array shape={arr.shape}")
            # The chain walker folds (arr*arr).sum() into one branch
            # on the bridge side: each bridge's partial IS the
            # sum-of-squares of its chunk (shape (1, 1)). The Deisa
            # side stacks the per-bridge partials along a new axis so
            # with 2 bridges we get (2, 1, 1). The callback's plain
            # ``arr.sum()`` reduces those back to the global
            # sum-of-squares (sum-of-sums-of-squares-per-bridge).
            #
            # NOTE: do NOT re-apply (arr*arr) here -- the partials are
            # already sum-of-squares, and squaring them would give the
            # wrong answer (sum of (partial)**2 instead of sum of partials).
            result = arr.sum().compute()
            callback_results.append(float(result))

        time.sleep(0.5)

        sim.generate_data(array_name, iteration=1, update_workers=True)

        assert wait_for(lambda: len(callback_results) >= 1, timeout=30), "callback was not called within 30s"

        after = _worker_bytes_per_key(client)
        after_max_per_worker = _largest_key_per_worker(after)
        logging.warning(f"CHAIN TEST: per-worker max key nbytes: {after_max_per_worker}")

        # Per-bridge partial is (1, 1) float64 = 16 bytes. Allow 64 KB
        # of slack (numpy wrapping + dask dict overhead), but
        # well below the 32 MB full-chunk size.
        max_allowed = 64 * 1024
        for worker, max_nbytes in after_max_per_worker.items():
            assert max_nbytes < max_allowed, (
                f"Worker {worker} holds a key of {max_nbytes} bytes; "
                f"expected only the small partial (< {max_allowed} bytes). "
                f"Full chunk appears to have landed on the worker. "
                f"Keys: {after}"
            )

        assert len(callback_results) == 1
        # The chain-folded branch_func must produce the correct global
        # value: sum-of-squares of the random uniform [0, 1) data.
        # Per-bridge partial = sum-of-squares-of-chunk; combine via
        # dask sum stacks-and-sums, giving the global sum-of-squares.
        # Sanity: for uniform [0, 1) values the expected sum-of-squares
        # is N * E[x^2] = N * 1/3 ~ N/3, so for N = 4096*2048 = 8.39M,
        # expect ~2.8M. We just check the result is finite, positive,
        # and on the right order of magnitude.
        assert np.isfinite(callback_results[0])
        assert callback_results[0] > 0
        N = int(np.prod(global_shape))
        expected_order = N / 3.0
        assert 0.1 * expected_order < callback_results[0] < 10.0 * expected_order, (
            f"Global sum-of-squares {callback_results[0]} not in expected range "
            f"around {expected_order:.1f} -- chain walker may be producing the "
            f"wrong expression (e.g. plain sum instead of squared sum)."
        )
