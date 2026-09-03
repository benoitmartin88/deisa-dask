# =============================================================================
# Memory-measurement tests for the precompute feature.
#
# Goal: prove that when a callback is registered with precompute=True and the
# analyzer detects reducible operations, the *full* array chunk never lands
# on a worker. Only the small per-bridge reduction partials should appear in
# worker memory. The full chunk stays on the bridge process (which is the
# simulator here).
#
# We use Dask's built-in memory counters (worker.data + nbytes) instead of
# psutil so the test has no external dependency. The test runs against a
# LocalCluster so worker memory is observable from the test process.
# =============================================================================
import logging
import os
import time
from typing import Any, Dict, List

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

    def test_precompute_worker_only_sees_partials(self, env_setup_2workers):
        """With precompute=True and a callback that reduces to a scalar, only
        the per-bridge partials (scalar size, ~8 bytes) should appear on
        workers. The full chunk (~8 MB) must NOT appear.
        """
        client, cluster = env_setup_2workers
        # Use a chunk big enough that "big" vs "small" is unmistakable.
        # 2048 * 2048 * 8 = 32 MB per chunk. Two bridges => 64 MB total
        # in the legacy path, 16 bytes (two scalars) on the precompute path.
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

        @deisa.register(array_name, precompute=True)
        def _cb(window: list[DeisaArray]) -> None:
            # `window[-1]` is the dask array the topic handler built. It
            # should be the STACKED per-bridge partials (shape (2,)) when
            # precompute is active, NOT the full chunk shape.
            arr = window[-1]
            logging.warning(f"PRECOMPUTE TEST: callback received dask array shape={arr.shape}")
            # Compute the global sum via dask's natural reduction over
            # the stacked per-bridge partials.
            s = arr.sum().compute()
            callback_results.append(float(s))

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
        logging.warning(f"PRECOMPUTE TEST: per-worker max key nbytes: {after_max_per_worker}")
        logging.warning(f"PRECOMPUTE TEST: per-worker keys: {after}")

        # The full chunk is 32 MB; the partial is a scalar (~8 bytes).
        # Allow some slack for numpy wrapping (a scalar's nbytes can show
        # as the array's dtype size, typically 8 bytes for float64).
        max_allowed = 64 * 1024  # 64 KB -- 5000x smaller than the chunk
        for worker, max_nbytes in after_max_per_worker.items():
            assert max_nbytes < max_allowed, (
                f"Worker {worker} holds a key of {max_nbytes} bytes; "
                f"expected only the small partial (< {max_allowed} bytes). "
                f"Full chunk appears to have landed on the worker -- precompute "
                f"is not doing its job. Keys: {after}"
            )

        # The callback must have returned the correct global sum.
        # generate_data fills the array with random values in [0, 1).
        # The exact value differs run-to-run, but the global sum is finite
        # and > 0, and we got exactly one callback invocation.
        assert len(callback_results) == 1, f"Expected exactly one callback invocation, got {len(callback_results)}"
        assert np.isfinite(callback_results[0])
        assert callback_results[0] > 0

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

        @deisa.register(array_name)
        # NOTE: no precompute=True
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
