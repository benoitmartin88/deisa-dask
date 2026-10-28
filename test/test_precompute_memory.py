# =============================================================================
# Memory-measurement tests for the reduction delivery path (the oracle port).
#
# Goal: prove that a callback that reduces the delivered operand never makes the
# FULL array chunk land on a dask worker. Only the small per-bridge payload (a
# scalar from the eager menu, or a chunk-local partial from an on-demand plan)
# should appear in worker memory.
#
# This file used to measure the AST precompute path: the analyzer planned the
# reductions at registration, the bridges shipped their partials with the topic
# event, and the test asserted that the worker store held only ~8-byte scalars.
# The runtime path plans nothing from the source -- the callback receives a
# lazily-reducible ``DeisaCollection`` and its reductions either read the eager
# menu shipped with the delivery or dispatch an on-demand plan -- so the SAME
# callback bodies are measured here against the runtime delivery. What changes
# is what the callback is handed (a collection describing the WHOLE array, not a
# ``(n_bridges,)`` stack of partials), which is why the shape assertions below
# now carry the discrimination instead of the stack shape.
#
# The memory assertions are the ones the plan calls the standing invariant
# (plan §7 P2): per-worker max key nbytes < 64 KB, and the full chunk never
# resident on a worker. The bound is 64 KB rather than the analyzer-era 8 bytes
# because an on-demand AXIS reduction legitimately lands a chunk-local partial,
# e.g. ``(1, 2048) float64`` = 16 KB, while the full chunk of this harness is
# 32 MB (three orders of magnitude larger).
#
# Every test here runs with RUNTIME_REDUCTION_DELIVERY_ENABLED on (flipped
# through the module, the pattern test_runtime_wiring.py established), except
# the control test, which pins that ``precompute=False`` still means the legacy
# full-chunk scatter even while the flag is on (decision F3).
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

from deisa.dask import Deisa, runtime_dispatch
from deisa.dask import runtime_errors as errors
from deisa.dask.collection import DeisaCollection
from deisa.dask.runtime_compose import compose_request, probe_partial
from deisa.dask.runtime_dispatch import runtime_delivery_enabled

logging.basicConfig(level=logging.DEBUG)

# Skip on the github-only windowless context
pytestmark = pytest.mark.timeout(60)

# The memory invariant of the runtime path: one worker never holds more than a
# chunk-local payload. The analyzer-era bound was 8 bytes; an on-demand axis
# partial is legitimately bigger, the full chunk (32 MB here) is not.
MAX_KEY_BYTES = 64 * 1024


def _worker_bytes_per_key(client: Client) -> Dict[str, Dict[str, Any]]:
    """Return, per worker, a dict ``{key: nbytes}`` for every key currently
    resident in that worker's in-memory data store. Includes zeros -- only in-memory keys are visible; spilled-to-disk
    keys are ignored on purpose so we capture what is actually consuming RAM.
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


def _largest_key_per_worker(per_worker: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    """Max nbytes per worker (None entries contribute 0)."""
    out = {}
    for worker, keymap in per_worker.items():
        out[worker] = max((v or 0) for v in keymap.values()) if keymap else 0
    return out


def _assert_only_partials_on_workers(after: Dict[str, Dict[str, Any]], chunk_bytes: int, label: str) -> None:
    """Assert the standing invariant: no worker key approaches the full chunk, and none exceeds the budget.

    Both halves are asserted because they fail differently: a worker holding the chunk shows up as a key at least half
    the chunk size (the legacy control demonstrates the harness sees exactly that), while a key between the budget and
    the chunk size would mean the payload is bigger than one chunk-local partial.
    """
    max_per_worker = _largest_key_per_worker(after)
    logging.warning(f"RUNTIME MEMORY TEST ({label}): per-worker max key nbytes: {max_per_worker}")
    logging.warning(f"RUNTIME MEMORY TEST ({label}): per-worker keys: {after}")
    for worker, max_nbytes in max_per_worker.items():
        assert max_nbytes < MAX_KEY_BYTES, (
            f"Worker {worker} holds a key of {max_nbytes} bytes; the runtime path delivers chunk-local payloads only "
            f"(< {MAX_KEY_BYTES} bytes). Keys: {after}"
        )
        assert max_nbytes < chunk_bytes // 2, (
            f"Worker {worker} holds a key of {max_nbytes} bytes, at least half the {chunk_bytes}-byte chunk: the FULL "
            f"chunk landed on a worker. Keys: {after}"
        )


def _make_callback(op: str, callback_results: List[float]) -> Callable:
    """Compile ``def _cb(window): ...`` reducing ``window[-1]`` via ``arr.<op>()``.

    The body is deliberately the analyzer-era body, unchanged: the same callback source must keep working under the
    runtime delivery, which is the point of the port. ``__source__`` is still attached because the legacy registration
    path reads it; the runtime path plans from the object model instead.
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


def _make_compute_callback(body: str, results: List[Any]) -> Callable:
    """Compile ``def _cb(window): <body>`` appending into ``results``.

    ``body`` is the dedented statement list (``textwrap.indent`` re-indents it inside the function). Used by the
    multi-reduction / axis tests whose callbacks call several reductions and store their outputs.
    """
    src = textwrap.dedent(f"def _cb(window):\n{textwrap.indent(body, '    ')}")
    scope: Dict[str, Any] = {"results": results, "np": np}
    code = compile(src, "<test_precompute_memory>", "exec")
    exec(code, scope)
    fn = scope["_cb"]
    fn.__source__ = src  # type: ignore[attr-defined]
    return fn


@pytest.fixture(scope="function", autouse=True)
def runtime_delivery(monkeypatch):
    """Run every test in this file on the RUNTIME delivery path.

    ``runtime_delivery_enabled`` reads the module global at call time, so flipping it through the module reaches every
    consumer (the topic handler, the register path and the bridge's boundary collective). The assertion pins that the
    port is really on the runtime path -- a silently flag-off test would measure the analyzer instead.
    """
    monkeypatch.setattr(runtime_dispatch, "RUNTIME_REDUCTION_DELIVERY_ENABLED", True)
    assert runtime_delivery_enabled() is True
    yield True


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


class TestRuntimeReductionMemory:
    """End-to-end tests that measure worker memory to confirm the full chunk
    never crosses the bridge -> worker boundary on the runtime reduction path.

    Everything the analyzer-era file measured still holds; what changed is the
    OBJECT the callback receives. ``window[-1]`` is now a ``DeisaCollection``
    describing the whole array, whose ``.compute()`` on a reduction returns the
    combined FINAL value -- so a delivered ``(2,)`` partial stack no longer
    exists, and the per-test discriminator is the collection's global shape plus
    the memory measurement.
    """

    @pytest.mark.parametrize(
        "op",
        ["sum", "mean", "var"],
    )
    def test_runtime_worker_only_sees_partials(self, env_setup_2workers, op):
        """A ``arr.<op>().compute()`` callback leaves only chunk-local payloads on the workers (~8-16 bytes).

        The full chunk (~32 MB) must NOT appear. Parametrized over the EAGER-MENU reductions (``sum`` / ``mean`` /
        ``var``): the menu is the zero-round-trip half of the runtime delivery, so one send is enough to answer a full
        reduction. ``std`` and every AXIS reduction are NOT menu entries -- they take the on-demand path, whose answer
        arrives at the NEXT send boundary -- so their payloads are measured directly (no cluster sample can see them
        in a one-send run) by ``test_on_demand_payloads_are_chunk_local`` below.

        The callback result is asserted against the true global value computed from the same data ``generate_data``
        returned (a loose ``x >= 0.0`` predicate would accept a buggy ``0.0`` result).
        """
        client, cluster = env_setup_2workers
        # Use a chunk big enough that "big" vs "small" is unmistakable.
        # 2048 * 2048 * 8 = 32 MB per chunk. Two bridges => 64 MB total
        # in the legacy path, a scalar per bridge on the runtime path.
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

        deisa.register(array_name)(_make_callback(op, callback_results))

        # Wait for bridges and deisa to handshake.
        time.sleep(0.5)

        # Snapshot worker memory BEFORE we send any data.
        before = _worker_bytes_per_key(client)
        before_max = max(_largest_key_per_worker(before).values())
        assert before_max == 0, f"Workers should start empty, but found max key of {before_max} bytes: {before}"

        # Send one iteration and keep the global ground-truth array.
        global_data = sim.generate_data(array_name, iteration=1, update_workers=True)

        # Wait for the callback to fire (it sets callback_results).
        assert wait_for(lambda: len(callback_results) >= 1, timeout=30), "callback was not called within 30s"

        # Inspect worker memory AFTER the send.
        _assert_only_partials_on_workers(_worker_bytes_per_key(client), int(np.prod(chunk_shape) * 8), label=f"{op}")

        # The callback must have fired exactly once (one iteration) and
        # returned the true reduction value for this op (var used to deliver 0.0,
        # which the old ``x >= 0.0`` predicates accepted).
        # ``generate_data`` fills the array with random values in [0, 1).
        assert len(callback_results) == 1, f"Expected exactly one callback invocation, got {len(callback_results)}"
        truth = {"sum": np.sum, "mean": np.mean, "var": np.var}[op](global_data)
        assert np.isclose(callback_results[0], truth, rtol=1e-5, atol=1e-9), (
            f"callback result {callback_results[0]!r} is not the true {op!r} of the global data "
            f"(expected {truth!r}). The runtime delivery must return the combined FINAL value, "
            f"never a value a re-applied callback op computes from a mis-shaped delivered operand."
        )

        # NOTE: we deliberately do NOT call deisa.execute_callbacks() here:
        # TestSimulation.__del__ closes the bridges via async_close_bridges,
        # which handles the lifecycle. Calling execute_callbacks() here would
        # hang waiting for a bridge-done event that only fires on close.

    def test_legacy_control_worker_sees_full_chunk(self, env_setup_2workers):
        """Control test: with ``precompute=False`` the full chunk DOES land on workers, even with the flag on.

        Two things are pinned at once: the memory harness is sensitive enough to see the difference, and
        ``precompute=False`` still selects the legacy full-chunk scatter while the runtime flag is on (decision F3 --
        the escape hatch must survive the redesign).
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

        @deisa.register(array_name, precompute=False)
        # NOTE: precompute=False opts out of the reduction path and falls back to
        # the legacy full-chunk scatter path, exactly as under the analyzer.
        def _cb(window: list[DeisaArray]) -> None:
            arr = window[-1]
            logging.warning(f"LEGACY CONTROL TEST: callback received shape={arr.shape}")
            # The legacy delivery hands over the full global array.
            assert arr.shape == global_shape, f"Expected full global shape {global_shape}, got {arr.shape}"
            callback_results.append(float(arr.sum().compute()))

        time.sleep(0.5)
        sim.generate_data(array_name, iteration=1, update_workers=True)

        assert wait_for(lambda: len(callback_results) >= 1, timeout=30)

        after = _worker_bytes_per_key(client)
        after_max_per_worker = _largest_key_per_worker(after)
        logging.warning(f"LEGACY CONTROL TEST: per-worker max key nbytes: {after_max_per_worker}")

        # The full chunk (32 MB) MUST appear on the workers when precompute
        # is disabled -- otherwise the test harness is broken.
        chunk_bytes = int(np.prod(chunk_shape) * 8)  # float64
        max_seen = max(after_max_per_worker.values())
        assert max_seen >= chunk_bytes // 2, (
            f"Expected at least one worker to hold ~{chunk_bytes} bytes "
            f"(the full chunk), but max key seen was {max_seen} bytes. "
            f"Per-worker: {after_max_per_worker}. Harness is broken -- "
            f"this control confirms the runtime test above is meaningful."
        )

        assert len(callback_results) == 1
        # See note in test_runtime_worker_only_sees_partials about
        # why we do not call execute_callbacks() here.

    def test_multi_array_runtime_end_to_end(self, env_setup_2workers):
        """Two arrays registered in ONE callback: both reductions are correct and rooted in their OWN array.

        The analyzer-era discriminator was a ``(2,)`` partial stack per array. Under the runtime delivery there is no
        stack: the callback receives one collection per array, describing the WHOLE array. The discriminators are
        therefore (a) the collection's global shape -- a legacy full-chunk tiled delivery would report the chunk shape,
        (b) per-array attribution, which is what keeps the two roots' attempts apart, and (c) the delivered values.
        """
        client, cluster = env_setup_2workers
        chunk_shape = (8, 16)
        global_shape = (16, 16)

        sim = TestSimulation(
            client,
            mpi_parallelism=(2, 1),
            arrays_metadata={
                "x": {"global_shape": global_shape, "chunk_shape": chunk_shape},
                "y": {"global_shape": global_shape, "chunk_shape": chunk_shape},
            },
            wait_for_go=False,
        )

        deisa = Deisa(wait_for_go=False)

        results: List[Any] = []

        @deisa.register("x", "y")
        def _cb(x_windows, y_windows):
            x = x_windows[-1]
            y = y_windows[-1]
            results.append(
                (
                    float(x.sum().compute()),
                    float(y.sum().compute()),
                    tuple(x.shape),
                    tuple(y.shape),
                    (x.name, y.name),
                    (isinstance(x, DeisaCollection), isinstance(y, DeisaCollection)),
                )
            )

        time.sleep(0.5)

        x_global, y_global = sim.generate_data("x", "y", iteration=1, update_workers=True)

        assert wait_for(lambda: len(results) >= 1, timeout=30), "callback was not called within 30s"

        x_sum, y_sum, x_shape, y_shape, names, are_collections = results[0]
        # Float64 summation ORDER differs between the chunk-local folds and a
        # direct full-array sum, so compare with a loose relative tolerance.
        assert np.isclose(x_sum, float(np.sum(x_global)))
        assert np.isclose(y_sum, float(np.sum(y_global)))
        # The runtime hands each callback ONE collection per array: the global
        # shape (never the chunk shape, never a partial stack) and the array name
        # are what prove per-array attribution reached the callback.
        assert x_shape == global_shape, f"x delivered shape {x_shape}, expected the GLOBAL {global_shape}"
        assert y_shape == global_shape, f"y delivered shape {y_shape}, expected the GLOBAL {global_shape}"
        assert names == ("x", "y"), f"per-array attribution lost: {names}"
        assert are_collections == (True, True)

    def test_multi_reduction_callback_receives_all_reductions(self, env_setup_2workers):
        """A 3-reduction callback receives all three true values.

        Every reduction must be served independently: if one reduction's attempt shadowed another's (a shared key), the
        other values would be far from the truth.
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

        results: List[Any] = []
        deisa.register(array_name)(
            _make_compute_callback(
                "arr = window[-1]\n"
                "s = arr.sum().compute()\n"
                "m = arr.mean().compute()\n"
                "mx = arr.max().compute()\n"
                "results.append((float(s), float(m), float(mx)))",
                results,
            )
        )

        time.sleep(0.5)
        global_data = sim.generate_data(array_name, iteration=1, update_workers=True)

        assert wait_for(lambda: len(results) >= 1, timeout=30), "callback was not called within 30s"
        s, m, mx = results[0]
        assert np.isclose(s, float(np.sum(global_data)), rtol=1e-5, atol=1e-9), f"sum {s} != {np.sum(global_data)}"
        assert np.isclose(m, float(np.mean(global_data)), rtol=1e-5, atol=1e-9), f"mean {m} != {np.mean(global_data)}"
        assert np.isclose(mx, float(np.max(global_data)), rtol=1e-5, atol=1e-9), f"max {mx} != {np.max(global_data)}"

    def test_same_op_two_axes_have_distinct_keys_and_chunk_local_payloads(self):
        """``arr.sum()`` and ``arr.sum(axis=0)`` carry DIFFERENT keys and different chunk-local partials.

        The e2e half of this test cannot run in a one-send harness: an AXIS reduction is not an eager-menu entry, so it
        takes the on-demand path, whose answer the bridge serves at the NEXT send boundary (``send`` ->
        ``_serve_pending_plans``; the two-send transport behaviour is pinned in ``test_runtime_transport.py``). What the
        memory invariant needs -- the SIZE of what crosses -- is fully determined by the composed request, so it is
        measured here without a cluster.

        Two attempts that shared a key would collapse in the bridge's output-key index and one reduction would answer
        with the other's partial.
        """
        chunk_shape = (2048, 2048)
        chunk_bytes = int(np.prod(chunk_shape) * 8)
        collection = DeisaCollection.from_metadata(
            "temperature", 1, {"global_shape": (chunk_shape[0] * 2, chunk_shape[1]), "chunk_shape": chunk_shape}
        )

        full = compose_request(collection.sum())
        axis0 = compose_request(collection.sum(axis=0))
        assert full.key != axis0.key, f"both sums share the key {full.key!r}"
        assert axis0.key.endswith("axis0"), axis0.key
        assert full.axes == () and axis0.axes == (0,)

        shape, dtype = probe_partial(axis0, chunk_shape, np.dtype(np.float64))
        assert shape == (1, chunk_shape[1]), shape
        assert dtype == np.dtype(np.float64)
        axis_bytes = int(np.prod(shape)) * np.dtype(np.float64).itemsize
        assert axis_bytes == chunk_shape[1] * 8
        assert axis_bytes < MAX_KEY_BYTES, f"axis partial {axis_bytes} bytes exceeds the {MAX_KEY_BYTES}-byte budget"
        assert chunk_bytes == axis_bytes * chunk_shape[0], "the chunk is 2048 axis partials -- the payload is not it"

    def test_multiple_callbacks_same_array_each_correct(self, env_setup_2workers):
        """Two callbacks on the same array each get their own result.

        The runtime builds a FRESH collection per callback (role 4 of the retired dispatch view, proven structurally in
        the ported ``test_precompute.py``), so one callback's ``sum`` cannot be answered from the other's ``mean``
        attempt: the attempt is keyed per (root, op, axes).
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

        sum_results: List[float] = []
        mean_results: List[float] = []
        deisa.register(array_name)(_make_callback("sum", sum_results))
        deisa.register(array_name)(_make_callback("mean", mean_results))

        time.sleep(0.5)
        global_data = sim.generate_data(array_name, iteration=1, update_workers=True)

        assert wait_for(lambda: len(sum_results) >= 1 and len(mean_results) >= 1, timeout=30), (
            "callbacks were not both called within 30s"
        )
        assert np.isclose(sum_results[0], float(np.sum(global_data)), rtol=1e-5, atol=1e-9), (
            f"sum callback {sum_results[0]} != {np.sum(global_data)}"
        )
        assert np.isclose(mean_results[0], float(np.mean(global_data)), rtol=1e-5, atol=1e-9), (
            f"mean callback {mean_results[0]} != {np.mean(global_data)}"
        )

    def test_chained_and_sliced_reductions_on_a_delivered_collection(self):
        """``(arr * arr).sum()`` is SERVED and ``arr[2:5].sum()`` is REFUSED -- the registration test ported.

        The original test asserted the analyzer's registration gate refused both as non-direct reductions. The runtime
        path classifies them differently, and both halves are pinned here on a real delivered collection:

        - ``(arr * arr).sum()`` is CORRECT: the pointwise program travels with the composed request, so the bridge folds
          ``chunk ** 2`` before summing. This is the analyzer's ``deliver_direct`` refusal made unnecessary.
        - ``arr[2:5].sum()`` is refused at the ITEM ACCESS with ``UnsupportedReductionError`` -- a slice of a registered
          field is a raw read, and the access is where the runtime can name it.

        Transitional note (K6a): the registration gate is still the analyzer until the deletion card lands, so
        ``deisa.register`` of these sources still raises the analyzer's class. This test therefore exercises the
        runtime PLANNER on the object a callback is delivered, which is what survives the deletion.
        """
        from utils import _dispatched

        field = np.random.default_rng(11).normal(size=(8, 8))
        collection = _dispatched(field)
        assert np.isclose(float((collection * collection).sum().compute()), float(np.sum(field**2)))

        with pytest.raises(errors.UnsupportedReductionError) as refusal:
            collection[2:5].sum().compute()
        assert "raw data" in str(refusal.value)

    def test_on_demand_payloads_are_chunk_local(self):
        """``std`` and the AXIS reductions are NOT menu entries: they ship a chunk-local partial, never the chunk.

        The eager menu covers :data:`EAGER_MENU_OPS` (full folds only), so ``arr.std()`` and every
        ``arr.<op>(axis=...)`` take the on-demand path. The bridge answers those at the NEXT send boundary (``send`` ->
        ``_serve_pending_plans``), so a one-send worker-store sample cannot observe them; the payload is fully
        determined by the composed request, and it is measured here:

        - ``arr.std()`` -> the ``(1, 1)`` moment partial (8 bytes; the moment combine keeps the reduced axes),
        - ``arr.sum(axis=0)`` on a (2048, 2048) chunk -> ``(1, 2048) float64`` = 16 KB,
        - the full chunk is 32 MB, i.e. 2048 axis partials.

        Everything that crosses is therefore under the 64 KB budget, and the chunk is never of them.
        """
        from deisa.dask.runtime_dispatch import EAGER_MENU_OPS

        chunk_shape = (2048, 2048)
        chunk_bytes = int(np.prod(chunk_shape) * 8)
        collection = DeisaCollection.from_metadata(
            "temperature", 1, {"global_shape": (chunk_shape[0] * 2, chunk_shape[1]), "chunk_shape": chunk_shape}
        )

        # The menu is full folds only -- this is WHY std/axis need the on-demand path (and why the one-send harness
        # cannot sample them).
        assert set(EAGER_MENU_OPS) == {"sum", "min", "max", "mean", "var", "prod"}
        assert "std" not in EAGER_MENU_OPS

        for expression, expected_shape in (
            (collection.std(), (1, 1)),
            (collection.sum(axis=0), (1, chunk_shape[1])),
            (collection.mean(axis=1), (chunk_shape[0], 1)),
        ):
            request = compose_request(expression)
            shape, dtype = probe_partial(request, chunk_shape, np.dtype(np.float64))
            assert shape == expected_shape, f"{request.key}: partial {shape}, expected {expected_shape}"
            assert np.dtype(dtype) == np.dtype(np.float64)
            nbytes = max(1, int(np.prod(shape))) * np.dtype(dtype).itemsize
            assert nbytes < MAX_KEY_BYTES, f"{request.key} ships {nbytes} bytes, over the {MAX_KEY_BYTES}-byte budget"
            assert nbytes < chunk_bytes // 2, f"{request.key} ships at least half the {chunk_bytes}-byte chunk"

        axis_bytes = chunk_shape[1] * 8  # (1, 2048) float64
        assert axis_bytes < MAX_KEY_BYTES, "the budget must admit a chunk-local axis partial"
        assert chunk_bytes == axis_bytes * chunk_shape[0], "the chunk is 2048 axis partials -- the payload is not it"
