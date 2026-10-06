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
Worker peak-memory benchmark: what each path leaves RESIDENT on the analytics side, measured by Dask's own
diagnostics (``distributed.diagnostics.memory_sampler.MemorySampler``), not by external polling.

The two paths (the semantics of ``benchmark/mergeable_pca/transfer_comparison.py``, in memory terms)
----------------------------------------------------------------------------------------------------------------
- ``legacy``  -- the full ``n_block x d`` chunk is the unit of transfer: every block is allocated co-resident
  (``n_blocks`` blocks at once) and scattered to the worker, and the scattered keys are HELD for the entire
  sampled window. The full chunk is resident on the analytics side by construction.
- ``summary`` -- the mergeable path computes before it ships: each block is allocated ONE AT A TIME,
  summarized by :func:`~deisa.dask.mergeable_pca.local_pca`, the small summary is scattered and held, and the
  block is released before the next one is built. Only one block plus a few KiB of summaries is ever resident.

Why the payload must be HELD inside the sampled window
------------------------------------------------------
On ``processes=False`` LocalCluster the scatter is zero-copy (the worker holds references to the arrays the
client already owns), so a scatter-then-cancel round trip moves no bytes at all: a transient measurement of it
reads ~0 and ``peak = max(samples)`` then measures baseline jitter, not the workload -- which is exactly what a
probe of this box showed (a 122 MiB scattered-then-cancelled array produced ~0-1.1 MiB of sustained RSS delta).
The measurement therefore allocates the payload INSIDE the window and keeps the futures referenced until after
sampling ends, so the sampled peak is the residency the path actually implies. On a distributed cluster the
legacy path would additionally pay one wire + deserialization copy; this box cannot reproduce that and no
number here claims to.

What is sampled
---------------
Three concurrent :class:`~distributed.diagnostics.memory_sampler.MemorySampler` series over the SAME window,
one per ``distributed.scheduler.MemoryState`` measure: ``process`` (the worker's process RSS as reported by
the worker's own accounting -- with the in-process worker this is this pytest process), ``managed`` (the
``sizeof`` of the keys the worker holds) and ``unmanaged`` (``process - managed``). The row's
``peak_worker_mib`` is the ``process`` peak; ``managed_mib`` / ``unmanaged_mib`` are the peaks of their own
series; every series' sample count is recorded so a window that sampled too rarely cannot masquerade as a
measurement. The scheduler's per-worker ``metrics`` read at the end of the hold corroborate each row.

Determinism
-----------
The assert at the end (requirement: ``summary_peak < legacy_peak`` on the tall config) compares within-config
peaks of identical arrays, with two guards against cross-window contamination: before each window the
scheduler's reported ``managed_bytes`` is polled below 1 MiB (so a previous window's released payload cannot
inflate the next peak), and the allocator is trimmed between windows so freed blocks leave the process RSS.
``managed`` is the structural channel: it is the worker's own key accounting, so the legacy summary's
``managed_mib`` must cover the whole scattered payload.

Run
---
    .venv/bin/python -m pytest test/benchmarks/test_worker_memory.py -q

Not ``--benchmark-only``: this harness samples memory windows, it does not run pytest-benchmark rounds. The
artifact is written through the shared measurement-suite writer to
``benchmark/mergeable_pca/results/worker_memory.json`` -- this file's own artifact name, not a script
artifact's, so (unlike the smoke benchmark) the real results directory is the right destination.
"""

from __future__ import annotations

import gc
import time
from contextlib import ExitStack
from typing import Any, Callable

import pytest
from conftest import _load_measurement_common

from deisa.dask.mergeable_pca import local_pca, merge_tree

#: Sampling interval of every ``MemorySampler`` series. <=0.1 s per the card; 0.05 s gives ~50+ samples over
#: a hold of :data:`HOLD_S` plus the workload, so a peak is many samples deep, never a single point.
INTERVAL_S = 0.05

#: How long the held payload stays resident inside the sampled window after the workload's last transfer, so
#: the sampler (which reads the scheduler's own accounting) cannot miss it to reporting lag.
HOLD_S = 2.5

#: A window that produced fewer samples than this did not sample the workload; its peak would be one point
#: taken before the payload was resident -- the failure mode the probe flagged -- so the run refuses it.
MIN_SAMPLES_PER_WINDOW = 10

#: Before each window, wait until the scheduler reports at most this much managed memory, so a previous
#: window's released payload cannot leak into the next window's peak.
SETTLE_MANAGED_BYTES = 1024 * 1024

#: Bound on the settle poll; on timeout the run proceeds but stamps ``settled_before_window: false`` on the row.
SETTLE_TIMEOUT_S = 10.0

#: The ``distributed.scheduler.MemoryState`` measures sampled, one ``MemorySampler`` series each.
MEASURES: tuple[str, ...] = ("process", "managed", "unmanaged")

#: The measured grid: (n_block rows, d features, local rank, number of blocks). Small by design (requirement:
#: keep the footprint small); the tall config is the one the deterministic assert runs on. The tall ``n_block``
#: is the card's ``n_block * 50`` on this grid's smallest row count (32 x 50), with the card's d=128, R=8.
CONFIGS: tuple[dict[str, Any], ...] = (
    {"label": "32x32-r4", "n_block": 32, "d": 32, "local_rank": 4, "n_blocks": 4, "tall": False},
    {"label": "256x64-r8", "n_block": 256, "d": 64, "local_rank": 8, "n_blocks": 8, "tall": False},
    {"label": "1600x128-r8", "n_block": 1600, "d": 128, "local_rank": 8, "n_blocks": 8, "tall": True},
)

#: Paths, as the artifact rows name them.
PATHS: tuple[str, ...] = ("legacy", "summary")


def _one_block(config: dict[str, Any], seed: int, index: int) -> Any:
    """Build block ``index`` of a config deterministically, via the shared ``make_block``.

    The same ``(config, index)`` builds the identical array in both paths, so the two paths differ ONLY in
    residency, never in bytes.

    - ``:param config:`` One entry of :data:`CONFIGS`.
    - ``:param seed:`` The config's base seed.
    - ``:param index:`` Block index; the block's seed is ``seed + index``.
    - ``:return:`` The ``(n_block, d)`` float64 block.
    """
    measurement_common = _load_measurement_common()
    return measurement_common.make_block(
        n_block=int(config["n_block"]),
        n_features=int(config["d"]),
        rank=int(config["local_rank"]),
        seed=int(seed) + int(index),
    )


def _trim_allocator() -> bool:
    """Return freed heap to the OS where the platform allows it (glibc ``malloc_trim``).

    Without this, a large freed block may stay in the allocator's arena and inflate every later window's RSS
    baseline -- the legacy window would then contaminate the summary window measured after it.

    - ``:return:`` Whether a trim actually ran (false on non-glibc platforms; the run is honest about it).
    """
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        return bool(libc.malloc_trim(0))
    except Exception:  # noqa: BLE001 - any failure means "no trim available", not a broken run
        return False


def _scheduler_managed_bytes(client: Any) -> int:
    """The scheduler's cluster-wide ``managed_bytes`` right now, summed over workers.

    - ``:param client:`` The connected ``Client``.
    - ``:return:`` Managed bytes (0 when the field is absent, so the settle poll then passes immediately).
    """
    workers = client.scheduler_info()["workers"]
    return int(sum(int(w["metrics"].get("managed_bytes", 0)) for w in workers.values()))


def _wait_memory_settled(client: Any) -> bool:
    """Poll until the scheduler's reported managed memory drops below :data:`SETTLE_MANAGED_BYTES`.

    - ``:param client:`` The connected ``Client``.
    - ``:return:`` True when settled, False on timeout (the caller stamps that on the row).
    """
    deadline = time.monotonic() + SETTLE_TIMEOUT_S
    while time.monotonic() < deadline:
        if _scheduler_managed_bytes(client) <= SETTLE_MANAGED_BYTES:
            return True
        time.sleep(0.2)
    return False


def _scheduler_metrics_mib(client: Any) -> dict[str, float]:
    """The scheduler's end-of-hold per-worker accounting, in MiB, summed over workers.

    Read INSIDE the sampled window while the payload is resident; this is the corroboration channel for the
    ``MemorySampler`` series (same scheduler accounting, one instantaneous read instead of a window peak).

    - ``:param client:`` The connected ``Client``.
    - ``:return:`` ``{"managed": .., "unmanaged": .., "process": ..}`` in MiB.
    """
    totals = {"managed": 0, "unmanaged": 0, "process": 0}
    for worker in client.scheduler_info()["workers"].values():
        metrics = worker["metrics"]
        for key in totals:
            totals[key] += int(metrics.get(f"{key}_bytes" if key != "process" else "memory", 0))
    return {key: value / 2**20 for key, value in totals.items()}


def _sample_window(workload: Callable[[], dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Sample one window: run ``workload`` under one concurrent ``MemorySampler`` series per measure.

    Every series starts before the workload's first allocation and stops after it, so the window covers the
    payload's whole residency. The workload returns what the caller needs to release afterwards (futures,
    payload size) plus the end-of-hold scheduler metrics; the caller keeps those references alive until the
    window has closed, which is what makes the payload resident while sampled.

    - ``:param workload:`` Zero-argument callable run inside the window.
    - ``:return:`` ``(samples, workload_result)`` where ``samples`` maps each measure to its peak bytes,
      sample count and series length.
    """
    from distributed.diagnostics.memory_sampler import MemorySampler

    samplers = {measure: MemorySampler() for measure in MEASURES}
    with ExitStack() as stack:
        for measure, sampler in samplers.items():
            stack.enter_context(sampler.sample(measure, measure=measure, interval=INTERVAL_S))
        result = workload()

    samples: dict[str, Any] = {}
    for measure, sampler in samplers.items():
        series = sampler.to_pandas()[measure].dropna()
        samples[measure] = {
            "peak_bytes": float(series.max()),
            # First sample of the window: sampling starts BEFORE the workload's first allocation, so this is the
            # window's pre-workload baseline and ``peak - baseline`` is the workload-attributable RSS peak.
            "baseline_bytes": float(series.iloc[0]),
            "n_samples": int(series.size),
        }
    return samples, result


def _legacy_workload(client: Any, config: dict[str, Any], seed: int) -> dict[str, Any]:
    """The legacy path's window: allocate ALL blocks co-resident, scatter them, and hold the keys.

    The full ``n_block x d`` chunk is the unit the legacy path ships, so the full chunk is what must be
    resident while sampled. Blocks are built inside the window (their allocation is part of the residency),
    scattered with ``hash=False`` (distinct keys, no dedup), and every future stays referenced until the
    caller releases them after sampling ends.

    - ``:param client:`` The connected ``Client``.
    - ``:param config:`` One entry of :data:`CONFIGS`.
    - ``:param seed:`` The config's base seed.
    - ``:return:`` Futures to cancel, the payload's byte size, and the end-of-hold scheduler metrics.
    """
    n_blocks = int(config["n_blocks"])
    blocks = [_one_block(config, seed, i) for i in range(n_blocks)]
    payload_bytes = sum(int(block.nbytes) for block in blocks)
    futures = client.scatter(blocks, hash=False)
    time.sleep(HOLD_S)
    return {"futures": futures, "payload_bytes": payload_bytes, "metrics": _scheduler_metrics_mib(client)}


def _summary_workload(client: Any, config: dict[str, Any], seed: int) -> dict[str, Any]:
    """The mergeable path's window: summarize each block as it is built, ship the summary, release the block.

    Blocks never co-reside: each is allocated, reduced by :func:`~deisa.dask.mergeable_pca.local_pca` to a
    rank-``local_rank`` summary, the summary is scattered (and held), and the block is dropped before the next
    is built -- the analytics side never holds the chunk. The held summaries are then gathered and reduced by
    :func:`~deisa.dask.mergeable_pca.merge_tree`, the path's analytics-side step.

    - ``:param client:`` The connected ``Client``.
    - ``:param config:`` One entry of :data:`CONFIGS`.
    - ``:param seed:`` The config's base seed.
    - ``:return:`` Futures to cancel, the payload's byte size (summaries only), and the scheduler metrics.
    """
    n_blocks = int(config["n_blocks"])
    futures = []
    summaries = []
    summary_bytes = 0
    for i in range(n_blocks):
        block = _one_block(config, seed, i)
        summary = local_pca(block, rank=int(config["local_rank"]))
        summaries.append(summary)
        summary_bytes += int(summary.components.nbytes + summary.singular_values.nbytes + summary.mean.nbytes)
        futures.append(client.scatter(summary, hash=False))
        del block
    held = client.gather(futures)
    merge_tree(held)
    del held
    payload_bytes = summary_bytes
    time.sleep(HOLD_S)
    return {"futures": futures, "payload_bytes": payload_bytes, "metrics": _scheduler_metrics_mib(client)}


def _release(client: Any, result: dict[str, Any]) -> bool:
    """Release one window's held payload and settle the cluster before the next window.

    Cancels the held futures, drops the references, collects, trims the allocator, then polls the scheduler
    until its reported managed memory is back under :data:`SETTLE_MANAGED_BYTES`.

    - ``:param client:`` The connected ``Client``.
    - ``:param result:`` The workload result carrying the held futures.
    - ``:return:`` Whether the cluster settled below the threshold before :data:`SETTLE_TIMEOUT_S`.
    """
    client.cancel(list(result["futures"]))
    del result["futures"]
    gc.collect()
    _trim_allocator()
    return _wait_memory_settled(client)


def _measure_config_path(client: Any, config: dict[str, Any], seed: int, path: str) -> dict[str, Any]:
    """Measure one (config, path) cell: sampled window, release, and the artifact row.

    - ``:param client:`` The connected ``Client``.
    - ``:param config:`` One entry of :data:`CONFIGS`.
    - ``:param seed:`` The config's base seed.
    - ``:param path:`` ``"legacy"`` or ``"summary"``.
    - ``:return:`` One artifact row in the card's schema, plus the sample counts and corroboration metrics.
    """
    workload = _legacy_workload if path == "legacy" else _summary_workload
    samples, result = _sample_window(lambda: workload(client, config, seed))
    settled = _release(client, result)
    mib = 2**20

    block_bytes = int(config["n_block"]) * int(config["d"]) * 8
    row: dict[str, Any] = {
        "config": config["label"],
        "path": path,
        "n_block": int(config["n_block"]),
        "d": int(config["d"]),
        "local_rank": int(config["local_rank"]),
        "n_blocks": int(config["n_blocks"]),
        "tall": bool(config["tall"]),
        "block_mib": block_bytes / mib,
        "total_payload_mib": block_bytes * int(config["n_blocks"]) / mib,
        "scattered_payload_mib": result["payload_bytes"] / mib,
        "peak_worker_mib": samples["process"]["peak_bytes"] / mib,
        "managed_mib": samples["managed"]["peak_bytes"] / mib,
        "unmanaged_mib": samples["unmanaged"]["peak_bytes"] / mib,
        # Raw peaks measure what the process held; the DELTA over the window's first sample measures what the
        # WORKLOAD added. Both are recorded: the paper quotes the peaks, the regression assert uses the deltas,
        # because the unmanaged baseline drifts upward across windows within one run (allocator/BLAS arena
        # growth) and an assert on raw peaks would flip on that drift, not on the workload.
        "peak_delta_worker_mib": (samples["process"]["peak_bytes"] - samples["process"]["baseline_bytes"]) / mib,
        "peak_delta_managed_mib": (samples["managed"]["peak_bytes"] - samples["managed"]["baseline_bytes"]) / mib,
        "baseline_worker_mib": samples["process"]["baseline_bytes"] / mib,
        "sample_count": samples["process"]["n_samples"],
        "sample_count_managed": samples["managed"]["n_samples"],
        "sample_count_unmanaged": samples["unmanaged"]["n_samples"],
        "scheduler_metrics_mib": result["metrics"],
        "settled_before_window": settled,
    }
    for measure in MEASURES:
        if samples[measure]["n_samples"] < MIN_SAMPLES_PER_WINDOW:
            raise AssertionError(
                f"{config['label']}/{path}: {measure} series produced only "
                f"{samples[measure]['n_samples']} sample(s) over a {HOLD_S}s hold at {INTERVAL_S}s interval; "
                "the peak would be a pre-workload point, not a measurement"
            )
    return row


def _artifact_payload(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    """Build the provenance-stamped artifact payload in the shared measurement-suite shape.

    The ``timing_policy`` is restated as NOT applicable (this artifact measures memory windows, not timed
    rounds) so the stamped default cannot be read as a repeat count this run never used; no row carries a
    timed repeat count, and :func:`measurement_common.write_result`'s gates run unchanged.

    - ``:param rows:`` One row per (config, path).
    - ``:param seed:`` The seed the data fixtures drew from.
    - ``:return:`` The full artifact payload.
    """
    measurement_common = _load_measurement_common()
    trimmed = _trim_allocator()
    return {
        "provenance": measurement_common.provenance(
            script="worker_memory",
            description=(
                "Worker peak memory per scatter path: full-chunk legacy scatter vs mergeable-summary scatter, "
                "sampled by distributed.diagnostics.memory_sampler.MemorySampler (process/managed/unmanaged) "
                "over a held-payload window, on a processes=False LocalCluster."
            ),
            extra={
                "timing_policy": {
                    "clock": "not_applicable (memory artifact: no timed rounds)",
                    "warmup_rounds": 0,
                    "timed_repeats": 0,
                    "statistic": "max over the sampled window",
                    "dispersion": "none (peak statistic); per-row sample_count records how deep the peak is",
                    "note": (
                        "provenance()'s default TIMING_POLICY is overridden here because no timed round ran; a "
                        "repeat count in this artifact would be a claim about a measurement that never happened"
                    ),
                },
                "inputs": {
                    "configs": [
                        {k: cfg[k] for k in ("label", "n_block", "d", "local_rank", "n_blocks", "tall")}
                        for cfg in CONFIGS
                    ],
                    "paths": list(PATHS),
                    "seed": int(seed),
                    "seed_rule": "config base seed = seed; block i seed = base seed + i; "
                    "identical arrays in both paths",
                    "sampler_interval_s": INTERVAL_S,
                    "hold_s": HOLD_S,
                    "min_samples_per_window": MIN_SAMPLES_PER_WINDOW,
                    "harness": "pytest test/benchmarks/test_worker_memory.py",
                },
                "memory_policy": {
                    "sampler": (
                        "distributed.diagnostics.memory_sampler.MemorySampler over the scheduler's "
                        "MemoryState measures -- Dask's own diagnostics, no external polling"
                    ),
                    "measures": list(MEASURES),
                    "peak_worker_mib_definition": "peak of the 'process' series: "
                    "the worker's process memory as the worker reports it",
                    "residency_design": (
                        "the scattered payload is allocated INSIDE the sampled window and held (futures stay "
                        "referenced until sampling ends): legacy holds n_blocks co-resident blocks, summary "
                        "holds one block at a time plus the scattered summaries"
                    ),
                    "zero_copy_note": (
                        "on processes=False LocalCluster the scatter is zero-copy: the held payload is resident "
                        "once and attributed to the worker's keys; a distributed cluster would additionally pay "
                        "one wire+deserialize copy on the legacy path, which this box cannot measure and no row "
                        "claims to"
                    ),
                    "cross_window_guards": {
                        "settle_poll": (
                            "before each window the scheduler's managed_bytes is polled below "
                            f"{SETTLE_MANAGED_BYTES} bytes (timeout {SETTLE_TIMEOUT_S}s); rows stamp "
                            "settled_before_window"
                        ),
                        "allocator_trim_ran": bool(trimmed),
                    },
                },
            },
        ),
        "results": rows,
    }


@pytest.mark.benchmark(group="worker_memory")
def test_worker_peak_memory_grid(bench_cluster, bench_seed):
    """Measure every (config, path) cell, write the artifact, and assert the tall-config inequality.

    The artifact is written BEFORE the deterministic assert, so a failed assert still leaves the measured rows
    on disk for inspection. The assert is the CI regression check: on the tall config the summary path must
    beat the legacy path on the two channels that measure the workload rather than the process's absolute
    state -- the raw ``managed`` peak (the worker's own key accounting, structurally the payload size) and the
    ``process`` peak delta over the window's pre-workload baseline (raw process peaks are recorded in the
    artifact but not asserted: the unmanaged baseline drifts upward across windows, and an assert on raw peaks
    would decide on that drift, not on the workload).

    - ``:param bench_cluster:`` The session-scoped in-process ``Client``.
    - ``:param bench_seed:`` The suite seed every fixture draws from.
    """
    client = bench_cluster
    rows: list[dict[str, Any]] = []
    for config_index, config in enumerate(CONFIGS):
        # A per-config base seed keeps every config's arrays deterministic and independent of run order; the
        # SAME (config, index) builds the identical array in both paths, so only residency differs.
        seed = int(bench_seed) + 1000 * config_index
        for path in PATHS:
            rows.append(_measure_config_path(client, config, seed, path))
            print(
                f"  {config['label']:>12} {path:>7}: peak={rows[-1]['peak_worker_mib']:8.2f} MiB "
                f"delta={rows[-1]['peak_delta_worker_mib']:7.2f} managed={rows[-1]['managed_mib']:8.2f} "
                f"unmanaged={rows[-1]['unmanaged_mib']:8.2f} MiB n={rows[-1]['sample_count']}",
                flush=True,
            )

    measurement_common = _load_measurement_common()
    payload = _artifact_payload(rows, bench_seed)
    artifact = measurement_common.write_result("worker_memory", payload)
    assert artifact.exists(), f"artifact not written at {artifact}"
    reread = measurement_common.json.loads(artifact.read_text(encoding="utf-8"))
    assert reread["provenance"]["script"] == "worker_memory"
    assert reread["provenance"]["deisa_dask_commit"]
    assert len(reread["results"]) == len(CONFIGS) * len(PATHS)

    tall = next(config for config in CONFIGS if config["tall"])
    legacy = next(r for r in rows if r["config"] == tall["label"] and r["path"] == "legacy")
    summary = next(r for r in rows if r["config"] == tall["label"] and r["path"] == "summary")

    # The measurement captured the legacy payload at all: the worker's managed accounting must cover the
    # scattered chunk (zero-copy keys carry their true sizeof), else the window sampled nothing.
    assert legacy["managed_mib"] >= 0.8 * legacy["scattered_payload_mib"], (
        f"legacy managed peak {legacy['managed_mib']:.2f} MiB covers less than 80% of the scattered payload "
        f"{legacy['scattered_payload_mib']:.2f} MiB: the window did not capture the held chunk"
    )
    # The CI regression check, on channels that measure the WORKLOAD rather than the process's absolute state:
    # - managed is the worker's own key accounting, structurally the payload size and independent of the
    #   unmanaged baseline, so its raw peak is already baseline-free;
    # - the process delta is the raw peak minus the window's first sample (taken before the first allocation),
    #   so allocator/BLAS baseline drift between windows cancels instead of deciding the assert.
    # The raw process peaks are recorded in the artifact for the paper but are NOT asserted: the summary
    # window's unmanaged peak can exceed the legacy window's without either workload holding more (a probe on
    # this box measured the summary window's unmanaged peak ABOVE legacy's while its managed accounting was
    # three orders of magnitude smaller).
    assert summary["managed_mib"] < legacy["managed_mib"], (
        f"tall config {tall['label']}: summary managed peak {summary['managed_mib']:.3f} MiB is not below "
        f"legacy managed peak {legacy['managed_mib']:.3f} MiB"
    )
    assert summary["peak_delta_worker_mib"] < legacy["peak_delta_worker_mib"], (
        f"tall config {tall['label']}: summary workload-attributable RSS peak "
        f"{summary['peak_delta_worker_mib']:.2f} MiB "
        f"is not below legacy {legacy['peak_delta_worker_mib']:.2f} MiB "
        f"(summary holds one {summary['block_mib']:.2f} MiB block at a time, legacy holds "
        f"{legacy['total_payload_mib']:.2f} MiB co-resident)"
    )
