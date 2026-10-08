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
Time-to-first-result benchmark: the per-step latency the design trades for the transfer reduction.

Ported from ``benchmark/mergeable_pca/time_to_first_result.py`` (the old self-timed script) onto pytest-benchmark.
What the "time to FIRST result" is here: the bridge does its per-step work -- summarize the chunk locally, merge
the summaries -- and the first thing a downstream callback can consume is that merged pipeline result, so the
per-step latency the artifact exposes is the sum of the measured local SVD and merge-tree medians, against the
legacy arm of the same trade (prepare the full chunks + the transfer both designs pay, estimated from measured
wire bytes at an assumed link rate). The old script timed each stage with its own ``time_repeated`` loop; that
timing layer is replaced by :func:`benchmark.pedantic` and its stats, while the measuring semantics (what is
inside each timed region, what is prepared outside it) are the script's.

One ``benchmark`` fixture measures one thing, so the old script's three ``time_repeated`` stages become three
benchmark tests over the same parametrized configurations; the pipeline latency is the SUM of measured medians
of the two stages that make it, exactly as the script computed ``pca_compute_s = local_svd_s + merge_s`` -- a sum
of medians, never a separately-timed blend that would double-count suite noise.

The two sibling artifacts
--------------------------
The legacy copy of the old script's artifact onto this harness keeps::

- ``pca_bytes`` / ``legacy_bytes`` / ``compression_ratio_legacy_vs_pca_wire`` -- measured through the same
  serializer the bridge uses, so the crossover comparison between the two harnesses runs over the same wire.
- ``crossover_mbps``: the link rate at which the byte saving exactly pays for the extra local compute. It is
  machine-independent: the script is in-process, this benchmark is in-process plus a scatter of the summaries,
  so WHERE each run sits relative to the crossover decides which one wins.

Warmup
-------
Kept at :data:`WARMUP_ROUNDS` rounds -- one discarded warmup round before timing, exactly what the old script
asked ``time_repeated`` for. During those untimed rounds the underlying data contract is exercised on
serialization and merge; ``provenance.inputs.warmup_rounds`` records it.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from conftest import _load_measurement_common

from deisa.dask.mergeable_pca import local_pca, merge_tree

#: Blocks the local rank summarizes per configuration -- one bridge per block, so ``n_block`` leaves per tree.
N_BLOCKS = 8

#: Rows per leaf block: the card's small config -- ``n_block <= 8`` rows per leaf and the tree carries
#: :data:`N_BLOCKS` leaves, so the whole pipeline stays a few milliseconds per step.
N_BLOCKS_LEAF_ROWS = 8

#: Feature dimensions swept. The card keeps the sweep small (n_block<=8, d<=64): each configuration is timed
#: through the three stages of the design, and the artifact crosses the same gates as the script's.
FEATURE_DIMS: tuple[int, ...] = (32, 64)

#: Local ranks timed: the paper's rank-truncation ceiling in the old script (8), a truncation below it (4), and
#: the full local rank (``None``, meaning ``min(n_block, d)``) -- the SVD cost is what truncation buys down.
LOCAL_RANKS: tuple[int | None, ...] = (4, 8, None)

#: Assumed link rate, megabits per second. Recorded in the artifacts verbatim -- an input, never a silent
#: constant -- exactly as the old script's ``DEFAULT_MBPS`` did.
DEFAULT_MBPS = 1000.0

#: Warmup rounds discarded before timing. The old script's policy kept one warmup round minimum; the artifact's
#: provenance states it explicitly so the reader can compare against the script's ``warmup_rounds = 1``.
WARMUP_ROUNDS = 1

#: Timed rounds per configuration per stage. The old script timed three repeats through ``time_repeated``;
#: pytest-benchmark adds real dispersion behind the same median. Five rounds keep the module under the five
#: minute budget: the artifacts then carry genuine iqr/stddev, and the repeat-count gate holds.
ROUNDS = 5

#: Intrinsic rank of the synthetic signal, as a fraction of ``d`` -- the value the script experiments used.
INTRINSIC_RANK_FRACTION = 4.0

#: Module-level row accumulators, keyed per stage. pytest-benchmark's fixture is single-use and the sweep is
#: three parametrized tests, so rows accumulate across invocations; each test rewrites the artifact for its
#: own stage after every point, so the LAST write of one artifact carries that stage's full sweep. This mirrors
#: the accumulator pattern the block-scaling benchmark uses.
_ROWS_SVD: list[dict[str, Any]] = []
_ROWS_MERGE: list[dict[str, Any]] = []
_ROWS_PIPELINE: list[dict[str, Any]] = []
_ROWS_CROSSOVER: list[dict[str, Any]] = []

#: Module-level cache of the per-configuration measurement context. Built ONCE per (n_block, d, local_rank) by
#: the first stage to run at that point, reused by every later stage/round. Holds the fully-built blocks
#: (untimed), the leaf summaries per rank, the serialized legacy and summary payload sizes, and the summary
#: sizes. No timing lives here, only data.
_POLICY_CACHE: dict[tuple[int, int, int | None, float], dict[str, Any]] = {}

#: The sweep's fixed inputs, stamped verbatim into the artifact's provenance.
_SWEEP_INPUTS: dict[str, Any] = {
    "n_blocks_per_config": N_BLOCKS,
    "feature_dims": list(FEATURE_DIMS),
    "local_ranks": [None if r is None else int(r) for r in LOCAL_RANKS],
    "intrinsic_rank_fraction_of_d": 1.0 / INTRINSIC_RANK_FRACTION,
    "assumed_link_mbps": DEFAULT_MBPS,
    "timed_repeats": ROUNDS,
    "warmup_rounds": WARMUP_ROUNDS,
    "harness": "pytest-benchmark",
}


def _config_key(n_block_per_leaf: int, d: int, local_rank: int | None) -> tuple[int, int, int | None, float]:
    """Key into the per-configuration cache; the assumed Mbps is part of it because it feeds the transfer estimate.

    - ``:param n_block_per_leaf:`` Rows per leaf block.
    - ``:param d:`` Feature dimension.
    - ``:param local_rank:`` Retained local rank or ``None`` for full local rank.
    """
    return (int(n_block_per_leaf), int(d), local_rank, float(DEFAULT_MBPS))


def _regime_of(n_rows: int, d: int) -> str:
    """Classify ``(rows, features)`` as tall / square / flat, the sweep's own regime labels renamed, in ONE place.

    Duplicates the legacy ``regime_of`` rule so the artifact's words match the sweep's dictionary key -- which
    is exactly how the script's ``regime_of`` reads too.

    - ``:param n_rows:`` Rows in the block.
    - ``:param d:`` Feature dimension.
    """
    if n_rows > d:
        return "tall"
    if n_rows < d:
        return "flat"
    return "square"


def _context(n_block_per_leaf: int, d: int, local_rank: int | None) -> dict[str, Any]:
    """Build (or reuse) the per-configuration measurement context: blocks at the correct shapes plus every
    serialized size except the assumed-rate-dependent transfer estimates, which are derived where they are
    needed. Timed stages never build *blocks* inside the timed region -- the sweep is over the three stages,
    not over data construction.

    The blocks are drawn from the legacy master seed, at the legacy stream ids the script used
    (``SEED + 41 + 100 * i``), so a row here and a script row at the same configuration describe the SAME data.

    - ``:param n_block_per_leaf:`` Rows per leaf block.
    - ``:param d:`` Feature dimension.
    - ``:param local_rank:`` Retained local rank, or ``None`` for the full local rank ``min(n_block, d)``.
    - ``:return:`` The context dict with the blocks, derived sizes and transfer-time estimates at the sweep's
      assumed rate. Includes both leaves (rank-truncated) and their wire sizes plus the merged tree result.
    """
    key = _config_key(n_block_per_leaf, d, local_rank)
    cached = _POLICY_CACHE.get(key)
    if cached is not None:
        return cached

    measurement_common = _load_measurement_common()
    seed = measurement_common.SEED
    intrinsic = max(1, d // int(INTRINSIC_RANK_FRACTION))
    blocks = [
        measurement_common.make_block(n_block=n_block_per_leaf, n_features=d, rank=intrinsic, seed=seed + 41 + 100 * i)
        for i in range(N_BLOCKS)
    ]

    rank_arg = None if local_rank is None else min(int(local_rank), d)
    summaries = [local_pca(block, rank=rank_arg) for block in blocks]
    legacy_payload = blocks[0]
    legacy_wire = measurement_common.serialized_nbytes(legacy_payload) * N_BLOCKS
    pca_wire = sum(measurement_common.serialized_nbytes(summary) for summary in summaries)

    context: dict[str, Any] = {
        "blocks": blocks,
        "summaries": summaries,
        "legacy_payload": legacy_payload,
        "legacy_wire_bytes": float(legacy_wire),
        "pca_wire_bytes": float(pca_wire),
        "local_rank_requested": local_rank,
        "local_rank_effective": int(summaries[0].rank),
        "intrinsic_rank": int(intrinsic),
        "regime": _regime_of(n_block_per_leaf, d),
    }
    _POLICY_CACHE[key] = context
    return context


def _transfer_seconds(wire_bytes: float) -> float:
    """Bandwidth-scaled transfer estimate: measured wire bytes divided by the assumed rate.

    Repeated verbatim from the script, so both artifacts derive the transfer term the same way: the network is
    never measured on this box, the estimate is, and the assumed rate is an input recorded in every artifact.

    - ``:param wire_bytes:`` The measured, serialized payload size the bridge or the legacy path would cross.
    """
    bytes_per_second = DEFAULT_MBPS * 1e6 / 8.0
    return wire_bytes / bytes_per_second


def _pipeline_row(
    n_block_per_leaf: int,
    d: int,
    local_rank: int | None,
    context: dict[str, Any],
    summary_s: float | None,
    merge_s: float | None,
    legacy_prepare_s: float | None,
) -> dict[str, Any]:
    """Walk a stage's row into the artifact format for the per-step pipeline latency.

    Every artifact row carries the per-configuration identity, the pipeline latency assembled from the measured
    medians according to the stage, and the transfer estimates both designs pay individually and together --
    exactly which rows the script's ``_print`` table and the crossover section read, computed from the measured
    medians the pytest-benchmark harness produced.

    - ``:param n_block_per_leaf:`` Rows per leaf block.
    - ``:param d:`` Feature dimension.
    - ``:param local_rank:`` Retained local rank requested.
    - ``:param context:`` The per-configuration context: wire sizes and payload shapes.
    - ``:param summary_s:`` The local-SVD stage's median, or ``None`` when that stage has not produced one at
      this point yet; assembled together below.
    - ``:param merge_s:`` The merge-tree stage's median, or ``None``.
    - ``:param legacy_prepare_s:`` The legacy-prepare stage's median, or ``None``.
    - ``:return:`` The artifact row, with whatever latency members are available so far at this stage.
    """
    legacy_wire = context["legacy_wire_bytes"]
    pca_wire = context["pca_wire_bytes"]
    legacy_transfer_s = _transfer_seconds(legacy_wire)
    pca_transfer_s = _transfer_seconds(pca_wire)
    legacy_prepare = legacy_prepare_s
    row: dict[str, Any] = {
        "n_block": int(n_block_per_leaf),
        "n_features": int(d),
        "n_blocks": N_BLOCKS,
        "n_block_over_d": float(n_block_per_leaf) / float(d),
        "regime": context["regime"],
        "local_rank_requested": local_rank,
        "local_rank_effective": context["local_rank_effective"],
        "intrinsic_rank": context["intrinsic_rank"],
        "legacy_wire_bytes": context["legacy_wire_bytes"],
        "pca_wire_bytes": context["pca_wire_bytes"],
        "compression_ratio_legacy_vs_pca_wire": (None if pca_wire == 0 else float(legacy_wire / pca_wire)),
        "legacy_transfer_seconds_estimate": legacy_transfer_s,
        "pca_transfer_seconds_estimate": pca_transfer_s,
        "local_svd_seconds_median": summary_s,
        "merge_tree_seconds_median": merge_s,
        "legacy_prepare_seconds_median": legacy_prepare,
    }
    # The pipeline latency is the sum of measured medians of the stages the pipeline consists of. Where one
    # stage's median has not been measured at this point (a partial sweep write), the fraction that has IS
    # recorded rather than silently dropped -- the artifact never shows a number for a stage it did not run.
    measured = [value for value in (summary_s, merge_s) if value is not None]
    row["pca_pipeline_seconds_estimate"] = float(sum(measured)) if measured else None
    if legacy_prepare is None:
        row["legacy_total_seconds_estimate"] = None
        row["pca_faster_than_legacy"] = None
        row["speedup_estimate"] = None
    else:
        legacy_total = float(legacy_prepare) + legacy_transfer_s
        row["legacy_total_seconds_estimate"] = legacy_total
        if not measured:
            # Without a measured pipeline, no stage of the trade is attributable: the verdict is explicitly
            # unknown rather than an implied zero.
            row["pca_faster_than_legacy"] = None
            row["speedup_estimate"] = None
        else:
            pipeline = row["pca_pipeline_seconds_estimate"]
            assert pipeline is not None
            row["pca_faster_than_legacy"] = bool(pipeline < legacy_total)
            row["speedup_estimate"] = None if pipeline == 0 else float(legacy_total / pipeline)
    return row


def _result_payload(script: str, description: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble the provenance-stamped artifact payload for one stage's rows.

    The artifact goes through :func:`measurement_common.write_result`, so the sign-invariant gate and the
    repeat-count gate run on it exactly as they did on the script's artifact -- repeated here because the
    three-stage accumulator pattern (one artifact per stage) files the same stage's rows in one place, unlike
    the script's single artifact carrying every stage.

    - ``:param script:`` Artifact stem.
    - ``:param description:`` One line stating what the artifact measures.
    - ``:param rows:`` That stage's measured rows, each already carrying its medians.
    """
    measurement_common = _load_measurement_common()
    wins = [row for row in rows if row.get("pca_faster_than_legacy") is True]
    losses = [row for row in rows if row.get("pca_faster_than_legacy") is False]
    unknown = [row for row in rows if row.get("pca_faster_than_legacy") is None]
    return {
        "provenance": measurement_common.provenance(
            script=script,
            description=description,
            extra={
                "inputs": {
                    **_SWEEP_INPUTS,
                    "timed_repeats": ROUNDS,
                    "warmup_rounds": WARMUP_ROUNDS,
                    "iterations_per_round": 1,
                    "statistic": "median",
                    "clock": "pytest-benchmark timer (timeit.default_timer unless overridden)",
                    "configurations_measured": len(rows),
                },
                "timing_policy": {
                    "clock": "pytest-benchmark timer (timeit.default_timer unless overridden)",
                    "warmup_rounds": WARMUP_ROUNDS,
                    "timed_repeats": ROUNDS,
                    "iterations_per_round": 1,
                    "statistic": "median",
                    "dispersion": "min/max/iqr/stddev reported alongside the median",
                    "note": "measured by pytest-benchmark; the warmup rounds are its own, kept at the old "
                    "script's warmup policy of one discarded round",
                    "source": "pytest-benchmark",
                },
            },
        ),
        "verdict": {
            "configurations_measured": len(rows),
            "pca_faster": len(wins),
            "pca_slower": len(losses),
            "attribution_unknown": len(unknown),
            "honest_summary": "the design is a TRADE, not a free win: where pca_slower is non-zero the local SVD "
            "cost exceeds the byte saving at the assumed link rate and those rows are kept, not dropped. Rows "
            "whose timing stages have not all run yet carry attribution_unknown rather than an implied verdict.",
        },
        "results": rows,
    }


def _write_stage_artifact(script: str, description: str, rows: list[dict[str, Any]]) -> None:
    """Write one stage's rows through the shared artifact writer and assert the artifact round-trips.

    Sub-calls through :func:`measurement_common.write_result` -- the same writer the scripts and the other
    benchmark modules use, so the gates and the provenance stamp are one authority, not three.

    - ``:param script:`` Artifact stem, e.g. ``"time_to_first_result_local_svd"``.
    - ``:param description:`` What the artifact measures, one line.
    - ``:param rows:`` The stage's rows.
    """

    measurement_common = _load_measurement_common()
    path = measurement_common.write_result(script, _result_payload(script, description, rows))
    with path.open(encoding="utf-8") as fp:
        stored = json.load(fp)
    assert stored["provenance"]["script"] == script
    assert len(stored["results"]) == len(rows)
    assert measurement_common.enforce_sign_invariant_results(stored) is None


# ==============================================================================================================
# Stage 1: the local SVD the bridge runs per step
# ==============================================================================================================
@pytest.mark.benchmark(group="time_to_first_result", min_rounds=ROUNDS, warmup=WARMUP_ROUNDS)
@pytest.mark.parametrize("d", FEATURE_DIMS)
@pytest.mark.parametrize("local_rank", LOCAL_RANKS)
def test_bench_local_svd_stage(benchmark, local_rank: int | None, d: int) -> None:
    """Benchmark the bridge-side local ``local_pca`` stage for one configuration.

    This is the cost the design introduces and the legacy path never pays: per-step summarize of every block
    into a mergeable summary. The blocks are built OUTSIDE the timed region by the shared context so the
    timed region is exactly ``[local_pca(b, rank) for b in blocks]`` -- the same list comprehension the script
    timed through ``time_repeated``.

    - ``:param benchmark:`` The pytest-benchmark fixture.
    - ``:param local_rank:`` Retained local rank, ``None`` handled by the sweep: the module docstring's
      truncation trade (truncated smaller, full exact).
    - ``:param d:`` Feature dimension.
    """
    context = _context(N_BLOCKS_LEAF_ROWS, d, local_rank)
    blocks = context["blocks"]
    rank_arg = None if local_rank is None else min(int(local_rank), d)

    def summarize_once() -> list[Any]:
        return [local_pca(block, rank=rank_arg) for block in blocks]

    summaries = benchmark.pedantic(summarize_once, rounds=ROUNDS, warmup_rounds=WARMUP_ROUNDS)
    assert len(summaries) == N_BLOCKS, "the timed region must summarize all blocks"
    assert summaries[0].rank == context["local_rank_effective"]

    stats = benchmark.stats
    samples = [float(value) for value in stats.stats.data]
    assert stats.get("rounds") == ROUNDS == len(samples)
    _ROWS_SVD.append(
        _pipeline_row(
            N_BLOCKS_LEAF_ROWS,
            d,
            local_rank,
            context,
            summary_s=float(stats.get("median")),
            merge_s=None,
            legacy_prepare_s=None,
        )
    )
    _ROWS_SVD[-1]["seconds_all"] = samples
    _ROWS_SVD[-1]["timed_repeats"] = ROUNDS
    _write_stage_artifact(
        "time_to_first_result_local_svd",
        "Bridge-side local SVD per step (list of local_pca over all blocks), one artifact per configuration.",
        list(_ROWS_SVD),
    )


# ==============================================================================================================
# Stage 2: the merge tree, the first thing a downstream callback can consume
# ==============================================================================================================
@pytest.mark.benchmark(group="time_to_first_result", min_rounds=ROUNDS, warmup=WARMUP_ROUNDS)
@pytest.mark.parametrize("d", FEATURE_DIMS)
@pytest.mark.parametrize("local_rank", LOCAL_RANKS)
def test_bench_merge_tree_stage(benchmark, local_rank: int | None, d: int) -> None:
    """Benchmark the merge-tree reduction over the SAME leaves stage 1 summarizes -- the callback's entry point.

    A merge tree over ``N_BLOCKS`` leaves: the O(B * d^3) reduction, independent of the sample count, whose
    output is the pipeline result a downstream callback consumes first. The leaves come from the SAME
    ``.summaries`` list the context built per configuration, so stage 1 and this stage measure the same data.

    - ``:param benchmark:`` The pytest-benchmark fixture.
    - ``:param local_rank:`` Retained local rank requested.
    - ``:param d:`` Feature dimension.
    """
    context = _context(N_BLOCKS_LEAF_ROWS, d, local_rank)
    summaries = context["summaries"]

    merged = benchmark.pedantic(merge_tree, args=(summaries,), rounds=ROUNDS, warmup_rounds=WARMUP_ROUNDS)

    # The tree really reduced: one root over the pooled sample count, no truncation past the saturation ceiling.
    assert merged.n_samples == N_BLOCKS * N_BLOCKS_LEAF_ROWS
    assert merged.rank <= d

    stats = benchmark.stats
    samples = [float(value) for value in stats.stats.data]
    assert stats.get("rounds") == ROUNDS == len(samples)
    _ROWS_MERGE.append(
        {
            "n_block": N_BLOCKS_LEAF_ROWS,
            "n_features": int(d),
            "n_blocks": N_BLOCKS,
            "n_block_over_d": float(N_BLOCKS_LEAF_ROWS) / float(d),
            "regime": context["regime"],
            "local_rank_requested": local_rank,
            "local_rank_effective": context["local_rank_effective"],
            "intrinsic_rank": context["intrinsic_rank"],
            "merge_tree_seconds_median": float(stats.get("median")),
            "seconds_all": samples,
            "timed_repeats": ROUNDS,
        }
    )
    _write_stage_artifact(
        "time_to_first_result_merge_tree",
        "Merge-tree reduction per step over prebuilt leaf summaries -- the stage whose output the first "
        "downstream callback consumes.",
        list(_ROWS_MERGE),
    )


# ==============================================================================================================
# Stage 3: the crossover row and the full pipeline vs legacy
# ==============================================================================================================
@pytest.mark.benchmark(group="time_to_first_result", min_rounds=ROUNDS, warmup=WARMUP_ROUNDS)
@pytest.mark.parametrize("d", FEATURE_DIMS)
@pytest.mark.parametrize("local_rank", LOCAL_RANKS)
def test_bench_legacy_prepare_stage(benchmark, local_rank: int | None, d: int) -> None:
    """Benchmark the legacy path's per-step compute: serialize the full chunks it would have scattered.

    The legacy scatter ships full chunks, so its per-step compute at the bridge is the serialization prep the
    design's analysis charged it (``serialized_nbytes`` per block, repeated). Nothing else: the legacy path
    does NOT pay an SVD, and the script measured its prepare step exactly this way. The payload does not
    depend on the rank (the chunk is the same whatever the PCA arm truncates to), so the timed region is the
    script's ``serialized_nbytes(legacy_payload)`` -- multiplied by the bridge count exactly where the script
    did, in the artifact row.

    - ``:param benchmark:`` The pytest-benchmark fixture.
    - ``:param local_rank:`` Kept parametrized so this stage's row keys line up with the summary stages' row
      keys (the trade compares them per configuration); do not ``del`` it, the row identity uses it.
    - ``:param d:`` Feature dimension.
    """
    context = _context(N_BLOCKS_LEAF_ROWS, d, None)
    legacy_payload = context["legacy_payload"]

    serialized = benchmark.pedantic(
        _load_measurement_common().serialized_nbytes, args=(legacy_payload,), rounds=ROUNDS, warmup_rounds=WARMUP_ROUNDS
    )
    assert isinstance(serialized, int), "legacy prepare is a serialization measurement, output count in bytes"

    stats = benchmark.stats
    samples = [float(value) for value in stats.stats.data]
    assert stats.get("rounds") == ROUNDS == len(samples)
    legacy_prepare = float(stats.get("median")) * N_BLOCKS

    row = _pipeline_row(
        N_BLOCKS_LEAF_ROWS,
        d,
        local_rank,
        context,
        summary_s=None,
        merge_s=None,
        legacy_prepare_s=legacy_prepare,
    )
    row["seconds_all"] = samples
    row["timed_repeats"] = ROUNDS
    _ROWS_CROSSOVER.append(row)
    _write_stage_artifact(
        "time_to_first_result_crossover",
        "Legacy full-chunk scatter prepare per step: the serialization the design replaces, measured against "
        "the same serializer.",
        list(_ROWS_CROSSOVER),
    )


def _merge_first_stage(d: int, local_rank: int | None) -> Any:
    """Build the callback's first consumable result at one configuration, over the pre-built leaves.

    Cross-reading helper between stage artifacts: the pipeline tests run their own pipeline per round; this
    builds the SAME merged result without touching any timing. Used only where a test needs the merged root
    outside a timed region, so the merge semantics stay in ONE call.

    - ``:param d:`` Feature dimension.
    - ``:param local_rank:`` Rank request the merged summary is built from.
    - ``:return:`` The merged summary, ranked ``min(n_block, d)`` at full local rank.
    """
    return merge_tree(_context(N_BLOCKS_LEAF_ROWS, d, local_rank)["summaries"])


# ==============================================================================================================
# The full pipeline, timed end to end: what a callback waits for per step
# ==============================================================================================================
@pytest.mark.benchmark(group="time_to_first_result", min_rounds=ROUNDS, warmup=WARMUP_ROUNDS)
@pytest.mark.parametrize("d", FEATURE_DIMS)
@pytest.mark.parametrize("local_rank", LOCAL_RANKS)
def test_bench_pipeline_latency(benchmark, local_rank: int | None, d: int) -> None:
    """Benchmark the whole per-step pipeline: local SVD of every block, then the merge tree.

    One timed round runs ``[local_pca(block) for blocks] + merge_tree(summaries)`` in ONE measurement, so the
    artifact carries the latency a downstream callback actually experiences per step against the transfer the
    design REDUCED. This is the number the crossover row reads and the single honest "time to first result";
    the stage tests remain because attribution -- SVD vs merge vs legacy prepare -- is the point of the card.

    - ``:param benchmark:`` The pytest-benchmark fixture.
    - ``:param local_rank:`` Retained local rank requested.
    - ``:param d:`` Feature dimension.
    """
    context = _context(N_BLOCKS_LEAF_ROWS, d, local_rank)
    blocks = context["blocks"]
    rank_arg = None if local_rank is None else min(int(local_rank), d)

    def pipeline_once() -> Any:
        summaries = [local_pca(block, rank=rank_arg) for block in blocks]
        return merge_tree(summaries)

    merged = benchmark.pedantic(pipeline_once, rounds=ROUNDS, warmup_rounds=WARMUP_ROUNDS)
    assert merged.n_samples == N_BLOCKS * N_BLOCKS_LEAF_ROWS

    stats = benchmark.stats
    samples = [float(value) for value in stats.stats.data]
    assert stats.get("rounds") == ROUNDS == len(samples)
    measured_pipeline = float(stats.get("median"))

    # The legacy arm is NEVER separately timed in this benchmark: the script timed it and its per-step
    # estimate derives from measured bytes at the assumed rate, which the crossover artifact carries.
    pipeline_row = {
        "n_block": N_BLOCKS_LEAF_ROWS,
        "n_features": int(d),
        "n_blocks": N_BLOCKS,
        "n_block_over_d": float(N_BLOCKS_LEAF_ROWS) / float(d),
        "regime": context["regime"],
        "local_rank_requested": local_rank,
        "local_rank_effective": context["local_rank_effective"],
        "intrinsic_rank": context["intrinsic_rank"],
        "pipeline_seconds_median": measured_pipeline,
        "pca_transfer_seconds_estimate": _transfer_seconds(context["pca_wire_bytes"]),
        "pca_total_seconds_estimate": measured_pipeline + _transfer_seconds(context["pca_wire_bytes"]),
        "legacy_total_seconds_estimate": (_legacy_arm_estimate(d, local_rank) if _ROWS_CROSSOVER else None),
        "speedup_estimate": None,
        "seconds_all": samples,
        "timed_repeats": ROUNDS,
    }
    if pipeline_row["legacy_total_seconds_estimate"] is not None:
        pipeline_row["speedup_estimate"] = (
            None if measured_pipeline == 0 else (pipeline_row["legacy_total_seconds_estimate"] / measured_pipeline)
        )
    _ROWS_PIPELINE.append(pipeline_row)

    # The pipeline measured END TO END per step -- the "time to first result" the metric means. When the
    # crossover rows carry a legacy prepare median, the speedup against the legacy arm is stated per row.
    crossover_rows = _crossover_artifact_rows(pipeline_row)
    _ROWS_PIPELINE[-1].update(crossover_rows)
    _write_stage_artifact(
        "time_to_first_result_pipeline",
        "Whole-per-step-pipeline latency: local SVD of all blocks plus the merge tree, the latency a "
        "downstream callback waits through per output step.",
        list(_ROWS_PIPELINE),
    )


# ==============================================================================================================
# Crossover rows / revisit: the crossover_mbps the old script derives
# ==============================================================================================================
def _legacy_arm_estimate(d: int, local_rank: int | None) -> float | None:
    """Legacy arm of the trade: prepare (median, from the crossover stage rows) + transfer at the assumed rate.

    Reads the crossover stage's accumulated rows; ``None`` when that stage has not run at this configuration
    yet. No separate timing is imposed here -- the crossover stage owns the legacy measurement.

    - ``:param d:`` Feature dimension.
    - ``:param local_rank:`` Rank request the row reads.
    - ``:return:`` The legacy arm's seconds estimate, or ``None``.
    """
    for row in _ROWS_CROSSOVER:
        if (
            row["n_features"] == int(d)
            and row["local_rank_requested"] == local_rank
            and row["n_block"] == N_BLOCKS_LEAF_ROWS
        ):
            return row["legacy_total_seconds_estimate"]
    return None


def _crossover_artifact_rows(pipeline_row: dict[str, Any]) -> dict[str, Any]:
    """Derive the crossover block for one pipeline row, in the script's definition.

    Solving ``pipeline + pca_wire / r == legacy_total + legacy_wire / r`` for ``r``, per configuration, equals
    the script's crossover: above it the byte saving dominates; below, the local SVD does. The equation needs
    the legacy arm at the SAME configuration; without it the crossover is honestly ``None`` rather than
    silently absent.

    - ``:param pipeline_row:`` The row the crossover is derived from.
    - ``:return:`` The keys ``crossover_mbps``, ``bytes_saved``, ``crossover_derivation`` for the row.
    """
    n_block = pipeline_row["n_block"]
    n_features = pipeline_row["n_features"]
    local_rank_requested = pipeline_row["local_rank_requested"]
    assert isinstance(n_block, int) and isinstance(n_features, int)
    context = _context(n_block, n_features, local_rank_requested)
    legacy_total = pipeline_row.get("legacy_total_seconds_estimate")
    if legacy_total is None:
        return {"crossover_mbps": None, "bytes_saved": None, "crossover_derivation": "legacy arm not yet measured"}
    legacy_total_s = float(legacy_total)
    saved = float(context["legacy_wire_bytes"]) - float(context["pca_wire_bytes"])
    legacy_transfer_s = _transfer_seconds(float(context["legacy_wire_bytes"]))
    extra_compute = float(pipeline_row["pipeline_seconds_median"]) - (legacy_total_s - legacy_transfer_s)
    rate_bps = (saved / extra_compute) if extra_compute > 0 else None
    return {
        "crossover_mbps": None if rate_bps is None else rate_bps * 8.0 / 1e6,
        "bytes_saved": saved,
        "crossover_derivation": "pipeline + pca_wire/r vs legacy_total + legacy_wire/r; measured bytes, "
        "assumed rate implicit in the legacy arm's transfer term",
    }
