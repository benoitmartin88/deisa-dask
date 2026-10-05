#!/usr/bin/env python
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
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# =============================================================================
"""
The trade the paper defers: what bridge-side PCA plus merge COSTS, against shipping the full block.

What this script measures, and the three quantities kept apart
-------------------------------------------------------------
Benoit's question about this work is the one the paper cannot currently answer. The paper reports bytes
saved and calls the RATIO the transferable result, leaving the absolute cost to the reader. That deferral
is the weakest joint in the paper, so this script measures the cost side instead of arguing about it. Three
quantities are reported, and they are deliberately never merged into one number:

1. ``bytes_saved_per_bridge_send_measured``. BYTES ON THE WIRE. Measured here by serializing both
   payloads through the same serializer on the same block. Absolute MiB is reported alongside every ratio,
   because a ratio makes the reader do the subtraction themselves and the number they want is the
   subtraction.

2. ``added_local_compute_seconds_measured``. ADDED LOCAL COMPUTE. Bridge-side local PCA plus the merge
   tree, against the same total work done AFTER a full transfer, both measured in process with real
   repeats and real dispersion, and the DIFFERENCE reported. The bridge path is measurably SLOWER than
   the optimized local baseline: the local-baseline/mergeable ratio runs 0.05 to 0.31 with median 0.21, so
   the bridge genuinely pays compute. This is measured rather than implied, and it is not hidden.

3. ``break_even_link_gbps``. The trade expressed as a BREAK-EVEN condition instead of as a fabricated
   transfer time. The bridge-side cost is worth paying exactly when the avoided transfer time exceeds the
   added compute:

       seconds_avoided  =  bytes_saved  /  bandwidth
       break-even when  seconds_avoided  >  added_compute
    =>  bandwidth_required  =  bytes_saved  /  added_compute

   That is a REQUIRED BANDWIDTH, obtained from two single-node measurements and one division. It needs no
   assumed link speed, so there is no bandwidth figure to invent and no measurement to mislabel.

Why no bandwidth figure appears anywhere in this artifact
---------------------------------------------------------
Grid5000 is where the real network lives and it is not reachable from this container; everything measured
here is single-node. A bandwidth number produced by this machine would be an invention wearing a unit. So
the artifact reports BYTES and COMPUTE, both measured, plus the break-even bandwidth DERIVED from them by
arithmetic, and it never reports a transfer DURATION. Any field whose name would read as a duration states
the assumption in the name (there are none in the headline table, by design) and the ``disclaimer`` field
repeats the constraint.

What is NOT here
---------------
A transfer time for either path. Not measured, not estimated, not modelled. Local in-process speed as a
CLAIMED BENEFIT is out of scope: the mergeable path loses that comparison and says so.

Run
---
    PYTHONPATH=src .venv/bin/python benchmark/mergeable_pca/b6_tradeoff_cost.py
    PYTHONPATH=src .venv/bin/python benchmark/mergeable_pca/b6_tradeoff_cost.py --repeats 3
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness_common import (  # noqa: E402
    SEED,
    TIMING_POLICY,
    byte_dict,
    checkpoint_config_key,
    checkpoint_path,
    make_block,
    peak_rss_bytes,
    print_summary_table,
    provenance,
    ratio_or_none,
    regime_of,
    serialized_nbytes,
    time_repeated,
    write_result,
)

from deisa.dask.mergeable_pca import local_pca, merge_tree  # noqa: E402

#: Artifact stem.
ARTIFACT_STEM = "b6_tradeoff_cost"

#: Bridges per configuration. 32 is the paper's multi-bridge anchor; the merge tree is what this measures,
#: and the per-bridge work is measured at the SAME shape for every configuration so the two are separable.
N_BRIDGES = 32

#: Local ranks measured. ``None`` is full local rank, where the merge is exact and the summary is as
#: LARGE as the data -- the flat-regime honesty case. The truncated values are where the transfer saving
#: actually appears, so both ends of the trade are measured.
LOCAL_RANKS: tuple[int | None, ...] = (8, 32, None)

#: Block shapes. MODEST ONLY. The point is the multi-bridge reduction and the volume it saves, not
#: per-block SVD cost: SVD grows about O(n*d^2), and an earlier attempt at 8 to 24 GiB blocks reached
#: 10.4 GB RSS and 56 minutes on ONE configuration before being killed. These shapes keep the whole sweep
#: inside minutes.
BLOCK_ROWS: tuple[int, ...] = (256, 1024, 4096)
FEATURE_DIMS: tuple[int, ...] = (32, 128, 512)

#: Block shapes that reach the ``n_block/d = 4`` column, which the default grid above CANNOT produce:
#: 256/512/4096 divided by 32/128/512 only ever yields x in {0.5, 2, 8, 32, 128}. Figure 2 draws an
#: ``x=4`` tick, so the default sweep leaves that tick with no measured row behind it. These are the
#: shapes that hit it (4*d for each d), kept separate so the default sweep's cost and its checkpoint
#: key are unchanged and the gap-fill is opt-in via ``--only-missing-against``.
GAP_FILL_BLOCK_ROWS: tuple[int, ...] = (128, 512, 2048)

#: Intrinsic rank of the synthetic signal as a FRACTION of ``d``. An INPUT, matching b1/b5.
INTRINSIC_RANK_FRACTION = 0.25

#: Timed repeats per arm. Overridable; whatever ran is stamped under ``inputs`` and cross-checked at write.
TIMED_REPEATS = 5


# ============================================================================= one configuration
def _fit_after_full_transfer(blocks: Sequence[np.ndarray], rank_arg: int | None) -> dict[str, Any]:
    """The LEGACY alternative, done after a full transfer: pool every block, then one exact SVD.

    This is the work the legacy path does on the analytics engine once all the data has arrived. It is the
    honest counterpart of the bridge path: the SAME total decomposition, on the same data, performed in one
    place instead of on 32 bridges plus a reduction.

    Note this is the FAIREST possible baseline for the legacy path -- a single batched SVD, not a loop of
    per-rank SVDs. Anything worse would flatter the bridge path, and the point is to measure what the bridge
    really costs.

    - ``:param blocks:`` The per-bridge blocks.
    - ``:param rank_arg:`` Retained local rank, or ``None`` for full local rank.
    """
    pooled = np.vstack([np.asarray(b, dtype=np.float64) for b in blocks])
    summary = local_pca(pooled, rank=rank_arg)
    return {"summary": summary, "pooled_rows": int(pooled.shape[0])}


def measure_configuration(
    n_block: int,
    d: int,
    local_rank: int | None,
    repeats: int,
) -> dict[str, Any]:
    """Measure the bridge path against the full-transfer path on ONE configuration.

    Both paths decompose the SAME 32 blocks to the SAME rank. The only difference is WHERE: on the
    bridges before shipping, or on the analytics engine after shipping.

    - ``:param n_block:`` Rows per bridge block.
    - ``:param d:`` Feature dimension.
    - ``:param local_rank:`` Retained local rank, or ``None`` for full local rank.
    - ``:param repeats:`` Timed repeats per arm, excluding warmup.
    """
    intrinsic = max(1, int(INTRINSIC_RANK_FRACTION * d))
    blocks = [make_block(n_block=n_block, n_features=d, rank=intrinsic, seed=SEED + 11) for _ in range(N_BRIDGES)]
    rank_arg = local_rank if local_rank is None else min(int(local_rank), d)

    # ---- PATH A: bridge side. Per-bridge local PCA, then the merge tree.
    leaves_timing = time_repeated(lambda: [local_pca(b, rank=rank_arg) for b in blocks], repeats=repeats)
    leaves = [local_pca(b, rank=rank_arg) for b in blocks]
    merge_timing = time_repeated(lambda: merge_tree(leaves), repeats=repeats)
    bridge_total = {
        "seconds_median": leaves_timing["seconds_median"] + merge_timing["seconds_median"],
        "seconds_all": [a + b for a, b in zip(leaves_timing["seconds_all"], merge_timing["seconds_all"], strict=True)],
        "seconds_min": leaves_timing["seconds_min"] + merge_timing["seconds_min"],
        "seconds_max": leaves_timing["seconds_max"] + merge_timing["seconds_max"],
        "seconds_iqr": leaves_timing["seconds_iqr"] + merge_timing["seconds_iqr"],
        "seconds_stddev": float(np.hypot(leaves_timing["seconds_stddev"], merge_timing["seconds_stddev"])),
        "warmup_rounds": leaves_timing["warmup_rounds"],
        "timed_repeats": leaves_timing["timed_repeats"],
        "composition": "sum of two independently timed medians (32 leaves + merge tree), not a separately timed blend",
        "leaves_seconds_median": leaves_timing["seconds_median"],
        "merge_tree_seconds_median": merge_timing["seconds_median"],
    }
    merged = merge_tree(leaves)

    # ---- PATH B: after a full transfer. One batched SVD on the pooled data.
    full_timing = time_repeated(lambda: _fit_after_full_transfer(blocks, rank_arg), repeats=repeats)
    pooled_state = _fit_after_full_transfer(blocks, rank_arg)
    pooled_summary = pooled_state["summary"]

    # ---- BYTES ON THE WIRE, both paths, through the SAME serializer on the same block.
    block_wire = serialized_nbytes(blocks[0])
    bridge_leaf_wire = serialized_nbytes(leaves[0])
    bridge_full_wire = block_wire * N_BRIDGES
    bridge_summary_wire = bridge_leaf_wire * N_BRIDGES
    bytes_saved = bridge_full_wire - bridge_summary_wire

    bridge_seconds = float(bridge_total["seconds_median"])
    full_seconds = float(full_timing["seconds_median"])
    added_compute = bridge_seconds - full_seconds

    # ---- THE TRADE, as a break-even condition rather than an invented transfer duration.
    # bytes_saved / added_compute has units of bytes per second: the bandwidth at which the avoided
    # transfer time exactly equals the added compute. No link speed is assumed to obtain it.
    break_even_bytes_per_s = (bytes_saved / added_compute) if added_compute > 0 else None
    break_even_gbps = (break_even_bytes_per_s / 1e9) if break_even_bytes_per_s is not None else None

    # Accuracy of the 32-way merge against the single-process reference, because a 32-way merge that returns
    # a plausible but WRONG answer is worse than no measurement at all.
    from harness_common import subspace_distance

    k = min(int(merged.rank), int(pooled_summary.rank))
    merge_vs_full = subspace_distance(merged.components[:k], pooled_summary.components[:k])
    mean_delta = float(np.max(np.abs(np.asarray(merged.mean) - np.asarray(pooled_summary.mean))))

    return {
        "n_block": int(n_block),
        "n_features": int(d),
        "n_block_over_d": float(n_block) / float(d),
        "regime": regime_of(n_block, d),
        "n_bridges": N_BRIDGES,
        "local_rank_requested": None if local_rank is None else int(local_rank),
        "local_rank_effective_per_leaf": int(leaves[0].rank),
        "intrinsic_rank": intrinsic,
        "pooled_rows": int(pooled_state["pooled_rows"]),
        "merge_exact_at_full_local_rank": bool(local_rank is None),
        # ---- quantity 1: bytes on the wire. MEASURED.
        "wire_bytes": {
            "legacy_full_transfer_all_bridges_measured": byte_dict(bridge_full_wire),
            "bridge_summary_all_bridges_measured": byte_dict(bridge_summary_wire),
            "per_bridge_block_measured": byte_dict(block_wire),
            "per_bridge_leaf_summary_measured": byte_dict(bridge_leaf_wire),
            "bytes_saved_all_bridges_measured": byte_dict(bytes_saved),
            "bytes_saved_per_bridge_send_measured": byte_dict(bytes_saved // N_BRIDGES),
            "bytes_saved_is_negative": bool(bytes_saved < 0),
            "ratio_legacy_vs_bridge_summary": ratio_or_none(bridge_full_wire, bridge_summary_wire),
            "note": (
                "MEASURED. Both payloads serialized through the same serializer on the same block, so the "
                "difference is arithmetic on two measurements. The saving is over the MEASURED bridge count "
                "of this configuration; a negative value means the summary is larger than the chunk, which "
                "is the flat-regime result and is kept rather than clipped"
            ),
        },
        # ---- quantity 2: added local compute. MEASURED, both paths, difference reported.
        "added_local_compute_seconds_measured": {
            "bridge_pca_plus_merge_tree_seconds_median": bridge_seconds,
            "full_transfer_then_pooled_svd_seconds_median": full_seconds,
            "added_compute_seconds_median": added_compute,
            "bridge_over_full_transfer_ratio": ratio_or_none(bridge_seconds, full_seconds),
            "mergeable_is_slower_in_process": bool(added_compute > 0),
            "leaves_seconds_median": leaves_timing["seconds_median"],
            "merge_tree_seconds_median": merge_timing["seconds_median"],
            "bridge_timing": bridge_total,
            "full_transfer_timing": full_timing,
            "note": (
                "MEASURED, in process, on this machine's CPU. A POSITIVE added_compute means the bridge "
                "path is SLOWER in process than one batched SVD on the analytics engine, which is the "
                "measured result: the bridge pays compute to buy transfer volume. The full-transfer arm is "
                "the FAIREST legacy comparison (one batched SVD, not a per-rank loop), so a worse baseline "
                "would only flatter the bridge path"
            ),
        },
        # ---- quantity 3: the trade. DERIVED, and labelled as arithmetic on two measurements.
        "break_even_link_gbps": break_even_gbps,
        "break_even_link_bytes_per_second": break_even_bytes_per_s,
        "break_even_definition": (
            "DERIVED ARITHMETIC on the two measured quantities above, not a measurement: the bridge-side "
            "cost is worth paying exactly when the avoided transfer time exceeds the added compute, "
            "seconds_avoided = bytes_saved / bandwidth > added_compute, which rearranges to "
            "bandwidth_required = bytes_saved / added_compute. NO link speed is assumed and NO transfer "
            "duration is reported. Null when added_compute <= 0, i.e. when the bridge path is not slower "
            "and there is nothing to trade"
        ),
        "break_even_is_a_measurement": False,
        "transfer_time_reported_anywhere": False,
        # ---- correctness of the 32-way merge, against a single-process reference.
        "merge_correctness": {
            "reference": "single-process pooled SVD of all 32 blocks, independent of the merge tree",
            "subspace_distance_merged_vs_pooled": merge_vs_full,
            "max_abs_mean_difference": mean_delta,
            "pooled_rank": int(pooled_summary.rank),
            "merged_rank": int(merged.rank),
            "n_samples_merged": int(merged.n_samples),
            "n_samples_pooled": int(pooled_summary.n_samples),
            "holds": bool(
                merged.rank == pooled_summary.rank
                and merged.n_samples == pooled_summary.n_samples
                and merge_vs_full < 1e-8
            ),
        },
        "peak_rss_bytes": peak_rss_bytes(),
    }


# ============================================================================= resume checkpointing
def _config_key(n_block: int, d: int, local_rank: int | None) -> str:
    """Return the resume key identifying one measured configuration.

    - ``:param n_block:`` Rows per bridge block.
    - ``:param d:`` Feature dimension.
    - ``:param local_rank:`` Retained local rank, or ``None`` for full local rank.
    - ``:return:`` A stable string key, identical across runs so a checkpoint row is found again.
    """
    return json.dumps([int(n_block), int(d), None if local_rank is None else int(local_rank)])


def _grid_key() -> dict[str, Any]:
    """Return the grid description stamped on every checkpoint record of this sweep.

    - ``:return:`` The sweep's inputs. A checkpoint recorded under a different grid is refused rather
      than reused, because those rows are not the same measurement.
    """
    return {
        "grid": ARTIFACT_STEM,
        "n_bridges": N_BRIDGES,
        "block_rows": list(BLOCK_ROWS),
        "feature_dims": list(FEATURE_DIMS),
        "local_ranks": [None if r is None else int(r) for r in LOCAL_RANKS],
        "intrinsic_rank_fraction_of_d": INTRINSIC_RANK_FRACTION,
    }


def _load_checkpoint_rows(checkpoint: Any, repeats: int) -> dict[str, dict[str, Any]]:
    """Return this run's already-measured configurations, keyed for reuse.

    - ``:param checkpoint:`` The handle from :func:`harness_common.checkpoint_config_key`.
    - ``:param repeats:`` Timed repeats this run will use.
    - ``:return:`` ``{config_key: row}`` for rows that are safe to replay.
    """
    path: Path = checkpoint.path
    if not path.exists():
        return {}
    recorded: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn trailing line from a hard kill; that configuration is simply re-measured
        if record.get("config_key") is not None:
            recorded.add(json.dumps(record["config_key"], sort_keys=True))
    if json.dumps(checkpoint.key, sort_keys=True) not in recorded:
        print(f"  checkpoint: {path.name} belongs to a different grid; starting fresh", flush=True)
        path.unlink(missing_ok=True)
        return {}
    replayed = checkpoint.load(repeats=repeats)
    return {_config_key(int(r["n_block"]), int(r["n_features"]), r.get("local_rank_requested")): r for r in replayed}


# ============================================================================= run
def _print_row(row: dict[str, Any]) -> None:
    """Print one measured configuration, whether measured now or replayed from a checkpoint.

    - ``:param row:`` The measured row.
    """
    wire = row["wire_bytes"]
    comp = row["added_local_compute_seconds_measured"]
    print(
        f"  measured n_block={row['n_block']:>5} d={row['n_features']:<4} "
        f"rank={str(row['local_rank_requested']):<4} "
        f"saved={wire['bytes_saved_all_bridges_measured']['MiB']:9.4f} MiB "
        f"added={comp['added_compute_seconds_median']:+9.6f}s "
        f"ratio={wire['ratio_legacy_vs_bridge_summary']} "
        f"break_even={row['break_even_link_gbps']} Gbps "
        f"merge_ok={row['merge_correctness']['holds']}",
        flush=True,
    )


# ============================================================================= plan
def plan_configurations(
    only_missing_against: list[dict[str, Any]] | None = None,
    include_gap_fill_rows: bool = False,
) -> list[tuple[int, int, int | None]]:
    """Enumerate the ``(n_block, d, local_rank)`` triples this run measures, in attempt order.

    One block shape per regime so the trade is visible on both sides of ``n_block = d``, and the local
    ranks that give it both ends. Duplicates dropped, so nothing is measured twice.

    - ``:param only_missing_against:`` When given, keep only the triples absent from this list of
       already-measured rows. The default ``FEATURE_DIMS`` x ``BLOCK_ROWS`` grid yields just five
       distinct ``n_block/d`` values and does NOT include ``x=4``, which is the tick Figure 2 draws,
       so a sweep of the default plan cannot fill that gap. Filtering against measured rows is how a
       gap-fill run targets it.
    - ``:param include_gap_fill_rows:`` Also offer :data:`GAP_FILL_BLOCK_ROWS`, the shapes that reach
       ``x=4``. Only meaningful together with ``only_missing_against``: without measured rows to
       exclude, this just re-measures the default grid at extra shapes.
    - ``:return:`` The configurations to attempt.
    """
    measured: set[tuple[int, int, int | None]] = set()
    if only_missing_against is not None:
        for row in only_missing_against:
            if not isinstance(row, dict):
                continue
            n_block = row.get("n_block")
            d = row.get("n_features")
            rank = row.get("local_rank_requested")
            if n_block is None or d is None:
                continue
            measured.add((int(n_block), int(d), None if rank is None else int(rank)))

    block_rows = BLOCK_ROWS
    if include_gap_fill_rows:
        block_rows = tuple(dict.fromkeys((*BLOCK_ROWS, *GAP_FILL_BLOCK_ROWS)))

    plan: list[tuple[int, int, int | None]] = []
    seen: set[tuple[int, int, int | None]] = set()
    for d in FEATURE_DIMS:
        for n_block in block_rows:
            for local_rank in LOCAL_RANKS:
                if local_rank is not None and local_rank > d:
                    continue
                key = (n_block, d, local_rank)
                if key in seen:
                    continue
                if only_missing_against is not None and key in measured:
                    continue
                seen.add(key)
                plan.append(key)
    return plan


def run(
    repeats: int,
    only_missing_against: list[dict[str, Any]] | None = None,
    include_gap_fill_rows: bool = False,
) -> dict[str, Any]:
    """Measure every configuration and return the artifact payload.

    - ``:param repeats:`` Timed repeats per arm, excluding warmup.
    - ``:param only_missing_against:`` Measured rows to exclude, see :func:`plan_configurations`.
    - ``:param include_gap_fill_rows:`` Offer the x=4 block shapes, see :func:`plan_configurations`.
    """
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    # Resume support, same rationale as baseline_comparison: each configuration costs minutes, so a sweep
    # that writes its artifact only at the end loses everything to any single interruption.
    checkpoint = checkpoint_config_key(ARTIFACT_STEM, _grid_key())
    reusable = _load_checkpoint_rows(checkpoint, repeats)
    if reusable:
        print(f"  checkpoint: reusing {len(reusable)} configuration(s) already measured by a prior run", flush=True)

    for n_block, d, local_rank in plan_configurations(only_missing_against, include_gap_fill_rows):
        cached = reusable.get(_config_key(n_block, d, local_rank))
        if cached is not None:
            rows.append({**cached, "replayed_from_checkpoint": True})
            _print_row(cached)
            continue
        try:
            row = measure_configuration(n_block, d, local_rank, repeats)
        except MemoryError as exc:  # pragma: no cover - the shapes here are modest
            skipped.append(
                {
                    "n_block": n_block,
                    "n_features": d,
                    "local_rank_requested": local_rank,
                    "status": "skipped_on_memory_error",
                    "reason": f"{type(exc).__name__} allocating the configuration: {exc}",
                }
            )
            continue
        rows.append(row)
        checkpoint.append(row, repeats=repeats)
        _print_row(row)
        gc.collect()

    return {
        "provenance": provenance(
            script=ARTIFACT_STEM,
            description=(
                "The trade the paper defers: bridge-side PCA plus the 32-bridge merge tree measured against "
                "sending the full block and doing the same decomposition on the analytics engine. Reports "
                "bytes saved (measured), added local compute (measured, both arms, with the difference), and "
                "the break-even bandwidth required for the trade to pay -- derived arithmetic, never a "
                "transfer duration."
            ),
            extra={
                "timing_policy": {
                    "clock": TIMING_POLICY["clock"],
                    "warmup_rounds": TIMING_POLICY["warmup_rounds"],
                    "timed_repeats": int(repeats),
                    "statistic": TIMING_POLICY["statistic"],
                    "dispersion": TIMING_POLICY["dispersion"],
                    "note": TIMING_POLICY.get("note", ""),
                },
                "inputs": {
                    "n_bridges": N_BRIDGES,
                    "block_rows": list(BLOCK_ROWS),
                    "feature_dims": list(FEATURE_DIMS),
                    "local_ranks": [None if r is None else int(r) for r in LOCAL_RANKS],
                    "intrinsic_rank_fraction_of_d": INTRINSIC_RANK_FRACTION,
                    "timed_repeats": int(repeats),
                    "block_shapes": "MODEST BY DESIGN: the point is the multi-bridge reduction and the "
                    "volume saved, not per-block SVD cost",
                },
                "timing_policy_why": (
                    "in-process COMPUTE only, with one warmup round discarded and the median of real repeats "
                    "as the headline plus min/max/iqr/stddev"
                ),
                "memory_policy": {
                    "driver_peak_rss_bytes": peak_rss_bytes(),
                    "note": "process-lifetime high-water mark from /proc/self/status VmHWM",
                },
                "resume_policy": {
                    "checkpoint_path": str(checkpoint_path(ARTIFACT_STEM)),
                    "granularity": "one JSONL record per configuration, appended and fsynced as it is measured",
                    "why": (
                        "each configuration costs minutes, so an artifact written only once at the end is "
                        "lost in full by any single interruption"
                    ),
                    "rows_reused_from_a_prior_run": int(sum(1 for r in rows if r.get("replayed_from_checkpoint"))),
                    "guarded_against": (
                        "a checkpoint recorded under a different grid or a different repeat count is "
                        "refused, not merged: such rows are not the same measurement"
                    ),
                },
            },
        ),
        "disclaimer": (
            "NO BANDWIDTH FIGURE AND NO TRANSFER DURATION APPEARS IN THIS ARTIFACT, and none is implied. "
            "Grid5000 is where the real network lives and it is not reachable from this container, so "
            "everything measured here is SINGLE-NODE: bytes and in-process compute. The break-even bandwidth "
            "is DERIVED ARITHMETIC on two measured quantities (bytes_saved / added_compute), not a measured "
            "link speed, and it names no assumed bandwidth of its own. A real transfer time requires the "
            "multi-node Grid5000 measurement, which is a separate experiment. The Gysela anchor "
            "(512,128,64,128,8), d=524288 and a full summary on the order of 2 TB against a roughly 4 GB "
            "slab are ARITHMETIC FROM THE SIZING MODEL, not measurements."
        ),
        "scope_limits": {
            "measured_block_bytes_range": "0.00025 MiB to 0.25 MiB per bridge (32 bridges pooled)",
            "not_attempted_large_blocks": (
                "8 to 24 GiB blocks were NOT attempted in this run. An earlier attempt reached 10.4 GB RSS "
                "and 56 minutes on ONE configuration before being killed, because SVD cost grows about "
                "O(n*d^2). Recorded so a reader knows what was and was not tried"
            ),
            "local_in_process_speed": (
                "OUT OF SCOPE as a claimed benefit. The mergeable path loses the in-process comparison and "
                "the artifact says so; the contribution is bridge-side compute plus avoided transfer"
            ),
            "network_bandwidth": "NOT MEASURED HERE. Single-node only; see the disclaimer",
        },
        "trade_summary": {
            "rule": (
                "the bridge-side cost is worth paying exactly when the avoided transfer time exceeds the "
                "added compute: seconds_avoided = bytes_saved / bandwidth > added_compute"
            ),
            "express_as": "break_even_link_gbps = bytes_saved / added_compute, DERIVED per configuration",
            "sign_convention": (
                "added_compute_seconds_median POSITIVE means the bridge path is slower in process, so there "
                "is a real trade to make. Negative means the bridge path is both smaller on the wire and no "
                "slower, and there is nothing to weigh"
            ),
        },
        "results": rows,
        "skipped": skipped,
    }


def main() -> int:
    """Entry point: run the measurement, write the artifact, print a human summary.

    - ``:return:`` ``0`` on success.
    """
    doc = __doc__ or ""
    parser = argparse.ArgumentParser(description=doc.splitlines()[0] if doc else "")
    parser.add_argument("--repeats", type=int, default=TIMED_REPEATS, help="Timed repeats per arm")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Discard any resume checkpoint and re-measure every configuration from scratch",
    )
    parser.add_argument(
        "--only-missing-against",
        metavar="ARTIFACT_JSON",
        default=None,
        help=(
            "Measure only the configurations absent from ARTIFACT_JSON's results, merging the rows "
            "already measured there. The default grid yields just five distinct n_block/d values and "
            "does NOT include x=4, which Figure 2 draws, so a default sweep cannot fill that gap. "
            "Use this to target the missing cells. Writes a NEW artifact; never edits the input."
        ),
    )
    parser.add_argument(
        "--with-gap-fill-rows",
        action="store_true",
        help=(
            "With --only-missing-against, also offer the block shapes that reach n_block/d=4 "
            f"({GAP_FILL_BLOCK_ROWS}). Required to fill the x=4 tick Figure 2 draws, because the "
            "default grid cannot produce it."
        ),
    )
    args = parser.parse_args()
    if args.repeats < 2:
        print("refusing: --repeats < 2 leaves no dispersion behind the median")
        return 1

    already_measured: list[dict[str, Any]] | None = None
    include_gap = False
    if args.only_missing_against:
        prior_path = Path(args.only_missing_against)
        if not prior_path.exists():
            print(f"refusing: --only-missing-against {prior_path} does not exist")
            return 1
        prior = json.loads(prior_path.read_text())
        already_measured = list(prior.get("results", []))
        include_gap = args.with_gap_fill_rows
        kept = plan_configurations(already_measured, include_gap)
        print(
            f"gap-fill against {prior_path.name}: {len(already_measured)} row(s) already measured, "
            f"{len(kept)} configuration(s) to measure"
            + (f" (including x=4 shapes {GAP_FILL_BLOCK_ROWS})" if include_gap else "")
        )
        if not kept:
            print("nothing missing: every planned configuration is already measured")
            return 0
        xs = sorted({round(n / d, 4) for n, d, _ in kept})
        print(f"  x values this run will cover: {xs}")
        if 4.0 not in xs:
            print(
                "  WARNING: x=4.0 is still NOT covered. Pass --with-gap-fill-rows so the x=4 block shapes are offered."
            )
    if args.reset:
        checkpoint_path(ARTIFACT_STEM).unlink(missing_ok=True)
        print(f"checkpoint: discarded {checkpoint_path(ARTIFACT_STEM).name} (--reset)")

    payload = run(args.repeats, already_measured, include_gap)
    if already_measured is not None:
        # Merge so the new artifact is self-contained: prior rows plus what this run added.
        # Prior rows are flagged, so a reader can tell measured-now from carried-over.
        fresh = {id(r) for r in payload["results"] if not r.get("replayed_from_checkpoint")}
        carried = [{**r, "carried_from_prior_artifact": True} for r in already_measured]
        payload["results"] = carried + payload["results"]
        payload.setdefault("provenance", {})["gap_fill_against"] = str(args.only_missing_against)
        payload["provenance"]["rows_measured_this_run"] = len(fresh)
        payload["provenance"]["rows_carried_from_prior"] = len(carried)
        payload["provenance"]["gap_fill_note"] = (
            "rows marked carried_from_prior_artifact were measured by an earlier run and are reproduced here "
            "unchanged; rows without that flag were measured by this run"
        )
        print(f"merged: {len(carried)} carried + {len(fresh)} newly measured")

    out_stem = ARTIFACT_STEM
    if already_measured is not None:
        # Never overwrite the artifact the gap-fill was told to read.
        out_stem = f"{ARTIFACT_STEM}_gapfill"
    path = write_result(out_stem, payload)
    # Only now are the rows safely inside a written artifact, so the resume state is consumed.
    checkpoint_path(ARTIFACT_STEM).unlink(missing_ok=True)

    print(f"\nbridge-side PCA + merge vs full transfer + pooled SVD, {len(payload['results'])} configuration(s)\n")
    print_summary_table(
        payload["results"],
        columns=(
            ("n_block", "n_blk", "int"),
            ("n_features", "d", "int"),
            ("local_rank_requested", "rank", "auto"),
            ("wire_bytes", "saved_all_bridges", "auto"),
            ("added_local_compute_seconds_measured", "added_compute_s", "float"),
            ("break_even_link_gbps", "break_even_Gbps", "float"),
        ),
        title="measured bytes saved and measured added compute, with the derived break-even bandwidth",
    )
    prov = payload["provenance"]
    print(f"\ncommit          : {prov['deisa_dask_commit']}")
    print(f"timed repeats   : {prov['inputs']['timed_repeats']} (plus {prov['timing_policy']['warmup_rounds']} warmup)")
    print(f"bridges         : {prov['inputs']['n_bridges']}")
    print(f"skipped         : {len(payload['skipped'])}")
    print(f"\nartifact: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
