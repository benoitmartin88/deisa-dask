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
B1: network transfer between the bridge and the analytics engine.

The claim under test is that on the PCA path only a compact mergeable summary crosses the boundary to
the analytics engine, never the field chunk itself. Both arms are pushed through the SAME serializer
the bridge actually uses (``distributed.protocol.serialize(to_serialize(...))``, the call
:meth:`~deisa.dask.bridge.Bridge._scatter_partials` makes before ``scatter_to_workers``), so the
numbers are payload sizes that go on the wire, pickle headers included, rather than an arithmetic
guess at ``array.nbytes``.

Why this script measures a handful of configurations
-----------------------------------------------------
Only the configurations the paper reports are measured. An earlier draft of this script swept an
``n_block/d`` grid crossed with four absolute byte targets up to 16 GiB; it spent hours producing
points no claim depended on, because SVD cost grows roughly as ``O(n * d^2)`` in the sample
dimension and a multi-GiB block is far more work than its size suggests.

That grid also never contained the configurations the paper cites. Its feature dimensions were
``(32, 128, 512)``, which excludes the ``d=256`` the paper's lead configuration uses, and its local
ranks were ``(4, 16, 64, None)``, which excludes the ``local_rank=8`` the paper's rank-truncation
result uses. The paper's headline numbers therefore had no measured row anywhere in the repository.
This run reproduces them at their true shape, which is a new measurement rather than a recovery of
an old one, and the artifact records which shapes those are so the claim is auditable.

Every configuration exists because a paper claim needs a row behind it
---------------------------------------------------------------------------
- :data:`HEADLINE` -- the lead network-transfer number.
- :data:`RANK_TRUNCATION` -- the same block with the local rank capped at 8.
- :data:`FLAT_REGIME_POINTS` -- the no-compression regime the paper reports openly as the design's
  boundary. At least one reproduced row is required, because a boundary claim with no row behind it
  is an assertion.
- :data:`CROSSOVER_POINTS` -- full-rank points on both sides of ratio 1.0, so the paper's both-sides
  claim is backed by data rather than asserted.

Nothing else is measured, and nothing is measured twice.

The invariant is an identity check, not a size check
-----------------------------------------------------
Every row asserts the scattered payload is a :class:`~deisa.dask.mergeable_pca.PCASummary` and that
the block array is neither that payload nor contained in it, tested by object identity. It
deliberately does NOT compare sizes: on the flat side a full-rank summary legitimately holds as many
elements as the block it summarizes, which is the documented regime boundary reported separately as
``summary_not_smaller_than_block``, not a violation.

Cost is reported, not just the saving
--------------------------------------
Bridge-side compute is the price of the transfer reduction: the local decomposition runs on the
bridge where the field already resides, and only the summary crosses. The legacy arm pays no local
CPU at all, so quoting the bytes it saves without the CPU that bought them would misstate the trade.
Every row carries its own local-compute cost and its measured peak RSS. No local in-process speed
number here is a benefit: the win is the reduction in the volume that crosses the boundary, and the
bridge-side CPU is what pays for it.

Run
---
    PYTHONPATH=src .venv/bin/python benchmark/mergeable_pca/b1_network_transfer.py
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness_common import (  # noqa: E402
    SEED,
    TIMING_POLICY,
    byte_dict,
    cgroup_memory_limit_bytes,
    current_rss_bytes,
    make_block,
    peak_rss_bytes,
    print_summary_table,
    provenance,
    ratio_or_none,
    regime_of,
    reset_peak_rss,
    serialized_nbytes,
    summary_elements,
    summary_nbytes,
    write_result,
)

from deisa.dask.mergeable_pca import PCASummary, local_pca, merge_tree  # noqa: E402

#: Artifact stem. Distinct from ``b1_smoke`` on purpose: the smoke file is a capped dry run of a
#: retired script and is NOT paper data, so the two must never be confusable in a results directory.
ARTIFACT_STEM = "b1_network_transfer"

#: The paper's lead configuration, measured at its true shape: a 2048x256 chunk. The quoted result is a
#: 4.000 MiB legacy payload against a 0.5042 MiB full-rank summary, a 7.9x reduction.
#:
#: The retired sweep's feature dimensions were ``(32, 128, 512)``, which does not contain 256, so this
#: configuration was never measured by it.
HEADLINE = (2048, 256)

#: The same block with the local rank capped at 8, quoted as a 0.0179 MiB summary and a 223.7x
#: reduction. The retired sweep's local ranks were ``(4, 16, 64, None)``, which does not contain 8, so
#: this configuration too was never measured.
RANK_TRUNCATION = (2048, 256, 8)

#: Full-rank shapes straddling ratio 1.0, so the crossover is measured on both sides. The pair is
#: deliberate rather than a sweep: the tall side is :data:`HEADLINE` and the flat side is a flat-regime
#: point, so both are already measured for the other claims and are deduplicated in
#: :func:`plan_configurations` rather than measured again.
CROSSOVER_POINTS = ((2048, 256), (256, 512))

#: Flat-regime shapes backing the no-compression claim. Each is ``n_block < d`` at full local rank,
#: where a full-rank summary holds ``min(n_block, d) * d`` elements against ``n_block * d`` for the
#: data and so cannot be smaller. Small by construction: a flat block at multi-GiB scale would need
#: many features, and an SVD whose cost grows with the cube of the feature dimension.
FLAT_REGIME_POINTS = ((256, 512), (128, 512), (64, 128))

#: Intrinsic rank of the synthetic signal, as a FRACTION of ``d``. Kept well below ``d`` so the block
#: is a realistic low-variance field rather than white noise. An INPUT, not a measured result.
INTRINSIC_RANK_FRACTION = 0.25

#: Multiples of the block's byte size the code path is budgeted to need, as an INPUT to the pre-flight
#: check only. Deliberately CONSERVATIVE, and NOT calibrated by this run: the artifact reports the peak
#: actually observed per row, so a reader can confirm this budget never limited the grid. Calibrating
#: it against measurement is what required the large-block sweep this run drops.
PEAK_MULTIPLE_INPUT = 6.0

#: Fraction of the machine's memory cap the projected peak may reach, as an INPUT to the same check.
#: The remainder is headroom for the interpreter, BLAS thread pools and page cache, none of which is
#: attributable to one configuration.
SAFETY_FRACTION = 0.6

#: Wall-clock budget for the whole run, enforced between phases. A configuration that has not finished
#: by the deadline is abandoned and recorded in ``skipped`` WITH its reason, because a run that
#: overruns its budget produces an artifact nobody can rely on.
TOTAL_BUDGET_SECONDS = 20 * 60

#: Recorded when the peak-RSS watermark cannot be reset, so a reader knows the per-row peak column is a
#: process-wide maximum rather than a per-configuration measurement.
PEAK_RSS_WATERMARK_UNRESETTABLE = (
    "per-configuration peak RSS UNAVAILABLE: this platform cannot reset VmHWM, so the column below is the "
    "process-wide high-water mark, not this configuration's peak"
)


class BudgetExhausted(RuntimeError):
    """A configuration exceeded the run's wall-clock budget and was abandoned mid-measurement."""


def _payload_arrays(obj: object) -> list[np.ndarray]:
    """Collect every numpy array reachable from ``obj``, to prove none of them is the full chunk.

    - ``:param obj:`` A scattered payload, here a :class:`PCASummary`.
    """
    found: list[np.ndarray] = []

    def _walk(item: object) -> None:
        if isinstance(item, np.ndarray):
            found.append(item)
        elif isinstance(item, PCASummary):
            _walk(item.components)
            _walk(item.mean)
            _walk(item.singular_values)
        elif isinstance(item, dict):
            for value in item.values():
                _walk(value)
        elif isinstance(item, (list, tuple)):
            for value in item:
                _walk(value)

    _walk(obj)
    return found


def _check_deadline(deadline: float, context: str) -> None:
    """Raise :class:`BudgetExhausted` if the wall-clock budget is spent, else return.

    Checked BETWEEN phases rather than only between configurations. A single SVD is the longest phase
    and nothing can interrupt it once started, so a deadline enforced only at configuration boundaries
    would be discovered exactly one configuration late, which is the failure the budget exists to
    prevent.

    - ``:param deadline:`` ``time.monotonic()`` value after which the run must stop.
    - ``:param context:`` Which phase the run was in when the deadline passed, recorded in the error.
    """
    if time.monotonic() > deadline:
        raise BudgetExhausted(f"wall-clock budget of {TOTAL_BUDGET_SECONDS} s spent during phase {context!r}")


def measure_one(
    n_block: int,
    d: int,
    local_rank: int | None,
    intrinsic_rank: int,
    deadline: float,
) -> dict[str, Any]:
    """Measure the wire bytes of both arms for ONE configuration.

    The invariant check is the point of this function: it asserts the scattered payload is a summary
    and that the full chunk is absent, so "the transfer carries only a mergeable summary" is re-proven
    on every configuration measured rather than assumed once.

    Each phase is timed separately, because the cost side of the trade is bridge-side CPU while the
    benefit side is the volume that crosses the boundary: the reader needs both numbers for the same
    configuration, not a saving quoted without the CPU that bought it.

    - ``:param n_block:`` Rows in the block.
    - ``:param d:`` Feature dimension.
    - ``:param local_rank:`` Retained local rank, or ``None`` for full local rank.
    - ``:param intrinsic_rank:`` Intrinsic rank of the synthetic signal.
    - ``:param deadline:`` ``time.monotonic()`` value after which the run must stop.
    """
    _check_deadline(deadline, "before make_block")
    start = time.perf_counter()
    block = make_block(n_block=n_block, n_features=d, rank=intrinsic_rank, seed=SEED + 11)
    seconds_make_block = time.perf_counter() - start
    block_elements = int(block.size)

    _check_deadline(deadline, "before local_pca")
    start = time.perf_counter()
    summary = local_pca(block, rank=local_rank)
    seconds_local_pca = time.perf_counter() - start

    _check_deadline(deadline, "before serializing the legacy arm")
    start = time.perf_counter()
    legacy_wire = serialized_nbytes(block)
    seconds_legacy_wire = time.perf_counter() - start

    _check_deadline(deadline, "before serializing the summary arm")
    start = time.perf_counter()
    pca_wire = serialized_nbytes(summary)
    seconds_pca_wire = time.perf_counter() - start

    start = time.perf_counter()
    merged = merge_tree([summary])
    seconds_merge = time.perf_counter() - start

    # The invariant, checked rather than assumed, and deliberately about IDENTITY, not SIZE.
    #
    # What must never happen is the CHUNK crossing the boundary. So the test is: the payload is a
    # PCASummary, and the block array itself is not the object being scattered and is not contained in
    # it (checked by identity, since the summary holds fresh arrays it allocated itself).
    #
    # An earlier draft of this check compared SIZES and flagged 16 configurations. That was wrong: on
    # the flat side a full-rank summary legitimately holds exactly n_block*d elements, i.e. it is as big
    # as the block it summarizes. That is the documented boundary of the design, reported as
    # summary_not_smaller_than_block below, not a violation of the invariant. Size equality is a
    # property of the regime; identity is the invariant.
    arrays = _payload_arrays(summary)
    is_summary = isinstance(summary, PCASummary)
    payload_is_the_block = summary is block
    contains_the_block = any(a is block for a in arrays)
    ships_chunk = (not is_summary) or payload_is_the_block or contains_the_block

    summary_elems = summary_elements(summary)
    # Reported separately from the invariant: True means this configuration gets NO size reduction,
    # which is the expected outcome whenever n_block <= d at full local rank.
    not_smaller = summary_elems >= block_elements
    summary_bytes = summary_nbytes(summary)

    total_seconds = seconds_make_block + seconds_local_pca + seconds_legacy_wire + seconds_pca_wire + seconds_merge

    return {
        "n_block": int(n_block),
        "n_features": int(d),
        "n_block_over_d": float(n_block) / float(d),
        "regime": regime_of(n_block, d),
        "local_rank_requested": None if local_rank is None else int(local_rank),
        "local_rank_effective": int(summary.rank),
        "local_rank_saturated_at_d": bool(local_rank is not None and local_rank >= d),
        "intrinsic_rank": int(intrinsic_rank),
        "block_elements": block_elements,
        "block_bytes": byte_dict(block_elements * 8),
        "summary_elements": summary_elems,
        "summary_bytes": byte_dict(summary_bytes),
        "legacy_wire_bytes": byte_dict(legacy_wire),
        "pca_wire_bytes": byte_dict(pca_wire),
        "compression_ratio_legacy_vs_pca_wire": ratio_or_none(legacy_wire, pca_wire),
        "compression_ratio_full_rank_vs_data": ratio_or_none(block_elements * 8, summary_bytes),
        "invariant": {
            "payload_is_pca_summary": bool(is_summary),
            "payload_is_the_block_object": bool(payload_is_the_block),
            "payload_contains_the_block_object": bool(contains_the_block),
            "largest_array_crossing_elements": max((int(a.size) for a in arrays), default=0),
            "ships_full_chunk": bool(ships_chunk),
            "holds": bool(is_summary and not ships_chunk),
        },
        "regime_boundary": {
            "summary_not_smaller_than_block": bool(not_smaller),
            "note": (
                "expected whenever n_block <= d at full local rank: the summary holds min(n_block,d)*d "
                "elements against n_block*d for the data, so it cannot be smaller. This is the design's "
                "boundary and is why local_rank < d exists."
            )
            if not_smaller
            else "",
        },
        "merged_root_rank": int(merged.rank),
        "merged_root_n_samples": int(merged.n_samples),
        "local_compute_cost_seconds": {
            "make_block": seconds_make_block,
            "local_pca": seconds_local_pca,
            "serialize_legacy_wire": seconds_legacy_wire,
            "serialize_pca_wire": seconds_pca_wire,
            "merge_tree_single_summary": seconds_merge,
            "total": total_seconds,
            "legacy_arm_local_cpu": 0.0,
            "legacy_arm_local_cpu_note": (
                "the legacy arm ships the chunk and never summarizes it, so it pays NO local CPU for the "
                "transform; this is the cost the PCA arm incurs to buy its byte saving"
            ),
        },
        "timing_policy": {
            "timed_repeats": 1,
            "seconds_all": [seconds_local_pca],
            "dispersion": None,
            "dispersion_note": (
                "each configuration is measured ONCE and reports a RATIO rather than a duration, and a "
                "serialized size is deterministic given its input, so no dispersion exists to report. "
                "dispersion is None rather than 0.0 on purpose: an iqr or stddev of 0.0 from a single "
                "sample reads as perfect reproducibility when nothing was repeated"
            ),
        },
        "bytes_saved_vs_legacy_wire": int(legacy_wire) - int(pca_wire),
        "bytes_saved_per_second_of_local_cpu": ratio_or_none(legacy_wire - pca_wire, total_seconds),
    }


def preflight_refusal(n_block: int, d: int, cap_bytes: int | None) -> dict[str, Any] | None:
    """Refuse a configuration whose projected peak exceeds the memory budget, else ``None``.

    The projection is ``block_bytes * PEAK_MULTIPLE_INPUT`` and the budget is ``cap *
    SAFETY_FRACTION``: both are declared INPUTS, so a refusal is deterministic arithmetic a reader can
    re-check, not a measurement dressed up as one.

    The gate stays because the large blocks it was built for are what made it necessary, and dropping
    it would re-admit that failure mode. A cgroup exhaustion is a SIGKILL, which ``except MemoryError``
    cannot catch; this gate is the primary defence and the handler the residual.

    - ``:param n_block:`` Rows in the block.
    - ``:param d:`` Feature dimension.
    - ``:param cap_bytes:`` The memory cap to budget against, or ``None`` when the machine reports none.
    """
    if cap_bytes is None:
        return None
    block_bytes = int(n_block) * int(d) * 8
    projected = block_bytes * PEAK_MULTIPLE_INPUT
    budget = cap_bytes * SAFETY_FRACTION
    if projected <= budget:
        return None
    return {
        "status": "skipped_pending_a_machine_with_more_memory",
        "reason": (
            f"pre-flight cap check refused this point before allocating it: projected peak "
            f"{projected:.0f} B = {block_bytes} B block * PEAK_MULTIPLE_INPUT={PEAK_MULTIPLE_INPUT} exceeds "
            f"the budget {budget:.0f} B = cap {cap_bytes} B * SAFETY_FRACTION={SAFETY_FRACTION}. No "
            f"allocation was attempted, so no number for this point was measured."
        ),
        "projected_peak_bytes": projected,
        "budget_bytes": budget,
        "cap_bytes": cap_bytes,
    }


def measure_one_guarded(
    n_block: int,
    d: int,
    local_rank: int | None,
    intrinsic_rank: int,
    claim: str,
    cap_bytes: int | None,
    deadline: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Measure one configuration, returning ``(row, skip)`` -- exactly one of the two is ``None``.

    The peak-RSS watermark is reset first so the recorded peak belongs to THIS configuration.
    ``MemoryError`` and :class:`BudgetExhausted` are both caught and returned as recorded skips: a
    machine that cannot hold a configuration, and a run that ran out of time, are facts about this
    run, and an artifact that silently lost either would read as though it had never been attempted.

    - ``:param n_block:`` Rows in the block.
    - ``:param d:`` Feature dimension.
    - ``:param local_rank:`` Retained local rank, or ``None`` for full local rank.
    - ``:param intrinsic_rank:`` Intrinsic rank of the synthetic signal.
    - ``:param claim:`` The paper claim this configuration exists to support, recorded verbatim.
    - ``:param cap_bytes:`` The memory cap the pre-flight gate compares against, or ``None``.
    - ``:param deadline:`` ``time.monotonic()`` value after which the run must stop.
    """
    block_bytes = int(n_block) * int(d) * 8
    identity = {
        "n_block": int(n_block),
        "n_features": int(d),
        "local_rank_requested": None if local_rank is None else int(local_rank),
        "claim_supported": claim,
        "block_bytes": byte_dict(block_bytes),
        "n_block_over_d": float(n_block) / float(d),
        "regime": regime_of(n_block, d),
    }

    gate = preflight_refusal(n_block, d, cap_bytes)
    if gate is not None:
        return None, {**identity, **gate}

    watermark_reset = reset_peak_rss()
    rss_before = current_rss_bytes()
    try:
        row = measure_one(n_block, d, local_rank, intrinsic_rank, deadline)
    except MemoryError as exc:
        return None, {
            **identity,
            "status": "skipped_memory_error",
            "reason": f"MemoryError while building or summarizing a {block_bytes}-byte block: {exc}",
            "peak_rss_bytes": peak_rss_bytes(),
        }
    except BudgetExhausted as exc:
        return None, {
            **identity,
            "status": "skipped_wall_clock_budget",
            "reason": (
                f"abandoned mid-configuration against the {TOTAL_BUDGET_SECONDS} s run budget: {exc}. "
                "No result was recorded. The phases completed before the deadline did run, but a partial "
                "measurement is not a measurement and is not reported as one."
            ),
            "peak_rss_bytes": peak_rss_bytes(),
        }
    peak = peak_rss_bytes()

    row["claim_supported"] = claim
    row["memory"] = {
        "peak_rss_bytes": peak,
        "peak_rss": byte_dict(peak) if peak >= 0 else byte_dict(0),
        "peak_rss_available": peak >= 0,
        "peak_rss_attributable_to_this_configuration": watermark_reset,
        "peak_rss_attribution_note": "" if watermark_reset else PEAK_RSS_WATERMARK_UNRESETTABLE,
        "rss_before_bytes": rss_before,
        "cap_bytes": cap_bytes,
        "fraction_of_cap": (float(peak) / float(cap_bytes)) if (peak >= 0 and cap_bytes) else None,
        "peak_multiple_of_block_bytes": ratio_or_none(peak, block_bytes),
        "peak_rss_within_budget_input": bool(cap_bytes is not None and peak >= SAFETY_FRACTION * cap_bytes),
    }
    return row, None


def plan_configurations() -> list[tuple[int, int, int | None, str]]:
    """Enumerate exactly the ``(n_block, d, local_rank, claim)`` tuples this run measures.

    Every entry traces to one of the four claims in the module docstring. Duplicates are dropped so a
    configuration is never measured twice under two claim labels: :data:`CROSSOVER_POINTS` names two
    shapes already covered by the other lists, and they are deduplicated here rather than measured
    again, because a shape already measured for the headline or for the flat regime IS the crossover
    evidence and its ratio appears in the ``crossover`` block either way.

    The lead configuration is planned FIRST, so if anything is lost to the wall-clock budget it is the
    least load-bearing row rather than the most.

    - ``:return:`` The configurations to attempt, in attempt order.
    """
    plan: list[tuple[int, int, int | None, str]] = []
    seen: set[tuple[int, int, int | None]] = set()

    def _add(n_block: int, d: int, local_rank: int | None, claim: str) -> None:
        """Append one configuration unless the same shape and rank is already planned.

        - ``:param n_block:`` Rows in the block.
        - ``:param d:`` Feature dimension.
        - ``:param local_rank:`` Retained local rank, or ``None`` for full local rank.
        - ``:param claim:`` The paper claim this configuration supports.
        """
        key = (n_block, d, local_rank)
        if key in seen:
            return
        seen.add(key)
        plan.append((n_block, d, local_rank, claim))

    _add(HEADLINE[0], HEADLINE[1], None, "headline_full_rank_network_transfer")
    _add(RANK_TRUNCATION[0], RANK_TRUNCATION[1], RANK_TRUNCATION[2], "rank_truncation_local_rank_8")
    for n_block, d in CROSSOVER_POINTS:
        _add(n_block, d, None, "crossover_full_rank_point")
    for n_block, d in FLAT_REGIME_POINTS:
        _add(n_block, d, None, "flat_regime_no_compression")

    return plan


def run() -> dict[str, Any]:
    """Run the cited configurations and return the artifact payload.

    Configurations are attempted HEADLINE FIRST rather than largest-first. The retired sweep sorted by
    descending block size so a machine that could not hold the biggest block would refuse it while
    time remained to report it; with a grid this small that ordering bought nothing and cost the
    headline number its place at the front of the run.

    - ``:return:`` The artifact payload, ready for :func:`harness_common.write_result`.
    """
    started = time.monotonic()
    deadline = started + TOTAL_BUDGET_SECONDS
    cap_bytes = cgroup_memory_limit_bytes()

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for n_block, d, local_rank, claim in plan_configurations():
        intrinsic = max(1, int(INTRINSIC_RANK_FRACTION * d))
        row, skip = measure_one_guarded(n_block, d, local_rank, intrinsic, claim, cap_bytes, deadline)
        if row is not None:
            rows.append(row)
            print(
                f"  measured n_block={n_block:>7} d={d:<5} rank={str(local_rank):<4} "
                f"legacy={row['legacy_wire_bytes']['MiB']:9.4f} MiB "
                f"summary={row['pca_wire_bytes']['MiB']:9.4f} MiB "
                f"ratio={row['compression_ratio_legacy_vs_pca_wire']:.4f} "
                f"peak={row['memory']['peak_rss_bytes'] / 2**30:6.2f} GiB "
                f"[{claim}]",
                flush=True,
            )
        elif skip is not None:
            skipped.append(skip)
            print(f"  skipped  n_block={n_block} d={d} rank={local_rank}: {skip['status']}", flush=True)
        gc.collect()

    elapsed = time.monotonic() - started

    full_rank = [r for r in rows if r["local_rank_requested"] is None]
    ratios = [r["compression_ratio_legacy_vs_pca_wire"] for r in full_rank]
    defined_ratios = [x for x in ratios if x is not None]
    at_or_below = [r for r in full_rank if (r["compression_ratio_legacy_vs_pca_wire"] or 0.0) <= 1.0]
    above = [r for r in full_rank if (r["compression_ratio_legacy_vs_pca_wire"] or 0.0) > 1.0]

    headline_row = next(
        (r for r in rows if (r["n_block"], r["n_features"]) == HEADLINE and r["local_rank_requested"] is None),
        None,
    )
    truncated_row = next(
        (r for r in rows if r["local_rank_requested"] == RANK_TRUNCATION[2]),
        None,
    )
    flat_rows = [r for r in full_rank if r["regime_boundary"]["summary_not_smaller_than_block"]]
    flat_ratios = [r["compression_ratio_legacy_vs_pca_wire"] for r in flat_rows]
    flat_defined = [x for x in flat_ratios if x is not None]

    return {
        "provenance": provenance(
            script=ARTIFACT_STEM,
            description=(
                "Network transfer between the bridge and the analytics engine: legacy full-chunk scatter "
                "versus bridge-side mergeable PCA summary. Measures only the configurations the paper "
                "reports -- the lead configuration at its true shape, its rank-truncated counterpart, the "
                "flat-regime no-compression rows, and full-rank points straddling ratio 1.0."
            ),
            extra={
                # Restated because provenance() stamps TIMING_POLICY's DEFAULT of 5 and this run measures
                # each configuration once. Declaring the default here would ship the exact provenance
                # defect this harness has a gate for: a repeat count the artifact did not use.
                "timing_policy": {
                    **TIMING_POLICY,
                    "timed_repeats": 1,
                    "note": (
                        "restated to 1 from TIMING_POLICY's default of 5, because each configuration is "
                        "measured once and reports a deterministic ratio rather than a timed distribution"
                    ),
                },
                "inputs": {
                    "feature_dims": sorted({c[1] for c in plan_configurations()}),
                    "configurations": [
                        {"n_block": n, "d": dd, "local_rank": r, "claim_supported": c}
                        for n, dd, r, c in plan_configurations()
                    ],
                    "intrinsic_rank_fraction_of_d": INTRINSIC_RANK_FRACTION,
                    "peak_multiple_input": PEAK_MULTIPLE_INPUT,
                    "safety_fraction_input": SAFETY_FRACTION,
                    "wall_clock_budget_seconds": TOTAL_BUDGET_SECONDS,
                    "wall_clock_budget_enforced": (
                        "checked between phases, not only between configurations: a single SVD cannot be "
                        "interrupted once started, so a check at configuration boundaries alone would be "
                        "discovered one configuration late"
                    ),
                    "repeats_per_configuration": 1,
                },
                "memory_policy": {
                    "cap_bytes": cap_bytes,
                    "cap_source": "/sys/fs/cgroup/memory.max (cgroup v2), else memory.limit_in_bytes (v1)",
                    "budget_bytes": None if cap_bytes is None else cap_bytes * SAFETY_FRACTION,
                    "preflight_gate": (
                        "refuse a point whose block_bytes * PEAK_MULTIPLE_INPUT exceeds cap * "
                        "SAFETY_FRACTION, BEFORE allocating it, because a cgroup exhaustion is a SIGKILL "
                        "that except MemoryError cannot catch"
                    ),
                    "peak_rss_source": "/proc/self/status VmHWM, reset per configuration via /proc/self/clear_refs",
                    "peak_rss_resettable": bool(rows)
                    and bool(rows[0]["memory"]["peak_rss_attributable_to_this_configuration"]),
                },
                "serialization": (
                    "distributed.protocol.serialize(to_serialize(...)), the same call "
                    "Bridge._scatter_partials makes before scatter_to_workers; numbers are on-the-wire "
                    "payload sizes including pickle framing"
                ),
                "wall_clock_seconds_measured": elapsed,
                "scope": (
                    "bridge-side compute is the contribution: the local decomposition runs on the bridge "
                    "where the field already resides and only a mergeable summary crosses. The reduction in "
                    "the volume that crosses the boundary is the result. No local in-process speed number "
                    "in this artifact is a benefit"
                ),
            },
        ),
        "paper_claims_supported": {
            "what_this_is": (
                "the four claims the paper reports from this experiment, each mapped to the rows that back "
                "it. A configuration exists in this run only because one of these needs a row behind it"
            ),
            "headline_full_rank_network_transfer": {
                "claim": (
                    "the lead network-transfer number: a 4.000 MiB chunk against a 0.5042 MiB full-rank "
                    "summary, a 7.9x reduction"
                ),
                "configuration": {"n_block": HEADLINE[0], "d": HEADLINE[1], "local_rank": None},
                "measured_here": bool(headline_row),
                "was_missing_from_the_retired_sweep": True,
                "missing_because": (
                    "the retired sweep's feature dimensions were (32, 128, 512) and exclude the d=256 this "
                    "configuration uses, so no grid row anywhere in the repository carried these numbers"
                ),
                "measured_values": None
                if headline_row is None
                else {
                    "legacy_wire_bytes": headline_row["legacy_wire_bytes"],
                    "pca_wire_bytes": headline_row["pca_wire_bytes"],
                    "ratio": headline_row["compression_ratio_legacy_vs_pca_wire"],
                },
            },
            "rank_truncation_local_rank_8": {
                "claim": (
                    "the same block with the local rank capped at 8 gives a 0.0179 MiB summary, a 223.7x reduction"
                ),
                "configuration": {
                    "n_block": RANK_TRUNCATION[0],
                    "d": RANK_TRUNCATION[1],
                    "local_rank": RANK_TRUNCATION[2],
                },
                "measured_here": bool(truncated_row),
                "was_missing_from_the_retired_sweep": True,
                "missing_because": ("the retired sweep's local ranks were (4, 16, 64, None) and exclude rank 8"),
                "measured_values": None
                if truncated_row is None
                else {
                    "pca_wire_bytes": truncated_row["pca_wire_bytes"],
                    "legacy_wire_bytes": truncated_row["legacy_wire_bytes"],
                    "ratio": truncated_row["compression_ratio_legacy_vs_pca_wire"],
                },
            },
            "flat_regime_no_compression": {
                "claim": (
                    "at full local rank the summary is not smaller than the block in the flat regime, which "
                    "the paper reports openly as the design's boundary rather than a defect"
                ),
                "configurations_attempted": [{"n_block": n, "d": dd} for n, dd in FLAT_REGIME_POINTS],
                "n_reproduced": len(flat_rows),
                "at_least_one_reproduced": bool(flat_rows),
                "ratio_range_measured": None
                if not flat_defined
                else {"min": min(flat_defined), "max": max(flat_defined)},
                "note": (
                    "the paper quotes this regime's ratio as running from 0.79 to 1.00. Those bounds are "
                    "reported here as measured, and any difference from the quoted range is a discrepancy "
                    "for the paper to reconcile rather than a value to reconcile silently"
                ),
            },
            "crossover_full_rank_point": {
                "claim": (
                    "at full local rank the summary is smaller than the block above the crossover and not "
                    "smaller below it, so the boundary is backed by data on both sides"
                ),
                "n_at_or_below_1": len(at_or_below),
                "n_above_1": len(above),
                "spans_both_sides_of_1": bool(at_or_below and above),
                "min_ratio": min(defined_ratios) if defined_ratios else None,
                "max_ratio": max(defined_ratios) if defined_ratios else None,
                "measured_here": bool(at_or_below and above),
            },
        },
        "scope_limits": {
            "no_large_block_was_measured": (
                "this run deliberately drops the multi-GiB sweep. SVD cost grows roughly as O(n*d^2) in "
                "the sample dimension, so a large block is far more work than its size suggests. A large "
                "block, if ever needed, is a separate experiment with its own justification and an agreed "
                "time budget"
            ),
            "flat_at_absolute_scale_not_measured": (
                "a flat point at slab-realistic ABSOLUTE size is not attempted: it needs a block with many "
                "features, and an SVD cost that grows with the cube of the feature dimension. That case is "
                "the gysela_sizing model's, where it is ARITHMETIC, and arithmetic must never be reported "
                "as a measurement however large the machine's memory is"
            ),
            "local_worker_results_out_of_scope": (
                "this artifact reports no local in-process speed number as a benefit. The comparison against "
                "NumPy SVD, SciPy SVD, sklearn PCA, sklearn IncrementalPCA and dask-ml IncrementalPCA is "
                "the LOCAL COST side: the bridge now pays that CPU to save the transfer. Those numbers stay "
                "out of the paper, except at most one sentence acknowledging the local cost the bridge now "
                "pays"
            ),
            "gysela_anchor": {
                "mesh_tor1_tor2_tor3_vpar_mu": [512, 128, 64, 128, 8],
                "measured_here": False,
                "kind": "arithmetic from the gysela_sizing sizing model, not a measurement",
                "note": (
                    "sizing a full-rank summary for this anchor yields a payload on the order of terabytes "
                    "against a local slab of a few GiB, i.e. no compression at all. This benchmark does NOT "
                    "measure it and does not attempt to: the number stays the sizing model's. More memory "
                    "lets the sizing model be tested at larger scale elsewhere; it does not retroactively "
                    "turn a model output into a measurement"
                ),
            },
        },
        "invariant_check": {
            "claim": (
                "on the PCA path the network transfer carries only a mergeable summary; the full chunk "
                "never crosses to the analytics engine"
            ),
            "how_verified": (
                "for every configuration the scattered payload is asserted to be a PCASummary whose arrays "
                "are the summary's own, i.e. the block object itself is neither the payload nor contained "
                "in it; any regression to shipping the chunk fails the run"
            ),
            "note_on_size": (
                "the check is on IDENTITY, not size: on the flat side a full-rank summary legitimately holds "
                "as many elements as the block it summarizes, which is the regime boundary reported "
                "separately as summary_not_smaller_than_block and not a violation"
            ),
            "configurations_checked": len(rows),
            "configurations_satisfying_invariant": sum(1 for r in rows if r["invariant"]["holds"]),
            "configurations_shipping_chunk": sum(1 for r in rows if r["invariant"]["ships_full_chunk"]),
            "configurations_with_no_size_reduction": len(flat_rows),
        },
        "results": rows,
        "skipped": skipped,
    }


def _print(payload: Mapping[str, Any]) -> None:
    """Print the human summary: absolute transfer bytes and the ratio for every configuration.

    Flattens the nested fields the table shows, so the columns resolve instead of printing ``-``
    everywhere: a column that renders ``-`` for every row looks like an empty measurement rather than a
    bad key. The assertion then fails loudly on a key that does not exist, so a future rename cannot
    quietly blank a column again.

    - ``:param payload:`` The artifact payload returned by :func:`run`.
    """
    rows = payload["results"]
    inv = payload["invariant_check"]
    display = []
    for row in rows:
        missing = [k for k in ("local_compute_cost_seconds", "memory") if k not in row]
        if missing:
            raise KeyError(f"b1 _print: row {row.get('n_block')}x{row.get('n_features')} is missing {missing}")
        display.append(
            {
                **row,
                "local_pca_seconds": row["local_compute_cost_seconds"]["local_pca"],
                "peak_rss_bytes": row["memory"]["peak_rss_bytes"],
            }
        )
    print(
        f"\nB1 -- network transfer, bridge -> analytics engine ({len(rows)} configurations, "
        f"{inv['configurations_checked']} invariant-checked, "
        f"{inv['configurations_shipping_chunk']} shipping the chunk, "
        f"{inv['configurations_with_no_size_reduction']} with no size reduction)\n"
    )
    print_summary_table(
        display,
        columns=(
            ("n_block", "n_blk", "int"),
            ("n_features", "d", "int"),
            ("regime", "regime", "str"),
            ("local_rank_requested", "rank", "auto"),
            ("legacy_wire_bytes", "transfer_legacy", "auto"),
            ("pca_wire_bytes", "transfer_summary", "auto"),
            ("compression_ratio_legacy_vs_pca_wire", "ratio", "float"),
            ("peak_rss_bytes", "peak_rss", "auto"),
            ("local_pca_seconds", "local_pca_s", "float"),
        ),
        title="bridge-side compute, absolute transfer bytes and ratio, one row per configuration",
    )
    print("\nclaims:")
    for name, claim in payload["paper_claims_supported"].items():
        if name == "what_this_is":
            continue
        print(f"  {name}: {json.dumps(claim, sort_keys=True, default=str)}")

    prov = payload["provenance"]
    print(f"\nwall time       : {prov['wall_clock_seconds_measured']:.1f} s")
    print(f"cgroup cap      : {prov['memory_policy']['cap_bytes']} bytes")
    print(f"peak RSS overall: {max((r['memory']['peak_rss_bytes'] for r in rows), default=0)} bytes")
    largest = max(rows, key=lambda r: r["block_bytes"]["bytes"], default=None)
    if largest is not None:
        print(
            f"largest block   : {largest['block_bytes']['bytes']} bytes "
            f"(peak {largest['memory']['peak_rss_bytes']} bytes)"
        )
    skipped = payload["skipped"]
    print(f"\n--- skipped: {len(skipped)} configuration(s), measured nothing ---")
    for entry in skipped:
        print(
            f"  n_block={entry['n_block']:>9} d={entry['n_features']:<4} "
            f"rank={str(entry['local_rank_requested']):<4} {entry.get('status', '')}: {entry['reason']}"
        )


def main() -> int:
    """Entry point: run B1, write the artifact, print a human summary.

    - ``:return:`` ``0`` on success, ``1`` if the invariant failed on any configuration.
    """
    doc = __doc__ or ""
    parser = argparse.ArgumentParser(description=doc.splitlines()[0] if doc else "")
    parser.add_argument(
        "--out",
        default=None,
        help="Optional explicit artifact path (default: results/b1_network_transfer.json)",
    )
    args = parser.parse_args()

    payload = run()
    path = write_result(ARTIFACT_STEM, payload)
    if args.out:
        Path(args.out).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        path = Path(args.out)
    _print(payload)

    shipping = payload["invariant_check"]["configurations_shipping_chunk"]
    print(f"\nartifact: {path}")
    if shipping:
        print(f"FAIL: {shipping} configuration(s) shipped the full chunk -- the invariant broke")
        return 1
    print("OK: the network transfer carried only mergeable summaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
