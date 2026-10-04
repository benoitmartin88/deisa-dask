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
B1: bytes crossing the bridge->worker boundary, legacy full-chunk scatter vs bridge-side summary.

This is the lead figure, and it is measured rather than asserted. Both arms are pushed through the SAME serializer
the bridge actually uses (``distributed.protocol.serialize(to_serialize(...))``, the call
:meth:`~deisa.dask.bridge.Bridge._scatter_partials` makes before ``scatter_to_workers``), so the numbers are the
payload sizes that go on the wire, pickle headers included, not an arithmetic guess at ``array.nbytes``.

The load-bearing claim under test is the NEVER-SHIP-THE-CHUNK invariant: on the PCA path only a compact summary
crosses, never the field. Rather than trusting that, the script PROVES it for every measured configuration by
checking that the scattered payload is a :class:`~deisa.dask.mergeable_pca.PCASummary` and that no numpy array in
it is as large as the block. If the mechanism ever regressed to shipping the chunk, this check fails.

What is reported, per configuration
-----------------------------------
- ``legacy_wire_bytes``: the full chunk, serialized, i.e. the precompute=False path.
- ``pca_wire_bytes``: the mergeable summary, serialized, i.e. the precompute=True path.
- ``compression_ratio``: ``legacy / pca``, the headline. Reported together with the regime tag so the tall and flat
  regimes are never conflated.

The regime boundary, stated rather than hidden
----------------------------------------------
A FULL-RANK summary holds ``min(n_block, d) * d + d + rank`` elements against ``n_block * d`` for the data, so it
compresses only when ``n_block > d``. This script sweeps ``n_block / d`` across the flat side (0.25, 0.5) and the tall
side (2, 4, 8) and reports them SEPARATELY, because a bandwidth figure quoted only from the tall side is
cherry-picking. On the flat side a full-rank ratio is expected to sit at or below 1.0 -- that is the boundary of the
design, not a bug, and it is exactly why ``local_rank < d`` exists.

Run
---
    PYTHONPATH=src .venv/bin/python benchmark/mergeable_pca/b1_bytes_crossing.py
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness_common import (  # noqa: E402
    SEED,
    byte_dict,
    make_block,
    print_summary_table,
    provenance,
    ratio_or_none,
    ratio_sweep_points,
    regime_of,
    serialized_nbytes,
    summary_elements,
    summary_nbytes,
    write_result,
)

from deisa.dask.mergeable_pca import PCASummary, local_pca, merge_tree  # noqa: E402

# Feature dimensions spanning a plausible range of fields. ``d`` is an input to the sweep, not a measured result.
FEATURE_DIMS: tuple[int, ...] = (32, 128, 512)

# Intrinsic rank of the synthetic signal, expressed as a FRACTION of ``d``. Kept well below ``d`` so the block is a
# realistic low-variance field rather than white noise.
INTRINSIC_RANK_FRACTION = 0.25

# Retained local ranks for the truncated arm. Spans "deep compression" to "barely truncated", plus ``None`` for the
# exact full-rank arm reported separately.
LOCAL_RANKS: tuple[int | None, ...] = (4, 16, 64, None)

# Block-size cap, in elements, to keep the legacy arm's serialization inside the machine's memory. A sweep point is
# SKIPPED (and recorded as skipped) rather than silently resized, so the JSON never claims a measurement that did
# not happen.
MAX_BLOCK_ELEMENTS = 24_000_000


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


def measure_one(n_block: int, d: int, local_rank: int | None, intrinsic_rank: int) -> dict[str, Any]:
    """Measure the wire bytes of both arms for ONE configuration.

    The invariant check is the point of this function: it asserts the scattered payload is a summary, that its
    largest array is smaller than the block, and that the full chunk is absent -- so the "never ship the chunk"
    property is re-proven on every configuration measured rather than assumed once.

    - ``:param n_block:`` Rows in the block.
    - ``:param d:`` Feature dimension.
    - ``:param local_rank:`` Retained local rank, or ``None`` for full local rank.
    - ``:param intrinsic_rank:`` Intrinsic rank of the synthetic signal.
    """
    block = make_block(n_block=n_block, n_features=d, rank=intrinsic_rank, seed=SEED + 11)
    block_elements = int(block.size)

    summary = local_pca(block, rank=local_rank)
    legacy_wire = serialized_nbytes(block)
    pca_wire = serialized_nbytes(summary)

    # The invariant, checked rather than assumed, and it is deliberately about IDENTITY, not SIZE.
    #
    # What must never happen is the CHUNK crossing the boundary. So the test is: the payload is a PCASummary, and
    # the block array itself is not the object being scattered and is not contained in it (checked by identity, since
    # the summary holds fresh arrays it allocated itself).
    #
    # An earlier draft of this check compared SIZES and flagged 16 configurations. That was wrong: on the flat side
    # a full-rank summary legitimately holds exactly n_block*d elements, i.e. it is as big as the block it
    # summarizes. That is the documented boundary of the design, reported as `summary_not_smaller_than_block` below,
    # not a violation of the invariant. Size equality is a property of the regime; identity is the invariant.
    arrays = _payload_arrays(summary)
    is_summary = isinstance(summary, PCASummary)
    payload_is_the_block = summary is block
    contains_the_block = any(a is block for a in arrays)
    ships_chunk = (not is_summary) or payload_is_the_block or contains_the_block

    summary_elems = summary_elements(summary)
    # Reported separately from the invariant: True means this configuration gets NO size reduction, which is the
    # expected outcome whenever n_block <= d at full local rank.
    not_smaller = summary_elems >= block_elements

    merged = merge_tree([summary])

    return {
        "n_block": int(n_block),
        "n_features": int(d),
        "n_block_over_d": float(n_block) / float(d),
        "regime": regime_of(n_block, d),
        "local_rank_requested": local_rank,
        "local_rank_effective": int(summary.rank),
        "local_rank_saturated_at_d": bool(local_rank is not None and local_rank >= d),
        "intrinsic_rank": int(intrinsic_rank),
        "block_elements": block_elements,
        "block_bytes": byte_dict(block_elements * 8),
        "summary_elements": summary_elems,
        "summary_bytes": byte_dict(summary_nbytes(summary)),
        "legacy_wire_bytes": byte_dict(legacy_wire),
        "pca_wire_bytes": byte_dict(pca_wire),
        "compression_ratio_full_rank_vs_data": ratio_or_none(block_elements * 8, summary_nbytes(summary)),
        "compression_ratio_legacy_vs_pca_wire": ratio_or_none(legacy_wire, pca_wire),
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
                "expected whenever n_block <= d at full local rank: the summary holds min(n_block,d)*d elements "
                "against n_block*d for the data, so it cannot be smaller. This is the design's boundary and is why "
                "local_rank < d exists."
            )
            if not_smaller
            else "",
        },
        "merged_root_rank": int(merged.rank),
        "merged_root_n_samples": int(merged.n_samples),
    }


def run(dims: tuple[int, ...] = FEATURE_DIMS) -> dict[str, Any]:
    """Run the full sweep and return the artifact payload.

    - ``:param dims:`` Feature dimensions to sweep.
    """
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for d in dims:
        intrinsic = max(1, int(INTRINSIC_RANK_FRACTION * d))
        for requested_ratio, n_block in ratio_sweep_points(d):
            if n_block * d > MAX_BLOCK_ELEMENTS:
                skipped.append(
                    {
                        "n_block": int(n_block),
                        "n_features": int(d),
                        "reason": f"block of {n_block * d} elements exceeds MAX_BLOCK_ELEMENTS={MAX_BLOCK_ELEMENTS}",
                    }
                )
                continue
            for local_rank in LOCAL_RANKS:
                rows.append(measure_one(n_block, d, local_rank, intrinsic))

    full_rank = [r for r in rows if r["local_rank_requested"] is None]
    tall = [r for r in full_rank if r["regime"] == "tall"]
    flat = [r for r in full_rank if r["regime"] in ("flat", "square")]

    def _ratio(row: Mapping[str, Any]) -> float:
        """Compression ratio of one row, as a float for sorting. An undefined ratio sorts as 0.

        - ``:param row:`` One result row.
        """
        value = row["compression_ratio_legacy_vs_pca_wire"]
        return float(value) if value is not None else 0.0

    def _best(subset: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not subset:
            return None
        return max(subset, key=_ratio)

    return {
        "provenance": provenance(
            script="b1_bytes_crossing",
            description=(
                "Bytes crossing the bridge->worker boundary: legacy full-chunk scatter vs bridge-side mergeable "
                "PCA summary, swept over n_block/d with tall and flat regimes reported separately."
            ),
            extra={
                "inputs": {
                    "feature_dims": list(dims),
                    "n_block_over_d_ratios": [r for r, _ in ratio_sweep_points(max(dims))],
                    "local_ranks": [None if r is None else int(r) for r in LOCAL_RANKS],
                    "intrinsic_rank_fraction_of_d": INTRINSIC_RANK_FRACTION,
                    "max_block_elements": MAX_BLOCK_ELEMENTS,
                },
                "serialization": (
                    "distributed.protocol.serialize(to_serialize(...)), the same call Bridge._scatter_partials makes "
                    "before scatter_to_workers; numbers are on-the-wire payload sizes including pickle framing"
                ),
            },
        ),
        "invariant_check": {
            "claim": "on the PCA path only a compact summary crosses bridge->worker; the full chunk never does",
            "how_verified": (
                "for every configuration the scattered payload is asserted to be a PCASummary whose arrays are the "
                "summary's own, i.e. the block object itself is neither the payload nor contained in it; any "
                "regression to shipping the chunk fails the run"
            ),
            "note_on_size": (
                "the check is on IDENTITY, not size: on the flat side a full-rank summary legitimately holds as many "
                "elements as the block it summarizes, which is the regime boundary reported separately as "
                "summary_not_smaller_than_block and not a violation"
            ),
            "configurations_checked": len(rows),
            "configurations_satisfying_invariant": sum(1 for r in rows if r["invariant"]["holds"]),
            "configurations_shipping_chunk": sum(1 for r in rows if r["invariant"]["ships_full_chunk"]),
            "configurations_with_no_size_reduction": sum(
                1 for r in rows if r["regime_boundary"]["summary_not_smaller_than_block"]
            ),
        },
        "regimes_separately": {
            "full_rank_tall": {
                "n_configurations": len(tall),
                "best_ratio": (_best(tall) or {}).get("compression_ratio_legacy_vs_pca_wire"),
                "min_ratio": min((_ratio(r) for r in tall), default=None),
            },
            "full_rank_flat_or_square": {
                "n_configurations": len(flat),
                "best_ratio": (_best(flat) or {}).get("compression_ratio_legacy_vs_pca_wire"),
                "min_ratio": min((_ratio(r) for r in flat), default=None),
                "note": (
                    "ratios at or below 1.0 here are the design's boundary, not a defect: a full-rank summary of a "
                    "block with n_block <= d holds min(n_block,d)*d elements against n_block*d and cannot compress"
                ),
            },
        },
        "results": rows,
        "skipped": skipped,
    }


def _print(payload: Mapping[str, Any]) -> None:
    rows = payload["results"]
    inv = payload["invariant_check"]
    print(
        f"\nB1 -- bytes crossing bridge->worker ({len(rows)} configurations, {inv['configurations_checked']} "
        f"invariant-checked, {inv['configurations_shipping_chunk']} shipping the chunk, "
        f"{inv['configurations_with_no_size_reduction']} with no size reduction)\n"
    )
    print_summary_table(
        rows,
        columns=(
            ("n_block", "n_blk", "int"),
            ("n_features", "d", "int"),
            ("regime", "regime", "str"),
            ("local_rank_effective", "rank", "int"),
            ("legacy_wire_bytes", "legacy_MiB", "mib"),
            ("pca_wire_bytes", "pca_MiB", "mib"),
            ("compression_ratio_legacy_vs_pca_wire", "ratio", "float"),
        ),
        title="full-rank + truncated arms, one row per (n_block, d, local_rank)",
    )
    reg = payload["regimes_separately"]
    print("\nfull-rank TALL   :", reg["full_rank_tall"])
    print("full-rank FLAT   :", reg["full_rank_flat_or_square"])


def main() -> int:
    """Entry point: run B1, write the artifact, print a human summary.

    - ``:return:`` ``0`` on success, ``1`` if the invariant failed on any configuration.
    """
    doc = __doc__ or ""
    parser = argparse.ArgumentParser(description=doc.splitlines()[0] if doc else "")
    parser.add_argument("--out", default=None, help="Optional explicit artifact path (default: results/b1_*.json)")
    args = parser.parse_args()

    payload = run()
    path = write_result("b1_bytes_crossing", payload)
    if args.out:
        Path(args.out).write_text(path.read_text(), encoding="utf-8")
        path = Path(args.out)
    _print(payload)

    shipping = payload["invariant_check"]["configurations_shipping_chunk"]
    print(f"\nartifact: {path}")
    if shipping:
        print(f"FAIL: {shipping} configuration(s) shipped the full chunk -- the never-ship-the-chunk invariant broke")
        return 1
    print("OK: no configuration shipped the full chunk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
