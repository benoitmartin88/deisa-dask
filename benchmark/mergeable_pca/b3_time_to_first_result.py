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
B3: time-to-first-result per timestep, INCLUDING the local SVD cost the bridge now pays.

The honest claim is a TRADE, not a free win, and this script exists so the cost side cannot be hidden. Moving the
decomposition to the bridge buys bandwidth at the price of an ``O(n_block * d)`` local SVD that the legacy
full-chunk scatter never performed: the legacy path ships bytes and lets a worker do whatever it wants, while the
PCA path computes before it ships. Every row therefore reports FOUR separate quantities, never a single blended
number:

- ``local_svd_seconds``: the ``local_pca`` call alone -- the NEW cost the design introduces.
- ``merge_tree_seconds``: the ``O(B * d^3)`` reduction -- independent of the total sample count N.
- ``legacy_scatter_seconds``: the cost of the path the design replaces.
- ``pca_pipeline_seconds``: ``local_svd + merge_tree + serialized scatter size``, the whole bridge-side pipeline.

and the verdict column ``faster_than_legacy`` is computed from those, so the direction of the trade is a measurement
rather than an assertion. On this machine the local SVD usually dominates and the pipeline can LOSE; where that
happens the artifact says so, because a benchmark that only reported the wins would misrepresent the design.

What is deliberately NOT claimed
--------------------------------
This is an in-process measurement of the computation and the payload size, not of the network. The wall-clock of a
real ``scatter_to_workers`` over a socket depends on the interconnect, which this box does not reproduce. So the
transfer term is charged at the MEASURED SERIALIZED BYTE COUNT as a bandwidth-scaled quantity
(``serialized_bytes / assumed_bytes_per_second``) and the assumed rate is an explicit, overridable input recorded in
the artifact -- never a silently chosen constant. What the reader gets is the crossover rate at which the trade
flips, which is machine-independent and is the honest way to state this result.

Run
---
    PYTHONPATH=src .venv/bin/python benchmark/mergeable_pca/b3_time_to_first_result.py
    PYTHONPATH=src .venv/bin/python benchmark/mergeable_pca/b3_time_to_first_result.py --mbps 125
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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
    summary_nbytes,
    time_repeated,
    write_result,
)

from deisa.dask.mergeable_pca import local_pca, merge_tree  # noqa: E402

# Feature dimensions for the timing sweep.
FEATURE_DIMS: tuple[int, ...] = (32, 128, 512)

# Blocks in the tree.
N_BLOCKS = 8

# Default assumed link rate, in megabits per second, used ONLY to turn measured bytes into a transfer-time estimate.
# It is an INPUT recorded verbatim in the artifact and overridable with --mbps. A 1000 Mbit/s figure is a
# conservative 1 Gb Ethernet-class link; the point is that the reader can substitute their own and the crossover rate
# is reported independently of it.
DEFAULT_MBPS = 1000.0

# Local ranks timed: full (the exact path) and two truncations, since the SVD cost is what truncation buys down.
LOCAL_RANKS: tuple[int | None, ...] = (8, 32, None)


def measure(n_block: int, d: int, local_rank: int | None, intrinsic_rank: int, mbps: float) -> dict[str, Any]:
    """Time every stage of both pipelines for ONE configuration.

    Each stage is timed independently so the trade is attributable, and the composite numbers are sums of the
    measured medians rather than a separately-timed blend (which would double-count harness noise).

    - ``:param n_block:`` Rows per block.
    - ``:param d:`` Feature dimension.
    - ``:param local_rank:`` Retained local rank, or ``None`` for full local rank.
    - ``:param intrinsic_rank:`` Intrinsic rank of the synthetic signal.
    - ``:param mbps:`` Assumed link rate in Mbit/s for the transfer-time estimate.
    """
    blocks = [
        make_block(n_block=n_block, n_features=d, rank=intrinsic_rank, seed=SEED + 41 + 100 * i)
        for i in range(N_BLOCKS)
    ]

    rank_arg = local_rank if local_rank is None else min(int(local_rank), d)
    summary_timing = time_repeated(lambda: [local_pca(b, rank=rank_arg) for b in blocks], warmup_rounds=1, repeats=3)
    summaries = [local_pca(b, rank=rank_arg) for b in blocks]
    merge_timing = time_repeated(lambda: merge_tree(summaries), warmup_rounds=1, repeats=3)

    # The legacy path's compute is only what it takes to get the bytes into a payload; the design under test replaces a
    # network transfer of the full blocks with a network transfer of summaries, so the legacy compute arm is the
    # baseline constant rather than an SVD it never performs.
    legacy_payload = blocks[0]
    legacy_bytes = serialized_nbytes(legacy_payload) * N_BLOCKS
    pca_bytes = sum(serialized_nbytes(s) for s in summaries)
    legacy_scatter_timing = time_repeated(lambda: serialized_nbytes(legacy_payload), warmup_rounds=1, repeats=3)

    bytes_per_second = mbps * 1e6 / 8.0
    legacy_transfer_s = legacy_bytes / bytes_per_second
    pca_transfer_s = pca_bytes / bytes_per_second

    summary_elems = sum(s.components.size + s.mean.size + s.singular_values.size for s in summaries)
    block_elems = sum(int(b.size) for b in blocks)

    legacy_prepare_s = legacy_scatter_timing["seconds_median"] * N_BLOCKS
    local_svd_s = summary_timing["seconds_median"]
    merge_s = merge_timing["seconds_median"]
    pca_compute_s = local_svd_s + merge_s

    return {
        "n_block": int(n_block),
        "n_features": int(d),
        "n_block_over_d": float(n_block) / float(d),
        "regime": regime_of(n_block, d),
        "n_blocks": N_BLOCKS,
        "local_rank_requested": local_rank,
        "local_rank_effective": int(summaries[0].rank),
        "intrinsic_rank": int(intrinsic_rank),
        "block_elements_total": block_elems,
        "summary_elements_total": summary_elems,
        "legacy_bytes": byte_dict(legacy_bytes),
        "pca_bytes": byte_dict(pca_bytes),
        "compression_ratio_legacy_vs_pca_wire": ratio_or_none(legacy_bytes, pca_bytes),
        "local_svd_seconds_median": local_svd_s,
        "merge_tree_seconds_median": merge_s,
        "local_svd_seconds": summary_timing,
        "merge_tree_seconds": merge_timing,
        "legacy_prepare_seconds": legacy_scatter_timing,
        "legacy_prepare_seconds_median": legacy_prepare_s,
        "legacy_transfer_seconds_estimate": legacy_transfer_s,
        "pca_transfer_seconds_estimate": pca_transfer_s,
        "legacy_compute_seconds_estimate": legacy_prepare_s,
        "pca_compute_seconds_estimate": pca_compute_s,
        "legacy_total_seconds_estimate": legacy_prepare_s + legacy_transfer_s,
        "pca_total_seconds_estimate": pca_compute_s + pca_transfer_s,
        "speedup_estimate": ratio_or_none(legacy_prepare_s + legacy_transfer_s, pca_compute_s + pca_transfer_s),
        "local_svd_share_of_pca_pipeline": ratio_or_none(local_svd_s, pca_compute_s + pca_transfer_s),
        "pca_faster_than_legacy": bool((pca_compute_s + pca_transfer_s) < (legacy_prepare_s + legacy_transfer_s)),
        "summary_bytes_raw": summary_nbytes(summaries[0]),
    }


def crossover_mbps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rate at which the PCA pipeline's byte saving exactly pays for its local SVD.

    Solves ``local_svd + merge + (pca_bytes / r) == legacy_prepare + (legacy_bytes / r)`` for the link rate ``r``,
    giving ``r == (legacy_bytes - pca_bytes) / (local_svd + merge - legacy_prepare)``. Above that rate the byte saving
    dominates and the PCA path wins; below it the local SVD dominates and it loses. Reported per configuration because
    it depends on the shape, and it is the machine-independent way to state this trade.

    - ``:param rows:`` Result rows from :func:`measure`.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        saved_bytes = row["legacy_bytes"]["bytes"] - row["pca_bytes"]["bytes"]
        extra_compute = (
            row["local_svd_seconds"]["seconds_median"]
            + row["merge_tree_seconds"]["seconds_median"]
            - row["legacy_prepare_seconds"]["seconds_median"]
        )
        rate_bps = (saved_bytes / extra_compute) if extra_compute > 0 else None
        out.append(
            {
                "n_block": row["n_block"],
                "n_features": row["n_features"],
                "regime": row["regime"],
                "local_rank_requested": row["local_rank_requested"],
                "bytes_saved": saved_bytes,
                "extra_compute_seconds": extra_compute,
                "crossover_mbps": None if rate_bps is None else rate_bps * 8.0 / 1e6,
                "interpretation": (
                    "link rate above which the byte saving dominates the local SVD cost; None when the design does no "
                    "byte saving (flat regime at full rank) or when it costs no extra compute"
                ),
            }
        )
    return out


def run(dims: tuple[int, ...] = FEATURE_DIMS, mbps: float = DEFAULT_MBPS) -> dict[str, Any]:
    """Run the full timing sweep and return the artifact payload.

    - ``:param dims:`` Feature dimensions to sweep.
    - ``:param mbps:`` Assumed link rate in Mbit/s, recorded in the artifact.
    """
    rows: list[dict[str, Any]] = []
    for d in dims:
        intrinsic = max(1, d // 4)
        for _, n_block in ratio_sweep_points(d):
            if n_block * d * N_BLOCKS > 24_000_000:
                continue
            for local_rank in LOCAL_RANKS:
                rows.append(measure(n_block, d, local_rank, intrinsic, mbps))

    wins = [r for r in rows if r["pca_faster_than_legacy"]]
    losses = [r for r in rows if not r["pca_faster_than_legacy"]]

    return {
        "provenance": provenance(
            script="b3_time_to_first_result",
            description=(
                "Time-to-first-result per timestep for the bridge-side PCA pipeline versus the legacy full-chunk "
                "scatter, reporting the local SVD cost the design newly pays rather than hiding it."
            ),
            extra={
                "inputs": {
                    "feature_dims": list(dims),
                    "n_blocks": N_BLOCKS,
                    "local_ranks": [None if r is None else int(r) for r in LOCAL_RANKS],
                    "assumed_link_mbps": mbps,
                },
                "what_is_measured": (
                    "local_pca, merge_tree and the serialized payload size are measured in-process; the NETWORK "
                    "transfer is NOT measured on this box and is estimated from measured bytes divided by the assumed "
                    "link rate above. No number here claims a real interconnect was exercised."
                ),
                "the_trade": (
                    "the PCA path pays an O(n_block*d) local SVD the legacy path never performs. It wins only when the "
                    "byte saving dominates that cost; crossover_mbps reports the rate at which that flips."
                ),
            },
        ),
        "verdict": {
            "configurations_measured": len(rows),
            "pca_faster": len(wins),
            "pca_slower": len(losses),
            "honest_summary": (
                "the design is a TRADE, not a free win. Where pca_slower is non-zero the local SVD cost exceeds the "
                "byte saving at the assumed link rate, and those rows are kept in the artifact rather than dropped."
            ),
        },
        "results": rows,
        "crossover": crossover_mbps(rows),
    }


def _print(payload: Mapping[str, Any]) -> None:
    print("\nB3 -- time to first result: legacy full-chunk scatter vs bridge-side PCA\n")
    print_summary_table(
        payload["results"],
        columns=(
            ("n_block", "n_blk", "int"),
            ("n_features", "d", "int"),
            ("regime", "regime", "str"),
            ("local_rank_effective", "rank", "int"),
            ("local_svd_seconds_median", "local_svd_s", "float"),
            ("merge_tree_seconds_median", "merge_s", "float"),
            ("legacy_transfer_seconds_estimate", "legacy_xfer_s", "float"),
            ("pca_transfer_seconds_estimate", "pca_xfer_s", "float"),
            ("legacy_total_seconds_estimate", "legacy_tot_s", "float"),
            ("pca_total_seconds_estimate", "pca_tot_s", "float"),
            ("speedup_estimate", "speedup", "float"),
            ("pca_faster_than_legacy", "wins", "str"),
        ),
        title="transfer times are ESTIMATES from measured bytes at the assumed link rate; compute times are measured",
    )
    print("\nverdict:", payload["verdict"])


def main() -> int:
    """Entry point: run B3, write the artifact, print a human summary.

    - ``:return:`` ``0`` on success.
    """
    doc = __doc__ or ""
    parser = argparse.ArgumentParser(description=doc.splitlines()[0] if doc else "")
    parser.add_argument("--mbps", type=float, default=DEFAULT_MBPS, help="Assumed link rate in Mbit/s (recorded)")
    parser.add_argument("--out", default=None, help="Optional explicit artifact path")
    args = parser.parse_args()

    payload = run(mbps=args.mbps)
    path = write_result("b3_time_to_first_result", payload)
    if args.out:
        Path(args.out).write_text(path.read_text(), encoding="utf-8")
        path = Path(args.out)
    _print(payload)
    print(f"\nartifact: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
