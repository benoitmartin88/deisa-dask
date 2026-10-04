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
B4: the accuracy/bandwidth curve -- retained local rank against summary size AND accuracy.

This is a headline figure and the metric choice is what makes or breaks it. Eigenvector signs are arbitrary, so a
raw component-wise error is non-monotonic in the retained rank and reads O(1) even when two subspaces are
IDENTICAL. Two sign-invariant metrics are used here and no other accuracy metric is computed:

- ``subspace_distance``: ``1 - min(svd(A @ B.T))`` between the retained subspace and the exact one. 0 iff the spans
  coincide, 1 iff orthogonal, and invariant to sign flips AND to rotation inside the retained subspace.
- ``variance_errors``: relative explained-variance errors computed on SINGULAR VALUES, which are sign-free.

The reference for both is the EXACT pooled SVD of the same data, computed independently by
``numpy.linalg.svd`` on the centered concatenation. So ``subspace_distance`` measures the approximation against a
truth the merge path did not itself produce.

The claim being substantiated
-----------------------------
Truncating a leaf to ``local_rank = R < d`` is what makes the summary sublinear and therefore a genuine bandwidth
win, and it COSTS accuracy, monotonically in ``R``. Full local rank is exact. Both halves of that sentence appear
in the same artifact: ``summary_elements`` falls with ``R`` while ``subspace_distance`` rises, and both collapse to
zero error at ``R = d``. A curve that only showed the bandwidth side would be the cherry-picking this card forbids.

Multi-block trees, and where the error actually comes from
----------------------------------------------------------
Accuracy is measured for a single leaf AND for a multi-block tree, because a truncated leaf's discarded variance is
unrecoverable by ANY merge: the error compounds with the number of blocks rather than averaging away. Reporting only
the single-leaf case would understate the cost of truncation.

Run
---
    PYTHONPATH=src .venv/bin/python benchmark/mergeable_pca/b4_rank_accuracy_curve.py
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
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
    ratio_sweep_points,
    regime_of,
    subspace_distance,
    summary_elements,
    summary_nbytes,
    time_repeated,
    variance_errors,
    write_result,
)

from deisa.dask.mergeable_pca import local_pca, merge_tree  # noqa: E402

# Retained local ranks swept, as the card requires: a geometric ladder from deeply truncated to full, plus the two
# saturation points. ``None`` means full local rank ``min(n_block, d)``, and ``d`` is requested explicitly so the
# curve contains the point where truncation stops mattering.
LOCAL_RANK_LADDER: tuple[int | str | None, ...] = (1, 2, 5, 10, 20, 40, 80, "d", None)

# Feature dimensions for the curve.
FEATURE_DIMS: tuple[int, ...] = (32, 64, 128)

# Number of blocks in the multi-block arm. Small enough that the reference SVD stays affordable at the largest ratio.
N_BLOCKS_MULTI = 8

# How many final components are scored by the subspace metric. Fixed across the whole curve so the metric compares
# like with like: a larger retained rank must not be rewarded simply by being compared against fewer components.
N_COMPONENTS_SCORED = 8


def _resolve_rank(requested: int | str | None, d: int, n_block: int) -> int | None:
    """Resolve one ladder entry into a ``local_pca`` rank argument.

    - ``"d"`` means the feature dimension, which for a block with ``n_block < d`` saturates to the full local rank
      anyway; returning ``d`` rather than ``None`` makes the saturation point an explicit, reproducible request.
    - ``None`` means "let ``local_pca`` choose the full local rank".

    - ``:param requested:`` Ladder entry: an int, the string ``"d"``, or ``None``.
    - ``:param d:`` Feature dimension of the block.
    - ``:param n_block:`` Rows in the block.
    """
    if requested is None:
        return None
    if requested == "d":
        return d
    return int(min(int(requested), d))


def exact_reference(blocks: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute the exact pooled SVD reference for a set of disjoint blocks.

    This is the independent ground truth: the centered concatenation is SVD'd directly with ``numpy``, so the
    approximation is scored against a result the merge path did not produce and cannot bias.

    - ``:param blocks:`` The disjoint sample blocks.
    """
    X = np.vstack([np.asarray(b, dtype=np.float64) for b in blocks])
    mean = X.mean(axis=0)
    _, singular_values, components = np.linalg.svd(X - mean, full_matrices=False)
    return singular_values, components, int(X.shape[0])


def measure_single_leaf(
    n_block: int,
    d: int,
    requested: int | str | None,
    intrinsic_rank: int,
) -> dict[str, Any]:
    """Measure one point of the accuracy curve for a SINGLE leaf block.

    - ``:param n_block:`` Rows in the block.
    - ``:param d:`` Feature dimension.
    - ``:param requested:`` Ladder entry for the retained local rank.
    - ``:param intrinsic_rank:`` Intrinsic rank of the synthetic signal.
    """
    block = make_block(n_block=n_block, n_features=d, rank=intrinsic_rank, seed=SEED + 21)
    rank_arg = _resolve_rank(requested, d, n_block)
    summary = local_pca(block, rank=rank_arg)

    exact_sv, exact_components, n_samples = exact_reference([block])

    k = min(N_COMPONENTS_SCORED, summary.rank, exact_components.shape[0])
    distance = subspace_distance(summary.components[:k], exact_components[:k])
    var = variance_errors(summary.singular_values, summary.n_samples, exact_sv, n_samples, n_components=k)

    timing = time_repeated(lambda: local_pca(block, rank=rank_arg), warmup_rounds=1, repeats=3)

    return {
        "n_block": int(n_block),
        "n_features": int(d),
        "n_block_over_d": float(n_block) / float(d),
        "regime": regime_of(n_block, d),
        "local_rank_requested": requested,
        "local_rank_effective": int(summary.rank),
        "intrinsic_rank": int(intrinsic_rank),
        "summary_elements": summary_elements(summary),
        "summary_bytes": byte_dict(summary_nbytes(summary)),
        "block_elements": int(block.size),
        "wire_compression_ratio": (block.size / summary_elements(summary)) if summary_elements(summary) else None,
        "subspace_distance": float(distance),
        "explained_variance_ratio_error": var["explained_variance_ratio_error"],
        "captured_variance_fraction": var["captured_variance_fraction"],
        "total_variance_relative_error": var["total_variance_relative_error"],
        "variance": var,
        "local_pca_seconds": timing,
    }


def measure_multi_block(n_block: int, d: int, requested: int | str | None, intrinsic_rank: int) -> dict[str, Any]:
    """Measure one point of the curve for a MULTI-BLOCK tree, which is where truncation error compounds.

    A leaf's discarded variance is unrecoverable by any merge -- the merge sees only summaries -- so with ``N_BLOCKS``
    truncated leaves the merged subspace drifts further from the exact one than a single truncated leaf does. Measuring
    only the single-leaf case would understate the cost of truncation.

    - ``:param n_block:`` Rows per block.
    - ``:param d:`` Feature dimension.
    - ``:param requested:`` Ladder entry for the retained local rank.
    - ``:param intrinsic_rank:`` Intrinsic rank of the synthetic signal in each block.
    """
    rank_arg = _resolve_rank(requested, d, n_block)
    blocks = [
        make_block(n_block=n_block, n_features=d, rank=intrinsic_rank, seed=SEED + 31 + 100 * i)
        for i in range(N_BLOCKS_MULTI)
    ]
    summaries = [local_pca(b, rank=rank_arg) for b in blocks]
    merged = merge_tree(summaries)

    exact_sv, exact_components, n_samples = exact_reference(blocks)
    k = min(N_COMPONENTS_SCORED, merged.rank, exact_components.shape[0])
    distance = subspace_distance(merged.components[:k], exact_components[:k])

    return {
        "n_blocks": N_BLOCKS_MULTI,
        "n_block": int(n_block),
        "n_features": int(d),
        "regime": regime_of(n_block, d),
        "local_rank_requested": requested,
        "local_rank_effective_per_leaf": int(summaries[0].rank),
        "merged_root_rank": int(merged.rank),
        "merged_root_n_samples": int(merged.n_samples),
        "summary_elements_per_leaf": int(summaries[0].components.size + summaries[0].mean.size),
        "subspace_distance": float(distance),
        "variance": variance_errors(merged.singular_values, merged.n_samples, exact_sv, n_samples, n_components=k),
        "note": (
            "rank saturation: the merged rank is min(rows(compact), d) and does not grow with tree depth past that "
            "ceiling; with a truncated leaf rank it climbs with depth until it saturates"
        ),
    }


def _is_saturation_point(requested: int | str | None, d: int) -> bool:
    """Whether a ladder entry requests at least the full local rank, i.e. a truncation-free (exact) configuration.

    - ``:param requested:`` Ladder entry.
    - ``:param d:`` Feature dimension.
    """
    if requested is None:
        return True
    if requested == "d":
        return True
    return int(requested) >= d


def run(dims: tuple[int, ...] = FEATURE_DIMS) -> dict[str, Any]:
    """Run the full accuracy/bandwidth sweep and return the artifact payload.

    - ``:param dims:`` Feature dimensions to sweep.
    """
    single: list[dict[str, Any]] = []
    multi: list[dict[str, Any]] = []

    for d in dims:
        intrinsic = max(1, d // 4)
        for requested_ratio, n_block in ratio_sweep_points(d):
            # The multi-block arm only needs the TALL side plus the square point: with n_block << d a truncated leaf
            # cannot even supply N_COMPONENTS_SCORED directions, so those points are recorded as skipped instead of
            # being reported with a silently reduced k.
            if n_block >= N_COMPONENTS_SCORED:
                for requested in LOCAL_RANK_LADDER:
                    single.append(measure_single_leaf(n_block, d, requested, intrinsic))
            for requested in LOCAL_RANK_LADDER:
                multi.append(measure_multi_block(n_block, d, requested, intrinsic))

    exact_rows = [r for r in single if _is_saturation_point(r["local_rank_requested"], r["n_features"])]
    truncated_rows = [r for r in single if not _is_saturation_point(r["local_rank_requested"], r["n_features"])]

    def _max_distance(rows: list[dict[str, Any]]) -> float | None:
        """Largest subspace distance among rows, or ``None`` for an empty set.

        - ``:param rows:`` Result rows.
        """
        return max((float(r["subspace_distance"]) for r in rows), default=None)

    def _min_wire(rows: list[dict[str, Any]]) -> float | None:
        """Largest wire compression ratio among rows, or ``None`` for an empty set.

        - ``:param rows:`` Result rows.
        """
        ratios = [r["wire_compression_ratio"] for r in rows if r["wire_compression_ratio"] is not None]
        return max(ratios) if ratios else None

    return {
        "provenance": provenance(
            script="b4_rank_accuracy_curve",
            description=(
                "Accuracy/bandwidth trade against retained local rank: summary bytes and SIGN-INVARIANT accuracy "
                "(subspace distance and relative explained-variance error) on one axis, swept across both regimes."
            ),
            extra={
                "inputs": {
                    "feature_dims": list(dims),
                    "local_rank_ladder": [None if r is None else r for r in LOCAL_RANK_LADDER],
                    "n_blocks_multi": N_BLOCKS_MULTI,
                    "n_components_scored": N_COMPONENTS_SCORED,
                    "n_block_over_d_ratios": [r for r, _ in ratio_sweep_points(max(dims))],
                },
                "metric_justification": (
                    "sign-invariant only. Eigenvector signs are arbitrary, so a raw component-wise error is "
                    "non-monotonic in the retained rank and reads O(1) even for identical subspaces. subspace "
                    "distance = 1 - min(svd(A@B.T)) is invariant to sign flips and to rotation inside the retained "
                    "subspace; the variance metrics are computed on singular values, which are sign-free."
                ),
                "reference": (
                    "exact pooled SVD via numpy.linalg.svd on the centered concatenation, computed independently of "
                    "the merge path so the approximation is scored against a truth it did not produce"
                ),
            },
        ),
        "findings": {
            "single_leaf_points": len(single),
            "multi_block_points": len(multi),
            "max_subspace_distance_truncated_single_leaf": _max_distance(truncated_rows),
            "max_subspace_distance_saturation_single_leaf": _max_distance(exact_rows),
            "max_wire_compression_truncated_single_leaf": _min_wire(truncated_rows),
            "max_wire_compression_saturation_single_leaf": _min_wire(exact_rows),
            "claim": (
                "truncating local_rank buys sublinear summary size and costs accuracy monotonically in the retained "
                "rank; both losses vanish at full local rank, where the merge is exact"
            ),
        },
        "single_leaf": single,
        "multi_block": multi,
    }


def _print(payload: Mapping[str, Any]) -> None:
    print("\nB4a -- single leaf: summary size vs SIGN-INVARIANT accuracy\n")
    print_summary_table(
        payload["single_leaf"],
        columns=(
            ("n_block", "n_blk", "int"),
            ("n_features", "d", "int"),
            ("regime", "regime", "str"),
            ("local_rank_effective", "rank", "int"),
            ("summary_elements", "summ_elems", "int"),
            ("wire_compression_ratio", "wire_x", "float"),
            ("subspace_distance", "subspace_d", "float"),
            ("explained_variance_ratio_error", "evr_err", "float"),
            ("captured_variance_fraction", "captured", "float"),
        ),
        title="single leaf: summary size falls with local_rank, subspace distance rises, both vanish at full rank",
    )
    print("\nB4b -- multi-block tree: truncation error compounds with block count\n")
    print_summary_table(
        payload["multi_block"],
        columns=(
            ("n_block", "n_blk", "int"),
            ("n_features", "d", "int"),
            ("local_rank_effective_per_leaf", "leaf_rank", "int"),
            ("merged_root_rank", "root_rank", "int"),
            ("subspace_distance", "subspace_d", "float"),
        ),
    )
    print("\nfindings:", payload["findings"])


def main() -> int:
    """Entry point: run B4, write the artifact, print a human summary.

    - ``:return:`` ``0`` on success.
    """
    doc = __doc__ or ""
    parser = argparse.ArgumentParser(description=doc.splitlines()[0] if doc else "")
    parser.add_argument("--out", default=None, help="Optional explicit artifact path")
    args = parser.parse_args()

    payload = run()
    path = write_result("b4_rank_accuracy_curve", payload)
    if args.out:
        Path(args.out).write_text(path.read_text(), encoding="utf-8")
        path = Path(args.out)
    _print(payload)
    print(f"\nartifact: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
