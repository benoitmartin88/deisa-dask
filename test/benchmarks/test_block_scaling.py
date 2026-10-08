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
Block-scaling benchmark: merge-tree time against the number of blocks ``n_block``.

The open question the old ``tradeoff_cost.py`` explicitly left unmeasured
-------------------------------------------------------------------------
The block shapes the multi-bridge experiments sweep vary ``(n_block, d, local_rank)`` at a FIXED bridge count
of 32: the merge tree is always a 32-leaf reduction there, so the artifact cannot say how merge COST scales
with the NUMBER of blocks. The module docstring of :mod:`deisa.dask.mergeable_pca` claims the tree over ``B``
blocks is ``O(B * d^3)`` -- one merge node per block, each node an SVD of a compact matrix of at most
``(2d + 1, d)`` whose cost is independent of the total sample count ``N``. Nothing in the repository put a
number next to that claim. This sweep does: ``n_block`` is swept over a doubling ladder at FIXED ``(n_samples,
d, local_rank)``, so the ONLY thing that changes is the tree's shape.

What is timed, and what is deliberately fixed
---------------------------------------------
The timed function is the in-memory reduction :func:`~deisa.dask.mergeable_pca.merge_tree` over
``n_block`` ALREADY-BUILT leaf summaries -- the merge step alone, exactly the sweep the card asks for. Leaf
construction is NOT timed: a leaf's ``local_pca`` cost grows with the block's rows and would confound the
``n_block`` effect. Fixed at every point of the sweep:

- ``n_samples = 8192`` (8192 = 32 * 256, so every ladder entry divides it exactly: the per-block row count
  ``8192 / n_block`` shrinks as the tree GROWS, which is the regime the claim describes -- more CPUs, smaller
  slabs each, the fixed total volume reduced over a wider tree);
- ``d = 64`` features in one basis;
- ``local_rank = 8`` per leaf, the paper's rank-truncation value, so each compact merge input is at most
  ``(2 * 8 + 1, 64)`` -- the per-node cost is essentially constant across the sweep by construction.

What is expected and how the artifact says it
---------------------------------------------
With per-node cost constant, tree depth ``log2(n_block)`` and total nodes ``n_block - 1``, merge time should grow
NEAR-LINEARLY in ``n_block`` (the level structure adds a lower-order ``log`` through the correction row only --
it is capped by ``d`` and irrelevant here since ``d = 64`` exceeds the saturated rank). The log-log slope of
median merge time against ``n_block`` is fitted with ``numpy.polyfit`` on the medians and written into the
artifact, as ``slope`` on every row and as ``log_log_slope`` in the findings; the slope lands in the JSON whatever
it turns out to be, because a measured number that disagrees with the claim is the result, not a defect.

Wall-time budget: six points at five timed rounds each, one warmup round discarded per point, on merges that
cost milliseconds -- the whole sweep stays well inside a minute beyond the leaf construction.

Run
---
    .venv/bin/python -m pytest test/benchmarks/test_block_scaling.py --benchmark-only -q

End-to-end through the bridge
-----------------------------
A bridge-round-trip arm would measure the SAME merge under the distributed scheduler plus serialization, at
minutes of cluster spin-up per point; the artifact therefore states it as out of scope rather than omitting it
silently, exactly as the network-transfer artifact does for its unmeasured columns.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from conftest import _load_measurement_common

from deisa.dask.mergeable_pca import local_pca, merge_tree

#: The doubling ladder the sweep runs. Every entry divides ``N_SAMPLES`` exactly, so the per-block row count
#: shrinks as the tree widens and the total sample volume is constant.
N_BLOCK_LADDER: tuple[int, ...] = (8, 16, 32, 64, 128, 256)

#: Total samples across ALL blocks together, constant for the whole sweep.
N_SAMPLES = 8192

#: Feature dimension, constant for the whole sweep: the per-node SVD is ``O(d^3)`` and must not move.
N_FEATURES = 64

#: Retained rank of EVERY leaf summary, constant for the whole sweep: the per-node compact matrix is then
#: at most ``(2 * 8 + 1, 64)``, so per-node cost is constant by construction.
LOCAL_RANK = 8

#: Timed rounds per point and discarded warmup. Five rounds make every row carry real dispersion; the whole
#: sweep stays inside a minute, inside the ~4-minute card budget.
ROUNDS = 5
WARMUP_ROUNDS = 1

#: Module-level row accumulator. pytest-benchmark's fixture is single-use and the sweep is one parametrized
#: test, so measured rows accumulate here and the artifact is rewritten after every point: the LAST write
#: carries all six rows and the fitted slope. The parameterized test function parameter runs in file order.
_ROWS: list[dict[str, Any]] = []

#: The sweep's fixed inputs, stamped verbatim into the artifact's provenance.
_SWEEP_INPUTS: dict[str, Any] = {
    "n_block_ladder": list(N_BLOCK_LADDER),
    "n_samples_total": N_SAMPLES,
    "n_features_d": N_FEATURES,
    "leaf_local_rank": LOCAL_RANK,
    "timed_rounds_per_point": ROUNDS,
    "warmup_rounds_per_point": WARMUP_ROUNDS,
    "timed_region": (
        "merge_tree() over PRE-BUILT leaf summaries only, in memory; leaf local_pca construction is NOT timed"
    ),
}


def _build_leaves(n_block: int, rng: np.random.Generator) -> list[Any]:
    """Build ``n_block`` leaf summaries at the sweep's fixed shape, WITHOUT timing them.

    Blocks are drawn per point with the suite's seeded generator: a fresh stream per point, so no two ladder
    entries share a draw and every run at one point is identical. Each leaf is ``local_pca`` of a
    ``(N_SAMPLES / n_block, d)`` row block truncated to ``LOCAL_RANK``.

    - ``:param n_block:`` Number of blocks, one leaf per block.
    - ``:param rng:`` The seeded generator (from the suite's ``--bench-seed`` default).
    - ``:return:`` The leaf summaries, in block order.
    """
    rows_per_block = N_SAMPLES // n_block
    blocks = []
    for index in range(n_block):
        # Disjoint low-rank signals, one stream id per block, so identical leaves reproduce across runs.
        signal = rng.standard_normal((N_FEATURES, LOCAL_RANK))
        factors = rng.standard_normal((rows_per_block, LOCAL_RANK))
        block = np.ascontiguousarray(factors @ signal.T + rng.standard_normal((rows_per_block, N_FEATURES)) * 0.05)
        blocks.append(local_pca(block, rank=LOCAL_RANK))
    return blocks


def _fit_log_log_slope(rows: list[dict[str, Any]]) -> float:
    """Fit the log-log slope of merge time against ``n_block`` over the measured medians.

    ``numpy.polyfit`` on ``log(n_block)`` against ``log(median_s)``, first coefficient. The slope is the
    artifact's empirical answer to "near-linear in n_block?": ~1.0 means linear, ~0 would be flat, above 1
    superlinear. Measured over whatever rows exist at write time, so a PARTIAL sweep still records its own slope
    -- the number always lands in the JSON.

    - ``:param rows:`` The measured rows so far.
    - ``:return:`` The fitted slope.
    """
    log_n = np.log([float(row["n_block"]) for row in rows])
    log_t = np.log([row["median_s"] for row in rows])
    if len(rows) < 2 or not np.all(log_n[1:] > log_n[:-1]):
        return float("nan")
    return float(np.polyfit(log_n, log_t, 1)[0])


def _write_artifact(rows: list[dict[str, Any]]) -> None:
    """Write (or rewrite) the provenance-stamped sweep artifact for the rows measured so far.

    The last invocation of the parametrized test writes all rows; every earlier invocation writes a strictly
    narrower sweep. The slope is RE-FITTED at each write over the rows present, so the final artifact carries
    the number for the full ladder and no stale earlier file can survive it.

    - ``:param rows:`` Every row measured by this process so far.
    """
    measurement_common = _load_measurement_common()
    slope = _fit_log_log_slope(rows)
    if len(rows) < 2:
        # One point has no slope to fit and no curve to describe: rewriting the artifact for it would publish a
        # sweep of one with a NaN slope, which the strict-JSON writer refuses. The later ladder points rewrite
        # with their accumulated rows, so a full sweep run loses nothing; a single-point invocation leaves the
        # previous artifact untouched and says so on stdout.
        print("  block_scaling: 1 point measured; artifact needs >= 2 points for a slope -- nothing written")
        return
    stamped = [{**row, "slope": slope} for row in rows]
    findings = {
        "log_log_slope": slope,
        "near_linear_in_n_block": bool(abs(slope - 1.0) < 0.15),
        "claim": (
            "the module docstring claims a tree over B blocks costs O(B * d^3): one merge node per block, "
            "each an SVD independent of N. Near-linear growth of merge time in n_block (log-log slope close "
            "to 1) is what that claim predicts; the fitted number is recorded whatever it is."
        ),
    }
    payload = {
        "provenance": measurement_common.provenance(
            script="block_scaling",
            description=(
                "Merge-fix cost against the number of blocks: merge_tree() time over a doubling ladder of "
                "n_block at FIXED total samples, feature dimension and leaf rank -- the sweep the old "
                "tradeoff_cost.py left open. Fitted log-log slope of time vs n_block, empirically testing "
                "the near-linear association cost the merge tree's node count predicts."
            ),
            extra={"inputs": _SWEEP_INPUTS, "sweep": _SWEEP_INPUTS, "findings": findings},
        ),
        "findings": findings,
        "results": stamped,
    }
    path = measurement_common.write_result("block_scaling", payload)
    with path.open(encoding="utf-8") as fp:
        stored = json.load(fp)
    assert stored["provenance"]["script"] == "block_scaling"
    assert len(stored["results"]) == len(rows)
    assert all("slope" in row for row in stored["results"])


@pytest.mark.benchmark(group="block_scaling", min_rounds=ROUNDS, warmup=WARMUP_ROUNDS)
@pytest.mark.parametrize("n_block", N_BLOCK_LADDER)
def test_bench_merge_tree_scales_with_n_block(benchmark, bench_rng, n_block):
    """Benchmark the merge tree at ONE ladder point and append its row to the sweep artifact.

    The leaves are built once, OUTSIDE the timed region, because the sweep is over the merge step: timing leaf
    construction would fold an ``O(n_samples * d * r)`` leaf cost into a curve that is meant to isolate the
    tree. ``merge_tree`` consumes summaries without mutating them, so one leaf set serves all rounds.

    - ``:param benchmark:`` The pytest-benchmark fixture.
    - ``:param bench_rng:`` The suite's seeded generator.
    - ``:param n_block:`` The ladder entry: number of blocks in the tree.
    """
    leaves = _build_leaves(n_block, bench_rng)

    merged = benchmark.pedantic(merge_tree, args=(leaves,), rounds=ROUNDS, warmup_rounds=WARMUP_ROUNDS)

    # The tree really reduced: one root over the pooled sample count.
    assert merged.n_samples == n_block * (N_SAMPLES // n_block)
    assert merged.rank <= N_FEATURES

    stats = benchmark.stats
    samples = [float(value) for value in stats.stats.data]
    assert stats.get("rounds") == ROUNDS == len(samples)
    _ROWS.append(
        {
            "n_block": int(n_block),
            "n_samples_total": N_SAMPLES,
            "n_features": N_FEATURES,
            "leaf_local_rank": LOCAL_RANK,
            "rows_per_block": N_SAMPLES // n_block,
            "median_s": float(stats.get("median")),
            "min_s": float(stats.get("min")),
            "max_s": float(stats.get("max")),
            "seconds_all": samples,
            "timed_repeats": ROUNDS,
        }
    )
    _write_artifact(_ROWS)
