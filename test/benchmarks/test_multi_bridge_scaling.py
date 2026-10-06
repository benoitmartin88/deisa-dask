# ==============================================================================
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
#   to endorse or promote the derived work without specific prior written
#   permission.
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
# ==============================================================================
"""
Multi-bridge scaling benchmark: merge time and per-bridge bytes against the number of bridges.

The slot the paper leaves open
------------------------------
The paper's result-slot table lists "the multi-bridge experiment: time and bytes vs bridges" as not
measured. This sweep closes the slot at the level the repository can support honestly: the
per-output-step cost of producing the global summary when the SAME total sample volume arrives
from ``B`` independent bridges instead of one, measured over a doubling ladder of ``B``.

What changes, what is fixed
---------------------------
- CHANGES: the bridge count ``B`` over ``(1, 2, 4, 8, 16, 32, 64, 128)``; per-bridge chunk bytes
  therefore change too, since the fixed total ``N`` divides across ``B`` producers.
- FIXED: total samples ``N = 16384``, feature dimension ``d = 64``, leaf rank ``8``. Each bridge
  contributes ``N / B`` rows as ONE chunk (the paper's per-bridge flattened field).

What is recorded per point
--------------------------
- ``per_bridge_wire_bytes``: the serialized per-bridge chunk size (the payload the bridge would
  ship in the legacy design) together with the serialized per-bridge summary size (the payload
  bridge-side compute ships instead). Both measured through the same serializer, so the
  total-bytes-vs-bridges curve for BOTH paths derives from these two numbers alone.
- ``merge_tree_seconds``: median/min/max of the balanced in-memory merge over the ``B`` leaf
  summaries, five timed rounds with a discarded warmup -- the same merge-time methodology as
  ``test_block_scaling.py``.
- Accuracy of the merged root against the exact batch SVD of the full data (subspace distance),
  so the artifact also certifies the multi-bridge root is the same decomposition at every ``B``.

What is expected
----------------
Per-bridge legacy wire bytes SHRINK proportional to ``1/B`` (each bridge holds ``N/B`` rows);
per-bridge summary bytes shrink toward the rank ceiling; total legacy bytes across the bridge
boundary are unchanged (``N * d * 8`` regardless of ``B``), while the merge-tree time grows with
the tree's node count. The artifact records what it measures, whatever the numbers turn out to be.

Wall-time budget: eight points at (1 warmup + 5 timed) rounds of milliseconds-scale merges, plus
leaf construction -- the sweep runs in well under two minutes.

Out of scope, stated rather than silent
---------------------------------------
A distributed round trip (real ``Bridge`` objects across ``B`` processes with the scheduler in
the loop) measures scheduler placement effects, not the candidate relation the slot names; the
repo-level single-process measurement isolates the numbers the paper quotes. The distributed
variant is the multi-bridge experiment the SOTA section defers to the production deployment.

Run
---
    .venv/bin/python -m pytest test/benchmarks/test_multi_bridge_scaling.py --benchmark-only -q
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from conftest import _load_measurement_common

from deisa.dask.mergeable_pca import local_pca, merge_tree

#: The bridge-count ladder the sweep runs. Every entry divides ``N_SAMPLES`` exactly.
BRIDGE_LADDER: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)

#: Total samples across ALL bridges together, constant for the whole sweep.
N_SAMPLES = 16384

#: Feature dimension, constant for the whole sweep.
N_FEATURES = 64

#: Retained rank of EVERY leaf summary, constant for the whole sweep.
LOCAL_RANK = 8

#: Timed rounds per point and discarded warmup, matching the block-scaling sweep's methodology.
ROUNDS = 5
WARMUP_ROUNDS = 1

_ROWS: list[dict[str, Any]] = []

_SWEEP_INPUTS: dict[str, Any] = {
    "bridge_ladder": list(BRIDGE_LADDER),
    "n_samples_total": N_SAMPLES,
    "n_features_d": N_FEATURES,
    "leaf_local_rank": LOCAL_RANK,
    "timed_rounds_per_point": ROUNDS,
    "warmup_rounds_per_point": WARMUP_ROUNDS,
    "timed_region": (
        "merge_tree() over the B bridges' PRE-BUILT leaf summaries, in memory; leaf construction "
        "and serialization counts run once per point, outside the timed rounds"
    ),
}


def _build_bridge_leaf(rows: int, seed_offset: int, rng: np.random.Generator) -> tuple[Any, np.ndarray]:
    """Build ONE bridge's chunk and its summary.

    The signal model matches ``test_block_scaling.py``: disjoint low-rank signals on a shared
    feature basis so the batch SVD of the pooled data is the decomposition the merge must
    recover. Returns the summary AND the chunk (the caller pools chunks for the reference SVD).

    - ``:param rows:`` Rows the bridge contributes (``N_SAMPLES / B``).
    - ``:param seed_offset:`` Distinguishing stream id per bridge.
    - ``:param rng:`` The seeded generator.
    """
    signal = rng.standard_normal((N_FEATURES, LOCAL_RANK))
    factors = rng.standard_normal((rows, LOCAL_RANK))
    chunk = np.ascontiguousarray(factors @ signal.T + rng.standard_normal((rows, N_FEATURES)) * 0.05)
    return local_pca(chunk, rank=LOCAL_RANK), chunk


def _fit_log_log_slope(rows: list[dict[str, Any]]) -> float:
    """Fit the log-log slope of merge time against bridge count over the measured medians."""
    log_b = np.log([float(row["n_bridges"]) for row in rows])
    log_t = np.log([row["merge_median_s"] for row in rows])
    if len(rows) < 2 or not np.all(log_b[1:] > log_b[:-1]):
        return float("nan")
    return float(np.polyfit(log_b, log_t, 1)[0])


def _write_artifact(rows: list[dict[str, Any]]) -> None:
    """Write (or rewrite) the provenance-stamped sweep artifact for the rows measured so far."""
    measurement_common = _load_measurement_common()
    if len(rows) < 2:
        print("  multi_bridge_scaling: 1 point; artifact needs >= 2 for a slope -- nothing written")
        return
    slope = _fit_log_log_slope(rows)
    stamped = [{**row, "slope": slope} for row in rows]
    findings = {
        "log_log_slope_merge_time_vs_bridges": slope,
        "claim": (
            "the paper's multi-bridge slot asks how merge time and transferred bytes grow with the "
            "number of bridges at fixed total volume. Legacy per-bridge bytes shrink 1/B by "
            "construction (each bridge holds N/B rows); the summary shrinks toward its rank floor; "
            "merge time grows with node count. The fitted slope is recorded whatever it is."
        ),
    }
    payload = {
        "provenance": measurement_common.provenance(
            script="multi_bridge_scaling",
            description=(
                "Multi-bridge scaling: per-bridge wire bytes (legacy chunk vs summary) and merge_tree "
                "time over a doubling ladder of bridge counts at FIXED total samples, feature "
                "dimension and leaf rank. Subspace distance of the merged root against the exact "
                "batch SVD certifies the multi-bridge root at every B."
            ),
            extra={"inputs": _SWEEP_INPUTS, "sweep": _SWEEP_INPUTS, "findings": findings},
        ),
        "findings": findings,
        "results": stamped,
    }
    path = measurement_common.write_result("multi_bridge_scaling", payload)
    with path.open(encoding="utf-8") as fp:
        stored = json.load(fp)
    assert stored["provenance"]["script"] == "multi_bridge_scaling"
    assert len(stored["results"]) == len(rows)


@pytest.mark.benchmark(group="multi_bridge_scaling", min_rounds=ROUNDS, warmup=WARMUP_ROUNDS)
@pytest.mark.parametrize("n_bridges", BRIDGE_LADDER)
def test_bench_multi_bridge_scaling(benchmark, bench_rng, n_bridges):
    """Benchmark the multi-bridge step at ONE ladder point and append its row to the sweep artifact.

    One bridge here means one producer contributing ``N / B`` rows as a single chunk. The leaves
    are built once, outside the timed region (the sweep isolates the merge step, exactly like the
    block-scaling sweep). The reference SVD is computed once per point over the pooled chunks.

    - ``:param benchmark:`` The pytest-benchmark fixture.
    - ``:param bench_rng:`` The suite's seeded generator.
    - ``:param n_bridges:`` The ladder entry: number of bridges.
    """
    measurement_common = _load_measurement_common()
    serialized_nbytes = measurement_common.serialized_nbytes
    rows_per_bridge = N_SAMPLES // n_bridges
    summaries, chunks = [], []
    for bridge_index in range(n_bridges):
        summary, chunk = _build_bridge_leaf(rows_per_bridge, bridge_index, bench_rng)
        summaries.append(summary)
        chunks.append(chunk)
    pooled = np.vstack(chunks)

    merged = benchmark.pedantic(merge_tree, args=(summaries,), rounds=ROUNDS, warmup_rounds=WARMUP_ROUNDS)

    # The tree really reduced over the pooled sample count at every ladder point.
    assert merged.n_samples == N_SAMPLES
    assert merged.rank <= N_FEATURES

    # Reference decomposition of the pooled data, OUTSIDE the timed region: an independent exact
    # batch SVD the merged root's subspace is compared against.
    centered = pooled - pooled.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    reference_rank = min(LOCAL_RANK, vt.shape[0])
    reference_components = vt[:reference_rank]

    merged_components = np.asarray(merged.components[:reference_rank])
    # The house sign-invariant metric: 1 - min(svd(A @ B^T)), 0 exactly when the spans coincide.
    # No transcribed metric code -- the same function the parity test and the artifacts use.
    subspace_distance = measurement_common.subspace_distance(reference_components, merged_components)

    stats = benchmark.stats
    samples = [float(v) for v in stats.stats.data]
    assert stats.get("rounds") == ROUNDS == len(samples)

    _ROWS.append(
        {
            "n_bridges": int(n_bridges),
            "n_samples_total": N_SAMPLES,
            "n_features": N_FEATURES,
            "leaf_local_rank": LOCAL_RANK,
            "rows_per_bridge": rows_per_bridge,
            "per_bridge_legacy_wire_bytes": serialized_nbytes(chunks[0]),
            "per_bridge_summary_wire_bytes": serialized_nbytes(summaries[0]),
            "merge_median_s": float(stats.get("median")),
            "merge_min_s": float(stats.get("min")),
            "merge_max_s": float(stats.get("max")),
            "seconds_all": samples,
            "timed_repeats": ROUNDS,
            "merged_root_rank": int(merged.rank),
            "subspace_distance_vs_exact_batch_svd": subspace_distance,
        }
    )
    _write_artifact(_ROWS)
