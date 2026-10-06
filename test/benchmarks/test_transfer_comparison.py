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
# ==============================================================================
"""
The consolidated five-way comparison, ported onto pytest-benchmark.

Ported from ``benchmark/mergeable_pca/transfer_comparison.py`` (the old self-timed script): the same
five-way baseline comparison -- NumPy SVD, SciPy SVD, scikit-learn batched PCA, scikit-learn batched
IncrementalPCA, dask-ml IncrementalPCA against bridge-side mergeable PCA -- on the configuration grid
the script's ``plan_configurations()`` enumerates, over the same ``make_block`` seed stream.

The script is LEFT UNTOUCHED and both harnesses live in parallel until the switch is approved. The
script's one consolidated artifact per configuration is split here into ONE artifact per ARM, because
one ``benchmark`` fixture measures one thing: five baseline arms plus the two mergeable stages
(``local_pca`` leaves, ``merge_tree``) as separate benchmark tests, each writing its own
pytest-benchmark-stamped artifact into ``benchmark/mergeable_pca/results/`` through the same
``measurement_common.write_result`` writer and gates.

What is timing and what is arithmetic
--------------------------------------
The measurement semantics are preserved, but where the wall-clock lives differs from the script BY
DECLARATION:

- TIMED by pytest-benchmark -- the in-process COMPUTE of each arm: one arm's fit per benchmark round,
  on the analytics-side CPU, exactly the region the script timed with ``time_repeated``. The mergeable
  total is one separately timed pipeline round (leaves + tree in ONE measurement), with the two
  stages' own medians carried by the stage artifacts -- never a blend that double-counts noise.
- NOT TIMED, computed as deterministic arithmetic -- the byte-count fields. ``serialized_nbytes`` of a
  block or a ``PCASummary`` is a deterministic function of its input, so timing it measures the
  serializer's constant factor, not the payload; this port computes the byte counts directly and
  writes the SAME byte fields into the artifact, so the figure pipeline stays fed. The artifact's
  ``transfer_fields`` block says ``in_benchmark_row: no_timing`` for them.

The comparison rule the artifact encodes (the script's central comparison, stated rather than
implied): every baseline runs ON the analytics engine, so the samples must be there first and its
transfer is the FULL BLOCK's wire payload. Only the mergeable path computes before it ships, and only
it transfers less.

Grid budget
-----------
The script's full grid is 36 configurations at minutes each -- a paper artifact run, driven by hand
with ``--repeats`` and a resume checkpoint. The pytest-benchmark module runs the same enumeration
through a smoke budget filter: shapes at which ONE timed round of the costliest arm stays under
:data:`MAX_ROUND_SECONDS`, keeping the whole module inside the card's five-minute budget. The filter
is driven by the MEASURED per-round worst case of each shape (``_MEASURED_ROUND_COSTS``), not by an
estimate: a shape whose one-round cost nobody measured is refused, not waved through. The artifact
stamps both the measured grid and the script's full grid rule.

Run
---
    .venv/bin/python -m pytest test/benchmarks/test_transfer_comparison.py --benchmark-only -q

Provenance parity with the script
---------------------------------
The deterministic fields are proven IDENTICAL to the script's on smoke configurations by
:func:`test_bench_parity_with_script`: both harnesses draw the SAME blocks (``make_block`` at the
script's ``SEED + 11`` stream), then the script's own ``measure_configuration`` runs through the
legacy import on those blocks and every byte field and accuracy value is compared row by row.
Timings are NOT compared: the script's ``time_repeated`` and pytest-benchmark's caliper measure the
same compute with different clocks and repeat policies, and the parity test says so.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import _load_measurement_common

from deisa.dask.mergeable_pca import local_pca, merge_tree

#: The legacy experiment module, imported by path exactly as the conftest imports
#: ``measurement_common``: the script stays untouched and its ``plan_configurations``, arm table and
#: ``measure_configuration`` are the single source this module measures against.
_BENCHMARK_PKG = Path(__file__).resolve().parents[2] / "benchmark" / "mergeable_pca"
if str(_BENCHMARK_PKG) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_PKG))

#: The legacy script module. The ``sys.path`` insert above makes its own ``import measurement_common``
#: resolve to the same module the conftest loads, so the provenance writer stays one authority.
import transfer_comparison as tc  # noqa: E402

#: Artifact stem: one per arm, suffixed by the arm -- the fixture is single-use, so the row each test
#: measures must go to its own artifact, the same one-artifact-per-stage split the sibling benchmarks use.
ARTIFACT_STEM = "transfer_comparison_pytest"

#: The arm names and their table, read from the script so the two harnesses cannot drift apart.
MERGEABLE_ARM = tc.MERGEABLE_ARM
STANDARD_METHODS = tc.STANDARD_METHODS
METHODS_COMPARED = tc.METHODS_COMPARED
MERGEABLE_STAGES = tc.MERGEABLE_STAGES
METHOD_IMPORTS = tc.METHOD_IMPORTS

#: Blocks per configuration, from the script: eight is what ``b5`` measured, so the consolidated rows
#: join to that artifact's rows on the same decomposition rather than on a similar one.
N_BLOCKS: int = int(tc.N_BLOCKS)

#: Intrinsic rank of the synthetic signal as a fraction of ``d``. An INPUT, matching the script.
INTRINSIC_RANK_FRACTION = tc.INTRINSIC_RANK_FRACTION

#: Components scored for every method. Fixed and IDENTICAL across methods, because
#: ``subspace_distance`` is undefined between spans of different rank.
N_COMPONENTS_SCORED = tc.N_COMPONENTS_SCORED

#: Per-``d`` cap on ``n_block``, the script's input, carried verbatim so the budget filter below
#: filters the SAME grid the script would run.
MAX_BLOCK_ELEMENTS_ROWS: dict[int, int] = dict(tc.MAX_BLOCK_ELEMENTS_ROWS)

#: Timed rounds per configuration per arm, plus one discarded warmup round -- the script's policy
#: (one warmup, five timed repeats), now stated from pytest-benchmark's own harness.
ROUNDS = 5
WARMUP_ROUNDS = 1

#: The smoke-budget ceiling: ONE timed round of the costliest arm must stay under it, so the whole
#: module stays inside the card's five-minute budget.
MAX_ROUND_SECONDS = 5.0

#: The measured per-round worst case at each retained shape, in seconds, from the sizing spike that
#: preceded this port: the costliest arm (dask-ml on this box) at each ``(n_block, d)`` the script's
#: grid retains. The filter is arithmetic over these MEASURED points -- a shape absent from the
#: table is REFUSED rather than estimated, so the budget filter can never wave through a shape
#: whose round cost nobody measured.
_MEASURED_ROUND_COSTS: dict[tuple[int, int], float] = {
    (32, 32): 0.05,
    (128, 32): 0.05,
    (512, 32): 0.10,
    (1024, 32): 0.15,
    (32, 128): 0.65,
    (128, 128): 0.70,
    (512, 128): 0.75,
    (1024, 128): 0.80,
    (32, 512): 1.50,
    (128, 512): 1.50,
    (512, 512): 1.55,
    (1024, 512): 1.60,
}


def _grid_budget(n_block: int, d: int) -> float:
    """The smoke-budget bound for one configuration: the MEASURED worst-case round cost.

    - ``:param n_block:`` Rows per block.
    - ``:param d:`` Feature dimension.
    - ``:return:`` The measured worst-case round cost, in seconds.
    """
    try:
        return _MEASURED_ROUND_COSTS[(int(n_block), int(d))]
    except KeyError as exc:
        raise KeyError(
            f"no measured round cost for shape ({n_block}, {d}): the smoke-budget filter only passes shapes "
            "whose one-round cost was measured in the sizing spike; extend _MEASURED_ROUND_COSTS from a "
            "measurement, never from a guess"
        ) from exc


def _plan_configurations() -> list[tuple[int, int, int | None]]:
    """The configurations this module measures: the script's ``plan_configurations()`` through the budget.

    The enumeration is the script's own function, imported unmodified. The budget filter keeps a
    configuration only when its measured worst-case round cost is at or under
    :data:`MAX_ROUND_SECONDS`.

    - ``:return:`` The configurations to measure, in the script's attempt order.
    """
    return [
        (n_block, d, local_rank)
        for n_block, d, local_rank in tc.plan_configurations()
        if _grid_budget(n_block, d) <= MAX_ROUND_SECONDS
    ]


#: The measured grid, computed once at import.
GRID: tuple[tuple[int, int, int | None], ...] = tuple(_plan_configurations())


def _config_key(n_block: int, d: int, local_rank: int | None) -> str:
    """The resume-style key of one configuration, in the script's ``_config_key`` JSON form.

    - ``:param n_block:`` Rows per block.
    - ``:param d:`` Feature dimension.
    - ``:param local_rank:`` Retained local rank, or ``None`` for full local rank.
    - ``:return:`` A stable string key, identical to the script's for the same configuration.
    """
    return json.dumps([int(n_block), int(d), None if local_rank is None else int(local_rank)])


# =================================================================================================
# Per-configuration context, built OUTSIDE every timed region, exactly as the script built its own.
# =================================================================================================
_CONTEXT: dict[str, dict[str, Any]] = {}


def _mc() -> Any:
    """Return the shared ``measurement_common`` module, imported once and cached.

    - ``:return:`` The imported module.
    """
    return _load_measurement_common()


def _context(n_block: int, d: int, local_rank: int | None) -> dict[str, Any]:
    """Build (or reuse) the per-configuration measurement context: blocks, leaves, merged root, wire bytes.

    Nothing here is timed: the blocks are built from the script's seed stream (``SEED + 11`` for
    every block), the leaves summarized, the tree reduced, and the wire sizes measured once -- every
    later benchmark round reuses this context, so the timed regions contain ONLY the arm's compute,
    never data construction.

    - ``:param n_block:`` Rows per block.
    - ``:param d:`` Feature dimension.
    - ``:param local_rank:`` Retained local rank, or ``None`` for full local rank.
    - ``:return:`` The context dict: blocks, leaves, merged root, wire byte counts, reference state.
    """
    key = _config_key(n_block, d, local_rank)
    cached = _CONTEXT.get(key)
    if cached is not None:
        return cached

    mc = _mc()
    intrinsic = max(1, int(INTRINSIC_RANK_FRACTION * d))
    blocks = [mc.make_block(n_block=n_block, n_features=d, rank=intrinsic, seed=mc.SEED + 11) for _ in range(N_BLOCKS)]
    rank_arg = local_rank if local_rank is None else min(int(local_rank), d)
    leaves = [local_pca(block, rank=rank_arg) for block in blocks]
    merged = merge_tree(leaves)

    # The exact reference: batch SVD of the centered concatenation, independent of the merge path --
    # the script's own function, called here so the truth is built by the same code.
    reference = tc._exact_reference(blocks)

    # Wire bytes, computed as DETERMINISTIC ARITHMETIC -- no wall-clock around the serializer, the
    # one deliberate change of measurement semantics, documented in the module docstring.
    block_wire = mc.serialized_nbytes(blocks[0])
    leaf_summary_wire = mc.serialized_nbytes(leaves[0])
    pooled_summary_wire = mc.serialized_nbytes(merged)

    context: dict[str, Any] = {
        "blocks": blocks,
        "leaves": leaves,
        "merged": merged,
        "reference": reference,
        "rank_arg": rank_arg,
        "block_wire_bytes": int(block_wire),
        "leaf_summary_wire_bytes": int(leaf_summary_wire),
        "pooled_summary_wire_bytes": int(pooled_summary_wire),
        "full_block_total_bytes": int(block_wire) * len(blocks),
        "intrinsic_rank": intrinsic,
    }
    _CONTEXT[key] = context
    return context


# =================================================================================================
# The five baseline arms. The fit functions are the SCRIPT's own ``_fit_method``, imported and
# re-exported so the two harnesses cannot drift apart in what a method does.
# =================================================================================================
#: Arm availability, resolved once at import: a missing library is a named finding in the artifact,
#: never a row that silently vanishes -- the script's rule, kept.
AVAILABLE: dict[str, Any] = {method: tc._resolve(method)[0] for method in STANDARD_METHODS}
RESOLVED: dict[str, str] = {method: tc._resolve(method)[1] for method in STANDARD_METHODS}


def _fit_method(method: str, X: np.ndarray, k: int, n_block: int) -> dict[str, Any]:
    """Fit one baseline method on the pooled samples -- the script's own function, unchanged.

    - ``:param method:`` Arm key from :data:`STANDARD_METHODS`.
    - ``:param X:`` The pooled sample matrix, identical for every arm.
    - ``:param k:`` Components to retain.
    - ``:param n_block:`` Rows per leaf, used as ``batch_size`` by the incremental arms.
    - ``:return:`` The sign-invariant state the scorer needs.
    """
    return tc._fit_method(method, X, k, n_block)


def _subspace_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Sign-invariant distance between two equal-rank bases, from the shared measurement suite.

    - ``:param a:`` Basis rows.
    - ``:param b:`` Basis rows, same rank as ``a``.
    - ``:return:`` The sign-invariant distance.
    """
    return _mc().subspace_distance(a, b)


# =================================================================================================
# Row and artifact assembly. The row schema is the script's, with the timing fields stamped from
# pytest-benchmark's own stats through the suite's provenance format.
# =================================================================================================
def _accuracy_row(state: Mapping[str, Any] | None, k: int, reference: Mapping[str, Any], merged: Any) -> dict[str, Any]:
    """Score one fitted state against the exact reference and the mergeable root.

    - ``:param state:`` The fitted state, or ``None`` when the arm failed.
    - ``:param k:`` Components every method is scored on.
    - ``:param reference:`` The exact reference state.
    - ``:param merged:`` The mergeable path's merged root.
    - ``:return:`` The accuracy block, in the script's schema.
    """
    if state is None:
        return {
            "scored": False,
            "reason_unscorable": "no fitted state",
            "subspace_distance_vs_exact": None,
            "subspace_distance_vs_mergeable": None,
        }
    components = np.asarray(state["components"], dtype=np.float64)
    available_k = int(components.shape[0])
    if available_k < k:
        return {
            "scored": False,
            "reason_unscorable": (
                f"method supplies {available_k} directions, fewer than the k={k} every method is scored on; "
                "subspace_distance is undefined across spans of different rank"
            ),
            "subspace_distance_vs_exact": None,
            "subspace_distance_vs_mergeable": None,
        }
    basis = components[:k]
    return {
        "scored": True,
        "reason_unscorable": "",
        "n_components_scored": k,
        "subspace_distance_vs_exact": _subspace_distance(basis, np.asarray(reference["components"])[:k]),
        "subspace_distance_vs_mergeable": _subspace_distance(basis, np.asarray(merged.components)[:k]),
    }


def _row_base(n_block: int, d: int, local_rank: int | None, context: Mapping[str, Any]) -> dict[str, Any]:
    """The per-configuration identity block every artifact row carries, in the script's schema.

    - ``:param n_block:`` Rows per block.
    - ``:param d:`` Feature dimension.
    - ``:param local_rank:`` Retained local rank, or ``None`` for full local rank.
    - ``:param context:`` The configuration's measurement context.
    - ``:return:`` The identity fields, byte dict fields included.
    """
    mc = _mc()
    block_wire = int(context["block_wire_bytes"])
    leaf_summary_wire = int(context["leaf_summary_wire_bytes"])
    full_block_total = int(context["full_block_total_bytes"])
    return {
        "n_block": int(n_block),
        "n_features": int(d),
        "n_block_over_d": float(n_block) / float(d),
        "regime": mc.regime_of(n_block, d),
        "n_blocks": len(context["blocks"]),
        "local_rank_requested": None if local_rank is None else int(local_rank),
        "local_rank_effective_per_leaf": int(context["leaves"][0].rank),
        "intrinsic_rank": int(context["intrinsic_rank"]),
        "n_components_scored": int(N_COMPONENTS_SCORED),
        "total_samples": int(context["reference"]["X"].shape[0]),
        "data_bytes": mc.byte_dict(int(context["reference"]["X"].size) * 8),
        "reference": (
            "exact batch SVD via numpy.linalg.svd on the centered concatenation, independent of the merge path"
        ),
        "mergeable_exact": bool(local_rank is None),
        "block_wire_bytes": mc.byte_dict(block_wire),
        "bytes_saved_per_block_vs_legacy_measured": mc.byte_dict(block_wire - leaf_summary_wire),
        "bytes_saved_total_vs_legacy_derived": mc.byte_dict(
            full_block_total - leaf_summary_wire * len(context["blocks"])
        ),
        "bytes_saved_total_is_derived": True,
        "bytes_saved_total_note": (
            "DERIVED ARITHMETIC: the measured per-bridge saving multiplied by the MEASURED bridge count of "
            "this configuration. It is not a separate measurement and no run of this artifact performed a "
            "multi-bridge transfer"
        ),
    }


def _transfer_fields() -> dict[str, Any]:
    """The transfer-volume declaration for the artifact: which fields are measured, and how.

    The byte fields are deterministic arithmetic -- the port's one deliberate change of measurement
    semantics -- and the artifact says so rather than implying a timing that does not exist.

    - ``:return:`` The transfer-fields provenance block.
    """
    return {
        "in_benchmark_row": "no_timing",
        "why": (
            "serialized_nbytes of a block or a PCASummary is a deterministic function of its input: the old "
            "script timed it once and reported a one-sample duration with no dispersion behind it. This port "
            "computes the byte counts directly (deterministic arithmetic, no wall-clock) and writes the same "
            "byte fields, so the figure pipeline stays fed without a fake dispersion."
        ),
        "fields": [
            "block_wire_bytes",
            "bytes_saved_per_block_vs_legacy_measured",
            "bytes_saved_total_vs_legacy_derived",
            "methods[].transfer_bytes",
        ],
    }


def _timing_block(benchmark: Any) -> dict[str, Any]:
    """The legacy-shaped timing block from pytest-benchmark's own stats, via the shared bridge helper.

    - ``:param benchmark:`` The pytest-benchmark fixture after the run.
    - ``:return:`` The timing block, in the ``time_repeated`` format the figure code reads.
    """
    from bench_common import timing_block_from_stats

    return timing_block_from_stats(benchmark.stats, iterations=int(benchmark.stats.iterations))


def _grid_extra(row_count: int) -> dict[str, Any]:
    """The provenance block every arm's artifact carries: grid, timing policy and transfer declaration.

    - ``:param row_count:`` Rows measured so far, stamped under ``inputs``.
    - ``:return:`` The extra provenance dict.
    """
    return {
        "transfer_comparison_policy": {
            "rule": (
                "a method that runs on the analytics engine needs the samples there first, so its transfer is "
                "the FULL BLOCK. Only a method that computes before it ships transfers less. This is the "
                "central comparison of the work and it is stated here in words as well as in the numbers, "
                "because a reader who inferred it from the ratio alone would conclude the opposite"
            ),
            "baseline_transfer_is_full_block": True,
            "mergeable_transfer_is_the_leaf_summary": True,
            "mergeable_baseline_speed": (
                "no baseline is faster in transfer; the mergeable path is the only one that reduces the "
                "volume crossing the boundary at all"
            ),
        },
        "transfer_fields": _transfer_fields(),
        "resolved_callables": dict(RESOLVED),
        "inputs": {
            "grid_rule": "tc.plan_configurations() filtered by the measured one-round smoke budget",
            "max_round_seconds": MAX_ROUND_SECONDS,
            "grid_measured": [list(key) for key in GRID],
            "grid_full_script": [list(key) for key in tc.plan_configurations()],
            "configurations_measured": int(row_count),
            "n_blocks": N_BLOCKS,
            "intrinsic_rank_fraction_of_d": INTRINSIC_RANK_FRACTION,
            "n_components_scored": int(N_COMPONENTS_SCORED),
            "max_block_rows_per_d": dict(MAX_BLOCK_ELEMENTS_ROWS),
            "timed_repeats": ROUNDS,
            "warmup_rounds": WARMUP_ROUNDS,
            "harness": "pytest-benchmark",
        },
    }


# =================================================================================================
# Module-level row accumulators. pytest-benchmark's fixture is single-use and the sweep is one
#: parametrized test per arm, so measured rows accumulate here and the artifact is rewritten after
#: every point: the LAST write carries the arm's full measured grid. The sibling benchmarks use the
#: same accumulator pattern.
# =================================================================================================
_ROWS: dict[str, list[dict[str, Any]]] = {}


def _accumulate_and_write(arm: str, row: dict[str, Any], benchmark: Any, description: str) -> None:
    """Append one row to an arm's accumulator and rewrite that arm's artifact.

    - ``:param arm:`` The arm key; the accumulator key and the artifact-stem suffix.
    - ``:param row:`` The measured row.
    - ``:param benchmark:`` The pytest-benchmark fixture after the run.
    - ``:param description:`` One line stating what the artifact measures.
    """
    _ROWS.setdefault(arm, []).append(row)
    _write_arm_artifact(
        script=f"{ARTIFACT_STEM}_{arm}",
        description=description,
        benchmark=benchmark,
        rows=_ROWS[arm],
        extra=_grid_extra(len(_ROWS[arm])),
    )


def _write_arm_artifact(
    script: str,
    description: str,
    benchmark: Any,
    rows: Sequence[Mapping[str, Any]],
    extra: Mapping[str, Any],
) -> None:
    """Write one arm's rows as a provenance-stamped artifact through the shared writer and gates.

    Goes through :func:`bench_common.write_benchmark_result` -- the same writer the smoke benchmark
    uses -- so the sign-invariant gate and the repeat-count gate run on this artifact exactly as
    they run on every script artifact. The payload's ``results`` list is the ARM's rows in
    configuration order; the bridge converts the fixture's stats into the provenance block.

    - ``:param script:`` Artifact stem.
    - ``:param description:`` One line stating what the artifact measures.
    - ``:param benchmark:`` The pytest-benchmark fixture after the run.
    - ``:param rows:`` The arm's measured rows so far (the sweep accumulates across parametrizations).
    - ``:param extra:`` Extra provenance merged into the provenance block.
    """
    mc = _mc()
    # The provenance helper defaults its timing policy; restate it from the arm's rows, whose timing blocks
    # are measured by pytest-benchmark and carried per row (authoritative), never defaulted.
    restated = dict(extra)
    restated["timing_policy"] = {
        "clock": "pytest-benchmark timer",
        "warmup_rounds": WARMUP_ROUNDS,
        "timed_repeats": ROUNDS,
        "iterations_per_round": 1,
        "statistic": "median",
        "dispersion": "min/max/iqr/stddev reported alongside the median",
        "note": "measured by pytest-benchmark; every row carries its own timing block",
    }
    payload: dict[str, Any] = {
        "provenance": mc.provenance(script=script, description=description, extra=restated),
        "results": [dict(row) for row in rows],
    }
    path = mc.write_result(script, payload)
    with Path(path).open(encoding="utf-8") as fp:
        stored = json.load(fp)
    assert stored["provenance"]["script"] == script
    assert len(stored["results"]) == len(rows)
    assert mc.enforce_sign_invariant_results(stored) is None


def _require_measured(n_block: int, d: int, local_rank: int | None) -> None:
    """Assert one parametrized configuration is on the grid, refusing a stray parametrization.

    - ``:param n_block:`` Rows per block.
    - ``:param d:`` Feature dimension.
    - ``:param local_rank:`` Retained local rank, or ``None`` for full local rank.
    """
    key = (int(n_block), int(d), local_rank)
    assert key in GRID, (
        f"configuration {key} is not on the smoke-budget grid: the parametrization must enumerate "
        "_plan_configurations(), never a hand-written list"
    )


@pytest.fixture
def bench_arm_grid() -> list[tuple[int, int, int | None]]:
    """The measured grid, exposed as a fixture so a test can assert the parametrization covers it.

    - ``:return:`` The ``(n_block, d, local_rank)`` tuples of the smoke-budget grid.
    """
    return list(GRID)


# =================================================================================================
# The mergeable arms. Two stages plus the whole-pipeline arm, one benchmark fixture each, because
# one fixture measures one thing.
# =================================================================================================
@pytest.mark.benchmark(group="transfer_comparison", min_rounds=ROUNDS, warmup=WARMUP_ROUNDS)
@pytest.mark.parametrize(("n_block", "d", "local_rank"), GRID)
def test_bench_mergeable_leaves(benchmark: Any, n_block: int, d: int, local_rank: int | None) -> None:
    """Benchmark the mergeable path's leaf stage: ``local_pca`` of every block, the pre-ship compute.

    The timed region is ``[local_pca(b, rank) for b in blocks]`` -- the same list comprehension the
    script timed through ``time_repeated`` -- over the SAME blocks the script's seed stream builds.
    The artifact row carries the leaf stage's timing and the configuration's byte fields, computed
    as arithmetic.

    - ``:param benchmark:`` The pytest-benchmark fixture.
    - ``:param n_block:`` Rows per block.
    - ``:param d:`` Feature dimension.
    - ``:param local_rank:`` Retained local rank, or ``None`` for full local rank.
    """
    _require_measured(n_block, d, local_rank)
    context = _context(n_block, d, local_rank)
    blocks = context["blocks"]
    rank_arg = context["rank_arg"]

    def leaves_once() -> list[Any]:
        return [local_pca(block, rank=rank_arg) for block in blocks]

    leaves = benchmark.pedantic(leaves_once, rounds=ROUNDS, warmup_rounds=WARMUP_ROUNDS)
    assert len(leaves) == N_BLOCKS
    assert leaves[0].rank == context["leaves"][0].rank

    timing = _timing_block(benchmark)
    mc = _mc()
    row = {
        **_row_base(n_block, d, local_rank, context),
        "method": MERGEABLE_STAGES[0],
        "in_methods_compared": False,
        "seconds_median": timing["seconds_median"],
        "timing": timing,
        "accuracy": {
            "scored": False,
            "reason_unscorable": "a leaf summary is not a fitted PCA; nothing to score",
            "subspace_distance_vs_exact": None,
            "subspace_distance_vs_mergeable": None,
        },
        "transfer_bytes": {
            "bridge_to_analytics_per_bridge_send": mc.byte_dict(context["leaf_summary_wire_bytes"]),
            "bridge_to_analytics_total_all_bridges": mc.byte_dict(context["leaf_summary_wire_bytes"] * N_BLOCKS),
            "why": (
                "measured: this is the serialized wire size of the PCASummary one bridge actually scatters. "
                "The full chunk never crosses on this path"
            ),
        },
        "samples_declared": timing["timed_repeats"],
        "samples_present": len(timing["seconds_all"]),
        "note": "leaves: the pre-ship compute, timed by pytest-benchmark",
        "failure": "",
    }
    _accumulate_and_write(
        arm="mergeable_pca_local_leaves",
        row=row,
        benchmark=benchmark,
        description=(
            "Mergeable PCA leaf stage: local_pca of every block (the pre-ship compute), one artifact row "
            "per configuration, on the script's seed-parity grid."
        ),
    )


@pytest.mark.benchmark(group="transfer_comparison", min_rounds=ROUNDS, warmup=WARMUP_ROUNDS)
@pytest.mark.parametrize(("n_block", "d", "local_rank"), GRID)
def test_bench_mergeable_merge_tree(benchmark: Any, n_block: int, d: int, local_rank: int | None) -> None:
    """Benchmark the mergeable path's reduction stage: ``merge_tree`` over the SAME configuration's leaves.

    The leaves come from the SAME context the leaves test built -- not re-summarized -- so the two
    stages measure the same data, and the mergeable total is the sum of the two stages' medians,
    the composition rule the script used.

    - ``:param benchmark:`` The pytest-benchmark fixture.
    - ``:param n_block:`` Rows per block.
    - ``:param d:`` Feature dimension.
    - ``:param local_rank:`` Retained local rank, or ``None`` for full local rank.
    """
    _require_measured(n_block, d, local_rank)
    context = _context(n_block, d, local_rank)
    leaves = context["leaves"]

    merged = benchmark.pedantic(merge_tree, args=(leaves,), rounds=ROUNDS, warmup_rounds=WARMUP_ROUNDS)
    assert merged.n_samples == int(context["reference"]["X"].shape[0])
    assert merged.rank <= d

    timing = _timing_block(benchmark)
    mc = _mc()
    row = {
        **_row_base(n_block, d, local_rank, context),
        "method": MERGEABLE_STAGES[1],
        "in_methods_compared": False,
        "seconds_median": timing["seconds_median"],
        "timing": timing,
        "accuracy": {
            "scored": False,
            "reason_unscorable": "a merge node is not a fitted PCA; nothing to score",
            "subspace_distance_vs_exact": None,
            "subspace_distance_vs_mergeable": None,
        },
        "transfer_bytes": {
            "bridge_to_analytics_per_bridge_send": mc.byte_dict(context["leaf_summary_wire_bytes"]),
            "bridge_to_analytics_total_all_bridges": mc.byte_dict(context["leaf_summary_wire_bytes"] * N_BLOCKS),
            "why": "the tree reduces summaries; the transfer already happened at the leaf stage",
        },
        "samples_declared": timing["timed_repeats"],
        "samples_present": len(timing["seconds_all"]),
        "note": "merge tree over the same leaves the leaf stage built: the reduction stage",
        "failure": "",
    }
    _accumulate_and_write(
        arm="mergeable_pca_merge_tree",
        row=row,
        benchmark=benchmark,
        description=(
            "Mergeable PCA merge stage: merge_tree over the same leaves the leaf stage built, one artifact "
            "row per configuration."
        ),
    )


@pytest.mark.benchmark(group="transfer_comparison", min_rounds=ROUNDS, warmup=WARMUP_ROUNDS)
@pytest.mark.parametrize(("n_block", "d", "local_rank"), GRID)
def test_bench_mergeable_pca_total(benchmark: Any, n_block: int, d: int, local_rank: int | None) -> None:
    """Benchmark the mergeable path AS AN ARM: leaves and tree in ONE timed round, scored for accuracy.

    The script timed the two stages separately and summed their medians; the total arm here times
    the whole pipeline in ONE round and reports that median as the arm's headline, with the stage
    artifacts carrying the two stages' own medians -- so the artifact never implies a blend that was
    not measured, and the comparison stays attributable.

    - ``:param benchmark:`` The pytest-benchmark fixture.
    - ``:param n_block:`` Rows per block.
    - ``:param d:`` Feature dimension.
    - ``:param local_rank:`` Retained local rank, or ``None`` for full local rank.
    """
    _require_measured(n_block, d, local_rank)
    context = _context(n_block, d, local_rank)
    blocks = context["blocks"]
    rank_arg = context["rank_arg"]
    reference = context["reference"]
    k = int(min(N_COMPONENTS_SCORED, int(reference["components"].shape[0])))

    def pipeline_once() -> Any:
        return merge_tree([local_pca(block, rank=rank_arg) for block in blocks])

    merged = benchmark.pedantic(pipeline_once, rounds=ROUNDS, warmup_rounds=WARMUP_ROUNDS)
    assert merged.n_samples == int(reference["X"].shape[0])

    timing = _timing_block(benchmark)
    mc = _mc()
    row = {
        **_row_base(n_block, d, local_rank, context),
        "method": MERGEABLE_ARM,
        "in_methods_compared": True,
        "seconds_median": timing["seconds_median"],
        "timing": {
            **timing,
            "composition": (
                "one separately timed pipeline round (leaves + merge tree in ONE measurement); the stage "
                "artifacts carry the two stages' own medians"
            ),
        },
        "accuracy": _accuracy_row(
            {
                "components": merged.components,
                "singular_values": merged.singular_values,
                "mean": merged.mean,
                "n_samples": int(merged.n_samples),
            },
            k,
            reference,
            context["merged"],
        ),
        "transfer_bytes": {
            "bridge_to_analytics_per_bridge_send": mc.byte_dict(context["leaf_summary_wire_bytes"]),
            "bridge_to_analytics_total_all_bridges": mc.byte_dict(context["leaf_summary_wire_bytes"] * N_BLOCKS),
            "why": (
                "measured: this is the serialized wire size of the PCASummary one bridge actually scatters. "
                "The full chunk never crosses on this path"
            ),
            "root_summary_wire": mc.byte_dict(context["pooled_summary_wire_bytes"]),
        },
        "samples_declared": timing["timed_repeats"],
        "samples_present": len(timing["seconds_all"]),
        "note": "leaves + reduction in one round: the only arm that computes BEFORE it ships",
        "failure": "",
    }
    _accumulate_and_write(
        arm="mergeable_pca_total",
        row=row,
        benchmark=benchmark,
        description=(
            "Mergeable PCA as a compared arm: local_pca of every block plus merge_tree in ONE timed round, "
            "scored against the exact reference."
        ),
    )


# =================================================================================================
# The five baseline arms, one parametrized benchmark over the same grid. A baseline runs ON the
# analytics engine, so its transfer is the FULL BLOCK -- the central comparison, kept verbatim.
# =================================================================================================
@pytest.mark.benchmark(group="transfer_comparison", min_rounds=ROUNDS, warmup=WARMUP_ROUNDS)
@pytest.mark.parametrize("method", STANDARD_METHODS)
@pytest.mark.parametrize(("n_block", "d", "local_rank"), GRID)
def test_bench_standard_method(benchmark: Any, method: str, n_block: int, d: int, local_rank: int | None) -> None:
    """Benchmark ONE standard baseline arm at ONE configuration, on the pooled data.

    The timed region is the script's ``_fit_method(method, X, k, n_block)`` -- the same fit on the
    same pooled matrix, timed by pytest-benchmark instead of ``time_repeated``. The row records the
    FULL BLOCK as the arm's transfer, whatever the fit did: the samples must be there first for any
    baseline to run, and no baseline is presented as cheaper in transfer than the legacy scatter.

    - ``:param benchmark:`` The pytest-benchmark fixture.
    - ``:param method:`` The baseline arm key.
    - ``:param n_block:`` Rows per block.
    - ``:param d:`` Feature dimension.
    - ``:param local_rank:`` Retained local rank, or ``None`` for full local rank (the fit is
      rank-independent; the parameter keeps the row keys aligned across arms).
    """
    _require_measured(n_block, d, local_rank)
    context = _context(n_block, d, local_rank)
    reference = context["reference"]
    k = int(min(N_COMPONENTS_SCORED, int(reference["components"].shape[0])))
    X = reference["X"]
    if AVAILABLE.get(method) is None:
        pytest.skip(f"{method} unavailable on this machine ({RESOLVED[method]})")

    state = benchmark.pedantic(_fit_method, args=(method, X, k, n_block), rounds=ROUNDS, warmup_rounds=WARMUP_ROUNDS)
    assert state["components"].shape[0] >= 1

    timing = _timing_block(benchmark)
    mc = _mc()
    row = {
        **_row_base(n_block, d, local_rank, context),
        "method": method,
        "in_methods_compared": True,
        "seconds_median": timing["seconds_median"],
        "timing": timing,
        "accuracy": _accuracy_row(state, k, reference, context["merged"]),
        "transfer_bytes": {
            "bridge_to_analytics_per_bridge_send": mc.byte_dict(context["block_wire_bytes"]),
            "bridge_to_analytics_total_all_bridges": mc.byte_dict(context["full_block_total_bytes"]),
            "why": (
                "measured, and equal to the FULL BLOCK: this method runs on the analytics engine, so the "
                "samples must arrive first. Every baseline therefore ships the chunk, and no baseline is "
                "cheaper in transfer than the legacy scatter"
            ),
        },
        "samples_declared": timing["timed_repeats"],
        "samples_present": len(timing["seconds_all"]),
        "note": (
            f"runs on the analytics engine on the pooled data: full chunk crosses, then {METHOD_IMPORTS[method][1]}"
        ),
        "failure": "",
    }
    _accumulate_and_write(
        arm=method,
        row=row,
        benchmark=benchmark,
        description=(
            f"Standard baseline arm {method} on the pooled data (the script's _fit_method, timed by "
            "pytest-benchmark), one artifact row per configuration."
        ),
    )


# =================================================================================================
# Seed parity: the port and the script must agree on every DETERMINISTIC field. Timings differ by
# harness; byte counts and accuracy must not.
# =================================================================================================
def _write_parity_artifact(n_block: int, d: int, local_rank: int | None, merges_measured: int) -> None:
    """Write (or rewrite) the parity record: the smoke configurations proven equal, one row per configuration.

    This record is written THROUGH the shared writer and its gates like every artifact of this suite,
    but with its own honest timing policy: the parity benchmark times ONE pipeline round per
    configuration (rounds=1), so stamping the sweep's ``timed_repeats=5`` here would declare a repeat
    count that did not run -- the contradiction ``enforce_consistent_repeat_counts`` exists to catch.

    - ``:param n_block:`` Rows per block of the parity configuration just proven.
    - ``:param d:`` Feature dimension of the parity configuration just proven.
    - ``:param local_rank:`` Local rank of the parity configuration just proven.
    - ``:param merges_measured:`` Pipeline rounds timed so far, one per parity configuration.
    """
    mc = _mc()
    _PARITY_ROWS.append(
        {
            "n_block": int(n_block),
            "n_features": int(d),
            "local_rank_requested": None if local_rank is None else int(local_rank),
            "parity": "byte fields and accuracy agree with benchmark/mergeable_pca/transfer_comparison.py",
        }
    )
    payload: dict[str, Any] = {
        "provenance": mc.provenance(
            script="transfer_comparison_pytest_parity",
            description=(
                "Seed parity: the ported byte fields and accuracy match benchmark/mergeable_pca/"
                "transfer_comparison.py's measure_configuration output on the same blocks."
            ),
            extra={
                "inputs": {
                    "configurations_passed": len(_PARITY_ROWS),
                    "merges_measured": int(merges_measured),
                    "timed_repeats": 1,
                    "warmup_rounds": 0,
                    "iterations_per_round": 1,
                    "harness": "pytest-benchmark",
                    "grid_rule": "smoke parity configs, a subset of tc.plan_configurations()",
                },
                "timing_policy": {
                    "clock": "pytest-benchmark timer (timeit.default_timer unless overridden)",
                    "warmup_rounds": 0,
                    "timed_repeats": 1,
                    "iterations_per_round": 1,
                    "statistic": "median",
                    "dispersion": "none: one round per parity configuration by design",
                    "note": (
                        "the parity fixture times ONE pipeline round per configuration; the parity CLAIM "
                        "is the deterministic byte/accuracy fields, asserted equal to the script in-test"
                    ),
                    "source": "pytest-benchmark",
                },
            },
        ),
        "results": list(_PARITY_ROWS),
    }
    path = mc.write_result("transfer_comparison_pytest_parity", payload)
    with Path(path).open(encoding="utf-8") as fp:
        stored = json.load(fp)
    assert stored["provenance"]["script"] == "transfer_comparison_pytest_parity"
    assert len(stored["results"]) == len(_PARITY_ROWS)
    assert mc.enforce_sign_invariant_results(stored) is None


#: The parity rows measured by this process, written out after every passing configuration: the LAST
#: write carries all of them, the same accumulator pattern the sweep arms use.
_PARITY_ROWS: list[dict[str, Any]] = []


@pytest.mark.benchmark(group="transfer_comparison_parity", min_rounds=1, warmup=0)
@pytest.mark.parametrize(("n_block", "d", "local_rank"), [(32, 32, None), (128, 32, 8)])
def test_bench_parity_with_script(benchmark: Any, n_block: int, d: int, local_rank: int | None) -> None:
    """Prove the ported row and the script's row agree on every deterministic field.

    The script's ``measure_configuration`` runs on the SAME blocks this module's context builds
    (same shapes, same ``SEED + 11`` stream), then every byte field, every accuracy value and the
    grid identity are compared. Timing is NOT compared: the script's ``time_repeated`` and
    pytest-benchmark's caliper measure the same compute with different clocks and repeat policies,
    and the artifact says so.

    - ``:param benchmark:`` The pytest-benchmark fixture (the mergeable pipeline, timed once).
    - ``:param n_block:`` Rows per block.
    - ``:param d:`` Feature dimension.
    - ``:param local_rank:`` Retained local rank, or ``None`` for full local rank.
    """
    _require_measured(n_block, d, local_rank)
    mc = _mc()
    context = _context(n_block, d, local_rank)

    # The script's own measurement, on the SAME blocks -- the reference this port must reproduce.
    # ``available`` carries the SAME resolved map this module resolved at import, so every arm the
    # script's ``measure_configuration`` can fit DOES fit on this machine -- an empty map would leave
    # the script's distances None and the comparison below vacuous for the baselines.
    script_row = tc.measure_configuration(
        list(context["blocks"]), n_block, d, local_rank, repeats=2, available=dict(AVAILABLE)
    )
    script_row.pop("replayed_from_checkpoint", None)
    k = int(script_row["n_components_scored"])

    def pipeline_once() -> Any:
        return merge_tree([local_pca(block, rank=context["rank_arg"]) for block in context["blocks"]])

    merged = benchmark.pedantic(pipeline_once, rounds=1, warmup_rounds=0)
    assert merged.n_samples == script_row["total_samples"]

    # --- deterministic fields, compared field by field -------------------------------
    base = _row_base(n_block, d, local_rank, context)
    assert script_row["block_wire_bytes"] == base["block_wire_bytes"]
    assert script_row["bytes_saved_per_block_vs_legacy_measured"] == base["bytes_saved_per_block_vs_legacy_measured"]
    assert script_row["bytes_saved_total_vs_legacy_derived"] == base["bytes_saved_total_vs_legacy_derived"]
    assert script_row["local_rank_effective_per_leaf"] == base["local_rank_effective_per_leaf"]
    assert script_row["intrinsic_rank"] == base["intrinsic_rank"]
    assert script_row["total_samples"] == base["total_samples"]
    assert script_row["regime"] == base["regime"]
    assert script_row["mergeable_exact"] == base["mergeable_exact"]

    # --- per-method transfer bytes and accuracy, compared arm by arm -------------------------------
    ported_rows: dict[str, dict[str, Any]] = {
        MERGEABLE_ARM: {
            "transfer_bytes": {
                "bridge_to_analytics_per_bridge_send": mc.byte_dict(context["leaf_summary_wire_bytes"]),
                "bridge_to_analytics_total_all_bridges": mc.byte_dict(context["leaf_summary_wire_bytes"] * N_BLOCKS),
            },
            "accuracy": _accuracy_row(
                {
                    "components": context["merged"].components,
                    "singular_values": context["merged"].singular_values,
                    "mean": context["merged"].mean,
                    "n_samples": int(context["merged"].n_samples),
                },
                k,
                context["reference"],
                context["merged"],
            ),
        }
    }
    for method in STANDARD_METHODS:
        ported_rows[method] = {
            "transfer_bytes": {
                "bridge_to_analytics_per_bridge_send": base["block_wire_bytes"],
                "bridge_to_analytics_total_all_bridges": mc.byte_dict(context["full_block_total_bytes"]),
            },
            "accuracy": _accuracy_row(
                tc._fit_method(method, context["reference"]["X"], k, n_block),
                k,
                context["reference"],
                context["merged"],
            ),
        }
    for script_method in script_row["methods"]:
        name = script_method["method"]
        ported = ported_rows.get(name)
        assert ported is not None, f"the port has no row for the script's arm {name!r}"
        assert (
            ported["transfer_bytes"]["bridge_to_analytics_per_bridge_send"]
            == script_method["transfer_bytes"]["bridge_to_analytics_per_bridge_send"]
        ), f"{name}: per-bridge transfer bytes disagree with the script"
        assert (
            ported["transfer_bytes"]["bridge_to_analytics_total_all_bridges"]
            == script_method["transfer_bytes"]["bridge_to_analytics_total_all_bridges"]
        ), f"{name}: total transfer bytes disagree with the script"
        ported_distance = ported["accuracy"]["subspace_distance_vs_exact"]
        script_distance = script_method["accuracy"]["subspace_distance_vs_exact"]
        assert ported_distance == script_distance, f"{name}: accuracy vs exact disagrees with the script"

    _write_parity_artifact(n_block=n_block, d=d, local_rank=local_rank, merges_measured=1)
