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
JSON-artifact bridge: pytest-benchmark stats written in the legacy measurement-suite artifact format.

The old scripts emit one JSON per experiment through :func:`measurement_common.write_result`: a ``provenance`` block,
an ``inputs`` block, and a ``results`` list whose timing rows carry ``seconds_median``/``seconds_min``/``seconds_max``
/``seconds_iqr``/``seconds_stddev``/``seconds_all`` plus the ``warmup_rounds``/``timed_repeats`` policy counts. The
figure code (``plot_figures.py``) reads exactly those keys, so the format is the contract.

:func:`write_benchmark_result` converts one pytest-benchmark :class:`~pytest_benchmark.fixture.BenchmarkFixture`'s
``stats`` into that same shape and writes it through the SAME writer, so the sign-invariant metric gate and the
repeat-count gate of the measurement suite run on pytest-benchmark artifacts too -- a bridge artifact cannot ship a
sign-sensitive metric or a repeat count that contradicts its sample list, by construction rather than by review.

The timing policy is stated from pytest-benchmark's own numbers, never from :data:`measurement_common.TIMING_POLICY`:
the clock is pytest-benchmark's timer, the repeats are its ``rounds``, and the warmup is whatever the harness ran,
stated as seen. Overriding the default policy block is the documented contract of :func:`measurement_common.provenance`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from conftest import _load_measurement_common


def timing_block_from_stats(stats: Any, iterations: int = 1) -> dict[str, Any]:
    """Render pytest-benchmark stats as one legacy ``time_repeated``-shaped timing block.

    Every field the scripts emit and the figure code reads is filled from the measured stats; nothing is defaulted.
    ``seconds_all`` carries the per-round sample list so ``enforce_consistent_repeat_counts`` can cross-check the
    declared count against it, exactly as it does for script artifacts.

    pytest-benchmark 5.x wraps the samples in an inner ``Stats`` object: ``Metadata.get`` forwards to it and
    ``Metadata.stats.data`` carries the raw per-round durations, so both are read from there.

    - ``:param stats:`` A ``BenchmarkFixture.stats`` (pytest-benchmark 5.x ``Metadata``).
    - ``:param iterations:`` Iterations per round the fixture calibrated to (default 1), recorded as ``iterations``.
    - ``:return:`` The timing block, in the ``time_repeated`` format.
    """
    samples = [float(value) for value in stats.stats.data]
    rounds = stats.get("rounds")
    if rounds != len(samples):
        raise ValueError(
            f"timing_block_from_stats: stats reports rounds={rounds} but carries {len(samples)} sample(s). "
            "The timing block must never declare a repeat count its own sample list contradicts."
        )
    return {
        "seconds_median": float(stats.get("median")),
        "seconds_min": float(stats.get("min")),
        "seconds_max": float(stats.get("max")),
        "seconds_iqr": float(stats.get("iqr")),
        "seconds_stddev": float(stats.get("stddev")),
        "seconds_all": samples,
        "warmup_rounds": 0,
        "timed_repeats": rounds,
        "iterations_per_round": int(iterations),
        "statistic": "median",
        "clock": "pytest-benchmark timer (timeit.default_timer unless overridden)",
        "source": "pytest-benchmark",
    }


def write_benchmark_result(
    script: str,
    description: str,
    benchmark: Any,
    extra: dict[str, Any] | None = None,
    out_dir: Path | str | None = None,
) -> Path:
    """Write one pytest-benchmark run as a provenance-stamped artifact in the legacy format.

    The payload is the shape every figure reads: ``provenance`` (from the shared :func:`measurement_common.provenance`,
    with ``timing_policy`` restated from the ACTUAL pytest-benchmark run), ``inputs`` naming the repeat counts, and a
    ``results`` list holding the converted timing block under the benchmark's name. Writing goes through
    :func:`measurement_common.write_result`, so the sign-invariant gate and the repeat-count gate both run here.

    - ``:param script:`` Artifact stem, e.g. ``"bench_smoke"``.
    - ``:param description:`` One line stating what the artifact measures.
    - ``:param benchmark:`` The ``benchmark`` fixture after the run (its ``stats`` and ``fullname`` are read).
    - ``:param extra:`` Optional extra provenance (inputs, sweep definition), merged into the provenance block.
    - ``:param out_dir:`` Output directory; ``None`` writes beside the script artifacts in
      ``benchmark/mergeable_pca/results/`` (gitignored), keeping every artifact of the suite in one place.
    - ``:return:`` The path the artifact was written to.
    """
    measurement_common = _load_measurement_common()

    if benchmark.stats is None or not benchmark.stats:
        raise ValueError(
            f"write_benchmark_result: the benchmark fixture for {script!r} carries no stats. A benchmark fixture is "
            "single-use: write the artifact inside the same test that ran it."
        )
    block = timing_block_from_stats(benchmark.stats, iterations=int(benchmark.stats.iterations))

    timing_policy = {
        "clock": block["clock"],
        "warmup_rounds": block["warmup_rounds"],
        "timed_repeats": block["timed_repeats"],
        "iterations_per_round": block["iterations_per_round"],
        "statistic": "median",
        "dispersion": "min/max/iqr/stddev reported alongside the median",
        "note": "measured by pytest-benchmark: rounds are its timed repeats, calibration may add iterations per round",
    }
    payload: dict[str, Any] = {
        "provenance": measurement_common.provenance(
            script=script,
            description=description,
            extra={
                "inputs": {
                    "timed_repeats": block["timed_repeats"],
                    "warmup_rounds": block["warmup_rounds"],
                    "iterations_per_round": block["iterations_per_round"],
                    "harness": "pytest-benchmark",
                    "benchmark_fullname": benchmark.fullname,
                },
                "timing_policy": timing_policy,
            },
        ),
        "results": [
            {
                "name": benchmark.name,
                "fullname": benchmark.fullname,
                **block,
            }
        ],
    }
    if extra:
        payload["provenance"].update(extra)

    # The shared writer runs enforce_sign_invariant_results and enforce_consistent_repeat_counts before writing,
    # so a bridge artifact is held to exactly the same gates as a script artifact.
    if out_dir is None:
        return measurement_common.write_result(script, payload)
    out = Path(out_dir)
    measurement_common.RESULTS_DIR = out
    try:
        return measurement_common.write_result(script, payload)
    finally:
        measurement_common.RESULTS_DIR = (
            measurement_common.Path(__file__).resolve().parents[2] / "benchmark" / "mergeable_pca" / "results"
        )
