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
Smoke benchmark: prove the pytest-benchmark harness measures and the artifact bridge writes a valid artifact.

One fit of :class:`~deisa.dask.mergeable_pca.MergeablePCA` over a small synthetic block, on the in-memory path
(the single-leaf base case of the merge tree), then one round trip through the artifact bridge, whose output must
parse back and satisfy the same structural invariants the script artifacts satisfy. The numbers are smoke sizes --
this test proves the harness works, it does not produce a paper measurement.
"""

from __future__ import annotations

import json

import pytest
from bench_common import write_benchmark_result
from conftest import _load_measurement_common

from deisa.dask.mergeable_pca import MergeablePCA


@pytest.mark.benchmark(group="smoke", min_rounds=5)
def test_bench_mergeable_pca_fit_smoke(benchmark, bench_data, tmp_path):
    """Benchmark a small MergeablePCA fit and write the provenance-stamped artifact from its stats.

    The artifact is written to ``tmp_path`` -- a smoke run must not overwrite a real script artifact in the results
    directory -- and is then read back and structurally checked: provenance stamped, sign-invariant metrics only,
    and a repeat count that matches the measured sample list.
    """
    data = bench_data["data"]

    def fit_once() -> MergeablePCA:
        return MergeablePCA(n_components=4).fit(data)

    estimator = benchmark(fit_once)
    assert estimator.n_components_ == 4
    assert estimator.n_samples_ == data.shape[0]

    measurement_common = _load_measurement_common()
    artifact = write_benchmark_result(
        script="bench_smoke",
        description="pytest-benchmark smoke: MergeablePCA fit on a small synthetic low-rank block (harness check).",
        benchmark=benchmark,
        out_dir=tmp_path,
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))

    # The artifact round-trips and carries the same provenance the script artifacts carry.
    assert payload["provenance"]["script"] == "bench_smoke"
    assert payload["provenance"]["deisa_dask_commit"]
    assert payload["provenance"]["timing_policy"]["timed_repeats"] == len(payload["results"][0]["seconds_all"])
    # Sign-invariant metrics only: the gate ran at write time; this asserts nothing regressed past it.
    assert measurement_common.enforce_sign_invariant_results(payload) is None
