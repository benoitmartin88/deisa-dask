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
Shared fixtures for the pytest-benchmark suite under ``test/benchmarks/``.

Discovery design (why the default suite never runs these)
---------------------------------------------------------
These tests are timing measurements, so they must never run as part of a correctness run: ``pytest test/`` collects
nothing from this directory because ``pytest.ini`` adds ``benchmarks`` to ``norecursedirs``, which stops recursive
discovery by directory name while leaving an explicit file argument untouched. Benchmark runs therefore name the file
or the directory directly::

    .venv/bin/python -m pytest test/benchmarks/test_bench_smoke.py --benchmark-only -q

The ``benchmark`` marker is registered in ``pytest.ini`` and is applied to every benchmark test here so a
``-m "not benchmark"`` filter stays available; it is deliberately NOT enforced through ``addopts``, because the
Bencher CI workflow runs ``pytest benchmark/``, where existing tests carry the same marker and a global deselect
would silently empty that run.

Fixtures
--------
- :func:`bench_cluster` -- session-scoped ``LocalCluster`` (1 worker, 1 thread, in-process) plus its ``Client``, for
  benchmarks that need the distributed scheduler without paying spin-up per test.
- :func:`bench_rng` -- a ``numpy.random.default_rng`` seeded from :option:`--bench-seed` (default:
  :data:`measurement_common.SEED`), function-scoped so every benchmark draws an identical, reproducible stream.
- :func:`bench_data` -- a synthetic low-rank block built with that generator, the same construction
  :func:`measurement_common.make_block` uses.

The artifact bridge lives in :mod:`bench_common`, next to this conftest.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

#: The legacy measurement module is imported through ``sys.path`` exactly as the experiment scripts do it, so
#: ``measurement_common.py`` stays untouched and the artifact it writes keeps one single source of provenance.
_BENCHMARK_PKG = Path(__file__).resolve().parents[2] / "benchmark" / "mergeable_pca"

#: Seed default: the master seed of the legacy measurement suite, so a pytest-benchmark artifact and a script
#: artifact drawn at the same stream id describe the same synthetic data.
_DEFAULT_SEED = 20261004


def _load_measurement_common() -> Any:
    """Import the legacy ``measurement_common`` module by path, once, without touching the scripts.

    The scripts put their own package directory on ``sys.path`` and ``import measurement_common``; doing the same
    here keeps a single provenance/JSON writer for both harnesses. A module named ``measurement_common`` could in
    principle already be importable from an earlier ``sys.path`` insert by a script run in the same interpreter, so
    the package directory is inserted at position 0 first, exactly like the scripts do.

    - ``:return:`` The imported module.
    """
    if str(_BENCHMARK_PKG) not in sys.path:
        sys.path.insert(0, str(_BENCHMARK_PKG))
    import measurement_common

    return measurement_common


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register ``--bench-seed`` for the benchmark suite.

    - ``:param parser:`` pytest's option parser.
    """
    group = parser.getgroup("deisa-benchmark", "deisa pytest-benchmark suite")
    group.addoption(
        "--bench-seed",
        type=int,
        default=None,
        help="Seed for every synthetic benchmark dataset (default: the measurement_common.SEED master seed).",
    )


@pytest.fixture(scope="session")
def bench_seed(request: pytest.FixtureRequest) -> int:
    """The seed every benchmark data fixture draws from, or the suite default.

    - ``:param request:`` The pytest request, used to read :option:`--bench-seed`.
    """
    return (
        int(request.config.getoption("--bench-seed"))
        if request.config.getoption("--bench-seed") is not None
        else _DEFAULT_SEED
    )


@pytest.fixture(scope="session")
def bench_cluster() -> Any:
    """Session-scoped ``LocalCluster`` + ``Client``: one worker, one thread, in-process, dashboard off.

    ``processes=False`` keeps every benchmark inside this interpreter: no worker start-up cost, no serialization of
    the fixtures, and the SVD timings measure compute rather than process spin-up. ``dashboard_address=":0"`` binds
    the dashboard to an ephemeral port (disabled in practice, no fixed-port collision).

    - ``:yield:`` The connected ``Client``; the cluster is closed with it.
    """
    from distributed import Client, LocalCluster

    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=1,
        processes=False,
        dashboard_address=":0",
    )
    client = Client(cluster)
    try:
        yield client
    finally:
        client.close()
        cluster.close()


@pytest.fixture
def bench_rng(bench_seed: int) -> np.random.Generator:
    """A fresh ``default_rng`` per benchmark test, so every test sees the same stream for a given seed.

    Function scope on purpose: a session-scoped generator would make a benchmark's data depend on how many tests
    ran before it, which is exactly the non-reproducibility the seed exists to prevent.

    - ``:param bench_seed:`` Seed from :func:`bench_seed`.
    - ``:yield:`` A ``numpy.random.Generator``.
    """
    return np.random.default_rng(bench_seed)


@pytest.fixture
def bench_data(bench_rng: np.random.Generator) -> dict[str, Any]:
    """One small synthetic low-rank block plus its known principal subspace, for the smoke benchmarks.

    The construction is ``low-rank signal + isotropic noise`` on float64, the same shape
    :func:`measurement_common.make_block` builds, so an artifact written from this fixture is comparable with the
    script artifacts. The smoke sizes are deliberately small: a smoke benchmark proves the harness measures, it does
    not produce a paper number.

    - ``:param bench_rng:`` The seeded generator.
    - ``:yield:`` ``{"data": (n_block, n_features) float64 array, "true_components": (rank, n_features) rows}``.
    """
    n_block, n_features, rank = 64, 32, 4
    signal = bench_rng.standard_normal((n_features, rank))
    factors = bench_rng.standard_normal((n_block, rank))
    low_rank = factors @ signal.T
    noise = bench_rng.standard_normal((n_block, n_features)) * (0.05 * float(np.std(low_rank)))
    data = np.ascontiguousarray(low_rank + noise, dtype=np.float64)
    # The exact principal subspace of the noise-free signal, as ROWS -- the same orthonormal-row convention the
    # summaries use -- so a benchmark can assert sign-invariant accuracy if it wants to.
    true_components, _ = np.linalg.qr(signal.T)
    return {"data": data, "true_components": true_components.T.copy()}
