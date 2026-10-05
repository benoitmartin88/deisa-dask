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
Shared measurement utilities for the mergeable-PCA benchmark harness.

Every experiment script in this package imports its machine/version/seed metadata, its JSON writer, its
sign-INVARIANT accuracy metrics and its timing policy from here, so the four that matter -- that every figure
traces to one provenance block, that no script invents its own accuracy metric, and that no script reports a
timing measured under an unstated warmup policy -- are enforced by construction rather than by review.

The no-fabricated-numbers rule
------------------------------
Nothing in this package contains a measured constant. Shapes, seeds, sweep points and warmup counts are INPUTS
and are named as such; every byte count, ratio, timing and accuracy value in the emitted JSON is COMPUTED at run
time by the script that reports it. :func:`provenance` stamps the machine, the library versions, the seed and the
timing policy into every artifact so a reader can re-run and compare.

Why accuracy is measured with sign-invariant metrics only
---------------------------------------------------------
Eigenvectors are only defined up to a sign, and a PCA component's sign is arbitrary (it flips with the SVD's
internal choice, with LAPACK version, with memory layout). A raw component-wise error therefore reads O(1) even
for two EXACTLY equal subspaces whose signs happen to differ, and it is non-monotonic in the retained rank. Using
it would make the accuracy/bandwidth curve meaningless, so this module exposes exactly two metrics, both
invariant to sign, rotation within the retained subspace, and basis choice:

- :func:`subspace_distance` -- ``1 - min(svd(A @ B.T))`` for two orthonormal row bases of equal rank. It is 0 iff
  the spans coincide and 1 iff they are orthogonal, and it cannot be gamed by a sign flip.
- :func:`variance_errors` -- relative explained-variance error against the exact pooled SVD, computed on the
  SINGULAR VALUES (a sign-free quantity), not on components.

:func:`enforce_sign_invariant_results` is called by every experiment before it writes: it refuses to emit a
result set that carries a raw component error, so the metric choice cannot regress silently.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ----------------------------------------------------------------------------- reproducibility inputs
#: Master seed. Every synthetic field is drawn from ``default_rng(SEED + <stream id>)`` so a script's data is
#: reproducible on its own and two scripts sharing a stream cannot silently share a draw.
SEED = 20261004

#: The gysela code generation these sizing numbers describe, so a reader knows WHICH sources were sized against.
#: The Fortran GYSELA decomposition is superseded and is deliberately absent: it is not measured and not cited.
GYSELA_SOURCES: dict[str, dict[str, str]] = {
    "gyselalibxx": {
        "repository": "https://github.com/gyselax/gyselalibxx",
        "branch": "devel",
        "commit": "b9aad37bc0d174ce37e95021573d9b846251d86a",
        "role": "MPILayout distribution semantics (src/mpi_parallelisation/mpilayout.hpp)",
    },
    "gysela-mini-app_io": {
        "repository": "https://github.com/gyselax/gysela-mini-app_io",
        "branch": "(detached)",
        "commit": "f39e2a57456e82aabefed140ccd97bd06453f747",
        "role": "field and MPI layout types (src/C++/geometry.hpp)",
    },
}

#: Directory every script writes its raw JSON into. Created on demand by :func:`write_result`.
RESULTS_DIR = Path(__file__).resolve().parent / "results"

#: Default timing policy, applied to every timed measurement unless a script overrides it and says so in its JSON.
#: Stated here once so no script can quietly time a cold import and call it a result.
TIMING_POLICY: dict[str, Any] = {
    "clock": "time.perf_counter",
    "warmup_rounds": 1,
    "timed_repeats": 5,
    "statistic": "median",
    "dispersion": "min/max/iqr/stddev reported alongside the median",
    "note": "one warmup round is discarded before timing so BLAS thread pools and page faults are not charged "
    "to the measurement; the median of the timed repeats is the headline and the spread is reported",
}


# ----------------------------------------------------------------------------- provenance
def _library_version(name: str) -> str:
    """Return an installed distribution's version, or a marked MISSING string.

    Recorded as MISSING rather than omitted so an artifact never silently implies a dependency was present. The
    point of the baselines card is that an unavailable baseline is a FINDING; a version table with the row
    silently dropped would hide exactly that.

    - ``:param name:`` Distribution name as importlib.metadata knows it (``"dask-ml"``, ``"scikit-learn"``).
    """
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "MISSING (not installed)"
    except Exception as exc:  # pragma: no cover - defensive: a broken metadata dir must not kill a benchmark
        return f"MISSING ({type(exc).__name__})"


def _repo_root(start: Path) -> Path:
    """Walk up from ``start`` to the nearest ancestor holding a ``.git`` entry, else return ``start``.

    A fixed ``parents[N]`` is wrong the moment the harness moves, and it fails SILENTLY: ``git rev-parse`` in a
    non-repository directory returns non-zero and the artifact records ``UNAVAILABLE (git rev-parse failed)``, which
    looks like a missing git rather than a wrong index. Searching for the marker makes the stamp self-locating.

    - ``:param start:`` Resolved path inside the repository, normally this module's own file.
    """
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start


def _git_commit(repo: Path) -> str:
    """Return a repository's HEAD commit, or a marked UNAVAILABLE string.

    Used to stamp the deisa-dask revision the measurement was taken on, which is what makes a number
    attributable to a code state rather than to a branch name.

    - ``:param repo:`` Path to the repository to interrogate.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - git absent or timed out
        return f"UNAVAILABLE ({type(exc).__name__})"
    if out.returncode != 0:
        return "UNAVAILABLE (git rev-parse failed)"
    return out.stdout.strip()


def enforce_consistent_repeat_counts(payload: Mapping[str, Any]) -> None:
    """Refuse to emit an artifact whose recorded repeat counts contradict each other or the timed samples.

    ``provenance()`` always stamps ``timing_policy`` with :data:`TIMING_POLICY`'s DEFAULT ``timed_repeats``, so a
    script invoked with ``--repeats 3`` writes ``timing_policy.timed_repeats = 5`` next to
    ``inputs.timed_repeats = 3``. That is a provenance defect: a reader cannot tell how many timed samples produced a
    median. It also hides the consequence, which is worse -- with ONE repeat there is no spread to report, so every
    ``seconds_iqr`` and ``seconds_stddev`` is a structural 0.0 that reads as "perfectly reproducible" when in fact
    nothing was repeated.

    Three counts must agree: the timing policy default, the count the script recorded under ``inputs``, and the
    length of each timed sample list. This compares them and raises naming the field that disagrees.

    - ``:param payload:`` The artifact about to be written.
    """
    policy = payload.get("provenance", {}).get("timing_policy", {})
    declared = policy.get("timed_repeats")
    inputs = payload.get("provenance", {}).get("inputs", {})
    used = inputs.get("timed_repeats")
    if declared is not None and used is not None and int(declared) != int(used):
        raise ValueError(
            f"enforce_consistent_repeat_counts: timing_policy.timed_repeats={declared} contradicts "
            f"provenance.inputs.timed_repeats={used}. provenance() stamps the DEFAULT policy, so a run with a "
            f"different repeat count must override it explicitly; otherwise the artifact reports a repeat count it "
            f"did not use and the median has no dispersion behind it."
        )

    seen: set[int] = set()

    def _walk(node: Any) -> None:
        if isinstance(node, Mapping):
            reps, samples = node.get("timed_repeats"), node.get("seconds_all")
            if reps is not None and isinstance(samples, list):
                if int(reps) != len(samples):
                    raise ValueError(
                        f"enforce_consistent_repeat_counts: timing block claims timed_repeats={reps} but carries "
                        f"{len(samples)} sample(s) in seconds_all. A single sample makes seconds_iqr and "
                        f"seconds_stddev structural zeros, which is not evidence of reproducibility."
                    )
                seen.add(int(reps))
            for value in node.values():
                _walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                _walk(item)

    _walk(payload.get("results", []))
    if used is not None and seen and int(used) not in seen:
        raise ValueError(
            f"enforce_consistent_repeat_counts: provenance.inputs.timed_repeats={used} but the measured rows timed "
            f"{sorted(seen)} sample(s). The declared repeat count is not what ran."
        )


def _total_memory_bytes() -> int:
    """Total physical RAM in bytes, via ``psutil`` when present and ``os.sysconf`` otherwise.

    Scaling numbers without the machine's memory are meaningless, so this is a hard requirement of every
    artifact rather than a nicety.
    """
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except Exception:  # pragma: no cover - non-POSIX fallback
        return -1


#: cgroup v2 files that bound this process's memory, probed in order. A benchmark that scales its blocks up to a
#: large fraction of the box must record the CAP it ran under, not just the physical RAM: the cap is what can
#: actually kill the run, and on a shared host the two differ.
_CGROUP_MEMORY_FILES: tuple[str, ...] = (
    "/sys/fs/cgroup/memory.max",  # cgroup v2
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
)


def cgroup_memory_limit_bytes() -> int | None:
    """Memory CAP on this process in bytes, or ``None`` when no cgroup cap is in force.

    Distinguishes three cases rather than collapsing them, because only two of the three are numbers:

    - a finite ``memory.max`` is the usable cap;
    - cgroup v2 spells "no limit" as the literal string ``"max"``, which is NOT a byte count and must never be
      parsed as one;
    - an absent file means no cgroup cap at all, i.e. the physical RAM is the only bound.

    Returning ``None`` rather than a sentinel keeps a reader from quoting ``-1`` or ``"max"`` as a cap.
    """
    for candidate in _CGROUP_MEMORY_FILES:
        try:
            raw = Path(candidate).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not raw or raw == "max":
            return None
        try:
            return int(raw)
        except ValueError:
            continue
    return None


def cgroup_memory_current_bytes() -> int | None:
    """Bytes currently charged to this cgroup, or ``None`` when unavailable."""
    for candidate in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            raw = Path(candidate).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return None


def peak_rss_bytes() -> int:
    """This process's peak resident set size in bytes, from the kernel's own high-water mark.

    ``/proc/self/status``'s ``VmHWM`` is used rather than ``resource.getrusage``, because ``getrusage`` reports
    ``ru_maxrss`` which is ALSO a process-lifetime maximum: once one big configuration has run, it can no longer
    attribute a peak to the configuration measured after it. Both are monotone over the process lifetime, so
    neither can answer "what did this call allocate".

    Returns ``-1`` off Linux, where neither file exists, rather than raising: a missing peak is a defect to be
    recorded, not a reason to lose a measurement that already succeeded.
    """
    try:
        with open("/proc/self/status", encoding="utf-8") as fp:
            for line in fp:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:  # pragma: no cover - non-Linux
        return -1
    return -1


def current_rss_bytes() -> int:
    """This process's current resident set size in bytes, or ``-1`` when unavailable."""
    try:
        with open("/proc/self/status", encoding="utf-8") as fp:
            for line in fp:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:  # pragma: no cover - non-Linux
        return -1
    return -1


def reset_peak_rss() -> bool:
    """Reset the peak-RSS high-water mark to the current RSS, so the NEXT configuration's peak is attributable.

    Writes ``5`` to ``/proc/self/clear_refs``, the documented way to zero the ``VmHWM`` watermark. Without this a
    sweep that grows its blocks has exactly one attributable peak -- the largest -- and every smaller row's
    "peak" is really just that one number, which would make the per-configuration memory column a lie for every
    row except the last.

    Returns ``False`` when the reset is not permitted (a hardened kernel, a non-Linux host, a read-only
    ``procfs``). A caller must then treat its peak column as process-wide rather than per-configuration, because
    that is exactly what it has.
    """
    try:
        with open("/proc/self/clear_refs", "w", encoding="ascii") as fp:
            fp.write("5\n")
    except OSError:
        return False
    return True


def machine_info() -> dict[str, Any]:
    """Describe the machine the measurement ran on: CPU, memory, and the BLAS/LAPACK that does the SVD.

    ``numpy.show_config`` output is included because the SVD timing and the roundoff floor both depend on the
    LAPACK build; a time-to-result number without it cannot be compared across machines.
    """
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)
    except Exception:
        physical = None
        logical = os.cpu_count()

    blas = "unavailable"
    try:
        cfg = np.__config__.CONFIG  # numpy >= 2
        blas = str(cfg.get("Build Dependencies", {}).get("blas", {}).get("name", "unknown"))
        lapack = str(cfg.get("Build Dependencies", {}).get("lapack", {}).get("name", "unknown"))
    except Exception:
        lapack = "unavailable"

    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count_logical": logical,
        "cpu_count_physical": physical,
        "total_memory_bytes": _total_memory_bytes(),
        "cgroup_memory_limit_bytes": cgroup_memory_limit_bytes(),
        "cgroup_memory_current_bytes": cgroup_memory_current_bytes(),
        "numpy_blas": blas,
        "numpy_lapack": lapack,
        "thread_env": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }


def provenance(script: str, description: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build the provenance block every artifact carries.

    Contains the UTC timestamp, the machine, the versions of every library a baseline or the measurement itself
    depends on, the seed, the timing policy, the deisa-dask commit, and the pinned gysela sources. A reader can
    re-run the script and compare like for like.

    ``timing_policy`` reports :data:`TIMING_POLICY`'s DEFAULTS, so a script that runs fewer repeats than the default
    MUST restate it or the artifact claims a repeat count it did not use. The repeat count actually in force is
    stamped by the caller under ``inputs``; ``timed_repeats_matches_measured`` cross-checks the two at write time so
    the contradiction cannot ship unnoticed.

    - ``:param script:`` Name of the experiment script, e.g. ``"b1_bytes"``.
    - ``:param description:`` One line stating what the artifact measures.
    - ``:param extra:`` Optional additional provenance (inputs, sweep definition, policy overrides).
    """
    payload: dict[str, Any] = {
        "script": script,
        "description": description,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "machine": machine_info(),
        "timing_policy": dict(TIMING_POLICY),
        "versions": {
            "numpy": _library_version("numpy"),
            "dask": _library_version("dask"),
            "distributed": _library_version("distributed"),
            "deisa-dask": _library_version("deisa-dask"),
            "scipy": _library_version("scipy"),
            "scikit-learn": _library_version("scikit-learn"),
            "dask-ml": _library_version("dask-ml"),
            "psutil": _library_version("psutil"),
        },
        "deisa_dask_commit": _git_commit(_repo_root(Path(__file__).resolve())),
        "gysela_sources": GYSELA_SOURCES,
        "disclaimer": (
            "The gysela mesh extents used by the sizing experiments are SYNTHETIC parameter points chosen to span "
            "a range of shapes in the same family as the application's index ranges. They are NOT read from a "
            "production input file and no gysela build was run on this machine; the arrays are simulated in numpy "
            "from the TYPE and LAYOUT semantics of the pinned C++ sources."
        ),
    }
    if extra:
        payload.update(dict(extra))
    return payload


# ----------------------------------------------------------------------------- JSON output
class NpEncoder(json.JSONEncoder):
    """JSON encoder for numpy scalars and arrays, so results need no lossy pre-conversion.

    NaN and infinity are emitted as the JSON-invalid tokens ``NaN``/``Infinity`` by default, which would make the
    artifact unreadable by a strict parser. They are instead mapped to strings so a reader sees WHY a value is
    missing instead of a parse error.
    """

    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            value = float(o)
            if np.isnan(value):
                return "NaN"
            if np.isinf(value):
                return "Infinity" if value > 0 else "-Infinity"
            return value
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def write_result(script: str, payload: Mapping[str, Any]) -> Path:
    """Write one artifact to ``results/<script>.json`` and return its path.

    Enforces the sign-invariant rule first (:func:`enforce_sign_invariant_results`), so a regression that
    reintroduces a raw component error fails loudly here instead of quietly shipping a broken headline figure.

    - ``:param script:`` Script name, used as the artifact file stem.
    - ``:param payload:`` The full artifact, provenance block included.
    """
    enforce_sign_invariant_results(payload)
    enforce_consistent_repeat_counts(payload)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{script}.json"
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, cls=NpEncoder, allow_nan=False)
    return path


#: Keys that must never appear in an emitted result set. A raw component error is non-monotonic and reads O(1)
#: even for identical subspaces whose eigenvector signs differ, so its presence invalidates the curve it appears in.
FORBIDDEN_METRIC_KEYS = frozenset(
    {
        "component_error",
        "component_error_max",
        "raw_component_error",
        "component_max_abs_error",
        "max_abs_component_error",
        "componentwise_error",
        "sign_sensitive_error",
    }
)


def enforce_sign_invariant_results(payload: Mapping[str, Any], _path: str = "result") -> None:
    """Refuse to emit a result set containing a sign-SENSITIVE accuracy metric.

    Walks the artifact recursively and raises on any forbidden key. This is a cheap structural gate: it cannot
    judge whether a metric is mathematically sound, but it does guarantee nobody reintroduces raw component error
    as an "accuracy" figure without the run failing.

    - ``:param payload:`` The artifact about to be written.
    - ``:param _path:`` Dotted location, used only to build the failure message.
    """
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if isinstance(key, str) and key.lower() in FORBIDDEN_METRIC_KEYS:
                raise ValueError(
                    f"enforce_sign_invariant_results: found sign-sensitive metric {key!r} at {_path}. Eigenvector "
                    "signs are arbitrary, so a raw component error is non-monotonic and reads O(1) even for "
                    "identical subspaces. Use subspace_distance() and variance_errors() instead."
                )
            enforce_sign_invariant_results(value, f"{_path}.{key}")
    elif isinstance(payload, (list, tuple)):
        for index, item in enumerate(payload):
            enforce_sign_invariant_results(item, f"{_path}[{index}]")


# ----------------------------------------------------------------------------- sign-invariant accuracy
def subspace_distance(basis_a: np.ndarray, basis_b: np.ndarray) -> float:
    """Sign-invariant distance between two subspaces, ``1 - min(svd(A @ B.T))``.

    ``A`` and ``B`` are ORTHONORMAL ROW BASES of equal length ``k`` (i.e. ``A @ A.T == B @ B.T == I``), as produced by
    the rows of ``numpy.linalg.svd``'s ``Vh``. The smallest singular value of ``A @ B.T`` is the cosine of the
    smallest principal angle between the two spans, so ``1 - min(svd(A @ B.T))`` is 0 exactly when the spans
    coincide and 1 exactly when they are orthogonal. Unlike a component-wise error it is invariant to sign flips
    AND to any rotation inside the retained subspace, which is the property that makes it the right accuracy
    measure for a truncated PCA.

    - ``:param basis_a:`` ``(k, d)`` orthonormal rows.
    - ``:param basis_b:`` ``(k, d)`` orthonormal rows, same ``k`` as ``basis_a``.
    """
    A = np.asarray(basis_a, dtype=np.float64)
    B = np.asarray(basis_b, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(
            f"subspace_distance: both bases must be 2-dimensional (rank, n_features), got ndim {A.ndim} and {B.ndim}."
        )
    if A.shape != B.shape:
        raise ValueError(
            f"subspace_distance: bases must have equal shape to compare spans of equal rank, got {A.shape} and "
            f"{B.shape}. Compare the same number of retained components on both sides."
        )
    k = A.shape[0]
    if k == 0:
        return 0.0
    sigma_min = float(np.linalg.svd(A @ B.T, compute_uv=False)[-1])
    # Guard the rounding of the singular value against leaving [0, 1]: the metric is a cosine and must read as one.
    return float(min(1.0, max(0.0, 1.0 - sigma_min)))


def variance_errors(
    singular_values: np.ndarray,
    n_samples: int,
    reference_singular_values: np.ndarray,
    reference_n_samples: int,
    n_components: int | None = None,
) -> dict[str, float]:
    """Sign-invariant explained-variance errors of one summary against an exact pooled SVD reference.

    Operates on SINGULAR VALUES, which are sign-free, so nothing here can be perturbed by an eigenvector sign flip.
    Three numbers are returned:

    - ``explained_variance_ratio_error``: sum of absolute differences of the per-component explained-variance
      ratios, against the exact reference. 0 means the variance profile matches exactly.
    - ``captured_variance_fraction``: fraction of the TOTAL variance that the retained components capture, so a
      truncation below 1 quantifies how much variance the approximation threw away.
    - ``total_variance_relative_error``: relative error of the total variance itself, which detects a mean-correction
      error (a broken between-block correction shows up here even when the subspace is right).

    - ``:param singular_values:`` Retained singular values of the approximation, descending.
    - ``:param n_samples:`` Sample count the approximation represents.
    - ``:param reference_singular_values:`` FULL singular values of the exact pooled centered data.
    - ``:param reference_n_samples:`` Sample count of the reference.
    - ``:param n_components:`` How many leading components to score, or ``None`` for all supplied.
    """
    approx = np.asarray(singular_values, dtype=np.float64)
    exact = np.asarray(reference_singular_values, dtype=np.float64)
    keep = approx.shape[0] if n_components is None else min(int(n_components), approx.shape[0], exact.shape[0])

    # ddof=1 matches scikit-learn's PCA; a single sample has no unbiased variance, hence the guard.
    approx_var = approx[:keep] ** 2 / (n_samples - 1) if n_samples > 1 else np.zeros(keep)
    exact_var = exact[:keep] ** 2 / (reference_n_samples - 1) if reference_n_samples > 1 else np.zeros(keep)

    approx_total = float(np.sum(approx**2)) / (n_samples - 1) if n_samples > 1 else 0.0
    exact_total = float(np.sum(exact**2)) / (reference_n_samples - 1) if reference_n_samples > 1 else 0.0

    if exact_total > 0.0:
        approx_ratio = approx_var / exact_total
        exact_ratio = exact_var / exact_total
        ratio_error = float(np.sum(np.abs(approx_ratio - exact_ratio)))
        total_error = float(abs(approx_total - exact_total) / exact_total)
    else:
        ratio_error = float("nan")
        total_error = float("nan")

    if approx_total > 0.0 and exact_total > 0.0:
        captured = float(np.sum(approx_var) / exact_total)
    else:
        captured = float("nan")

    return {
        "n_components_scored": int(keep),
        "explained_variance_ratio_error": ratio_error,
        "captured_variance_fraction": captured,
        "total_variance_relative_error": total_error,
        "approx_total_variance": approx_total,
        "reference_total_variance": exact_total,
    }


# ----------------------------------------------------------------------------- regimes and shapes
def regime_of(n_block: int, n_features: int) -> str:
    """Classify a block shape as ``"tall"``, ``"square"`` or ``"flat"`` by its ``n_block / d`` ratio.

    The split is not cosmetic. A FULL-RANK (``local_rank = d``) summary carries ``min(n_block, d) * d + d``
    elements against ``n_block * d`` for the data, so it compresses only when ``n_block > d``. Reporting the tall
    regime alone would be cherry-picking, which is why every experiment sweeps both and tags each row with this.

    - ``:param n_block:`` Rows in the block (samples).
    - ``:param n_features:`` Columns in the block (features).
    """
    if n_block > n_features:
        return "tall"
    if n_block < n_features:
        return "flat"
    return "square"


def summary_nbytes(summary: Any, itemsize: int = 8) -> int:
    """Bytes a :class:`~deisa.dask.mergeable_pca.PCASummary` occupies as raw float64 payload.

    Counts exactly the arrays that cross the boundary: ``components``, ``mean`` and ``singular_values``. The
    ``n_samples`` count rides in the pickle header, not in an array, so it is not double counted.

    - ``:param summary:`` Any object exposing ``components``, ``mean`` and ``singular_values`` arrays.
    - ``:param itemsize:`` Bytes per element of those arrays.
    """
    total = 0
    for field in ("components", "mean", "singular_values"):
        value = getattr(summary, field, None)
        if value is not None:
            total += int(np.asarray(value).size) * itemsize
    return total


def summary_elements(summary: Any) -> int:
    """Element COUNT of a summary's arrays, the unit the compression ratios are computed in.

    - ``:param summary:`` Any object exposing ``components``, ``mean`` and ``singular_values`` arrays.
    """
    total = 0
    for field in ("components", "mean", "singular_values"):
        value = getattr(summary, field, None)
        if value is not None:
            total += int(np.asarray(value).size)
    return total


def make_block(
    n_block: int,
    n_features: int,
    rank: int,
    seed: int,
    noise: float = 0.05,
) -> np.ndarray:
    """Build a deterministic synthetic block with a KNOWN intrinsic rank, for the accuracy experiments.

    The block is ``low-rank signal + isotropic noise``: a rank-``r`` signal drawn once and broadcast, plus a noise
    matrix scaled so the signal dominates. That gives an exactly known principal subspace, so the sign-invariant
    distance of an approximation to the exact one is a meaningful, interpretable number rather than a comparison
    between two arbitrary bases.

    - ``:param n_block:`` Rows (samples).
    - ``:param n_features:`` Columns (features).
    - ``:param rank:`` Intrinsic rank of the signal, clipped to ``min(n_block, n_features)``.
    - ``:param seed:`` Seed for this stream.
    - ``:param noise:`` Relative amplitude of the isotropic noise.
    """
    rng = np.random.default_rng(seed)
    effective_rank = max(1, min(int(rank), n_block, n_features))
    signal = rng.standard_normal((n_features, effective_rank))
    factors = rng.standard_normal((n_block, effective_rank))
    low_rank = factors @ signal.T
    noise = rng.standard_normal((n_block, n_features)) * (noise * float(np.std(low_rank)))
    return np.ascontiguousarray(low_rank + noise, dtype=np.float64)


# ----------------------------------------------------------------------------- timing
def time_repeated(
    func: Callable[[], Any],
    warmup_rounds: int | None = None,
    repeats: int | None = None,
) -> dict[str, Any]:
    """Time ``func`` under the shared warmup/repeat policy and report the spread, not just the headline.

    Returns the median as the headline statistic plus min/max/iqr/stddev, so a reader can judge whether a
    difference between two arms is larger than the run-to-run noise. Reporting a bare median would make a 2%
    difference look like signal.

    - ``:param func:`` Zero-argument callable to time; its return value is discarded.
    - ``:param warmup_rounds:`` Untimed warmup calls, or ``None`` for :data:`TIMING_POLICY`'s value.
    - ``:param repeats:`` Timed calls, or ``None`` for :data:`TIMING_POLICY`'s value.
    """
    warm = TIMING_POLICY["warmup_rounds"] if warmup_rounds is None else int(warmup_rounds)
    reps = TIMING_POLICY["timed_repeats"] if repeats is None else int(repeats)
    if reps < 1:
        raise ValueError(f"time_repeated: repeats must be >= 1, got {reps}.")

    for _ in range(warm):
        func()

    samples: list[float] = []
    for _ in range(reps):
        start = time.perf_counter()
        func()
        samples.append(time.perf_counter() - start)

    ordered = sorted(samples)
    return {
        "seconds_median": statistics.median(samples),
        "seconds_min": ordered[0],
        "seconds_max": ordered[-1],
        "seconds_iqr": float(np.percentile(samples, 75) - np.percentile(samples, 25)),
        "seconds_stddev": float(np.std(samples)),
        "seconds_all": samples,
        "warmup_rounds": warm,
        "timed_repeats": reps,
    }


# ----------------------------------------------------------------------------- wire bytes
def _frame_nbytes(frame: Any) -> int:
    """Recursive byte count over one serialized frame, which may be a buffer, a list or a dict.

    ``distributed.protocol.serialize`` returns frames whose type depends on the payload (a memoryview for a bare
    array, a list of frames for a nested object), so a single ``len(frame)`` is not enough and a wrong answer here
    would silently corrupt the lead figure.

    - ``:param frame:`` One frame of a serialized message.
    """
    nbytes = getattr(frame, "nbytes", None)
    if nbytes is not None and not isinstance(frame, (bytes, bytearray)):
        return int(nbytes)
    if isinstance(frame, (bytes, bytearray, memoryview)):
        return len(memoryview(frame))
    if isinstance(frame, (list, tuple)):
        return sum(_frame_nbytes(item) for item in frame)
    if isinstance(frame, Mapping):
        return sum(_frame_nbytes(item) for item in frame.values())
    return 0


def serialized_nbytes(obj: Any) -> int:
    """Bytes ``obj`` occupies ON THE WIRE, measured with the SAME serializer the bridge scatters with.

    Uses ``distributed.protocol.serialize(to_serialize(obj))`` -- the exact call
    :meth:`~deisa.dask.bridge.Bridge._scatter_partials` makes before ``scatter_to_workers`` -- so the number is the
    payload size the scheduler actually puts on the wire, including any framing, not an arithmetic guess at
    ``array.nbytes``. For a bare float64 array the two differ slightly (the pickle adds a header), and the wire
    number is the honest one.

    - ``:param obj:`` The object that would be scattered.
    """
    from distributed.protocol import serialize, to_serialize

    message = serialize(to_serialize(obj))
    frames = message[3] if isinstance(message, (list, tuple)) and len(message) == 4 else message
    return int(sum(_frame_nbytes(frame) for frame in frames))


def byte_dict(nbytes: int) -> dict[str, float]:
    """Render a byte count in every unit a reader might quote, so no artifact needs a hand conversion.

    - ``:param nbytes:`` Byte count.
    """
    value = float(nbytes)
    return {
        "bytes": value,
        "KiB": value / 2**10,
        "MiB": value / 2**20,
        "GiB": value / 2**30,
    }


def ratio_or_none(numerator: float, denominator: float) -> float | None:
    """``numerator / denominator``, or ``None`` when the denominator is zero.

    Returns ``None`` rather than infinity so the artifact stays strictly valid JSON and a reader sees an explicit
    "undefined" instead of a number that looks meaningful.

    - ``:param numerator:`` The quantity being scaled.
    - ``:param denominator:`` The scaling base.
    """
    if denominator == 0:
        return None
    return float(numerator / denominator)


def safe_import(module: str, attribute: str | None = None) -> Any:
    """Import an optional benchmark dependency, returning ``None`` when it is unavailable.

    B7's baselines are mandatory, so an unavailable one is recorded as an explicit finding in the artifact rather
    than crashing the run: a missing baseline must be visible in the data, not in a traceback.

    - ``:param module:`` Module path, e.g. ``"dask_ml.decomposition"``.
    - ``:param attribute:`` Attribute to take from the module, or ``None`` for the module itself.
    """
    try:
        mod = importlib.import_module(module)
    except Exception:
        return None
    if attribute is None:
        return mod
    return getattr(mod, attribute, None)


def ratio_sweep_points(d: int) -> list[tuple[float, int]]:
    """The ``n_block / d`` ratios every regime sweep must span, paired with the resulting ``n_block``.

    The card requires roughly 0.25, 0.5, 1, 2, 4 and 8. Ratios below 1 are the FLAT regime where a full-rank summary
    is not a compression at all, and ratios above 1 are the TALL regime where it is. ``n_block`` is rounded to an
    integer and forced to be at least 1, because a ratio is meaningless if the block has no rows.

    - ``:param d:`` Feature dimension of the block.
    """
    if d < 1:
        raise ValueError(f"ratio_sweep_points: feature dimension must be >= 1, got {d}.")
    points: list[tuple[float, int]] = []
    for ratio in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        n_block = max(1, int(round(ratio * d)))
        points.append((ratio, n_block))
    return points


def _render_bytes(value: float) -> str:
    """Render a byte count in a unit that suits its magnitude, so one column can hold five orders of size.

    A fixed unit is wrong the moment a sweep spans it: a byte dict printed only in MiB reads "0.00" for the small
    end of a sweep whose large end is tens of GiB. Each unit is used only within its own decade, so the smallest
    non-zero value never renders as "0.00".

    - ``:param value:`` Byte count, in the unit ``value`` was passed in.
    """
    for threshold, unit, scale in ((2**30, "GiB", 2**30), (2**20, "MiB", 2**20), (2**10, "KiB", 2**10)):
        if value >= threshold:
            return f"{value / scale:.2f} {unit}"
    return f"{value:.0f} B" if value > 0 else "0 B"


def print_summary_table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[tuple[str, Any, str]],
    title: str = "",
    stream: Any = None,
) -> None:
    """Print result rows as a fixed-width table, for a human running the script.

    The JSON is the artifact; this only shows its shape without opening it, and the values are rendered from the
    same rows that are written to JSON -- nothing is recomputed or reformatted numerically, so the display cannot
    become a second, divergent source of numbers. Byte-valued columns are rendered in a unit chosen per row by
    :func:`_render_bytes` for legibility; the JSON keeps the exact byte count.

    - ``:param rows:`` Result rows.
    - ``:param columns:`` ``(key, header, kind)`` triples in display order, where ``kind`` is one of ``"int"``,
        ``"float"``, ``"str"``, ``"mib"`` (renders a ``{"MiB": ...}`` byte dict) or ``"auto"`` (renders a
        ``{"bytes": ...}`` byte dict in whichever unit suits the value).
    - ``:param title:`` Optional heading printed above the table.
    - ``:param stream:`` Output stream, or ``None`` for stdout.
    """
    out = stream if stream is not None else sys.stdout

    def _cell(row: Mapping[str, Any], key: str, kind: str) -> str:
        if key not in row:
            return "-"
        value = row[key]
        if kind in ("mib", "auto"):
            if isinstance(value, Mapping) and "bytes" in value:
                exact = float(value["bytes"])
                return _render_bytes(exact) if kind == "auto" else f"{float(value['MiB']):.2f}"
            if kind == "auto" and isinstance(value, (int, float)):
                return _render_bytes(float(value))
            return "-"
        if kind == "float":
            if value is None:
                return "-"
            try:
                return f"{float(value):.4g}"
            except (TypeError, ValueError):
                return str(value)
        if kind == "int":
            return str(value)
        return str(value)

    widths = {key: len(header) for key, header, _ in columns}
    rendered: list[list[str]] = []
    for row in rows:
        line = [_cell(row, key, kind) for key, _, kind in columns]
        for (key, _, _), text in zip(columns, line, strict=True):
            widths[key] = max(widths[key], len(text))
        rendered.append(line)

    if title:
        print(title, file=out)
    print("  ".join(header.ljust(widths[key]) for key, header, _ in columns), file=out)
    print("  ".join("-" * widths[key] for key, _, _ in columns), file=out)
    for line in rendered:
        print("  ".join(text.ljust(widths[key]) for text, (key, _, _) in zip(line, columns, strict=True)), file=out)
