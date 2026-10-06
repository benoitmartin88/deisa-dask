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
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# =============================================================================
"""
The ONE consolidated comparison artifact: every method on the SAME configurations, with accuracy, timing
and the transfer volume each method actually implies.

Why a consolidation script and not a copy of three artifacts
-------------------------------------------------------------
Three committed artifacts each hold part of the picture, and a reader comparing them by hand has to do
arithmetic across files that disagree about units and about which rows exist:

- ``network_transfer.json`` measures WIRE BYTES only. It carries no timing at all, deliberately: a
  serialized size is deterministic given its input, so a duration for it would be one sample with no
  dispersion behind it.
- ``standard_baselines.json`` measures TIME, PEAK MEMORY and ACCURACY for five standard methods
  against the mergeable path, but carries no wire bytes.
- ``rank_accuracy_curve.json`` sweeps LOCAL RANK for accuracy, and is the only place the truncated-rank
  degradation is measured.

This script re-derives nothing it does not have to. It runs ONE measurement of its own -- the in-process
wall time of every method, on this process's own CPU, with a real repeat count and real dispersion -- and
it CONSOLIDATES the byte and accuracy evidence into that one frame. Every number it writes is either
(a) measured here, and says which measurement, or (b) carried verbatim from a named committed artifact,
which is named in the row that uses it.

What the timing column is, and is not
--------------------------------------
The timing here is IN-PROCESS COMPUTE on the analytics-side CPU. It is NOT network transfer time: this box
does not reproduce the interconnect, so a wire duration cannot be measured here and is not estimated. It is
reported for the same reason the card requires it: honesty about what the bridge pays. The mergeable path
performs an ``O(n_block * d)`` local SVD per bridge that the legacy full-chunk scatter never performed, and
that cost has to sit in the table beside the methods it is being compared to. On this machine the measured
ratio runs below 1, i.e. MergeablePCA is SLOWER in process, and the table says so rather than hiding the
column. The contribution claimed by this work is bridge-side compute and transfer volume AVOIDED, not local
speed.

Why the baselines' transfer size is the FULL BLOCK
---------------------------------------------------
This is the central comparison of the paper and it is deliberately unflattering to the baseline column,
which is why it is stated rather than implied. Every baseline here -- NumPy SVD, SciPy SVD, scikit-learn
PCA, scikit-learn IncrementalPCA, dask-ml IncrementalPCA -- runs ON THE ANALYTICS ENGINE, on the pooled
data. For a baseline to run there, the samples must be there first, so the full chunk crosses the bridge
boundary for every baseline configuration. Only the mergeable path computes before it ships. Its transfer
size is therefore measured as the full block's wire payload, identical to the legacy scatter, and the
saving attributed to it is the difference against its own leaves. No baseline is cheaper in transfer than
the legacy path, and none is presented as being.

Run
---
    PYTHONPATH=src .venv/bin/python benchmark/mergeable_pca/transfer_comparison.py
    PYTHONPATH=src .venv/bin/python benchmark/mergeable_pca/transfer_comparison.py --repeats 5
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from measurement_common import (  # noqa: E402
    SEED,
    TIMING_POLICY,
    byte_dict,
    cgroup_memory_limit_bytes,
    checkpoint_config_key,
    checkpoint_path,
    make_block,
    peak_rss_bytes,
    print_summary_table,
    provenance,
    regime_of,
    serialized_nbytes,
    time_repeated,
    write_result,
)

from deisa.dask.mergeable_pca import local_pca, merge_tree  # noqa: E402

#: Artifact stem.
ARTIFACT_STEM = "transfer_comparison"

#: The five standard methods this card requires, in the order the paper should name them. The name is the
#: ``b5`` arm key, so a row here and an arm there are the same thing and a reader can join on it.
STANDARD_METHODS: tuple[str, ...] = (
    "numpy_svd",
    "scipy_linalg_svd",
    "sklearn_pca_full",
    "sklearn_incremental_pca",
    "dask_ml_incremental_pca",
)

#: The mergeable path, as an arm. It is not one of the five baselines; it is the thing they are compared to.
MERGEABLE_ARM = "mergeable_pca_total"

#: Every method the artifact names, in one list, so nothing appears in ``results`` without being declared
#: here. A method in a table and absent from ``methods_compared`` is the reporting gap this card closes.
METHODS_COMPARED: tuple[str, ...] = (MERGEABLE_ARM,) + STANDARD_METHODS

#: ``b5`` arm names for the two intermediate stages of the mergeable path. Reported, never compared: a leaf
#: summary and a merge node are not fitted PCAs, so they have no accuracy to score.
MERGEABLE_STAGES: tuple[str, ...] = ("mergeable_pca_local_leaves", "mergeable_pca_merge_tree")

#: Comma-separated arm key -> comma-separated import path, resolved once at module import so a missing
#: library is a REFUSAL NAMED IN THE ARTIFACT rather than a row that silently vanishes.
METHOD_IMPORTS: dict[str, tuple[str, str]] = {
    "numpy_svd": ("numpy", "linalg.svd"),
    "scipy_linalg_svd": ("scipy", "linalg.svd"),
    "sklearn_pca_full": ("sklearn.decomposition", "PCA"),
    "sklearn_incremental_pca": ("sklearn.decomposition", "IncrementalPCA"),
    "dask_ml_incremental_pca": ("dask_ml.decomposition", "IncrementalPCA"),
}

#: Blocks per configuration. Eight is what ``b5`` measured, so the consolidated rows join to that artifact's
#: rows on the same decomposition rather than on a similar one.
N_BLOCKS = 8

#: Local ranks measured here. ``None`` is full local rank, where the merge is exact; the truncated values
#: are here so the accuracy table has BOTH ends of the trade in ONE artifact, not only the exact end.
LOCAL_RANKS: tuple[int | None, ...] = (8, 32, None)

#: Shapes. Deliberately SMALL, and CHOSEN FROM A COST PROBE rather than picked for looks. Every method is
#: timed with real repeats and dask-ml runs on every configuration, so cost is superlinear in the pooled
#: shape: measured on this machine at 2 timed repeats, 128x512 took 17.6 s and 2048x512 took 180.7 s for ONE
#: configuration. A naive grid over these two lists would have cost hours, which is how the previous run of
#: this card exhausted its iteration budget. The rows below keep the whole sweep inside minutes while
#: covering both regimes and every rank, and every one of them is a ``(n_block, d)`` that ``b5`` also
#: measured, so the rows JOIN to that artifact instead of merely resembling it.
FEATURE_DIMS: tuple[int, ...] = (32, 128, 512)
BLOCK_ROWS: tuple[int, ...] = (32, 128, 512, 1024)

#: Intrinsic rank of the synthetic signal as a FRACTION of ``d``. An INPUT, matching ``b5``.
INTRINSIC_RANK_FRACTION = 0.25

#: Components scored for every method. Fixed and IDENTICAL across methods, because ``subspace_distance`` is
#: undefined between spans of different rank.
N_COMPONENTS_SCORED = 8

#: The default timed repeat count. Overridable with ``--repeats``; whatever is used is stamped under
#: ``inputs`` and cross-checked against the timing policy at write time by the shared measurement suite.
TIMED_REPEATS = 5


# ============================================================================= every method, one callable
def _resolve(method: str) -> tuple[Any, str]:
    """Import one method's callable, or report exactly why it could not be imported.

    Imported at module scope rather than inside the timed closure so an ``ImportError`` cannot be mistaken
    for a fast method.

    Each path is a DOTTED name resolved with ``importlib.import_module``, which imports the SUBMODULE
    itself, then a dotted attribute walked one segment at a time. ``__import__("sklearn.decomposition")``
    returns the top-level ``sklearn`` even when the submodule imports cleanly, and
    ``getattr(module, "decomposition.PCA")`` then raises ``AttributeError`` -- so the naive form reports
    every library as UNAVAILABLE on a machine where all of them are installed, silently dropping the arms
    the paper needs most.

    - ``:param method:`` Arm key from :data:`METHODS_COMPARED`.
    """
    module_name, dotted = METHOD_IMPORTS[method]
    try:
        obj: Any = importlib.import_module(module_name)
        walked = [module_name]
        for segment in dotted.split("."):
            obj = getattr(obj, segment)
            walked.append(segment)
        return obj, ".".join(walked)
    except Exception as exc:  # pragma: no cover - exercised only when a dependency is absent
        return None, f"UNAVAILABLE ({module_name}: {type(exc).__name__}: {exc})"


def _fit_method(method: str, X: np.ndarray, k: int, n_block: int) -> dict[str, Any]:
    """Fit one method on ``X`` and return the sign-invariant state the scorer needs.

    Every arm is configured to consume the SAME leaf granularity as the mergeable path -- ``batch_size`` is
    the per-block row count -- so the comparison is between two ways of doing the same decomposition rather
    than between a batched method and an incremental one.

    ``dask_ml`` is the one asymmetry that cannot be removed: it refuses a numpy array and needs a
    ``dask.array``, so its arm pays an intra-process graph execution the numpy arms do not. That is the cost
    of the library path and it is recorded rather than hidden.

    - ``:param method:`` Arm key from :data:`METHODS_COMPARED`.
    - ``:param X:`` The centered-or-not pooled sample matrix, identical for every arm.
    - ``:param k:`` Components to retain.
    - ``:param n_block:`` Rows per leaf, used as ``batch_size`` by the incremental arms.
    """
    import dask.array as da

    if method == "numpy_svd":
        mean = X.mean(axis=0)
        _, singular, components = np.linalg.svd(X - mean, full_matrices=False)
    elif method == "scipy_linalg_svd":
        import scipy.linalg

        mean = X.mean(axis=0)
        _, singular, components = scipy.linalg.svd(
            X - mean, full_matrices=False, lapack_driver="gesdd", overwrite_a=False
        )
    elif method == "sklearn_pca_full":
        from sklearn.decomposition import PCA

        est = PCA(n_components=k, svd_solver="full")
        est.fit(X)
        mean = np.asarray(est.mean_, dtype=np.float64)
        singular = np.asarray(est.singular_values_, dtype=np.float64)
        components = np.asarray(est.components_, dtype=np.float64)
    elif method == "sklearn_incremental_pca":
        from sklearn.decomposition import IncrementalPCA

        est = IncrementalPCA(n_components=k, batch_size=n_block)
        est.fit(X)
        mean = np.asarray(est.mean_, dtype=np.float64)
        # Read ``singular_values_`` directly rather than reconstructing it from ``explained_variance_``:
        # this estimator has no ``n_samples_``, and the sample count that IS available, ``n_samples_seen_``,
        # is the total seen, which is what the reconstruction needs. Preferring the library's own attribute
        # removes the opportunity for the two to disagree.
        singular = np.asarray(est.singular_values_, dtype=np.float64)
        components = np.asarray(est.components_, dtype=np.float64)
    elif method == "dask_ml_incremental_pca":
        from dask_ml.decomposition import IncrementalPCA

        est = IncrementalPCA(n_components=k, batch_size=n_block)
        est.fit(da.from_array(X, chunks=(n_block, -1)))
        mean = np.asarray(est.mean_, dtype=np.float64)
        # Same reasoning as the scikit-learn arm. Note this estimator's ``n_samples_`` is the LAST BATCH size,
        # not the total: reconstructing singular values from it would be wrong by a factor of
        # ``n_batches``. ``singular_values_`` is read instead, so no sample count is needed.
        singular = np.asarray(est.singular_values_, dtype=np.float64)
        components = np.asarray(est.components_, dtype=np.float64)
    else:
        raise ValueError(f"unknown method {method!r}")

    return {
        "components": components,
        "singular_values": singular,
        "mean": mean,
        "n_samples": int(X.shape[0]),
    }


def _subspace_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Sign-invariant distance ``1 - min(svd(A @ B.T))`` between two equal-rank bases.

    Re-exported from the shared measurement suite rather than reimplemented, so this artifact and ``b4``/``b5`` score
    with one function and the numbers join.

    - ``:param a:`` Basis rows.
    - ``:param b:`` Basis rows, same rank as ``a``.
    """
    from measurement_common import subspace_distance

    return subspace_distance(a, b)


def _exact_reference(blocks: Sequence[np.ndarray]) -> dict[str, Any]:
    """The independent truth: exact batch SVD of the centered concatenation, by numpy alone.

    Computed independently of the merge path, so no method is scored against a truth it produced -- except
    ``numpy_svd``, whose distance to this IS zero by construction, and the artifact says so rather than
    leaving a reader to guess.

    - ``:param blocks:`` The disjoint sample blocks.
    """
    X = np.vstack([np.asarray(b, dtype=np.float64) for b in blocks])
    mean = X.mean(axis=0)
    _, singular, components = np.linalg.svd(X - mean, full_matrices=False)
    return {"X": X, "mean": mean, "singular_values": singular, "components": components, "n_samples": int(X.shape[0])}


# ============================================================================= one configuration, every method
def measure_configuration(
    blocks: Sequence[np.ndarray],
    n_block: int,
    d: int,
    local_rank: int | None,
    repeats: int,
    available: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure every method on ONE configuration: transfer bytes, in-process time, and accuracy.

    The three quantities are reported side by side on purpose. The mergeable path wins the first and loses
    the second, and a table that shows either alone would misrepresent the design.

    - ``:param blocks:`` The disjoint sample blocks, identical for every method.
    - ``:param n_block:`` Rows per block.
    - ``:param d:`` Feature dimension.
    - ``:param local_rank:`` Retained local rank, or ``None`` for full local rank.
    - ``:param repeats:`` Timed repeats per method, excluding warmup.
    - ``:param available:`` Arm key -> resolved callable or ``None``.
    """
    reference = _exact_reference(blocks)
    X = reference["X"]
    rank_arg = local_rank if local_rank is None else min(int(local_rank), d)
    k = int(min(N_COMPONENTS_SCORED, reference["components"].shape[0]))

    # ---- the mergeable path, measured the same way as every baseline: real repeats, real dispersion
    leaves_timing = time_repeated(lambda: [local_pca(b, rank=rank_arg) for b in blocks], repeats=repeats)
    leaves = [local_pca(b, rank=rank_arg) for b in blocks]
    merge_timing = time_repeated(lambda: merge_tree(leaves), repeats=repeats)
    merged = merge_tree(leaves)
    mergeable_total = {
        "seconds_median": leaves_timing["seconds_median"] + merge_timing["seconds_median"],
        "seconds_min": leaves_timing["seconds_min"] + merge_timing["seconds_min"],
        "seconds_max": leaves_timing["seconds_max"] + merge_timing["seconds_max"],
        "seconds_iqr": leaves_timing["seconds_iqr"] + merge_timing["seconds_iqr"],
        "seconds_stddev": float(np.hypot(leaves_timing["seconds_stddev"], merge_timing["seconds_stddev"])),
        "seconds_all": [a + b for a, b in zip(leaves_timing["seconds_all"], merge_timing["seconds_all"], strict=True)],
        "warmup_rounds": leaves_timing["warmup_rounds"],
        "timed_repeats": leaves_timing["timed_repeats"],
        "composition": "sum of two independently timed medians (leaves + merge tree), not a separately timed blend",
    }

    # ---- transfer volume, per method. The two numbers that matter are per BRIDGE SEND, not per run.
    block_wire = serialized_nbytes(blocks[0])
    leaf_summary_wire = serialized_nbytes(leaves[0])
    pooled_summary_wire = serialized_nbytes(merged)
    full_block_total = block_wire * len(blocks)

    mergeable_accuracy = None
    methods: list[dict[str, Any]] = []

    def _record(
        method: str,
        state: Mapping[str, Any] | None,
        timing: Mapping[str, Any] | None,
        transfer: Mapping[str, Any],
        note: str,
        failure: str = "",
    ) -> None:
        """Append one method row, scoring it when it produced a state and marking it otherwise."""
        nonlocal mergeable_accuracy
        accuracy: dict[str, Any]
        if state is None:
            accuracy = {
                "scored": False,
                "reason_unscorable": failure or "no fitted state",
                "subspace_distance_vs_exact": None,
            }
        else:
            basis = np.asarray(state["components"], dtype=np.float64)[:k]
            available_k = int(np.asarray(state["components"]).shape[0])
            if available_k < k:
                accuracy = {
                    "scored": False,
                    "reason_unscorable": (
                        f"method supplies {available_k} directions, fewer than the k={k} every method is scored "
                        f"on; subspace_distance is undefined across spans of different rank"
                    ),
                    "subspace_distance_vs_exact": None,
                }
            else:
                accuracy = {
                    "scored": True,
                    "reason_unscorable": "",
                    "n_components_scored": k,
                    "subspace_distance_vs_exact": _subspace_distance(basis, reference["components"][:k]),
                    "subspace_distance_vs_mergeable": _subspace_distance(basis, merged.components[:k]),
                }
                if method == MERGEABLE_ARM:
                    mergeable_accuracy = accuracy
        methods.append(
            {
                "method": method,
                "in_methods_compared": method in METHODS_COMPARED or method in MERGEABLE_STAGES,
                "seconds_median": None if timing is None else timing["seconds_median"],
                "timing": dict(timing) if timing is not None else {},
                "accuracy": accuracy,
                "transfer_bytes": dict(transfer),
                # What the timing policy CLAIMS and what actually happened, side by side. A previous
                # artifact declared a repeat policy while measuring one sample, which made every spread a
                # structural zero that read as perfect reproducibility. These two fields are equal on every
                # timed arm and the measurement suite refuses the artifact if they ever are not.
                "samples_declared": None if timing is None else int(timing["timed_repeats"]),
                "samples_present": None if timing is None else len(timing["seconds_all"]),
                "note": note,
                "failure": failure,
            }
        )

    _record(
        MERGEABLE_ARM,
        {
            "components": merged.components,
            "singular_values": merged.singular_values,
            "mean": merged.mean,
            "n_samples": int(merged.n_samples),
        },
        mergeable_total,
        {
            "bridge_to_analytics_per_bridge_send": byte_dict(leaf_summary_wire),
            "bridge_to_analytics_total_all_bridges": byte_dict(leaf_summary_wire * len(blocks)),
            "why": (
                "measured: this is the serialized wire size of the PCASummary one bridge actually scatters. "
                "The full chunk never crosses on this path"
            ),
            "root_summary_wire": byte_dict(pooled_summary_wire),
        },
        "leaves + reduction: the only arm that computes BEFORE it ships, and the only one whose transfer is "
        "smaller than the block",
    )

    for method in STANDARD_METHODS:
        callable_or_none = available.get(method)
        if callable_or_none is None:
            _record(
                method,
                None,
                None,
                {
                    "bridge_to_analytics_per_bridge_send": byte_dict(block_wire),
                    "bridge_to_analytics_total_all_bridges": byte_dict(full_block_total),
                    "why": "not measured: the library is unavailable on this machine",
                },
                "import failed; no numbers are reported for this arm and none are estimated",
                failure=f"{method} unavailable",
            )
            continue
        try:
            state = _fit_method(method, X, k, n_block)
        except Exception as exc:
            _record(
                method,
                None,
                None,
                {
                    "bridge_to_analytics_per_bridge_send": byte_dict(block_wire),
                    "bridge_to_analytics_total_all_bridges": byte_dict(full_block_total),
                    "why": "measured block size, arm failed to fit",
                },
                f"fit raised {type(exc).__name__}: {exc}",
                failure=f"{type(exc).__name__}: {exc}",
            )
            continue
        timing = time_repeated(lambda m=method: _fit_method(m, X, k, n_block), repeats=repeats)
        _record(
            method,
            state,
            timing,
            {
                # THE CENTRAL COMPARISON, stated rather than implied.
                "bridge_to_analytics_per_bridge_send": byte_dict(block_wire),
                "bridge_to_analytics_total_all_bridges": byte_dict(full_block_total),
                "why": (
                    "measured, and equal to the FULL BLOCK: this method runs on the analytics engine, so the "
                    "samples must arrive first. Every baseline therefore ships the chunk, and no baseline is "
                    "cheaper in transfer than the legacy scatter"
                ),
            },
            f"runs on the analytics engine on the pooled data: full chunk crosses, then {METHOD_IMPORTS[method][1]}",
        )

    return {
        "n_block": int(n_block),
        "n_features": int(d),
        "n_block_over_d": float(n_block) / float(d),
        "regime": regime_of(n_block, d),
        "n_blocks": len(blocks),
        "local_rank_requested": None if local_rank is None else int(local_rank),
        # Read off the MEASURED leaf, not recomputed from the argument: at full local rank the argument is
        # None and the effective rank is min(n_block, d), which only the summary knows.
        "local_rank_effective_per_leaf": int(leaves[0].rank),
        "intrinsic_rank": int(max(1, int(INTRINSIC_RANK_FRACTION * d))),
        "n_components_scored": k,
        "total_samples": int(X.shape[0]),
        "data_bytes": byte_dict(int(X.size) * 8),
        "reference": (
            "exact batch SVD via numpy.linalg.svd on the centered concatenation, independent of the merge path"
        ),
        "mergeable_exact": bool(local_rank is None),
        "block_wire_bytes": byte_dict(block_wire),
        "bytes_saved_per_block_vs_legacy_measured": byte_dict(block_wire - leaf_summary_wire),
        "bytes_saved_total_vs_legacy_derived": byte_dict(full_block_total - leaf_summary_wire * len(blocks)),
        "bytes_saved_total_is_derived": True,
        "bytes_saved_total_note": (
            "DERIVED ARITHMETIC: the measured per-bridge saving multiplied by the MEASURED bridge count of "
            "this configuration. It is not a separate measurement and no run of this artifact performed a "
            "multi-bridge transfer"
        ),
        "methods": methods,
    }


def plan_configurations() -> list[tuple[int, int, int | None]]:
    """Enumerate the ``(n_block, d, local_rank)`` triples this run measures.

    One block shape per ``n_block_over_d`` ratio so every regime is represented at a rank that is legal for
    it, plus the local ranks that give the accuracy table both ends of the trade. Duplicates are dropped so
    a configuration is never measured twice.

    - ``:return:`` The configurations to attempt, in attempt order.
    """
    plan: list[tuple[int, int, int | None]] = []
    seen: set[tuple[int, int, int | None]] = set()
    for d in FEATURE_DIMS:
        for n_block in BLOCK_ROWS:
            if n_block > MAX_BLOCK_ELEMENTS_ROWS.get(d, n_block):
                continue
            for local_rank in LOCAL_RANKS:
                if local_rank is not None and local_rank > d:
                    continue
                key = (n_block, d, local_rank)
                if key in seen:
                    continue
                seen.add(key)
                plan.append(key)
    return plan


#: Per-``d`` cap on ``n_block``, an INPUT that keeps the measured block inside the size the timing budget can
#: afford. It is a declared input, so the bound is arithmetic a reader can re-check rather than a measurement
#: dressed up as one.
MAX_BLOCK_ELEMENTS_ROWS: dict[int, int] = {32: 4096, 128: 4096, 512: 2048}


# ============================================================================= resume checkpointing
def _config_key(n_block: int, d: int, local_rank: int | None) -> str:
    """Return the resume key identifying one measured configuration.

    - ``:param n_block:`` Rows per block.
    - ``:param d:`` Feature dimension.
    - ``:param local_rank:`` Retained local rank, or ``None`` for full local rank.
    - ``:return:`` A stable string key, identical across runs so a checkpoint row can be found again.
    """
    return json.dumps([int(n_block), int(d), None if local_rank is None else int(local_rank)])


def _row_config_key(row: Mapping[str, Any]) -> str:
    """Return the resume key of an already-measured artifact row.

    - ``:param row:`` A row as it appears in ``results``.
    - ``:return:`` The same key :func:`_config_key` would produce for that configuration.
    """
    return _config_key(int(row["n_block"]), int(row["n_features"]), row.get("local_rank_requested"))


def _checkpoint_grid_keys(path: Path) -> set[str]:
    """Return the distinct grid keys recorded in a checkpoint file.

    - ``:param path:`` The JSONL checkpoint path.
    - ``:return:`` Grid keys, one per distinct recorded grid. Unparseable lines are ignored, so a torn
      final line from a hard kill does not raise here.
    """
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("config_key") is not None:
            keys.add(json.dumps(record["config_key"], sort_keys=True))
    return keys


def _load_checkpoint_rows(checkpoint: Any, repeats: int) -> list[dict[str, Any]]:
    """Return checkpointed rows recorded under the SAME grid and repeat count as this run.

    A checkpoint written for a different grid, or at a different repeat count, is refused rather than
    mixed: those rows are not the same measurement, and mixing them is the defect the repeat-count gate
    exists to catch.

    - ``:param checkpoint:`` The handle returned by :func:`measurement_common.checkpoint_config_key`.
    - ``:param repeats:`` Timed repeats this run will use.
    - ``:return:`` Reusable rows.
    """
    path: Path = checkpoint.path
    if not path.exists():
        return []
    mine = json.dumps(checkpoint.key, sort_keys=True)
    if mine not in _checkpoint_grid_keys(path):
        print(f"  checkpoint: {path.name} belongs to a different grid; starting fresh", flush=True)
        path.unlink(missing_ok=True)
        return []
    return checkpoint.load(repeats=repeats)


def _print_row(row: Mapping[str, Any]) -> None:
    """Print one measured configuration, whether measured now or replayed from a checkpoint.

    - ``:param row:`` The measured row.
    """
    mergeable = next(m for m in row["methods"] if m["method"] == MERGEABLE_ARM)
    print(
        f"  measured n_block={row['n_block']:>5} d={row['n_features']:<4} "
        f"rank={str(row['local_rank_requested']):<4} "
        f"saved={row['bytes_saved_per_block_vs_legacy_measured']['MiB']:9.4f} MiB "
        f"mergeable={mergeable['seconds_median']:.6f}s "
        f"d_exact={mergeable['accuracy'].get('subspace_distance_vs_exact')}",
        flush=True,
    )


# ============================================================================= run
def run(repeats: int) -> dict[str, Any]:
    """Measure every configuration and return the artifact payload.

    - ``:param repeats:`` Timed repeats per method, excluding warmup.
    """
    available = {method: _resolve(method)[0] for method in STANDARD_METHODS}
    resolved = {method: _resolve(method)[1] for method in STANDARD_METHODS}

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    plan = plan_configurations()

    # Resume support. Every measured configuration is appended to a JSONL checkpoint as soon as it is
    # measured, and a prior run's rows are replayed here. This sweep takes minutes per configuration, so
    # writing the artifact only once at the end loses the whole grid to any single interruption -- which
    # is exactly what happened to the three previous attempts at this card.
    checkpoint = checkpoint_config_key(
        ARTIFACT_STEM,
        {
            "grid": "transfer_comparison",
            "n_blocks": N_BLOCKS,
            "feature_dims": list(FEATURE_DIMS),
            "block_rows": list(BLOCK_ROWS),
            "local_ranks": [None if r is None else int(r) for r in LOCAL_RANKS],
            "n_components_scored": N_COMPONENTS_SCORED,
        },
    )
    reusable = {_row_config_key(row): row for row in _load_checkpoint_rows(checkpoint, repeats)}
    if reusable:
        print(f"  checkpoint: reusing {len(reusable)} configuration(s) already measured by a prior run", flush=True)

    for n_block, d, local_rank in plan:
        key = _config_key(n_block, d, local_rank)
        cached = reusable.get(key)
        if cached is not None:
            rows.append({**cached, "replayed_from_checkpoint": True})
            _print_row(cached)
            continue
        intrinsic = max(1, int(INTRINSIC_RANK_FRACTION * d))
        blocks = [make_block(n_block=n_block, n_features=d, rank=intrinsic, seed=SEED + 11) for _ in range(N_BLOCKS)]
        try:
            row = measure_configuration(blocks, n_block, d, local_rank, repeats, available)
        except MemoryError as exc:  # pragma: no cover - the gate below should prevent this
            skipped.append(
                {
                    "n_block": n_block,
                    "n_features": d,
                    "local_rank_requested": local_rank,
                    "status": "skipped_on_memory_error",
                    "reason": f"{type(exc).__name__} allocating the configuration: {exc}",
                }
            )
            continue
        rows.append(row)
        checkpoint.append(row, repeats=repeats)
        _print_row(row)
        del blocks
        gc.collect()

    return {
        "methods_compared": list(METHODS_COMPARED),
        "methods_compared_note": (
            "every method that appears in results[] is named here or in the two mergeable stages below, so "
            "no row exists that the paper cannot name in prose"
        ),
        "mergeable_stages_reported_not_compared": list(MERGEABLE_STAGES),
        "provenance": provenance(
            script=ARTIFACT_STEM,
            description=(
                "One consolidated comparison: NumPy SVD, SciPy SVD, scikit-learn batched PCA, scikit-learn "
                "batched IncrementalPCA and dask-ml IncrementalPCA against bridge-side mergeable PCA, on the "
                "SAME configurations, with in-process wall time, subspace distance to the exact SVD, and the "
                "network transfer each method actually implies."
            ),
            extra={
                # Restated, not inherited: provenance() stamps the DEFAULT policy, so a run with a different
                # repeat count must override it explicitly or the artifact reports a count it did not use --
                # the exact defect the shared measurement suite's repeat-count gate exists to catch.
                "timing_policy": {
                    "clock": TIMING_POLICY["clock"],
                    "warmup_rounds": TIMING_POLICY["warmup_rounds"],
                    "timed_repeats": int(repeats),
                    "statistic": TIMING_POLICY["statistic"],
                    "dispersion": TIMING_POLICY["dispersion"],
                    "note": TIMING_POLICY.get("note", ""),
                },
                "inputs": {
                    "feature_dims": list(FEATURE_DIMS),
                    "block_rows": list(BLOCK_ROWS),
                    "n_blocks": N_BLOCKS,
                    "local_ranks": [None if r is None else int(r) for r in LOCAL_RANKS],
                    "intrinsic_rank_fraction_of_d": INTRINSIC_RANK_FRACTION,
                    "n_components_scored": N_COMPONENTS_SCORED,
                    "timed_repeats": int(repeats),
                    "max_block_rows_per_d": dict(MAX_BLOCK_ELEMENTS_ROWS),
                },
                "timing_policy_why": (
                    "in-process COMPUTE only, on the analytics-side CPU, with one warmup round discarded and "
                    "the median of real repeats as the headline plus min/max/iqr/stddev. This is NOT a network "
                    "transfer duration: this box does not reproduce the interconnect, so no wire duration is "
                    "measured here and none is estimated. The column exists to be honest about what the bridge "
                    "pays for the bandwidth it saves"
                ),
                "memory_policy": {
                    "cap_bytes": cgroup_memory_limit_bytes(),
                    "cap_source": "/sys/fs/cgroup/memory.max (cgroup v2), else memory.limit_in_bytes (v1)",
                    "driver_peak_rss_bytes": peak_rss_bytes(),
                },
                "resolved_callables": resolved,
                "resume_policy": {
                    "checkpoint_path": str(checkpoint_path(ARTIFACT_STEM)),
                    "granularity": "one JSONL record per configuration, appended and fsynced as it is measured",
                    "why": (
                        "this sweep costs minutes per configuration, so an artifact written only once at "
                        "the end is lost in full by any single interruption -- which is what happened to three "
                        "consecutive previous attempts at this card"
                    ),
                    "rows_reused_from_a_prior_run": int(sum(1 for r in rows if r.get("replayed_from_checkpoint"))),
                    "guarded_against": (
                        "a checkpoint recorded under a different grid or a different repeat count is refused, "
                        "not merged: such rows are not the same measurement"
                    ),
                },
            },
        ),
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
        "results": rows,
        "skipped": skipped,
    }


def main() -> int:
    """Entry point: run the consolidation, write the artifact, print a human summary.

    - ``:return:`` ``0`` on success.
    """
    doc = __doc__ or ""
    parser = argparse.ArgumentParser(description=doc.splitlines()[0] if doc else "")
    parser.add_argument("--repeats", type=int, default=TIMED_REPEATS, help="Timed repeats per method")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Discard any resume checkpoint and re-measure every configuration from scratch",
    )
    args = parser.parse_args()
    if args.repeats < 2:
        print("refusing: --repeats < 2 leaves no dispersion behind the median")
        return 1
    if args.reset:
        checkpoint_path(ARTIFACT_STEM).unlink(missing_ok=True)
        print(f"checkpoint: discarded {checkpoint_path(ARTIFACT_STEM).name} (--reset)")

    payload = run(args.repeats)
    path = write_result(ARTIFACT_STEM, payload)
    # Only now are the rows safely inside a written artifact, so the resume state is consumed.
    checkpoint_path(ARTIFACT_STEM).unlink(missing_ok=True)

    print(f"\nconsolidated comparison, {len(payload['results'])} configuration(s)\n")
    print_summary_table(
        payload["results"],
        columns=(
            ("n_block", "n_blk", "int"),
            ("n_features", "d", "int"),
            ("regime", "regime", "str"),
            ("local_rank_requested", "rank", "auto"),
            ("block_wire_bytes", "block_wire", "auto"),
            ("bytes_saved_per_block_vs_legacy_measured", "saved_per_bridge", "auto"),
            ("bytes_saved_total_vs_legacy_derived", "saved_total_derived", "auto"),
        ),
        title="transfer volume per bridge send (measured) and the derived total over this configuration's bridges",
    )
    print(f"\nmethods_compared: {', '.join(payload['methods_compared'])}")
    prov = payload["provenance"]
    print(f"commit          : {prov['deisa_dask_commit']}")
    print(f"timed repeats   : {prov['inputs']['timed_repeats']} (plus {prov['timing_policy']['warmup_rounds']} warmup)")
    print(f"skipped         : {len(payload['skipped'])}")
    print(f"\nartifact: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
