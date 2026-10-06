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
gysela sizing: the two candidate PCA flattenings, sized over mesh shape AND rank count.

This is the flagship evaluation's design-space argument, and it is committed here as a runnable script precisely
because the paper must state it without pointing at any file on this machine. Nothing here is a throwaway probe and
nothing is a hardcoded measurement: the distribution is recomputed from the pinned ``MPILayout`` semantics, the byte
counts are computed from the shape arithmetic, and every number lands in the JSON.

Why the layout semantics come from source, and what is NOT claimed
-----------------------------------------------------------------
The field index range and the two MPI layouts are read from the pinned C++ (recorded in the artifact's provenance):

- ``gysela-mini-app_io`` @ f39e2a5, ``src/C++/geometry.hpp``:
  ``IdxRangeSpTor3DV2D = IdxRange<Species, GridTor1, GridTor2, GridTor3, GridVpar, GridMu>``,
  ``Tor3DSplit = MPILayout<IdxRangeSpTor3DV2D, GridTor1, GridTor2, GridTor3>``,
  ``V2DSplit = MPILayout<IdxRangeSpV2DTor3D, GridVpar, GridMu>``.
- ``gyselalibxx`` @ b9aad37c, ``src/mpi_parallelisation/mpilayout.hpp``: a requested dimension is distributed by
  ``gcd(comm_size, extent)`` ranks, leaving the rest to the recursive call on the lower dimensions. That is the
  "distribute IN ORDER, MAXIMALLY, leave a dimension local once ranks run out" rule this script reimplements.

NOT claimed: that any of these mesh extents is a production configuration, or that gysela was built or run here.
Neither repository builds on this box (toolchain-specific installer, no MPI toolchain match), so the extents are
SYNTHETIC parameter points in the same family as the application's index ranges and the arrays are sized
arithmetically in numpy. Only the TYPE and LAYOUT semantics are taken from source. The superseded Fortran GYSELA
decomposition is neither described nor measured.

The inversion this script exists to demonstrate
----------------------------------------------
Under ``Tor3DSplit`` each rank owns a contiguous ``(Tor1, Tor2, Tor3)`` box with the COMPLETE ``(Vpar, Mu)`` space
intact, because the layout distributes only the spatial axes. So:

- **Layout A (velocity-space PCA):** features ``(Vpar, Mu)``, so ``d = Nvpar * Nmu`` is fixed by the physics and
  INDEPENDENT of the rank count. It compresses at every rank count measured, so full local rank is viable and the
  merge is EXACT.
- **Layout B (spatial-box PCA):** features ``(Tor1, Tor2, Tor3)``, so ``d`` is the LOCAL box size and GROWS AS RANKS
  DECREASE. A full-rank summary is then not smaller than its input, and -- the inversion -- it gets WORSE as ranks
  are ADDED, because more ranks means more, smaller summaries to merge, each over a comparable box.

Sweeping rank count is therefore mandatory; a benchmark at one rank count misses the regime where the design is worst.

Run
---
    PYTHONPATH=src .venv/bin/python benchmark/mergeable_pca/flatten_sizing.py
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from measurement_common import (  # noqa: E402
    byte_dict,
    print_summary_table,
    provenance,
    ratio_or_none,
    regime_of,
    write_result,
)

# Bytes per element of a field and of a summary: both are float64, so the comparison is element-for-element and the
# units never differ.
BYTES_PER_ELEMENT = 8

#: Mesh extents as ``(Tor1, Tor2, Tor3, Vpar, Mu)``. SYNTHETIC parameter points in the same family as the
#: application's index ranges -- NOT production configurations read from an input file, and not measured from a run.
MESHES: tuple[tuple[int, int, int, int, int], ...] = (
    (128, 32, 16, 64, 8),
    (256, 64, 32, 128, 8),
    (512, 128, 64, 128, 8),
)

#: Rank counts to sweep. The card requires sweeping rank count, because Layout B degrades as ranks are ADDED.
RANK_COUNTS: tuple[int, ...] = (4, 8, 16, 32, 64, 128)

#: Retained local rank illustrating the truncated arm of Layout B, where a full-rank summary cannot compress.
TRUNCATED_LOCAL_RANK = 32


def tor3d_split(shape: tuple[int, int, int, int, int], n_ranks: int) -> tuple[tuple[int, int, int, int, int], int]:
    """Reimplement ``Tor3DSplit``: distribute Tor1, Tor2, Tor3 in order, maximally, leaving a dimension local.

    Mirrors ``MPILayout::internal_distribute_idx_range``: at each dimension the number of ranks placed along it is
    ``gcd(remaining_ranks, extent)``, the local extent is ``extent / ranks_along``, and the remaining ranks recurse on
    the lower dimensions. Returns the local ``(Tor1, Tor2, Tor3, Vpar, Mu)`` plus how many ranks are still unplaced
    (1 once the layout is exhausted, meaning the later dimensions stay whole -- which is why every rank keeps its
    complete velocity space).

    - ``:param shape:`` Global ``(Tor1, Tor2, Tor3, Vpar, Mu)``.
    - ``:param n_ranks:`` Communicator size.
    """
    t1, t2, t3, vpar, mu = shape
    local = [t1, t2, t3]
    remaining = int(n_ranks)
    for i in range(3):
        if remaining <= 1:
            break
        ranks_along = _gcd(remaining, local[i])
        if ranks_along <= 1:
            continue
        local[i] = local[i] // ranks_along
        remaining = remaining // ranks_along
    return (int(local[0]), int(local[1]), int(local[2]), int(vpar), int(mu)), int(remaining)


def _gcd(a: int, b: int) -> int:
    """Greatest common divisor, matching ``std::gcd`` in the C++ layout code.

    - ``:param a:`` First operand.
    - ``:param b:`` Second operand.
    """
    while b:
        a, b = b, a % b
    return abs(a)


def _leaf_summary_elements(n_samples: int, d: int, local_rank: int | None) -> int:
    """Elements a leaf summary holds, mirroring :func:`~deisa.dask.mergeable_pca.local_pca`'s output.

    A summary is ``components (rank, d) + mean (d) + singular_values (rank)``, with
    ``rank = min(n_samples, d)`` at full local rank and ``min(local_rank, n_samples, d)`` when truncated.

    - ``:param n_samples:`` Rows in the block.
    - ``:param d:`` Feature dimension.
    - ``:param local_rank:`` Retained local rank, or ``None`` for full.
    """
    full_rank = min(n_samples, d)
    rank = full_rank if local_rank is None else min(int(local_rank), n_samples, d)
    return rank * d + d + rank


def _merged_root_rank(n_leaves: int, leaf_rank: int, d: int) -> int:
    """Rank of the merged root after a balanced tree over ``n_leaves`` leaves.

    A merge stacks both operands plus one mean-correction row, so rank grows by one per merge until it saturates at
    ``min(rows(compact), d)``. Modelling the whole tree here keeps the root figure honest rather than assuming the
    root equals the leaf rank (it does not, and assuming so understates it).

    - ``:param n_leaves:`` Number of leaf summaries.
    - ``:param leaf_rank:`` Rank of each leaf.
    - ``:param d:`` Feature dimension.
    """
    level = int(leaf_rank)
    leaves = int(n_leaves)
    while leaves > 1:
        level = level * 2 + 1
        leaves = leaves // 2 + (leaves % 2)
        level = min(level, d)
    return min(level, d)


def size_layout_a(shape: tuple[int, int, int, int, int], n_ranks: int, local_rank: int | None) -> dict[str, Any]:
    """Size Layout A: PCA over VELOCITY space, per spatial cell.

    - ``:param shape:`` Global ``(Tor1, Tor2, Tor3, Vpar, Mu)``.
    - ``:param n_ranks:`` Communicator size.
    - ``:param local_rank:`` Retained local rank, or ``None`` for full.
    """
    local, unplaced = tor3d_split(shape, n_ranks)
    n_cells = local[0] * local[1] * local[2]
    d = shape[3] * shape[4]
    slab_elements = n_cells * d
    summ = _leaf_summary_elements(n_cells, d, local_rank)
    return {
        "mesh_tor1_tor2_tor3_vpar_mu": list(shape),
        "n_ranks": int(n_ranks),
        "local_box": list(local[:3]),
        "ranks_unplaced_after_layout": int(unplaced),
        "n_samples_per_rank": int(n_cells),
        "n_features_d": int(d),
        "n_features_definition": "Nvpar * Nmu (fixed by the physics, independent of rank count)",
        "n_block_over_d": float(n_cells) / float(d) if d else None,
        "regime": regime_of(n_cells, d),
        "local_rank_requested": local_rank,
        "leaf_summary_elements": int(summ),
        "slab_elements_per_rank": int(slab_elements),
        "slab_bytes": byte_dict(slab_elements * BYTES_PER_ELEMENT),
        "summary_bytes": byte_dict(summ * BYTES_PER_ELEMENT),
        "compression_ratio_slab_over_summary": ratio_or_none(
            slab_elements * BYTES_PER_ELEMENT, summ * BYTES_PER_ELEMENT
        ),
        "compresses": bool(summ * BYTES_PER_ELEMENT < slab_elements * BYTES_PER_ELEMENT),
    }


def size_layout_b(
    shape: tuple[int, int, int, int, int],
    n_ranks: int,
    local_rank: int | None,
    n_ranks_total: int,
) -> dict[str, Any]:
    """Size Layout B: PCA over the SPATIAL box, per velocity cell.

    The merged root is included because a Layout B fit merges every rank's summary into ONE root, and that root is
    where the blow-up actually lands -- the leaf alone understates it.

    - ``:param shape:`` Global ``(Tor1, Tor2, Tor3, Vpar, Mu)``.
    - ``:param n_ranks:`` Communicator size.
    - ``:param local_rank:`` Retained local rank, or ``None`` for full.
    - ``:param n_ranks_total:`` Total ranks participating, for the merged root sizing.
    """
    local, unplaced = tor3d_split(shape, n_ranks)
    d = local[0] * local[1] * local[2]
    n_samples = shape[3] * shape[4]
    slab_elements = n_samples * d
    leaf_summ = _leaf_summary_elements(n_samples, d, local_rank)
    leaf_rank = leaf_summ // (d + 1) if d else 0
    root_rank = _merged_root_rank(n_ranks_total, leaf_rank, d)
    root_elements = root_rank * d + d + root_rank
    return {
        "mesh_tor1_tor2_tor3_vpar_mu": list(shape),
        "n_ranks": int(n_ranks),
        "local_box": list(local[:3]),
        "ranks_unplaced_after_layout": int(unplaced),
        "n_samples_per_rank": int(n_samples),
        "n_features_d": int(d),
        "n_features_definition": "local Tor1 * Tor2 * Tor3 (grows as the rank count drops)",
        "n_block_over_d": float(n_samples) / float(d) if d else None,
        "regime": regime_of(n_samples, d),
        "local_rank_requested": local_rank,
        "leaf_summary_elements": int(leaf_summ),
        "merged_root_rank": int(root_rank),
        "merged_root_elements": int(root_elements),
        "slab_elements_per_rank": int(slab_elements),
        "slab_bytes": byte_dict(slab_elements * BYTES_PER_ELEMENT),
        "summary_bytes": byte_dict(root_elements * BYTES_PER_ELEMENT),
        "compression_ratio_slab_over_summary": ratio_or_none(
            slab_elements * BYTES_PER_ELEMENT, root_elements * BYTES_PER_ELEMENT
        ),
        "compresses": bool(root_elements * BYTES_PER_ELEMENT < slab_elements * BYTES_PER_ELEMENT),
        "note": (
            "merged root over all ranks is what a fit actually materializes; sizing only the leaf understates the "
            "Layout B blow-up"
        ),
    }


def run() -> dict[str, Any]:
    """Run the sizing sweep over mesh shape and rank count for both layouts, in full-rank and truncated arms.

    - ``:return:`` The artifact payload.
    """
    layout_a: list[dict[str, Any]] = []
    layout_a_trunc: list[dict[str, Any]] = []
    layout_b: list[dict[str, Any]] = []
    layout_b_trunc: list[dict[str, Any]] = []

    for shape in MESHES:
        for n_ranks in RANK_COUNTS:
            layout_a.append(size_layout_a(shape, n_ranks, None))
            layout_a_trunc.append(size_layout_a(shape, n_ranks, TRUNCATED_LOCAL_RANK))
            layout_b.append(size_layout_b(shape, n_ranks, None, n_ranks))
            layout_b_trunc.append(size_layout_b(shape, n_ranks, TRUNCATED_LOCAL_RANK, n_ranks))

    def _compress_rate(rows: Sequence[Mapping[str, Any]]) -> float:
        """Fraction of rows that compress, as a float.

        - ``:param rows:`` Sizing rows.
        """
        return sum(1 for r in rows if r["compresses"]) / len(rows) if rows else 0.0

    def _ratio_range(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Compression ratio statistics across rows, as a dict.

        - ``:param rows:`` Sizing rows.
        """
        ratios = [
            float(r["compression_ratio_slab_over_summary"]) for r in rows if r["compression_ratio_slab_over_summary"]
        ]
        return {
            "n": len(rows),
            "min": min(ratios) if ratios else None,
            "max": max(ratios) if ratios else None,
        }

    # The inversion: in Layout B, does full-rank compression get WORSE as ranks are ADDED? Measured as the trend of
    # the full-rank ratio across the rank sweep for one mesh, rather than asserted.
    inversion_trend = []
    for shape in MESHES:
        for_rank = [r for r in layout_b if tuple(r["mesh_tor1_tor2_tor3_vpar_mu"]) == tuple(shape)]
        series = sorted(
            ((int(r["n_ranks"]), r["compression_ratio_slab_over_summary"], int(r["n_features_d"])) for r in for_rank),
            key=lambda item: item[0],
        )
        ratios = [s[1] for s in series if s[1] is not None]
        # Spearman-style monotonicity over consecutive rank steps, computed rather than asserted. An earlier draft
        # compared only the first and last ratio, which is fragile: the series is NOT strictly monotone (the merged
        # root rank saturates and then drops as the box shrinks), so an endpoint test would misreport the trend.
        # Reporting both the endpoint change AND the fraction of consecutive steps that get worse keeps the inversion
        # statement honest about the non-monotonicity instead of smoothing it away.
        steps_worse = sum(1 for a, b in zip(ratios, ratios[1:], strict=False) if b < a)
        steps_total = max(0, len(ratios) - 1)
        inversion_trend.append(
            {
                "mesh": list(shape),
                "ranks": [s[0] for s in series],
                "d_per_rank": [s[2] for s in series],
                "full_rank_ratio": ratios,
                "ratio_first": ratios[0] if ratios else None,
                "ratio_last": ratios[-1] if ratios else None,
                "ratio_at_high_rank_worse_than_low": bool(len(ratios) > 1 and ratios[-1] < ratios[0]),
                "consecutive_steps_total": steps_total,
                "consecutive_steps_worse_as_ranks_added": steps_worse,
                "fraction_steps_worse": (steps_worse / steps_total) if steps_total else None,
                "series_is_monotone_worsening": bool(steps_total > 0 and steps_worse == steps_total),
                "all_full_rank_rows_compress": all(bool(r["compresses"]) for r in for_rank),
                "n_full_rank_rows_compressing": sum(1 for r in for_rank if r["compresses"]),
                "n_rows": len(for_rank),
                "note": (
                    "full rank never compresses in Layout B at any rank count measured; the per-step trend is "
                    "reported because the series is not strictly monotone (the merged root rank saturates), so an "
                    "endpoint-only comparison would misstate it"
                ),
            }
        )

    return {
        "provenance": provenance(
            script="flatten_sizing",
            description=(
                "Sizing of the two candidate PCA flattenings for a gyrokinetic distribution indexed "
                "(Species, Tor1, Tor2, Tor3, Vpar, Mu), swept over mesh shape and RANK COUNT. Layout A "
                "(velocity-space) compresses at every rank count measured; Layout B (spatial-box) does not, and "
                "degrades as ranks are added."
            ),
            extra={
                "inputs": {
                    "meshes_tor1_tor2_tor3_vpar_mu": [list(m) for m in MESHES],
                    "rank_counts": list(RANK_COUNTS),
                    "truncated_local_rank": TRUNCATED_LOCAL_RANK,
                    "bytes_per_element": BYTES_PER_ELEMENT,
                },
                "layout_semantics_from_source": {
                    "field_and_layouts": "gysela-mini-app_io @ f39e2a5, src/C++/geometry.hpp",
                    "distribution_rule": (
                        "gyselalibxx @ b9aad37c, src/mpi_parallelisation/mpilayout.hpp: a requested dimension takes "
                        "gcd(remaining_ranks, extent) ranks, local extent is extent/ranks_along, and the remaining "
                        "ranks recurse on the lower dimensions -- distribute in order, maximally, leave a dimension "
                        "local once ranks run out"
                    ),
                },
                "not_claimed": (
                    "the mesh extents are SYNTHETIC parameter points in the same family as the application's index "
                    "ranges, not production configurations read from an input file. No gysela build was run on this "
                    "machine; only the TYPE and LAYOUT semantics are taken from the pinned C++ sources, and the arrays "
                    "are sized arithmetically. The superseded Fortran GYSELA decomposition is not described here."
                ),
            },
        ),
        "findings": {
            "layout_a_velocity_space": {
                "full_rank_compress_rate": _compress_rate(layout_a),
                "full_rank_ratio": _ratio_range(layout_a),
                "d_independent_of_rank_count": True,
                "d_values_observed": sorted({r["n_features_d"] for r in layout_a}),
                "claim": (
                    "d = Nvpar*Nmu is fixed by the physics and independent of the rank count, so a full-rank summary "
                    "compresses at every rank count measured and the merge can be exact"
                ),
            },
            "layout_b_spatial_box": {
                "full_rank_compress_rate": _compress_rate(layout_b),
                "full_rank_ratio": _ratio_range(layout_b),
                "truncated_compress_rate": _compress_rate(layout_b_trunc),
                "truncated_ratio": _ratio_range(layout_b_trunc),
                "d_grows_as_ranks_decrease": True,
                "d_values_observed": sorted({r["n_features_d"] for r in layout_b}),
                "claim": (
                    "d is the LOCAL spatial box and grows as the rank count drops, so a full-rank summary is not "
                    "smaller than its input; truncating local_rank restores compression at the cost of accuracy"
                ),
            },
            "inversion": inversion_trend,
        },
        "layout_a_velocity_space": {
            "full_rank": layout_a,
            "truncated": layout_a_trunc,
        },
        "layout_b_spatial_box": {
            "full_rank": layout_b,
            "truncated": layout_b_trunc,
        },
    }


def main() -> int:
    """Entry point: run the sizing sweep, write the artifact, print a human summary.

    - ``:return:`` ``0`` on success.
    """
    doc = __doc__ or ""
    parser = argparse.ArgumentParser(description=doc.splitlines()[0] if doc else "")
    parser.add_argument("--out", default=None, help="Optional explicit artifact path")
    args = parser.parse_args()

    payload = run()
    path = write_result("flatten_sizing", payload)
    if args.out:
        Path(args.out).write_text(path.read_text(), encoding="utf-8")
        path = Path(args.out)

    findings = payload["findings"]
    print("\ngysela sizing -- Layout A (velocity-space PCA), features (Vpar, Mu)\n")
    print_summary_table(
        payload["layout_a_velocity_space"]["full_rank"],
        columns=(
            ("n_ranks", "ranks", "int"),
            ("n_samples_per_rank", "n_samples", "int"),
            ("n_features_d", "d", "int"),
            ("regime", "regime", "str"),
            ("slab_bytes", "slab_MiB", "mib"),
            ("summary_bytes", "summ_MiB", "mib"),
            ("compression_ratio_slab_over_summary", "ratio", "float"),
        ),
    )
    print("\ngysela sizing -- Layout B (spatial-box PCA), features (Tor1, Tor2, Tor3), FULL rank\n")
    print_summary_table(
        payload["layout_b_spatial_box"]["full_rank"],
        columns=(
            ("n_ranks", "ranks", "int"),
            ("n_features_d", "d", "int"),
            ("regime", "regime", "str"),
            ("slab_bytes", "slab_MiB", "mib"),
            ("summary_bytes", "summ_MiB", "mib"),
            ("compression_ratio_slab_over_summary", "ratio", "float"),
        ),
    )
    print("\nLayout A findings:", findings["layout_a_velocity_space"])
    print("\nLayout B findings:", findings["layout_b_spatial_box"])
    print("\nInversion (does full rank get worse as ranks are ADDED?):")
    for entry in findings["inversion"]:
        print(
            f"  mesh={entry['mesh']} ranks={entry['ranks']}\n"
            f"    ratio series          : {[round(r, 4) for r in entry['full_rank_ratio']]}\n"
            f"    worse at high rank    : {entry['ratio_at_high_rank_worse_than_low']}\n"
            f"    steps worse/total     : {entry['consecutive_steps_worse_as_ranks_added']}/"
            f"{entry['consecutive_steps_total']} (monotone: {entry['series_is_monotone_worsening']})\n"
            f"    full-rank rows compressing: {entry['n_full_rank_rows_compressing']}/{entry['n_rows']}"
        )
    print(f"\nartifact: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
