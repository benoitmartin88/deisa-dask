"""Build the Figure 2 source artifact by combining every measured row we have.

Figure 2 plots one number per (feature dimension, block aspect ratio): the bytes the legacy path
would put on the wire for one bridge, against the bytes a bridge summary occupies. Those two
quantities were measured by two different scripts, and neither artifact alone covers the figure:

  - ``network_transfer.json`` has 4 feature dimensions (32, 128, 256, 512) but only 5 distinct
    aspect ratios, and no ``x=4``, which the figure draws as a tick.
  - ``tradeoff_cost_gapfill.json`` supplies the missing ``x=4`` column and many more ratios, but
    only 3 feature dimensions (32, 128, 512).

So the figure is built from both. This script does NOT measure anything: it only projects rows onto
the two fields the figure needs and records where each row came from, so a reader can tell measured
from carried and can see that no value in the figure was invented here.

Basis note, which matters for correctness: the single-bridge sweep
    reports ``block_bytes``/``summary_bytes`` for a single
bridge, while the multi-bridge sweep reports
    ``*_all_bridges_measured`` totals over 32 bridges. Those are not comparable.
The figure is per-bridge, so this script uses the multi-bridge sweep's ``per_bridge_block_measured`` and
``per_bridge_leaf_summary_measured``, which are the same quantity the single-bridge sweep records.

- ``:param results_dir:`` Directory holding the committed result JSON.
- ``:param out_path:`` Where to write the combined artifact.
- ``:return:`` The payload written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: The two aspect-ratio ticks the figure labels, kept for the coverage report below.
FIGURE_TICKS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def _load(results_dir: Path, name: str) -> dict[str, Any]:
    path = results_dir / name
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _project(row: dict[str, Any], source: str) -> dict[str, Any] | None:
    """Project one measured row onto the fields the figure plots, or ``None`` if it cannot.

    Both input schemas are accepted so the figure can draw every configuration we have measured,
    whichever script measured it.
    """
    n_block = row.get("n_block")
    d = row.get("n_features")
    rank = row.get("local_rank_requested")
    if n_block is None or d is None:
        return None
    # The figure fixes the summary rank at full local rank, so the rank is not a second free
    # variable. Anything measured at a truncated rank is deliberately left out.
    if rank is not None:
        return None

    if "block_bytes" in row and "summary_bytes" in row:
        # the single-bridge sweep schema: already per-bridge.
        block = row["block_bytes"]["bytes"]
        summary = row["summary_bytes"]["bytes"]
    else:
        wire = row.get("wire_bytes") or {}
        block_key = "per_bridge_block_measured"
        summary_key = "per_bridge_leaf_summary_measured"
        if block_key not in wire or summary_key not in wire:
            return None
        block = wire[block_key]["bytes"]
        summary = wire[summary_key]["bytes"]

    return {
        "n_block": n_block,
        "n_features": d,
        "n_block_over_d": row.get("n_block_over_d", n_block / d),
        # Recorded explicitly so the invariant is visible in the artifact and not only in this code.
        "local_rank_requested": None,
        # Kept in the same nested shape the single-bridge sweep uses, so plot_figures.py reads either schema unchanged.
        "block_bytes": {"bytes": float(block)},
        "summary_bytes": {"bytes": float(summary)},
        "intrinsic_rank": row.get("intrinsic_rank"),
        "measured_by": source,
    }


def build(results_dir: Path) -> dict[str, Any]:
    """Combine every measured row into the artifact the figure is drawn from.

    - ``:param results_dir:`` Directory holding the committed result JSON.
    - ``:return:`` The combined payload.
    """
    sources = (
        ("tradeoff_cost_gapfill.json", "tradeoff_cost"),
        ("network_transfer.json", "network_transfer"),
    )

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for filename, label in sources:
        payload = _load(results_dir, filename)
        for raw in payload.get("results", []):
            projected = _project(raw, label)
            if projected is None:
                continue
            key = (projected["n_block"], projected["n_features"])
            if key in seen:
                # the multi-bridge sweep is listed first, so it wins. Duplicates are the same configuration measured
                # by both scripts; keeping one avoids double-weighting a point in the figure.
                continue
            seen.add(key)
            rows.append(projected)

    rows.sort(key=lambda r: (r["n_features"], r["n_block_over_d"]))

    coverage = {}
    for tick in FIGURE_TICKS:
        hits = [r for r in rows if abs(r["n_block_over_d"] - tick) < 1e-9]
        coverage[f"{tick:g}"] = {
            "rows": len(hits),
            "n_features": sorted({r["n_features"] for r in hits}),
        }

    return {
        "figure": "network transfer vs block aspect ratio, per bridge, at full local rank",
        "basis": (
            "per bridge, not per run: the single-bridge sweep's block_bytes/summary_bytes and the multi-bridge sweep's "
            "per_bridge_block_measured/per_bridge_leaf_summary_measured are the same quantity, so "
            "the two artifacts are combined on that common basis"
        ),
        "generated_by": "benchmark/mergeable_pca/build_figure2_source.py",
        "measures_anything": False,
        "provenance": {
            "sources": [f for f, _ in sources if (results_dir / f).exists()],
            "rows_total": len(rows),
            "rows_by_source": {label: sum(1 for r in rows if r["measured_by"] == label) for _, label in sources},
        },
        "tick_coverage": coverage,
        "results": rows,
    }


def main() -> int:
    here = Path(__file__).resolve().parent
    results_dir = here / "results"
    out_path = results_dir / "fig_network_transfer.json"

    payload = build(results_dir)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    prov = payload["provenance"]
    print(f"wrote {out_path.name}: {prov['rows_total']} row(s) from {prov['sources']}")
    print(f"  by source: {prov['rows_by_source']}")
    print("  tick coverage:")
    for tick, info in payload["tick_coverage"].items():
        state = "MEASURED" if info["rows"] else "MISSING"
        print(f"    x={tick:<5} {info['rows']} row(s) d={info['n_features']}  {state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
