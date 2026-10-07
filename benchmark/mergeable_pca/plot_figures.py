#!/usr/bin/env python3
"""Generate the mergeable-PCA paper's data figures from the benchmark artifacts.

Every plotted value is read from a JSON result file written by the benchmark
measurement suite (``benchmark/mergeable_pca/results/*.json``); nothing in this script is
a hand-entered number.  The figures are vector PDFs sized for a single-column
ACM layout (3.3 in) with type set to remain legible at final print size.

The three data figures are:

  network_transfer.png the network transfer across the coupling boundary, legacy
                       full-chunk scatter versus the bridge-side mergeable summary,
                       full local rank, over the block aspect ratio.
  local_rank_curve.png local rank R against summary size and against the two
                       sign-invariant accuracy metrics, for a single leaf and
                       for the eight-leaf merge.
  flatten_sizing.png    the two flattenings sized over the two meshes that fit memory:
                       compression ratio (chunk bytes / summary bytes) versus rank count.

Run:
    python -m benchmark.mergeable_pca.plot_figures \
        --results benchmark/mergeable_pca/results \
        --out benchmark/mergeable_pca/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import ticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# --------------------------------------------------------------------------
# House style: single ACM column, small serif type, vector output.
# --------------------------------------------------------------------------
ACM_COLUMN_IN = 3.3  # inches, ACM single-column text width

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 7.5,
        "legend.fontsize": 6.2,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.2,
        "axes.grid": True,
        "grid.alpha": 0.30,
        "grid.linewidth": 0.4,
        "legend.frameon": False,
        "legend.handlelength": 1.7,
        "legend.handletextpad": 0.4,
        "legend.labelspacing": 0.25,
        "legend.columnspacing": 1.0,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)

# Okabe-Ito colourblind-safe palette plus a grey.
COL = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#4D4D4D",
}


def _load(results_dir: Path, name: str) -> dict:
    with open(results_dir / name) as fh:
        return json.load(fh)


def _rank_series(points, ykey, rankkey="local_rank_effective"):
    """Return (ranks, values) ordered by effective rank, one point per rank."""
    by_rank = {}
    for r in points:
        by_rank[r[rankkey]] = r
    ranks = sorted(by_rank)
    return ranks, [ykey(by_rank[k]) for k in ranks]


# --------------------------------------------------------------------------
# Figure B1: the network transfer across the coupling boundary.
# --------------------------------------------------------------------------
def figure_network_transfer(results_dir: Path, out_dir: Path) -> Path:
    # The combined artifact is built by build_figure2_source.py from every measured row we have.
    # Neither input artifact alone covers the figure: b1 has four feature dimensions but no x=4,
    # which the figure labels, and the gap-fill run supplies x=4 but only three feature dimensions.
    # Every row in the combined artifact carries measured_by, so the provenance survives to the plot.
    b1 = _load(results_dir, "fig_network_transfer.json")
    # build_figure2_source.py already keeps only the rows measured at full local rank, so that is
    # the invariant here rather than a filter over a field the combined rows do not carry.
    rows = list(b1["results"])
    dims = sorted({r["n_features"] for r in rows})

    # Four feature dimensions are measured, so four colours; grey is a fallback.
    colours = {32: COL["blue"], 128: COL["orange"], 256: COL["red"], 512: COL["green"]}

    fig, ax = plt.subplots(figsize=(ACM_COLUMN_IN, 2.5))
    for d in dims:
        c = colours.get(d, COL["grey"])
        pts = sorted((r for r in rows if r["n_features"] == d), key=lambda r: r["n_block_over_d"])
        x = [r["n_block_over_d"] for r in pts]
        # The measured payload bytes, on the same basis as the ratio the paper quotes.
        legacy = [r["block_bytes"]["bytes"] for r in pts]
        summary = [r["summary_bytes"]["bytes"] for r in pts]
        # A line joins only configurations that share a feature dimension and number
        # more than one; a lone measured point is a marker, never an extrapolation.
        line_legacy = "-" if len(x) > 1 else "none"
        line_summary = "--" if len(x) > 1 else "none"
        ax.plot(x, legacy, line_legacy, color=c, marker="o", clip_on=False)
        ax.plot(x, summary, line_summary, color=c, marker="s", mfc="white", clip_on=False)

    ax.axvspan(1.0, 14.0, color="0.85", alpha=0.4, zorder=0, linewidth=0)
    ax.axvline(1.0, color="0.35", ls=":", lw=0.8, zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128])
    ax.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8", "16", "32", "64", "128"], fontsize=5.2)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(2, 5)))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    # The combined artifact spans x from 0.25 to 128 and per-bridge block bytes from ~3.3e4 to
    # ~8.4e6, so the limits cover the measured range rather than the narrower original sweep.
    ax.set_xlim(0.18, 160)
    ax.set_ylim(2e3, 3e7)
    ax.set_xlabel(r"blocks per feature, $n_{\mathrm{block}}/d$")
    ax.set_ylabel("network transfer (bytes)")

    path_leg = [
        Line2D([], [], color="0.2", ls="-", marker="o", label="full chunk"),
        Line2D([], [], color="0.2", ls="--", marker="s", mfc="white", label="bridge summary"),
    ] + [Line2D([], [], color=colours.get(d, COL["grey"]), ls="-", marker="o", label=rf"$d={d}$") for d in dims]
    # One figure-level legend above the axes: add_artist legend pairs are silently
    # dropped by savefig(bbox="tight") in this matplotlib, a figure-level legend is
    # not, and no line is ever covered.
    fig.legend(
        handles=path_leg,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.985),
        ncol=6,
        fontsize=5.8,
        borderpad=0.1,
        frameon=False,
        handlelength=1.5,
        columnspacing=0.7,
    )
    # High band, x where only the d=512 solid reaches: text sits at 1.1e7, x=1.7, well above the
    # smaller-d solids and below their legacy line only where the reader already sees the band.
    ax.text(4.5, 1.1e7, "tall", color="0.35", fontsize=6.5, ha="center")
    fig.savefig(out_dir / "network_transfer.png")
    plt.close(fig)
    return out_dir / "network_transfer.png"


# --------------------------------------------------------------------------
# Figure B4: local-rank accuracy and bandwidth curve.
# --------------------------------------------------------------------------
def figure_local_rank(results_dir: Path, out_dir: Path) -> Path:
    b4 = _load(results_dir, "rank_accuracy_curve.json")
    d = 128
    ratio = 8.0  # tall regime: n_block/d = 8

    # Panel (a): single leaf, summary size and relative explained-variance error.
    sl = [r for r in b4["single_leaf"] if r["n_features"] == d and r["n_block_over_d"] == ratio]
    xa, bytes_a = _rank_series(sl, lambda r: r["summary_bytes"]["bytes"])
    _, err_a = _rank_series(sl, lambda r: r["total_variance_relative_error"])

    # Panel (b): eight-leaf merge, subspace distance and merged root rank.
    mb = [r for r in b4["multi_block"] if r["n_features"] == d and r["regime"] == "tall" and r["n_blocks"] == 8]
    n_block = max(r["n_block"] for r in mb)
    mb = [r for r in mb if r["n_block"] == n_block]
    xb, dist_b = _rank_series(mb, lambda r: r["subspace_distance"], rankkey="local_rank_effective_per_leaf")
    _, rank_b = _rank_series(mb, lambda r: r["merged_root_rank"], rankkey="local_rank_effective_per_leaf")

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(ACM_COLUMN_IN, 4.15),
        sharex=True,
        gridspec_kw={"hspace": 0.6},
    )

    # ---- (a) single leaf -------------------------------------------------
    c0, c1 = COL["blue"], COL["red"]
    ax1.plot(xa, bytes_a, "-o", color=c0)
    ax1.set_yscale("log")
    ax1.set_ylabel("summary size (bytes)", color=c0)
    ax1.tick_params(axis="y", colors=c0)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks([1, 2, 5, 10, 20, 40, 80, 128])
    ax1.set_xticklabels(["1", "2", "5", "10", "20", "40", "80", "d"])
    ax1.xaxis.set_minor_locator(ticker.NullLocator())
    ax1.set_title(r"single leaf, $d=128$, $n_{\mathrm{block}}/d=8$", loc="left")

    ax1b = ax1.twinx()
    ax1b.plot(xa, err_a, "-^", color=c1)
    ax1b.set_ylabel("relative explained-\nvariance error", color=c1)
    ax1b.tick_params(axis="y", colors=c1)
    ax1b.set_ylim(-0.02, 1.0)
    ax1b.grid(False)
    ax1b.annotate(
        r"exact at $R\geq d$",
        xy=(128, 0.0),
        xytext=(9, 0.5),
        fontsize=6.2,
        color="0.3",
        arrowprops=dict(arrowstyle="->", color="0.4", lw=0.6),
    )
    ax1.text(0.02, 0.90, "(a)", transform=ax1.transAxes, fontsize=7.5, fontweight="bold")
    ax1.legend(
        handles=[
            Line2D([], [], color=c0, ls="-", marker="o", label="summary size (bytes)"),
            Line2D([], [], color=c1, ls="-", marker="^", label="variance error (dimensionless)"),
        ],
        loc="center left",
        bbox_to_anchor=(0.02, 0.62),
        fontsize=5.8,
    )

    # ---- (b) eight-leaf merge -------------------------------------------
    ax2.plot(xb, rank_b, "-o", color=c0)
    ax2.set_ylabel("merged root rank", color=c0)
    ax2.tick_params(axis="y", colors=c0)
    ax2.set_xlabel(r"local rank $R$ (dimensionless)")
    ax2.set_title(r"eight-leaf merge, $d=128$, $n_{\mathrm{block}}/d=8$", loc="left")
    ax2.text(0.02, 0.90, "(b)", transform=ax2.transAxes, fontsize=7.5, fontweight="bold")

    ax2b = ax2.twinx()
    ax2b.plot(xb, dist_b, "-^", color=c1)
    ax2b.set_yscale("log")
    ax2b.set_ylabel("principal-subspace\ndistance (dimensionless)", color=c1)
    ax2b.tick_params(axis="y", colors=c1)
    ax2b.grid(False)
    ax2b.set_ylim(1e-16, 4.0)
    ax2.legend(
        handles=[
            Line2D([], [], color=c0, ls="-", marker="o", label="merged root rank"),
            Line2D([], [], color=c1, ls="-", marker="^", label=r"$1-\sigma_{\min}(Q_a^{\top}Q_b)$"),
        ],
        loc="center left",
        bbox_to_anchor=(0.02, 0.66),
        fontsize=5.8,
    )

    fig.savefig(out_dir / "local_rank_curve.png")
    plt.close(fig)
    return out_dir / "local_rank_curve.png"


# --------------------------------------------------------------------------
# Figure sizing: the two flattenings sized over two mesh sizes that fit memory side by
# side. The largest example mesh (512x128x64x128x8) never fit in memory on any machine we
# had, so plotting it next to the smaller ones implied a run that did not happen. The
# figure now shows only measured-capable meshes; the sizing table of the paper carries the
# larger-mesh arithmetic.
# --------------------------------------------------------------------------
def figure_sizing(results_dir: Path, out_dir: Path) -> Path:
    g = _load(results_dir, "flatten_sizing.json")
    meshes = [(128, 32, 16, 64, 8), (256, 64, 32, 128, 8)]

    def series(layout, sub, mesh):
        rows = [r for r in g[layout][sub] if tuple(r["mesh_tor1_tor2_tor3_vpar_mu"]) == mesh]
        rows = sorted(rows, key=lambda r: r["n_ranks"])
        return ([r["n_ranks"] for r in rows], [r["compression_ratio_slab_over_summary"] for r in rows])

    fig, ax = plt.subplots(figsize=(ACM_COLUMN_IN, 2.5))
    # Colour = layout, solid = full rank, dashed = truncated local rank R=32.
    # Line weight distinguishes the two mesh sizes; the heavier line is the larger mesh.
    style = {
        (128, 32, 16, 64, 8): {"lw": 1.0},
        (256, 64, 32, 128, 8): {"lw": 2.0},
    }
    specs = [
        ("layout_a_velocity_space", "full_rank", COL["blue"], "-", "o", "velocity axis, full rank"),
        ("layout_a_velocity_space", "truncated", COL["blue"], "--", "s", "velocity axis, capped rank 32"),
        ("layout_b_spatial_box", "full_rank", COL["red"], "-", "o", "spatial axis, full rank"),
        ("layout_b_spatial_box", "truncated", COL["red"], "--", "s", "spatial axis, capped rank 32"),
    ]
    for layout, sub, c, ls, mk, label in specs:
        for mesh in meshes:
            x, y = series(layout, sub, mesh)
            first = mesh == meshes[0]
            ax.plot(x, y, ls, color=c, marker=mk, lw=style[mesh]["lw"], label=label if first else None)

    # One extra legend entry pair naming the two mesh sizes via line weight.
    from matplotlib.lines import Line2D

    weight_leg = [
        Line2D([], [], color="0.3", lw=1.0, label="mesh 128$\\times$32$\\times$16$\\times$64$\\times$8"),
        Line2D([], [], color="0.3", lw=2.0, label="mesh 256$\\times$64$\\times$32$\\times$128$\\times$8"),
    ]

    ax.axhline(1.0, color="0.35", ls=":", lw=0.9, zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([4, 8, 16, 32, 64, 128])
    ax.set_xticklabels(["4", "8", "16", "32", "64", "128"])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(2, 5)))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlim(3.2, 150)
    ax.set_ylim(0.008, 34000)
    ax.set_xlabel("number of MPI ranks")
    ax.set_ylabel("chunk bytes / summary bytes")
    ax.text(9.0, 0.5, "break-even (ratio 1)", fontsize=6.0, color="0.35", ha="center")
    # One legend band BELOW the axes: interior legends sit over the curves whichever
    # corner we pick, because every quadrant of this chart carries a line. The two mesh
    # sizes ride on line weight, which the paper caption explains.
    ax.legend(
        handles=[
            Line2D([], [], color=COL["blue"], ls="-", marker="o", label="velocity axis, full rank"),
            Line2D([], [], color=COL["blue"], ls="--", marker="s", label="velocity axis, capped rank 32"),
            Line2D([], [], color=COL["red"], ls="-", marker="o", label="spatial axis, full rank"),
            Line2D([], [], color=COL["red"], ls="--", marker="s", label="spatial axis, capped rank 32"),
        ]
        + weight_leg,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.16),
        ncol=2,
        fontsize=5.8,
        borderpad=0.1,
        frameon=False,
        handlelength=1.6,
        columnspacing=0.8,
    )
    fig.savefig(out_dir / "flatten_sizing.png")
    plt.close(fig)
    return out_dir / "flatten_sizing.png"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("benchmark/mergeable_pca/results"))
    ap.add_argument("--out", type=Path, default=Path("benchmark/mergeable_pca/figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Rebuild the Figure 2 source from the committed measurements first, so the plot can never be
    # drawn from a stale or hand-edited artifact. build_figure2_source.py measures nothing; it only
    # combines rows that were already measured and records where each one came from.
    source = args.results / "fig_network_transfer.json"
    if not source.exists():
        print("building the Figure 2 source from the committed measurements")
        import build_figure2_source

        payload = build_figure2_source.build(args.results)
        source.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"  wrote {source.name}: {payload['provenance']['rows_total']} row(s)")
        for tick, info in payload["tick_coverage"].items():
            if not info["rows"]:
                print(f"  WARNING: x={tick} has no measured row, so that tick is unsupported")

    made = [
        figure_network_transfer(args.results, args.out),
        figure_local_rank(args.results, args.out),
        figure_sizing(args.results, args.out),
        figure_baselines(args.results, args.out),
    ]
    for m in made:
        print("wrote", m)


# --------------------------------------------------------------------------
# Figure: baselines — time and accuracy of the mergeable pipeline against the
# standard alternatives, over the same measured parameters.
# --------------------------------------------------------------------------
def figure_baselines(results_dir: Path, out_dir: Path) -> Path:
    """Two panels from standard_baselines.json, all arms with timing dispersion.

    Panel (a): per-problem total seconds (median with min/max whiskers) against the
    d=128 tall ladder for the mergeable pipeline total and the measured alternatives.
    Panel (b): principal-subspace distance against the exact batch SVD reference,
    same ladder. The mergeable path must sit at roundoff: that is the point.
    """
    sb = _load(results_dir, "standard_baselines.json")
    rows = sb["results"]

    # Panel arms, in drawing order: the path under test first, then the alternatives.
    draws = [
        "mergeable_pca_total",
        "numpy_svd",
        "sklearn_pca_full",
        "sklearn_incremental_pca",
        "dask_ml_incremental_pca",
    ]
    labels = {
        "mergeable_pca_total": "mergeable PCA (total)",
        "numpy_svd": "numpy batch SVD",
        "sklearn_pca_full": "sklearn PCA (full)",
        "sklearn_incremental_pca": "sklearn IncrementalPCA",
        "dask_ml_incremental_pca": "dask-ml IncrementalPCA",
    }
    colours = {
        "mergeable_pca_total": COL["blue"],
        "numpy_svd": COL["green"],
        "sklearn_pca_full": COL["orange"],
        "sklearn_incremental_pca": COL["red"],
        "dask_ml_incremental_pca": COL["purple"],
    }
    markers = {
        "mergeable_pca_total": "o",
        "numpy_svd": "s",
        "sklearn_pca_full": "^",
        "sklearn_incremental_pca": "v",
        "dask_ml_incremental_pca": "D",
    }

    # Full local rank (requested None) is the precision operating point: the mergeable path is
    # exact there, which is the claim the baselines figure argues. Truncated-rank rows belong to
    # the rank-accuracy figure, not this one.
    operating_rank: int | None = None

    def _series(arm: str, d: int, ykey, with_err: bool = False):
        # One row per (n_block, arm) at the operating rank, so rank variations of the same shape
        # do not overplot each other.
        pts = [
            r
            for r in rows
            if r["n_features"] == d and r["regime"] == "tall" and r["local_rank_requested"] == operating_rank
        ]
        by_n = {}
        for r in pts:
            a = next(a for a in r["arms"] if a["arm"] == arm)
            if a.get("measured") and (ykey(a) is not None):
                by_n[r["n_block"]] = (a, r)
        ns = sorted(by_n)
        ys, los, his = [], [], []
        for n in ns:
            a, r = by_n[n]
            y = ykey(a)
            ys.append(y)
            t = a.get("timing") or {}
            if with_err and t.get("seconds_all"):
                alls = sorted(t["seconds_all"])
                los.append(max(y - alls[0], 0.0))
                his.append(max(alls[-1] - y, 0.0))
            else:
                los.append(0.0)
                his.append(0.0)
        return ns, ys, los, his

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(ACM_COLUMN_IN, 4.3), sharex=True, gridspec_kw={"hspace": 0.55})

    # ---- (a) time ----------------------------------------------------------
    # Two feature dimensions share each arm's marker; d is carried by fill (solid d=128, open d=32).
    ladders = sorted({r["n_features"] for r in rows if r["regime"] == "tall"}, reverse=True)[:2]
    for arm in draws:
        for i, d in enumerate(ladders):
            ns, ys, lo, hi = _series(arm, d, lambda a: a["seconds_median"], with_err=True)
            if not ns:
                continue
            c = colours.get(arm, COL["grey"])
            filled = i == 0
            ax1.errorbar(
                ns,
                ys,
                yerr=[lo, hi],
                marker=markers.get(arm, "o"),
                color=c,
                label=labels.get(arm, arm) if filled else None,
                ls="-" if filled else "--",
                mfc=c if filled else "white",
                capsize=1.8,
                elinewidth=0.5,
                markersize=2.8,
            )
    ax1.set_yscale("log")
    ax1.set_xscale("log", base=2)
    ax1.xaxis.set_minor_locator(ticker.NullLocator())
    ax1.set_ylabel("time per fit (s)")
    ax1.set_title(
        "(a) end-to-end time, tall, 8 blocks, full rank\n"
        "(solid $d=128$, dashed $d=32$; median, min/max over 5 repeats)",
        loc="left",
        fontsize=6.4,
        pad=18,
    )
    ax1.legend(fontsize=5.2, loc="lower left", borderpad=0.15, ncols=3, bbox_to_anchor=(0.0, 1.02), frameon=False)

    # ---- (b) accuracy ------------------------------------------------------
    for arm in draws:
        for i, d in enumerate(ladders):
            ns, ys, _, _ = _series(arm, d, lambda a: (a.get("accuracy") or {}).get("subspace_distance_vs_exact"))
            if not ns:
                continue
            c = colours.get(arm, COL["grey"])
            filled = i == 0
            ax2.plot(
                ns,
                ys,
                marker=markers.get(arm, "o"),
                color=c,
                ls="-" if filled else "--",
                mfc=c if filled else "white",
                markersize=2.8,
            )
    # error bars on accuracy: the metrics' floor is numerical roundoff; show a roundoff band
    ax2.set_yscale("log")
    ax2.set_xscale("log", base=2)
    ax2.xaxis.set_minor_locator(ticker.NullLocator())
    ax2.axhline(1e-15, color="0.4", ls=":", lw=0.7)
    ax2.text(
        ax2.get_xlim()[1] * 0.55,
        2.1e-15,
        "IEEE-754 double roundoff",
        fontsize=5.0,
        color="0.35",
        ha="center",
        va="bottom",
    )
    ax2.set_xlabel(r"rows per block $n_{\mathrm{block}}$ (8 blocks, full local rank)")
    ax2.set_ylabel("principal-subspace\ndistance vs exact SVD")
    ax2.set_title(
        "(b) principal-subspace distance vs exact batch SVD",
        loc="left",
        fontsize=6.6,
    )
    ax2.set_ylim(1e-16, 4.0)
    ax2.set_xticks([64, 128, 256, 512, 1024])
    ax2.set_xticklabels([str(n) for n in (64, 128, 256, 512, 1024)])
    ax2.xaxis.set_minor_locator(ticker.NullLocator())

    fig.savefig(out_dir / "baselines.png")
    plt.close(fig)
    return out_dir / "baselines.png"


if __name__ == "__main__":
    main()
