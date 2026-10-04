#!/usr/bin/env python3
"""Generate the mergeable-PCA paper's data figures from the benchmark artifacts.

Every plotted value is read from a JSON result file written by the benchmark
harness (``benchmark/mergeable_pca/results/*.json``); nothing in this script is
a hand-entered number.  The figures are vector PDFs sized for a single-column
ACM layout (3.3 in) with type set to remain legible at final print size.

The three data figures are:

  bytes_crossed.pdf    bytes crossing the coupling boundary, legacy full-chunk
                       scatter versus the bridge-side mergeable summary, full
                       local rank, over the n_block/d sweep.
  local_rank_curve.pdf local rank R against summary size and against the two
                       sign-invariant accuracy metrics, for a single leaf and
                       for the eight-leaf merge.
  gysela_sizing.pdf    the two gysela flattenings for the production-scale mesh:
                       compression ratio (slab / summary) versus rank count.

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
# Figure B1: bytes crossing the boundary.
# --------------------------------------------------------------------------
def figure_bytes(results_dir: Path, out_dir: Path) -> Path:
    b1 = _load(results_dir, "b1_bytes_crossing.json")
    # full local rank only, so the summary rank is not a second free variable.
    rows = [r for r in b1["results"] if r["local_rank_requested"] is None]
    dims = sorted({r["n_features"] for r in rows})

    fig, ax = plt.subplots(figsize=(ACM_COLUMN_IN, 2.5))
    colours = {32: COL["blue"], 128: COL["orange"], 512: COL["green"]}
    for d in dims:
        c = colours[d]
        pts = sorted((r for r in rows if r["n_features"] == d), key=lambda r: r["n_block_over_d"])
        x = [r["n_block_over_d"] for r in pts]
        legacy = [r["legacy_wire_bytes"]["bytes"] for r in pts]
        summary = [r["pca_wire_bytes"]["bytes"] for r in pts]
        ax.plot(x, legacy, "-", color=c, marker="o", clip_on=False)
        ax.plot(x, summary, "--", color=c, marker="s", clip_on=False)

    ax.axvspan(1.0, 9.0, color="0.85", alpha=0.4, zorder=0, linewidth=0)
    ax.axvline(1.0, color="0.35", ls=":", lw=0.8, zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 4, 8])
    ax.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8"])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(2, 5)))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlim(0.2, 11)
    ax.set_ylim(1.2e3, 4e7)
    ax.set_xlabel(r"block aspect ratio $n_{\mathrm{block}}/d$ (dimensionless)")
    ax.set_ylabel("bytes crossing the boundary (bytes)")
    ax.set_title("Full local rank: tall blocks compress, flat blocks do not")

    path_leg = [
        Line2D([], [], color="0.2", ls="-", marker="o", label="legacy full chunk"),
        Line2D([], [], color="0.2", ls="--", marker="s", label="bridge summary"),
    ]
    dim_leg = [Line2D([], [], color=colours[d], ls="-", marker="o", label=rf"$d={d}$") for d in dims]
    l1 = ax.legend(handles=path_leg, loc="upper left", fontsize=5.9, borderpad=0.1)
    ax.add_artist(l1)
    ax.legend(handles=dim_leg, loc="lower right", fontsize=5.9, borderpad=0.1)
    ax.text(2.95, 6.5e6, "tall", color="0.35", fontsize=6.5, ha="center")
    ax.text(0.5, 6.5e6, "flat /\nsquare", color="0.35", fontsize=6.5, ha="center")
    fig.savefig(out_dir / "bytes_crossed.pdf")
    plt.close(fig)
    return out_dir / "bytes_crossed.pdf"


# --------------------------------------------------------------------------
# Figure B4: local-rank accuracy and bandwidth curve.
# --------------------------------------------------------------------------
def figure_local_rank(results_dir: Path, out_dir: Path) -> Path:
    b4 = _load(results_dir, "b4_rank_accuracy_curve.json")
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

    fig.savefig(out_dir / "local_rank_curve.pdf")
    plt.close(fig)
    return out_dir / "local_rank_curve.pdf"


# --------------------------------------------------------------------------
# Figure sizing: the two gysela flattenings on the production mesh.
# --------------------------------------------------------------------------
def figure_sizing(results_dir: Path, out_dir: Path) -> Path:
    g = _load(results_dir, "gysela_sizing.json")
    mesh = [512, 128, 64, 128, 8]  # production-scale mesh named in the paper

    def series(layout, sub):
        rows = [r for r in g[layout][sub] if r["mesh_tor1_tor2_tor3_vpar_mu"] == mesh]
        rows = sorted(rows, key=lambda r: r["n_ranks"])
        return ([r["n_ranks"] for r in rows], [r["compression_ratio_slab_over_summary"] for r in rows])

    fig, ax = plt.subplots(figsize=(ACM_COLUMN_IN, 2.5))
    specs = [
        ("layout_a_velocity_space", "full_rank", COL["blue"], "-", "o", "A velocity: full rank"),
        ("layout_a_velocity_space", "truncated", COL["blue"], "--", "s", r"A velocity: $R=32$"),
        ("layout_b_spatial_box", "full_rank", COL["red"], "-", "o", "B spatial: full rank"),
        ("layout_b_spatial_box", "truncated", COL["red"], "--", "s", r"B spatial: $R=32$"),
    ]
    for layout, sub, c, ls, mk, label in specs:
        x, y = series(layout, sub)
        ax.plot(x, y, ls, color=c, marker=mk, label=label)

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
    ax.set_xlabel("number of ranks")
    ax.set_ylabel("compression ratio\n(slab / summary, dimensionless)")
    ax.set_title("Production mesh 512$\\times$128$\\times$64$\\times$128$\\times$8")
    ax.text(5.3, 1.35, "break-even (ratio 1)", fontsize=6.0, color="0.35")
    ax.legend(loc="center left", bbox_to_anchor=(0.015, 0.45))
    fig.savefig(out_dir / "gysela_sizing.pdf")
    plt.close(fig)
    return out_dir / "gysela_sizing.pdf"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("benchmark/mergeable_pca/results"))
    ap.add_argument("--out", type=Path, default=Path("benchmark/mergeable_pca/figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    made = [
        figure_bytes(args.results, args.out),
        figure_local_rank(args.results, args.out),
        figure_sizing(args.results, args.out),
    ]
    for m in made:
        print("wrote", m)


if __name__ == "__main__":
    main()
