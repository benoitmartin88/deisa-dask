"""Producer-side CPU overhead: the cost a simulation rank pays for its per-step local PCA.

The designer's burden the paper claims: the producer does ONE ``local_pca`` of its own chunk per output
step, and the merge cost belongs to the analytics side, not the producer. This benchmark times exactly
that producer-side step on a simulated per-step cadence, for both flattening conventions the Gysela
application uses (the paper's velocity-space Layout A and spatial Layout B) and at the three chunk
volumes the grand-scale sweep covers.

Per configuration and per chunk volume it reports, all from a single repeated-timing policy with
median / min / max / IQR / stddev so a reader can separate signal from run noise:
    - ``flatten_seconds``      : reshape/transpose cost of turning the 5-D chunk into (n, d)
    - ``local_pca_seconds``    : centering + SVD (+ truncation) cost
    - ``total_seconds``        : flatten + PCA, the full producer-side step
    - ``chunk_scale_GiBps``    : producer throughput in chunk-volume per second
    - ``pca_flops_approx``     : the O(n_block * d^2) flop count of the economy SVD, for the paper's
                                 complexity claim

All work happens on a GPU-free CPU path identical to what a bridge callable executes
(``local_pca_from_chunk`` -> ``local_pca``), so the numbers ARE the bridge-side CPU cost.

Writes ``results/producer_overhead.json``.
"""

import os

# Pin BLAS to the true core count once: this is a STANDALONE producer measurement, not a pool,
# so one process legitimately uses all cores, matching how a simulation rank would call LAPACK.
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count() or 4))

import json
import sys
from pathlib import Path

import numpy as np
from measurement_common import time_repeated

from deisa.dask.mergeable_pca import local_pca_from_chunk

_BENCH_DIR = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)  # noqa: E402  (must precede the local import above)

ARTIFACT = Path(__file__).with_name("results") / "producer_overhead.json"

#: The mesh the mini-app initializes, scaled down proportionally to per-rank chunks that fit RAM.
#: The full 2048x256x64x128x8 field is 100 TB-class; per-rank chunks in the grand-scale sweep are
#: 512 MiB..4 GiB. We take per-rank chunks of 512 MiB, 2 GiB and 4 GiB at d=512 with 5-D axes laid
#: out so samples = 3 axes and features = 2 axes (the flatten policy's shapes).
CHUNK_SHAPES = (
    # (label, shape, feature_axes)  -- shape named in the 5-D convention (r, theta, phi, vpar, mu).
    ("512MiB_layoutA", (32, 16, 512, 96, 88), (3, 4)),  # features = velocity axes (d = 96*88... no, keep d=512)
)


def build_5d_chunk(shape, feature_axes, seed=7):
    """A deterministic 5-D chunk with the given axis extents (kept small enough to fit)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape)


def flatten_ref_5d(chunk, feature_axes):
    """Reference flattening identical in cost model to _flatten_chunk_to_2d: transpose + reshape."""
    ndim = chunk.ndim
    feature_axes = tuple(int(a) % ndim for a in feature_axes)
    sample_axes = tuple(a for a in range(ndim) if a not in feature_axes)
    permutation = sample_axes + feature_axes
    n_samples = int(np.prod([chunk.shape[a] for a in sample_axes]))
    n_features = int(np.prod([chunk.shape[a] for a in feature_axes]))
    return np.reshape(np.transpose(chunk, axes=permutation), (n_samples, n_features))


def measure_chunk(label, shape, feature_axes, rank, repeats=None):
    chunk = build_5d_chunk(shape, feature_axes, seed=7)

    def flatten_step():
        return flatten_ref_5d(chunk, feature_axes)

    flat = flatten_step()
    n, d = flat.shape

    def pca_step():
        return local_pca_from_chunk(chunk, rank=rank, feature_axes=feature_axes)

    t_flat = time_repeated(flatten_step, repeats=repeats)
    t_pca = time_repeated(pca_step, repeats=repeats)

    total_med = t_flat["seconds_median"] + t_pca["seconds_median"]
    volume = chunk.nbytes
    flops = 2.0 * n * d * d  # economy SVD ~ O(n d^2), constant folded for reporting
    return {
        "label": label,
        "shape": list(shape),
        "feature_axes": list(feature_axes),
        "n_samples": n,
        "d": d,
        "rank_cap": rank,
        "chunk_GiB": volume / 2**30,
        "flatten_seconds": t_flat,
        "local_pca_seconds": t_pca,
        "total_seconds_median": total_med,
        "chunk_scale_GiBps": volume / 2**30 / total_med,
        "pca_flops_approx": flops,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=None)
    args = parser.parse_args()

    rows = []
    # Layout A (velocity features): the velocity axes (vpar, mu) are the FEATURES, d = 32*16 = 512,
    # and the spatial axes scale the sample count. Chunk volumes match the grand-scale rungs.
    # Layout B (spatial features): the spatial axes (r, theta, phi) are the features, the velocity
    # axes are the samples, so d grows and n stays small -- the paper's non-compressing regime.
    layouts = [
        # (label, shape, feature_axes, rank)
        ("layoutA_512MiB", (512, 2, 128, 32, 16), (3, 4), None),  # n=131072, d=512, 512 MiB
        ("layoutA_2GiB", (1024, 8, 64, 32, 16), (3, 4), None),  # n=524288, d=512, 2 GiB
        ("layoutA_4GiB", (2048, 8, 64, 32, 16), (3, 4), None),  # n=1048576, d=512, 4 GiB
        ("layoutA_512MiB_cap32", (512, 2, 128, 32, 16), (3, 4), 32),
        ("layoutA_2GiB_cap32", (1024, 8, 64, 32, 16), (3, 4), 32),
        ("layoutA_4GiB_cap32", (2048, 8, 64, 32, 16), (3, 4), 32),
        # Layout B: features are spatial; the per-chunk volumes are matched by growing velocity
        # extents (the realistic case: vpar/mu refine faster than the box grows).
        ("layoutB_512MiB", (32, 64, 32, 64, 16), (0, 1, 2), None),  # d=65536, n=1024, 0.5 GiB
        ("layoutB_8GiB_a", (128, 64, 64, 32, 64), (0, 1, 2), None),  # d=524288, n=2048, 8 GiB
        ("layoutB_8GiB_b", (128, 64, 64, 32, 64), (0, 1, 2), None),  # repeat rung for run-to-run noise
    ]
    for label, shape, feature_axes, rank in layouts:
        print(f"=== {label}: shape {shape}, feature_axes {feature_axes}, rank {rank} ===", flush=True)
        rows.append(measure_chunk(label, shape, feature_axes, rank, repeats=args.repeats))
        r = rows[-1]
        print(
            f"    flatten {r['flatten_seconds']['seconds_median'] * 1000:.1f} ms | "
            f"pca {r['local_pca_seconds']['seconds_median'] * 1000:.1f} ms | "
            f"{r['chunk_GiB']:.2f} GiB | {r['chunk_scale_GiBps']:.2f} GiB/s",
            flush=True,
        )

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "script": "producer_overhead",
        "results": rows,
        "note": (
            "producer-side CPU overhead per output step: flatten + local PCA on 5-D Gysela-shaped "
            "chunks in a standalone process with all cores for one LAPACK call (how a simulation "
            "rank runs); both layouts of the paper's evaluation, full local rank and cap R=32"
        ),
    }
    ARTIFACT.write_text(json.dumps(payload, indent=2))
    print(f"artifact written: {ARTIFACT}")


if __name__ == "__main__":
    main()
