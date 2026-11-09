"""Extended grand-scale sweep: 8/16/32 GiB with capped summaries (and a full-rank 8 GiB arm).

Extends ``grand_scale_transfer.py``, which stops at 4 GiB because each rung there builds FULL local-rank
summaries sequentially. Three changes push the measured frontier up:

1. The BLAS backend is pinned to one thread per process, and the assignment happens BEFORE numpy is
   imported because OpenBLAS pins its thread pool once at import time; setting the same variables
   later has no effect. Without the pin, a 32-way pool each spawning ~40 OpenBLAS threads
   oversubscribes 30x (observed load average 610 on a 40-core host) and inflates a 0.6 s SVD to
   ~14 s, which poisons both the timing and the host for everything else.
2. Leaf builds run in parallel across processes; the leaves are independent. ``merge_tree`` stays
   sequential in the parent process.
3. The headline rungs cap the summary at R=32, the operating point the paper evaluates, which shrinks
   each summary from 2 MiB (512x512, full rank) to 128 KiB (32x512) and keeps the summary pool small
   at large totals. One full-rank arm at 8 GiB is kept as a direct comparison against the old 4 GiB
   full-rank result.

Writes ``results/network_transfer_grand_scale_extended.json`` with, per rung: the measured wire bytes,
the compression ratio, leaf-build and merge-tree wall times, process peak RSS, and the configuration.
"""

import os

# MUST run before numpy is imported: OpenBLAS reads these at import and never re-reads them.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from measurement_common import serialized_nbytes

from deisa.dask.mergeable_pca import PCASummary, local_pca, merge_tree

_BENCH_DIR = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)  # noqa: E402  (must precede the local import above)

ARTIFACT = Path(__file__).with_name("results") / "network_transfer_grand_scale_extended.json"

#: (total GiB label, blocks, d, rows-per-block, rank-cap) per rung; 16 MiB blocks at d=512.
RUNGS = (
    (8, 512, 512, 4096, 32),
    (16, 1024, 512, 4096, 32),
    (32, 2048, 512, 4096, 32),
    (8, 512, 512, 4096, None),  # full-rank arm, comparable to the old 4 GiB full-rank rungs
)

_WORKER: dict = {}


def _init_worker(d: int, rows_per_block: int, rank: int | None, seeds_base: int) -> None:
    """Record the rung config in the worker. BLAS threading is already pinned at module import."""
    _WORKER.update(d=d, rows=rows_per_block, rank=rank, base=seeds_base)


def _build_leaf_and_summary(i: int) -> PCASummary:
    block = np.random.default_rng(_WORKER["base"] + i).standard_normal((_WORKER["rows"], _WORKER["d"]))
    try:
        return local_pca(block, rank=_WORKER["rank"])
    finally:
        del block


def build_leaves_parallel(
    blocks: int, d: int, rows_per_block: int, rank: int | None, seeds_base: int, procs: int
) -> tuple[list[PCASummary], float]:
    """Build ``blocks`` leaf summaries in parallel and return (summaries, wall seconds)."""
    summaries: list = [None] * blocks
    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=procs, initializer=_init_worker, initargs=(d, rows_per_block, rank, seeds_base)
    ) as pool:
        for i, summary in enumerate(pool.map(_build_leaf_and_summary, range(blocks), chunksize=8)):
            summaries[i] = summary
    wall = time.perf_counter() - t0
    assert all(s is not None for s in summaries)
    return summaries, wall


def measure_rung(blocks: int, d: int, rows_per_block: int, rank: int | None, seeds_base: int, procs: int) -> dict:
    """Build ``blocks`` leaves in parallel, merge sequentially, and report byte + time accounting."""
    # One reference block for the wire-size math; all blocks share the shape, so one measurement
    # times the block count gives exact totals without ever holding more than one block here.
    ref = np.random.default_rng(seeds_base).standard_normal((rows_per_block, d))
    legacy_wire = serialized_nbytes(ref)
    ref_summary = local_pca(ref, rank=rank)
    del ref
    summary_wire = serialized_nbytes(ref_summary)
    del ref_summary

    leaves, t_build = build_leaves_parallel(blocks, d, rows_per_block, rank, seeds_base, procs)
    t0 = time.perf_counter()
    root = merge_tree(leaves)
    t_merge = time.perf_counter() - t0

    legacy_total = legacy_wire * blocks
    summary_total = summary_wire * blocks
    return {
        "total_data_GiB": legacy_total / 2**30,
        "blocks": blocks,
        "d": d,
        "rows_per_block": rows_per_block,
        "rank_cap": rank,
        "per_block_bytes": rows_per_block * d * 8,
        "total_summary_bytes": summary_total,
        "summary_compression_ratio": legacy_total / summary_total,
        "leaf_build_seconds_parallel": t_build,
        "leaf_procs": procs,
        "merge_tree_seconds": t_merge,
        "root_rank": int(root.rank),
        "peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--procs", type=int, default=min(32, (os.cpu_count() or 4)))
    parser.add_argument("--rungs", type=int, default=None, help="Run only the first N rungs.")
    args = parser.parse_args()

    rungs = RUNGS if args.rungs is None else RUNGS[: args.rungs]
    rows = []
    for _label, blocks, d, rows_per_block, rank in rungs:
        print(
            f"=== rung {blocks} blocks x 16 MiB, d={d}, cap={rank}, procs={args.procs} ===",
            flush=True,
        )
        rows.append(measure_rung(blocks, d, rows_per_block, rank, 1000, args.procs))
        print(rows[-1], flush=True)

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "script": "network_transfer_grand_scale_extended",
        "results": rows,
        "note": (
            "extended grand-scale sweep: 8/16/32 GiB totals at d=512, 16 MiB blocks, cap R=32 "
            "plus one full-rank 8 GiB arm; parallel leaf builds (1 BLAS thread/proc), "
            "sequential merge_tree"
        ),
    }
    ARTIFACT.write_text(json.dumps(payload, indent=2))
    print(f"artifact written: {ARTIFACT}")


if __name__ == "__main__":
    main()
