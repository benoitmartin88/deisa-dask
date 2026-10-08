"""Grand-scale network transfer sweep: 1/2/4 GiB total data.

Repeats the legacy-vs-summary wire comparison at the byte volumes the production
application produces per output step, as far as this machine's memory allows
(the container holds a 100 GiB cgroup cap; totals beyond 4 GiB were not
attempted here). Each run holds the total in 16 MiB blocks at d=512, full local
rank, builds one summary per block sequentially, then merges the whole pool.

Writes ``results/network_transfer_grand_scale.json`` with, per total: the
measured legacy and summary wire bytes, the payload ratio, the leaf-build and
merge-tree wall times, and the process peak RSS.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
from measurement_common import serialized_nbytes

from deisa.dask.mergeable_pca import local_pca, merge_tree

#: (blocks, d, rows-per-block) per rung of the ladder; 16 MiB blocks each.
RUNGS = ((64, 512, 4096), (128, 512, 4096), (256, 512, 4096))

ARTIFACT = Path(__file__).with_name("results") / "network_transfer_grand_scale.json"


def measure() -> list[dict]:
    rows: list[dict] = []
    for blocks, d, rows_per_block in RUNGS:
        t0 = time.perf_counter()
        summaries, wire = [], []
        for i in range(blocks):
            block = np.random.default_rng(1000 + i).standard_normal((rows_per_block, d))
            summary = local_pca(block, rank=None)
            wire.append((serialized_nbytes(block), serialized_nbytes(summary)))
            del block
            summaries.append(summary)
        t_build = time.perf_counter() - t0
        t0 = time.perf_counter()
        root = merge_tree(summaries)
        t_merge = time.perf_counter() - t0
        legacy_total = sum(b for b, _ in wire)
        summary_total = sum(s for _, s in wire)
        rows.append(
            {
                "total_data_bytes_total": legacy_total,
                "total_data_GiB": legacy_total / 2**30,
                "blocks": blocks,
                "per_block_bytes": rows_per_block * d * 8,
                "total_summary_bytes": summary_total,
                "summary_compression_ratio": legacy_total / summary_total,
                "leaf_build_seconds": t_build,
                "merge_tree_seconds": t_merge,
                "root_rank": int(root.rank),
                "peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            }
        )
        print(rows[-1], flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    rows = measure()
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "script": "network_transfer_grand_scale",
        "results": rows,
        "note": (
            "grand-scale sweep: 1/2/4 GiB total data at d=512, 16 MiB blocks, "
            "full local rank; leaf build + merge_tree wall times"
        ),
    }
    ARTIFACT.write_text(json.dumps(payload, indent=2))
    print(f"artifact written: {ARTIFACT}")


if __name__ == "__main__":
    main()
