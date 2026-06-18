from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from model.data import FEATURE_FILES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect raw FakeNewsNet-derived graph data.")
    parser.add_argument("--data-root", default="database/data")
    parser.add_argument("--dataset", choices=["gossipcop", "politifact"], default="politifact")
    return parser.parse_args()


def sparse_shape(path: Path) -> list[int]:
    with np.load(path) as archive:
        return archive["shape"].astype(int).tolist()


def edge_count(path: Path) -> int:
    with path.open("r") as handle:
        return sum(1 for _ in handle)


def label_counts(labels: np.ndarray) -> dict[str, int]:
    counts = np.bincount(labels.astype(int), minlength=2)
    return {"real_0": int(counts[0]), "fake_1": int(counts[1])}


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.data_root) / args.dataset / "raw"
    labels = np.load(raw_dir / "graph_labels.npy")
    node_graph_id = np.load(raw_dir / "node_graph_id.npy")

    breaks = np.flatnonzero(node_graph_id[1:] != node_graph_id[:-1]) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [len(node_graph_id)]))
    graph_sizes = ends - starts

    split_summary = {}
    for split in ["train", "val", "test"]:
        indices = np.load(raw_dir / f"custom_{split}_idx.npy")
        split_summary[split] = {
            "graphs": int(len(indices)),
            "labels": label_counts(labels[indices]),
        }

    summary = {
        "dataset": args.dataset,
        "graphs": int(len(labels)),
        "nodes": int(len(node_graph_id)),
        "edges": edge_count(raw_dir / "A.txt"),
        "labels": label_counts(labels),
        "graph_size": {
            "min": int(graph_sizes.min()),
            "median": float(np.median(graph_sizes)),
            "mean": float(graph_sizes.mean()),
            "max": int(graph_sizes.max()),
        },
        "splits": split_summary,
        "features": {
            name: sparse_shape(raw_dir / filename)
            for name, filename in FEATURE_FILES.items()
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

