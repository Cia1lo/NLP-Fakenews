from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model.data import RawGraphStore
from model.metrics import classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train graph-size-only control classifier.")
    parser.add_argument("--data-root", default="database/data")
    parser.add_argument("--dataset", choices=["gossipcop", "politifact"], default="politifact")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--output-dir", default="outputs/size_control")
    parser.add_argument("--max-iter", type=int, default=1000)
    return parser.parse_args()


def graph_size_features(store: RawGraphStore) -> tuple[np.ndarray, list[str]]:
    node_count = (store.graph_ends - store.graph_starts).astype(np.float32)
    edge_count = (store.edge_ptr[1:] - store.edge_ptr[:-1]).astype(np.float32)
    edge_per_node = edge_count / np.maximum(node_count, 1.0)
    x = np.column_stack(
        [
            node_count,
            edge_count,
            np.log1p(node_count),
            np.log1p(edge_count),
            edge_per_node,
        ]
    ).astype(np.float32)
    names = [
        "node_count",
        "edge_count",
        "log_node_count",
        "log_edge_count",
        "edge_per_node",
    ]
    return x, names


def evaluate_split(
    model: object,
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    y_true = y[indices]
    prob = model.predict_proba(x[indices])[:, 1]
    pred = (prob >= 0.5).astype(np.int64)
    return classification_metrics(y_true, pred, prob)


def write_split_predictions(
    path: Path,
    model: object,
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
) -> None:
    prob = model.predict_proba(x[indices])[:, 1]
    pred = (prob >= 0.5).astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("graph_id", "node_count", "y_true", "y_pred", "prob_fake"),
        )
        writer.writeheader()
        for graph_id, node_count, label, prediction, fake_prob in zip(
            indices,
            x[indices, 0],
            y[indices],
            pred,
            prob,
            strict=True,
        ):
            writer.writerow(
                {
                    "graph_id": int(graph_id),
                    "node_count": int(node_count),
                    "y_true": int(label),
                    "y_pred": int(prediction),
                    "prob_fake": float(fake_prob),
                }
            )


def main() -> None:
    args = parse_args()
    store = RawGraphStore(args.data_root, args.dataset, features=())
    x, feature_names = graph_size_features(store)
    y = store.labels

    train_idx = store.split_indices("train")
    val_idx = store.split_indices("val")
    test_idx = store.split_indices("test")

    class_weight = "balanced" if args.class_weight == "balanced" else None
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight=class_weight,
            max_iter=args.max_iter,
            random_state=args.seed,
        ),
    )
    model.fit(x[train_idx], y[train_idx])

    val_metrics = evaluate_split(model, x, y, val_idx)
    test_metrics = evaluate_split(model, x, y, test_idx)

    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    write_split_predictions(output_dir / "val_predictions.csv", model, x, y, val_idx)
    write_split_predictions(output_dir / "test_predictions.csv", model, x, y, test_idx)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "best_epoch": 0,
                "best_val_macro_f1": val_metrics["macro_f1"],
                "val": val_metrics,
                "test": test_metrics,
                "args": {
                    **vars(args),
                    "features": ["graph_size"],
                    "graph_layers": 0,
                    "pooling": "size_only",
                    "trainer": "size_control",
                    "size_features": feature_names,
                },
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "trainer": "size_control",
                "feature_names": feature_names,
                "train_graphs": int(len(train_idx)),
                "val_macro_f1": val_metrics["macro_f1"],
                "test_macro_f1": test_metrics["macro_f1"],
            },
            indent=2,
        )
    )
    print(f"saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
