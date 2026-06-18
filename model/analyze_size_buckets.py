from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model.metrics import classification_metrics

METRIC_COLUMNS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "macro_f1",
    "auc",
)

GROUP_COLUMNS = (
    "dataset",
    "experiment",
    "trainer",
    "features",
    "graph_layers",
    "pooling",
    "bucket",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze model metrics by graph size bucket.")
    parser.add_argument("--input-dir", default="outputs/ablations")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--buckets", type=int, default=3)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def infer_path_metadata(input_dir: Path, prediction_path: Path) -> tuple[str, int | None, str]:
    rel = prediction_path.relative_to(input_dir)
    parts = rel.parts
    if len(parts) == 3:
        experiment, dataset, _ = parts
        return experiment, None, dataset
    if len(parts) == 4 and parts[1].startswith("seed_"):
        experiment, seed_part, dataset, _ = parts
        return experiment, int(seed_part.removeprefix("seed_")), dataset
    raise ValueError(f"Unexpected prediction path layout: {prediction_path}")


def bucket_labels(count: int) -> list[str]:
    if count == 3:
        return ["small", "medium", "large"]
    return [f"bucket_{idx + 1}" for idx in range(count)]


def add_size_buckets(df: pd.DataFrame, buckets: int) -> pd.DataFrame:
    bucket_count = min(buckets, len(df))
    labels = bucket_labels(bucket_count)
    ranked = df["node_count"].rank(method="first")
    df = df.copy()
    df["bucket"] = pd.qcut(ranked, q=bucket_count, labels=labels)
    return df


def load_run_metadata(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        return {}
    with metrics_path.open() as handle:
        payload = json.load(handle)
    return payload.get("args", {})


def feature_text(features: Any) -> str:
    if isinstance(features, str):
        return features
    if isinstance(features, list | tuple):
        return " ".join(str(item) for item in features)
    return ""


def metrics_for_bucket(df: pd.DataFrame) -> dict[str, float]:
    metrics = classification_metrics(
        df["y_true"].to_numpy(dtype=np.int64),
        df["y_pred"].to_numpy(dtype=np.int64),
        df["prob_fake"].to_numpy(dtype=np.float32),
    )
    return {name: metrics.get(name) for name in METRIC_COLUMNS}


def collect_bucket_rows(input_dir: Path, split: str, buckets: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prediction_path in sorted(input_dir.rglob(f"{split}_predictions.csv")):
        experiment, seed_from_path, dataset = infer_path_metadata(input_dir, prediction_path)
        metrics_path = prediction_path.with_name("metrics.json")
        metadata = load_run_metadata(metrics_path)
        seed = metadata.get("seed", seed_from_path)
        trainer = metadata.get("trainer", "graph")
        predictions = pd.read_csv(prediction_path)
        predictions = add_size_buckets(predictions, buckets)

        for bucket, bucket_df in predictions.groupby("bucket", observed=True):
            metrics = metrics_for_bucket(bucket_df)
            rows.append(
                {
                    "dataset": dataset,
                    "experiment": experiment,
                    "seed": seed,
                    "trainer": trainer,
                    "features": feature_text(metadata.get("features", "")),
                    "graph_layers": metadata.get("graph_layers"),
                    "pooling": metadata.get("pooling", "mean_max"),
                    "bucket": str(bucket),
                    "n": int(len(bucket_df)),
                    "fake_rate": float(bucket_df["y_true"].mean()),
                    "node_count_min": int(bucket_df["node_count"].min()),
                    "node_count_mean": float(bucket_df["node_count"].mean()),
                    "node_count_max": int(bucket_df["node_count"].max()),
                    **metrics,
                }
            )
    return rows


def aggregate_bucket_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[column] for column in GROUP_COLUMNS)
        groups.setdefault(key, []).append(row)

    aggregated: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        row = dict(zip(GROUP_COLUMNS, key, strict=True))
        row["n_seeds"] = len(group)
        row["seeds"] = " ".join(sorted(str(item["seed"]) for item in group))
        for column in ("n", "fake_rate", "node_count_min", "node_count_mean", "node_count_max"):
            values = [float(item[column]) for item in group]
            row[f"{column}_mean"] = statistics.mean(values)
            row[f"{column}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        for metric in METRIC_COLUMNS:
            values = [float(item[metric]) for item in group if item.get(metric) is not None]
            row[f"{metric}_mean"] = statistics.mean(values) if values else None
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregated.append(row)
    return aggregated


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    rows = collect_bucket_rows(input_dir, args.split, args.buckets)
    aggregate_rows = aggregate_bucket_rows(rows)

    split_prefix = f"{args.split}_size_bucket"
    detail_csv = output_dir / f"{split_prefix}_summary.csv"
    aggregate_csv = output_dir / f"{split_prefix}_summary_by_experiment.csv"
    detail_json = output_dir / f"{split_prefix}_summary.json"
    aggregate_json = output_dir / f"{split_prefix}_summary_by_experiment.json"

    write_csv(detail_csv, rows)
    write_csv(aggregate_csv, aggregate_rows)
    detail_json.write_text(json.dumps(rows, indent=2))
    aggregate_json.write_text(json.dumps(aggregate_rows, indent=2))

    print(f"saved bucket summary: {detail_csv}")
    print(f"saved bucket summary: {aggregate_csv}")


if __name__ == "__main__":
    main()
