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

MATCH_COLUMNS = (
    "matched_pairs",
    "matched_n",
    "matched_fake_rate",
    "match_distance_mean",
    "match_distance_std",
    "match_distance_max",
    "node_count_real_mean",
    "node_count_fake_mean",
    "match_value_real_mean",
    "match_value_fake_mean",
    "match_smd_before",
    "match_smd_after",
)

GROUP_COLUMNS = (
    "dataset",
    "experiment",
    "trainer",
    "features",
    "graph_layers",
    "pooling",
    "match_on",
    "caliper",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate predictions on a real/fake graph-size matched subset."
    )
    parser.add_argument("--input-dir", default="outputs/ablations")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--match-on",
        choices=["log_node_count", "node_count"],
        default="log_node_count",
        help="Graph-size variable used for nearest-neighbor matching.",
    )
    parser.add_argument(
        "--caliper",
        type=float,
        default=None,
        help="Optional maximum absolute distance on the matching variable.",
    )
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


def match_values(df: pd.DataFrame, match_on: str) -> pd.Series:
    if match_on == "node_count":
        return df["node_count"].astype(float)
    if match_on == "log_node_count":
        return np.log1p(df["node_count"].astype(float))
    raise ValueError(f"Unsupported match variable: {match_on}")


def standardized_mean_difference(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) == 0 or len(b) == 0:
        return None
    pooled = np.sqrt((np.var(a, ddof=0) + np.var(b, ddof=0)) / 2.0)
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def greedy_class_match(
    df: pd.DataFrame,
    match_on: str,
    caliper: float | None,
) -> tuple[pd.DataFrame, dict[str, float | int | None]]:
    df = df.copy()
    df["match_value"] = match_values(df, match_on)
    real = df[df["y_true"] == 0].copy()
    fake = df[df["y_true"] == 1].copy()
    if real.empty or fake.empty:
        return df.iloc[0:0].copy(), {
            "matched_pairs": 0,
            "matched_n": 0,
            "matched_fake_rate": None,
            "match_distance_mean": None,
            "match_distance_std": None,
            "match_distance_max": None,
            "node_count_real_mean": None,
            "node_count_fake_mean": None,
            "match_value_real_mean": None,
            "match_value_fake_mean": None,
            "match_smd_before": standardized_mean_difference(
                real["match_value"].to_numpy(),
                fake["match_value"].to_numpy(),
            ),
            "match_smd_after": None,
        }

    if len(real) <= len(fake):
        anchors = real.sort_values("match_value")
        candidates = fake.sort_values("match_value")
    else:
        anchors = fake.sort_values("match_value")
        candidates = real.sort_values("match_value")

    unused = set(candidates.index.tolist())
    matched_indices: list[int] = []
    distances: list[float] = []
    for anchor_idx, anchor in anchors.iterrows():
        if not unused:
            break
        candidate_indices = np.array(list(unused), dtype=np.int64)
        candidate_values = candidates.loc[candidate_indices, "match_value"].to_numpy()
        candidate_distances = np.abs(candidate_values - float(anchor["match_value"]))
        best_pos = int(np.argmin(candidate_distances))
        best_distance = float(candidate_distances[best_pos])
        if caliper is not None and best_distance > caliper:
            continue
        candidate_idx = int(candidate_indices[best_pos])
        unused.remove(candidate_idx)
        matched_indices.extend([int(anchor_idx), candidate_idx])
        distances.append(best_distance)

    matched = df.loc[matched_indices].copy()
    matched_real = matched[matched["y_true"] == 0]
    matched_fake = matched[matched["y_true"] == 1]
    before_smd = standardized_mean_difference(
        real["match_value"].to_numpy(),
        fake["match_value"].to_numpy(),
    )
    after_smd = standardized_mean_difference(
        matched_real["match_value"].to_numpy(),
        matched_fake["match_value"].to_numpy(),
    )
    stats = {
        "matched_pairs": len(distances),
        "matched_n": len(matched),
        "matched_fake_rate": float(matched["y_true"].mean()) if len(matched) else None,
        "match_distance_mean": statistics.mean(distances) if distances else None,
        "match_distance_std": statistics.stdev(distances) if len(distances) > 1 else 0.0,
        "match_distance_max": max(distances) if distances else None,
        "node_count_real_mean": (
            float(matched_real["node_count"].mean()) if len(matched_real) else None
        ),
        "node_count_fake_mean": (
            float(matched_fake["node_count"].mean()) if len(matched_fake) else None
        ),
        "match_value_real_mean": (
            float(matched_real["match_value"].mean()) if len(matched_real) else None
        ),
        "match_value_fake_mean": (
            float(matched_fake["match_value"].mean()) if len(matched_fake) else None
        ),
        "match_smd_before": before_smd,
        "match_smd_after": after_smd,
    }
    return matched, stats


def matched_metrics(df: pd.DataFrame) -> dict[str, float | None]:
    if df.empty or df["y_true"].nunique() < 2:
        return {name: None for name in METRIC_COLUMNS}
    metrics = classification_metrics(
        df["y_true"].to_numpy(dtype=np.int64),
        df["y_pred"].to_numpy(dtype=np.int64),
        df["prob_fake"].to_numpy(dtype=np.float32),
    )
    return {name: metrics.get(name) for name in METRIC_COLUMNS}


def collect_matched_rows(
    input_dir: Path,
    split: str,
    match_on: str,
    caliper: float | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prediction_path in sorted(input_dir.rglob(f"{split}_predictions.csv")):
        experiment, seed_from_path, dataset = infer_path_metadata(input_dir, prediction_path)
        metrics_path = prediction_path.with_name("metrics.json")
        metadata = load_run_metadata(metrics_path)
        seed = metadata.get("seed", seed_from_path)
        trainer = metadata.get("trainer", "graph")
        predictions = pd.read_csv(prediction_path)
        matched, match_stats = greedy_class_match(predictions, match_on, caliper)
        rows.append(
            {
                "dataset": dataset,
                "experiment": experiment,
                "seed": seed,
                "trainer": trainer,
                "features": feature_text(metadata.get("features", "")),
                "graph_layers": metadata.get("graph_layers"),
                "pooling": metadata.get("pooling", "mean_max"),
                "match_on": match_on,
                "caliper": caliper,
                "original_n": int(len(predictions)),
                "original_fake_rate": float(predictions["y_true"].mean()),
                **match_stats,
                **matched_metrics(matched),
            }
        )
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[column] for column in GROUP_COLUMNS)
        groups.setdefault(key, []).append(row)

    aggregated: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        row = dict(zip(GROUP_COLUMNS, key, strict=True))
        row["n_seeds"] = len(group)
        row["seeds"] = " ".join(sorted(str(item["seed"]) for item in group))
        for column in ("original_n", "original_fake_rate", *MATCH_COLUMNS, *METRIC_COLUMNS):
            values = [float(item[column]) for item in group if item.get(column) is not None]
            row[f"{column}_mean"] = statistics.mean(values) if values else None
            row[f"{column}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
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
    rows = collect_matched_rows(input_dir, args.split, args.match_on, args.caliper)
    aggregate = aggregate_rows(rows)

    split_prefix = f"{args.split}_matched"
    detail_csv = output_dir / f"{split_prefix}_summary.csv"
    aggregate_csv = output_dir / f"{split_prefix}_summary_by_experiment.csv"
    detail_json = output_dir / f"{split_prefix}_summary.json"
    aggregate_json = output_dir / f"{split_prefix}_summary_by_experiment.json"

    write_csv(detail_csv, rows)
    write_csv(aggregate_csv, aggregate)
    detail_json.write_text(json.dumps(rows, indent=2))
    aggregate_json.write_text(json.dumps(aggregate, indent=2))

    print(f"saved matched summary: {detail_csv}")
    print(f"saved matched summary: {aggregate_csv}")


if __name__ == "__main__":
    main()
