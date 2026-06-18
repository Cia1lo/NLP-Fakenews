from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

METRICS = ("macro_f1", "auc")

GROUP_COLUMNS = (
    "dataset",
    "experiment",
    "features",
    "graph_layers",
    "pooling",
    "explanation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize occlusion-style explanation deltas from robustness results."
    )
    parser.add_argument("--input", default="outputs/ablations/test_robustness_summary.csv")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def explanation_name(perturbation: str) -> str:
    if perturbation.startswith("zero_"):
        return f"occlude_{perturbation.removeprefix('zero_')}"
    if perturbation.startswith("noise_"):
        return f"noise_{perturbation.removeprefix('noise_')}"
    if perturbation.startswith("edge_drop_"):
        return perturbation
    return perturbation


def numeric(row: dict[str, str], column: str) -> float | None:
    value = row.get(column)
    if value in {None, ""}:
        return None
    return float(value)


def collect_explanation_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    clean_by_run: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        if row["perturbation"] == "clean":
            clean_by_run[(row["dataset"], row["experiment"], row["seed"])] = row

    explanation_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["perturbation"] == "clean":
            continue
        clean = clean_by_run.get((row["dataset"], row["experiment"], row["seed"]))
        if clean is None:
            continue
        output: dict[str, Any] = {
            "dataset": row["dataset"],
            "experiment": row["experiment"],
            "seed": row["seed"],
            "features": row["features"],
            "graph_layers": row["graph_layers"],
            "pooling": row["pooling"],
            "perturbation": row["perturbation"],
            "explanation": explanation_name(row["perturbation"]),
        }
        for metric in METRICS:
            clean_value = numeric(clean, metric)
            perturbed_value = numeric(row, metric)
            output[f"clean_{metric}"] = clean_value
            output[f"perturbed_{metric}"] = perturbed_value
            output[f"delta_{metric}"] = (
                clean_value - perturbed_value
                if clean_value is not None and perturbed_value is not None
                else None
            )
        explanation_rows.append(output)
    return explanation_rows


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
        for metric in METRICS:
            for prefix in ("clean", "perturbed", "delta"):
                column = f"{prefix}_{metric}"
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
    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    rows = collect_explanation_rows(load_rows(input_path))
    aggregate = aggregate_rows(rows)

    detail_csv = output_dir / "explainability_summary.csv"
    aggregate_csv = output_dir / "explainability_summary_by_experiment.csv"
    detail_json = output_dir / "explainability_summary.json"
    aggregate_json = output_dir / "explainability_summary_by_experiment.json"

    write_csv(detail_csv, rows)
    write_csv(aggregate_csv, aggregate)
    detail_json.write_text(json.dumps(rows, indent=2))
    aggregate_json.write_text(json.dumps(aggregate, indent=2))
    print(f"saved explainability summary: {detail_csv}")
    print(f"saved explainability summary: {aggregate_csv}")


if __name__ == "__main__":
    main()
