from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Experiment:
    name: str
    features: tuple[str, ...] = ()
    graph_layers: int = 2
    pooling: str = "mean_max"
    reverse_edges: bool = True
    trainer: str = "graph"


CORE_EXPERIMENTS = (
    Experiment("bert_profile_graph", ("bert", "profile")),
    Experiment("bert_only_graph", ("bert",)),
    Experiment("profile_only_graph", ("profile",)),
    Experiment("bert_profile_no_graph", ("bert", "profile"), graph_layers=0),
)

POOLING_CONTROL_EXPERIMENTS = (
    Experiment("bert_profile_graph_mean_pool", ("bert", "profile"), pooling="mean"),
    Experiment("bert_profile_graph_root_pool", ("bert", "profile"), pooling="root"),
    Experiment(
        "bert_profile_no_graph_mean_pool",
        ("bert", "profile"),
        graph_layers=0,
        pooling="mean",
    ),
    Experiment(
        "bert_profile_no_graph_root_pool",
        ("bert", "profile"),
        graph_layers=0,
        pooling="root",
    ),
)

POOLING_SPLIT_EXPERIMENTS = (
    Experiment("bert_only_graph_mean_pool", ("bert",), pooling="mean"),
    Experiment("bert_only_graph_root_pool", ("bert",), pooling="root"),
    Experiment("bert_only_no_graph_mean_pool", ("bert",), graph_layers=0, pooling="mean"),
    Experiment("bert_only_no_graph_root_pool", ("bert",), graph_layers=0, pooling="root"),
    Experiment("profile_only_graph_mean_pool", ("profile",), pooling="mean"),
    Experiment("profile_only_graph_root_pool", ("profile",), pooling="root"),
    Experiment("profile_only_no_graph_mean_pool", ("profile",), graph_layers=0, pooling="mean"),
    Experiment("profile_only_no_graph_root_pool", ("profile",), graph_layers=0, pooling="root"),
)

SIZE_CONTROL_EXPERIMENTS = (
    Experiment(
        "graph_size_control",
        graph_layers=0,
        pooling="size_only",
        trainer="size_control",
    ),
)

SIZE_ONLY_EXPERIMENTS = (
    Experiment(
        "size_only",
        graph_layers=0,
        pooling="size_only",
        trainer="size_control",
    ),
)

CONTROL_EXPERIMENTS = SIZE_CONTROL_EXPERIMENTS + POOLING_CONTROL_EXPERIMENTS
CORE_CONTROL_EXPERIMENTS = CORE_EXPERIMENTS + CONTROL_EXPERIMENTS
POOLING_SPLIT_CONTROL_EXPERIMENTS = POOLING_SPLIT_EXPERIMENTS + SIZE_ONLY_EXPERIMENTS

FULL_EXPERIMENTS = CORE_CONTROL_EXPERIMENTS + (
    Experiment("content_only_graph", ("content",)),
    Experiment("spacy_only_graph", ("spacy",)),
    Experiment("bert_content_profile_graph", ("bert", "content", "profile")),
    Experiment("all_features_graph", ("bert", "content", "spacy", "profile")),
)

METRIC_COLUMNS = (
    "best_epoch",
    "best_val_macro_f1",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_macro_f1",
    "test_auc",
)

SUMMARY_COLUMNS = (
    "dataset",
    "experiment",
    "seed",
    "trainer",
    *METRIC_COLUMNS,
    "features",
    "graph_layers",
    "pooling",
    "reverse_edges",
)

GROUP_COLUMNS = (
    "dataset",
    "experiment",
    "trainer",
    "features",
    "graph_layers",
    "pooling",
    "reverse_edges",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fake news detection ablation suite.")
    parser.add_argument("--data-root", default="database/data")
    parser.add_argument("--datasets", nargs="+", default=["politifact", "gossipcop"])
    parser.add_argument(
        "--suite",
        choices=[
            "core",
            "controls",
            "core_controls",
            "pooling_splits",
            "pooling_splits_controls",
            "full",
        ],
        default="core",
    )
    parser.add_argument("--output-dir", default="outputs/ablations")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--size-max-iter", type=int, default=1000)
    return parser.parse_args()


def experiments_for_suite(suite: str) -> tuple[Experiment, ...]:
    if suite == "core":
        return CORE_EXPERIMENTS
    if suite == "controls":
        return CONTROL_EXPERIMENTS
    if suite == "core_controls":
        return CORE_CONTROL_EXPERIMENTS
    if suite == "pooling_splits":
        return POOLING_SPLIT_EXPERIMENTS
    if suite == "pooling_splits_controls":
        return POOLING_SPLIT_CONTROL_EXPERIMENTS
    if suite == "full":
        return FULL_EXPERIMENTS
    raise ValueError(f"Unsupported suite: {suite}")


def seeds_from_args(args: argparse.Namespace) -> tuple[int, ...]:
    if args.seeds:
        return tuple(args.seeds)
    return (args.seed,)


def run_dir_for(
    output_dir: Path,
    experiment: Experiment,
    seed: int,
    multi_seed: bool,
) -> Path:
    run_dir = output_dir / experiment.name
    if multi_seed:
        run_dir = run_dir / f"seed_{seed}"
    return run_dir


def append_optional_int(cmd: list[str], flag: str, value: int | None) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def run_graph_experiment(
    args: argparse.Namespace,
    dataset: str,
    experiment: Experiment,
    seed: int,
    run_dir: Path,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "model.train",
        "--data-root",
        args.data_root,
        "--dataset",
        dataset,
        "--features",
        *experiment.features,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--graph-layers",
        str(experiment.graph_layers),
        "--pooling",
        experiment.pooling,
        "--dropout",
        str(args.dropout),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--patience",
        str(args.patience),
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
        "--class-weight",
        args.class_weight,
        "--output-dir",
        str(run_dir),
    ]
    append_optional_int(cmd, "--max-train-batches", args.max_train_batches)
    append_optional_int(cmd, "--max-eval-batches", args.max_eval_batches)
    if not experiment.reverse_edges:
        cmd.append("--no-reverse-edges")
    subprocess.run(cmd, check=True)


def run_size_control(
    args: argparse.Namespace,
    dataset: str,
    seed: int,
    run_dir: Path,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "model.train_size_control",
        "--data-root",
        args.data_root,
        "--dataset",
        dataset,
        "--seed",
        str(seed),
        "--class-weight",
        args.class_weight,
        "--output-dir",
        str(run_dir),
        "--max-iter",
        str(args.size_max_iter),
    ]
    subprocess.run(cmd, check=True)


def run_experiment(
    args: argparse.Namespace,
    dataset: str,
    experiment: Experiment,
    seed: int,
    multi_seed: bool,
) -> Path:
    run_dir = run_dir_for(Path(args.output_dir), experiment, seed, multi_seed)
    metrics_path = run_dir / dataset / "metrics.json"
    if args.skip_existing and metrics_path.exists():
        print(f"skip existing: {metrics_path}")
        return metrics_path

    print(f"run: dataset={dataset} experiment={experiment.name} seed={seed}", flush=True)
    if experiment.trainer == "graph":
        run_graph_experiment(args, dataset, experiment, seed, run_dir)
    elif experiment.trainer == "size_control":
        run_size_control(args, dataset, seed, run_dir)
    else:
        raise ValueError(f"Unsupported trainer: {experiment.trainer}")
    return metrics_path


def infer_path_metadata(output_dir: Path, metrics_path: Path) -> tuple[str, int | None, str]:
    rel = metrics_path.relative_to(output_dir)
    parts = rel.parts
    if len(parts) == 3:
        experiment, dataset, _ = parts
        return experiment, None, dataset
    if len(parts) == 4 and parts[1].startswith("seed_"):
        experiment, seed_part, dataset, _ = parts
        return experiment, int(seed_part.removeprefix("seed_")), dataset
    raise ValueError(f"Unexpected metrics path layout: {metrics_path}")


def collect_results(
    output_dir: Path,
    metrics_paths: list[Path] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    paths = metrics_paths if metrics_paths is not None else sorted(output_dir.rglob("metrics.json"))
    for metrics_path in paths:
        with metrics_path.open() as handle:
            payload = json.load(handle)
        args_payload = payload.get("args", {})
        test = payload["test"]
        run_name, seed_from_path, dataset = infer_path_metadata(output_dir, metrics_path)
        seed = args_payload.get("seed", seed_from_path)
        features = args_payload.get("features", ())
        if isinstance(features, str):
            feature_text = features
        else:
            feature_text = " ".join(features)
        trainer = args_payload.get("trainer", "graph")
        reverse_edges = (
            None
            if trainer == "size_control"
            else not bool(args_payload.get("no_reverse_edges", False))
        )
        rows.append(
            {
                "dataset": dataset,
                "experiment": run_name,
                "seed": seed,
                "trainer": trainer,
                "best_epoch": payload["best_epoch"],
                "best_val_macro_f1": payload["best_val_macro_f1"],
                "test_accuracy": test.get("accuracy"),
                "test_precision": test.get("precision"),
                "test_recall": test.get("recall"),
                "test_f1": test.get("f1"),
                "test_macro_f1": test.get("macro_f1"),
                "test_auc": test.get("auc"),
                "features": feature_text,
                "graph_layers": args_payload.get("graph_layers"),
                "pooling": args_payload.get("pooling", "mean_max"),
                "reverse_edges": reverse_edges,
            }
        )
    return rows


def aggregate_results(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[column] for column in GROUP_COLUMNS)
        groups.setdefault(key, []).append(row)

    aggregated: list[dict[str, object]] = []
    for key, group in sorted(groups.items()):
        row: dict[str, object] = dict(zip(GROUP_COLUMNS, key, strict=True))
        seeds = sorted(str(item["seed"]) for item in group)
        row["n_seeds"] = len(group)
        row["seeds"] = " ".join(seeds)
        for metric in METRIC_COLUMNS:
            values = [float(item[metric]) for item in group if item.get(metric) is not None]
            if values:
                row[f"{metric}_mean"] = statistics.mean(values)
                row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
        aggregated.append(row)
    return aggregated


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: tuple[str, ...] | None = None,
) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(output_dir: Path, rows: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    aggregate_json = output_dir / "summary_by_experiment.json"
    aggregate_csv = output_dir / "summary_by_experiment.csv"

    aggregate_rows = aggregate_results(rows)
    summary_json.write_text(json.dumps(rows, indent=2))
    aggregate_json.write_text(json.dumps(aggregate_rows, indent=2))
    write_csv(summary_csv, rows, SUMMARY_COLUMNS)
    write_csv(aggregate_csv, aggregate_rows)

    print(f"saved summary: {summary_json}")
    print(f"saved summary: {summary_csv}")
    print(f"saved aggregate summary: {aggregate_json}")
    print(f"saved aggregate summary: {aggregate_csv}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    seeds = seeds_from_args(args)
    multi_seed = len(seeds) > 1
    metrics_paths: list[Path] = []
    for dataset in args.datasets:
        for experiment in experiments_for_suite(args.suite):
            for seed in seeds:
                metrics_paths.append(run_experiment(args, dataset, experiment, seed, multi_seed))
    write_summary(output_dir, collect_results(output_dir, metrics_paths))


if __name__ == "__main__":
    main()
