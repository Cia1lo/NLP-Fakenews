from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model.data import FakeNewsGraphDataset, GraphBatch, RawGraphStore, graph_collate
from model.device import select_device, torch
from model.metrics import classification_metrics
from model.models import HeteroGraphFakeNewsModel

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
    "perturbation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate robustness of saved graph checkpoints under test-time perturbations."
    )
    parser.add_argument("--input-dir", default="outputs/ablations")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--edge-drop-rates", nargs="+", type=float, default=[0.25, 0.5])
    return parser.parse_args()


def infer_path_metadata(input_dir: Path, checkpoint_path: Path) -> tuple[str, int | None, str]:
    rel = checkpoint_path.relative_to(input_dir)
    parts = rel.parts
    if len(parts) == 3:
        experiment, dataset, _ = parts
        return experiment, None, dataset
    if len(parts) == 4 and parts[1].startswith("seed_"):
        experiment, seed_part, dataset, _ = parts
        return experiment, int(seed_part.removeprefix("seed_")), dataset
    raise ValueError(f"Unexpected checkpoint path layout: {checkpoint_path}")


def load_metrics_args(checkpoint_path: Path) -> dict[str, Any]:
    metrics_path = checkpoint_path.with_name("metrics.json")
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


def checkpoint_paths(
    input_dir: Path,
    datasets: set[str] | None,
    experiments: set[str] | None,
) -> list[Path]:
    paths: list[Path] = []
    for checkpoint_path in sorted(input_dir.rglob("best.pt")):
        experiment, _, dataset = infer_path_metadata(input_dir, checkpoint_path)
        if datasets is not None and dataset not in datasets:
            continue
        if experiments is not None and experiment not in experiments:
            continue
        paths.append(checkpoint_path)
    return paths


def make_loader(
    data_root: str,
    dataset: str,
    split: str,
    features: list[str],
    batch_size: int,
    num_workers: int,
    add_reverse_edges: bool,
) -> DataLoader:
    store = RawGraphStore(data_root, dataset, features)
    ds = FakeNewsGraphDataset(
        data_root=store.data_root,
        dataset=store.dataset,
        split=split,
        features=store.feature_names,
        add_reverse_edges=add_reverse_edges,
        store=store,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=graph_collate,
        pin_memory=torch.cuda.is_available(),
    )


def perturbations(
    features: list[str],
    graph_layers: int,
    edge_drop_rates: list[float],
) -> list[str]:
    names = ["clean"]
    if graph_layers > 0:
        names.extend(f"edge_drop_{rate:g}" for rate in edge_drop_rates)
    for feature in features:
        names.append(f"zero_{feature}")
        names.append(f"noise_{feature}")
    return names


def apply_perturbation(
    batch: GraphBatch,
    perturbation: str,
    feature_slices: dict[str, tuple[int, int]],
    noise_std: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = batch.x
    edge_index = batch.edge_index
    if perturbation == "clean":
        return x, edge_index

    if perturbation.startswith("edge_drop_"):
        rate = float(perturbation.removeprefix("edge_drop_"))
        if edge_index.numel() == 0:
            return x, edge_index
        keep = torch.rand(edge_index.size(1), generator=generator) >= rate
        return x, edge_index[:, keep]

    if perturbation.startswith("zero_"):
        feature = perturbation.removeprefix("zero_")
        start, end = feature_slices[feature]
        perturbed = x.clone()
        perturbed[:, start:end] = 0
        return perturbed, edge_index

    if perturbation.startswith("noise_"):
        feature = perturbation.removeprefix("noise_")
        start, end = feature_slices[feature]
        perturbed = x.clone()
        block = perturbed[:, start:end]
        scale = block.std(dim=0, keepdim=True).clamp_min(1e-6)
        noise = torch.randn(block.shape, generator=generator, dtype=block.dtype) * scale * noise_std
        perturbed[:, start:end] = block + noise
        return perturbed, edge_index

    raise ValueError(f"Unsupported perturbation: {perturbation}")


@torch.no_grad()
def evaluate_perturbation(
    model: HeteroGraphFakeNewsModel,
    loader: DataLoader,
    device: torch.device,
    perturbation: str,
    feature_slices: dict[str, tuple[int, int]],
    noise_std: float,
    seed: int,
) -> dict[str, float]:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    y_prob_fake: list[np.ndarray] = []

    for batch in tqdm(loader, desc=perturbation, leave=False):
        x, edge_index = apply_perturbation(
            batch,
            perturbation,
            feature_slices,
            noise_std,
            generator,
        )
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch_ids = batch.batch.to(device)
        labels = batch.y.to(device)
        output = model(x, edge_index, batch_ids)
        prob = torch.softmax(output.logits, dim=1)
        y_true.append(labels.cpu().numpy())
        y_pred.append(prob.argmax(dim=1).cpu().numpy())
        y_prob_fake.append(prob[:, 1].cpu().numpy())

    return classification_metrics(
        np.concatenate(y_true),
        np.concatenate(y_pred),
        np.concatenate(y_prob_fake),
    )


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[HeteroGraphFakeNewsModel, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint["args"]
    model = HeteroGraphFakeNewsModel(
        feature_slices=checkpoint["feature_slices"],
        hidden_dim=int(args["hidden_dim"]),
        graph_layers=int(args["graph_layers"]),
        dropout=float(args["dropout"]),
        pooling=str(args["pooling"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def run_checkpoint(
    input_dir: Path,
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    experiment, seed_from_path, dataset = infer_path_metadata(input_dir, checkpoint_path)
    metadata = load_metrics_args(checkpoint_path)
    model, checkpoint = load_model(checkpoint_path, device)
    train_args = checkpoint["args"]
    features = list(train_args["features"])
    seed = int(metadata.get("seed", seed_from_path if seed_from_path is not None else args.seed))
    add_reverse_edges = not bool(train_args.get("no_reverse_edges", False))
    loader = make_loader(
        data_root=str(train_args.get("data_root", "database/data")),
        dataset=dataset,
        split=args.split,
        features=features,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        add_reverse_edges=add_reverse_edges,
    )

    rows: list[dict[str, Any]] = []
    perturbation_names = perturbations(
        features,
        int(train_args["graph_layers"]),
        args.edge_drop_rates,
    )
    for perturbation in perturbation_names:
        metrics = evaluate_perturbation(
            model,
            loader,
            device,
            perturbation,
            checkpoint["feature_slices"],
            args.noise_std,
            seed=args.seed + seed,
        )
        rows.append(
            {
                "dataset": dataset,
                "experiment": experiment,
                "seed": seed,
                "trainer": "graph",
                "features": feature_text(features),
                "graph_layers": train_args.get("graph_layers"),
                "pooling": train_args.get("pooling"),
                "perturbation": perturbation,
                **{metric: metrics.get(metric) for metric in METRIC_COLUMNS},
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
    device = select_device(args.device)
    datasets = set(args.datasets) if args.datasets else None
    experiments = set(args.experiments) if args.experiments else None

    rows: list[dict[str, Any]] = []
    for checkpoint_path in checkpoint_paths(input_dir, datasets, experiments):
        print(f"evaluate robustness: {checkpoint_path}")
        rows.extend(run_checkpoint(input_dir, checkpoint_path, args, device))

    aggregate = aggregate_rows(rows)
    split_prefix = f"{args.split}_robustness"
    detail_csv = output_dir / f"{split_prefix}_summary.csv"
    aggregate_csv = output_dir / f"{split_prefix}_summary_by_experiment.csv"
    detail_json = output_dir / f"{split_prefix}_summary.json"
    aggregate_json = output_dir / f"{split_prefix}_summary_by_experiment.json"

    write_csv(detail_csv, rows)
    write_csv(aggregate_csv, aggregate)
    detail_json.write_text(json.dumps(rows, indent=2))
    aggregate_json.write_text(json.dumps(aggregate, indent=2))
    print(f"saved robustness summary: {detail_csv}")
    print(f"saved robustness summary: {aggregate_csv}")


if __name__ == "__main__":
    main()
