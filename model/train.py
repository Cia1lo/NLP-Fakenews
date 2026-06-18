from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model.data import FakeNewsGraphDataset, RawGraphStore, graph_collate, normalize_features
from model.device import device_report, select_device, torch
from model.metrics import classification_metrics
from model.models import HeteroGraphFakeNewsModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fake news graph classifier.")
    parser.add_argument(
        "--data-root",
        default="database/data",
        help="Path containing gossipcop/politifact.",
    )
    parser.add_argument("--dataset", choices=["gossipcop", "politifact"], default="politifact")
    parser.add_argument(
        "--features",
        nargs="+",
        default=["bert", "profile"],
        help="Feature groups to use.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument(
        "--pooling",
        choices=["mean_max", "mean", "max", "root"],
        default="mean_max",
        help="Graph-level readout. root uses the first node in each propagation graph.",
    )
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--no-reverse-edges", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(
    store: RawGraphStore,
    split: str,
    batch_size: int,
    num_workers: int,
    add_reverse_edges: bool,
    shuffle: bool,
) -> tuple[FakeNewsGraphDataset, DataLoader]:
    ds = FakeNewsGraphDataset(
        data_root=store.data_root,
        dataset=store.dataset,
        split=split,
        features=store.feature_names,
        add_reverse_edges=add_reverse_edges,
        store=store,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=graph_collate,
        pin_memory=torch.cuda.is_available(),
    )
    return ds, loader


def class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    weights = counts.sum() / (len(counts) * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_graphs = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(output.logits, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * batch.y.numel()
        total_graphs += batch.y.numel()
        if max_batches is not None and batch_idx >= max_batches:
            break

    return total_loss / max(total_graphs, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split: str,
    max_batches: int | None = None,
    prediction_path: Path | None = None,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    graph_ids: list[np.ndarray] = []
    node_counts: list[np.ndarray] = []
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    y_prob_fake: list[np.ndarray] = []

    for batch_idx, batch in enumerate(tqdm(loader, desc=split, leave=False), start=1):
        batch = batch.to(device)
        output = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(output.logits, batch.y)
        prob = torch.softmax(output.logits, dim=1)

        losses.append(float(loss.item()) * batch.y.numel())
        graph_ids.append(batch.graph_id.cpu().numpy())
        node_counts.append(batch.node_count.cpu().numpy())
        y_true.append(batch.y.cpu().numpy())
        y_pred.append(prob.argmax(dim=1).cpu().numpy())
        y_prob_fake.append(prob[:, 1].cpu().numpy())
        if max_batches is not None and batch_idx >= max_batches:
            break

    labels = np.concatenate(y_true)
    predictions = np.concatenate(y_pred)
    fake_prob = np.concatenate(y_prob_fake)
    metrics = classification_metrics(labels, predictions, fake_prob)
    metrics["loss"] = sum(losses) / max(len(labels), 1)
    if prediction_path is not None:
        write_predictions(
            prediction_path,
            graph_ids=np.concatenate(graph_ids),
            node_counts=np.concatenate(node_counts),
            labels=labels,
            predictions=predictions,
            fake_prob=fake_prob,
        )
    return metrics


def write_predictions(
    path: Path,
    graph_ids: np.ndarray,
    node_counts: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    fake_prob: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("graph_id", "node_count", "y_true", "y_pred", "prob_fake"),
        )
        writer.writeheader()
        for graph_id, node_count, label, prediction, prob in zip(
            graph_ids,
            node_counts,
            labels,
            predictions,
            fake_prob,
            strict=True,
        ):
            writer.writerow(
                {
                    "graph_id": int(graph_id),
                    "node_count": int(node_count),
                    "y_true": int(label),
                    "y_pred": int(prediction),
                    "prob_fake": float(prob),
                }
            )


def main() -> None:
    args = parse_args()
    features = normalize_features(args.features)
    set_seed(args.seed)
    device = select_device(args.device)
    add_reverse_edges = not args.no_reverse_edges
    store = RawGraphStore(args.data_root, args.dataset, features)

    train_ds, train_loader = make_loader(
        store,
        "train",
        args.batch_size,
        args.num_workers,
        add_reverse_edges,
        shuffle=True,
    )
    _, val_loader = make_loader(
        store,
        "val",
        args.batch_size,
        args.num_workers,
        add_reverse_edges,
        shuffle=False,
    )
    _, test_loader = make_loader(
        store,
        "test",
        args.batch_size,
        args.num_workers,
        add_reverse_edges,
        shuffle=False,
    )

    model = HeteroGraphFakeNewsModel(
        feature_slices=train_ds.feature_slices,
        hidden_dim=args.hidden_dim,
        graph_layers=args.graph_layers,
        dropout=args.dropout,
        pooling=args.pooling,
    ).to(device)

    weights = (
        class_weights(train_ds.split_labels(), device)
        if args.class_weight == "balanced"
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    best_val = -1.0
    best_epoch = 0
    stale_epochs = 0

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "features": features,
                "device": str(device),
                "train_graphs": len(train_ds),
                "input_dim": train_ds.input_dim,
                "feature_slices": train_ds.feature_slices,
                "graph_layers": args.graph_layers,
                "pooling": args.pooling,
                "device_report": device_report(),
            },
            indent=2,
        )
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_batches=args.max_train_batches,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            split="val",
            max_batches=args.max_eval_batches,
        )
        score = val_metrics["macro_f1"]
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_macro_f1={score:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if score > best_val:
            best_val = score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "feature_slices": train_ds.feature_slices,
                    "val_metrics": val_metrics,
                    "epoch": epoch,
                },
                best_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    final_val_metrics = evaluate(
        model,
        val_loader,
        criterion,
        device,
        split="val",
        max_batches=args.max_eval_batches,
        prediction_path=output_dir / "val_predictions.csv",
    )
    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        split="test",
        max_batches=args.max_eval_batches,
        prediction_path=output_dir / "test_predictions.csv",
    )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_val,
                "val": final_val_metrics,
                "test": test_metrics,
                "args": vars(args),
            },
            indent=2,
        )
    )
    print("test", json.dumps(test_metrics, indent=2))
    print(f"saved checkpoint: {best_path}")
    print(f"saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
