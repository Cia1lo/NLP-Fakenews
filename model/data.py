from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset

FEATURE_FILES = {
    "bert": "new_bert_feature.npz",
    "content": "new_content_feature.npz",
    "spacy": "new_spacy_feature.npz",
    "profile": "new_profile_feature.npz",
}


@dataclass(frozen=True)
class GraphSample:
    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    graph_id: int


class RawGraphStore:
    """Shared raw graph arrays and sparse feature matrices for one dataset."""

    def __init__(
        self,
        data_root: str | Path,
        dataset: str,
        features: Iterable[str] = ("bert", "profile"),
    ) -> None:
        self.data_root = Path(data_root)
        self.dataset = dataset
        self.feature_names = normalize_features(features, allow_empty=True)
        self.raw_dir = self.data_root / dataset / "raw"
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw dataset directory not found: {self.raw_dir}")

        self.labels = np.load(self.raw_dir / "graph_labels.npy").astype(np.int64)
        self.node_graph_id = np.load(self.raw_dir / "node_graph_id.npy").astype(np.int64)
        self.graph_starts, self.graph_ends = self._build_graph_slices(self.node_graph_id)
        self.num_graphs = len(self.labels)
        if len(self.graph_starts) != self.num_graphs:
            raise ValueError(
                f"Graph count mismatch: labels={self.num_graphs}, "
                f"node_graph_id groups={len(self.graph_starts)}"
            )

        self.features = self._load_features()
        self.feature_slices = self._build_feature_slices()
        self.edge_src, self.edge_dst, self.edge_ptr = self._load_edges(self.raw_dir / "A.txt")

    @property
    def input_dim(self) -> int:
        return sum(
            self.feature_slices[name][1] - self.feature_slices[name][0]
            for name in self.feature_names
        )

    def split_indices(self, split: str) -> np.ndarray:
        return np.load(self.raw_dir / f"custom_{split}_idx.npy").astype(np.int64)

    def dense_feature(self, name: str, start: int, end: int) -> np.ndarray:
        return np.asarray(self.features[name][start:end].toarray(), dtype=np.float32)

    def _load_features(self) -> dict[str, sparse.csr_matrix]:
        loaded: dict[str, sparse.csr_matrix] = {}
        expected_rows = len(self.node_graph_id)
        for name in self.feature_names:
            matrix = sparse.load_npz(self.raw_dir / FEATURE_FILES[name]).tocsr()
            if matrix.shape[0] != expected_rows:
                raise ValueError(
                    f"{FEATURE_FILES[name]} row count mismatch: "
                    f"{matrix.shape[0]} != {expected_rows}"
                )
            loaded[name] = matrix
        return loaded

    def _build_feature_slices(self) -> dict[str, tuple[int, int]]:
        slices: dict[str, tuple[int, int]] = {}
        offset = 0
        for name in self.feature_names:
            dim = int(self.features[name].shape[1])
            slices[name] = (offset, offset + dim)
            offset += dim
        return slices

    @staticmethod
    def _build_graph_slices(node_graph_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        breaks = np.flatnonzero(node_graph_id[1:] != node_graph_id[:-1]) + 1
        starts = np.concatenate(([0], breaks)).astype(np.int64)
        ends = np.concatenate((breaks, [len(node_graph_id)])).astype(np.int64)

        expected = np.arange(len(starts), dtype=np.int64)
        observed = node_graph_id[starts]
        if not np.array_equal(observed, expected):
            raise ValueError(
                "node_graph_id must be grouped by consecutive graph ids starting at 0."
            )
        return starts, ends

    def _load_edges(self, edge_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        edges = np.loadtxt(edge_path, delimiter=",", dtype=np.int64)
        if edges.ndim == 1:
            edges = edges.reshape(1, 2)

        src = edges[:, 0]
        dst = edges[:, 1]
        src_gid = self.node_graph_id[src]
        dst_gid = self.node_graph_id[dst]
        if not np.array_equal(src_gid, dst_gid):
            raise ValueError(
                "A.txt contains cross-graph edges, which this loader does not support."
            )

        order = np.argsort(src_gid, kind="stable")
        src = src[order]
        dst = dst[order]
        edge_gid = src_gid[order]
        counts = np.bincount(edge_gid, minlength=self.num_graphs).astype(np.int64)
        edge_ptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        return src, dst, edge_ptr


@dataclass
class GraphBatch:
    x: torch.Tensor
    edge_index: torch.Tensor
    batch: torch.Tensor
    y: torch.Tensor
    graph_id: torch.Tensor
    node_count: torch.Tensor

    def to(self, device: torch.device | str) -> GraphBatch:
        return GraphBatch(
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            batch=self.batch.to(device),
            y=self.y.to(device),
            graph_id=self.graph_id.to(device),
            node_count=self.node_count.to(device),
        )


def normalize_features(features: Iterable[str], allow_empty: bool = False) -> tuple[str, ...]:
    names = tuple(features)
    unknown = sorted(set(names) - set(FEATURE_FILES))
    if unknown:
        valid = ", ".join(sorted(FEATURE_FILES))
        raise ValueError(f"Unknown feature(s): {unknown}. Valid features: {valid}")
    if not names and not allow_empty:
        raise ValueError("At least one feature must be selected.")
    return names


def graph_collate(samples: list[GraphSample]) -> GraphBatch:
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    graph_ids: list[int] = []
    node_counts: list[int] = []
    edge_indices: list[torch.Tensor] = []
    batch_ids: list[torch.Tensor] = []

    offset = 0
    for batch_id, sample in enumerate(samples):
        num_nodes = sample.x.size(0)
        xs.append(sample.x)
        ys.append(sample.y)
        graph_ids.append(sample.graph_id)
        node_counts.append(num_nodes)
        batch_ids.append(torch.full((num_nodes,), batch_id, dtype=torch.long))

        if sample.edge_index.numel() > 0:
            edge_indices.append(sample.edge_index + offset)
        offset += num_nodes

    edge_index = (
        torch.cat(edge_indices, dim=1)
        if edge_indices
        else torch.empty((2, 0), dtype=torch.long)
    )

    return GraphBatch(
        x=torch.cat(xs, dim=0),
        edge_index=edge_index,
        batch=torch.cat(batch_ids, dim=0),
        y=torch.stack(ys).long(),
        graph_id=torch.tensor(graph_ids, dtype=torch.long),
        node_count=torch.tensor(node_counts, dtype=torch.long),
    )


class FakeNewsGraphDataset(Dataset[GraphSample]):
    """Read FakeNewsNet-style raw graph files using custom split indices."""

    def __init__(
        self,
        data_root: str | Path,
        dataset: str,
        split: str,
        features: Iterable[str] = ("bert", "profile"),
        add_reverse_edges: bool = True,
        store: RawGraphStore | None = None,
    ) -> None:
        self.store = store or RawGraphStore(data_root, dataset, features)
        self.data_root = self.store.data_root
        self.dataset = self.store.dataset
        self.split = split
        self.feature_names = self.store.feature_names
        self.add_reverse_edges = add_reverse_edges
        self.indices = self.store.split_indices(split)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> GraphSample:
        graph_id = int(self.indices[idx])
        start = int(self.store.graph_starts[graph_id])
        end = int(self.store.graph_ends[graph_id])

        parts = [self.store.dense_feature(name, start, end) for name in self.feature_names]
        x = torch.from_numpy(np.concatenate(parts, axis=1)).float()

        edge_start = int(self.store.edge_ptr[graph_id])
        edge_end = int(self.store.edge_ptr[graph_id + 1])
        src = self.store.edge_src[edge_start:edge_end] - start
        dst = self.store.edge_dst[edge_start:edge_end] - start
        edge_index = np.stack([src, dst], axis=0).astype(np.int64, copy=False)

        if self.add_reverse_edges and edge_index.size:
            reverse = edge_index[[1, 0], :]
            edge_index = np.concatenate([edge_index, reverse], axis=1)

        return GraphSample(
            x=x,
            edge_index=torch.from_numpy(edge_index).long(),
            y=torch.tensor(int(self.store.labels[graph_id]), dtype=torch.long),
            graph_id=graph_id,
        )

    @property
    def input_dim(self) -> int:
        return self.store.input_dim

    @property
    def feature_slices(self) -> dict[str, tuple[int, int]]:
        return self.store.feature_slices

    def split_labels(self) -> np.ndarray:
        return self.store.labels[self.indices]
