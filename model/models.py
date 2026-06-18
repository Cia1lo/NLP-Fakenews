from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    modality_weights: torch.Tensor | None = None


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureFusion(nn.Module):
    """Project heterogeneous node features into one hidden space with soft gates."""

    def __init__(
        self,
        feature_slices: dict[str, tuple[int, int]],
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.feature_names = tuple(feature_slices)
        self.feature_slices = feature_slices
        self.encoders = nn.ModuleDict()
        for name, (start, end) in feature_slices.items():
            self.encoders[name] = MLP(end - start, hidden_dim, hidden_dim, dropout)
        self.gate = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = []
        for name in self.feature_names:
            start, end = self.feature_slices[name]
            encoded.append(self.encoders[name](x[:, start:end]))

        stacked = torch.stack(encoded, dim=1)
        weights = torch.softmax(self.gate(stacked).squeeze(-1), dim=1)
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        return fused, weights


class MeanGraphConv(nn.Module):
    """GraphSAGE-style mean aggregation implemented with native PyTorch ops."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            out = self.self_proj(x)
        else:
            src, dst = edge_index
            agg = torch.zeros_like(x)
            agg.index_add_(0, dst, x[src])
            degree = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            degree.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
            agg = agg / degree.clamp_min(1).unsqueeze(-1)
            out = self.self_proj(x) + self.neighbor_proj(agg)
        return self.norm(x + self.dropout(self.act(out)))


def global_mean_max_pool(
    x: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    mean = torch.zeros(num_graphs, x.size(1), device=x.device, dtype=x.dtype)
    mean.index_add_(0, batch, x)
    counts = graph_counts(batch, num_graphs, x.dtype).clamp_min(1)
    mean = mean / counts.unsqueeze(-1)

    max_pool = torch.stack(
        [x[batch == graph_id].max(dim=0).values for graph_id in range(num_graphs)],
        dim=0,
    )
    return torch.cat([mean, max_pool], dim=1)


def global_mean_pool(
    x: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    mean = torch.zeros(num_graphs, x.size(1), device=x.device, dtype=x.dtype)
    mean.index_add_(0, batch, x)
    counts = graph_counts(batch, num_graphs, x.dtype).clamp_min(1)
    return mean / counts.unsqueeze(-1)


def global_max_pool(
    x: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    return torch.stack(
        [x[batch == graph_id].max(dim=0).values for graph_id in range(num_graphs)],
        dim=0,
    )


def graph_counts(batch: torch.Tensor, num_graphs: int, dtype: torch.dtype) -> torch.Tensor:
    counts = torch.zeros(num_graphs, device=batch.device, dtype=dtype)
    counts.index_add_(0, batch, torch.ones(batch.numel(), device=batch.device, dtype=dtype))
    return counts


def root_pool(x: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    counts = graph_counts(batch, num_graphs, x.dtype).to(torch.long)
    offsets = torch.cumsum(counts, dim=0) - counts
    return x[offsets]


def pooling_output_dim(hidden_dim: int, pooling: str) -> int:
    if pooling == "mean_max":
        return hidden_dim * 2
    if pooling in {"mean", "max", "root"}:
        return hidden_dim
    raise ValueError(f"Unsupported pooling mode: {pooling}")


def graph_pool(
    x: torch.Tensor,
    batch: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    num_graphs = int(batch.max().item()) + 1
    if pooling == "mean_max":
        return global_mean_max_pool(x, batch, num_graphs)
    if pooling == "mean":
        return global_mean_pool(x, batch, num_graphs)
    if pooling == "max":
        return global_max_pool(x, batch, num_graphs)
    if pooling == "root":
        return root_pool(x, batch, num_graphs)
    raise ValueError(f"Unsupported pooling mode: {pooling}")



class HeteroGraphFakeNewsModel(nn.Module):
    def __init__(
        self,
        feature_slices: dict[str, tuple[int, int]],
        hidden_dim: int = 128,
        graph_layers: int = 2,
        dropout: float = 0.3,
        pooling: str = "mean_max",
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        pooling_output_dim(hidden_dim, pooling)
        self.pooling = pooling
        self.fusion = FeatureFusion(feature_slices, hidden_dim, dropout)
        self.convs = nn.ModuleList(
            MeanGraphConv(hidden_dim, dropout) for _ in range(graph_layers)
        )
        self.classifier = nn.Sequential(
            nn.Linear(pooling_output_dim(hidden_dim, pooling), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> ModelOutput:
        x, modality_weights = self.fusion(x)
        for conv in self.convs:
            x = conv(x, edge_index)
        graph_repr = graph_pool(x, batch, self.pooling)
        logits = self.classifier(graph_repr)
        return ModelOutput(logits=logits, modality_weights=modality_weights)
