from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn

Tensor = torch.Tensor


class MapFeatureHead(nn.Module):
    """Small MLP for the faithful-transplantation auxiliary score."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


def binary_enhancer_loss(logits: Tensor, labels: Tensor) -> Tensor:
    labels = labels.to(device=logits.device, dtype=logits.dtype)
    return nn.functional.binary_cross_entropy_with_logits(logits, labels)
