from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

Tensor = torch.Tensor


@dataclass
class ScoreNormalizer:
    lo: Optional[float] = None
    hi: Optional[float] = None
    eps: float = 1e-6

    def fit(self, scores: Tensor) -> "ScoreNormalizer":
        values = scores.detach().float()
        self.lo = float(values.min().item())
        self.hi = float(values.max().item())
        return self

    def transform(self, scores: Tensor) -> Tensor:
        if self.lo is None or self.hi is None:
            raise RuntimeError("ScoreNormalizer must be fit before transform")
        scale = self.hi - self.lo
        if abs(scale) < self.eps:
            scale = self.eps
        return (scores - self.lo) / scale

    def fit_transform(self, scores: Tensor) -> Tensor:
        return self.fit(scores).transform(scores)


def normalizer_from_metadata(metadata: dict) -> ScoreNormalizer:
    return ScoreNormalizer(lo=float(metadata["lo"]), hi=float(metadata["hi"]))


def fuse_scores(
    base_scores: Tensor,
    aux_scores: Tensor,
    *,
    beta: float = 1.0,
    base_normalizer: Optional[ScoreNormalizer] = None,
    aux_normalizer: Optional[ScoreNormalizer] = None,
) -> Tensor:
    base_norm = (base_normalizer or ScoreNormalizer().fit(base_scores)).transform(base_scores)
    aux_norm = (aux_normalizer or ScoreNormalizer().fit(aux_scores)).transform(aux_scores)
    return base_norm + beta * aux_norm
