from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch

Tensor = torch.Tensor
NumberOrTensor = Union[float, Tensor]


@dataclass(frozen=True)
class NormalScoreStats:
    mean: float
    std: float
    eps: float = 1e-6

    @classmethod
    def from_scores(cls, scores: Tensor) -> "NormalScoreStats":
        scores = scores.detach().float()
        return cls(float(scores.mean().item()), float(scores.std(unbiased=False).clamp_min(1e-6).item()))

    def z_score(self, scores: Tensor) -> Tensor:
        return (scores - self.mean) / (self.std + self.eps)


@dataclass(frozen=True)
class HardnessConfig:
    z_target: float = 2.0
    tau_z: float = 0.75
    overlap_target: float = 0.4
    tau_overlap: float = 0.15
    tau_perturb: float = 0.03
    tau_stability: float = 0.2
    keep_z: Tuple[float, float] = (1.0, 3.0)
    reject_easy_z: float = 4.0
    min_mask_area: float = 0.005
    max_mask_area: float = 0.15
    min_overlap: float = 0.15
    max_stability_sigma: float = 0.25


def hardness_score(
    z: NumberOrTensor,
    *,
    overlap: Optional[NumberOrTensor] = None,
    perturb: Optional[NumberOrTensor] = None,
    stability: Optional[NumberOrTensor] = None,
    config: Optional[HardnessConfig] = None,
) -> Tensor:
    """Compute the report's near-boundary hard-sample score."""

    cfg = config or HardnessConfig()
    z_t = _as_tensor(z)
    overlap_t = _as_tensor(cfg.overlap_target if overlap is None else overlap, like=z_t)
    perturb_t = _as_tensor(0.0 if perturb is None else perturb, like=z_t)
    stability_t = _as_tensor(0.0 if stability is None else stability, like=z_t)

    z_term = torch.exp(-torch.abs(z_t - cfg.z_target) / cfg.tau_z)
    overlap_term = torch.exp(-torch.abs(overlap_t - cfg.overlap_target) / cfg.tau_overlap)
    perturb_term = torch.exp(-perturb_t.clamp_min(0.0) / cfg.tau_perturb)
    stability_term = torch.exp(-stability_t.clamp_min(0.0) / cfg.tau_stability)
    return z_term * overlap_term * perturb_term * stability_term


def accept_hard(
    meta: Dict[str, Tensor],
    stats: NormalScoreStats,
    *,
    config: Optional[HardnessConfig] = None,
) -> Tensor:
    """Return a boolean mask for candidates that pass hard-sample gates."""

    cfg = config or HardnessConfig()
    z = stats.z_score(meta["score_cand"])
    keep = (z >= cfg.keep_z[0]) & (z <= cfg.keep_z[1]) & (z < cfg.reject_easy_z)

    if "mask_area" in meta:
        keep = keep & (meta["mask_area"] >= cfg.min_mask_area) & (meta["mask_area"] <= cfg.max_mask_area)
    if "mask_overlap" in meta:
        keep = keep & (meta["mask_overlap"] >= cfg.min_overlap)
    if "stability" in meta:
        keep = keep & (meta["stability"] <= cfg.max_stability_sigma * stats.std)
    return keep


def attach_hardness(
    meta: Dict[str, Tensor],
    stats: NormalScoreStats,
    *,
    config: Optional[HardnessConfig] = None,
) -> Dict[str, Tensor]:
    cfg = config or HardnessConfig()
    z = stats.z_score(meta["score_cand"])
    meta = dict(meta)
    meta["z"] = z
    meta["hardness"] = hardness_score(
        z,
        overlap=meta.get("mask_overlap"),
        perturb=meta.get("perturb_l1"),
        stability=meta.get("stability"),
        config=cfg,
    )
    meta["accepted"] = accept_hard(meta, stats, config=cfg)
    return meta


def _as_tensor(value: NumberOrTensor, like: Optional[Tensor] = None) -> Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if like is None:
        return torch.tensor(value, dtype=torch.float32)
    return torch.full_like(like, float(value))
