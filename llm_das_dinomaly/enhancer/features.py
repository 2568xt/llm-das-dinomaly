from __future__ import annotations

from typing import Iterable, Optional

import torch

Tensor = torch.Tensor


def map_statistics(anomaly_map: Tensor, *, eps: float = 1e-6) -> Tensor:
    """Return compact per-image anomaly-map statistics."""

    if anomaly_map.ndim == 3:
        anomaly_map = anomaly_map.unsqueeze(1)
    if anomaly_map.ndim != 4:
        raise ValueError("anomaly_map must have shape [B,1,H,W] or [B,H,W]")

    b, _, h, w = anomaly_map.shape
    flat = anomaly_map.flatten(1)
    norm = _minmax_by_sample(anomaly_map, eps=eps)
    norm_flat = norm.flatten(1)
    k1 = max(1, int(flat.shape[1] * 0.01))
    k5 = max(1, int(flat.shape[1] * 0.05))

    prob = norm_flat / norm_flat.sum(dim=1, keepdim=True).clamp_min(eps)
    entropy = -(prob * (prob + eps).log()).sum(dim=1) / torch.log(
        torch.tensor(float(flat.shape[1]), device=flat.device, dtype=flat.dtype)
    )
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=flat.device, dtype=flat.dtype),
        torch.linspace(-1.0, 1.0, w, device=flat.device, dtype=flat.dtype),
        indexing="ij",
    )
    mass = norm.flatten(2).sum(dim=2).clamp_min(eps)
    cx = (norm[:, 0] * xx).flatten(1).sum(dim=1) / mass[:, 0]
    cy = (norm[:, 0] * yy).flatten(1).sum(dim=1) / mass[:, 0]
    active_area = (norm_flat > 0.5).float().mean(dim=1)

    return torch.stack(
        [
            flat.mean(dim=1),
            flat.std(dim=1, unbiased=False),
            flat.max(dim=1).values,
            torch.topk(flat, k=k1, dim=1).values.mean(dim=1),
            torch.topk(flat, k=k5, dim=1).values.mean(dim=1),
            active_area,
            entropy,
            cx,
            cy,
        ],
        dim=1,
    )


def pooled_feature_statistics(features: Optional[Iterable[Tensor]]) -> Tensor:
    """Pool each fused feature group into scalar distribution summaries."""

    if features is None:
        raise ValueError("features cannot be None")

    stats = []
    for feat in features:
        if feat.ndim == 3:
            values = feat
        elif feat.ndim == 4:
            values = feat.flatten(1)
        else:
            raise ValueError("features must be [B,N,C] or [B,C,H,W]")
        stats.extend(
            [
                values.mean(dim=1),
                values.std(dim=1, unbiased=False),
                values.amax(dim=1),
                values.amin(dim=1),
            ]
        )
    if not stats:
        raise ValueError("at least one feature group is required")
    pooled = [s.mean(dim=1, keepdim=True) if s.ndim == 2 else s.unsqueeze(1) for s in stats]
    return torch.cat(pooled, dim=1)


def build_enhancer_features(
    base_score: Tensor,
    anomaly_map: Tensor,
    *,
    encoder_groups: Optional[Iterable[Tensor]] = None,
    extra: Optional[Tensor] = None,
) -> Tensor:
    parts = [base_score.reshape(-1, 1), map_statistics(anomaly_map)]
    if encoder_groups is not None:
        parts.append(pooled_feature_statistics(encoder_groups))
    if extra is not None:
        if extra.ndim == 1:
            extra = extra.unsqueeze(1)
        parts.append(extra)
    return torch.cat(parts, dim=1)


def _minmax_by_sample(x: Tensor, *, eps: float = 1e-6) -> Tensor:
    flat = x.flatten(1)
    lo = flat.min(dim=1).values.view(-1, 1, 1, 1)
    hi = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return (x - lo) / (hi - lo + eps)
