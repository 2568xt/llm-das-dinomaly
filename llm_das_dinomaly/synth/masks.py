from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


@dataclass(frozen=True)
class MaskConfig:
    area_ratio: Tuple[float, float] = (0.005, 0.12)
    components: Tuple[int, int] = (1, 4)
    min_side: int = 4
    max_aspect: float = 4.0


def sample_rectangular_mask(
    batch: int,
    height: int,
    width: int,
    config: Optional[MaskConfig] = None,
    *,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Sample one or more connected rectangular components per image."""

    cfg = config or MaskConfig()
    device = device or torch.device("cpu")
    mask = torch.zeros(batch, 1, height, width, device=device)
    min_components, max_components = cfg.components

    for b in range(batch):
        n_components = int(
            torch.randint(min_components, max_components + 1, (1,), generator=generator).item()
        )
        for _ in range(n_components):
            area_ratio = _uniform(cfg.area_ratio[0], cfg.area_ratio[1], device, generator)
            area = max(cfg.min_side * cfg.min_side, int(area_ratio * height * width / n_components))
            aspect = _uniform(1.0 / cfg.max_aspect, cfg.max_aspect, device, generator)
            box_h = int(max(cfg.min_side, min(height, round((area / aspect) ** 0.5))))
            box_w = int(max(cfg.min_side, min(width, round((area * aspect) ** 0.5))))
            top = int(torch.randint(0, max(1, height - box_h + 1), (1,), generator=generator).item())
            left = int(torch.randint(0, max(1, width - box_w + 1), (1,), generator=generator).item())
            mask[b, :, top : top + box_h, left : left + box_w] = 1.0
    return mask


def sample_patch_mask(
    batch: int,
    height: int,
    width: int,
    n_patch: Tuple[int, int] = (4, 64),
    *,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Sample sparse masks on a patch/token grid."""

    device = device or torch.device("cpu")
    total = height * width
    lo, hi = n_patch
    mask = torch.zeros(batch, 1, height, width, device=device)
    for b in range(batch):
        k = int(torch.randint(lo, min(hi, total) + 1, (1,), generator=generator).item())
        idx = torch.randperm(total, generator=generator, device=device)[:k]
        mask.view(batch, 1, total)[b, 0, idx] = 1.0
    return mask


def feather_mask(mask: Tensor, radius: int = 5) -> Tensor:
    if radius <= 1:
        return mask.float()
    if radius % 2 == 0:
        radius += 1
    pad = radius // 2
    soft = F.avg_pool2d(mask.float(), kernel_size=radius, stride=1, padding=pad)
    if soft.max() > 0:
        soft = soft / soft.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
    return soft.clamp(0.0, 1.0)


def _uniform(
    low: float,
    high: float,
    device: torch.device,
    generator: Optional[torch.Generator],
) -> float:
    return float(torch.empty((), device=device).uniform_(low, high, generator=generator).item())
