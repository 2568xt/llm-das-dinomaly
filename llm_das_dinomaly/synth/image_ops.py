from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from llm_das_dinomaly.synth.masks import MaskConfig, feather_mask, sample_rectangular_mask

Tensor = torch.Tensor


@dataclass(frozen=True)
class ImageMutationConfig:
    max_shift: int = 12
    blur_kernel: int = 7
    noise_sigma: float = 0.08
    color_strength: float = 0.25
    feather_radius: int = 7
    mask: MaskConfig = MaskConfig()


def copy_shift(x: Tensor, mask: Tensor, *, max_shift: int = 12, generator: Optional[torch.Generator] = None) -> Tensor:
    shifted = torch.empty_like(x)
    for b in range(x.shape[0]):
        dy = int(torch.randint(-max_shift, max_shift + 1, (1,), generator=generator).item())
        dx = int(torch.randint(-max_shift, max_shift + 1, (1,), generator=generator).item())
        shifted[b] = torch.roll(x[b], shifts=(dy, dx), dims=(-2, -1))
    return torch.where(mask.bool(), shifted, x)


def local_blur(x: Tensor, mask: Tensor, *, kernel_size: int = 7) -> Tensor:
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    blurred = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
    return torch.where(mask.bool(), blurred, x)


def local_noise(
    x: Tensor,
    mask: Tensor,
    *,
    sigma: float = 0.08,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator) * sigma
    return x + noise * mask


def local_color_jitter(
    x: Tensor,
    mask: Tensor,
    *,
    strength: float = 0.25,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    batch = x.shape[0]
    gain = torch.empty(batch, x.shape[1], 1, 1, device=x.device, dtype=x.dtype)
    bias = torch.empty_like(gain)
    gain.uniform_(1.0 - strength, 1.0 + strength, generator=generator)
    bias.uniform_(-strength, strength, generator=generator)
    jittered = x * gain + bias
    return torch.where(mask.bool(), jittered, x)


def feather_blend(x_ref: Tensor, x_mut: Tensor, mask: Tensor, *, radius: int = 7) -> Tensor:
    alpha = feather_mask(mask, radius=radius).to(device=x_ref.device, dtype=x_ref.dtype)
    return x_ref * (1.0 - alpha) + x_mut * alpha


def clamp_like_reference(x: Tensor, x_ref: Tensor) -> Tensor:
    lo = x_ref.flatten(1).min(dim=1).values.view(-1, 1, 1, 1)
    hi = x_ref.flatten(1).max(dim=1).values.view(-1, 1, 1, 1)
    return torch.max(torch.min(x, hi), lo)


def apply_local_op(
    x: Tensor,
    mask: Tensor,
    op: str,
    config: Optional[ImageMutationConfig] = None,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    cfg = config or ImageMutationConfig()
    if op == "copy_shift":
        return copy_shift(x, mask, max_shift=cfg.max_shift, generator=generator)
    if op == "local_blur":
        return local_blur(x, mask, kernel_size=cfg.blur_kernel)
    if op == "local_noise":
        return local_noise(x, mask, sigma=cfg.noise_sigma, generator=generator)
    if op == "color_jitter_local":
        return local_color_jitter(x, mask, strength=cfg.color_strength, generator=generator)
    raise ValueError(f"unknown image operation: {op}")


def propose_image_mutation(
    x: Tensor,
    *,
    mask: Optional[Tensor] = None,
    op: Optional[str] = None,
    config: Optional[ImageMutationConfig] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor, Dict[str, object]]:
    """Create one local synthetic anomaly proposal for a batch."""

    cfg = config or ImageMutationConfig()
    if mask is None:
        mask = sample_rectangular_mask(
            x.shape[0],
            x.shape[-2],
            x.shape[-1],
            cfg.mask,
            device=x.device,
            generator=generator,
        )
    else:
        mask = _coerce_mask(mask, x)

    if op is None:
        ops = ("copy_shift", "local_blur", "local_noise", "color_jitter_local")
        op_idx = int(torch.randint(0, len(ops), (1,), generator=generator).item())
        op = ops[op_idx]

    mutated = apply_local_op(x, mask, op, cfg, generator=generator)
    blended = feather_blend(x, mutated, mask, radius=cfg.feather_radius)
    blended = clamp_like_reference(blended, x)
    return blended, mask, {"op": op, "mask_area": mask.flatten(1).mean(dim=1)}


def _coerce_mask(mask: Tensor, x: Tensor) -> Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4:
        raise ValueError("mask must have shape [H,W], [B,H,W], or [B,1,H,W]")
    mask = mask.to(device=x.device, dtype=x.dtype)
    if mask.shape[0] == 1 and x.shape[0] > 1:
        mask = mask.expand(x.shape[0], -1, -1, -1)
    if mask.shape[-2:] != x.shape[-2:]:
        mask = F.interpolate(mask, size=x.shape[-2:], mode="nearest")
    return mask[:, :1].clamp(0.0, 1.0)
