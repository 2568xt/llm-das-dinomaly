from __future__ import annotations

from typing import Optional, Tuple

import torch

from llm_das_dinomaly.synth.masks import sample_patch_mask

Tensor = torch.Tensor


def token_swap_or_mix(
    features: Tensor,
    patch_mask: Optional[Tensor] = None,
    *,
    n_patch: Tuple[int, int] = (4, 64),
    mix_alpha: float = 0.5,
    noise_sigma: float = 0.1,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor]:
    """Inject local token mixing/noise into fused feature maps."""

    if features.ndim != 4:
        raise ValueError("features must have shape [B,C,H,W]")
    if patch_mask is None:
        patch_mask = sample_patch_mask(
            features.shape[0],
            features.shape[-2],
            features.shape[-1],
            n_patch=n_patch,
            device=features.device,
            generator=generator,
        )
    else:
        patch_mask = patch_mask.to(device=features.device, dtype=features.dtype)
        if patch_mask.ndim == 3:
            patch_mask = patch_mask.unsqueeze(1)

    shifted = torch.roll(features, shifts=(1, -1), dims=(-2, -1))
    noise = torch.randn(features.shape, device=features.device, dtype=features.dtype, generator=generator)
    noise = noise * (features.flatten(2).std(dim=2, keepdim=True).unsqueeze(-1) + 1e-6) * noise_sigma
    mixed = features * (1.0 - mix_alpha) + shifted * mix_alpha + noise
    return features * (1.0 - patch_mask) + mixed * patch_mask, patch_mask
