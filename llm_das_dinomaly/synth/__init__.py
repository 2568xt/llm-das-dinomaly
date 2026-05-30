"""Synthetic anomaly policies."""

from llm_das_dinomaly.synth.image_ops import propose_image_mutation
from llm_das_dinomaly.synth.masks import MaskConfig, sample_patch_mask, sample_rectangular_mask

__all__ = [
    "MaskConfig",
    "propose_image_mutation",
    "sample_patch_mask",
    "sample_rectangular_mask",
]
