from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from llm_das_dinomaly.synth.image_ops import ImageMutationConfig, propose_image_mutation

Tensor = torch.Tensor


@dataclass(frozen=True)
class ImageSynthesisConfig:
    n_try: int = 16
    max_per_image: int = 4
    mutation: ImageMutationConfig = ImageMutationConfig()


def synthesize_image_candidates(
    x: Tensor,
    *,
    config: Optional[ImageSynthesisConfig] = None,
    generator: Optional[torch.Generator] = None,
) -> List[Dict[str, object]]:
    """Generate rule-based candidates before hard filtering."""

    cfg = config or ImageSynthesisConfig()
    records: List[Dict[str, object]] = []
    for _ in range(cfg.n_try):
        x_syn, mask, meta = propose_image_mutation(x, config=cfg.mutation, generator=generator)
        records.append({"x": x_syn, "mask": mask, "meta": meta})
    return records[: cfg.max_per_image]
