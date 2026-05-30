from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from llm_das_dinomaly.search.hardness import HardnessConfig, NormalScoreStats, attach_hardness
from llm_das_dinomaly.synth.image_ops import ImageMutationConfig, propose_image_mutation

Tensor = torch.Tensor


@dataclass(frozen=True)
class SearchConfig:
    budget: int = 24
    mutation: ImageMutationConfig = ImageMutationConfig()
    hardness: HardnessConfig = HardnessConfig()


@dataclass
class Candidate:
    x: Tensor
    mask: Tensor
    meta: Dict[str, Tensor]

    @property
    def hardness(self) -> Tensor:
        return self.meta["hardness"]


def score_aware_search(
    wrapper,
    x_ref: Tensor,
    stats: NormalScoreStats,
    *,
    config: Optional[SearchConfig] = None,
    generator: Optional[torch.Generator] = None,
) -> Candidate:
    """Search local image mutations and keep the hardest accepted candidate."""

    cfg = config or SearchConfig()
    best: Optional[Candidate] = None
    best_score = None

    for _ in range(cfg.budget):
        x_cand, mask, op_meta = propose_image_mutation(
            x_ref,
            config=cfg.mutation,
            generator=generator,
        )
        meta = wrapper.score_candidates(x_ref, x_cand, synth_masks=mask)
        meta = attach_hardness(meta, stats, config=cfg.hardness)
        meta["op_index"] = torch.zeros_like(meta["hardness"])
        if isinstance(op_meta.get("mask_area"), torch.Tensor):
            meta["proposal_mask_area"] = op_meta["mask_area"]

        score = meta["hardness"].clone()
        score = torch.where(meta["accepted"], score, score * 0.25)
        scalar = score.mean()
        if best is None or scalar > best_score:
            best = Candidate(x=x_cand, mask=mask, meta=meta)
            best_score = scalar

    if best is None:
        raise RuntimeError("search budget produced no candidates")
    return best
