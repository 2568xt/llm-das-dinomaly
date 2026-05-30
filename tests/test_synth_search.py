from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_das_dinomaly.search import NormalScoreStats, SearchConfig, accept_hard, hardness_score, score_aware_search
from llm_das_dinomaly.synth import MaskConfig, propose_image_mutation, sample_patch_mask, sample_rectangular_mask
from llm_das_dinomaly.wrappers import DinomalyConfig, DinomalyWrapper


class DummyDinomaly(nn.Module):
    def forward(self, x):
        pooled = F.adaptive_avg_pool2d(x, (4, 4))
        return [pooled, pooled * 0.5], [pooled.roll(shifts=1, dims=-1), pooled * 0.25]


def test_mask_generators_area_and_shape():
    gen = torch.Generator().manual_seed(3)
    mask = sample_rectangular_mask(3, 28, 28, MaskConfig(area_ratio=(0.02, 0.03)), generator=gen)
    assert mask.shape == (3, 1, 28, 28)
    assert mask.float().mean() > 0

    patch_mask = sample_patch_mask(2, 4, 4, n_patch=(2, 4), generator=gen)
    assert patch_mask.shape == (2, 1, 4, 4)
    assert patch_mask.sum() >= 4


def test_image_mutation_changes_masked_pixels():
    gen = torch.Generator().manual_seed(4)
    x = torch.rand(1, 3, 28, 28, generator=gen)
    x_syn, mask, meta = propose_image_mutation(x, op="local_noise", generator=gen)
    assert x_syn.shape == x.shape
    assert mask.shape == (1, 1, 28, 28)
    assert "op" in meta
    assert (x_syn - x).abs().sum() > 0


def test_hardness_acceptance_and_search_smoke():
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4])
    stats = NormalScoreStats.from_scores(scores)
    meta = {
        "score_cand": torch.tensor([stats.mean + 2 * stats.std]),
        "mask_area": torch.tensor([0.05]),
        "mask_overlap": torch.tensor([0.4]),
        "perturb_l1": torch.tensor([0.01]),
    }
    assert hardness_score(torch.tensor([2.0]), overlap=torch.tensor([0.4])).shape == (1,)
    assert accept_hard(meta, stats).item()

    gen = torch.Generator().manual_seed(5)
    wrapper = DinomalyWrapper(
        DummyDinomaly(),
        DinomalyConfig(image_size=32, crop_size=28, patch_size=7, gaussian_kernel=3),
    )
    x = torch.rand(1, 3, 28, 28, generator=gen)
    candidate = score_aware_search(wrapper, x, stats, config=SearchConfig(budget=3), generator=gen)
    assert candidate.x.shape == x.shape
    assert candidate.mask.shape == (1, 1, 28, 28)
    assert "hardness" in candidate.meta
