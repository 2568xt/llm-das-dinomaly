from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_das_dinomaly.wrappers import DinomalyConfig, DinomalyWrapper


class DummyDinomaly(nn.Module):
    def forward(self, x):
        pooled = F.adaptive_avg_pool2d(x, (4, 4))
        enc = [pooled, pooled * 0.5]
        dec = [pooled.roll(shifts=1, dims=-1), pooled * 0.25]
        return enc, dec


def test_preprocess_crop_and_normalize_shape():
    wrapper = DinomalyWrapper(DummyDinomaly(), DinomalyConfig(image_size=32, crop_size=28, patch_size=7))
    x = torch.rand(2, 3, 20, 24)
    out = wrapper.preprocess(x)
    assert out.shape == (2, 3, 28, 28)
    assert torch.isfinite(out).all()


def test_predict_map_score_and_features():
    wrapper = DinomalyWrapper(
        DummyDinomaly(),
        DinomalyConfig(image_size=32, crop_size=28, patch_size=7, gaussian_kernel=3),
    )
    x = torch.rand(2, 3, 28, 28)
    features = wrapper.extract_features(x, which="both")
    assert len(features["encoder_groups"]) == 2
    assert features["encoder_groups"][0].shape[-2:] == (4, 4)

    anomaly_map = wrapper.predict_map(x)
    score = wrapper.predict_score(x)
    assert anomaly_map.shape == (2, 1, 28, 28)
    assert score.shape == (2,)
    assert (score >= 0).all()


def test_score_candidates_with_mask_overlap():
    wrapper = DinomalyWrapper(
        DummyDinomaly(),
        DinomalyConfig(image_size=32, crop_size=28, patch_size=7, gaussian_kernel=3),
    )
    x = torch.rand(1, 3, 28, 28)
    x_cand = x.roll(shifts=2, dims=-1)
    mask = torch.zeros(1, 1, 28, 28)
    mask[:, :, 4:12, 4:12] = 1
    meta = wrapper.score_candidates(x, x_cand, synth_masks=mask)
    assert {"score_ref", "score_cand", "score_delta", "map", "perturb_l1", "mask_area", "mask_overlap"} <= set(meta)
    assert meta["mask_area"].shape == (1,)
    assert meta["mask_overlap"].shape == (1,)
