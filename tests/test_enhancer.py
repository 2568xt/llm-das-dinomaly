from __future__ import annotations

import torch

from llm_das_dinomaly.enhancer import MapFeatureHead, ScoreNormalizer, build_enhancer_features, fuse_scores, map_statistics
from llm_das_dinomaly.enhancer.fusion import normalizer_from_metadata
from llm_das_dinomaly.enhancer.heads import binary_enhancer_loss


def test_map_statistics_and_feature_builder():
    anomaly_map = torch.rand(3, 1, 16, 16)
    stats = map_statistics(anomaly_map)
    assert stats.shape == (3, 9)

    base = torch.rand(3)
    feats = [torch.rand(3, 8, 4, 4), torch.rand(3, 8, 4, 4)]
    x = build_enhancer_features(base, anomaly_map, encoder_groups=feats)
    assert x.shape[0] == 3
    assert x.shape[1] > stats.shape[1]


def test_head_loss_and_score_fusion():
    features = torch.rand(4, 18)
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    head = MapFeatureHead(input_dim=18, hidden_dim=8)
    logits = head(features)
    loss = binary_enhancer_loss(logits, labels)
    assert loss.item() > 0

    base_norm = ScoreNormalizer().fit(torch.tensor([1.0, 2.0, 3.0]))
    fused = fuse_scores(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([3.0, 2.0, 1.0]), base_normalizer=base_norm)
    assert fused.shape == (3,)
    assert torch.isfinite(fused).all()


def test_normalizer_from_metadata_roundtrip():
    normalizer = normalizer_from_metadata({"lo": 0.25, "hi": 0.75})
    values = normalizer.transform(torch.tensor([0.25, 0.5, 0.75]))
    assert torch.allclose(values, torch.tensor([0.0, 0.5, 1.0]), atol=1e-5)
    assert ScoreNormalizer(lo=0.0, hi=1.0).transform(torch.tensor([0.5])).item() == 0.5
