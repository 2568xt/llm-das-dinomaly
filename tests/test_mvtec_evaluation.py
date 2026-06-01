from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from llm_das_dinomaly.evaluation.mvtec import _mean_category_metrics, evaluate_mvtec_detector
from llm_das_dinomaly.wrappers import DinomalyConfig, DinomalyWrapper


class EvalDummyDinomaly(nn.Module):
    def forward(self, x):
        pooled = F.adaptive_avg_pool2d(x, (4, 4))
        return [pooled, pooled * 0.5], [pooled.roll(shifts=1, dims=-1), pooled * 0.25]


class ModeTrackingDinomaly(nn.Module):
    def __init__(self):
        super().__init__()
        self.probe = nn.Dropout(p=0.5)
        self.forward_modes = []

    def forward(self, x):
        self.forward_modes.append((self.training, self.probe.training))
        pooled = F.adaptive_avg_pool2d(x, (4, 4))
        return [pooled, pooled * 0.5], [pooled.roll(shifts=1, dims=-1), pooled * 0.25]


class TrackingEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.probe = nn.Dropout(p=0.5)
        self.forward_modes = []

    def forward(self, x):
        self.forward_modes.append((self.training, self.probe.training))
        return x[:, 0]


class SentinelScoreWrapper(DinomalyWrapper):
    def __init__(self):
        super().__init__(
            EvalDummyDinomaly(),
            DinomalyConfig(image_size=32, crop_size=28, patch_size=7, gaussian_kernel=3, resize_mask=16),
        )
        self.predict_score_calls = 0

    def predict_map(self, x, *, resize_to=None, smooth=True):
        size = x.shape[-1] if resize_to is None else resize_to
        values = x.mean(dim=(1, 2, 3), keepdim=True)
        return values.expand(-1, 1, size, size).contiguous()

    def predict_score(self, x, *, topk_ratio=None, resize_to=None):
        self.predict_score_calls += 1
        return -x.mean(dim=(1, 2, 3))


def test_evaluate_mvtec_detector_writes_baseline_metrics(tmp_path):
    data_root = _fake_mvtec_test(tmp_path / "mvtec")
    wrapper = DinomalyWrapper(
        EvalDummyDinomaly(),
        DinomalyConfig(image_size=32, crop_size=28, patch_size=7, gaussian_kernel=3, resize_mask=16),
    )

    summary = evaluate_mvtec_detector(
        wrapper,
        data_root,
        categories=["bottle"],
        batch_size=1,
        device="cpu",
        resize_mask=16,
    )

    bottle = summary["categories"]["bottle"]
    assert bottle["num_images"] == 2
    assert bottle["num_anomalies"] == 1
    assert set(bottle["baseline"]) == {
        "image_auroc",
        "image_ap",
        "image_f1",
        "pixel_auroc",
        "pixel_ap",
        "pixel_f1",
        "pixel_aupro",
    }
    assert summary["mean"]["baseline"]["num_categories"] == 1


def test_evaluate_mvtec_detector_uses_predict_score_for_image_metrics(tmp_path):
    data_root = _fake_mvtec_test(tmp_path / "mvtec")
    wrapper = SentinelScoreWrapper()

    summary = evaluate_mvtec_detector(
        wrapper,
        data_root,
        categories=["bottle"],
        batch_size=2,
        device="cpu",
        resize_mask=16,
    )

    assert wrapper.predict_score_calls == 1
    assert summary["categories"]["bottle"]["baseline"]["image_auroc"] == 0.0


def test_evaluate_mvtec_detector_restores_wrapper_module_modes(tmp_path):
    data_root = _fake_mvtec_test(tmp_path / "mvtec")
    model = ModeTrackingDinomaly()
    wrapper = DinomalyWrapper(
        model,
        DinomalyConfig(image_size=32, crop_size=28, patch_size=7, gaussian_kernel=3, resize_mask=16),
    )
    wrapper.train()
    model.probe.eval()

    evaluate_mvtec_detector(
        wrapper,
        data_root,
        categories=["bottle"],
        batch_size=2,
        device="cpu",
        resize_mask=16,
    )

    assert model.forward_modes
    assert all(mode == (False, False) for mode in model.forward_modes)
    assert wrapper.training is True
    assert model.training is True
    assert model.probe.training is False


def test_evaluate_mvtec_detector_restores_enhancer_module_modes(tmp_path):
    data_root = _fake_mvtec_test(tmp_path / "mvtec")
    wrapper = DinomalyWrapper(
        EvalDummyDinomaly(),
        DinomalyConfig(image_size=32, crop_size=28, patch_size=7, gaussian_kernel=3, resize_mask=16),
    )
    enhancer = TrackingEnhancer()
    enhancer.train()
    enhancer.probe.eval()

    summary = evaluate_mvtec_detector(
        wrapper,
        data_root,
        categories=["bottle"],
        batch_size=2,
        device="cpu",
        resize_mask=16,
        enhancer_head=enhancer,
    )

    enhanced = summary["categories"]["bottle"]["enhanced"]
    assert enhanced["pixel_source"] == "base_dinomaly_map"
    assert enhancer.forward_modes
    assert all(mode == (False, False) for mode in enhancer.forward_modes)
    assert enhancer.training is True
    assert enhancer.probe.training is False


def test_mean_category_metrics_averages_numeric_values_only():
    summary = _mean_category_metrics(
        {
            "bottle": {
                "baseline": {
                    "image_auroc": 1.0,
                    "image_ap": None,
                    "pixel_source": "base_dinomaly_map",
                },
                "enhanced": {
                    "image_auroc": 0.25,
                    "image_f1": None,
                    "pixel_source": "base_dinomaly_map",
                },
            },
            "cable": {
                "baseline": {
                    "image_auroc": 0.5,
                    "image_ap": 0.75,
                    "pixel_source": "base_dinomaly_map",
                },
                "enhanced": {
                    "image_auroc": 0.75,
                    "image_f1": 1.0,
                    "pixel_source": "base_dinomaly_map",
                },
            },
        }
    )

    assert summary["baseline"]["num_categories"] == 2
    assert np.isclose(summary["baseline"]["image_auroc"], 0.75)
    assert np.isclose(summary["baseline"]["image_ap"], 0.75)
    assert "pixel_source" not in summary["baseline"]
    assert np.isclose(summary["enhanced"]["image_auroc"], 0.5)
    assert np.isclose(summary["enhanced"]["image_f1"], 1.0)
    assert "pixel_source" not in summary["enhanced"]


def _fake_mvtec_test(root: Path) -> Path:
    good_dir = root / "bottle" / "test" / "good"
    defect_dir = root / "bottle" / "test" / "broken_large"
    mask_dir = root / "bottle" / "ground_truth" / "broken_large"
    good_dir.mkdir(parents=True)
    defect_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(good_dir / "000.png")
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(defect_dir / "001.png")
    Image.new("L", (8, 8), color=255).save(mask_dir / "001_mask.png")
    return root
