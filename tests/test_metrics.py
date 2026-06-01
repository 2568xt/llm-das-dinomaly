import numpy as np

from llm_das_dinomaly.evaluation.metrics import (
    binary_average_precision,
    binary_auroc,
    binary_f1_max,
    metric_bundle,
    pixel_aupro,
)


def test_binary_metrics_perfect_scores():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert binary_auroc(labels, scores) == 1.0
    assert binary_average_precision(labels, scores) == 1.0
    assert binary_f1_max(labels, scores) == 1.0


def test_metric_bundle_returns_none_for_single_class_auroc():
    labels = np.array([0, 0, 0])
    scores = np.array([0.1, 0.2, 0.3])
    summary = metric_bundle(labels, scores)
    assert summary["auroc"] is None
    assert summary["ap"] == 0.0
    assert summary["f1"] == 0.0


def test_binary_metrics_are_tie_order_invariant():
    scores = np.array([0.5, 0.5])
    labels_pos_first = np.array([1, 0])
    labels_neg_first = np.array([0, 1])

    assert binary_average_precision(labels_pos_first, scores) == 0.5
    assert binary_average_precision(labels_neg_first, scores) == 0.5
    assert np.isclose(binary_f1_max(labels_pos_first, scores), 2.0 / 3.0)
    assert np.isclose(binary_f1_max(labels_neg_first, scores), 2.0 / 3.0)


def test_pixel_aupro_perfect_region_overlap():
    masks = np.zeros((1, 4, 4), dtype=np.uint8)
    masks[0, 1:3, 1:3] = 1
    scores = masks.astype(np.float32)
    assert pixel_aupro(masks, scores, max_fpr=0.3, num_thresholds=8) == 1.0


def test_pixel_aupro_constant_scores_returns_zero():
    masks = np.zeros((1, 2, 2), dtype=np.uint8)
    masks[0, 0, 0] = 1
    scores = np.zeros_like(masks, dtype=np.float32)

    assert pixel_aupro(masks, scores, max_fpr=0.3, num_thresholds=4) == 0.0


def test_pixel_aupro_uses_true_all_negative_origin():
    masks = np.zeros((1, 2, 2), dtype=np.uint8)
    masks[0, 0, 0] = 1
    scores = np.array([[[0.9, 0.9], [0.0, 0.0]]], dtype=np.float32)

    assert np.isclose(pixel_aupro(masks, scores, max_fpr=0.5, num_thresholds=4), 2.0 / 3.0)
