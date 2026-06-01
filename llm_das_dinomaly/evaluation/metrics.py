from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def binary_auroc(labels, scores) -> Optional[float]:
    y_true = _as_binary_vector(labels)
    y_score = _as_score_vector(scores)
    _validate_same_length(y_true, y_score)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _average_ranks(y_score)
    rank_sum = float(ranks[pos].sum())
    auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def binary_average_precision(labels, scores) -> float:
    y_true = _as_binary_vector(labels)
    y_score = _as_score_vector(scores)
    _validate_same_length(y_true, y_score)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return 0.0

    tp = 0
    fp = 0
    previous_recall = 0.0
    average_precision = 0.0
    for group_pos, group_neg in _binary_threshold_groups(y_true, y_score):
        tp += group_pos
        fp += group_neg
        precision = tp / max(tp + fp, 1)
        recall = tp / n_pos
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
    return float(average_precision)


def binary_f1_max(labels, scores) -> float:
    y_true = _as_binary_vector(labels)
    y_score = _as_score_vector(scores)
    _validate_same_length(y_true, y_score)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return 0.0

    tp = 0
    fp = 0
    best_f1 = 0.0
    for group_pos, group_neg in _binary_threshold_groups(y_true, y_score):
        tp += group_pos
        fp += group_neg
        precision = tp / max(tp + fp, 1)
        recall = tp / n_pos
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        best_f1 = max(best_f1, f1)
    return float(best_f1)


def metric_bundle(labels, scores) -> Dict[str, Optional[float]]:
    return {
        "auroc": binary_auroc(labels, scores),
        "ap": binary_average_precision(labels, scores),
        "f1": binary_f1_max(labels, scores),
    }


def pixel_aupro(gt_masks, score_maps, *, max_fpr: float = 0.3, num_thresholds: int = 200) -> Optional[float]:
    gt = np.asarray(gt_masks) > 0
    scores = np.asarray(score_maps, dtype=np.float64)
    if gt.shape != scores.shape:
        raise ValueError("gt_masks and score_maps must have the same shape")
    if gt.ndim != 3:
        raise ValueError("gt_masks and score_maps must have shape [N,H,W]")
    if max_fpr <= 0.0:
        raise ValueError("max_fpr must be positive")
    if int(gt.sum()) == 0 or int((~gt).sum()) == 0:
        return None

    regions = _connected_regions(gt)
    if not any(regions):
        return None

    thresholds = np.linspace(float(scores.max()), float(scores.min()), max(2, int(num_thresholds)))
    points = [(0.0, 0.0)]
    normal_pixels = max(int((~gt).sum()), 1)
    for threshold in thresholds:
        pred = scores >= threshold
        fpr = float(np.logical_and(pred, ~gt).sum() / normal_pixels)
        if fpr > max_fpr:
            continue
        overlaps = []
        for image_index, image_regions in enumerate(regions):
            for region in image_regions:
                region_size = max(int(region.sum()), 1)
                overlaps.append(float(np.logical_and(pred[image_index], region).sum() / region_size))
        if overlaps:
            points.append((fpr, float(np.mean(overlaps))))

    points = sorted(points)
    if points[-1][0] < max_fpr:
        points.append((max_fpr, points[-1][1]))

    xs = np.array([point[0] for point in points], dtype=np.float64)
    ys = np.array([point[1] for point in points], dtype=np.float64)
    return float(np.trapz(ys, xs) / max_fpr)


def _connected_regions(gt: np.ndarray) -> List[List[np.ndarray]]:
    return [_connected_regions_2d(mask) for mask in gt]


def _connected_regions_2d(mask: np.ndarray) -> List[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    regions: List[np.ndarray] = []
    height, width = mask.shape
    for row in range(height):
        for col in range(width):
            if seen[row, col] or not mask[row, col]:
                continue
            region = np.zeros(mask.shape, dtype=bool)
            stack = [(row, col)]
            seen[row, col] = True
            while stack:
                cur_row, cur_col = stack.pop()
                region[cur_row, cur_col] = True
                for next_row, next_col in (
                    (cur_row - 1, cur_col),
                    (cur_row + 1, cur_col),
                    (cur_row, cur_col - 1),
                    (cur_row, cur_col + 1),
                ):
                    if 0 <= next_row < height and 0 <= next_col < width:
                        if not seen[next_row, next_col] and mask[next_row, next_col]:
                            seen[next_row, next_col] = True
                            stack.append((next_row, next_col))
            regions.append(region)
    return regions


def _as_binary_vector(values) -> np.ndarray:
    arr = np.asarray(values).reshape(-1)
    return (arr > 0).astype(np.int64)


def _as_score_vector(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _average_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    start = 0
    while start < sorted_scores.shape[0]:
        end = start + 1
        while end < sorted_scores.shape[0] and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _binary_threshold_groups(labels: np.ndarray, scores: np.ndarray) -> List[Tuple[int, int]]:
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    groups: List[Tuple[int, int]] = []
    start = 0
    while start < sorted_scores.shape[0]:
        end = start + 1
        while end < sorted_scores.shape[0] and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group_labels = sorted_labels[start:end]
        group_pos = int((group_labels == 1).sum())
        group_neg = int((group_labels == 0).sum())
        groups.append((group_pos, group_neg))
        start = end
    return groups


def _validate_same_length(labels: np.ndarray, scores: np.ndarray) -> None:
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("labels and scores must have the same length")
