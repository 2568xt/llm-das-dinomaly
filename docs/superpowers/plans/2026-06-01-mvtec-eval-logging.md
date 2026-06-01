# MVTec Evaluation Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add baseline and enhancer evaluation on the original MVTec test split, and record metrics during enhancer training as well as after training.

**Architecture:** Keep `scripts/run_server_mvtec.sh` as a thin entrypoint and add reusable evaluation logic under `llm_das_dinomaly/`. The existing hard-sample pipeline will still produce synthetic hard-sample tensors and an enhancer checkpoint, while the new evaluation path will load `test/` images plus `ground_truth/` masks, score them with the frozen Dinomaly wrapper, optionally fuse image-level enhancer scores, and write metrics under `OUTPUT_ROOT/metrics/`.

**Tech Stack:** Python 3.8+, PyTorch, NumPy, PIL, existing `DinomalyWrapper`, existing `MapFeatureHead`, existing tensor-cache helpers, pytest.

---

## Current Source Findings

- `scripts/run_server_mvtec.sh` validates paths, runs `python -m llm_das_dinomaly.pipelines.server_mvtec --stage check`, then runs the same module with `--stage all`.
- `llm_das_dinomaly/pipelines/server_mvtec.py` currently has stages `check`, `hard-samples`, `enhancer`, and `all`.
- The current real outputs under `OUTPUT_ROOT` are `run_summary.json`, `hard_samples.pt`, `hard_samples_shards/`, and `enhancer.pt`.
- By default `hard_samples.pt` is compact. It stores tensors needed to train the enhancer, not a full exported image dataset. Image tensors are stored only when `CACHE_IMAGES=true`, and then they live in shard files.
- `train_enhancer_from_cache()` currently records `losses`, `final_loss`, and a training-cache `fused_score_mean`, but it does not evaluate on MVTec `test/`.
- The vendored official Dinomaly code has `MVTecDataset` and `evaluation_batch()` in `third_party/Dinomaly/`, but the project pipeline does not call them. The project should implement a wrapper-native evaluator outside `third_party/` so enhanced-score evaluation can be tested without editing vendor code.

## File Structure

- Modify `llm_das_dinomaly/data/mvtec.py`: add original MVTec test split indexing and mask loading.
- Create `llm_das_dinomaly/evaluation/__init__.py`: export evaluation functions.
- Create `llm_das_dinomaly/evaluation/metrics.py`: dependency-light AUROC/AP/F1/AUPRO helpers.
- Create `llm_das_dinomaly/evaluation/mvtec.py`: wrapper-native MVTec evaluator and JSON/JSONL metric writers.
- Modify `llm_das_dinomaly/enhancer/fusion.py`: add explicit normalizer construction from saved metadata.
- Modify `llm_das_dinomaly/pipelines/server_mvtec.py`: add `eval` stage, baseline evaluation, enhancer epoch callbacks, final evaluation, and metric summary fields.
- Modify `configs/server_mvtec.yaml`: add `evaluation:` config.
- Modify `configs/server_paths.example.env`: document evaluation overrides.
- Modify `README.md`: explain what the runner outputs and how to run/evaluate enhanced metrics.
- Modify tests under `tests/`: add focused tests for test-split loading, metrics, evaluator behavior, and training-time callback logging.

## Metrics Contract

The evaluator must record these fields:

```json
{
  "category": "bottle",
  "num_images": 83,
  "num_anomalies": 63,
  "baseline": {
    "image_auroc": 0.9912,
    "image_ap": 0.9971,
    "image_f1": 0.9843,
    "pixel_auroc": 0.9822,
    "pixel_ap": 0.7123,
    "pixel_f1": 0.6811,
    "pixel_aupro": 0.9410
  },
  "enhanced": {
    "image_auroc": 0.9930,
    "image_ap": 0.9978,
    "image_f1": 0.9865,
    "pixel_source": "base_dinomaly_map"
  }
}
```

`enhanced` is image-level in this implementation because `MapFeatureHead` outputs an auxiliary image score, not a new pixel anomaly map. Pixel metrics remain from the base Dinomaly map and must be labeled with `"pixel_source": "base_dinomaly_map"` instead of silently pretending the enhancer changed segmentation.

### Task 1: Add MVTec Test Split Support

**Files:**
- Modify: `llm_das_dinomaly/data/mvtec.py`
- Test: `tests/test_config_and_mvtec.py`

- [ ] **Step 1: Add failing test for MVTec test image and mask loading**

Append this test to `tests/test_config_and_mvtec.py`:

```python
from PIL import Image

from llm_das_dinomaly.data.mvtec import MVTecTestDataset, list_mvtec_test_images


def test_mvtec_test_indexer_loads_good_and_defect_masks(tmp_path):
    root = tmp_path / "mvtec"
    good_dir = root / "bottle" / "test" / "good"
    defect_dir = root / "bottle" / "test" / "broken_large"
    mask_dir = root / "bottle" / "ground_truth" / "broken_large"
    good_dir.mkdir(parents=True)
    defect_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(good_dir / "000.png")
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(defect_dir / "001.png")
    Image.new("L", (8, 8), color=255).save(mask_dir / "001_mask.png")

    records = list_mvtec_test_images(root, categories=["bottle"])
    assert [record.label for record in records] == [1, 0]
    assert records[0].mask_path == mask_dir / "001_mask.png"
    assert records[1].mask_path is None

    dataset = MVTecTestDataset(root, categories=["bottle"])
    image, mask, meta = dataset[0]
    assert image.mode == "RGB"
    assert mask.mode == "L"
    assert meta["category"] == "bottle"
    assert meta["defect_type"] == "broken_large"
    assert meta["label"] == 1

    _, good_mask, good_meta = dataset[1]
    assert good_mask.getbbox() is None
    assert good_meta["defect_type"] == "good"
    assert good_meta["label"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m pytest tests/test_config_and_mvtec.py::test_mvtec_test_indexer_loads_good_and_defect_masks -q
```

Expected: FAIL with `ImportError` or `AttributeError` because `MVTecTestDataset` and `list_mvtec_test_images` do not exist yet.

- [ ] **Step 3: Implement test split records and dataset**

Add this code to `llm_das_dinomaly/data/mvtec.py` after `MVTecImageRecord`:

```python
@dataclass(frozen=True)
class MVTecTestImageRecord:
    path: Path
    category: str
    defect_type: str
    split: str = "test"
    label: int = 0
    mask_path: Optional[Path] = None
```

Add these functions and class after `list_mvtec_train_good()`:

```python
def list_mvtec_test_images(
    root,
    *,
    categories: Optional[Sequence[str]] = None,
    limit_per_category: Optional[int] = None,
) -> List[MVTecTestImageRecord]:
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"MVTec root does not exist: {root}")

    selected = tuple(categories or MVTEC_CLASSES)
    records: List[MVTecTestImageRecord] = []
    missing = []
    for category in selected:
        test_dir = root / category / "test"
        if not test_dir.is_dir():
            missing.append(str(test_dir))
            continue
        category_records: List[MVTecTestImageRecord] = []
        for defect_dir in sorted(path for path in test_dir.iterdir() if path.is_dir()):
            defect_type = defect_dir.name
            label = 0 if defect_type == "good" else 1
            image_paths = sorted(path for path in defect_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
            for image_path in image_paths:
                mask_path = None
                if label:
                    mask_path = _find_mvtec_mask(root, category, defect_type, image_path)
                category_records.append(
                    MVTecTestImageRecord(
                        path=image_path,
                        category=category,
                        defect_type=defect_type,
                        label=label,
                        mask_path=mask_path,
                    )
                )
        if limit_per_category is not None:
            category_records = category_records[:limit_per_category]
        records.extend(category_records)

    if missing and not records:
        raise FileNotFoundError("no MVTec test images found. Checked:\n" + "\n".join(missing))
    if not records:
        raise FileNotFoundError(f"no MVTec test images found under {root}")
    return records


def _find_mvtec_mask(root: Path, category: str, defect_type: str, image_path: Path) -> Path:
    mask_dir = root / category / "ground_truth" / defect_type
    candidates = [
        mask_dir / f"{image_path.stem}_mask.png",
        mask_dir / f"{image_path.stem}.png",
        mask_dir / f"{image_path.stem}_mask.bmp",
        mask_dir / f"{image_path.stem}.bmp",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing MVTec ground-truth mask for {image_path}")
```

Add this class after `MVTecGoodDataset`:

```python
class MVTecTestDataset(Dataset):
    def __init__(
        self,
        root,
        *,
        categories: Optional[Sequence[str]] = None,
        limit_per_category: Optional[int] = None,
    ) -> None:
        self.records = list_mvtec_test_images(
            root,
            categories=categories,
            limit_per_category=limit_per_category,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image = Image.open(record.path).convert("RGB")
        if record.mask_path is None:
            mask = Image.new("L", image.size, color=0)
        else:
            mask = Image.open(record.mask_path).convert("L")
        return image, mask, {
            "path": str(record.path),
            "mask_path": None if record.mask_path is None else str(record.mask_path),
            "category": record.category,
            "defect_type": record.defect_type,
            "split": record.split,
            "label": record.label,
        }
```

- [ ] **Step 4: Export the test dataset**

Modify `llm_das_dinomaly/data/__init__.py` to include:

```python
from .mvtec import MVTecGoodDataset, MVTecTestDataset, list_mvtec_train_good, list_mvtec_test_images
```

The file should still export the existing cache helpers.

- [ ] **Step 5: Run test to verify it passes**

Run:

```bash
python3 -m pytest tests/test_config_and_mvtec.py::test_mvtec_test_indexer_loads_good_and_defect_masks -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add llm_das_dinomaly/data tests/test_config_and_mvtec.py
git commit -m "feat: index mvtec test split"
```

### Task 2: Add Dependency-Light Metric Helpers

**Files:**
- Create: `llm_das_dinomaly/evaluation/__init__.py`
- Create: `llm_das_dinomaly/evaluation/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write failing metric tests**

Create `tests/test_metrics.py`:

```python
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


def test_pixel_aupro_perfect_region_overlap():
    masks = np.zeros((1, 4, 4), dtype=np.uint8)
    masks[0, 1:3, 1:3] = 1
    scores = masks.astype(np.float32)
    assert pixel_aupro(masks, scores, max_fpr=0.3, num_thresholds=8) == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_metrics.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'llm_das_dinomaly.evaluation'`.

- [ ] **Step 3: Create evaluation package exports**

Create `llm_das_dinomaly/evaluation/__init__.py`:

```python
from .metrics import binary_average_precision, binary_auroc, binary_f1_max, metric_bundle, pixel_aupro

__all__ = [
    "binary_average_precision",
    "binary_auroc",
    "binary_f1_max",
    "metric_bundle",
    "pixel_aupro",
]
```

- [ ] **Step 4: Implement metric helpers**

Create `llm_das_dinomaly/evaluation/metrics.py`:

```python
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def binary_auroc(labels, scores) -> Optional[float]:
    y_true = _as_binary_vector(labels)
    y_score = _as_score_vector(scores)
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
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-y_score, kind="mergesort")
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true == 1)
    precision = tp / (np.arange(sorted_true.shape[0], dtype=np.float64) + 1.0)
    return float((precision * (sorted_true == 1)).sum() / n_pos)


def binary_f1_max(labels, scores) -> float:
    y_true = _as_binary_vector(labels)
    y_score = _as_score_vector(scores)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-y_score, kind="mergesort")
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true == 1).astype(np.float64)
    fp = np.cumsum(sorted_true == 0).astype(np.float64)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / n_pos
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(f1.max(initial=0.0))


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
    if int(gt.sum()) == 0 or int((~gt).sum()) == 0:
        return None

    regions = _connected_regions(gt)
    if not regions:
        return None
    thresholds = np.linspace(float(scores.max()), float(scores.min()), max(2, int(num_thresholds)))
    points = []
    normal_pixels = max(int((~gt).sum()), 1)
    for threshold in thresholds:
        pred = scores >= threshold
        fpr = float(np.logical_and(pred, ~gt).sum() / normal_pixels)
        if fpr > max_fpr:
            continue
        overlaps = []
        for image_index, image_regions in enumerate(regions):
            for region in image_regions:
                overlaps.append(float(np.logical_and(pred[image_index], region).sum() / max(int(region.sum()), 1)))
        if overlaps:
            points.append((fpr, float(np.mean(overlaps))))
    if not points:
        return None
    points = sorted(points)
    if points[0][0] > 0.0:
        points.insert(0, (0.0, points[0][1]))
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
python3 -m pytest tests/test_metrics.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add llm_das_dinomaly/evaluation tests/test_metrics.py
git commit -m "feat: add binary evaluation metrics"
```

### Task 3: Add Wrapper-Native MVTec Evaluator

**Files:**
- Modify: `llm_das_dinomaly/evaluation/__init__.py`
- Create: `llm_das_dinomaly/evaluation/mvtec.py`
- Test: `tests/test_mvtec_evaluation.py`

- [ ] **Step 1: Write failing evaluator test**

Create `tests/test_mvtec_evaluation.py`:

```python
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from llm_das_dinomaly.evaluation.mvtec import evaluate_mvtec_detector
from llm_das_dinomaly.wrappers import DinomalyConfig, DinomalyWrapper


class EvalDummyDinomaly(nn.Module):
    def forward(self, x):
        pooled = F.adaptive_avg_pool2d(x, (4, 4))
        return [pooled, pooled * 0.5], [pooled.roll(shifts=1, dims=-1), pooled * 0.25]


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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m pytest tests/test_mvtec_evaluation.py -q
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` because `llm_das_dinomaly.evaluation.mvtec` does not exist yet.

- [ ] **Step 3: Implement evaluator**

Create `llm_das_dinomaly/evaluation/mvtec.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from llm_das_dinomaly.data import MVTecTestDataset
from llm_das_dinomaly.enhancer import build_enhancer_features, fuse_scores
from llm_das_dinomaly.enhancer.fusion import ScoreNormalizer
from llm_das_dinomaly.evaluation.metrics import metric_bundle, pixel_aupro


def evaluate_mvtec_detector(
    wrapper,
    data_root,
    *,
    categories: Sequence[str],
    batch_size: int,
    device: str,
    resize_mask: Optional[int],
    enhancer_head: Optional[torch.nn.Module] = None,
    beta: float = 1.0,
    base_normalizer: Optional[ScoreNormalizer] = None,
    aux_normalizer: Optional[ScoreNormalizer] = None,
    limit_per_category: Optional[int] = None,
) -> Dict[str, Any]:
    category_summaries: Dict[str, Any] = {}
    for category in categories:
        dataset = MVTecTestDataset(data_root, categories=[category], limit_per_category=limit_per_category)
        category_summaries[category] = _evaluate_category(
            wrapper,
            dataset,
            batch_size=max(1, int(batch_size)),
            device=device,
            resize_mask=resize_mask,
            enhancer_head=enhancer_head,
            beta=beta,
            base_normalizer=base_normalizer,
            aux_normalizer=aux_normalizer,
        )
    return {
        "categories": category_summaries,
        "mean": _mean_category_metrics(category_summaries),
    }


def write_metric_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_metric_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _evaluate_category(
    wrapper,
    dataset: MVTecTestDataset,
    *,
    batch_size: int,
    device: str,
    resize_mask: Optional[int],
    enhancer_head: Optional[torch.nn.Module],
    beta: float,
    base_normalizer: Optional[ScoreNormalizer],
    aux_normalizer: Optional[ScoreNormalizer],
) -> Dict[str, Any]:
    labels: List[torch.Tensor] = []
    base_scores: List[torch.Tensor] = []
    aux_scores: List[torch.Tensor] = []
    pixel_labels: List[torch.Tensor] = []
    pixel_scores: List[torch.Tensor] = []
    pixel_label_images: List[np.ndarray] = []
    pixel_score_images: List[np.ndarray] = []
    enhancer_was_training = bool(enhancer_head.training) if enhancer_head is not None else False
    if enhancer_head is not None:
        enhancer_head.eval()

    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            images, masks, metas = _load_batch(dataset, start, batch_size)
            x = wrapper.preprocess(images).to(device)
            anomaly_map = wrapper.predict_map(x, resize_to=resize_mask)
            score = _topk_image_score(anomaly_map, topk_ratio=wrapper.cfg.topk_ratio)
            labels.append(torch.tensor([int(meta["label"]) for meta in metas], dtype=torch.long))
            base_scores.append(score.detach().cpu())
            mask_tensor = _mask_batch_to_tensor(masks, anomaly_map.shape[-2:]).cpu()
            pixel_labels.append(mask_tensor[:, 0].reshape(-1).long())
            pixel_scores.append(anomaly_map.detach().cpu()[:, 0].reshape(-1))
            pixel_label_images.extend(mask_tensor[:, 0].numpy().astype(np.uint8))
            pixel_score_images.extend(anomaly_map.detach().cpu()[:, 0].numpy())

            if enhancer_head is not None:
                encoder_groups = wrapper.extract_features(x, which="encoder")
                features = build_enhancer_features(score.detach().cpu(), anomaly_map.detach().cpu(), encoder_groups=[
                    feat.detach().cpu() for feat in encoder_groups
                ])
                aux_scores.append(torch.sigmoid(enhancer_head(features)).detach().cpu().reshape(-1))

    if enhancer_head is not None and enhancer_was_training:
        enhancer_head.train()

    label_tensor = torch.cat(labels).numpy()
    base_tensor = torch.cat(base_scores).numpy()
    pixel_label_tensor = torch.cat(pixel_labels).numpy()
    pixel_score_tensor = torch.cat(pixel_scores).numpy()

    baseline_image = metric_bundle(label_tensor, base_tensor)
    baseline_pixel = metric_bundle(pixel_label_tensor, pixel_score_tensor)
    baseline_aupro = pixel_aupro(np.stack(pixel_label_images), np.stack(pixel_score_images))
    summary: Dict[str, Any] = {
        "num_images": int(label_tensor.shape[0]),
        "num_anomalies": int(label_tensor.sum()),
        "baseline": {
            "image_auroc": baseline_image["auroc"],
            "image_ap": baseline_image["ap"],
            "image_f1": baseline_image["f1"],
            "pixel_auroc": baseline_pixel["auroc"],
            "pixel_ap": baseline_pixel["ap"],
            "pixel_f1": baseline_pixel["f1"],
            "pixel_aupro": baseline_aupro,
        },
    }

    if aux_scores:
        aux_tensor = torch.cat(aux_scores)
        fused = fuse_scores(
            torch.from_numpy(base_tensor).float(),
            aux_tensor.float(),
            beta=beta,
            base_normalizer=base_normalizer,
            aux_normalizer=aux_normalizer,
        ).numpy()
        enhanced_image = metric_bundle(label_tensor, fused)
        summary["enhanced"] = {
            "image_auroc": enhanced_image["auroc"],
            "image_ap": enhanced_image["ap"],
            "image_f1": enhanced_image["f1"],
            "pixel_source": "base_dinomaly_map",
        }
    return summary


def _load_batch(dataset: MVTecTestDataset, start: int, batch_size: int):
    rows = [dataset[idx] for idx in range(start, min(start + batch_size, len(dataset)))]
    images, masks, metas = zip(*rows)
    return list(images), list(masks), list(metas)


def _mask_batch_to_tensor(masks, size) -> torch.Tensor:
    tensors = []
    for mask in masks:
        arr = np.asarray(mask.resize((size[1], size[0])), dtype=np.float32)
        tensors.append(torch.from_numpy((arr > 127).astype(np.float32)).view(1, 1, size[0], size[1]))
    return torch.cat(tensors, dim=0)


def _topk_image_score(anomaly_map: torch.Tensor, *, topk_ratio: float) -> torch.Tensor:
    flat = anomaly_map.flatten(1)
    k = max(1, min(flat.shape[1], int(flat.shape[1] * topk_ratio)))
    return torch.topk(flat, k=k, dim=1).values.mean(dim=1)


def _mean_category_metrics(category_summaries: Dict[str, Any]) -> Dict[str, Any]:
    baseline = _mean_named_metrics(category_summaries.values(), section="baseline")
    out: Dict[str, Any] = {"baseline": baseline}
    enhanced_values = [summary for summary in category_summaries.values() if "enhanced" in summary]
    if enhanced_values:
        out["enhanced"] = _mean_named_metrics(enhanced_values, section="enhanced")
    return out


def _mean_named_metrics(summaries: Iterable[Dict[str, Any]], *, section: str) -> Dict[str, Any]:
    rows = [summary[section] for summary in summaries]
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (float, int)) or value is None})
    result: Dict[str, Any] = {"num_categories": len(rows)}
    for key in keys:
        values = [row[key] for row in rows if isinstance(row.get(key), (float, int))]
        result[key] = None if not values else float(np.mean(values))
    return result
```

- [ ] **Step 4: Export evaluator**

Modify `llm_das_dinomaly/evaluation/__init__.py`:

```python
from .metrics import binary_average_precision, binary_auroc, binary_f1_max, metric_bundle, pixel_aupro
from .mvtec import append_metric_jsonl, evaluate_mvtec_detector, write_metric_json

__all__ = [
    "append_metric_jsonl",
    "binary_average_precision",
    "binary_auroc",
    "binary_f1_max",
    "evaluate_mvtec_detector",
    "metric_bundle",
    "pixel_aupro",
    "write_metric_json",
]
```

- [ ] **Step 5: Run evaluator test**

Run:

```bash
python3 -m pytest tests/test_mvtec_evaluation.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add llm_das_dinomaly/evaluation tests/test_mvtec_evaluation.py
git commit -m "feat: evaluate mvtec test metrics"
```

### Task 4: Store Fusion Calibration And Invoke Epoch Evaluation

**Files:**
- Modify: `llm_das_dinomaly/enhancer/fusion.py`
- Modify: `llm_das_dinomaly/pipelines/server_mvtec.py`
- Test: `tests/test_enhancer.py`
- Test: `tests/test_server_pipeline.py`

- [ ] **Step 1: Add failing tests for normalizer metadata and epoch callbacks**

Append to `tests/test_enhancer.py`:

```python
from llm_das_dinomaly.enhancer.fusion import ScoreNormalizer, normalizer_from_metadata


def test_normalizer_from_metadata_roundtrip():
    normalizer = normalizer_from_metadata({"lo": 0.25, "hi": 0.75})
    values = normalizer.transform(torch.tensor([0.25, 0.5, 0.75]))
    assert torch.allclose(values, torch.tensor([0.0, 0.5, 1.0]), atol=1e-5)
    assert ScoreNormalizer(lo=0.0, hi=1.0).transform(torch.tensor([0.5])).item() == 0.5
```

Append to `tests/test_server_pipeline.py`:

```python
from llm_das_dinomaly.pipelines.server_mvtec import train_enhancer_from_cache


def test_train_enhancer_records_epoch_eval_callback(tmp_path):
    cache_path = tmp_path / "hard_samples.pt"
    save_tensor_cache(
        cache_path,
        {
            "enhancer_features": torch.randn(4, 3),
            "labels": torch.tensor([0.0, 1.0, 0.0, 1.0]),
            "base_scores": torch.tensor([0.1, 0.8, 0.2, 0.9]),
        },
        {"type": "hard_samples"},
    )
    calls = []

    def eval_callback(*, head, epoch, loss, fusion_calibration):
        calls.append((epoch, loss, fusion_calibration["base"]))
        return {"epoch": epoch, "mean": {"enhanced": {"image_auroc": 1.0}}}

    summary = train_enhancer_from_cache(
        cache_path,
        tmp_path / "enhancer.pt",
        epochs=2,
        hidden_dim=4,
        lr=1e-3,
        seed=3,
        show_progress=False,
        eval_callback=eval_callback,
    )

    assert [call[0] for call in calls] == [1, 2]
    assert len(summary["epoch_evaluations"]) == 2
    payload = torch.load(tmp_path / "enhancer.pt", map_location="cpu")
    assert "fusion_calibration" in payload
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_enhancer.py::test_normalizer_from_metadata_roundtrip tests/test_server_pipeline.py::test_train_enhancer_records_epoch_eval_callback -q
```

Expected: FAIL because `normalizer_from_metadata` and `eval_callback` support do not exist.

- [ ] **Step 3: Add normalizer metadata helper**

Append this to `llm_das_dinomaly/enhancer/fusion.py`:

```python
def normalizer_from_metadata(metadata: dict) -> ScoreNormalizer:
    return ScoreNormalizer(lo=float(metadata["lo"]), hi=float(metadata["hi"]))
```

- [ ] **Step 4: Add callback support and calibration in training**

Modify `train_enhancer_from_cache()` in `llm_das_dinomaly/pipelines/server_mvtec.py`:

```python
def train_enhancer_from_cache(
    cache_path: Path,
    output_path: Path,
    *,
    epochs: int,
    hidden_dim: int,
    lr: float,
    seed: int,
    show_progress: bool = True,
    eval_callback=None,
) -> Dict[str, Any]:
    seed_everything(seed)
    payload = load_tensor_cache(cache_path)
    x = payload["tensors"]["enhancer_features"].float()
    labels = payload["tensors"]["labels"].float()
    base_scores = payload["tensors"]["base_scores"].float()
    head = MapFeatureHead(input_dim=x.shape[1], hidden_dim=hidden_dim)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    losses = []
    epoch_evaluations = []
    progress = ProgressBar(epochs, label="enhancer training", enabled=show_progress)
    for epoch_idx in range(epochs):
        logits = head(x)
        loss = binary_enhancer_loss(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        with torch.no_grad():
            aux_scores = torch.sigmoid(head(x)).reshape(-1)
            fusion_calibration = _fit_fusion_calibration(base_scores, aux_scores)
        if eval_callback is not None:
            eval_payload = eval_callback(
                head=head,
                epoch=epoch_idx + 1,
                loss=losses[-1],
                fusion_calibration=fusion_calibration,
            )
            epoch_evaluations.append(eval_payload)
        progress.update(suffix=f"loss={losses[-1]:.6f}")
    progress.close()

    with torch.no_grad():
        aux = torch.sigmoid(head(x))
        final_calibration = _fit_fusion_calibration(base_scores, aux.reshape(-1))
        fused = fuse_scores(
            base_scores,
            aux.reshape(-1),
            base_normalizer=ScoreNormalizer(**final_calibration["base"]),
            aux_normalizer=ScoreNormalizer(**final_calibration["aux"]),
        )
    save_torch_payload(
        output_path,
        {
            "state_dict": head.state_dict(),
            "input_dim": x.shape[1],
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "losses": losses,
            "fusion_calibration": final_calibration,
        },
    )
    summary = {
        "checkpoint_path": str(output_path),
        "reused": False,
        "input_dim": int(x.shape[1]),
        "epochs": epochs,
        "final_loss": losses[-1],
        "fused_score_mean": float(fused.mean().item()),
        "fusion_calibration": final_calibration,
    }
    if epoch_evaluations:
        summary["epoch_evaluations"] = epoch_evaluations
    return summary
```

Add this helper below `_try_summarize_enhancer_checkpoint()`:

```python
def _fit_fusion_calibration(base_scores: torch.Tensor, aux_scores: torch.Tensor) -> Dict[str, Dict[str, float]]:
    base = ScoreNormalizer().fit(base_scores.reshape(-1))
    aux = ScoreNormalizer().fit(aux_scores.reshape(-1))
    return {
        "base": {"lo": float(base.lo), "hi": float(base.hi)},
        "aux": {"lo": float(aux.lo), "hi": float(aux.hi)},
    }
```

Also import `ScoreNormalizer` at the top of `server_mvtec.py`:

```python
from llm_das_dinomaly.enhancer.fusion import ScoreNormalizer, normalizer_from_metadata
```

- [ ] **Step 5: Include calibration when reusing enhancer checkpoint**

Modify `_try_summarize_enhancer_checkpoint()` in `llm_das_dinomaly/pipelines/server_mvtec.py`:

```python
def _try_summarize_enhancer_checkpoint(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = torch.load(path, map_location="cpu")
        losses = payload.get("losses", [])
        summary = {
            "checkpoint_path": str(path),
            "reused": True,
            "input_dim": int(payload["input_dim"]),
            "hidden_dim": int(payload.get("hidden_dim", 0)),
            "epochs": int(payload.get("epochs", len(losses))),
        }
        if losses:
            summary["final_loss"] = float(losses[-1])
        if "fusion_calibration" in payload:
            summary["fusion_calibration"] = payload["fusion_calibration"]
        return summary
    except Exception as exc:
        _quarantine_unreadable_file(path, "enhancer checkpoint", exc)
        return None
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
python3 -m pytest tests/test_enhancer.py::test_normalizer_from_metadata_roundtrip tests/test_server_pipeline.py::test_train_enhancer_records_epoch_eval_callback -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add llm_das_dinomaly/enhancer/fusion.py llm_das_dinomaly/pipelines/server_mvtec.py tests/test_enhancer.py tests/test_server_pipeline.py
git commit -m "feat: record enhancer eval calibration"
```

### Task 5: Wire Evaluation Into Server Pipeline

**Files:**
- Modify: `llm_das_dinomaly/pipelines/server_mvtec.py`
- Modify: `configs/server_mvtec.yaml`
- Modify: `configs/server_paths.example.env`
- Test: `tests/test_server_pipeline.py`

- [ ] **Step 1: Add failing pipeline test for eval stage**

Append this test to `tests/test_server_pipeline.py`:

```python
def test_server_pipeline_eval_stage_writes_metrics(tmp_path, monkeypatch):
    data_root = _fake_mvtec_with_test(tmp_path / "mvtec")
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"not-used-by-dummy-wrapper")
    dinomaly_root = tmp_path / "Dinomaly"
    dinomaly_root.mkdir()

    def build_dummy_wrapper(**kwargs):
        return _dummy_wrapper(), {"dummy": True}

    monkeypatch.setattr("llm_das_dinomaly.pipelines.server_mvtec.build_dinomaly_wrapper", build_dummy_wrapper)
    summary = run_pipeline(
        {
            "runtime": {"output_root": str(tmp_path / "out"), "device": "cpu", "progress": False},
            "data": {"root": str(data_root), "categories": ["bottle"], "limit_per_category": 1},
            "model": {"dinomaly_root": str(dinomaly_root), "checkpoint_path": str(checkpoint)},
            "evaluation": {"enabled": True, "batch_size": 1, "resize_mask": 16, "limit_per_category": 2},
        },
        stage="eval",
    )

    assert "evaluation" in summary
    assert (tmp_path / "out" / "metrics" / "eval_summary.json").is_file()
    assert summary["evaluation"]["baseline"]["mean"]["baseline"]["num_categories"] == 1


def _fake_mvtec_with_test(root: Path) -> Path:
    _fake_mvtec(root, count=1)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m pytest tests/test_server_pipeline.py::test_server_pipeline_eval_stage_writes_metrics -q
```

Expected: FAIL because parser and `run_pipeline()` do not support `eval` stage or metric writing yet.

- [ ] **Step 3: Add eval stage parsing**

Modify the parser in `main()`:

```python
parser.add_argument("--stage", choices=["check", "hard-samples", "enhancer", "eval", "all"], default="all")
```

- [ ] **Step 4: Read evaluation config**

Inside `run_pipeline()` after `enhancer_cfg = cfg.get("enhancer", {})`, add:

```python
    eval_cfg = cfg.get("evaluation", {})
```

After `show_progress = _as_bool(runtime.get("progress", True))`, add:

```python
    eval_enabled = _as_bool(eval_cfg.get("enabled", True))
    eval_batch_size = int(eval_cfg.get("batch_size", batch_size))
    eval_resize_mask = eval_cfg.get("resize_mask", 256)
    eval_resize_mask = None if str(eval_resize_mask).lower() in {"none", "0", "false"} else int(eval_resize_mask)
    eval_limit = eval_cfg.get("limit_per_category")
    eval_limit = None if eval_limit is None or str(eval_limit).lower() in {"none", "all", "-1"} else int(eval_limit)
    eval_beta = float(eval_cfg.get("beta", 1.0))
    metrics_dir = output_root / "metrics"
```

Import evaluator helpers at the top:

```python
from llm_das_dinomaly.evaluation.mvtec import append_metric_jsonl, evaluate_mvtec_detector, write_metric_json
```

- [ ] **Step 5: Add helper for loading enhancer head**

Add this helper below `_try_summarize_enhancer_checkpoint()`:

```python
def _load_enhancer_head(path: Path) -> Tuple[MapFeatureHead, Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    head = MapFeatureHead(input_dim=int(payload["input_dim"]), hidden_dim=int(payload["hidden_dim"]))
    head.load_state_dict(payload["state_dict"])
    head.eval()
    return head, payload
```

- [ ] **Step 6: Add helper for running and writing evaluation**

Add this helper near `_write_json()`:

```python
def _run_and_write_evaluation(
    *,
    wrapper,
    data_root: Path,
    categories: Sequence[str],
    batch_size: int,
    device: str,
    resize_mask: Optional[int],
    metrics_dir: Path,
    name: str,
    enhancer_head: Optional[MapFeatureHead] = None,
    beta: float = 1.0,
    fusion_calibration: Optional[Dict[str, Dict[str, float]]] = None,
    limit_per_category: Optional[int] = None,
) -> Dict[str, Any]:
    base_normalizer = None
    aux_normalizer = None
    if fusion_calibration is not None:
        base_normalizer = normalizer_from_metadata(fusion_calibration["base"])
        aux_normalizer = normalizer_from_metadata(fusion_calibration["aux"])
    payload = evaluate_mvtec_detector(
        wrapper,
        data_root,
        categories=categories,
        batch_size=batch_size,
        device=device,
        resize_mask=resize_mask,
        enhancer_head=enhancer_head,
        beta=beta,
        base_normalizer=base_normalizer,
        aux_normalizer=aux_normalizer,
        limit_per_category=limit_per_category,
    )
    write_metric_json(metrics_dir / f"{name}.json", payload)
    return payload
```

- [ ] **Step 7: Wire standalone eval stage**

Inside `run_pipeline()` after `get_wrapper()` is defined and before hard-sample stage handling, add:

```python
    if stage == "eval":
        baseline_eval = _run_and_write_evaluation(
            wrapper=get_wrapper(),
            data_root=data_root,
            categories=categories,
            batch_size=eval_batch_size,
            device=device,
            resize_mask=eval_resize_mask,
            metrics_dir=metrics_dir,
            name="eval_summary",
            limit_per_category=eval_limit,
        )
        summary["evaluation"] = {"baseline": baseline_eval}
        if enhancer_path.is_file():
            enhancer_head, enhancer_payload = _load_enhancer_head(enhancer_path)
            enhanced_eval = _run_and_write_evaluation(
                wrapper=get_wrapper(),
                data_root=data_root,
                categories=categories,
                batch_size=eval_batch_size,
                device=device,
                resize_mask=eval_resize_mask,
                metrics_dir=metrics_dir,
                name="eval_enhanced",
                enhancer_head=enhancer_head,
                beta=eval_beta,
                fusion_calibration=enhancer_payload.get("fusion_calibration"),
                limit_per_category=eval_limit,
            )
            summary["evaluation"]["enhanced"] = enhanced_eval
        _write_json(summary_path, summary)
        return summary
```

- [ ] **Step 8: Wire baseline, epoch, and final evaluation into `all`**

In the `stage in ("enhancer", "all")` block, before calling `train_enhancer_from_cache()`, create this callback when `eval_enabled` is true:

```python
            eval_callback = None
            if eval_enabled:
                baseline_eval = _run_and_write_evaluation(
                    wrapper=get_wrapper(),
                    data_root=data_root,
                    categories=categories,
                    batch_size=eval_batch_size,
                    device=device,
                    resize_mask=eval_resize_mask,
                    metrics_dir=metrics_dir,
                    name="baseline_eval",
                    limit_per_category=eval_limit,
                )
                summary.setdefault("evaluation", {})["baseline"] = baseline_eval

                def eval_callback(*, head, epoch, loss, fusion_calibration):
                    payload = _run_and_write_evaluation(
                        wrapper=get_wrapper(),
                        data_root=data_root,
                        categories=categories,
                        batch_size=eval_batch_size,
                        device=device,
                        resize_mask=eval_resize_mask,
                        metrics_dir=metrics_dir,
                        name=f"enhancer_epoch_{epoch:04d}",
                        enhancer_head=head,
                        beta=eval_beta,
                        fusion_calibration=fusion_calibration,
                        limit_per_category=eval_limit,
                    )
                    record = {"epoch": epoch, "loss": loss, "metrics": payload}
                    append_metric_jsonl(metrics_dir / "enhancer_epochs.jsonl", record)
                    return record
```

Pass the callback to `train_enhancer_from_cache()`:

```python
                eval_callback=eval_callback,
```

After `summary["enhancer"] = enhancer_summary`, add final evaluation when an enhancer checkpoint exists and evaluation is enabled:

```python
        if eval_enabled and enhancer_path.is_file():
            enhancer_head, enhancer_payload = _load_enhancer_head(enhancer_path)
            final_eval = _run_and_write_evaluation(
                wrapper=get_wrapper(),
                data_root=data_root,
                categories=categories,
                batch_size=eval_batch_size,
                device=device,
                resize_mask=eval_resize_mask,
                metrics_dir=metrics_dir,
                name="final_enhanced_eval",
                enhancer_head=enhancer_head,
                beta=eval_beta,
                fusion_calibration=enhancer_payload.get("fusion_calibration"),
                limit_per_category=eval_limit,
            )
            summary.setdefault("evaluation", {})["final_enhanced"] = final_eval
```

- [ ] **Step 9: Add config and env knobs**

Append this to `configs/server_mvtec.yaml`:

```yaml
evaluation:
  enabled: ${EVAL_ENABLED:-true}
  batch_size: ${EVAL_BATCH_SIZE:-2}
  resize_mask: ${EVAL_RESIZE_MASK:-256}
  limit_per_category: ${EVAL_LIMIT_PER_CATEGORY:-all}
  beta: ${EVAL_FUSION_BETA:-1.0}
```

Append these commented lines to `configs/server_paths.example.env`:

```bash
# EVAL_ENABLED=true
# EVAL_BATCH_SIZE=2
# EVAL_RESIZE_MASK=256
# EVAL_LIMIT_PER_CATEGORY=all
# EVAL_FUSION_BETA=1.0
```

- [ ] **Step 10: Run focused pipeline tests**

Run:

```bash
python3 -m pytest tests/test_server_pipeline.py::test_server_pipeline_eval_stage_writes_metrics tests/test_server_pipeline.py::test_train_enhancer_records_epoch_eval_callback -q
```

Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add llm_das_dinomaly/pipelines/server_mvtec.py configs/server_mvtec.yaml configs/server_paths.example.env tests/test_server_pipeline.py
git commit -m "feat: wire mvtec evaluation into server pipeline"
```

### Task 6: Document The New Runner Behavior

**Files:**
- Modify: `README.md`
- Modify: `docs/EXPERIMENT_PLAN.md`

- [ ] **Step 1: Update README runner output section**

Replace the paragraph that starts with `The run writes hard_samples.pt` in `README.md` with:

```markdown
The run writes `hard_samples.pt`, `enhancer.pt`, `run_summary.json`, and
evaluation metrics under `OUTPUT_ROOT`. It also writes incremental hard-sample
shards under `OUTPUT_ROOT/hard_samples_shards/`. By default,
`hard_samples.pt` is compact and contains only the tensors needed for enhancer
training; generated image/mask/map tensors are saved in per-shard files only
when `CACHE_IMAGES=true`.

Evaluation artifacts are written under `OUTPUT_ROOT/metrics/`:

- `baseline_eval.json`: Dinomaly wrapper metrics on the original MVTec `test/`
  split before enhancer fusion.
- `enhancer_epochs.jsonl`: one JSON line per enhancer epoch when training runs
  with evaluation enabled.
- `enhancer_epoch_0001.json`, `enhancer_epoch_0002.json`, ...: full per-class
  metric snapshots for each recorded epoch.
- `final_enhanced_eval.json`: final image-level fused enhancer metrics after
  training.

The enhancer currently changes image-level scores only. Pixel AUROC/AP/F1 are
reported from the base Dinomaly anomaly map and labeled as
`base_dinomaly_map` in enhanced summaries.
```

- [ ] **Step 2: Document eval-only command**

Add this snippet after the server run command in `README.md`:

```markdown
To evaluate an existing checkpoint and optional existing `enhancer.pt` without
regenerating hard samples, run:

```bash
python -m llm_das_dinomaly.pipelines.server_mvtec --config configs/server_mvtec.yaml --stage eval
```
```

- [ ] **Step 3: Update experiment plan evidence**

In `docs/EXPERIMENT_PLAN.md`, add this paragraph under `## Reporting`:

```markdown
For server MVTec runs, keep `OUTPUT_ROOT/metrics/baseline_eval.json`,
`OUTPUT_ROOT/metrics/enhancer_epochs.jsonl`, and
`OUTPUT_ROOT/metrics/final_enhanced_eval.json` with the server logs. The epoch
JSONL file is the evidence that the enhanced image-level score was evaluated
against the original MVTec `test/` split during training rather than only after
training.
```

- [ ] **Step 4: Run documentation grep check**

Run:

```bash
rg -n "enhancer_epochs|final_enhanced_eval|stage eval|CACHE_IMAGES" README.md docs/EXPERIMENT_PLAN.md
```

Expected: the command prints the new README and experiment-plan lines.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/EXPERIMENT_PLAN.md
git commit -m "docs: explain mvtec evaluation outputs"
```

### Task 7: Full Verification And PR

**Files:**
- No new file changes unless verification exposes a defect.

- [ ] **Step 1: Run focused tests**

Run:

```bash
python3 -m pytest tests/test_config_and_mvtec.py tests/test_metrics.py tests/test_mvtec_evaluation.py tests/test_enhancer.py tests/test_server_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 2: Run required repository checks**

Run:

```bash
python3 -m pytest
python3 -m compileall -q llm_das_dinomaly scripts tests
```

Expected: both commands exit 0.

- [ ] **Step 3: Run local eval-stage smoke path**

Run this on a fake or real MVTec-shaped local path with a monkeypatched test only if no real checkpoint is available locally:

```bash
python3 -m pytest tests/test_server_pipeline.py::test_server_pipeline_eval_stage_writes_metrics -q
```

Expected: PASS and `eval_summary.json` exists in the pytest temp output.

On the GPU server, after the PR is merged into `main`, run:

```bash
RUN_MODE=smoke EVAL_LIMIT_PER_CATEGORY=8 bash scripts/run_server_mvtec.sh configs/server_mvtec.yaml configs/server_paths.env
```

Expected: `OUTPUT_ROOT/metrics/baseline_eval.json`, `OUTPUT_ROOT/metrics/enhancer_epochs.jsonl`, and `OUTPUT_ROOT/metrics/final_enhanced_eval.json` exist and contain per-class plus mean image metrics.

- [ ] **Step 4: Review changed files**

Run:

```bash
git status --short
git diff --stat origin/main...HEAD
git log --oneline origin/main..HEAD
```

Expected: only intentional project files are changed; no `outputs/`, `checkpoints/`, datasets, caches, or server-local env files are staged.

- [ ] **Step 5: Push and open PR**

Run:

```bash
git push -u origin codex/add-mvtec-eval-logging
gh pr create --base main --head codex/add-mvtec-eval-logging --title "Add MVTec evaluation logging" --body "$(cat <<'PR_BODY'
## Problem

The server runner currently produces hard-sample caches and an enhancer checkpoint, but it does not evaluate baseline or enhanced scores on the original MVTec test split during training.

## Summary

- Add MVTec test split indexing and mask loading.
- Add wrapper-native image and pixel metric helpers.
- Add an eval stage plus baseline, per-epoch, and final enhanced metric logging.
- Document runner outputs and server evaluation commands.

## Test Commands And Results

- `python3 -m pytest`
- `python3 -m compileall -q llm_das_dinomaly scripts tests`

Both pass locally.

## Server Update Or Run Commands

- `RUN_MODE=smoke EVAL_LIMIT_PER_CATEGORY=8 bash scripts/run_server_mvtec.sh configs/server_mvtec.yaml configs/server_paths.env`
- Full server run can use `RUN_MODE=full MAX_SAMPLES=all EVAL_LIMIT_PER_CATEGORY=all` after smoke metrics look sane.

## Risks Or Follow-Up Notes

- Enhanced metrics are image-level because the current enhancer produces an auxiliary score, not a replacement pixel map.
- Pixel metrics in enhanced summaries are explicitly labeled as coming from the base Dinomaly anomaly map.
PR_BODY
)"
```

Expected: PR opens against `main`.

## Self-Review

- Spec coverage: The plan answers the current-output question, adds original MVTec test evaluation, records metrics during enhancer training, preserves server-local path handling, and keeps implementation outside `third_party/Dinomaly/`.
- Empty-step scan: Every task names concrete files, commands, expected results, and code snippets for the implementation path.
- Type consistency: `fusion_calibration` is consistently shaped as `{"base": {"lo": float, "hi": float}, "aux": {"lo": float, "hi": float}}`; `evaluate_mvtec_detector()` accepts `ScoreNormalizer` instances; `train_enhancer_from_cache()` passes the callback keyword arguments used by the tests.
