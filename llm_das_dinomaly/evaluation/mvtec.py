from __future__ import annotations

import json
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from llm_das_dinomaly.data import MVTecTestDataset
from llm_das_dinomaly.enhancer import build_enhancer_features, fuse_scores
from llm_das_dinomaly.enhancer.fusion import ScoreNormalizer
from llm_das_dinomaly.evaluation.metrics import metric_bundle, pixel_aupro
from llm_das_dinomaly.utils import ProgressBar


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
    num_workers: int = 0,
    pixel_metrics: bool = True,
    pixel_aupro_enabled: bool = False,
    show_progress: bool = True,
    progress_label: str = "mvtec eval",
    progress_path: Optional[Path] = None,
    test_dataset_cls=MVTecTestDataset,
) -> Dict[str, Any]:
    category_summaries: Dict[str, Any] = {}
    for category in categories:
        dataset = test_dataset_cls(
            data_root,
            categories=[category],
            limit_per_category=limit_per_category,
        )
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
            category=category,
            num_workers=num_workers,
            pixel_metrics=pixel_metrics,
            pixel_aupro_enabled=pixel_aupro_enabled,
            show_progress=show_progress,
            progress_label=progress_label,
        )
        if progress_path is not None:
            write_metric_json(
                progress_path,
                {
                    "categories": category_summaries,
                    "mean": _mean_category_metrics(category_summaries),
                    "completed_categories": list(category_summaries),
                    "total_categories": len(categories),
                },
            )

    return {
        "categories": category_summaries,
        "mean": _mean_category_metrics(category_summaries),
    }


def write_metric_json(path, payload: Dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_metric_jsonl(path, payload: Dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
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
    category: str,
    num_workers: int,
    pixel_metrics: bool,
    pixel_aupro_enabled: bool,
    show_progress: bool,
    progress_label: str,
) -> Dict[str, Any]:
    wrapper.to(device)
    dataloader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        collate_fn=_collate_batch,
    )

    labels: List[torch.Tensor] = []
    base_scores: List[torch.Tensor] = []
    aux_scores: List[torch.Tensor] = []
    pixel_labels: List[torch.Tensor] = []
    pixel_scores: List[torch.Tensor] = []
    pixel_label_images: List[np.ndarray] = []
    pixel_score_images: List[np.ndarray] = []

    enhancer_context = (
        _preserve_training_state(enhancer_head) if enhancer_head is not None else nullcontext()
    )
    progress = ProgressBar(
        len(dataloader),
        label=f"{progress_label} {category}",
        enabled=show_progress,
    )
    seen = 0
    with _preserve_training_state(wrapper), enhancer_context:
        with torch.no_grad():
            for images, masks, metas in dataloader:
                x = wrapper.preprocess(images).to(device)
                prediction = wrapper.predict_map_score_features(
                    x,
                    resize_to=resize_mask,
                    return_encoder=enhancer_head is not None,
                )
                anomaly_map = prediction["anomaly_map"]
                score = prediction["score"]

                batch_labels = torch.tensor([int(meta["label"]) for meta in metas], dtype=torch.long)
                labels.append(batch_labels)
                base_scores.append(score.detach().cpu())

                anomaly_cpu = anomaly_map.detach().cpu()
                if pixel_metrics:
                    mask_tensor = _mask_batch_to_tensor(masks, anomaly_map.shape[-2:]).cpu()
                    pixel_labels.append(mask_tensor[:, 0].reshape(-1).long())
                    pixel_scores.append(anomaly_cpu[:, 0].reshape(-1))
                    if pixel_aupro_enabled:
                        pixel_label_images.extend(mask_tensor[:, 0].numpy().astype(np.uint8))
                        pixel_score_images.extend(anomaly_cpu[:, 0].numpy())

                if enhancer_head is not None:
                    features = build_enhancer_features(
                        score.detach().cpu(),
                        anomaly_cpu,
                        encoder_groups=[feat.detach().cpu() for feat in prediction["encoder_groups"]],
                    )
                    features = features.to(_module_device(enhancer_head))
                    aux = torch.sigmoid(enhancer_head(features)).detach().cpu().reshape(-1)
                    aux_scores.append(aux)
                seen += len(metas)
                progress.update(suffix=f"images={seen}")
    progress.close()

    label_array = torch.cat(labels).numpy()
    base_tensor = torch.cat(base_scores)
    base_array = base_tensor.numpy()

    baseline_image = metric_bundle(label_array, base_array)
    baseline_metrics = {
        "image_auroc": baseline_image["auroc"],
        "image_ap": baseline_image["ap"],
        "image_f1": baseline_image["f1"],
        "pixel_auroc": None,
        "pixel_ap": None,
        "pixel_f1": None,
        "pixel_aupro": None,
    }
    if pixel_metrics:
        pixel_label_array = torch.cat(pixel_labels).numpy()
        pixel_score_array = torch.cat(pixel_scores).numpy()
        baseline_pixel = metric_bundle(pixel_label_array, pixel_score_array)
        baseline_metrics.update(
            {
                "pixel_auroc": baseline_pixel["auroc"],
                "pixel_ap": baseline_pixel["ap"],
                "pixel_f1": baseline_pixel["f1"],
                "pixel_aupro": (
                    pixel_aupro(np.stack(pixel_label_images), np.stack(pixel_score_images))
                    if pixel_aupro_enabled
                    else None
                ),
            }
        )

    summary: Dict[str, Any] = {
        "num_images": int(label_array.shape[0]),
        "num_anomalies": int(label_array.sum()),
        "baseline": baseline_metrics,
    }

    if aux_scores:
        aux_tensor = torch.cat(aux_scores)
        fused = fuse_scores(
            base_tensor.float(),
            aux_tensor.float(),
            beta=beta,
            base_normalizer=base_normalizer,
            aux_normalizer=aux_normalizer,
        ).numpy()
        enhanced_image = metric_bundle(label_array, fused)
        summary["enhanced"] = {
            "image_auroc": enhanced_image["auroc"],
            "image_ap": enhanced_image["ap"],
            "image_f1": enhanced_image["f1"],
            "pixel_source": "base_dinomaly_map",
        }

    return summary


def _load_batch(
    dataset: MVTecTestDataset,
    start: int,
    batch_size: int,
) -> Tuple[List[Any], List[Any], List[Dict[str, Any]]]:
    rows = [dataset[idx] for idx in range(start, min(start + batch_size, len(dataset)))]
    images, masks, metas = zip(*rows)
    return list(images), list(masks), list(metas)


def _collate_batch(rows: Sequence[Tuple[Any, Any, Dict[str, Any]]]):
    images, masks, metas = zip(*rows)
    return list(images), list(masks), list(metas)


def _mask_batch_to_tensor(masks: Sequence[Any], size: Tuple[int, int]) -> torch.Tensor:
    tensors = []
    for mask in masks:
        arr = np.asarray(mask, dtype=np.float32)
        tensor = torch.from_numpy((arr > 127).astype(np.float32)).view(1, 1, arr.shape[0], arr.shape[1])
        tensor = F.interpolate(tensor, size=size, mode="nearest")
        tensors.append(tensor)
    return torch.cat(tensors, dim=0)


def _mean_category_metrics(category_summaries: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "baseline": _mean_named_metrics(category_summaries.values(), section="baseline")
    }
    enhanced = [summary for summary in category_summaries.values() if "enhanced" in summary]
    if enhanced:
        out["enhanced"] = _mean_named_metrics(enhanced, section="enhanced")
    return out


def _mean_named_metrics(summaries: Iterable[Dict[str, Any]], *, section: str) -> Dict[str, Any]:
    rows = [summary[section] for summary in summaries]
    result: Dict[str, Any] = {"num_categories": len(rows)}
    keys = sorted({key for row in rows for key, value in row.items() if _is_meanable(value)})
    for key in keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (float, int))]
        result[key] = None if not values else float(np.mean(values))
    return result


def _is_meanable(value: Any) -> bool:
    return value is None or (isinstance(value, (float, int)) and not isinstance(value, bool))


def _module_device(module: torch.nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


@contextmanager
def _preserve_training_state(module: torch.nn.Module):
    modules = list(module.modules())
    states = [submodule.training for submodule in modules]
    try:
        module.eval()
        yield
    finally:
        for submodule, training in zip(modules, states):
            submodule.train(training)
