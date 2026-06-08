from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import torch

from llm_das_dinomaly.data import (
    MPDD_CLASSES,
    MPDDGoodDataset,
    MPDDTestDataset,
    MVTEC_CLASSES,
    MVTecGoodDataset,
    MVTecTestDataset,
    RotatedGoodDataset,
    VISA_CLASSES,
    ViSAGoodDataset,
    ViSATestDataset,
    load_tensor_cache,
    save_tensor_cache,
    save_torch_payload,
)
from llm_das_dinomaly.enhancer import (
    MapFeatureHead,
    ScoreNormalizer,
    build_enhancer_features,
    fuse_scores,
    normalizer_from_metadata,
)
from llm_das_dinomaly.enhancer.heads import binary_enhancer_loss
from llm_das_dinomaly.evaluation import append_metric_jsonl, evaluate_mvtec_detector, write_metric_json
from llm_das_dinomaly.integrations import build_dinomaly_wrapper
from llm_das_dinomaly.integrations.dinomaly import _architecture, _dinomaly_import_context, _init_trainable_layers
from llm_das_dinomaly.search import NormalScoreStats, SearchConfig, score_aware_search
from llm_das_dinomaly.utils import ProgressBar, load_yaml_config, require_path, seed_everything


DATASET_SPECS = {
    "mvtec": {
        "classes": MVTEC_CLASSES,
        "default_category": "bottle",
        "good_dataset_cls": MVTecGoodDataset,
        "test_dataset_cls": MVTecTestDataset,
    },
    "mpdd": {
        "classes": MPDD_CLASSES,
        "default_category": "bracket_black",
        "good_dataset_cls": MPDDGoodDataset,
        "test_dataset_cls": MPDDTestDataset,
    },
    "visa": {
        "classes": VISA_CLASSES,
        "default_category": "candle",
        "good_dataset_cls": ViSAGoodDataset,
        "test_dataset_cls": ViSATestDataset,
    },
}

FEW_SHOT_ROTATION_ANGLES = (0, 90, 180, 270)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run LLM-DAS Dinomaly server pipeline.")
    parser.add_argument(
        "--config",
        required=True,
    )
    parser.add_argument(
        "--stage",
        choices=["check", "base-train", "hard-samples", "enhancer", "eval", "all"],
        default="all",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_yaml_config(args.config)
        summary = run_pipeline(cfg, stage=args.stage)
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from None
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_pipeline(cfg: Dict[str, Any], *, stage: str = "all") -> Dict[str, Any]:
    runtime = cfg.get("runtime", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    hard_cfg = cfg.get("hard_samples", {})
    enhancer_cfg = cfg.get("enhancer", {})
    eval_cfg = cfg.get("evaluation", {})
    base_cfg = cfg.get("base_training", {})

    output_root = Path(runtime.get("output_root", "outputs/server_mvtec")).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "run_summary.json"

    seed = int(runtime.get("seed", 7))
    device = str(runtime.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    mode = str(runtime.get("mode", "smoke")).lower()
    generator = seed_everything(seed)

    dataset_name = str(data_cfg.get("dataset", runtime.get("dataset", "mvtec"))).strip().lower()
    dataset_spec = _dataset_spec(dataset_name)
    configured_data_root, few_shot_root, data_root = _resolve_data_roots(data_cfg)
    few_shot_active = few_shot_root is not None
    dinomaly_root = require_path(model_cfg["dinomaly_root"], kind="DINOMALY_ROOT")
    categories = data_cfg.get("categories") or [dataset_spec["default_category"]]
    limit_per_category = data_cfg.get("limit_per_category")
    if mode == "full":
        categories = (
            list(dataset_spec["classes"])
            if categories == [dataset_spec["default_category"]]
            else list(categories)
        )
        limit_per_category = None
    else:
        categories = list(categories)
    if few_shot_active:
        limit_per_category = None
    batch_size = int(runtime.get("batch_size", 16))
    show_progress = _as_bool(runtime.get("progress", True))
    eval_enabled = _as_bool(eval_cfg.get("enabled", True))
    eval_batch_size = int(eval_cfg.get("batch_size", batch_size))
    eval_num_workers = int(eval_cfg.get("num_workers", 0))
    eval_resize_mask = _resolve_optional_int(
        eval_cfg.get("resize_mask", 256),
        none_values={"none", "0", "false"},
    )
    eval_limit_per_category = _resolve_optional_int(
        eval_cfg.get("limit_per_category"),
        none_values={"none", "all", "-1"},
    )
    eval_beta = float(eval_cfg.get("beta", 1.0))
    eval_pixel_metrics = _as_bool(eval_cfg.get("pixel_metrics", True))
    eval_pixel_aupro = _as_bool(eval_cfg.get("pixel_aupro", False))
    eval_epoch_pixel_metrics = _as_bool(eval_cfg.get("epoch_pixel_metrics", False))
    metrics_dir = output_root / "metrics"

    base_dataset = dataset_spec["good_dataset_cls"](
        data_root,
        categories=categories,
        limit_per_category=limit_per_category,
    )
    rotation_angles = FEW_SHOT_ROTATION_ANGLES if few_shot_active else ()
    dataset = (
        RotatedGoodDataset(base_dataset, angles=rotation_angles)
        if rotation_angles
        else base_dataset
    )
    checkpoint_path, base_summary = _resolve_base_checkpoint(
        model_cfg,
        base_cfg,
        output_root=output_root,
        data_root=data_root,
        dinomaly_root=dinomaly_root,
        dataset_name=dataset_name,
        categories=categories,
        stage=stage,
        device=device,
        seed=seed,
        show_progress=show_progress,
        few_shot_active=few_shot_active,
    )
    summary: Dict[str, Any] = {
        "stage": stage,
        "mode": mode,
        "dataset": dataset_name,
        "seed": seed,
        "device": device,
        "data_root": str(data_root),
        "configured_data_root": None if configured_data_root is None else str(configured_data_root),
        "few_shot_root": None if few_shot_root is None else str(few_shot_root),
        "categories": categories,
        "num_normal_images": len(dataset),
        "num_base_normal_images": len(base_dataset),
        "output_root": str(output_root),
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "few_shot": {
            "active": few_shot_active,
            "effective_data_root": str(data_root),
            "root": None if few_shot_root is None else str(few_shot_root),
            "rotation_angles": list(rotation_angles),
            "base_normal_images": len(base_dataset),
            "rotated_normal_views": len(dataset),
        },
    }
    if base_summary:
        summary["base_checkpoint"] = base_summary
    if stage == "check":
        _write_json(summary_path, summary)
        return summary
    if stage == "base-train":
        if checkpoint_path is None:
            raise FileNotFoundError("CHECKPOINT_PATH is missing and base training did not produce a checkpoint")
        _write_json(summary_path, summary)
        return summary

    if checkpoint_path is None:
        raise FileNotFoundError(
            "CHECKPOINT_PATH is missing. Set CHECKPOINT_PATH or enable BASE_TRAIN_IF_MISSING for stage=all."
        )

    backbone = str(model_cfg.get("backbone", "dinov2reg_vit_base_14"))
    cache_context = _cache_context(
        dataset_name=dataset_name,
        categories=categories,
        data_root=data_root,
        checkpoint_path=checkpoint_path,
        backbone=backbone,
        few_shot_active=few_shot_active,
        rotation_angles=rotation_angles,
    )

    wrapper_pair = None

    def get_wrapper():
        nonlocal wrapper_pair
        if wrapper_pair is None:
            if show_progress:
                print("[llm-das-dinomaly] loading Dinomaly checkpoint...", file=sys.stderr, flush=True)
            wrapper_pair = build_dinomaly_wrapper(
                dinomaly_root=dinomaly_root,
                checkpoint_path=checkpoint_path,
                device=device,
                backbone=backbone,
                strict=bool(model_cfg.get("strict_checkpoint", False)),
            )
            summary["wrapper"] = wrapper_pair[1]
        return wrapper_pair[0]

    hard_cache = output_root / "hard_samples.pt"
    enhancer_path = output_root / "enhancer.pt"
    regenerate_hard = _as_bool(hard_cfg.get("regenerate", False))
    retrain_enhancer = _as_bool(enhancer_cfg.get("retrain", False))
    if few_shot_active and base_summary.get("trained"):
        regenerate_hard = True
        retrain_enhancer = True
    hard_search_budget = int(hard_cfg.get("search_budget", 4))
    hard_max_samples = _resolve_max_samples(hard_cfg.get("max_samples", 8), mode=mode)
    if few_shot_active:
        hard_max_samples = len(dataset)
    hard_target_samples = min(len(dataset), hard_max_samples)
    cache_images = _as_bool(hard_cfg.get("cache_images", False))
    hard_shard_size = int(hard_cfg.get("shard_size", 32))

    if stage == "eval":
        evaluation_summary: Dict[str, Any] = {}
        evaluation_summary["baseline"] = _run_and_write_evaluation(
            get_wrapper(),
            data_root,
            categories=categories,
            batch_size=eval_batch_size,
            device=device,
            resize_mask=eval_resize_mask,
            limit_per_category=eval_limit_per_category,
            beta=eval_beta,
            metrics_dir=metrics_dir,
            name="eval_summary",
            num_workers=eval_num_workers,
            pixel_metrics=eval_pixel_metrics,
            pixel_aupro=eval_pixel_aupro,
            show_progress=show_progress,
            test_dataset_cls=dataset_spec["test_dataset_cls"],
        )
        if enhancer_path.is_file():
            enhancer_summary = _try_summarize_enhancer_checkpoint(
                enhancer_path,
                cache_context=cache_context,
            )
            if enhancer_summary is not None:
                enhancer_head, enhancer_payload = _load_enhancer_head(enhancer_path)
                evaluation_summary["enhanced"] = _run_and_write_evaluation(
                    get_wrapper(),
                    data_root,
                    categories=categories,
                    batch_size=eval_batch_size,
                    device=device,
                    resize_mask=eval_resize_mask,
                    limit_per_category=eval_limit_per_category,
                    beta=eval_beta,
                    metrics_dir=metrics_dir,
                    name="eval_enhanced",
                    enhancer_head=enhancer_head,
                    fusion_calibration=enhancer_payload.get("fusion_calibration"),
                    num_workers=eval_num_workers,
                    pixel_metrics=eval_pixel_metrics,
                    pixel_aupro=eval_pixel_aupro,
                    show_progress=show_progress,
                    test_dataset_cls=dataset_spec["test_dataset_cls"],
                )
        summary["evaluation"] = evaluation_summary
        _write_json(summary_path, summary)
        return summary

    if stage in ("hard-samples", "all"):
        hard_summary = None
        if regenerate_hard:
            _clear_hard_sample_artifacts(hard_cache)
        elif hard_cache.is_file():
            hard_summary = _try_summarize_hard_cache(
                hard_cache,
                target_samples=hard_target_samples,
                search_budget=hard_search_budget,
                cache_images=cache_images,
                cache_context=cache_context,
            )
        if hard_summary is None:
            hard_summary = generate_hard_samples(
                get_wrapper(),
                dataset,
                hard_cache,
                batch_size=batch_size,
                device=device,
                generator=generator,
                search_budget=hard_search_budget,
                max_samples=hard_max_samples,
                cache_images=cache_images,
                shard_size=hard_shard_size,
                show_progress=show_progress,
                cache_context=cache_context,
            )
        summary["hard_samples"] = hard_summary

    if stage in ("enhancer", "all"):
        if not hard_cache.is_file():
            raise FileNotFoundError(f"hard sample cache does not exist: {hard_cache}")
        enhancer_summary = None
        if enhancer_path.is_file() and not retrain_enhancer:
            enhancer_summary = _try_summarize_enhancer_checkpoint(
                enhancer_path,
                cache_context=cache_context,
            )
            if eval_enabled and enhancer_summary is not None and "fusion_calibration" not in enhancer_summary:
                _warn(
                    "existing enhancer checkpoint is missing fusion_calibration; "
                    "retraining enhancer for enhanced evaluation"
                )
                enhancer_summary = None
        if eval_enabled:
            summary.setdefault("evaluation", {})["baseline"] = _run_and_write_evaluation(
                get_wrapper(),
                data_root,
                categories=categories,
                batch_size=eval_batch_size,
                device=device,
                resize_mask=eval_resize_mask,
                limit_per_category=eval_limit_per_category,
                beta=eval_beta,
                metrics_dir=metrics_dir,
                name="baseline_eval",
                num_workers=eval_num_workers,
                pixel_metrics=eval_pixel_metrics,
                pixel_aupro=eval_pixel_aupro,
                show_progress=show_progress,
                test_dataset_cls=dataset_spec["test_dataset_cls"],
            )
        if enhancer_summary is None:
            if eval_enabled:
                epoch_metrics_path = metrics_dir / "enhancer_epochs.jsonl"
                if epoch_metrics_path.exists():
                    epoch_metrics_path.unlink()

            def eval_callback(*, head, epoch, loss, fusion_calibration):
                record = {
                    "epoch": int(epoch),
                    "loss": float(loss),
                    "metrics": _run_and_write_evaluation(
                        get_wrapper(),
                        data_root,
                        categories=categories,
                        batch_size=eval_batch_size,
                        device=device,
                        resize_mask=eval_resize_mask,
                        limit_per_category=eval_limit_per_category,
                        beta=eval_beta,
                        metrics_dir=metrics_dir,
                        name=f"enhancer_epoch_{int(epoch):04d}",
                        enhancer_head=head,
                        fusion_calibration=fusion_calibration,
                        num_workers=eval_num_workers,
                        pixel_metrics=eval_epoch_pixel_metrics,
                        pixel_aupro=False,
                        show_progress=show_progress,
                        test_dataset_cls=dataset_spec["test_dataset_cls"],
                    ),
                }
                append_metric_jsonl(metrics_dir / "enhancer_epochs.jsonl", record)
                return record

            enhancer_summary = train_enhancer_from_cache(
                hard_cache,
                enhancer_path,
                epochs=int(enhancer_cfg.get("epochs", 1)),
                hidden_dim=int(enhancer_cfg.get("hidden_dim", 128)),
                lr=float(enhancer_cfg.get("lr", 1e-3)),
                seed=seed,
                show_progress=show_progress,
                eval_callback=eval_callback if eval_enabled else None,
                cache_context=cache_context,
            )
        summary["enhancer"] = enhancer_summary
        if eval_enabled and enhancer_path.is_file():
            enhancer_head, enhancer_payload = _load_enhancer_head(enhancer_path)
            summary.setdefault("evaluation", {})["final_enhanced"] = _run_and_write_evaluation(
                get_wrapper(),
                data_root,
                categories=categories,
                batch_size=eval_batch_size,
                device=device,
                resize_mask=eval_resize_mask,
                limit_per_category=eval_limit_per_category,
                beta=eval_beta,
                metrics_dir=metrics_dir,
                name="final_enhanced_eval",
                enhancer_head=enhancer_head,
                fusion_calibration=enhancer_payload.get("fusion_calibration"),
                num_workers=eval_num_workers,
                pixel_metrics=eval_pixel_metrics,
                pixel_aupro=eval_pixel_aupro,
                show_progress=show_progress,
                test_dataset_cls=dataset_spec["test_dataset_cls"],
            )

    _write_json(summary_path, summary)
    return summary


def _dataset_spec(dataset_name: str) -> Dict[str, Any]:
    if dataset_name not in DATASET_SPECS:
        expected = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"unsupported DATASET={dataset_name!r}; expected one of: {expected}")
    return DATASET_SPECS[dataset_name]


def _resolve_data_roots(data_cfg: Dict[str, Any]) -> Tuple[Optional[Path], Optional[Path], Path]:
    data_root_value = str(data_cfg.get("root", "") or "").strip()
    few_shot_value = str(data_cfg.get("few_shot_root", "") or "").strip()
    configured_data_root = Path(data_root_value).expanduser() if data_root_value else None
    if few_shot_value:
        few_shot_root = require_path(few_shot_value, kind="FEW_SHOT_ROOT")
        return configured_data_root, few_shot_root, few_shot_root
    if not data_root_value:
        raise FileNotFoundError("DATA_ROOT is not set and FEW_SHOT_ROOT is not set")
    data_root = require_path(data_root_value, kind="DATA_ROOT")
    return configured_data_root, None, data_root


def _resolve_base_checkpoint(
    model_cfg: Dict[str, Any],
    base_cfg: Dict[str, Any],
    *,
    output_root: Path,
    data_root: Path,
    dinomaly_root: Path,
    dataset_name: str,
    categories: Sequence[str],
    stage: str,
    device: str,
    seed: int,
    show_progress: bool,
    few_shot_active: bool = False,
) -> Tuple[Optional[Path], Dict[str, Any]]:
    backbone = str(model_cfg.get("backbone", "dinov2reg_vit_base_14"))
    explicit_checkpoint = str(model_cfg.get("checkpoint_path", "") or "").strip()
    train_if_missing = _as_bool(base_cfg.get("train_if_missing", False))
    force_retrain = _as_bool(base_cfg.get("force_retrain", False))
    total_iters = int(base_cfg.get("total_iters", 10000))
    eval_interval = int(base_cfg.get("eval_interval", 5000))
    batch_size = int(base_cfg.get("batch_size", 16))
    num_workers = int(base_cfg.get("num_workers", 4))
    checkpoint_dir = Path(base_cfg.get("checkpoint_dir") or (output_root / "base_checkpoints")).expanduser()

    if few_shot_active:
        output_path = checkpoint_dir / _base_checkpoint_filename(
            dataset_name=dataset_name,
            backbone=backbone,
            total_iters=total_iters,
        )
        should_train = stage in {"base-train", "all"}
        if should_train:
            training_summary = train_unified_dinomaly_checkpoint(
                dinomaly_root=dinomaly_root,
                data_root=data_root,
                output_path=output_path,
                categories=categories,
                backbone=backbone,
                device=device,
                total_iters=total_iters,
                eval_interval=eval_interval,
                batch_size=batch_size,
                num_workers=num_workers,
                seed=seed,
                show_progress=show_progress,
            )
            training_summary.update(
                {
                    "source": "few_shot_trained",
                    "trained": True,
                    "few_shot": True,
                    "ignored_checkpoint_path": explicit_checkpoint or None,
                    "checkpoint_search_skipped": True,
                    "train_if_missing": train_if_missing,
                    "force_retrain": True,
                }
            )
            return Path(training_summary["checkpoint_path"]), training_summary
        if output_path.is_file() and stage != "check":
            return output_path, {
                "source": "few_shot_run_checkpoint",
                "checkpoint_path": str(output_path),
                "trained": False,
                "few_shot": True,
                "ignored_checkpoint_path": explicit_checkpoint or None,
                "checkpoint_search_skipped": True,
                "train_if_missing": train_if_missing,
                "force_retrain": False,
            }
        if stage == "check":
            return None, {
                "source": "few_shot_will_train",
                "checkpoint_path": None,
                "trained": False,
                "few_shot": True,
                "will_train": True,
                "ignored_checkpoint_path": explicit_checkpoint or None,
                "checkpoint_search_skipped": True,
                "train_if_missing": train_if_missing,
                "force_retrain": True,
            }
        raise FileNotFoundError(
            "FEW_SHOT_ROOT is active but no run-local base checkpoint exists. "
            "Run stage=base-train or stage=all first."
        )

    if explicit_checkpoint:
        checkpoint = require_path(explicit_checkpoint, kind="CHECKPOINT_PATH", must_be_file=True)
        return checkpoint, {
            "source": "explicit",
            "checkpoint_path": str(checkpoint),
            "trained": False,
            "train_if_missing": train_if_missing,
            "force_retrain": force_retrain,
        }

    if not force_retrain:
        found = _find_existing_base_checkpoint(
            dataset_name=dataset_name,
            backbone=backbone,
            total_iters=total_iters,
            checkpoint_dirs=[checkpoint_dir, Path("checkpoints")],
        )
        if found is not None:
            return found, {
                "source": "search",
                "checkpoint_path": str(found),
                "trained": False,
                "train_if_missing": train_if_missing,
                "force_retrain": force_retrain,
            }

    should_train = stage == "base-train" or (stage == "all" and train_if_missing)
    if should_train:
        output_path = checkpoint_dir / _base_checkpoint_filename(
            dataset_name=dataset_name,
            backbone=backbone,
            total_iters=total_iters,
        )
        training_summary = train_unified_dinomaly_checkpoint(
            dinomaly_root=dinomaly_root,
            data_root=data_root,
            output_path=output_path,
            categories=categories,
            backbone=backbone,
            device=device,
            total_iters=total_iters,
            eval_interval=eval_interval,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            show_progress=show_progress,
        )
        training_summary.update(
            {
                "source": "trained",
                "train_if_missing": train_if_missing,
                "force_retrain": force_retrain,
            }
        )
        return Path(training_summary["checkpoint_path"]), training_summary

    if stage == "check" and train_if_missing:
        return None, {
            "source": "missing",
            "checkpoint_path": None,
            "trained": False,
            "will_train_if_missing": True,
            "train_if_missing": train_if_missing,
            "force_retrain": force_retrain,
            "checkpoint_search_dirs": [str(checkpoint_dir), "checkpoints"],
        }

    raise FileNotFoundError(
        "CHECKPOINT_PATH is not set and no existing "
        f"{dataset_name.upper()} unified checkpoint was found under {checkpoint_dir} or checkpoints/"
    )


def _find_existing_base_checkpoint(
    *,
    dataset_name: str,
    backbone: str,
    total_iters: int,
    checkpoint_dirs: Sequence[Path],
) -> Optional[Path]:
    names = [
        _base_checkpoint_filename(dataset_name=dataset_name, backbone=backbone, total_iters=total_iters),
        f"{dataset_name}_unified.pth",
        f"{dataset_name}_vitb_unified.pth",
        f"{dataset_name}_vitb_it{max(1, total_iters // 1000)}k.pth",
    ]
    for directory in checkpoint_dirs:
        directory = directory.expanduser()
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
        matches = sorted(directory.glob(f"{dataset_name}*unified*.pth")) if directory.is_dir() else []
        if matches:
            return matches[0]
        matches = sorted(directory.glob(f"{dataset_name}*.pth")) if directory.is_dir() else []
        if matches:
            return matches[0]
    return None


def _base_checkpoint_filename(*, dataset_name: str, backbone: str, total_iters: int) -> str:
    safe_backbone = "".join(char if char.isalnum() or char in "._-" else "_" for char in backbone)
    return f"{dataset_name}_unified_{safe_backbone}_it{max(1, total_iters // 1000)}k.pth"


def train_unified_dinomaly_checkpoint(
    *,
    dinomaly_root: Path,
    data_root: Path,
    output_path: Path,
    categories: Sequence[str],
    backbone: str,
    device: str,
    total_iters: int,
    eval_interval: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    show_progress: bool = True,
) -> Dict[str, Any]:
    seed_everything(seed)
    data_root = Path(data_root).expanduser().resolve()
    output_path = Path(output_path).expanduser()
    total_iters = max(1, int(total_iters))
    batch_size = max(1, int(batch_size))
    num_workers = max(0, int(num_workers))
    image_size = 448
    crop_size = 392
    metrics: List[Dict[str, Any]] = []
    losses: List[float] = []

    with _dinomaly_import_context(Path(dinomaly_root).expanduser().resolve()):
        from dataset import MVTecDataset, get_data_transforms
        from dinov1.utils import trunc_normal_
        from models import vit_encoder
        from models.uad import ViTill
        from models.vision_transformer import LinearAttention2
        from models.vision_transformer import Block as VitBlock
        from models.vision_transformer import bMlp
        from optimizers import StableAdamW
        from torch.utils.data import ConcatDataset, DataLoader
        from torchvision.datasets import ImageFolder
        from utils import WarmCosineScheduler, evaluation_batch, global_cosine_hm_percent
        from functools import partial
        import numpy as np
        import torch.nn as nn

        data_transform, gt_transform = get_data_transforms(image_size, crop_size)
        train_sets = []
        test_sets = []
        for category_index, category in enumerate(categories):
            train_path = data_root / category / "train"
            category_root = data_root / category
            if not train_path.is_dir():
                raise FileNotFoundError(f"MPDD train path does not exist: {train_path}")
            if not (category_root / "test").is_dir():
                raise FileNotFoundError(f"MPDD test path does not exist: {category_root / 'test'}")
            train_data = ImageFolder(root=str(train_path), transform=data_transform)
            train_data.classes = [category]
            train_data.class_to_idx = {category: category_index}
            train_data.samples = [(sample_path, category_index) for sample_path, _ in train_data.samples]
            train_sets.append(train_data)
            test_sets.append(
                MVTecDataset(
                    root=str(category_root),
                    transform=data_transform,
                    gt_transform=gt_transform,
                    phase="test",
                )
            )

        train_data = ConcatDataset(train_sets)
        if len(train_data) == 0:
            raise ValueError("base training found no train/good images")
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=len(train_data) >= batch_size,
        )
        if len(train_loader) == 0:
            raise ValueError("base training dataloader is empty")

        target_layers, fuse_encoder, fuse_decoder, embed_dim, num_heads = _architecture(backbone)
        encoder = vit_encoder.load(backbone)
        bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)])
        decoder = nn.ModuleList(
            [
                VitBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-8),
                    attn=LinearAttention2,
                )
                for _ in range(8)
            ]
        )
        model = ViTill(
            encoder=encoder,
            bottleneck=bottleneck,
            decoder=decoder,
            target_layers=target_layers,
            mask_neighbor_size=0,
            fuse_layer_encoder=fuse_encoder,
            fuse_layer_decoder=fuse_decoder,
        ).to(device)
        _init_trainable_layers(model, trunc_normal_)
        trainable = nn.ModuleList([model.bottleneck, model.decoder])
        optimizer = StableAdamW(
            [{"params": trainable.parameters()}],
            lr=2e-3,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
            amsgrad=True,
            eps=1e-10,
        )
        scheduler = WarmCosineScheduler(
            optimizer,
            base_value=2e-3,
            final_value=2e-4,
            total_iters=total_iters,
            warmup_iters=100,
        )

        progress = ProgressBar(total_iters, label="base training", enabled=show_progress)
        it = 0
        last_loss = None
        while it < total_iters:
            model.train()
            for images, _ in train_loader:
                images = images.to(device)
                en, de = model(images)
                p = min(0.9 * it / 1000, 0.9)
                loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                scheduler.step()
                last_loss = float(loss.item())
                losses.append(last_loss)
                it += 1
                progress.update(suffix=f"loss={last_loss:.6f}")

                should_eval = eval_interval > 0 and (it % eval_interval == 0 or it == total_iters)
                if should_eval:
                    metrics.append(
                        _evaluate_official_base_model(
                            model=model,
                            test_sets=test_sets,
                            categories=categories,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            device=device,
                            evaluation_batch=evaluation_batch,
                            data_loader_cls=DataLoader,
                            np_module=np,
                        )
                    )
                    model.train()
                if it >= total_iters:
                    break
        progress.close()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_path)

    return {
        "checkpoint_path": str(output_path),
        "trained": True,
        "categories": list(categories),
        "total_iters": total_iters,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "image_size": image_size,
        "crop_size": crop_size,
        "final_loss": last_loss,
        "losses_tail": losses[-10:],
        "metrics": metrics,
    }


def _evaluate_official_base_model(
    *,
    model,
    test_sets: Sequence[Any],
    categories: Sequence[str],
    batch_size: int,
    num_workers: int,
    device: str,
    evaluation_batch,
    data_loader_cls,
    np_module,
) -> Dict[str, Any]:
    rows = []
    for category, test_data in zip(categories, test_sets):
        test_loader = data_loader_cls(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = evaluation_batch(
            model,
            test_loader,
            device,
            max_ratio=0.01,
            resize_mask=256,
        )
        rows.append(
            {
                "category": category,
                "image_auroc": float(auroc_sp),
                "image_ap": float(ap_sp),
                "image_f1": float(f1_sp),
                "pixel_auroc": float(auroc_px),
                "pixel_ap": float(ap_px),
                "pixel_f1": float(f1_px),
                "pixel_aupro": float(aupro_px),
            }
        )

    mean = {}
    for key in ("image_auroc", "image_ap", "image_f1", "pixel_auroc", "pixel_ap", "pixel_f1", "pixel_aupro"):
        mean[key] = float(np_module.mean([row[key] for row in rows]))
    return {"categories": rows, "mean": mean}


def _cache_context(
    *,
    dataset_name: str,
    categories: Sequence[str],
    data_root: Path,
    checkpoint_path: Path,
    backbone: str,
    few_shot_active: bool = False,
    rotation_angles: Sequence[int] = (),
) -> Dict[str, Any]:
    context = {
        "dataset": dataset_name,
        "categories": list(categories),
        "data_root": str(Path(data_root).expanduser()),
        "checkpoint_path": str(Path(checkpoint_path).expanduser()),
        "backbone": backbone,
        "few_shot": bool(few_shot_active),
        "rotation_angles": [int(angle) for angle in rotation_angles],
    }
    if few_shot_active:
        context["checkpoint_fingerprint"] = _checkpoint_fingerprint(checkpoint_path)
    return context


def _normalize_cache_context(context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not context:
        return None
    normalized = {
        "dataset": str(context.get("dataset")),
        "categories": [str(category) for category in context.get("categories", [])],
        "data_root": str(context.get("data_root")),
        "checkpoint_path": str(context.get("checkpoint_path")),
        "backbone": str(context.get("backbone")),
        "few_shot": _as_bool(context.get("few_shot", False)),
        "rotation_angles": [int(angle) for angle in context.get("rotation_angles", [])],
    }
    if "checkpoint_fingerprint" in context:
        fingerprint = context.get("checkpoint_fingerprint") or {}
        normalized["checkpoint_fingerprint"] = {
            "size": int(fingerprint.get("size", -1)),
            "mtime_ns": int(fingerprint.get("mtime_ns", -1)),
        }
    return normalized


def _checkpoint_fingerprint(path: Path) -> Dict[str, int]:
    stat = Path(path).expanduser().stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _cache_context_matches(existing: Any, expected: Optional[Dict[str, Any]]) -> bool:
    normalized_expected = _normalize_cache_context(expected)
    if normalized_expected is None:
        return True
    return existing == normalized_expected


def generate_hard_samples(
    wrapper,
    dataset,
    output_path: Path,
    *,
    batch_size: int,
    device: str,
    generator: torch.Generator,
    search_budget: int,
    max_samples: int,
    cache_images: bool = False,
    shard_size: int = 32,
    show_progress: bool = True,
    cache_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    target_samples = min(len(dataset), max_samples)
    if target_samples <= 0:
        raise ValueError("max_samples must select at least one train/good image")
    batch_size = max(1, int(batch_size))
    shard_size = max(1, int(shard_size))
    target_batches = (target_samples + batch_size - 1) // batch_size
    shard_dir = _hard_sample_shard_dir(output_path)

    normal_scores = []
    normal_progress = ProgressBar(target_batches, label="normal scoring", enabled=show_progress)
    normal_seen = 0
    with torch.no_grad():
        for _, images, _ in _iter_dataset_batches(dataset, target_samples, batch_size):
            x = wrapper.preprocess(images).to(device)
            scores = wrapper.predict_score(x)
            normal_scores.append(scores.detach().cpu())
            normal_seen += x.shape[0]
            normal_progress.update(suffix=f"images={normal_seen}")
    normal_progress.close()

    all_scores = torch.cat(normal_scores, dim=0)
    stats = NormalScoreStats.from_scores(all_scores)
    completed_indices = _load_completed_sample_indices(
        shard_dir,
        target_samples=target_samples,
        search_budget=search_budget,
        cache_images=cache_images,
        cache_context=cache_context,
    )

    buffer = _new_hard_sample_buffer(cache_images=cache_images)
    generated_count = 0

    search_progress = ProgressBar(
        target_samples,
        label=f"hard search budget={search_budget}",
        enabled=show_progress,
    )
    cached_count = len(completed_indices)
    if cached_count:
        search_progress.update(cached_count, suffix=f"cached={cached_count}")

    with torch.no_grad():
        for start, images, metas in _iter_dataset_batches(dataset, target_samples, batch_size):
            pending = [
                (start + offset, image, meta)
                for offset, (image, meta) in enumerate(zip(images, metas))
                if start + offset not in completed_indices
            ]
            if not pending:
                continue

            x_batch = wrapper.preprocess([image for _, image, _ in pending]).to(device)
            for row, (idx, _, meta) in enumerate(pending):
                x_ref = x_batch[row : row + 1]
                candidate = score_aware_search(
                    wrapper,
                    x_ref,
                    stats,
                    config=SearchConfig(budget=search_budget),
                    generator=None if x_ref.is_cuda else generator,
                )

                normal_map = wrapper.predict_map(x_ref)
                normal_score = wrapper.predict_score(x_ref)
                normal_feats = wrapper.extract_features(x_ref, which="encoder")
                synth_map = wrapper.predict_map(candidate.x)
                synth_score = wrapper.predict_score(candidate.x)
                synth_feats = wrapper.extract_features(candidate.x, which="encoder")

                _append_hard_sample(
                    buffer,
                    sample_index=idx,
                    source_record=meta,
                    x_ref=x_ref,
                    candidate_x=candidate.x,
                    candidate_mask=candidate.mask,
                    hardness=candidate.hardness,
                    normal_score=normal_score,
                    normal_map=normal_map,
                    normal_feats=normal_feats,
                    synth_score=synth_score,
                    synth_map=synth_map,
                    synth_feats=synth_feats,
                )
                generated_count += 1
                search_progress.update(suffix=f"candidate={idx + 1} generated={generated_count}")

                if _hard_sample_buffer_len(buffer) >= shard_size:
                    saved_indices = _save_hard_sample_shard(
                        shard_dir,
                        buffer,
                        stats=stats,
                        search_budget=search_budget,
                        cache_images=cache_images,
                        cache_context=cache_context,
                    )
                    completed_indices.update(saved_indices)
                    buffer = _new_hard_sample_buffer(cache_images=cache_images)

    if _hard_sample_buffer_len(buffer):
        saved_indices = _save_hard_sample_shard(
            shard_dir,
            buffer,
            stats=stats,
            search_budget=search_budget,
            cache_images=cache_images,
            cache_context=cache_context,
        )
        completed_indices.update(saved_indices)
    search_progress.close()

    missing = sorted(set(range(target_samples)) - completed_indices)
    if missing:
        preview = ", ".join(str(idx) for idx in missing[:8])
        raise RuntimeError(f"hard sample shards are incomplete; missing sample indices: {preview}")

    summary = _merge_hard_sample_shards(
        shard_dir,
        output_path,
        target_samples=target_samples,
        stats=stats,
        search_budget=search_budget,
        cache_images=cache_images,
        cache_context=cache_context,
    )
    summary["generated_candidates"] = generated_count
    summary["resumed_from_shards"] = cached_count > 0
    return summary


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
    cache_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    seed_everything(seed)
    payload = load_tensor_cache(cache_path)
    x = payload["tensors"]["enhancer_features"].float()
    labels = payload["tensors"]["labels"].float().reshape(-1)
    base_scores = payload["tensors"]["base_scores"].float().reshape(-1)
    if x.shape[0] != labels.numel() or x.shape[0] != base_scores.numel():
        raise ValueError(
            "enhancer cache tensor counts must align: "
            f"features={x.shape[0]}, labels={labels.numel()}, base_scores={base_scores.numel()}"
        )
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
            epoch_evaluations.append(
                eval_callback(
                    head=head,
                    epoch=epoch_idx + 1,
                    loss=losses[-1],
                    fusion_calibration=fusion_calibration,
                )
            )
        progress.update(suffix=f"loss={losses[-1]:.6f}")
    progress.close()

    with torch.no_grad():
        aux = torch.sigmoid(head(x)).reshape(-1)
        final_calibration = _fit_fusion_calibration(base_scores, aux)
        fused = fuse_scores(
            base_scores,
            aux,
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
            "cache_context": _normalize_cache_context(cache_context),
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
        "cache_context": _normalize_cache_context(cache_context),
    }
    if epoch_evaluations:
        summary["epoch_evaluations"] = epoch_evaluations
    return summary


def _iter_dataset_batches(
    dataset,
    total: int,
    batch_size: int,
) -> Iterator[Tuple[int, List[Any], List[Dict[str, Any]]]]:
    for start in range(0, total, batch_size):
        items = [dataset[idx] for idx in range(start, min(start + batch_size, total))]
        images, metas = zip(*items)
        yield start, list(images), list(metas)


def _hard_sample_shard_dir(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}_shards"


def _new_hard_sample_buffer(*, cache_images: bool) -> Dict[str, List[Any]]:
    buffer: Dict[str, List[Any]] = {
        "sample_indices": [],
        "source_records": [],
        "hardness": [],
        "enhancer_features": [],
        "labels": [],
        "base_scores": [],
    }
    if cache_images:
        buffer.update(
            {
                "normal_images": [],
                "synthetic_images": [],
                "masks": [],
                "anomaly_maps": [],
            }
        )
    return buffer


def _hard_sample_buffer_len(buffer: Dict[str, List[Any]]) -> int:
    return len(buffer["sample_indices"])


def _append_hard_sample(
    buffer: Dict[str, List[Any]],
    *,
    sample_index: int,
    source_record: Dict[str, Any],
    x_ref: torch.Tensor,
    candidate_x: torch.Tensor,
    candidate_mask: torch.Tensor,
    hardness: torch.Tensor,
    normal_score: torch.Tensor,
    normal_map: torch.Tensor,
    normal_feats: Sequence[torch.Tensor],
    synth_score: torch.Tensor,
    synth_map: torch.Tensor,
    synth_feats: Sequence[torch.Tensor],
) -> None:
    source = dict(source_record)
    source["sample_index"] = int(sample_index)
    buffer["sample_indices"].append(int(sample_index))
    buffer["source_records"].append(source)
    buffer["hardness"].append(hardness.detach().cpu().reshape(-1))
    buffer["base_scores"].extend([normal_score.detach().cpu().reshape(-1), synth_score.detach().cpu().reshape(-1)])
    buffer["enhancer_features"].extend(
        [
            build_enhancer_features(
                normal_score.detach().cpu(),
                normal_map.detach().cpu(),
                encoder_groups=[feat.detach().cpu() for feat in normal_feats],
            ),
            build_enhancer_features(
                synth_score.detach().cpu(),
                synth_map.detach().cpu(),
                encoder_groups=[feat.detach().cpu() for feat in synth_feats],
            ),
        ]
    )
    buffer["labels"].extend([torch.zeros(1), torch.ones(1)])

    if "normal_images" in buffer:
        buffer["normal_images"].append(x_ref.detach().cpu())
        buffer["synthetic_images"].append(candidate_x.detach().cpu())
        buffer["masks"].append(candidate_mask.detach().cpu())
        buffer["anomaly_maps"].extend([normal_map.detach().cpu(), synth_map.detach().cpu()])


def _save_hard_sample_shard(
    shard_dir: Path,
    buffer: Dict[str, List[Any]],
    *,
    stats: NormalScoreStats,
    search_budget: int,
    cache_images: bool,
    cache_context: Optional[Dict[str, Any]] = None,
) -> Set[int]:
    indices = [int(idx) for idx in buffer["sample_indices"]]
    if not indices:
        return set()
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"shard-{min(indices):06d}-{max(indices) + 1:06d}.pt"
    tensors = {
        "sample_indices": torch.tensor(indices, dtype=torch.long),
        "hardness": torch.cat(buffer["hardness"], dim=0).float(),
        "enhancer_features": torch.cat(buffer["enhancer_features"], dim=0).float(),
        "labels": torch.cat(buffer["labels"], dim=0).float(),
        "base_scores": torch.cat(buffer["base_scores"], dim=0).float(),
    }
    if cache_images:
        tensors.update(
            {
                "normal_images": torch.cat(buffer["normal_images"], dim=0).float(),
                "synthetic_images": torch.cat(buffer["synthetic_images"], dim=0).float(),
                "masks": torch.cat(buffer["masks"], dim=0).float(),
                "anomaly_maps": torch.cat(buffer["anomaly_maps"], dim=0).float(),
            }
        )
    metadata = {
        "cache_version": 2,
        "type": "hard_sample_shard",
        "complete": True,
        "normal_stats": stats.__dict__,
        "num_candidates": len(indices),
        "source_records": list(buffer["source_records"]),
        "search_budget": int(search_budget),
        "cache_images": bool(cache_images),
        "cache_context": _normalize_cache_context(cache_context),
    }
    save_tensor_cache(shard_path, tensors, metadata)
    return set(indices)


def _merge_hard_sample_shards(
    shard_dir: Path,
    output_path: Path,
    *,
    target_samples: int,
    stats: NormalScoreStats,
    search_budget: int,
    cache_images: bool,
    cache_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    entries = {}
    used_shards = []
    for shard_path in _iter_shard_paths(shard_dir):
        payload = load_tensor_cache(shard_path)
        if not _is_compatible_hard_shard(
            payload,
            search_budget=search_budget,
            cache_images=cache_images,
            cache_context=cache_context,
        ):
            continue
        tensors = payload["tensors"]
        metadata = payload.get("metadata", {})
        indices = _payload_sample_indices(payload)
        records = metadata.get("source_records", [])
        used_shards.append(shard_path)
        for row, sample_index in enumerate(indices):
            if sample_index >= target_samples or sample_index in entries:
                continue
            feature_start = row * 2
            feature_end = feature_start + 2
            source = records[row] if row < len(records) else {"sample_index": int(sample_index)}
            entries[int(sample_index)] = {
                "hardness": tensors["hardness"][row : row + 1].detach().cpu(),
                "enhancer_features": tensors["enhancer_features"][feature_start:feature_end].detach().cpu(),
                "labels": tensors["labels"][feature_start:feature_end].detach().cpu(),
                "base_scores": tensors["base_scores"][feature_start:feature_end].detach().cpu(),
                "source_record": dict(source),
            }

    missing = sorted(set(range(target_samples)) - set(entries))
    if missing:
        preview = ", ".join(str(idx) for idx in missing[:8])
        raise RuntimeError(f"cannot finalize hard sample cache; missing shard entries: {preview}")

    ordered_indices = sorted(entries)
    tensors = {
        "sample_indices": torch.tensor(ordered_indices, dtype=torch.long),
        "hardness": torch.cat([entries[idx]["hardness"] for idx in ordered_indices], dim=0).float(),
        "enhancer_features": torch.cat(
            [entries[idx]["enhancer_features"] for idx in ordered_indices], dim=0
        ).float(),
        "labels": torch.cat([entries[idx]["labels"] for idx in ordered_indices], dim=0).float(),
        "base_scores": torch.cat([entries[idx]["base_scores"] for idx in ordered_indices], dim=0).float(),
    }
    metadata = {
        "cache_version": 2,
        "type": "hard_samples",
        "complete": True,
        "normal_stats": stats.__dict__,
        "num_candidates": len(ordered_indices),
        "source_records": [entries[idx]["source_record"] for idx in ordered_indices],
        "search_budget": int(search_budget),
        "cache_images": bool(cache_images),
        "image_tensors_location": "shards" if cache_images else "not_saved",
        "shard_dir": str(shard_dir),
        "num_shards": len(used_shards),
        "cache_context": _normalize_cache_context(cache_context),
    }
    save_tensor_cache(output_path, tensors, metadata)
    return _summarize_hard_cache_payload({"tensors": tensors, "metadata": metadata}, output_path, reused=False)


def _load_completed_sample_indices(
    shard_dir: Path,
    *,
    target_samples: int,
    search_budget: int,
    cache_images: bool,
    cache_context: Optional[Dict[str, Any]] = None,
) -> Set[int]:
    completed: Set[int] = set()
    for shard_path in _iter_shard_paths(shard_dir):
        try:
            payload = load_tensor_cache(shard_path)
        except Exception as exc:
            _warn(f"ignoring unreadable hard sample shard {shard_path}: {exc}")
            continue
        if not _is_compatible_hard_shard(
            payload,
            search_budget=search_budget,
            cache_images=cache_images,
            cache_context=cache_context,
        ):
            continue
        completed.update(idx for idx in _payload_sample_indices(payload) if idx < target_samples)
    return completed


def _iter_shard_paths(shard_dir: Path) -> List[Path]:
    if not shard_dir.is_dir():
        return []
    return sorted(shard_dir.glob("shard-*.pt"))


def _is_compatible_hard_shard(
    payload: Dict[str, Any],
    *,
    search_budget: int,
    cache_images: bool,
    cache_context: Optional[Dict[str, Any]] = None,
) -> bool:
    metadata = payload.get("metadata", {})
    if metadata.get("type") != "hard_sample_shard":
        return False
    if int(metadata.get("search_budget", -1)) != int(search_budget):
        return False
    if cache_images and not _as_bool(metadata.get("cache_images", False)):
        return False
    if not _cache_context_matches(metadata.get("cache_context"), cache_context):
        return False
    return True


def _payload_sample_indices(payload: Dict[str, Any]) -> List[int]:
    tensors = payload.get("tensors", {})
    if "sample_indices" in tensors:
        return [int(idx) for idx in tensors["sample_indices"].reshape(-1).tolist()]
    return [int(idx) for idx in payload.get("metadata", {}).get("sample_indices", [])]


def _try_summarize_hard_cache(
    path: Path,
    *,
    target_samples: int,
    search_budget: int,
    cache_images: bool,
    cache_context: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    try:
        payload = load_tensor_cache(path)
        summary = _summarize_hard_cache_payload(payload, path, reused=True)
        metadata = payload.get("metadata", {})
        if summary["num_candidates"] != target_samples:
            _warn(
                "existing hard sample cache has "
                f"{summary['num_candidates']} candidates but this run expects {target_samples}; regenerating"
            )
            return None
        if int(metadata.get("search_budget", search_budget)) != int(search_budget):
            _warn("existing hard sample cache uses a different SEARCH_BUDGET; regenerating")
            return None
        if cache_images and not _as_bool(metadata.get("cache_images", False)):
            _warn("existing hard sample cache has no image shards but CACHE_IMAGES=true; regenerating")
            return None
        if not _cache_context_matches(metadata.get("cache_context"), cache_context):
            _warn("existing hard sample cache was built for a different dataset/checkpoint; regenerating")
            return None
        return summary
    except Exception as exc:
        _quarantine_unreadable_file(path, "hard sample cache", exc)
        return None


def _summarize_hard_cache_payload(payload: Dict[str, Any], path: Path, *, reused: bool) -> Dict[str, Any]:
    tensors = payload.get("tensors", {})
    metadata = payload.get("metadata", {})
    if "enhancer_features" not in tensors or "labels" not in tensors or "base_scores" not in tensors:
        raise ValueError("hard sample cache is missing enhancer training tensors")

    hardness = tensors.get("hardness", torch.empty(0))
    num_candidates = int(metadata.get("num_candidates", int(hardness.numel())))
    normal_stats = metadata.get("normal_stats", {})
    return {
        "cache_path": str(path),
        "reused": bool(reused),
        "num_candidates": num_candidates,
        "normal_score_mean": _dict_float(normal_stats, "mean"),
        "normal_score_std": _dict_float(normal_stats, "std"),
        "accepted_proxy": int((hardness.float() > 0).sum().item()) if hardness.numel() else 0,
        "cache_images": _as_bool(metadata.get("cache_images", False)),
        "image_tensors_location": metadata.get("image_tensors_location", "cache"),
        "shard_dir": metadata.get("shard_dir"),
        "num_shards": metadata.get("num_shards"),
        "cache_context": metadata.get("cache_context"),
    }


def _try_summarize_enhancer_checkpoint(
    path: Path,
    *,
    cache_context: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    try:
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"enhancer checkpoint has unsupported payload type: {type(payload)!r}")
        if not _cache_context_matches(payload.get("cache_context"), cache_context):
            _warn("existing enhancer checkpoint was built for a different dataset/checkpoint; retraining")
            return None
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
        if "cache_context" in payload:
            summary["cache_context"] = payload["cache_context"]
        return summary
    except Exception as exc:
        _quarantine_unreadable_file(path, "enhancer checkpoint", exc)
        return None


def _load_enhancer_head(path: Path) -> Tuple[MapFeatureHead, Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        metadata = payload
    elif isinstance(payload, dict):
        state_dict = payload
        metadata = {"state_dict": payload}
    else:
        raise ValueError(f"enhancer checkpoint has unsupported payload type: {type(payload)!r}")

    if "input_dim" not in metadata:
        raise ValueError(f"enhancer checkpoint is missing input_dim: {path}")
    head = MapFeatureHead(
        input_dim=int(metadata["input_dim"]),
        hidden_dim=int(metadata.get("hidden_dim", 128)),
    )
    head.load_state_dict(state_dict)
    head.eval()
    return head, metadata


def _run_and_write_evaluation(
    wrapper,
    data_root: Path,
    *,
    categories: Sequence[str],
    batch_size: int,
    device: str,
    resize_mask: Optional[int],
    limit_per_category: Optional[int],
    beta: float,
    metrics_dir: Path,
    name: str,
    enhancer_head: Optional[MapFeatureHead] = None,
    fusion_calibration: Optional[Dict[str, Any]] = None,
    num_workers: int = 0,
    pixel_metrics: bool = True,
    pixel_aupro: bool = False,
    show_progress: bool = True,
    test_dataset_cls=MVTecTestDataset,
) -> Dict[str, Any]:
    base_normalizer = None
    aux_normalizer = None
    if enhancer_head is not None:
        if not isinstance(fusion_calibration, dict) or not {"base", "aux"} <= set(fusion_calibration):
            raise ValueError(
                "enhanced evaluation requires fusion_calibration with both 'base' and 'aux' normalizers"
            )
        if "base" in fusion_calibration:
            base_normalizer = normalizer_from_metadata(fusion_calibration["base"])
        if "aux" in fusion_calibration:
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
        num_workers=num_workers,
        pixel_metrics=pixel_metrics,
        pixel_aupro_enabled=pixel_aupro,
        show_progress=show_progress,
        progress_label=name,
        progress_path=metrics_dir / f"{name}.progress.json",
        test_dataset_cls=test_dataset_cls,
    )
    write_metric_json(metrics_dir / f"{name}.json", payload)
    return payload


def _fit_fusion_calibration(base_scores: torch.Tensor, aux_scores: torch.Tensor) -> Dict[str, Dict[str, float]]:
    base = ScoreNormalizer().fit(base_scores.reshape(-1))
    aux = ScoreNormalizer().fit(aux_scores.reshape(-1))
    return {
        "base": {"lo": float(base.lo), "hi": float(base.hi)},
        "aux": {"lo": float(aux.lo), "hi": float(aux.hi)},
    }


def _clear_hard_sample_artifacts(cache_path: Path) -> None:
    cache_path.unlink(missing_ok=True)
    shard_dir = _hard_sample_shard_dir(cache_path)
    if shard_dir.exists():
        shutil.rmtree(shard_dir)


def _quarantine_unreadable_file(path: Path, label: str, exc: Exception) -> None:
    if not path.exists():
        return
    target = path.with_name(f"{path.name}.corrupt")
    counter = 1
    while target.exists():
        target = path.with_name(f"{path.name}.corrupt.{counter}")
        counter += 1
    try:
        path.replace(target)
        _warn(f"moved unreadable {label} to {target}: {exc}")
    except OSError:
        _warn(f"unreadable {label} at {path}: {exc}")


def _dict_float(mapping: Any, key: str) -> Optional[float]:
    if isinstance(mapping, dict) and mapping.get(key) is not None:
        return float(mapping[key])
    return None


def _warn(message: str) -> None:
    print(f"[llm-das-dinomaly] {message}", file=sys.stderr, flush=True)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_max_samples(value: Any, *, mode: str) -> int:
    if mode == "full" and (value is None or str(value).lower() in {"all", "none", "-1"}):
        return 10**12
    return int(value)


def _resolve_optional_int(value: Any, *, none_values: Set[str]) -> Optional[int]:
    if value is None:
        return None
    if str(value).strip().lower() in none_values:
        return None
    return int(value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


if __name__ == "__main__":
    main()
