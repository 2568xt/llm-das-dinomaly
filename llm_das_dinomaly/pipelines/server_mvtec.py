from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from llm_das_dinomaly.data.mvtec import MVTEC_CLASSES
from llm_das_dinomaly.data import MVTecGoodDataset, save_tensor_cache
from llm_das_dinomaly.enhancer import MapFeatureHead, build_enhancer_features, fuse_scores
from llm_das_dinomaly.enhancer.heads import binary_enhancer_loss
from llm_das_dinomaly.integrations import build_dinomaly_wrapper
from llm_das_dinomaly.search import NormalScoreStats, SearchConfig, score_aware_search
from llm_das_dinomaly.utils import load_yaml_config, require_path, seed_everything


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run LLM-DAS Dinomaly MVTec server pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["check", "hard-samples", "enhancer", "all"], default="all")
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

    output_root = Path(runtime.get("output_root", "outputs/server_mvtec")).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "run_summary.json"

    seed = int(runtime.get("seed", 7))
    device = str(runtime.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    mode = str(runtime.get("mode", "smoke")).lower()
    generator = seed_everything(seed)

    data_root = require_path(data_cfg["root"], kind="DATA_ROOT")
    checkpoint_path = require_path(model_cfg["checkpoint_path"], kind="CHECKPOINT_PATH", must_be_file=True)
    dinomaly_root = require_path(model_cfg["dinomaly_root"], kind="DINOMALY_ROOT")
    categories = data_cfg.get("categories") or ["bottle"]
    limit_per_category = data_cfg.get("limit_per_category")
    if mode == "full":
        categories = list(MVTEC_CLASSES) if categories == ["bottle"] else categories
        limit_per_category = None
    batch_size = int(runtime.get("batch_size", 4))

    dataset = MVTecGoodDataset(data_root, categories=categories, limit_per_category=limit_per_category)
    summary: Dict[str, Any] = {
        "stage": stage,
        "mode": mode,
        "seed": seed,
        "device": device,
        "data_root": str(data_root),
        "categories": categories,
        "num_normal_images": len(dataset),
        "output_root": str(output_root),
        "checkpoint_path": str(checkpoint_path),
    }
    if stage == "check":
        _write_json(summary_path, summary)
        return summary

    wrapper, wrapper_meta = build_dinomaly_wrapper(
        dinomaly_root=dinomaly_root,
        checkpoint_path=checkpoint_path,
        device=device,
        backbone=str(model_cfg.get("backbone", "dinov2reg_vit_base_14")),
        strict=bool(model_cfg.get("strict_checkpoint", False)),
    )
    summary["wrapper"] = wrapper_meta

    hard_cache = output_root / "hard_samples.pt"
    if stage in ("hard-samples", "all"):
        hard_summary = generate_hard_samples(
            wrapper,
            dataset,
            hard_cache,
            batch_size=batch_size,
            device=device,
            generator=generator,
            search_budget=int(hard_cfg.get("search_budget", 4)),
            max_samples=_resolve_max_samples(hard_cfg.get("max_samples", 8), mode=mode),
        )
        summary["hard_samples"] = hard_summary

    if stage in ("enhancer", "all"):
        if not hard_cache.is_file():
            raise FileNotFoundError(f"hard sample cache does not exist: {hard_cache}")
        enhancer_summary = train_enhancer_from_cache(
            hard_cache,
            output_root / "enhancer.pt",
            epochs=int(enhancer_cfg.get("epochs", 1)),
            hidden_dim=int(enhancer_cfg.get("hidden_dim", 128)),
            lr=float(enhancer_cfg.get("lr", 1e-3)),
            seed=seed,
        )
        summary["enhancer"] = enhancer_summary

    _write_json(summary_path, summary)
    return summary


def generate_hard_samples(
    wrapper,
    dataset: MVTecGoodDataset,
    output_path: Path,
    *,
    batch_size: int,
    device: str,
    generator: torch.Generator,
    search_budget: int,
    max_samples: int,
) -> Dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_pil)
    normal_scores = []
    preprocessed = []
    meta_records = []
    with torch.no_grad():
        for images, metas in loader:
            x = wrapper.preprocess(images).to(device)
            scores = wrapper.predict_score(x)
            normal_scores.append(scores.detach().cpu())
            preprocessed.append(x.detach().cpu())
            meta_records.extend(metas)
            if sum(batch.shape[0] for batch in preprocessed) >= max(max_samples, batch_size):
                break

    all_scores = torch.cat(normal_scores, dim=0)
    stats = NormalScoreStats.from_scores(all_scores)
    x_all = torch.cat(preprocessed, dim=0)[:max_samples]
    meta_records = meta_records[: x_all.shape[0]]

    candidates = []
    masks = []
    hardness = []
    base_scores = []
    aux_maps = []
    feature_vectors = []
    labels = []

    for idx in range(x_all.shape[0]):
        x_ref = x_all[idx : idx + 1].to(device)
        candidate = score_aware_search(
            wrapper,
            x_ref,
            stats,
            config=SearchConfig(budget=search_budget),
            generator=None if x_ref.is_cuda else generator,
        )
        candidates.append(candidate.x.detach().cpu())
        masks.append(candidate.mask.detach().cpu())
        hardness.append(candidate.hardness.detach().cpu())

        normal_map = wrapper.predict_map(x_ref)
        normal_score = wrapper.predict_score(x_ref)
        normal_feats = wrapper.extract_features(x_ref, which="encoder")
        synth_map = wrapper.predict_map(candidate.x)
        synth_score = wrapper.predict_score(candidate.x)
        synth_feats = wrapper.extract_features(candidate.x, which="encoder")

        base_scores.extend([normal_score.detach().cpu(), synth_score.detach().cpu()])
        aux_maps.extend([normal_map.detach().cpu(), synth_map.detach().cpu()])
        feature_vectors.extend(
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
        labels.extend([torch.zeros(1), torch.ones(1)])

    tensors = {
        "normal_images": x_all.cpu(),
        "synthetic_images": torch.cat(candidates, dim=0),
        "masks": torch.cat(masks, dim=0),
        "hardness": torch.cat(hardness, dim=0),
        "enhancer_features": torch.cat(feature_vectors, dim=0),
        "labels": torch.cat(labels, dim=0),
        "base_scores": torch.cat(base_scores, dim=0),
        "anomaly_maps": torch.cat(aux_maps, dim=0),
    }
    metadata = {
        "normal_stats": stats.__dict__,
        "num_candidates": int(tensors["synthetic_images"].shape[0]),
        "source_records": meta_records,
        "search_budget": search_budget,
    }
    save_tensor_cache(output_path, tensors, metadata)
    return {
        "cache_path": str(output_path),
        "num_candidates": metadata["num_candidates"],
        "normal_score_mean": stats.mean,
        "normal_score_std": stats.std,
        "accepted_proxy": int((tensors["hardness"] > 0).sum().item()),
    }


def train_enhancer_from_cache(
    cache_path: Path,
    output_path: Path,
    *,
    epochs: int,
    hidden_dim: int,
    lr: float,
    seed: int,
) -> Dict[str, Any]:
    seed_everything(seed)
    payload = torch.load(cache_path, map_location="cpu")
    x = payload["tensors"]["enhancer_features"].float()
    labels = payload["tensors"]["labels"].float()
    head = MapFeatureHead(input_dim=x.shape[1], hidden_dim=hidden_dim)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        logits = head(x)
        loss = binary_enhancer_loss(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    with torch.no_grad():
        aux = torch.sigmoid(head(x))
        fused = fuse_scores(payload["tensors"]["base_scores"].float(), aux)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "input_dim": x.shape[1],
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "losses": losses,
        },
        output_path,
    )
    return {
        "checkpoint_path": str(output_path),
        "input_dim": int(x.shape[1]),
        "epochs": epochs,
        "final_loss": losses[-1],
        "fused_score_mean": float(fused.mean().item()),
    }


def _collate_pil(batch):
    images, metas = zip(*batch)
    return list(images), list(metas)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_max_samples(value: Any, *, mode: str) -> int:
    if mode == "full" and (value is None or str(value).lower() in {"all", "none", "-1"}):
        return 10**12
    return int(value)


if __name__ == "__main__":
    main()
