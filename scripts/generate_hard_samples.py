#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_das_dinomaly.data import save_tensor_cache
from llm_das_dinomaly.search import NormalScoreStats, SearchConfig, score_aware_search
from llm_das_dinomaly.utils import seed_everything
from llm_das_dinomaly.wrappers import DinomalyConfig, DinomalyWrapper


class DummyDinomaly(nn.Module):
    def forward(self, x):
        base = F.adaptive_avg_pool2d(x, (4, 4))
        enc = [base, base * 0.5]
        dec = [base.roll(shifts=1, dims=-1), base * 0.45]
        return enc, dec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run with a dummy Dinomaly-like model.")
    parser.add_argument("--config", type=Path, help="Server YAML config for real Dinomaly/MVTec generation.")
    parser.add_argument("--output", type=Path, default=Path("outputs/dry_run_hard.pt"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.config is not None and not args.dry_run:
        from llm_das_dinomaly.pipelines.server_mvtec import run_pipeline
        from llm_das_dinomaly.utils import load_yaml_config

        cfg = load_yaml_config(args.config)
        cfg.setdefault("runtime", {})["output_root"] = str(args.output.parent)
        run_pipeline(cfg, stage="hard-samples")
        return

    if not args.dry_run:
        raise SystemExit("Use --dry-run for dummy mode, or --config for real server mode.")

    generator = seed_everything(args.seed)
    cfg = DinomalyConfig(image_size=32, crop_size=28, patch_size=7, gaussian_kernel=3, device="cpu")
    wrapper = DinomalyWrapper(DummyDinomaly(), cfg)
    x = torch.rand(2, 3, 28, 28, generator=generator)
    normal_scores = wrapper.predict_score(x)
    stats = NormalScoreStats.from_scores(normal_scores)
    candidate = score_aware_search(wrapper, x[:1], stats, config=SearchConfig(budget=4), generator=generator)
    save_tensor_cache(
        args.output,
        {"x": candidate.x, "mask": candidate.mask, "hardness": candidate.hardness},
        {"wrapper": wrapper.metadata(), "seed": args.seed, "normal_stats": stats.__dict__},
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
