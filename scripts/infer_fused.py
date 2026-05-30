#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_das_dinomaly.enhancer import fuse_scores


def _parse_scores(text: str) -> torch.Tensor:
    return torch.tensor([float(x) for x in text.split(",")], dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Comma-separated base scores.")
    parser.add_argument("--aux", required=True, help="Comma-separated auxiliary scores.")
    parser.add_argument("--beta", type=float, default=1.0)
    args = parser.parse_args()

    fused = fuse_scores(_parse_scores(args.base), _parse_scores(args.aux), beta=args.beta)
    print(",".join(f"{x:.6f}" for x in fused.tolist()))


if __name__ == "__main__":
    main()
