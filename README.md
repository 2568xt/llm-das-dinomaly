# LLM-DAS Dinomaly

This repository organizes the engineering plan from `deep-research-report.md`
into a small, testable Python package. The goal is not to copy the original
tabular LLM-DAS code into Dinomaly. Instead, the code is structured around the
faithful-transplantation path from the report:

1. Wrap the full Dinomaly detector behind `DinomalyWrapper`.
2. Generate detector-aware synthetic anomalies in image and feature space.
3. Search for near-boundary hard synthetic samples.
4. Train a lightweight auxiliary enhancer.
5. Fuse normalized Dinomaly scores and enhancer scores.

The official Dinomaly model can be plugged into the wrapper later. The current
tree already includes dummy-model tests so the interfaces stay stable before a
full dataset/checkpoint is available.

## Layout

```text
llm_das_dinomaly/
  wrappers/      Dinomaly detector API: score, map, features, candidate scoring.
  synth/         Human-authored baseline mask, image, and feature policies.
  search/        Hardness scoring and score-aware search.
  enhancer/      Feature extraction, lightweight heads, score fusion.
  llm/           Prompt/cache records for later LLM Path B integration.
  data/          Tensor cache helpers for hard sample banks.
  utils/         Shared utility functions.
configs/         Default experiment knobs.
docs/            Architecture and wrapper API notes.
scripts/         Dry-run CLIs and future experiment entry points.
tests/           Unit tests using a dummy Dinomaly-like model.
```

Report-suggested names such as `synth/mask_generators.py` and
`enhancer/map_feature_head.py` are kept as compatibility entry points.

## Quick Check

```bash
python3 -m pytest
python3 scripts/generate_hard_samples.py --dry-run --output outputs/dry_run_hard.pt
python3 scripts/train_enhancer.py --dry-run
```

## Server MVTec Smoke Run

Clone the public repository on a GPU server:

```bash
git clone https://github.com/2568xt/llm-das-dinomaly.git
cd llm-das-dinomaly
git submodule update --init --recursive
```

Install this package in the same environment that can import and run the
official Dinomaly dependencies, then launch a smoke run:

```bash
pip install -e .
cp configs/server_paths.example.env configs/server_paths.env
# Edit configs/server_paths.env with server-local paths.
bash scripts/run_server_mvtec.sh configs/server_mvtec.yaml configs/server_paths.env
```

If `configs/server_paths.env` exists, `scripts/run_server_mvtec.sh` also loads
it automatically, so this shorter command works after the first edit:

```bash
bash scripts/run_server_mvtec.sh
```

Optional environment overrides:

- `DINOMALY_ROOT`: defaults to `third_party/Dinomaly`.
- `DEVICE`: defaults to `cuda`.
- `MVTEC_CATEGORY`: defaults to `bottle` in smoke mode.
- `RUN_MODE`: defaults to `smoke`.
- `MAX_SAMPLES`: defaults to `4`; use `all` for a full run.
- `SEARCH_BUDGET`: defaults to `4`.
- `PROGRESS`: defaults to `true`; set to `false` to disable terminal progress bars.

The run writes `hard_samples.pt`, `enhancer.pt`, and `run_summary.json` under
`OUTPUT_ROOT`.

## Integration Notes

Keep the visual Dinomaly environment separate from the LLM code-generation
environment. Dinomaly's reference stack is older, while modern LLM SDKs usually
move faster. Generated code should target only the documented wrapper API and be
saved with prompt, response, wrapper metadata, random seed, normal-score stats,
and hard-filter thresholds.
