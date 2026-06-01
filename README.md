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
- `HARD_SAMPLE_SHARD_SIZE`: defaults to `32`; hard samples are saved every
  shard so interrupted runs can resume.
- `CACHE_IMAGES`: defaults to `false`; set to `true` only when you need debug
  image/mask/map tensors saved in per-shard files.
- `REGENERATE_HARD_SAMPLES`: defaults to `false`; set to `true` to discard and
  rebuild the hard-sample cache.
- `RETRAIN_ENHANCER`: defaults to `false`; set to `true` to retrain even when
  `enhancer.pt` already exists.
- `PROGRESS`: defaults to `true`; set to `false` to disable terminal progress bars.
- `EVAL_ENABLED`: defaults to `true`; set to `false` for train-only smoke runs
  when the data root does not include MVTec `test/` and `ground_truth/`.

The run writes `hard_samples.pt`, `enhancer.pt`, `run_summary.json`, and
evaluation metrics under `OUTPUT_ROOT`. Hard sample shards are saved under
`OUTPUT_ROOT/hard_samples_shards/`. By default, `hard_samples.pt` is compact and
contains only the tensors needed for enhancer training; image, mask, and map
tensors are stored only when `CACHE_IMAGES=true`. Existing valid caches and
enhancer checkpoints are reused on the next run. If a previous run left an
unreadable cache file, the runner moves it aside with a `.corrupt` suffix and
rebuilds from any compatible shards that are present.

Metrics are written under `OUTPUT_ROOT/metrics/`, including
`baseline_eval.json`, `enhancer_epochs.jsonl`, per-epoch files such as
`enhancer_epoch_0001.json`, and `final_enhanced_eval.json`. Eval-only runs write
`eval_summary.json` and, when an enhancer checkpoint exists,
`eval_enhanced.json`:

```bash
python -m llm_das_dinomaly.pipelines.server_mvtec --config configs/server_mvtec.yaml --stage eval
```

The enhancer changes image-level scores only. Pixel metrics continue to come
from the base Dinomaly anomaly map and are labeled `base_dinomaly_map` in
enhanced summaries. Because `EVAL_ENABLED` defaults to `true`, the server data
root must include MVTec `test/` and `ground_truth/`; set `EVAL_ENABLED=false`
for train-only smoke runs.

## Integration Notes

Keep the visual Dinomaly environment separate from the LLM code-generation
environment. Dinomaly's reference stack is older, while modern LLM SDKs usually
move faster. Generated code should target only the documented wrapper API and be
saved with prompt, response, wrapper metadata, random seed, normal-score stats,
and hard-filter thresholds.
