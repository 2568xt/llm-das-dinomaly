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

For tracked experiments, put every experiment knob in the env file instead of
leaving it only in a one-off shell prefix. Inline environment overrides still
take precedence for ad-hoc checks, and the runner records their names and final
values in `OUTPUT_ROOT/effective_config.json`.

Optional environment overrides:

- `DATASET`: defaults to `mvtec` for the MVTec runner, `mpdd` for the MPDD
  runner, and `visa` for the ViSA runner.
- `FEW_SHOT_ROOT`: optional complete MVTec-like dataset root. When set, it
  replaces `DATA_ROOT` for base training, hard samples, enhancer training, and
  evaluation.
- `DINOMALY_ROOT`: defaults to `third_party/Dinomaly`.
- `DEVICE`: defaults to `cuda`.
- `MVTEC_CATEGORY`: defaults to `bottle` in smoke mode.
- `MPDD_CATEGORY`: defaults to `bracket_black` in MPDD smoke mode.
- `VISA_CATEGORY`: defaults to `candle` in ViSA smoke mode.
- `RUN_MODE`: defaults to `smoke`.
- `RUN_STAGE`: defaults to `all`; runner scripts also accept the stage as the
  third positional argument, for example `eval`.
- `BATCH_SIZE`: defaults to `16`, matching the official Dinomaly runner's
  batch size for the main server pipeline.
- `BASE_TRAIN_IF_MISSING`: defaults to `false` for MVTec and `true` for MPDD.
- `BASE_FORCE_RETRAIN`: defaults to `false`; set to `true` with an empty
  `CHECKPOINT_PATH` to rebuild the unified base checkpoint.
- `BASE_TOTAL_ITERS`: defaults to `10000` for MPDD base training.
- `BASE_EVAL_INTERVAL`: defaults to `5000` for MPDD base training.
- `FEW_SHOT_BASE_TOTAL_ITERS`: defaults to `2000` and overrides
  `BASE_TOTAL_ITERS` only when `FEW_SHOT_ROOT` is set.
- `FEW_SHOT_BASE_EVAL_INTERVAL`: defaults to `1000` and overrides
  `BASE_EVAL_INTERVAL` only when `FEW_SHOT_ROOT` is set.
- `FEW_SHOT_ENHANCER_EPOCHS`: defaults to `10` and overrides
  `ENHANCER_EPOCHS` only when `FEW_SHOT_ROOT` is set.
- `BASE_CHECKPOINT_DIR`: defaults to `OUTPUT_ROOT/base_checkpoints`.
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
- `ENHANCER_EPOCHS`, `ENHANCER_HIDDEN_DIM`, and `ENHANCER_LR`: override the
  lightweight enhancer training defaults.
- `PROGRESS`: defaults to `true`; set to `false` to disable terminal progress bars.
- `EVAL_ENABLED`: defaults to `true`; set to `false` for train-only smoke runs
  when the data root does not include MVTec `test/` and `ground_truth/`.
- `EVAL_BATCH_SIZE`: defaults to `16`, matching the official Dinomaly eval
  batch size.
- `EVAL_NUM_WORKERS`: defaults to `4`, matching the official Dinomaly test
  dataloader worker count.
- `EVAL_PIXEL_METRICS`: defaults to `true`; set to `false` for image-only
  fast checks.
- `EVAL_PIXEL_AUPRO`: defaults to `false`; set to `true` for full parity with
  Dinomaly's reported pixel AUPRO.
- `EVAL_EPOCH_PIXEL_METRICS`: defaults to `false` so per-epoch enhancer
  evaluations stay image-level and fast.

The run writes `effective_config.json`, `hard_samples.pt`, `enhancer.pt`,
`run_summary.json`, and evaluation metrics under `OUTPUT_ROOT`.
`effective_config.json` contains the resolved YAML config, final typed pipeline
settings, env-file path, env-file keys, and any inline override values seen by
the runner. Hard sample shards are saved under
`OUTPUT_ROOT/hard_samples_shards/`. By default, `hard_samples.pt` is compact and
contains only the tensors needed for enhancer training; image, mask, and map
tensors are stored only when `CACHE_IMAGES=true`. Existing valid caches and
enhancer checkpoints are reused on the next run only when their dataset,
categories, effective data root, few-shot/rotation settings, backbone, and base
checkpoint metadata match the current run. If a previous run left an
unreadable cache file, the runner moves it aside with a `.corrupt` suffix and
rebuilds from any compatible shards that are present.

Metrics are written under `OUTPUT_ROOT/metrics/`, including
`baseline_eval.json`, `enhancer_epochs.jsonl`, per-epoch files such as
`enhancer_epoch_0001.json`, and `final_enhanced_eval.json`. Eval-only runs write
`eval_summary.json` and, when an enhancer checkpoint exists,
`eval_enhanced.json`. During long evaluations, category-level partial results
are also written to matching `*.progress.json` files so server logs can be
tailed without guessing whether eval is still moving:

```bash
python -m llm_das_dinomaly.pipelines.server_mvtec --config configs/server_mvtec.yaml --stage eval
```

Fast smoke evaluation:

```bash
RUN_MODE=smoke EVAL_LIMIT_PER_CATEGORY=8 EVAL_BATCH_SIZE=16 EVAL_PIXEL_AUPRO=false \
bash scripts/run_server_mvtec.sh configs/server_mvtec.yaml configs/server_paths.env
```

Fast eval-only run after `enhancer.pt` exists:

```bash
set -a
source configs/server_paths.env
set +a
RUN_MODE=full EVAL_BATCH_SIZE=16 EVAL_PIXEL_AUPRO=false \
python -m llm_das_dinomaly.pipelines.server_mvtec --config configs/server_mvtec.yaml --stage eval
```

Full parity with Dinomaly's reported pixel AUPRO:

```bash
RUN_MODE=full EVAL_BATCH_SIZE=16 EVAL_RESIZE_MASK=256 EVAL_PIXEL_METRICS=true EVAL_PIXEL_AUPRO=true \
python -m llm_das_dinomaly.pipelines.server_mvtec --config configs/server_mvtec.yaml --stage eval
```

The enhancer changes image-level scores only. Pixel metrics continue to come
from the base Dinomaly anomaly map and are labeled `base_dinomaly_map` in
enhanced summaries. Because `EVAL_ENABLED` defaults to `true`, the server data
root must include MVTec `test/` and `ground_truth/`; set `EVAL_ENABLED=false`
for train-only smoke runs. Per-epoch enhancer evaluation is image-level by
default; set `EVAL_EPOCH_PIXEL_METRICS=true` only when the extra pixel-metric
cost is intentional.

## Few-Shot Rotation Runs

Set `FEW_SHOT_ROOT` when you want a few-shot experiment. The directory must be a
complete dataset root with `<category>/train/good`, `<category>/test/*`, and
`<category>/ground_truth/*`. The runner uses all images present under
`train/good`, expands them as `0/90/180/270` normal views before Dinomaly
preprocessing, trains a new unified base checkpoint for the run, and evaluates
on the same root's full test split.

In few-shot mode, `CHECKPOINT_PATH` is ignored for base-checkpoint reuse. The
DINOv2 pretrained encoder from the official Dinomaly recipe is still used; the
new checkpoint is the Dinomaly reconstruction model for this few-shot root.
Few-shot runs use shorter training defaults than full-data runs:
`FEW_SHOT_BASE_TOTAL_ITERS=2000`, `FEW_SHOT_BASE_EVAL_INTERVAL=1000`, and
`FEW_SHOT_ENHANCER_EPOCHS=10`.

Example MVTec few-shot run. Edit these values into
`configs/server_paths.env`, then run the script without a shell prefix:

```dotenv
FEW_SHOT_ROOT=/path/to/fewshot_mvtec_root
RUN_MODE=full
SEARCH_BUDGET=24
EVAL_BATCH_SIZE=32
EVAL_PIXEL_AUPRO=false
```

```bash
bash scripts/run_server_mvtec.sh configs/server_mvtec.yaml configs/server_paths.env
```

## Server MPDD Run

MPDD is expected to be already arranged in the same layout style as MVTec:
`<category>/train/good`, `<category>/test/*`, and
`<category>/ground_truth/*`. The supported MPDD categories are
`bracket_black`, `bracket_brown`, `bracket_white`, `connector`, `metal_plate`,
and `tubes`.

Prepare a server-local env file:

```bash
cp configs/server_paths_mpdd.example.env configs/server_paths_mpdd.env
# Edit DATA_ROOT, OUTPUT_ROOT, DEVICE, and optional CHECKPOINT_PATH.
```

If `CHECKPOINT_PATH` is empty, the MPDD runner first looks under
`BASE_CHECKPOINT_DIR` and `checkpoints/` for an MPDD unified checkpoint. If none
is found and `BASE_TRAIN_IF_MISSING=true`, it trains a unified Dinomaly-B
checkpoint with the MPDD classes before generating hard samples and training the
enhancer.

Recommended full MPDD run. Put these values in
`configs/server_paths_mpdd.env` for the experiment:

```dotenv
RUN_MODE=full
MAX_SAMPLES=all
SEARCH_BUDGET=24
EVAL_BATCH_SIZE=32
EVAL_PIXEL_AUPRO=false
BASE_TRAIN_IF_MISSING=true
```

```bash
bash scripts/run_server_mpdd.sh configs/server_mpdd.yaml configs/server_paths_mpdd.env
```

For quick path validation:

```dotenv
RUN_MODE=smoke
EVAL_LIMIT_PER_CATEGORY=8
EVAL_PIXEL_METRICS=false
```

```bash
bash scripts/run_server_mpdd.sh configs/server_mpdd.yaml configs/server_paths_mpdd.env
```

## Server ViSA Run

ViSA is expected to be preprocessed with the official Dinomaly/Spot-Diff split
into the MVTec-like `VisA_pytorch/1cls` layout. This repository does not prepare
raw ViSA plus CSV splits inside the server runner.

Prepare a server-local env file:

```bash
cp configs/server_paths_visa.example.env configs/server_paths_visa.env
# Edit DATA_ROOT or FEW_SHOT_ROOT, OUTPUT_ROOT, DEVICE, and optional CHECKPOINT_PATH.
```

Recommended ViSA few-shot run. Put these values in
`configs/server_paths_visa.env` for the experiment:

```dotenv
FEW_SHOT_ROOT=/path/to/fewshot_visa_1cls
RUN_MODE=full
SEARCH_BUDGET=24
EVAL_BATCH_SIZE=32
EVAL_PIXEL_AUPRO=false
```

```bash
bash scripts/run_server_visa.sh configs/server_visa.yaml configs/server_paths_visa.env
```

## Integration Notes

Keep the visual Dinomaly environment separate from the LLM code-generation
environment. Dinomaly's reference stack is older, while modern LLM SDKs usually
move faster. Generated code should target only the documented wrapper API and be
saved with prompt, response, wrapper metadata, random seed, normal-score stats,
and hard-filter thresholds.
