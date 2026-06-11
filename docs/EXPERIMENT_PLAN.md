# Experiment Plan

This plan preserves the report's separation between faithful transplantation
and higher-risk joint training.

## Required Runs

| Run | Change | Evidence to Collect |
|---|---|---|
| Baseline | Official Dinomaly only | Dataset, class, seed, image AUROC/AP/F1, pixel AUROC/AP/F1, AUPRO |
| Image-only | Image-space synthetic anomalies, hard filter, enhancer | Synthetic hit rate, z-band coverage, fused metrics |
| Feature-only | Feature-space synthetic anomalies, enhancer | Group selection, fused metrics |
| Search-only | Image-space proposals plus score-aware search | Acceptance rate, hardness distribution, fused metrics |
| Full faithful | Image + feature + search + enhancer fusion | Per-class paired difference against baseline |
| Joint finetune | Optional rank/map loss branch | Training stability and overfit checks |

## Default Order

1. MVTec for full module bring-up and visual QA.
2. MPDD for a lower-ceiling industrial dataset check with the unified Dinomaly
   base checkpoint path.
3. ViSA for broader generalization with the preprocessed `VisA_pytorch/1cls`
   MVTec-like layout.
4. Real-IAD only after smaller datasets show stable positive movement.

## Reporting

Use at least three seeds for final claims. Report mean and standard deviation
per dataset, plus paired class-level tests or bootstrap confidence intervals.
Keep synthetic sample visualizations with the numeric logs so that overly easy
or unrealistic synthetic defects can be rejected early.

Keep `OUTPUT_ROOT/metrics/baseline_eval.json`,
`OUTPUT_ROOT/metrics/enhancer_epochs.jsonl`, and
`OUTPUT_ROOT/metrics/final_enhanced_eval.json` with the server logs. The JSONL
file records per-epoch enhancer evaluation and proves that the enhanced
image-level score was evaluated against the same root's test split during
training. Per-epoch evaluation is image-level by default so training remains
fast; final evaluation keeps image metrics and pixel AUROC/AP/F1 enabled.

Use the fast default while iterating. Put temporary knobs in the server env file
for the run, then execute the runner:

```dotenv
RUN_MODE=smoke
EVAL_LIMIT_PER_CATEGORY=8
EVAL_BATCH_SIZE=16
EVAL_PIXEL_AUPRO=false
```

```bash
bash scripts/run_server_mvtec.sh configs/server_mvtec.yaml configs/server_paths.env
```

Use full Dinomaly-style pixel AUPRO only for final parity reports:

```dotenv
RUN_MODE=full
EVAL_BATCH_SIZE=16
EVAL_RESIZE_MASK=256
EVAL_PIXEL_METRICS=true
EVAL_PIXEL_AUPRO=true
```

```bash
bash scripts/run_server_mvtec.sh configs/server_mvtec.yaml configs/server_paths.env eval
```

For MPDD, use the MVTec-like server dataset layout and keep outputs separate
from MVTec so cache context checks can do their job:

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

For few-shot rotation experiments, set `FEW_SHOT_ROOT` to a complete dataset
root. This root replaces `DATA_ROOT`, trains a new unified base checkpoint for
the run, expands `train/good` into `0/90/180/270` normal views, and evaluates on
the same root's complete test set. Few-shot runs intentionally use shorter
defaults than full-data training: `FEW_SHOT_BASE_TOTAL_ITERS=2000`,
`FEW_SHOT_BASE_EVAL_INTERVAL=1000`, `FEW_SHOT_ENHANCER_EPOCHS=1`,
`FEW_SHOT_ENHANCER_HIDDEN_DIM=64`, and `FEW_SHOT_ENHANCER_LR=0.0001`.

```dotenv
FEW_SHOT_ROOT=/path/to/fewshot_root
RUN_MODE=full
SEARCH_BUDGET=24
EVAL_BATCH_SIZE=32
EVAL_PIXEL_AUPRO=false
EVAL_FUSION_BETA=0.05
EVAL_FUSION_BETA_SWEEP=0,0.01,0.05,0.1
EVAL_BETA_SELECTION_METRIC=image_auroc
```

```bash
bash scripts/run_server_mvtec.sh configs/server_mvtec.yaml configs/server_paths.env
```

For ViSA, prepare the dataset first with the official `1cls` split, then run the
ViSA server entry point:

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

For enhancer-effectiveness claims, keep the primary beta fixed and use beta
sweep as evidence rather than as an unbiased final selector:

```dotenv
EVAL_FUSION_BETA=0.05
EVAL_FUSION_BETA_SWEEP=0,0.01,0.05,0.1
EVAL_BETA_SELECTION_METRIC=image_auroc
```

After the run, generate the compact evidence table:

```bash
python scripts/summarize_enhancer_evidence.py "$OUTPUT_ROOT"
```

Report `diagnostic_best_beta` only as a diagnostic value selected on the eval
set. For final claims, use the fixed primary beta or add a separate validation
split before choosing beta.

Every runner writes `OUTPUT_ROOT/effective_config.json`, which records the
resolved YAML, final typed settings, env-file path, env-file keys, and any
inline overrides. Use that file with `run_summary.json` when comparing runs.
