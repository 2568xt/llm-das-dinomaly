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

Use the fast default while iterating:

```bash
RUN_MODE=smoke EVAL_LIMIT_PER_CATEGORY=8 EVAL_BATCH_SIZE=16 EVAL_PIXEL_AUPRO=false \
bash scripts/run_server_mvtec.sh configs/server_mvtec.yaml configs/server_paths.env
```

Use full Dinomaly-style pixel AUPRO only for final parity reports:

```bash
RUN_MODE=full EVAL_BATCH_SIZE=16 EVAL_RESIZE_MASK=256 EVAL_PIXEL_METRICS=true EVAL_PIXEL_AUPRO=true \
python -m llm_das_dinomaly.pipelines.server_mvtec --config configs/server_mvtec.yaml --stage eval
```

For MPDD, use the MVTec-like server dataset layout and keep outputs separate
from MVTec so cache context checks can do their job:

```bash
RUN_MODE=full MAX_SAMPLES=all SEARCH_BUDGET=24 EVAL_BATCH_SIZE=32 \
EVAL_PIXEL_AUPRO=false BASE_TRAIN_IF_MISSING=true \
bash scripts/run_server_mpdd.sh configs/server_mpdd.yaml configs/server_paths_mpdd.env
```

For few-shot rotation experiments, set `FEW_SHOT_ROOT` to a complete dataset
root. This root replaces `DATA_ROOT`, trains a new unified base checkpoint for
the run, expands `train/good` into `0/90/180/270` normal views, and evaluates on
the same root's complete test set:

```bash
FEW_SHOT_ROOT=/path/to/fewshot_root RUN_MODE=full SEARCH_BUDGET=24 EVAL_BATCH_SIZE=32 \
EVAL_PIXEL_AUPRO=false bash scripts/run_server_mvtec.sh configs/server_mvtec.yaml configs/server_paths.env
```

For ViSA, prepare the dataset first with the official `1cls` split, then run the
ViSA server entry point:

```bash
FEW_SHOT_ROOT=/path/to/fewshot_visa_1cls RUN_MODE=full SEARCH_BUDGET=24 EVAL_BATCH_SIZE=32 \
EVAL_PIXEL_AUPRO=false bash scripts/run_server_visa.sh configs/server_visa.yaml configs/server_paths_visa.env
```
