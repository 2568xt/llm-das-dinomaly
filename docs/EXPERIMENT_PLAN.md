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
2. VisA for generalization.
3. Real-IAD only after MVTec and VisA show stable positive movement.

## Reporting

Use at least three seeds for final claims. Report mean and standard deviation
per dataset, plus paired class-level tests or bootstrap confidence intervals.
Keep synthetic sample visualizations with the numeric logs so that overly easy
or unrealistic synthetic defects can be rejected early.

Keep `OUTPUT_ROOT/metrics/baseline_eval.json`,
`OUTPUT_ROOT/metrics/enhancer_epochs.jsonl`, and
`OUTPUT_ROOT/metrics/final_enhanced_eval.json` with the server logs. The JSONL
file records per-epoch enhancer evaluation and proves that the enhanced
image-level score was evaluated against the original MVTec test split during
training.
