# Completion Audit

This repository currently satisfies the code-organization deliverable from
`deep-research-report.md` at the scaffold and interface level.

## Implemented

- `DinomalyWrapper` with `preprocess`, `predict_map`, `predict_score`,
  `extract_features`, and `score_candidates`.
- Human-authored baseline synthesis modules for masks, local image mutations,
  and feature token mixing.
- Hardness scoring with z-band, mask area, overlap, perturbation, and optional
  stability gates.
- Score-aware search that keeps the best candidate under the hard-sample
  scoring rule.
- Lightweight enhancer feature construction, MLP head, binary loss, min-max
  normalizer, and score fusion.
- LLM Path B record templates for prompt/response/code/metadata persistence.
- Config, architecture docs, wrapper API docs, experiment plan, smoke scripts,
  and unit tests with a dummy Dinomaly-like model.
- Server MVTec entrypoint with YAML/env path placeholders, official Dinomaly
  submodule adapter, MVTec train/good indexer, hard-sample cache generation,
  enhancer training, and run summary writing.

## Not Yet Implemented

- Dataset-specific loaders for VisA and Real-IAD.
- Numeric equivalence test against official `evaluation_batch()` on a real
  checkpoint batch.
- LLM-generated policy execution sandbox.

Real execution still requires a GPU server with Dinomaly dependencies, an MVTec
dataset path, and a trained Dinomaly checkpoint path.
