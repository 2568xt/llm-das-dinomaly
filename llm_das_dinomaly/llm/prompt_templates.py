DINOMALY_DETECTOR_API_PROMPT = """\
You are writing detector-aware synthetic anomaly policies for an image UAD
detector. Use only the public DinomalyWrapper API:

- preprocess(images) -> Tensor[B,3,crop,crop]
- predict_score(x) -> Tensor[B]
- predict_map(x) -> Tensor[B,1,H,W]
- extract_features(x, which="encoder") -> list[Tensor[B,C,Hf,Wf]]
- score_candidates(x_ref, x_cands, synth_masks=None) -> dict[str, Tensor]

Generate pure Python functions that create local, bounded, near-boundary
synthetic anomalies. Do not access private model internals.
"""
