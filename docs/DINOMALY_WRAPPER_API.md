# DinomalyWrapper API

`DinomalyWrapper` exposes the detector as a small symbolic interface.

## Configuration

Default values match the report's Dinomaly facts:

- backbone: `dinov2reg_vit_base_14`
- resize: `448`
- center crop: `392`
- patch size: `14`
- target layers: `2..9`
- fused groups: `[[0,1,2,3], [4,5,6,7]]`
- image score: top 1 percent of anomaly-map pixels

For the default base model, the expected fused feature shape is approximately
`[B, 768, 28, 28]` per group because `392 / 14 = 28`.

## Public Methods

```python
x = wrapper.preprocess(images)
features = wrapper.extract_features(x, which="encoder")
amap = wrapper.predict_map(x)
score = wrapper.predict_score(x)
meta = wrapper.score_candidates(x_ref, x_cands, synth_masks=mask)
```

`score_candidates` returns:

- `score_ref`
- `score_cand`
- `score_delta`
- `map`
- `perturb_l1`
- `mask_area`, when masks are supplied
- `mask_overlap`, when masks are supplied

## Contract for Generated Policies

Generated or hand-written policies may call only the public methods above.
They should not inspect `wrapper.model` or depend on Dinomaly private modules.
This keeps generated synthesis code portable across official checkpoints,
future Dinomaly variants, and dummy test models.
