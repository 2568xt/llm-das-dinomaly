from __future__ import annotations

import torch

from llm_das_dinomaly.data import load_tensor_cache, save_tensor_cache
from llm_das_dinomaly.llm import GenerationRecord, save_generation_record


def test_tensor_cache_roundtrip(tmp_path):
    path = tmp_path / "cache.pt"
    save_tensor_cache(path, {"x": torch.ones(2, 3)}, {"seed": 1})
    loaded = load_tensor_cache(path)
    assert loaded["metadata"]["seed"] == 1
    assert loaded["tensors"]["x"].shape == (2, 3)


def test_generation_record_writes_reproducibility_files(tmp_path):
    record = GenerationRecord(
        prompt="p",
        response="r",
        code="def policy(): pass\n",
        model="dummy",
        wrapper_metadata={"config": {}},
        seed=7,
        normal_stats={"mean": 0.0, "std": 1.0},
        thresholds={"z": [1.0, 3.0]},
    )
    out = save_generation_record(record, tmp_path, name="run-001")
    assert (out / "policy.py").exists()
    assert (out / "metadata.json").exists()
