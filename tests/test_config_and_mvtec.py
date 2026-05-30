from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from llm_das_dinomaly.data import MVTecGoodDataset, list_mvtec_train_good
from llm_das_dinomaly.utils import ConfigError, expand_env, require_path


def test_expand_env_supports_defaults_and_required_values():
    cfg = {
        "root": "${DATA_ROOT}",
        "device": "${DEVICE:-cuda}",
        "nested": ["${CATEGORY:-bottle}"],
    }
    out = expand_env(cfg, env={"DATA_ROOT": "/tmp/mvtec"})
    assert out == {"root": "/tmp/mvtec", "device": "cuda", "nested": ["bottle"]}
    with pytest.raises(ConfigError, match="MISSING"):
        expand_env("${MISSING}", env={})


def test_require_path_errors_are_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="DATA_ROOT"):
        require_path(tmp_path / "missing", kind="DATA_ROOT")


def test_mvtec_train_good_indexer(tmp_path):
    good_dir = tmp_path / "bottle" / "train" / "good"
    good_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(good_dir / "000.png")
    Image.new("RGB", (8, 8)).save(good_dir / "001.jpg")
    records = list_mvtec_train_good(tmp_path, categories=["bottle"], limit_per_category=1)
    assert len(records) == 1
    assert records[0].category == "bottle"

    dataset = MVTecGoodDataset(tmp_path, categories=["bottle"])
    image, meta = dataset[0]
    assert image.size == (8, 8)
    assert meta["label"] == 0
