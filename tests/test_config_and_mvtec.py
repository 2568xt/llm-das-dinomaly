from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from llm_das_dinomaly.data import MPDDGoodDataset, MVTecGoodDataset, list_mpdd_train_good, list_mvtec_train_good
from llm_das_dinomaly.data.mpdd import MPDDTestDataset, list_mpdd_test_images
from llm_das_dinomaly.data.mvtec import MVTecTestDataset, list_mvtec_test_images
from llm_das_dinomaly.utils import ConfigError, expand_env, load_yaml_config, require_path


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


def test_server_mpdd_config_allows_empty_checkpoint_env(tmp_path):
    cfg = load_yaml_config(
        Path("configs/server_mpdd.yaml"),
        env={
            "DATA_ROOT": str(tmp_path / "mpdd"),
            "OUTPUT_ROOT": str(tmp_path / "out"),
        },
    )

    assert cfg["data"]["dataset"] == "mpdd"
    assert cfg["model"]["checkpoint_path"] == ""
    assert cfg["base_training"]["train_if_missing"] == "true"


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


def test_mvtec_test_indexer_loads_good_and_defect_masks(tmp_path):
    root = tmp_path / "mvtec"
    good_dir = root / "bottle" / "test" / "good"
    defect_dir = root / "bottle" / "test" / "broken_large"
    mask_dir = root / "bottle" / "ground_truth" / "broken_large"
    good_dir.mkdir(parents=True)
    defect_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(good_dir / "000.png")
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(defect_dir / "001.png")
    Image.new("L", (8, 8), color=255).save(mask_dir / "001_mask.png")

    records = list_mvtec_test_images(root, categories=["bottle"])
    assert [record.label for record in records] == [1, 0]
    assert records[0].mask_path == mask_dir / "001_mask.png"
    assert records[1].mask_path is None

    dataset = MVTecTestDataset(root, categories=["bottle"])
    image, mask, meta = dataset[0]
    assert image.mode == "RGB"
    assert mask.mode == "L"
    assert meta["path"] == str(defect_dir / "001.png")
    assert meta["mask_path"] == str(mask_dir / "001_mask.png")
    assert meta["category"] == "bottle"
    assert meta["defect_type"] == "broken_large"
    assert meta["split"] == "test"
    assert meta["label"] == 1

    _, good_mask, good_meta = dataset[1]
    assert good_mask.getbbox() is None
    assert good_meta["path"] == str(good_dir / "000.png")
    assert good_meta["mask_path"] is None
    assert good_meta["category"] == "bottle"
    assert good_meta["defect_type"] == "good"
    assert good_meta["split"] == "test"
    assert good_meta["label"] == 0


def test_mvtec_test_limit_keeps_good_and_defect_when_possible(tmp_path):
    root = tmp_path / "mvtec"
    good_dir = root / "bottle" / "test" / "good"
    defect_dir = root / "bottle" / "test" / "broken_large"
    mask_dir = root / "bottle" / "ground_truth" / "broken_large"
    good_dir.mkdir(parents=True)
    defect_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(good_dir / "000.png")
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(good_dir / "002.png")
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(defect_dir / "001.png")
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(defect_dir / "003.png")
    Image.new("L", (8, 8), color=255).save(mask_dir / "001_mask.png")
    Image.new("L", (8, 8), color=255).save(mask_dir / "003_mask.png")

    limited_records = list_mvtec_test_images(root, categories=["bottle"], limit_per_category=2)
    assert [record.label for record in limited_records] == [1, 0]


def test_mpdd_indexer_uses_mvtec_like_layout(tmp_path):
    root = tmp_path / "mpdd"
    train_dir = root / "bracket_black" / "train" / "good"
    good_dir = root / "bracket_black" / "test" / "good"
    defect_dir = root / "bracket_black" / "test" / "anomaly"
    mask_dir = root / "bracket_black" / "ground_truth" / "anomaly"
    train_dir.mkdir(parents=True)
    good_dir.mkdir(parents=True)
    defect_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(train_dir / "000.JPG")
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(good_dir / "001.JPG")
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(defect_dir / "002.JPG")
    Image.new("L", (8, 8), color=255).save(mask_dir / "002.png")

    train_records = list_mpdd_train_good(root, categories=["bracket_black"])
    test_records = list_mpdd_test_images(root, categories=["bracket_black"], limit_per_category=2)
    assert len(train_records) == 1
    assert train_records[0].category == "bracket_black"
    assert [record.label for record in test_records] == [1, 0]
    assert test_records[0].mask_path == mask_dir / "002.png"

    train_dataset = MPDDGoodDataset(root, categories=["bracket_black"])
    test_dataset = MPDDTestDataset(root, categories=["bracket_black"])
    image, meta = train_dataset[0]
    assert image.mode == "RGB"
    assert meta["category"] == "bracket_black"
    _, mask, test_meta = test_dataset[0]
    assert mask.mode == "L"
    assert test_meta["defect_type"] == "anomaly"
