from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from llm_das_dinomaly.pipelines.server_mvtec import run_pipeline


def test_server_pipeline_check_stage_with_fake_mvtec(tmp_path):
    data_root = _fake_mvtec(tmp_path / "mvtec")
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"not-used-in-check-stage")
    dinomaly_root = tmp_path / "Dinomaly"
    (dinomaly_root / "models").mkdir(parents=True)
    (dinomaly_root / "models" / "uad.py").write_text("# fake\n", encoding="utf-8")

    summary = run_pipeline(
        {
            "runtime": {"output_root": str(tmp_path / "out"), "device": "cpu", "batch_size": 1},
            "data": {"root": str(data_root), "categories": ["bottle"], "limit_per_category": 1},
            "model": {
                "dinomaly_root": str(dinomaly_root),
                "checkpoint_path": str(checkpoint),
                "backbone": "dinov2reg_vit_base_14",
            },
        },
        stage="check",
    )

    assert summary["num_normal_images"] == 1
    assert (tmp_path / "out" / "run_summary.json").exists()


def test_server_pipeline_reports_missing_checkpoint(tmp_path):
    data_root = _fake_mvtec(tmp_path / "mvtec")
    dinomaly_root = tmp_path / "Dinomaly"
    dinomaly_root.mkdir()
    with pytest.raises(FileNotFoundError, match="CHECKPOINT_PATH"):
        run_pipeline(
            {
                "runtime": {"output_root": str(tmp_path / "out"), "device": "cpu"},
                "data": {"root": str(data_root), "categories": ["bottle"], "limit_per_category": 1},
                "model": {
                    "dinomaly_root": str(dinomaly_root),
                    "checkpoint_path": str(tmp_path / "missing.pth"),
                },
            },
            stage="check",
        )


def test_full_mode_expands_default_bottle_to_all_classes(tmp_path):
    data_root = tmp_path / "mvtec"
    for category in ("bottle", "cable"):
        good_dir = data_root / category / "train" / "good"
        good_dir.mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(good_dir / "000.png")
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"not-used-in-check-stage")
    dinomaly_root = tmp_path / "Dinomaly"
    (dinomaly_root / "models").mkdir(parents=True)
    (dinomaly_root / "models" / "uad.py").write_text("# fake\n", encoding="utf-8")

    summary = run_pipeline(
        {
            "runtime": {"mode": "full", "output_root": str(tmp_path / "out"), "device": "cpu"},
            "data": {"root": str(data_root), "categories": ["bottle"], "limit_per_category": 1},
            "model": {"dinomaly_root": str(dinomaly_root), "checkpoint_path": str(checkpoint)},
        },
        stage="check",
    )
    assert "cable" in summary["categories"]


def _fake_mvtec(root: Path) -> Path:
    good_dir = root / "bottle" / "train" / "good"
    good_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(good_dir / "000.png")
    return root
