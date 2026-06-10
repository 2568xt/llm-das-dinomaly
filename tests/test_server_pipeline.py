from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from llm_das_dinomaly.data import MVTecGoodDataset, load_tensor_cache, save_tensor_cache, save_torch_payload
from llm_das_dinomaly.enhancer import MapFeatureHead
from llm_das_dinomaly.pipelines.server_mvtec import (
    _initialize_trainable_layers_then_move,
    _maybe_disable_legacy_cuda_fusers,
    _try_summarize_enhancer_checkpoint,
    _try_summarize_hard_cache,
    generate_hard_samples,
    run_pipeline,
    train_enhancer_from_cache,
)
from llm_das_dinomaly.wrappers import DinomalyConfig, DinomalyWrapper


class DummyDinomaly(nn.Module):
    def forward(self, x):
        pooled = F.adaptive_avg_pool2d(x, (4, 4))
        return [pooled, pooled * 0.5], [pooled.roll(shifts=1, dims=-1), pooled * 0.25]


def test_cuda_compat_helper_skips_non_cuda_device():
    summary = _maybe_disable_legacy_cuda_fusers("cpu")

    assert summary["device"] == "cpu"
    assert summary["disabled"] == []
    assert summary["errors"] == {}


def test_cuda_compat_helper_disables_available_legacy_fusers(monkeypatch):
    calls = []

    class FakeTorchC:
        def _jit_set_nvfuser_enabled(self, value):
            calls.append(("nvfuser", value))

        def _jit_set_texpr_fuser_enabled(self, value):
            calls.append(("texpr", value))

        def _jit_set_profiling_executor(self, value):
            calls.append(("profiling_executor", value))

        def _jit_set_profiling_mode(self, value):
            calls.append(("profiling_mode", value))

        def _jit_override_can_fuse_on_gpu(self, value):
            calls.append(("gpu_fusion", value))

    monkeypatch.setattr("llm_das_dinomaly.pipelines.server_mvtec.torch._C", FakeTorchC())

    summary = _maybe_disable_legacy_cuda_fusers("cuda:4")

    assert summary["device"] == "cuda:4"
    assert summary["errors"] == {}
    assert summary["unavailable"] == []
    assert set(summary["disabled"]) == {
        "nvfuser",
        "texpr_fuser",
        "profiling_executor",
        "profiling_mode",
        "gpu_fusion_override",
    }
    assert calls == [
        ("nvfuser", False),
        ("texpr", False),
        ("profiling_executor", False),
        ("profiling_mode", False),
        ("gpu_fusion", False),
    ]


def test_base_training_initializes_layers_before_device_move(monkeypatch):
    calls = []

    class TrackingModel(nn.Module):
        def to(self, device):
            calls.append(("to", device))
            return self

    def fake_init(model, trunc_normal):
        calls.append(("init", model, trunc_normal))

    def fake_trunc_normal():
        raise AssertionError("not called directly by this helper")

    monkeypatch.setattr("llm_das_dinomaly.pipelines.server_mvtec._init_trainable_layers", fake_init)
    model = TrackingModel()

    returned = _initialize_trainable_layers_then_move(model, fake_trunc_normal, "cuda:4")

    assert returned is model
    assert calls == [("init", model, fake_trunc_normal), ("to", "cuda:4")]


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


def test_mpdd_full_mode_expands_default_category_to_all_mpdd_classes(tmp_path):
    data_root = _fake_mpdd(tmp_path / "mpdd")
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"not-used-in-check-stage")
    dinomaly_root = tmp_path / "Dinomaly"
    (dinomaly_root / "models").mkdir(parents=True)
    (dinomaly_root / "models" / "uad.py").write_text("# fake\n", encoding="utf-8")

    summary = run_pipeline(
        {
            "runtime": {"mode": "full", "output_root": str(tmp_path / "out"), "device": "cpu"},
            "data": {
                "dataset": "mpdd",
                "root": str(data_root),
                "categories": ["bracket_black"],
                "limit_per_category": 1,
            },
            "model": {"dinomaly_root": str(dinomaly_root), "checkpoint_path": str(checkpoint)},
        },
        stage="check",
    )

    assert summary["dataset"] == "mpdd"
    assert summary["categories"] == [
        "bracket_black",
        "bracket_brown",
        "bracket_white",
        "connector",
        "metal_plate",
        "tubes",
    ]


def test_visa_full_mode_expands_default_category_to_all_visa_classes(tmp_path):
    data_root = _fake_visa(tmp_path / "visa")
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"not-used-in-check-stage")
    dinomaly_root = tmp_path / "Dinomaly"
    (dinomaly_root / "models").mkdir(parents=True)
    (dinomaly_root / "models" / "uad.py").write_text("# fake\n", encoding="utf-8")

    summary = run_pipeline(
        {
            "runtime": {"mode": "full", "output_root": str(tmp_path / "out"), "device": "cpu"},
            "data": {
                "dataset": "visa",
                "root": str(data_root),
                "categories": ["candle"],
                "limit_per_category": 1,
            },
            "model": {"dinomaly_root": str(dinomaly_root), "checkpoint_path": str(checkpoint)},
        },
        stage="check",
    )

    assert summary["dataset"] == "visa"
    assert summary["categories"] == [
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum",
    ]


def test_few_shot_root_replaces_data_root_and_adds_rotation_views(tmp_path):
    few_shot_root = _fake_mvtec(tmp_path / "fewshot", count=2)
    checkpoint = tmp_path / "old.pth"
    checkpoint.write_bytes(b"ignored-in-check-stage")
    dinomaly_root = tmp_path / "Dinomaly"
    (dinomaly_root / "models").mkdir(parents=True)
    (dinomaly_root / "models" / "uad.py").write_text("# fake\n", encoding="utf-8")

    summary = run_pipeline(
        {
            "runtime": {"output_root": str(tmp_path / "out"), "device": "cpu", "progress": False},
            "data": {
                "root": str(tmp_path / "missing-full-root"),
                "few_shot_root": str(few_shot_root),
                "categories": ["bottle"],
                "limit_per_category": 1,
            },
            "model": {"dinomaly_root": str(dinomaly_root), "checkpoint_path": str(checkpoint)},
        },
        stage="check",
    )

    assert summary["data_root"] == str(few_shot_root)
    assert summary["configured_data_root"] == str(tmp_path / "missing-full-root")
    assert summary["few_shot"]["active"] is True
    assert summary["few_shot"]["rotation_angles"] == [0, 90, 180, 270]
    assert summary["num_base_normal_images"] == 2
    assert summary["num_normal_images"] == 8
    assert summary["checkpoint_path"] is None
    assert summary["base_checkpoint"]["source"] == "few_shot_will_train"


def test_mpdd_check_stage_allows_missing_checkpoint_when_training_enabled(tmp_path):
    data_root = _fake_mpdd(tmp_path / "mpdd")
    dinomaly_root = tmp_path / "Dinomaly"
    (dinomaly_root / "models").mkdir(parents=True)
    (dinomaly_root / "models" / "uad.py").write_text("# fake\n", encoding="utf-8")

    summary = run_pipeline(
        {
            "runtime": {"output_root": str(tmp_path / "out"), "device": "cpu", "progress": False},
            "data": {"dataset": "mpdd", "root": str(data_root), "categories": ["bracket_black"]},
            "model": {"dinomaly_root": str(dinomaly_root), "checkpoint_path": ""},
            "base_training": {"train_if_missing": True, "checkpoint_dir": str(tmp_path / "base")},
        },
        stage="check",
    )

    assert summary["checkpoint_path"] is None
    assert summary["base_checkpoint"]["will_train_if_missing"] is True


def test_missing_checkpoint_errors_when_base_training_disabled(tmp_path):
    data_root = _fake_mpdd(tmp_path / "mpdd")
    dinomaly_root = tmp_path / "Dinomaly"
    (dinomaly_root / "models").mkdir(parents=True)
    (dinomaly_root / "models" / "uad.py").write_text("# fake\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="CHECKPOINT_PATH is not set"):
        run_pipeline(
            {
                "runtime": {"output_root": str(tmp_path / "out"), "device": "cpu", "progress": False},
                "data": {"dataset": "mpdd", "root": str(data_root), "categories": ["bracket_black"]},
                "model": {"dinomaly_root": str(dinomaly_root), "checkpoint_path": ""},
                "base_training": {"train_if_missing": False, "checkpoint_dir": str(tmp_path / "base")},
            },
            stage="check",
        )


def test_existing_base_checkpoint_is_found_when_checkpoint_path_is_empty(tmp_path):
    data_root = _fake_mpdd(tmp_path / "mpdd")
    dinomaly_root = tmp_path / "Dinomaly"
    (dinomaly_root / "models").mkdir(parents=True)
    (dinomaly_root / "models" / "uad.py").write_text("# fake\n", encoding="utf-8")
    checkpoint_dir = tmp_path / "base"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "mpdd_unified_dinov2reg_vit_base_14_it10k.pth"
    checkpoint.write_bytes(b"found")

    summary = run_pipeline(
        {
            "runtime": {"output_root": str(tmp_path / "out"), "device": "cpu", "progress": False},
            "data": {"dataset": "mpdd", "root": str(data_root), "categories": ["bracket_black"]},
            "model": {"dinomaly_root": str(dinomaly_root), "checkpoint_path": ""},
            "base_training": {"checkpoint_dir": str(checkpoint_dir), "total_iters": 10000},
        },
        stage="check",
    )

    assert summary["checkpoint_path"] == str(checkpoint)
    assert summary["base_checkpoint"]["source"] == "search"


def test_base_train_stage_invokes_training_when_checkpoint_missing(tmp_path, monkeypatch):
    data_root = _fake_mpdd(tmp_path / "mpdd")
    dinomaly_root = tmp_path / "Dinomaly"
    (dinomaly_root / "models").mkdir(parents=True)
    (dinomaly_root / "models" / "uad.py").write_text("# fake\n", encoding="utf-8")
    calls = []

    def fake_base_train(**kwargs):
        calls.append(kwargs)
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_bytes(b"trained")
        return {
            "checkpoint_path": str(kwargs["output_path"]),
            "trained": True,
            "total_iters": kwargs["total_iters"],
        }

    monkeypatch.setattr("llm_das_dinomaly.pipelines.server_mvtec.train_unified_dinomaly_checkpoint", fake_base_train)
    summary = run_pipeline(
        {
            "runtime": {"output_root": str(tmp_path / "out"), "device": "cpu", "progress": False},
            "data": {"dataset": "mpdd", "root": str(data_root), "categories": ["bracket_black"]},
            "model": {"dinomaly_root": str(dinomaly_root), "checkpoint_path": ""},
            "base_training": {
                "checkpoint_dir": str(tmp_path / "base"),
                "total_iters": 123,
                "eval_interval": 50,
            },
        },
        stage="base-train",
    )

    assert calls
    assert calls[0]["categories"] == ["bracket_black"]
    assert summary["base_checkpoint"]["source"] == "trained"
    assert summary["checkpoint_path"].endswith("mpdd_unified_dinov2reg_vit_base_14_it1k.pth")


def test_all_stage_trains_missing_mpdd_base_before_reusing_matching_artifacts(tmp_path, monkeypatch):
    data_root = _fake_mpdd(tmp_path / "mpdd")
    dinomaly_root = tmp_path / "Dinomaly"
    (dinomaly_root / "models").mkdir(parents=True)
    (dinomaly_root / "models" / "uad.py").write_text("# fake\n", encoding="utf-8")
    output_root = tmp_path / "out"
    checkpoint = tmp_path / "base" / "mpdd_unified_dinov2reg_vit_base_14_it1k.pth"
    context = {
        "dataset": "mpdd",
        "categories": ["bracket_black"],
        "data_root": str(data_root),
        "checkpoint_path": str(checkpoint),
        "backbone": "dinov2reg_vit_base_14",
        "few_shot": False,
        "rotation_angles": [],
    }
    save_tensor_cache(
        output_root / "hard_samples.pt",
        {
            "sample_indices": torch.tensor([0]),
            "hardness": torch.tensor([0.3]),
            "enhancer_features": torch.ones(2, 3),
            "labels": torch.tensor([0.0, 1.0]),
            "base_scores": torch.tensor([0.1, 0.2]),
        },
        {
            "normal_stats": {"mean": 0.1, "std": 0.01},
            "num_candidates": 1,
            "search_budget": 4,
            "cache_images": False,
            "cache_context": context,
        },
    )
    save_torch_payload(
        output_root / "enhancer.pt",
        {
            "state_dict": {},
            "input_dim": 3,
            "hidden_dim": 8,
            "epochs": 1,
            "losses": [0.2],
            "fusion_calibration": {
                "base": {"lo": 0.1, "hi": 0.2},
                "aux": {"lo": 0.3, "hi": 0.4},
            },
            "cache_context": context,
        },
    )

    def fake_base_train(**kwargs):
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_bytes(b"trained")
        return {"checkpoint_path": str(kwargs["output_path"]), "trained": True}

    def fail_build_wrapper(**kwargs):
        raise AssertionError("Dinomaly should not load when matching artifacts are reused")

    monkeypatch.setattr("llm_das_dinomaly.pipelines.server_mvtec.train_unified_dinomaly_checkpoint", fake_base_train)
    monkeypatch.setattr("llm_das_dinomaly.pipelines.server_mvtec.build_dinomaly_wrapper", fail_build_wrapper)
    summary = run_pipeline(
        {
            "runtime": {"output_root": str(output_root), "device": "cpu", "progress": False},
            "data": {"dataset": "mpdd", "root": str(data_root), "categories": ["bracket_black"]},
            "model": {"dinomaly_root": str(dinomaly_root), "checkpoint_path": ""},
            "base_training": {
                "train_if_missing": True,
                "checkpoint_dir": str(tmp_path / "base"),
                "total_iters": 123,
            },
            "hard_samples": {"search_budget": 4, "max_samples": 1},
            "enhancer": {"hidden_dim": 8},
            "evaluation": {"enabled": False},
        },
        stage="all",
    )

    assert summary["base_checkpoint"]["source"] == "trained"
    assert summary["hard_samples"]["reused"] is True
    assert summary["enhancer"]["reused"] is True


def test_few_shot_all_stage_trains_new_base_and_generates_rotated_hard_samples(tmp_path, monkeypatch):
    data_root = _fake_mvtec(tmp_path / "full", count=1)
    few_shot_root = _fake_mvtec(tmp_path / "fewshot", count=2)
    old_checkpoint = tmp_path / "old.pth"
    old_checkpoint.write_bytes(b"ignored")
    dinomaly_root = tmp_path / "Dinomaly"
    (dinomaly_root / "models").mkdir(parents=True)
    (dinomaly_root / "models" / "uad.py").write_text("# fake\n", encoding="utf-8")
    base_calls = []
    enhancer_calls = []

    def fake_base_train(**kwargs):
        base_calls.append(kwargs)
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_bytes(b"few-shot-trained")
        return {"checkpoint_path": str(kwargs["output_path"]), "trained": True}

    def fake_train_enhancer(cache_path, output_path, **kwargs):
        enhancer_calls.append(kwargs)
        return {"checkpoint_path": str(output_path), "epochs": kwargs["epochs"], "reused": False}

    def build_dummy_wrapper(**kwargs):
        return _dummy_wrapper(), {"backend": "dummy"}

    monkeypatch.setattr("llm_das_dinomaly.pipelines.server_mvtec.train_unified_dinomaly_checkpoint", fake_base_train)
    monkeypatch.setattr("llm_das_dinomaly.pipelines.server_mvtec.train_enhancer_from_cache", fake_train_enhancer)
    monkeypatch.setattr("llm_das_dinomaly.pipelines.server_mvtec.build_dinomaly_wrapper", build_dummy_wrapper)
    summary = run_pipeline(
        {
            "runtime": {"output_root": str(tmp_path / "out"), "device": "cpu", "progress": False, "batch_size": 4},
            "data": {
                "root": str(data_root),
                "few_shot_root": str(few_shot_root),
                "categories": ["bottle"],
                "limit_per_category": 1,
            },
            "model": {"dinomaly_root": str(dinomaly_root), "checkpoint_path": str(old_checkpoint)},
            "base_training": {"checkpoint_dir": str(tmp_path / "base"), "total_iters": 999, "eval_interval": 333},
            "few_shot_training": {"base_total_iters": 12, "base_eval_interval": 6, "enhancer_epochs": 2},
            "hard_samples": {"search_budget": 1, "max_samples": 1, "shard_size": 4},
            "enhancer": {"epochs": 99, "hidden_dim": 4},
            "evaluation": {"enabled": False},
        },
        stage="all",
    )

    assert base_calls
    assert base_calls[0]["data_root"] == few_shot_root
    assert base_calls[0]["total_iters"] == 12
    assert base_calls[0]["eval_interval"] == 6
    assert enhancer_calls[0]["epochs"] == 2
    assert summary["few_shot"]["training_budget"] == {
        "base_total_iters": 12,
        "base_eval_interval": 6,
        "enhancer_epochs": 2,
    }
    assert summary["base_checkpoint"]["source"] == "few_shot_trained"
    assert summary["base_checkpoint"]["ignored_checkpoint_path"] == str(old_checkpoint)
    assert summary["hard_samples"]["num_candidates"] == 8
    assert summary["hard_samples"]["reused"] is False
    payload = load_tensor_cache(tmp_path / "out" / "hard_samples.pt")
    rotations = [record["rotation_degrees"] for record in payload["metadata"]["source_records"]]
    assert rotations == [0, 90, 180, 270, 0, 90, 180, 270]
    assert payload["metadata"]["cache_context"]["few_shot"] is True
    assert payload["metadata"]["cache_context"]["rotation_angles"] == [0, 90, 180, 270]


def test_hard_sample_generation_writes_compact_shards_by_default(tmp_path):
    data_root = _fake_mvtec(tmp_path / "mvtec", count=3)
    dataset = MVTecGoodDataset(data_root, categories=["bottle"])
    output_path = tmp_path / "out" / "hard_samples.pt"

    summary = generate_hard_samples(
        _dummy_wrapper(),
        dataset,
        output_path,
        batch_size=2,
        device="cpu",
        generator=torch.Generator().manual_seed(11),
        search_budget=1,
        max_samples=3,
        shard_size=2,
        show_progress=False,
    )

    payload = load_tensor_cache(output_path)
    assert summary["num_candidates"] == 3
    assert summary["generated_candidates"] == 3
    assert summary["cache_images"] is False
    assert {"sample_indices", "hardness", "enhancer_features", "labels", "base_scores"} <= set(
        payload["tensors"]
    )
    assert "normal_images" not in payload["tensors"]
    assert "synthetic_images" not in payload["tensors"]
    assert len(list((tmp_path / "out" / "hard_samples_shards").glob("shard-*.pt"))) == 2


def test_hard_sample_generation_resumes_from_existing_shards(tmp_path):
    data_root = _fake_mvtec(tmp_path / "mvtec", count=3)
    dataset = MVTecGoodDataset(data_root, categories=["bottle"])
    output_path = tmp_path / "out" / "hard_samples.pt"

    generate_hard_samples(
        _dummy_wrapper(),
        dataset,
        output_path,
        batch_size=1,
        device="cpu",
        generator=torch.Generator().manual_seed(12),
        search_budget=1,
        max_samples=2,
        shard_size=2,
        show_progress=False,
    )
    summary = generate_hard_samples(
        _dummy_wrapper(),
        dataset,
        output_path,
        batch_size=1,
        device="cpu",
        generator=torch.Generator().manual_seed(12),
        search_budget=1,
        max_samples=3,
        shard_size=2,
        show_progress=False,
    )

    payload = load_tensor_cache(output_path)
    assert summary["resumed_from_shards"] is True
    assert summary["generated_candidates"] == 1
    assert payload["metadata"]["num_candidates"] == 3
    assert payload["tensors"]["sample_indices"].tolist() == [0, 1, 2]


def test_server_pipeline_reuses_existing_cache_and_enhancer(tmp_path, monkeypatch):
    data_root = _fake_mvtec(tmp_path / "mvtec")
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"not-used-when-cache-is-reused")
    dinomaly_root = tmp_path / "Dinomaly"
    dinomaly_root.mkdir()
    output_root = tmp_path / "out"
    save_tensor_cache(
        output_root / "hard_samples.pt",
        {
            "sample_indices": torch.tensor([0]),
            "hardness": torch.tensor([0.3]),
            "enhancer_features": torch.ones(2, 3),
            "labels": torch.tensor([0.0, 1.0]),
            "base_scores": torch.tensor([0.1, 0.2]),
        },
        {
            "normal_stats": {"mean": 0.1, "std": 0.01},
            "num_candidates": 1,
            "search_budget": 4,
            "cache_images": False,
            "cache_context": _expected_cache_context(data_root, checkpoint),
        },
    )
    save_torch_payload(
        output_root / "enhancer.pt",
        {
            "state_dict": {},
            "input_dim": 3,
            "hidden_dim": 8,
            "epochs": 1,
            "losses": [0.2],
            "fusion_calibration": {
                "base": {"lo": 0.1, "hi": 0.2},
                "aux": {"lo": 0.3, "hi": 0.4},
            },
            "cache_context": _expected_cache_context(data_root, checkpoint),
        },
    )

    def fail_build_wrapper(**kwargs):
        raise AssertionError("Dinomaly should not load when cache and enhancer are reused")

    monkeypatch.setattr("llm_das_dinomaly.pipelines.server_mvtec.build_dinomaly_wrapper", fail_build_wrapper)
    summary = run_pipeline(
        {
            "runtime": {"output_root": str(output_root), "device": "cpu", "progress": False},
            "data": {"root": str(data_root), "categories": ["bottle"], "limit_per_category": 1},
            "model": {"dinomaly_root": str(dinomaly_root), "checkpoint_path": str(checkpoint)},
            "hard_samples": {"search_budget": 4, "max_samples": 1},
            "enhancer": {"hidden_dim": 8},
            "evaluation": {"enabled": False},
        },
        stage="all",
    )

    assert summary["hard_samples"]["reused"] is True
    assert summary["enhancer"]["reused"] is True
    assert summary["enhancer"]["fusion_calibration"]["base"] == {"lo": 0.1, "hi": 0.2}
    assert "wrapper" not in summary


def test_cache_context_mismatch_rejects_hard_cache_and_enhancer(tmp_path):
    cache_path = tmp_path / "hard_samples.pt"
    enhancer_path = tmp_path / "enhancer.pt"
    save_tensor_cache(
        cache_path,
        {
            "sample_indices": torch.tensor([0]),
            "hardness": torch.tensor([0.3]),
            "enhancer_features": torch.ones(2, 3),
            "labels": torch.tensor([0.0, 1.0]),
            "base_scores": torch.tensor([0.1, 0.2]),
        },
        {
            "normal_stats": {"mean": 0.1, "std": 0.01},
            "num_candidates": 1,
            "search_budget": 4,
            "cache_images": False,
            "cache_context": {
                "dataset": "mpdd",
                "categories": ["bracket_black"],
                "data_root": "/mpdd",
                "checkpoint_path": "/ckpt/mpdd.pth",
                "backbone": "dinov2reg_vit_base_14",
                "few_shot": False,
                "rotation_angles": [],
            },
        },
    )
    save_torch_payload(
        enhancer_path,
        {
            "state_dict": {},
            "input_dim": 3,
            "hidden_dim": 8,
            "epochs": 1,
            "cache_context": {
                "dataset": "mpdd",
                "categories": ["bracket_black"],
                "data_root": "/mpdd",
                "checkpoint_path": "/ckpt/mpdd.pth",
                "backbone": "dinov2reg_vit_base_14",
                "few_shot": False,
                "rotation_angles": [],
            },
        },
    )

    expected = {
        "dataset": "mvtec",
        "categories": ["bottle"],
        "data_root": "/mvtec",
        "checkpoint_path": "/ckpt/mvtec.pth",
        "backbone": "dinov2reg_vit_base_14",
        "few_shot": False,
        "rotation_angles": [],
    }
    hard_summary = _try_summarize_hard_cache(
        cache_path,
        target_samples=1,
        search_budget=4,
        cache_images=False,
        cache_context=expected,
    )
    enhancer_summary = _try_summarize_enhancer_checkpoint(enhancer_path, cache_context=expected)

    assert hard_summary is None
    assert enhancer_summary is None


def test_server_pipeline_eval_stage_writes_metrics(tmp_path, monkeypatch):
    data_root = _fake_mvtec(tmp_path / "mvtec")
    _fake_mvtec_test(data_root)
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"not-used-by-dummy-wrapper")
    dinomaly_root = tmp_path / "Dinomaly"
    dinomaly_root.mkdir()

    def build_dummy_wrapper(**kwargs):
        return _dummy_wrapper(), {"backend": "dummy"}

    monkeypatch.setattr("llm_das_dinomaly.pipelines.server_mvtec.build_dinomaly_wrapper", build_dummy_wrapper)
    summary = run_pipeline(
        {
            "runtime": {"output_root": str(tmp_path / "out"), "device": "cpu", "progress": False},
            "data": {"root": str(data_root), "categories": ["bottle"], "limit_per_category": 1},
            "model": {"dinomaly_root": str(dinomaly_root), "checkpoint_path": str(checkpoint)},
            "evaluation": {
                "batch_size": 1,
                "num_workers": 0,
                "resize_mask": 16,
                "limit_per_category": "all",
                "pixel_metrics": False,
                "pixel_aupro": False,
            },
        },
        stage="eval",
    )

    assert summary["evaluation"]["baseline"]["mean"]["baseline"]["num_categories"] == 1
    assert summary["evaluation"]["baseline"]["categories"]["bottle"]["baseline"]["pixel_auroc"] is None
    assert (tmp_path / "out" / "metrics" / "eval_summary.json").exists()
    assert (tmp_path / "out" / "metrics" / "eval_summary.progress.json").exists()


def test_server_pipeline_enhanced_eval_requires_saved_calibration(tmp_path, monkeypatch):
    data_root = _fake_mvtec(tmp_path / "mvtec")
    _fake_mvtec_test(data_root)
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"not-used-by-dummy-wrapper")
    dinomaly_root = tmp_path / "Dinomaly"
    dinomaly_root.mkdir()
    output_root = tmp_path / "out"
    head = MapFeatureHead(input_dim=18, hidden_dim=4)
    save_torch_payload(
        output_root / "enhancer.pt",
        {
            "state_dict": head.state_dict(),
            "input_dim": 18,
            "hidden_dim": 4,
            "epochs": 1,
            "losses": [0.1],
            "cache_context": _expected_cache_context(data_root, checkpoint),
        },
    )

    def build_dummy_wrapper(**kwargs):
        return _dummy_wrapper(), {"backend": "dummy"}

    monkeypatch.setattr("llm_das_dinomaly.pipelines.server_mvtec.build_dinomaly_wrapper", build_dummy_wrapper)
    with pytest.raises(ValueError, match="requires fusion_calibration"):
        run_pipeline(
            {
                "runtime": {"output_root": str(output_root), "device": "cpu", "progress": False},
                "data": {"root": str(data_root), "categories": ["bottle"], "limit_per_category": 1},
                "model": {"dinomaly_root": str(dinomaly_root), "checkpoint_path": str(checkpoint)},
                "evaluation": {"batch_size": 1, "resize_mask": 16, "limit_per_category": "all"},
            },
            stage="eval",
        )


def test_server_pipeline_all_stage_writes_training_evaluation_metrics(tmp_path, monkeypatch):
    data_root = _fake_mvtec(tmp_path / "mvtec")
    _fake_mvtec_test(data_root)
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"not-used-by-dummy-wrapper")
    dinomaly_root = tmp_path / "Dinomaly"
    dinomaly_root.mkdir()
    output_root = tmp_path / "out"
    save_tensor_cache(
        output_root / "hard_samples.pt",
        {
            "sample_indices": torch.tensor([0]),
            "hardness": torch.tensor([0.3]),
            "enhancer_features": torch.ones(2, 3),
            "labels": torch.tensor([0.0, 1.0]),
            "base_scores": torch.tensor([0.1, 0.2]),
        },
        {
            "normal_stats": {"mean": 0.1, "std": 0.01},
            "num_candidates": 1,
            "search_budget": 4,
            "cache_images": False,
            "cache_context": _expected_cache_context(data_root, checkpoint),
        },
    )

    def build_dummy_wrapper(**kwargs):
        return _dummy_wrapper(), {"backend": "dummy"}

    def fake_train(cache_path, output_path, *, eval_callback=None, hidden_dim=4, **kwargs):
        head = MapFeatureHead(input_dim=18, hidden_dim=hidden_dim)
        fusion_calibration = {
            "base": {"lo": 0.0, "hi": 1.0},
            "aux": {"lo": 0.0, "hi": 1.0},
        }
        epoch_records = [
            eval_callback(head=head, epoch=1, loss=0.123, fusion_calibration=fusion_calibration)
        ]
        save_torch_payload(
            output_path,
            {
                "state_dict": head.state_dict(),
                "input_dim": 18,
                "hidden_dim": hidden_dim,
                "epochs": 1,
                "losses": [0.123],
                "fusion_calibration": fusion_calibration,
                "cache_context": _expected_cache_context(data_root, checkpoint),
            },
        )
        return {
            "checkpoint_path": str(output_path),
            "reused": False,
            "input_dim": 18,
            "hidden_dim": hidden_dim,
            "epochs": 1,
            "final_loss": 0.123,
            "fusion_calibration": fusion_calibration,
            "epoch_evaluations": epoch_records,
        }

    monkeypatch.setattr("llm_das_dinomaly.pipelines.server_mvtec.build_dinomaly_wrapper", build_dummy_wrapper)
    monkeypatch.setattr("llm_das_dinomaly.pipelines.server_mvtec.train_enhancer_from_cache", fake_train)

    stale_jsonl = output_root / "metrics" / "enhancer_epochs.jsonl"
    stale_jsonl.parent.mkdir(parents=True)
    stale_jsonl.write_text('{"epoch": 99, "stale": true}\n', encoding="utf-8")

    summary = run_pipeline(
        {
            "runtime": {"output_root": str(output_root), "device": "cpu", "progress": False},
            "data": {"root": str(data_root), "categories": ["bottle"], "limit_per_category": 1},
            "model": {"dinomaly_root": str(dinomaly_root), "checkpoint_path": str(checkpoint)},
            "hard_samples": {"search_budget": 4, "max_samples": 1},
            "enhancer": {"hidden_dim": 4, "retrain": True},
            "evaluation": {"batch_size": 1, "resize_mask": 16, "limit_per_category": "all"},
        },
        stage="all",
    )

    metrics_dir = output_root / "metrics"
    assert summary["evaluation"]["baseline"]["mean"]["baseline"]["num_categories"] == 1
    assert summary["evaluation"]["final_enhanced"]["mean"]["enhanced"]["num_categories"] == 1
    assert (
        summary["enhancer"]["epoch_evaluations"][0]["metrics"]["categories"]["bottle"]["baseline"]["pixel_auroc"]
        is None
    )
    assert (
        summary["evaluation"]["final_enhanced"]["categories"]["bottle"]["baseline"]["pixel_auroc"]
        is not None
    )
    assert (metrics_dir / "baseline_eval.json").exists()
    assert (metrics_dir / "baseline_eval.progress.json").exists()
    assert (metrics_dir / "enhancer_epoch_0001.json").exists()
    assert (metrics_dir / "enhancer_epochs.jsonl").exists()
    epoch_records = [
        line for line in (metrics_dir / "enhancer_epochs.jsonl").read_text(encoding="utf-8").splitlines() if line
    ]
    assert len(epoch_records) == 1
    assert '"epoch": 1' in epoch_records[0]
    assert "stale" not in epoch_records[0]
    assert (metrics_dir / "final_enhanced_eval.json").exists()


def test_server_pipeline_reused_real_enhancer_writes_final_enhanced_eval(tmp_path, monkeypatch):
    data_root = _fake_mvtec(tmp_path / "mvtec")
    _fake_mvtec_test(data_root)
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"not-used-by-dummy-wrapper")
    dinomaly_root = tmp_path / "Dinomaly"
    dinomaly_root.mkdir()
    output_root = tmp_path / "out"
    save_tensor_cache(
        output_root / "hard_samples.pt",
        {
            "sample_indices": torch.tensor([0]),
            "hardness": torch.tensor([0.3]),
            "enhancer_features": torch.ones(2, 18),
            "labels": torch.tensor([0.0, 1.0]),
            "base_scores": torch.tensor([0.1, 0.2]),
        },
        {
            "normal_stats": {"mean": 0.1, "std": 0.01},
            "num_candidates": 1,
            "search_budget": 4,
            "cache_images": False,
            "cache_context": _expected_cache_context(data_root, checkpoint),
        },
    )
    head = MapFeatureHead(input_dim=18, hidden_dim=4)
    save_torch_payload(
        output_root / "enhancer.pt",
        {
            "state_dict": head.state_dict(),
            "input_dim": 18,
            "hidden_dim": 4,
            "epochs": 1,
            "losses": [0.2],
            "fusion_calibration": {
                "base": {"lo": 0.0, "hi": 1.0},
                "aux": {"lo": 0.0, "hi": 1.0},
            },
            "cache_context": _expected_cache_context(data_root, checkpoint),
        },
    )

    def build_dummy_wrapper(**kwargs):
        return _dummy_wrapper(), {"backend": "dummy"}

    def fail_train(*args, **kwargs):
        raise AssertionError("existing enhancer should be reused")

    monkeypatch.setattr("llm_das_dinomaly.pipelines.server_mvtec.build_dinomaly_wrapper", build_dummy_wrapper)
    monkeypatch.setattr("llm_das_dinomaly.pipelines.server_mvtec.train_enhancer_from_cache", fail_train)
    summary = run_pipeline(
        {
            "runtime": {"output_root": str(output_root), "device": "cpu", "progress": False},
            "data": {"root": str(data_root), "categories": ["bottle"], "limit_per_category": 1},
            "model": {"dinomaly_root": str(dinomaly_root), "checkpoint_path": str(checkpoint)},
            "hard_samples": {"search_budget": 4, "max_samples": 1},
            "enhancer": {"hidden_dim": 4},
            "evaluation": {"batch_size": 1, "resize_mask": 16, "limit_per_category": "all"},
        },
        stage="all",
    )

    assert summary["enhancer"]["reused"] is True
    assert summary["evaluation"]["final_enhanced"]["mean"]["enhanced"]["num_categories"] == 1
    assert (output_root / "metrics" / "final_enhanced_eval.json").exists()


def test_server_pipeline_retrains_legacy_enhancer_missing_calibration(tmp_path, monkeypatch):
    data_root = _fake_mvtec(tmp_path / "mvtec")
    _fake_mvtec_test(data_root)
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"not-used-by-dummy-wrapper")
    dinomaly_root = tmp_path / "Dinomaly"
    dinomaly_root.mkdir()
    output_root = tmp_path / "out"
    save_tensor_cache(
        output_root / "hard_samples.pt",
        {
            "sample_indices": torch.tensor([0]),
            "hardness": torch.tensor([0.3]),
            "enhancer_features": torch.ones(2, 18),
            "labels": torch.tensor([0.0, 1.0]),
            "base_scores": torch.tensor([0.1, 0.2]),
        },
        {
            "normal_stats": {"mean": 0.1, "std": 0.01},
            "num_candidates": 1,
            "search_budget": 4,
            "cache_images": False,
            "cache_context": _expected_cache_context(data_root, checkpoint),
        },
    )
    head = MapFeatureHead(input_dim=18, hidden_dim=4)
    save_torch_payload(
        output_root / "enhancer.pt",
        {
            "state_dict": head.state_dict(),
            "input_dim": 18,
            "hidden_dim": 4,
                "epochs": 1,
                "losses": [0.2],
                "cache_context": _expected_cache_context(data_root, checkpoint),
        },
    )

    def build_dummy_wrapper(**kwargs):
        return _dummy_wrapper(), {"backend": "dummy"}

    def fake_train(cache_path, output_path, *, hidden_dim=4, eval_callback=None, **kwargs):
        replacement = MapFeatureHead(input_dim=18, hidden_dim=hidden_dim)
        fusion_calibration = {
            "base": {"lo": 0.0, "hi": 1.0},
            "aux": {"lo": 0.0, "hi": 1.0},
        }
        save_torch_payload(
            output_path,
            {
                "state_dict": replacement.state_dict(),
                "input_dim": 18,
                "hidden_dim": hidden_dim,
                "epochs": 1,
                "losses": [0.1],
                "fusion_calibration": fusion_calibration,
                "cache_context": _expected_cache_context(data_root, checkpoint),
            },
        )
        return {
            "checkpoint_path": str(output_path),
            "reused": False,
            "input_dim": 18,
            "hidden_dim": hidden_dim,
            "epochs": 1,
            "final_loss": 0.1,
            "fusion_calibration": fusion_calibration,
        }

    monkeypatch.setattr("llm_das_dinomaly.pipelines.server_mvtec.build_dinomaly_wrapper", build_dummy_wrapper)
    monkeypatch.setattr("llm_das_dinomaly.pipelines.server_mvtec.train_enhancer_from_cache", fake_train)
    summary = run_pipeline(
        {
            "runtime": {"output_root": str(output_root), "device": "cpu", "progress": False},
            "data": {"root": str(data_root), "categories": ["bottle"], "limit_per_category": 1},
            "model": {"dinomaly_root": str(dinomaly_root), "checkpoint_path": str(checkpoint)},
            "hard_samples": {"search_budget": 4, "max_samples": 1},
            "enhancer": {"hidden_dim": 4},
            "evaluation": {"batch_size": 1, "resize_mask": 16, "limit_per_category": "all"},
        },
        stage="all",
    )

    assert summary["enhancer"]["reused"] is False
    assert "fusion_calibration" in summary["enhancer"]
    assert summary["evaluation"]["final_enhanced"]["mean"]["enhanced"]["num_categories"] == 1


def test_train_enhancer_records_epoch_eval_callback(tmp_path):
    cache_path = tmp_path / "hard_samples.pt"
    save_tensor_cache(
        cache_path,
        {
            "enhancer_features": torch.randn(4, 3),
            "labels": torch.tensor([0.0, 1.0, 0.0, 1.0]),
            "base_scores": torch.tensor([[0.1], [0.8], [0.2], [0.9]]),
        },
        {"type": "hard_samples"},
    )
    calls = []

    def eval_callback(*, head, epoch, loss, fusion_calibration):
        calls.append((head, epoch, loss, fusion_calibration["base"]))
        return {"epoch": epoch, "mean": {"enhanced": {"image_auroc": 1.0}}}

    summary = train_enhancer_from_cache(
        cache_path,
        tmp_path / "enhancer.pt",
        epochs=2,
        hidden_dim=4,
        lr=1e-3,
        seed=3,
        show_progress=False,
        eval_callback=eval_callback,
    )

    assert [call[1] for call in calls] == [1, 2]
    assert len(summary["epoch_evaluations"]) == 2
    assert isinstance(summary["fused_score_mean"], float)
    assert torch.isfinite(torch.tensor(summary["fused_score_mean"]))
    payload = torch.load(tmp_path / "enhancer.pt", map_location="cpu")
    assert "fusion_calibration" in payload


def test_train_enhancer_rejects_mismatched_cache_lengths(tmp_path):
    cache_path = tmp_path / "hard_samples.pt"
    save_tensor_cache(
        cache_path,
        {
            "enhancer_features": torch.randn(4, 3),
            "labels": torch.tensor([0.0, 1.0, 0.0, 1.0]),
            "base_scores": torch.tensor([0.1, 0.8, 0.2]),
        },
        {"type": "hard_samples"},
    )

    with pytest.raises(ValueError, match="enhancer cache tensor counts must align"):
        train_enhancer_from_cache(
            cache_path,
            tmp_path / "enhancer.pt",
            epochs=1,
            hidden_dim=4,
            lr=1e-3,
            seed=3,
            show_progress=False,
        )


def _fake_mvtec(root: Path, *, count: int = 1) -> Path:
    good_dir = root / "bottle" / "train" / "good"
    good_dir.mkdir(parents=True)
    for idx in range(count):
        Image.new("RGB", (8 + idx, 8 + idx)).save(good_dir / f"{idx:03d}.png")
    return root


def _fake_mpdd(root: Path, *, count: int = 1) -> Path:
    for category in ("bracket_black", "bracket_brown"):
        good_dir = root / category / "train" / "good"
        test_good_dir = root / category / "test" / "good"
        defect_dir = root / category / "test" / "anomaly"
        mask_dir = root / category / "ground_truth" / "anomaly"
        good_dir.mkdir(parents=True)
        test_good_dir.mkdir(parents=True)
        defect_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        for idx in range(count):
            Image.new("RGB", (8 + idx, 8 + idx)).save(good_dir / f"{idx:03d}.JPG")
        Image.new("RGB", (8, 8)).save(test_good_dir / "000.JPG")
        Image.new("RGB", (8, 8), color=(255, 255, 255)).save(defect_dir / "001.JPG")
        Image.new("L", (8, 8), color=255).save(mask_dir / "001.png")
    return root


def _fake_visa(root: Path, *, count: int = 1) -> Path:
    for category in (
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum",
    ):
        good_dir = root / category / "train" / "good"
        test_good_dir = root / category / "test" / "good"
        defect_dir = root / category / "test" / "bad"
        mask_dir = root / category / "ground_truth" / "bad"
        good_dir.mkdir(parents=True)
        test_good_dir.mkdir(parents=True)
        defect_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        for idx in range(count):
            Image.new("RGB", (8 + idx, 8 + idx)).save(good_dir / f"{idx:03d}.JPG")
        Image.new("RGB", (8, 8)).save(test_good_dir / "000.JPG")
        Image.new("RGB", (8, 8), color=(255, 255, 255)).save(defect_dir / "001.JPG")
        Image.new("L", (8, 8), color=255).save(mask_dir / "001.png")
    return root


def _fake_mvtec_test(root: Path) -> Path:
    good_dir = root / "bottle" / "test" / "good"
    defect_dir = root / "bottle" / "test" / "broken_large"
    mask_dir = root / "bottle" / "ground_truth" / "broken_large"
    good_dir.mkdir(parents=True)
    defect_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(good_dir / "000.png")
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(defect_dir / "001.png")
    Image.new("L", (8, 8), color=255).save(mask_dir / "001_mask.png")
    return root


def _expected_cache_context(data_root: Path, checkpoint: Path, *, categories=None):
    return {
        "dataset": "mvtec",
        "categories": list(categories or ["bottle"]),
        "data_root": str(data_root),
        "checkpoint_path": str(checkpoint),
        "backbone": "dinov2reg_vit_base_14",
        "few_shot": False,
        "rotation_angles": [],
    }


def _dummy_wrapper() -> DinomalyWrapper:
    return DinomalyWrapper(
        DummyDinomaly(),
        DinomalyConfig(image_size=32, crop_size=28, patch_size=7, gaussian_kernel=3),
    )
