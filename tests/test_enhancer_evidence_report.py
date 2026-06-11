from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_report_module():
    script_path = Path("scripts/summarize_enhancer_evidence.py").resolve()
    spec = importlib.util.spec_from_file_location("summarize_enhancer_evidence", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_summarize_enhancer_evidence_reports_primary_and_best_beta(tmp_path):
    module = _load_report_module()
    output_root = tmp_path / "out"
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True)
    metric_path = metrics_dir / "final_enhanced_eval.json"
    metric_path.write_text(json.dumps(_metric_payload()), encoding="utf-8")
    (output_root / "run_summary.json").write_text(
        json.dumps({"dataset": "visa", "mode": "full", "categories": ["a", "b"]}),
        encoding="utf-8",
    )

    summary = module.summarize_enhancer_evidence(
        metric_path,
        run_summary_path=output_root / "run_summary.json",
    )
    csv_path = metrics_dir / "summary.csv"
    module.write_category_csv(csv_path, summary["categories"])

    assert summary["run"]["dataset"] == "visa"
    assert summary["primary_beta"]["key"] == "0.05"
    assert summary["diagnostic_best_beta"]["key"] == "0.1"
    assert summary["primary_beta"]["delta_vs_baseline"]["image_auroc"] == pytest.approx(0.015)
    assert summary["primary_beta"]["category_counts"] == {
        "positive": 1,
        "negative": 1,
        "zero": 0,
        "missing": 0,
        "num_categories": 2,
    }
    assert [row["category"] for row in summary["worst_categories"]] == ["b", "a"]
    assert csv_path.exists()
    assert "primary_delta_image_auroc" in csv_path.read_text(encoding="utf-8")


def test_summarize_enhancer_evidence_requires_beta_sweep_payload(tmp_path):
    module = _load_report_module()
    metric_path = tmp_path / "metrics.json"
    metric_path.write_text(json.dumps({"mean": {"baseline": {}}}), encoding="utf-8")

    with pytest.raises(ValueError, match="beta-sweep"):
        module.summarize_enhancer_evidence(metric_path)


def _metric_payload():
    return {
        "mean": {
            "baseline": {"image_auroc": 0.8, "image_ap": 0.7, "image_f1": 0.6},
            "enhanced_by_beta": {
                "0": {
                    "image_auroc": 0.8,
                    "image_ap": 0.7,
                    "image_f1": 0.6,
                    "delta_vs_baseline": {"image_auroc": 0.0, "image_ap": 0.0, "image_f1": 0.0},
                },
                "0.05": {
                    "image_auroc": 0.815,
                    "image_ap": 0.72,
                    "image_f1": 0.61,
                    "delta_vs_baseline": {
                        "image_auroc": 0.015,
                        "image_ap": 0.02,
                        "image_f1": 0.01,
                    },
                },
                "0.1": {
                    "image_auroc": 0.875,
                    "image_ap": 0.73,
                    "image_f1": 0.62,
                    "delta_vs_baseline": {
                        "image_auroc": 0.075,
                        "image_ap": 0.03,
                        "image_f1": 0.02,
                    },
                },
            },
            "beta_sweep": {
                "primary_beta_key": "0.05",
                "selection_metric": "image_auroc",
                "diagnostic_best_beta": {
                    "key": "0.1",
                    "beta": 0.1,
                    "selection_metric": "image_auroc",
                    "value": 0.875,
                    "delta_vs_baseline": 0.075,
                },
                "diagnostic_note": "diagnostic only",
            },
        },
        "categories": {
            "a": {
                "baseline": {"image_auroc": 0.8, "image_ap": 0.7, "image_f1": 0.6},
                "enhanced_by_beta": {
                    "0.05": {"image_auroc": 0.85, "image_ap": 0.72, "image_f1": 0.61},
                    "0.1": {"image_auroc": 0.84, "image_ap": 0.73, "image_f1": 0.62},
                },
            },
            "b": {
                "baseline": {"image_auroc": 0.9, "image_ap": 0.8, "image_f1": 0.7},
                "enhanced_by_beta": {
                    "0.05": {"image_auroc": 0.88, "image_ap": 0.79, "image_f1": 0.69},
                    "0.1": {"image_auroc": 0.91, "image_ap": 0.82, "image_f1": 0.71},
                },
            },
        },
    }
