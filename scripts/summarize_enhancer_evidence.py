from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


IMAGE_METRIC_KEYS = ("image_auroc", "image_ap", "image_f1")
DEFAULT_METRIC_NAMES = ("final_enhanced_eval", "eval_enhanced")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize enhancer beta-sweep evidence.")
    parser.add_argument("output_root", help="Server OUTPUT_ROOT containing run_summary.json and metrics/")
    parser.add_argument(
        "--metric-name",
        default=None,
        help="Metric JSON basename under OUTPUT_ROOT/metrics, without .json",
    )
    parser.add_argument("--json-out", default=None, help="Summary JSON path")
    parser.add_argument("--csv-out", default=None, help="Summary CSV path")
    parser.add_argument("--no-csv", action="store_true", help="Only write JSON")
    args = parser.parse_args(argv)

    output_root = Path(args.output_root).expanduser()
    metric_path = resolve_metric_path(output_root, metric_name=args.metric_name)
    run_summary_path = output_root / "run_summary.json"
    run_summary = run_summary_path if run_summary_path.is_file() else None

    try:
        summary = summarize_enhancer_evidence(metric_path, run_summary_path=run_summary)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from None

    json_out = Path(args.json_out).expanduser() if args.json_out else output_root / "metrics" / "enhancer_evidence_summary.json"
    write_json(json_out, summary)

    if not args.no_csv:
        csv_out = Path(args.csv_out).expanduser() if args.csv_out else output_root / "metrics" / "enhancer_evidence_summary.csv"
        write_category_csv(csv_out, summary.get("categories", []))

    print(json.dumps(summary, indent=2, sort_keys=True))


def resolve_metric_path(output_root: Path, *, metric_name: Optional[str] = None) -> Path:
    metrics_dir = Path(output_root) / "metrics"
    if metric_name:
        candidate = metrics_dir / f"{metric_name}.json"
        if not candidate.is_file():
            raise ValueError(f"metric file not found: {candidate}")
        return candidate
    for name in DEFAULT_METRIC_NAMES:
        candidate = metrics_dir / f"{name}.json"
        if candidate.is_file():
            return candidate
    expected = ", ".join(f"{name}.json" for name in DEFAULT_METRIC_NAMES)
    raise ValueError(f"no enhanced metric file found under {metrics_dir}; expected one of: {expected}")


def summarize_enhancer_evidence(
    metric_path: Path,
    *,
    run_summary_path: Optional[Path] = None,
) -> Dict[str, Any]:
    metric_path = Path(metric_path)
    payload = read_json(metric_path)
    mean = payload.get("mean", {})
    enhanced_by_beta = mean.get("enhanced_by_beta")
    if not isinstance(enhanced_by_beta, dict) or not enhanced_by_beta:
        raise ValueError(
            f"metric payload does not contain beta-sweep fields: {metric_path}. "
            "Run with EVAL_FUSION_BETA_SWEEP set."
        )

    beta_sweep = mean.get("beta_sweep", {})
    selection_metric = str(beta_sweep.get("selection_metric", "image_auroc"))
    primary_key = str(beta_sweep.get("primary_beta_key") or _first_beta_key(enhanced_by_beta))
    if primary_key not in enhanced_by_beta:
        primary_key = _first_beta_key(enhanced_by_beta)
    best = beta_sweep.get("diagnostic_best_beta") or _diagnostic_best_beta(
        enhanced_by_beta,
        selection_metric=selection_metric,
        baseline_metrics=mean.get("baseline", {}),
    )
    best_key = str(best.get("key")) if isinstance(best, dict) and best.get("key") in enhanced_by_beta else primary_key

    baseline_mean = mean.get("baseline", {})
    primary_mean = enhanced_by_beta[primary_key]
    best_mean = enhanced_by_beta[best_key]
    category_rows = _category_rows(
        payload.get("categories", {}),
        primary_key=primary_key,
        best_key=best_key,
        selection_metric=selection_metric,
    )

    run_summary = read_json(run_summary_path) if run_summary_path and Path(run_summary_path).is_file() else {}
    return {
        "source_metric_path": str(metric_path),
        "run_summary_path": str(run_summary_path) if run_summary_path else None,
        "run": _run_info(run_summary),
        "selection_metric": selection_metric,
        "baseline_mean": baseline_mean,
        "primary_beta": {
            "key": primary_key,
            "beta": float(primary_key),
            "mean": primary_mean,
            "delta_vs_baseline": _metric_delta(primary_mean, baseline_mean),
            "category_counts": _delta_counts(category_rows, f"primary_delta_{selection_metric}"),
        },
        "diagnostic_best_beta": {
            "key": best_key,
            "beta": float(best_key),
            "selection_metric": selection_metric,
            "mean": best_mean,
            "delta_vs_baseline": _metric_delta(best_mean, baseline_mean),
            "category_counts": _delta_counts(category_rows, f"best_delta_{selection_metric}"),
            "diagnostic_note": beta_sweep.get("diagnostic_note"),
        },
        "beta_sweep": beta_sweep,
        "all_beta_means": enhanced_by_beta,
        "categories": category_rows,
        "best_categories": _rank_categories(category_rows, f"primary_delta_{selection_metric}", reverse=True),
        "worst_categories": _rank_categories(category_rows, f"primary_delta_{selection_metric}", reverse=False),
    }


def write_category_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    row_list = list(rows)
    fieldnames = [
        "category",
        "selection_metric",
        "primary_beta_key",
        "best_beta_key",
    ]
    for prefix in ("baseline", "primary", "primary_delta", "best", "best_delta"):
        fieldnames.extend(f"{prefix}_{key}" for key in IMAGE_METRIC_KEYS)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in row_list:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _category_rows(
    categories: Dict[str, Any],
    *,
    primary_key: str,
    best_key: str,
    selection_metric: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for category, summary in sorted(categories.items()):
        baseline = summary.get("baseline", {})
        by_beta = summary.get("enhanced_by_beta", {})
        if primary_key not in by_beta or best_key not in by_beta:
            continue
        primary = by_beta[primary_key]
        best = by_beta[best_key]
        row: Dict[str, Any] = {
            "category": category,
            "selection_metric": selection_metric,
            "primary_beta_key": primary_key,
            "best_beta_key": best_key,
        }
        for key in IMAGE_METRIC_KEYS:
            row[f"baseline_{key}"] = baseline.get(key)
            row[f"primary_{key}"] = primary.get(key)
            row[f"primary_delta_{key}"] = _single_delta(primary.get(key), baseline.get(key))
            row[f"best_{key}"] = best.get(key)
            row[f"best_delta_{key}"] = _single_delta(best.get(key), baseline.get(key))
        rows.append(row)
    return rows


def _run_info(run_summary: Dict[str, Any]) -> Dict[str, Any]:
    if not run_summary:
        return {}
    return {
        "dataset": run_summary.get("dataset"),
        "mode": run_summary.get("mode"),
        "categories": run_summary.get("categories"),
        "few_shot": run_summary.get("few_shot"),
        "checkpoint_path": run_summary.get("checkpoint_path"),
        "effective_config_path": run_summary.get("effective_config_path"),
    }


def _metric_delta(metrics: Dict[str, Any], baseline_metrics: Dict[str, Any]) -> Dict[str, Optional[float]]:
    return {
        key: _single_delta(metrics.get(key), baseline_metrics.get(key))
        for key in IMAGE_METRIC_KEYS
    }


def _single_delta(value: Any, baseline: Any) -> Optional[float]:
    if isinstance(value, (float, int)) and isinstance(baseline, (float, int)):
        return float(value) - float(baseline)
    return None


def _delta_counts(rows: Iterable[Dict[str, Any]], field: str) -> Dict[str, int]:
    counts = {"positive": 0, "negative": 0, "zero": 0, "missing": 0}
    for row in rows:
        value = row.get(field)
        if value is None:
            counts["missing"] += 1
        elif value > 0:
            counts["positive"] += 1
        elif value < 0:
            counts["negative"] += 1
        else:
            counts["zero"] += 1
    counts["num_categories"] = sum(counts.values())
    return counts


def _rank_categories(rows: List[Dict[str, Any]], field: str, *, reverse: bool) -> List[Dict[str, Any]]:
    ranked = [row for row in rows if isinstance(row.get(field), (float, int))]
    ranked.sort(key=lambda row: float(row[field]), reverse=reverse)
    return ranked[:5]


def _diagnostic_best_beta(
    enhanced_by_beta: Dict[str, Any],
    *,
    selection_metric: str,
    baseline_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    candidates = []
    for key, metrics in enhanced_by_beta.items():
        value = metrics.get(selection_metric)
        if isinstance(value, (float, int)):
            candidates.append((float(value), str(key)))
    if not candidates:
        key = _first_beta_key(enhanced_by_beta)
        return {"key": key, "beta": float(key), "selection_metric": selection_metric, "value": None}
    value, key = max(candidates, key=lambda item: (item[0], -abs(float(item[1]))))
    return {
        "key": key,
        "beta": float(key),
        "selection_metric": selection_metric,
        "value": value,
        "delta_vs_baseline": _single_delta(value, baseline_metrics.get(selection_metric)),
    }


def _first_beta_key(enhanced_by_beta: Dict[str, Any]) -> str:
    return sorted(enhanced_by_beta, key=lambda key: float(key))[0]


def read_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
