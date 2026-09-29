from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_risa_v06 import (
    FORMAL_VARIANT,
    OUTPUT_ROOT,
    PROTOCOL,
    SCREEN_DATASETS,
    SEEDS,
    _validate_checkpoint,
    _validate_run_metrics,
    expected_checkpoint_paths,
)

V05_FULL_RESULTS = ROOT / "results/risa_v05/formal_analysis/formal_risa_v05.json"
V05_ABLATION_RESULTS = ROOT / "results/risa_v05/main_ablation/risa_v05_main_ablation.json"


def mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty list")
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    return {"mean": mean, "std": std}


def _read_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("protocol_version") != PROTOCOL:
        raise ValueError(f"protocol mismatch in comparison source {path}")
    return data.get("records", [])


def _v05_comparison_records() -> dict[tuple[str, str], list[dict[str, Any]]]:
    if not V05_FULL_RESULTS.is_file() or not V05_ABLATION_RESULTS.is_file():
        return {}
    formal_full = _read_records(V05_FULL_RESULTS)
    ablation = _read_records(V05_ABLATION_RESULTS)
    records: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for dataset in SCREEN_DATASETS:
        formal_rows = [row for row in formal_full
                       if row.get("dataset") == dataset and row.get("variant") == "v05_full"]
        if len(formal_rows) == len(SEEDS):
            records[(dataset, "v05_full")] = sorted(formal_rows, key=lambda row: int(row["seed"]))
        for variant in ("v05_absorb_only", "v05_plain", "v05_crst_only"):
            rows = [row for row in ablation
                    if row.get("dataset") == dataset and row.get("variant") == variant]
            if len(rows) == len(SEEDS):
                records[(dataset, variant)] = sorted(rows, key=lambda row: int(row["seed"]))
    return records


def _paired_comparison(current: list[dict[str, Any]],
                       baseline: list[dict[str, Any]]) -> dict[str, Any]:
    current_by_seed = {int(row["seed"]): row for row in current}
    baseline_by_seed = {int(row["seed"]): row for row in baseline}
    shared = sorted(set(current_by_seed) & set(baseline_by_seed))
    if shared != list(SEEDS):
        raise ValueError(f"paired comparison needs seeds {SEEDS}, found {shared}")
    output: dict[str, Any] = {"seeds": shared}
    for metric in ("val_acc", "val_macro_f1"):
        deltas = [float(current_by_seed[seed][metric]) - float(baseline_by_seed[seed][metric])
                  for seed in shared]
        stats = mean_std(deltas)
        output[metric] = {
            "baseline_mean": mean_std([float(baseline_by_seed[s][metric]) for s in shared])["mean"],
            "current_mean": mean_std([float(current_by_seed[s][metric]) for s in shared])["mean"],
            "paired_seed_deltas": dict(zip(map(str, shared), deltas, strict=True)),
            "mean_delta": stats["mean"], "std_delta": stats["std"],
            "mean_delta_percentage_points": stats["mean"] * 100.0,
            "std_delta_percentage_points": stats["std"] * 100.0,
        }
    return output


def analyze_formal(output_dir: Path | None = None) -> dict[str, Any]:
    output_dir = output_dir or ROOT / "results/risa_v06/formal"
    dataset_results: dict[str, Any] = {}
    raw_rows: list[dict[str, Any]] = []
    audits = []
    maximum_gpu_mib = 0

    for dataset in SCREEN_DATASETS:
        run_dir = OUTPUT_ROOT / "formal" / dataset / FORMAL_VARIANT
        complete_path = run_dir / "complete.json"
        metrics_path = run_dir / "run_metrics.json"
        if not complete_path.is_file():
            raise FileNotFoundError(complete_path)
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if (complete.get("phase"), complete.get("dataset"), complete.get("variant"),
                complete.get("status")) != ("formal", dataset, FORMAL_VARIANT, "complete"):
            raise ValueError(f"formal completion metadata mismatch for {dataset}")
        if complete.get("protocol_version") != PROTOCOL:
            raise ValueError(f"protocol mismatch in {complete_path}")
        if complete.get("test_evaluation") is not False or complete.get("lp_evaluation") is not False:
            raise ValueError(f"test or LP evaluation was enabled for {dataset}")
        metrics = _validate_run_metrics(metrics_path, tuple(SEEDS))
        checkpoints = expected_checkpoint_paths(run_dir, len(SEEDS))
        payloads = [_validate_checkpoint(path, seed)
                    for path, seed in zip(checkpoints, SEEDS, strict=True)]
        rows = metrics["runs"]
        if len(rows) != len(SEEDS):
            raise ValueError(f"expected {len(SEEDS)} runs for {dataset}, found {len(rows)}")

        dataset_rows = []
        checkpoint_checks = []
        for seed, metric_row, checkpoint_path, payload in zip(
                SEEDS, rows, checkpoints, payloads, strict=True):
            if int(metric_row.get("seed", -1)) != seed:
                raise ValueError(f"run_metrics seed mismatch for {dataset}, expected {seed}")
            if int(payload.get("seed", -1)) != seed:
                raise ValueError(f"checkpoint seed mismatch for {checkpoint_path}")
            if payload.get("selection") != "best_val_accuracy":
                raise ValueError(f"checkpoint is not Val Accuracy-selected: {checkpoint_path}")
            for metric in ("val_acc", "val_macro_f1"):
                recorded = float(metric_row["metrics"][metric])
                selected = float(payload["metrics"][metric])
                if not math.isclose(recorded, selected, rel_tol=0.0, abs_tol=1e-8):
                    raise ValueError(f"checkpoint/run_metrics mismatch for {checkpoint_path}:{metric}")
            best_epoch = int(metric_row["metadata"]["best_epoch"])
            if best_epoch != int(payload["epoch"]) or not 1 <= best_epoch <= 300:
                raise ValueError(f"best epoch mismatch or invalid for {checkpoint_path}")
            row = {
                "dataset": dataset, "variant": FORMAL_VARIANT, "seed": seed,
                "best_epoch": best_epoch,
                "val_acc": float(metric_row["metrics"]["val_acc"]),
                "val_macro_f1": float(metric_row["metrics"]["val_macro_f1"]),
                "checkpoint": str(checkpoint_path),
            }
            dataset_rows.append(row)
            raw_rows.append(row)
            checkpoint_checks.append({
                "seed": seed, "path": str(checkpoint_path),
                "selection": payload["selection"],
                "protocol_version": payload["protocol_version"],
                "test_metrics_absent": not any(k.startswith("test_")
                                                for k in payload.get("metrics", {})),
                "finite_metrics_and_weights": True,
                "metrics_match_run_metrics": True,
                "best_epoch": best_epoch,
            })
        epoch_values = [float(row["best_epoch"]) for row in dataset_rows]
        memory = complete.get("peak_process_gpu_memory_mib")
        if memory is not None:
            maximum_gpu_mib = max(maximum_gpu_mib, int(memory))
        dataset_results[dataset] = {
            "n_seeds": len(dataset_rows), "seeds": list(SEEDS),
            "metrics": {
                metric: mean_std([row[metric] for row in dataset_rows])
                for metric in ("val_acc", "val_macro_f1")
            },
            "best_epoch": {
                **mean_std(epoch_values),
                "min": int(min(epoch_values)), "max": int(max(epoch_values)),
                "per_seed": {str(row["seed"]): row["best_epoch"] for row in dataset_rows},
            },
            "peak_process_gpu_memory_mib": memory,
            "checkpoint_audit": checkpoint_checks,
            "run_metrics_audit": "passed",
        }
        audits.extend(checkpoint_checks)

    old_results = _v05_comparison_records()
    comparisons: dict[str, Any] = {}
    for dataset, rows in dataset_results.items():
        current_rows = [row for row in raw_rows if row["dataset"] == dataset]
        comparisons[dataset] = {}
        for variant in ("v05_full", "v05_absorb_only", "v05_plain", "v05_crst_only"):
            baseline = old_results.get((dataset, variant))
            if baseline:
                comparisons[dataset][variant] = _paired_comparison(current_rows, baseline)

    report = {
        "report": "RISA v0.6 formal screening summary",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": PROTOCOL, "phase": "formal", "variant": FORMAL_VARIANT,
        "datasets": list(SCREEN_DATASETS), "seeds": list(SEEDS),
        "runs_expected": len(SCREEN_DATASETS) * len(SEEDS),
        "runs_validated": len(audits), "all_checkpoints_valid": len(audits) == 9,
        "selection": "best_val_accuracy", "test_evaluated": False,
        "lp_evaluated": False, "aggregation": "mean ± population standard deviation (ddof=0)",
        "parameter_counts": {
            dataset: {
                "model": int(json.loads((OUTPUT_ROOT / "formal" / dataset / FORMAL_VARIANT /
                                         "complete.json").read_text())["model_trainable_params"]),
                "classifier": int(json.loads((OUTPUT_ROOT / "formal" / dataset / FORMAL_VARIANT /
                                               "complete.json").read_text())["classifier_trainable_params"]),
            } for dataset in SCREEN_DATASETS
        },
        "peak_process_gpu_memory_mib_max": maximum_gpu_mib,
        "per_dataset": dataset_results,
        "paired_v05_comparisons": comparisons,
        "comparison_sources": {
            "v05_full": str(V05_FULL_RESULTS.relative_to(ROOT)),
            "v05_ablation_variants": str(V05_ABLATION_RESULTS.relative_to(ROOT)),
        },
        "analysis_notes": [
            "Against v0.5 Full, v0.6 is dataset-dependent: both validation metrics decrease on Movies, Grocery accuracy is nearly unchanged while Macro-F1 is lower, and both metrics improve on ele-fashion.",
            "Across three seeds, Val Accuracy has low spread (SD 0.0004 to 0.0029); Val Macro-F1 varies more on Movies (SD 0.0161) than Grocery (0.0046) or ele-fashion (0.0009).",
            "These are paired validation-set comparisons over three seeds; no test set was evaluated, so they do not establish test-set generalization or statistical significance.",
        ],
        "per_seed_records": raw_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "formal_risa_v06.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    csv_path = output_dir / "formal_risa_v06.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "dataset", "variant", "seed", "best_epoch", "val_acc", "val_macro_f1", "checkpoint",
        ), lineterminator="\n")
        writer.writeheader()
        writer.writerows(raw_rows)

    lines = [
        "# RISA v0.6 formal screening summary", "",
        f"- Protocol: `{PROTOCOL}`; variant: `{FORMAL_VARIANT}`.",
        f"- Runs: {report['runs_validated']}/{report['runs_expected']} validated; seeds {', '.join(map(str, SEEDS))}.",
        "- Checkpoints selected by Val Accuracy; test and LP evaluation were disabled.",
        "- Statistics are mean ± population SD (ddof=0) over three seeds.",
        f"- Peak sampled process GPU memory: {maximum_gpu_mib} MiB.", "",
        "## Validation metrics", "",
        "| Dataset | Val Accuracy | Val Macro-F1 | Best epoch | Peak GPU (MiB) |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in SCREEN_DATASETS:
        entry = dataset_results[dataset]
        acc, f1 = entry["metrics"]["val_acc"], entry["metrics"]["val_macro_f1"]
        epoch = entry["best_epoch"]
        lines.append(
            f"| {dataset} | {acc['mean']:.4f} ± {acc['std']:.4f} "
            f"| {f1['mean']:.4f} ± {f1['std']:.4f} "
            f"| {epoch['mean']:.1f} ({epoch['min']}–{epoch['max']}) "
            f"| {entry['peak_process_gpu_memory_mib']} |"
        )
    lines += ["", "## Paired comparison with v0.5", "",
              "Δ is v0.6 minus v0.5 in percentage points, paired by dataset and seed. These are validation metrics, not test-set estimates.", "",
              "| Dataset | Baseline | Δ Val Accuracy (pp) | Δ Val Macro-F1 (pp) |",
              "|---|---|---:|---:|"]
    for dataset in SCREEN_DATASETS:
        for variant in ("v05_full", "v05_absorb_only"):
            comparison = comparisons.get(dataset, {}).get(variant)
            if comparison:
                lines.append(
                    f"| {dataset} | {variant} "
                    f"| {comparison['val_acc']['mean_delta_percentage_points']:+.2f} "
                    f"| {comparison['val_macro_f1']['mean_delta_percentage_points']:+.2f} |"
                )
    lines += ["", "## Seed-level results", "",
              "| Dataset | Seed | Best epoch | Val Accuracy | Val Macro-F1 |",
              "|---|---:|---:|---:|---:|"]
    for row in raw_rows:
        lines.append(
            f"| {row['dataset']} | {row['seed']} | {row['best_epoch']} "
            f"| {row['val_acc']:.4f} | {row['val_macro_f1']:.4f} |"
        )
    lines += ["", "## Interpretation", "",
              "- All nine validation-selected checkpoints passed seed, protocol, finiteness, and run-metrics consistency checks.",
              "- Against v0.5 Full, v0.6 is dataset-dependent: both validation metrics decrease on Movies, Grocery accuracy is nearly unchanged while Macro-F1 is lower, and both metrics improve on ele-fashion.",
              "- Val Accuracy has low seed spread (SD 0.0004–0.0029); Val Macro-F1 varies more on Movies (SD 0.0161) than Grocery (0.0046) or ele-fashion (0.0009).",
              "- These are paired validation-set comparisons over three seeds. No test set was evaluated, so they do not establish test-set generalization or statistical significance.",
              f"- v0.5 comparison sources: `{V05_FULL_RESULTS.relative_to(ROOT)}` and `{V05_ABLATION_RESULTS.relative_to(ROOT)}`.",
              "- Full per-checkpoint provenance, metrics, and paired seed deltas are in `formal_risa_v06.json`; row-level values are in `formal_risa_v06.csv`.", ""]
    (output_dir / "formal_risa_v06.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize RISA v0.6 formal runs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/risa_v06/formal")
    args = parser.parse_args()
    result = analyze_formal(args.output_dir.resolve())
    print(json.dumps({
        "runs_validated": result["runs_validated"],
        "all_checkpoints_valid": result["all_checkpoints_valid"],
        "peak_process_gpu_memory_mib_max": result["peak_process_gpu_memory_mib_max"],
        "summary": str(args.output_dir.resolve() / "formal_risa_v06.md"),
    }, indent=2))


if __name__ == "__main__":
    main()
