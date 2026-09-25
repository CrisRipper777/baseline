from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
READOUTS = ("terminal", "uniform", "gpr")
RUN_IDS = (1, 2, 3)
METRICS = ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
PAIRED_COMPARISONS = (
    ("uniform", "terminal"),
    ("gpr", "terminal"),
    ("gpr", "uniform"),
)


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def _load_checkpoint(root: Path, dataset: str, readout: str, run_id: int) -> dict[str, Any]:
    path = root / dataset / readout / f"best_run{run_id}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing run checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected_seed = 41 + run_id
    if payload.get("task") != "nc":
        raise ValueError(f"{path} is not an NC checkpoint")
    if payload.get("protocol_version") != "unified_full_graph_nc_v1":
        raise ValueError(f"{path} does not use unified_full_graph_nc_v1")
    if int(payload.get("seed", -1)) != expected_seed:
        raise ValueError(
            f"{path} has seed={payload.get('seed')}, expected {expected_seed}"
        )
    if payload.get("selection") != "best_val_accuracy":
        raise ValueError(f"{path} is not selected by validation accuracy")
    if payload.get("epoch") is None:
        raise ValueError(f"{path} has no best-epoch metadata")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{path} has no metrics payload")
    missing = set(METRICS) - metrics.keys()
    if missing:
        raise ValueError(f"{path} is missing metrics: {sorted(missing)}")
    return payload


def analyze(root: Path, datasets: tuple[str, ...] = DATASETS) -> dict[str, Any]:
    checkpoints: dict[tuple[str, str, int], dict[str, Any]] = {}
    for dataset in datasets:
        for readout in READOUTS:
            for run_id in RUN_IDS:
                checkpoints[(dataset, readout, run_id)] = _load_checkpoint(
                    root, dataset, readout, run_id
                )

    summary: dict[str, Any] = {
        "protocol_version": "unified_full_graph_nc_v1",
        "run_seeds": [42, 43, 44],
        "aggregation": "mean ± population std (ddof=0)",
        "per_dataset_readout": {},
        "paired_comparisons": {},
        "gpr_gammas": {},
    }
    for dataset in datasets:
        summary["per_dataset_readout"][dataset] = {}
        for readout in READOUTS:
            run_payloads = [
                checkpoints[(dataset, readout, run_id)] for run_id in RUN_IDS
            ]
            metric_stats = {
                metric: dict(zip(("mean", "std"), _mean_std([
                    float(payload["metrics"][metric]) for payload in run_payloads
                ]), strict=True))
                for metric in METRICS
            }
            summary["per_dataset_readout"][dataset][readout] = {
                "metrics": metric_stats,
                "runs": [
                    {
                        "run_id": run_id - 1,
                        "seed": int(payload["seed"]),
                        "best_epoch": int(payload["epoch"]),
                        "metrics": {
                            key: float(payload["metrics"][key]) for key in METRICS
                        },
                    }
                    for run_id, payload in zip(RUN_IDS, run_payloads, strict=True)
                ],
            }

        summary["paired_comparisons"][dataset] = {}
        for left, right in PAIRED_COMPARISONS:
            key = f"{left} - {right}"
            differences = {}
            for metric in METRICS:
                paired_values = [
                    float(checkpoints[(dataset, left, run_id)]["metrics"][metric])
                    - float(checkpoints[(dataset, right, run_id)]["metrics"][metric])
                    for run_id in RUN_IDS
                ]
                mean, std = _mean_std(paired_values)
                differences[metric] = {
                    "mean": mean,
                    "std": std,
                    "per_seed": paired_values,
                }
            summary["paired_comparisons"][dataset][key] = differences

        gamma_runs = []
        for run_id in RUN_IDS:
            payload = checkpoints[(dataset, "gpr", run_id)]
            state = payload.get("model_state", {})
            if "gamma_text" not in state or "gamma_visual" not in state:
                raise ValueError(
                    f"{dataset} gpr run {run_id} checkpoint lacks modality gamma parameters"
                )
            gamma_runs.append(
                {
                    "run_id": run_id - 1,
                    "seed": int(payload["seed"]),
                    "gamma_text": state["gamma_text"].detach().cpu().tolist(),
                    "gamma_visual": state["gamma_visual"].detach().cpu().tolist(),
                }
            )
        summary["gpr_gammas"][dataset] = gamma_runs
    return summary


def _format_pct(stats: dict[str, float]) -> str:
    return f"{100 * stats['mean']:.2f} ± {100 * stats['std']:.2f}"


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Multi-Order Bank NC summary",
        "",
        "Values are mean ± population std over seeds 42, 43, and 44.",
        "",
        "| Dataset | Readout | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset, readouts in summary["per_dataset_readout"].items():
        for readout, record in readouts.items():
            metrics = record["metrics"]
            lines.append(
                f"| {dataset} | {readout} | "
                f"{_format_pct(metrics['val_acc'])} | "
                f"{_format_pct(metrics['val_macro_f1'])} | "
                f"{_format_pct(metrics['test_acc'])} | "
                f"{_format_pct(metrics['test_macro_f1'])} |"
            )
    lines.extend(["", "## Paired differences", ""])
    for dataset, comparisons in summary["paired_comparisons"].items():
        lines.extend([f"### {dataset}", "", "| Comparison | Metric | Mean difference |", "|---|---|---:|"])
        for comparison, metrics in comparisons.items():
            for metric, stats in metrics.items():
                lines.append(
                    f"| {comparison} | {metric} | "
                    f"{100 * stats['mean']:.2f} ± {100 * stats['std']:.2f} |"
                )
        lines.append("")
    lines.extend(["## GPR modality coefficients", "", "See the JSON gamma export for each dataset and seed.", ""])
    return chr(10).join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/multi_order_bank_nc")
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    args = parser.parse_args()

    result = analyze(args.root)
    json_path = args.output_json or args.root / "summary.json"
    markdown_path = args.output_markdown or args.root / "summary.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_markdown(result), encoding="utf-8")
    print(f"Saved {json_path}")
    print(f"Saved {markdown_path}")


if __name__ == "__main__":
    main()
