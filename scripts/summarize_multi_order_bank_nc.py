from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from sklearn.metrics import f1_score


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
READOUTS = ("terminal", "uniform", "gpr")
FUSIONS = ("plain", "residual")
RUN_IDS = (1, 2, 3)
SEEDS = (42, 43, 44)
METRICS = ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
BASELINES = ("mlp", "gcn", "sage", "mmgcn", "mgat", "dip", "dgf", "dmgc", "lgmrec")
CONTRASTS = (
    ("uniform_plain - terminal_plain", "mob_uniform_plain", "mob_terminal_plain", "retention"),
    ("uniform_residual - terminal_residual", "mob_uniform_residual", "mob_terminal_residual", "retention"),
    ("gpr_plain - uniform_plain", "mob_gpr_plain", "mob_uniform_plain", "adaptive_response"),
    ("gpr_residual - uniform_residual", "mob_gpr_residual", "mob_uniform_residual", "adaptive_response"),
    ("terminal_residual - terminal_plain", "mob_terminal_residual", "mob_terminal_plain", "fusion"),
    ("uniform_residual - uniform_plain", "mob_uniform_residual", "mob_uniform_plain", "fusion"),
    ("gpr_residual - gpr_plain", "mob_gpr_residual", "mob_gpr_plain", "fusion"),
)


def variant_name(readout: str, fusion: str) -> str:
    return f"mob_{readout}_{fusion}"


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def _load_mob_checkpoint(root: Path, dataset: str, variant: str, run_id: int) -> dict[str, Any]:
    path = root / dataset / variant / f"best_run{run_id}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing factorial checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected_seed = SEEDS[run_id - 1]
    if payload.get("task") != "nc" or payload.get("protocol_version") != "unified_full_graph_nc_v1":
        raise ValueError(f"{path} does not use the frozen NC protocol")
    if int(payload.get("seed", -1)) != expected_seed:
        raise ValueError(f"{path} has seed={payload.get('seed')}, expected {expected_seed}")
    if payload.get("selection") != "best_val_accuracy" or payload.get("epoch") is None:
        raise ValueError(f"{path} lacks validation-selected checkpoint metadata")
    missing = set(METRICS) - payload.get("metrics", {}).keys()
    if missing:
        raise ValueError(f"{path} is missing metrics: {sorted(missing)}")
    return payload


def _load_baseline_runs(root: Path, dataset: str, model: str) -> list[dict[str, Any]]:
    path = root / dataset / model / "per_run_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing baseline run metrics: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_version") != "unified_full_graph_nc_v1":
        raise ValueError(f"{path} does not use unified_full_graph_nc_v1")
    if payload.get("run_seeds") != list(SEEDS) or len(payload.get("runs", [])) != 3:
        raise ValueError(f"{path} does not contain runs for seeds {SEEDS}")
    for run_id, run in enumerate(payload["runs"]):
        if int(run.get("seed", -1)) != SEEDS[run_id]:
            raise ValueError(f"{path} has unexpected run seed {run.get('seed')}")
        if set(METRICS) - run.get("metrics", {}).keys():
            raise ValueError(f"{path} has missing metrics in run {run_id + 1}")
    return payload["runs"]


def _metrics_of(run: dict[str, Any]) -> dict[str, float]:
    source = run["metrics"]
    return {name: float(source[name]) for name in METRICS}


def analyze(
    factorial_root: Path,
    datasets: tuple[str, ...] = DATASETS,
    baseline_root: Path | None = None,
) -> dict[str, Any]:
    mob: dict[tuple[str, str, int], dict[str, Any]] = {}
    baseline: dict[tuple[str, str, int], dict[str, Any]] = {}
    for dataset in datasets:
        for fusion in FUSIONS:
            for readout in READOUTS:
                variant = variant_name(readout, fusion)
                for run_id in RUN_IDS:
                    mob[(dataset, variant, run_id)] = _load_mob_checkpoint(
                        factorial_root, dataset, variant, run_id
                    )
        if baseline_root is not None:
            for model in BASELINES:
                for run_id, run in enumerate(_load_baseline_runs(baseline_root, dataset, model), start=1):
                    baseline[(dataset, model, run_id)] = run

    summary: dict[str, Any] = {
        "protocol_version": "unified_full_graph_nc_v1",
        "datasets": list(datasets),
        "seeds": list(SEEDS),
        "aggregation": "mean ± population std (ddof=0)",
        "factorial_jobs": len(datasets) * 6,
        "factorial_runs": len(datasets) * 6 * len(SEEDS),
        "baseline_jobs": len(datasets) * len(BASELINES) if baseline_root else 0,
        "baseline_runs": len(datasets) * len(BASELINES) * len(SEEDS) if baseline_root else 0,
        "factorial": {},
        "baselines": {},
        "paired_contrasts": {},
        "preregistered_support": {},
        "fusion_attribution_required": False,
    }
    factorial_rows = []
    per_run_rows = []
    baseline_rows = []

    for dataset in datasets:
        summary["factorial"][dataset] = {}
        for fusion in FUSIONS:
            for readout in READOUTS:
                variant = variant_name(readout, fusion)
                runs = [mob[(dataset, variant, run_id)] for run_id in RUN_IDS]
                metric_stats = {
                    metric: dict(zip(("mean", "std"), _mean_std([
                        float(run["metrics"][metric]) for run in runs
                    ]), strict=True))
                    for metric in METRICS
                }
                summary["factorial"][dataset][variant] = {
                    "metrics": metric_stats,
                    "runs": [
                        {
                            "run_id": run_id - 1,
                            "seed": int(run["seed"]),
                            "best_epoch": int(run["epoch"]),
                            "metrics": _metrics_of(run),
                        }
                        for run_id, run in zip(RUN_IDS, runs, strict=True)
                    ],
                }
                factorial_rows.append({
                    "dataset": dataset,
                    "variant": variant,
                    **{f"{metric}_mean": metric_stats[metric]["mean"] for metric in METRICS},
                    **{f"{metric}_std": metric_stats[metric]["std"] for metric in METRICS},
                })
                for run_id, run in zip(RUN_IDS, runs, strict=True):
                    per_run_rows.append({
                        "dataset": dataset,
                        "model": "multi_order_bank",
                        "variant": variant,
                        "run_id": run_id - 1,
                        "seed": int(run["seed"]),
                        "best_epoch": int(run["epoch"]),
                        **_metrics_of(run),
                    })

        if baseline_root is not None:
            summary["baselines"][dataset] = {}
            for model in BASELINES:
                runs = [baseline[(dataset, model, run_id)] for run_id in RUN_IDS]
                stats = {
                    metric: dict(zip(("mean", "std"), _mean_std([
                        float(run["metrics"][metric]) for run in runs
                    ]), strict=True))
                    for metric in METRICS
                }
                summary["baselines"][dataset][model] = {"metrics": stats, "runs": runs}
                metadata = [run.get("metadata", {}) for run in runs]
                first = metadata[0] if metadata else {}
                model_counts = sorted({int(item.get("model_parameters", -1)) for item in metadata})
                if len(model_counts) > 1:
                    raise ValueError(f"{dataset}/{model} parameter count changed across seeds: {model_counts}")
                groups = first.get("optimizer_groups", [])
                optimizer_lr = ",".join(str(group.get("lr")) for group in groups)
                optimizer_wd = ",".join(str(group.get("weight_decay")) for group in groups)
                row = {
                    "dataset": dataset,
                    "model": model,
                    "model_parameters": model_counts[0] if model_counts else "",
                    "classifier_parameters": first.get("classifier_parameters", ""),
                    "optimizer": first.get("optimizer", ""),
                    "optimizer_lr_groups": optimizer_lr,
                    "weight_decay_groups": optimizer_wd,
                    "hidden_dim": first.get("hidden_dim", ""),
                    "num_layers": first.get("num_layers", ""),
                    "other_depth_fields": json.dumps(first.get("other_depth_fields", {}), sort_keys=True),
                }
                row.update({f"{metric}_mean": stats[metric]["mean"] for metric in METRICS})
                row.update({f"{metric}_std": stats[metric]["std"] for metric in METRICS})
                baseline_rows.append(row)
                for run_id, run in enumerate(runs, start=1):
                    per_run_rows.append({
                        "dataset": dataset,
                        "model": model,
                        "variant": "baseline",
                        "run_id": run_id - 1,
                        "seed": int(run["seed"]),
                        "best_epoch": run.get("metadata", {}).get("best_epoch", ""),
                        **_metrics_of(run),
                        "model_parameters": run.get("metadata", {}).get("model_parameters", ""),
                        "classifier_parameters": run.get("metadata", {}).get("classifier_parameters", ""),
                        "optimizer": run.get("metadata", {}).get("optimizer", ""),
                        "optimizer_groups": json.dumps(run.get("metadata", {}).get("optimizer_groups", [])),
                        "hidden_dim": run.get("metadata", {}).get("hidden_dim", ""),
                        "num_layers": run.get("metadata", {}).get("num_layers", ""),
                        "other_depth_fields": json.dumps(run.get("metadata", {}).get("other_depth_fields", {}), sort_keys=True),
                    })

    per_contrast_rows = []
    for label, left_variant, right_variant, category in CONTRASTS:
        dataset_contrasts = {}
        dataset_val_means = []
        positive_dataset_means = 0
        positive_pairs = 0
        val_pairs = []
        for dataset in datasets:
            pair_values = {
                metric: [
                    float(mob[(dataset, left_variant, run_id)]["metrics"][metric])
                    - float(mob[(dataset, right_variant, run_id)]["metrics"][metric])
                    for run_id in RUN_IDS
                ]
                for metric in METRICS
            }
            means = {metric: _mean_std(values) for metric, values in pair_values.items()}
            dataset_contrasts[dataset] = {
                metric: {"mean": mean, "std": std, "per_seed": pair_values[metric]}
                for metric, (mean, std) in means.items()
            }
            val_mean = means["val_acc"][0]
            dataset_val_means.append(val_mean)
            positive_dataset_means += int(val_mean > 0)
            positive_pairs += sum(value > 0 for value in pair_values["val_acc"])
            val_pairs.extend(pair_values["val_acc"])
            per_contrast_rows.append({
                "contrast": label,
                "category": category,
                "dataset": dataset,
                **{f"{metric}_mean": means[metric][0] for metric in METRICS},
                **{f"{metric}_std": means[metric][1] for metric in METRICS},
                "positive_val_seed_pairs": sum(value > 0 for value in pair_values["val_acc"]),
                "val_acc_diffs": json.dumps(pair_values["val_acc"]),
            })
        overall_mean = float(np.mean(dataset_val_means))
        overall_pair_mean, overall_pair_std = _mean_std(val_pairs)
        if overall_mean >= 0.003 and positive_dataset_means >= 4 and positive_pairs >= 10:
            support = "STRONG_SUPPORT"
        elif overall_mean > 0 and positive_dataset_means >= 3:
            support = "MODERATE_SUPPORT"
        else:
            support = "UNSUPPORTED_OR_MIXED"
        summary["paired_contrasts"][label] = {
            "category": category,
            "per_dataset": dataset_contrasts,
            "val_acc_summary": {
                "mean_across_dataset_means": overall_mean,
                "mean_across_15_pairs": overall_pair_mean,
                "population_std_across_seed_pairs": overall_pair_std,
                "mean_pp": 100.0 * overall_mean,
                "positive_datasets": positive_dataset_means,
                "dataset_count": len(datasets),
                "positive_seed_pairs": positive_pairs,
                "seed_pair_count": len(val_pairs),
                "support": support,
            },
        }
        summary["preregistered_support"][label] = support

    fusion_labels = [label for label, _, _, category in CONTRASTS if category == "fusion"]
    fusion_attribution_required = False
    for label in fusion_labels:
        val = summary["paired_contrasts"][label]["val_acc_summary"]
        if val["support"] == "STRONG_SUPPORT" or (val["mean_pp"] >= 0.30 and val["positive_datasets"] >= 3):
            fusion_attribution_required = True
    summary["fusion_attribution_required"] = fusion_attribution_required

    for retention, response in (
        ("uniform_plain - terminal_plain", "uniform_residual - terminal_residual"),
        ("gpr_plain - uniform_plain", "gpr_residual - uniform_residual"),
    ):
        checks = []
        for label in (retention, response):
            val = summary["paired_contrasts"][label]["val_acc_summary"]
            checks.append(val["mean_across_dataset_means"] > 0 and val["positive_datasets"] >= 3)
        summary.setdefault("fusion_robust_multi_order", {})[retention + " / " + response] = bool(all(checks))

    summary["per_run_rows"] = per_run_rows
    summary["factorial_rows"] = factorial_rows
    summary["baseline_rows"] = baseline_rows
    summary["contrast_rows"] = per_contrast_rows
    return summary


def _load_nc_data(dataset: str, seed: int = 42):
    from src.data import load_mag_data

    config_dir = str((Path.cwd() / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=[f"dataset={dataset}", "task=nc", f"seed={seed}"])
    return cfg, load_mag_data(cfg, "nc", seed)


def _macro_f1(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    indices: torch.Tensor,
    eval_labels: list[int] | tuple[int, ...],
) -> float:
    y = labels[indices].cpu().numpy()
    pred = predictions[indices].cpu().numpy()
    return float(f1_score(y, pred, labels=list(eval_labels), average="macro", zero_division=0))


def collect_gpr_diagnostics(factorial_root: Path, datasets: tuple[str, ...], device_name: str = "cuda:0") -> dict[str, list[dict[str, Any]]]:
    from src.models import build_model
    from src.tasks.nc import _resolve_nc_eval_labels

    device = torch.device(device_name)
    gamma_rows: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []
    redundancy_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []

    for dataset in datasets:
        _, data = _load_nc_data(dataset, seed=42)
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        eval_labels = _resolve_nc_eval_labels(data)
        for fusion in FUSIONS:
            variant = variant_name("gpr", fusion)
            for run_id in RUN_IDS:
                payload = _load_mob_checkpoint(factorial_root, dataset, variant, run_id)
                model_cfg = OmegaConf.create({
                    "model": {
                        "name": "multi_order_bank",
                        "hidden_dim": 256,
                        "max_order": 3,
                        "num_layers": 3,
                        "dropout": 0.2,
                        "readout": "gpr",
                        "fusion_mode": "plain_mlp" if fusion == "plain" else "residual",
                    }
                })
                model = build_model(model_cfg, payload["data_info"]).to(device)
                model.load_state_dict(payload["model_state"])
                model.eval()
                classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
                classifier.load_state_dict(payload["head_state"])
                classifier.eval()
                with torch.no_grad():
                    analysis = model.analyze(x, edge_index)
                    states_t = analysis["S_text"]
                    states_v = analysis["S_visual"]
                    gamma_t = model.gamma_text.detach().cpu().numpy().astype(float)
                    gamma_v = model.gamma_visual.detach().cpu().numpy().astype(float)
                    eps = 1e-12
                    sum_t = float(np.abs(gamma_t).sum())
                    sum_v = float(np.abs(gamma_v).sum())
                    norm_t = gamma_t / (sum_t + eps)
                    norm_v = gamma_v / (sum_v + eps)
                    abs_t = np.abs(gamma_t) / (sum_t + eps)
                    abs_v = np.abs(gamma_v) / (sum_v + eps)
                    order_values = np.arange(4, dtype=float)
                    eff_t = float(np.dot(order_values, np.abs(gamma_t)) / (sum_t + eps))
                    eff_v = float(np.dot(order_values, np.abs(gamma_v)) / (sum_v + eps))
                    cosine = float(np.dot(norm_t, norm_v) / ((np.linalg.norm(norm_t) * np.linalg.norm(norm_v)) + eps))
                    gamma_rows.append({
                        "dataset": dataset,
                        "variant": variant,
                        "run_id": run_id - 1,
                        "seed": int(payload["seed"]),
                        "gamma_text": gamma_t.tolist(),
                        "gamma_visual": gamma_v.tolist(),
                        "gamma_text_signed_l1": norm_t.tolist(),
                        "gamma_visual_signed_l1": norm_v.tolist(),
                        "gamma_text_absolute_mass": abs_t.tolist(),
                        "gamma_visual_absolute_mass": abs_v.tolist(),
                        "effective_order_text": eff_t,
                        "effective_order_visual": eff_v,
                        "l1_profile_distance": float(np.abs(norm_t - norm_v).sum()),
                        "cosine_similarity": cosine,
                    })

                    for modality, states, gamma in (("text", states_t, model.gamma_text), ("visual", states_v, model.gamma_visual)):
                        contribution = torch.stack([
                            (gamma[k] * states[k]).norm(p=2, dim=-1).mean()
                            for k in range(4)
                        ]).detach().cpu().numpy().astype(float)
                        normalized_contribution = contribution / (contribution.sum() + eps)
                        for order in range(4):
                            contribution_rows.append({
                                "dataset": dataset,
                                "variant": variant,
                                "run_id": run_id - 1,
                                "seed": int(payload["seed"]),
                                "modality": modality,
                                "order": order,
                                "mean_node_l2_contribution": float(contribution[order]),
                                "normalized_contribution": float(normalized_contribution[order]),
                            })
                        for k in range(4):
                            for q in range(4):
                                mean_cosine = F.cosine_similarity(states[k], states[q], dim=-1).mean().item()
                                redundancy_rows.append({
                                    "dataset": dataset,
                                    "variant": variant,
                                    "run_id": run_id - 1,
                                    "seed": int(payload["seed"]),
                                    "modality": modality,
                                    "order_k": k,
                                    "order_q": q,
                                    "mean_nodewise_cosine": float(mean_cosine),
                                })

                    z_base = analysis["fused_z"]
                    pred_base = classifier(z_base).argmax(dim=-1).detach().cpu()
                    base_acc = float((pred_base[data.val_idx] == data.y[data.val_idx]).float().mean().item())
                    base_f1 = _macro_f1(data.y, pred_base, data.val_idx, eval_labels)
                    stored_val = payload["metrics"]
                    if abs(base_acc - float(stored_val["val_acc"])) > 1e-6 or abs(base_f1 - float(stored_val["val_macro_f1"])) > 1e-6:
                        raise ValueError(f"Recomputed validation metrics disagree with {dataset}/{variant}/run{run_id}")
                    sensitivity_rows.append({
                        "dataset": dataset, "variant": variant, "run_id": run_id - 1,
                        "seed": int(payload["seed"]), "order": "baseline",
                        "modality_perturbation": "none", "val_acc": base_acc,
                        "val_macro_f1": base_f1, "delta_val_acc": 0.0, "delta_val_macro_f1": 0.0,
                        "interpretation": "FROZEN_SENSITIVITY",
                    })
                    original_t = model.gamma_text.detach().clone()
                    original_v = model.gamma_visual.detach().clone()
                    perturbations = []
                    for order in range(4):
                        perturbations.extend([
                            (order, "text", True, False),
                            (order, "visual", False, True),
                            (order, "both", True, True),
                        ])
                    for order, label, zero_t, zero_v in perturbations:
                        model.gamma_text.copy_(original_t)
                        model.gamma_visual.copy_(original_v)
                        if zero_t:
                            model.gamma_text[order] = 0.0
                        if zero_v:
                            model.gamma_visual[order] = 0.0
                        z_t = model._readout(states_t, model.gamma_text)
                        z_v = model._readout(states_v, model.gamma_visual)
                        _, _, _, fused = model._fuse_modalities(z_t, z_v)
                        predictions = classifier(fused).argmax(dim=-1).detach().cpu()
                        acc = float((predictions[data.val_idx] == data.y[data.val_idx]).float().mean().item())
                        f1 = _macro_f1(data.y, predictions, data.val_idx, eval_labels)
                        sensitivity_rows.append({
                            "dataset": dataset, "variant": variant, "run_id": run_id - 1,
                            "seed": int(payload["seed"]), "order": order,
                            "modality_perturbation": label, "val_acc": acc,
                            "val_macro_f1": f1, "delta_val_acc": acc - base_acc,
                            "delta_val_macro_f1": f1 - base_f1,
                            "interpretation": "FROZEN_SENSITIVITY",
                        })
                    model.gamma_text.copy_(original_t)
                    model.gamma_visual.copy_(original_v)
                del model, classifier, analysis
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    return {
        "gamma_rows": gamma_rows,
        "contribution_rows": contribution_rows,
        "redundancy_rows": redundancy_rows,
        "sensitivity_rows": sensitivity_rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _pct(mean: float, std: float) -> str:
    return f"{100 * mean:.2f} ± {100 * std:.2f}"


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# NC factorial and baseline benchmark report",
        "",
        "NC checkpoints are selected only by validation accuracy. Test metrics are descriptive.",
        "All means and standard deviations use seeds 42, 43, and 44; standard deviation is population SD.",
        "",
        "## Multi-Order factorial metrics",
        "",
        "| Dataset | Variant | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset, variants in summary["factorial"].items():
        for variant, record in variants.items():
            metrics = record["metrics"]
            lines.append(
                f"| {dataset} | {variant} | "
                + " | ".join(_pct(metrics[name]["mean"], metrics[name]["std"]) for name in METRICS)
                + " |"
            )
    lines.extend(["", "## Preregistered paired contrasts", "", "Positive differences favor the left-hand variant.", ""])
    lines.extend(["| Contrast | Mean Val Acc difference (pp) | Positive datasets | Positive seed pairs | Support |", "|---|---:|---:|---:|---|"])
    for label, record in summary["paired_contrasts"].items():
        value = record["val_acc_summary"]
        lines.append(
            f"| {label} | {value['mean_pp']:.3f} | {value['positive_datasets']}/{value['dataset_count']} | "
            f"{value['positive_seed_pairs']}/{value['seed_pair_count']} | {value['support']} |"
        )
    lines.extend(["", "## Factorial decisions", ""])
    lines.append(f"- Fusion attribution trigger: {'FUSION_ATTRIBUTION_REQUIRED' if summary['fusion_attribution_required'] else 'not triggered'}.")
    for label, robust in summary.get("fusion_robust_multi_order", {}).items():
        lines.append(f"- Fusion-robust direction for {label}: {robust}.")
    lines.extend([
        "- Practical support labels are descriptive preregistered rules, not statistical significance claims.",
        "- Text/Visual GPR profiles, actual order contributions, hop cosine matrices, and frozen sensitivity are in the companion CSV files.",
        "- Frozen sensitivity sets a selected gamma coefficient to zero without retraining; it is not a retrained ablation or a causal necessity test.",
        "",
    ])
    if summary.get("baselines"):
        lines.extend(["## NC baseline summary", "", "| Dataset | Model | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 |", "|---|---|---:|---:|---:|---:|"])
        for dataset, models in summary["baselines"].items():
            for model, record in models.items():
                metrics = record["metrics"]
                lines.append(
                    f"| {dataset} | {model} | "
                    + " | ".join(_pct(metrics[name]["mean"], metrics[name]["std"]) for name in METRICS)
                    + " |"
                )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the frozen NC factorial and baseline benchmarks.")
    parser.add_argument("--factorial-root", type=Path, default=Path("outputs/mob_factorial_nc_v1"))
    parser.add_argument("--baseline-root", type=Path, default=Path("outputs/nc_benchmark_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("results/nc_benchmark_v1"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--skip-diagnostics", action="store_true")
    parser.add_argument("--without-baselines", action="store_true", help="summarize the factorial before the baseline benchmark")
    args = parser.parse_args()
    baseline_root = None if args.without_baselines else args.baseline_root
    summary = analyze(args.factorial_root, tuple(args.datasets), baseline_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for key, filename in (("per_run_rows", "nc_per_run.csv"), ("baseline_rows", "nc_baseline_table.csv"), ("factorial_rows", "mob_factorial_table.csv"), ("contrast_rows", "mob_paired_contrasts.csv")):
        _write_csv(args.output_root / filename, summary.pop(key))
    if not args.skip_diagnostics:
        diagnostics = collect_gpr_diagnostics(args.factorial_root, tuple(args.datasets), args.device)
        _write_csv(args.output_root / "mob_gamma_profiles.csv", diagnostics["gamma_rows"])
        _write_csv(args.output_root / "mob_order_contribution.csv", diagnostics["contribution_rows"])
        _write_csv(args.output_root / "mob_order_redundancy.csv", diagnostics["redundancy_rows"])
        _write_csv(args.output_root / "mob_order_frozen_sensitivity.csv", diagnostics["sensitivity_rows"])
    if summary["fusion_attribution_required"]:
        (args.output_root / "fusion_attribution.csv").write_text(
            "FUSION_ATTRIBUTION_REQUIRED; conditional variants must be implemented and run before reporting component effects.\n",
            encoding="utf-8",
        )
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_root / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(f"Wrote benchmark tables and report to {args.output_root}")


if __name__ == "__main__":
    main()
