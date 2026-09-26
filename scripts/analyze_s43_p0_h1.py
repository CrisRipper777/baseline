from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from sklearn.metrics import f1_score

from src.data import load_mag_data
from src.models import build_model

DATASETS = ("Movies", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
VARIANTS = (
    "p0_relation_only", "p0_residual", "p0_concat", "h1_dual_agg",
    "h1_dual_functional_static", "h1_dual_functional_global",
)
OUTPUT_ROOT = ROOT / "outputs/s43_p0_h1_v1"
RESULT_ROOT = ROOT / "results/s43_p0_h1_v1"


def _read_complete(dataset: str, variant: str) -> dict[str, Any]:
    path = OUTPUT_ROOT / "formal" / dataset / variant / "complete.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing formal job completion record: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("test_evaluation") is not False or record.get("protocol_version") != "unified_full_graph_nc_v1":
        raise ValueError(f"invalid or test-contaminated record {path}")
    if record.get("run_seeds") != list(SEEDS) or record.get("selection") != "best_val_accuracy":
        raise ValueError(f"wrong seed/selection metadata in {path}")
    return record


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def _std(values: list[float]) -> float:
    return statistics.pstdev(values) if values else math.nan


def _seed_metrics(record: dict[str, Any]) -> dict[tuple[str, int], dict[str, float]]:
    result = {}
    for run in record["run_metrics"]["runs"]:
        seed = int(run["seed"])
        metrics = run["metrics"]
        if set(metrics) != {"val_acc", "val_macro_f1"}:
            raise ValueError("formal run metrics must contain validation accuracy and macro-F1 only")
        result[(record["dataset"], seed)] = {
            "val_acc": float(metrics["val_acc"]),
            "val_macro_f1": float(metrics["val_macro_f1"]),
        }
    return result


def _paired_contrast(
    rows: list[dict[str, Any]], name: str, left: dict[tuple[str, int], dict[str, float]],
    right: dict[tuple[str, int], dict[str, float]],
) -> list[dict[str, Any]]:
    pairs = []
    for key in sorted(left):
        if key not in right:
            raise ValueError(f"unpaired run for {name}: {key}")
        pairs.append({
            "dataset": key[0], "seed": key[1], "contrast": name,
            "delta_val_acc": right[key]["val_acc"] - left[key]["val_acc"],
            "delta_val_macro_f1": right[key]["val_macro_f1"] - left[key]["val_macro_f1"],
        })
    rows.extend(pairs)
    for metric, source in (("val_acc", "delta_val_acc"), ("val_macro_f1", "delta_val_macro_f1")):
        deltas = [float(row[source]) for row in pairs]
        summary = {
            "dataset": "ALL", "seed": "ALL", "contrast": name,
            f"mean_delta_{metric}": _mean(deltas), f"population_std_delta_{metric}": _std(deltas),
            f"positive_seed_pairs_{metric}": sum(value > 0 for value in deltas),
            f"positive_dataset_means_{metric}": sum(
                _mean([float(row[source]) for row in pairs if row["dataset"] == dataset]) > 0
                for dataset in DATASETS
            ),
        }
        rows.append(summary)
    return pairs


def _table_rows(records: dict[tuple[str, str], dict[str, Any]], variants: tuple[str, ...]):
    rows = []
    all_runs = {}
    for dataset in DATASETS:
        for variant in variants:
            record = records[(dataset, variant)]
            per_seed = _seed_metrics(record)
            all_runs[(dataset, variant)] = per_seed
            for metric in ("val_acc", "val_macro_f1"):
                values = [run[metric] for run in per_seed.values()]
                rows.append({
                    "dataset": dataset, "variant": variant, "metric": metric,
                    "n_runs": len(values), "mean": _mean(values), "population_std": _std(values),
                    "trainable_params": record["trainable_params"],
                })
    return rows, all_runs


def _paired_index(all_runs, dataset: str, variant: str):
    return all_runs[(dataset, variant)]


def _load_dataset_and_model(dataset: str, variant: str, seed: int, checkpoint_path: Path):
    from src.tasks.nc import _resolve_nc_eval_labels

    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=[
            f"dataset={dataset}", "task=nc", "model=relation_basis_pilot",
            f"model.variant={variant}", "model.hidden_dim=256", "model.dropout=0.2",
            "seed=42", "num_runs=1", "device=cpu", "task.evaluate_test=false",
        ])
        # Formal seeds vary initialization only; the frozen dataset split is seed 42.
        data = load_mag_data(cfg, "nc", 42)
    data_info = {
        "input_dim": data.input_dim, "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("task") != "nc" or payload.get("protocol_version") != "unified_full_graph_nc_v1":
        raise ValueError(f"invalid checkpoint {checkpoint_path}")
    if int(payload.get("seed", -1)) != seed or payload.get("selection") != "best_val_accuracy":
        raise ValueError(f"wrong checkpoint selection metadata in {checkpoint_path}")
    if any(key.startswith("test_") for key in payload.get("metrics", {})):
        raise ValueError(f"test metrics found in checkpoint {checkpoint_path}")
    model = build_model(cfg, data_info)
    model.load_state_dict(payload["model_state"])
    classifier = nn.Linear(model.out_dim, int(data.num_classes))
    classifier.load_state_dict(payload["head_state"])
    model.eval(); classifier.eval()
    return cfg, data, model, classifier, _resolve_nc_eval_labels(data)


def _evaluate_logits(logits: torch.Tensor, data, labels: list[int]) -> dict[str, float]:
    idx = data.val_idx.detach().cpu().long()
    targets = data.y[idx].detach().cpu().long()
    pred = logits[idx].argmax(dim=-1).detach().cpu()
    return {
        "val_acc": float((pred == targets).float().mean()),
        "val_macro_f1": float(f1_score(targets.numpy(), pred.numpy(), labels=labels,
                                       average="macro", zero_division=0)),
        "true_label_ce": float(F.cross_entropy(logits[idx], targets)),
    }


def _functional_diagnostics(records: dict[tuple[str, str], dict[str, Any]]):
    diagnostics, interventions, alphas = [], [], []
    for dataset in DATASETS:
        for variant in ("h1_dual_functional_static", "h1_dual_functional_global"):
            for seed in SEEDS:
                checkpoint = OUTPUT_ROOT / "formal" / dataset / variant / f"best_run{SEEDS.index(seed) + 1}.pt"
                _, data, model, classifier, labels = _load_dataset_and_model(dataset, variant, seed, checkpoint)
                with torch.no_grad():
                    normal = model.analyze(data.x, data.edge_index)
                    normal_logits = classifier(normal["fused_z"])
                    normal_metrics = _evaluate_logits(normal_logits, data, labels)
                    val_idx = data.val_idx.detach().cpu().long()
                    modality_stats = {}
                    for modality in ("text", "visual"):
                        ra = normal[f"R_A_{modality}"][val_idx]
                        rd = normal[f"R_D_{modality}"][val_idx]
                        norm_a = torch.linalg.vector_norm(ra, dim=-1)
                        norm_d = torch.linalg.vector_norm(rd, dim=-1)
                        cosine = F.cosine_similarity(ra, rd, dim=-1, eps=1e-8)
                        mean_a, mean_d = float(norm_a.mean()), float(norm_d.mean())
                        mean_cos = float(cosine.mean())
                        row = {
                            "dataset": dataset, "variant": variant, "seed": seed,
                            "modality": modality, "mean_norm_R_A": mean_a, "mean_norm_R_D": mean_d,
                            "mean_cosine_R_A_R_D": mean_cos,
                            "std_cosine_R_A_R_D": float(cosine.std(unbiased=False)),
                            "R_A_to_R_D_mean_norm_ratio": mean_a / mean_d if mean_d > 0 else math.inf,
                            "flag_nan_or_inf": not bool(torch.isfinite(ra).all() and torch.isfinite(rd).all()),
                            "flag_zero_basis": mean_a == 0.0 or mean_d == 0.0,
                            "flag_cosine_near_one": abs(mean_cos - 1.0) <= 1e-3,
                        }
                        diagnostics.append(row)
                        modality_stats[modality] = row
                    for mode, intervention, swap in (
                        ("NORMAL", "normal", False), ("FORCE_A", "force_a", False),
                        ("FORCE_D", "force_d", False),
                    ):
                        result = normal if mode == "NORMAL" else model.analyze(
                            data.x, data.edge_index, basis_intervention=intervention
                        )
                        logits = normal_logits if mode == "NORMAL" else classifier(result["fused_z"])
                        metrics = _evaluate_logits(logits, data, labels)
                        pred = logits[val_idx].argmax(dim=-1)
                        flips = 0.0 if mode == "NORMAL" else float(
                            (pred != normal_logits[val_idx].argmax(dim=-1)).float().mean()
                        )
                        interventions.append({
                            "dataset": dataset, "variant": variant, "seed": seed, "intervention": mode,
                            **metrics, "delta_val_acc": metrics["val_acc"] - normal_metrics["val_acc"],
                            "delta_val_macro_f1": metrics["val_macro_f1"] - normal_metrics["val_macro_f1"],
                            "delta_true_label_ce": metrics["true_label_ce"] - normal_metrics["true_label_ce"],
                            "prediction_flip_rate_vs_normal": flips,
                            "interpretation": "frozen sensitivity; no retraining",
                        })
                    if variant == "h1_dual_functional_global":
                        result = model.analyze(data.x, data.edge_index, swap_modality_alpha=True)
                        logits = classifier(result["fused_z"])
                        metrics = _evaluate_logits(logits, data, labels)
                        pred = logits[val_idx].argmax(dim=-1)
                        normal_pred = normal_logits[val_idx].argmax(dim=-1)
                        interventions.append({
                            "dataset": dataset, "variant": variant, "seed": seed,
                            "intervention": "SWAP_MODALITY_ALPHA", **metrics,
                            "delta_val_acc": metrics["val_acc"] - normal_metrics["val_acc"],
                            "delta_val_macro_f1": metrics["val_macro_f1"] - normal_metrics["val_macro_f1"],
                            "delta_true_label_ce": metrics["true_label_ce"] - normal_metrics["true_label_ce"],
                            "prediction_flip_rate_vs_normal": float((pred != normal_pred).float().mean()),
                            "interpretation": "frozen sensitivity; no retraining",
                        })
                        alphas.append({
                            "dataset": dataset, "seed": seed,
                            "alpha_text_A": float(normal["alpha_text"][0]),
                            "alpha_text_D": float(normal["alpha_text"][1]),
                            "alpha_visual_A": float(normal["alpha_visual"][0]),
                            "alpha_visual_D": float(normal["alpha_visual"][1]),
                            "interpretation": "learned coefficient != causal contribution; read with FORCE_A/FORCE_D",
                        })
                del data, model, classifier
    return diagnostics, interventions, alphas


def _contrast_summary(rows: list[dict[str, Any]], name: str):
    return [row for row in rows if row.get("contrast") == name and row.get("dataset") == "ALL"]


def _status_p0(pairs: list[dict[str, Any]]) -> tuple[str, str]:
    acc_by_ds = {dataset: _mean([r["delta_val_acc"] for r in pairs if r["dataset"] == dataset]) for dataset in DATASETS}
    f1_by_ds = {dataset: _mean([r["delta_val_macro_f1"] for r in pairs if r["dataset"] == dataset]) for dataset in DATASETS}
    if all(value > 0 for value in acc_by_ds.values()) and all(value > 0 for value in f1_by_ds.values()):
        return "STRONG_SUPPORT", "Primary residual-minus-relation-only validation deltas are positive for both metrics on all four datasets."
    if any(value > 0 for value in acc_by_ds.values()) or any(value > 0 for value in f1_by_ds.values()):
        return "MIXED", "Primary validation deltas vary by metric or dataset."
    return "UNSUPPORTED", "Primary validation deltas are not positive at the dataset-mean level."


def _status_h1(pairs: list[dict[str, Any]], diagnostics: list[dict[str, Any]]):
    acc = _mean([row["delta_val_acc"] for row in pairs])
    f1 = _mean([row["delta_val_macro_f1"] for row in pairs])
    nondegenerate = bool(diagnostics) and all(
        not row["flag_nan_or_inf"] and not row["flag_zero_basis"] and not row["flag_cosine_near_one"]
        for row in diagnostics
    )
    ds_positive = all(
        _mean([r["delta_val_acc"] for r in pairs if r["dataset"] == dataset]) > 0 and
        _mean([r["delta_val_macro_f1"] for r in pairs if r["dataset"] == dataset]) > 0
        for dataset in DATASETS
    )
    if ds_positive and nondegenerate:
        return "STRONG_SUPPORT", "Capacity-matched static functional-basis deltas are positive by dataset for both validation metrics and diagnostics are non-degenerate."
    if acc >= 0 and f1 >= 0 and nondegenerate:
        return "MECHANISM_SUPPORT", "Mean validation deltas are non-negative and frozen functional diagnostics are non-degenerate; dataset/seed consistency is weaker."
    if any(row["delta_val_acc"] > 0 or row["delta_val_macro_f1"] > 0 for row in pairs) or nondegenerate:
        return "MIXED", "Performance direction and/or functional diagnostics are mixed."
    return "UNSUPPORTED", "Neither directional validation evidence nor non-degenerate functional diagnostics were observed."


def main():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    records = {(dataset, variant): _read_complete(dataset, variant)
               for dataset in DATASETS for variant in VARIANTS}
    p0_table, p0_runs = _table_rows(records, ("p0_relation_only", "p0_residual", "p0_concat"))
    h1_table, h1_runs = _table_rows(
        records, ("p0_residual", "h1_dual_agg", "h1_dual_functional_static", "h1_dual_functional_global")
    )
    _write_csv(RESULT_ROOT / "p0_table.csv", p0_table)
    _write_csv(RESULT_ROOT / "h1_table.csv", h1_table)

    p0_contrasts: list[dict[str, Any]] = []
    p0_pair_map = {}
    for name, left, right in (
        ("p0_residual - p0_relation_only", "p0_relation_only", "p0_residual"),
        ("p0_concat - p0_relation_only", "p0_relation_only", "p0_concat"),
        ("p0_concat - p0_residual", "p0_residual", "p0_concat"),
    ):
        l, r = {}, {}
        for dataset in DATASETS:
            l.update(_paired_index(p0_runs, dataset, left)); r.update(_paired_index(p0_runs, dataset, right))
        p0_pair_map[name] = _paired_contrast(p0_contrasts, name, l, r)
    _write_csv(RESULT_ROOT / "p0_paired_contrasts.csv", p0_contrasts)

    h1_contrasts: list[dict[str, Any]] = []
    h1_pair_map = {}
    for name, left, right in (
        ("dual_agg - single_agg", "p0_residual", "h1_dual_agg"),
        ("dual_functional_static - dual_agg", "h1_dual_agg", "h1_dual_functional_static"),
        ("dual_functional_global - dual_functional_static", "h1_dual_functional_static", "h1_dual_functional_global"),
        ("dual_functional_global - single_agg", "p0_residual", "h1_dual_functional_global"),
    ):
        l, r = {}, {}
        for dataset in DATASETS:
            l.update(_paired_index(h1_runs, dataset, left)); r.update(_paired_index(h1_runs, dataset, right))
        h1_pair_map[name] = _paired_contrast(h1_contrasts, name, l, r)
    _write_csv(RESULT_ROOT / "h1_paired_contrasts.csv", h1_contrasts)

    diagnostics, interventions, alphas = _functional_diagnostics(records)
    _write_csv(RESULT_ROOT / "h1_basis_diagnostics.csv", diagnostics)
    _write_csv(RESULT_ROOT / "h1_forced_basis_interventions.csv", interventions)
    _write_csv(RESULT_ROOT / "h1_global_coefficients.csv", alphas)

    complexity = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            row = records[(dataset, variant)]
            complexity.append({
                "dataset": dataset, "variant": variant,
                "trainable_params": row["trainable_params"],
                "model_trainable_params": row["model_trainable_params"],
                "classifier_trainable_params": row["classifier_trainable_params"],
                "peak_cuda_memory_mb": row["peak_cuda_memory_mb"],
                "peak_device_memory_mb": row["peak_device_memory_mb"],
                "wall_clock_seconds": row["wall_clock_seconds"],
                "mean_epoch_seconds_proxy": row["mean_epoch_seconds_proxy"],
                "best_epoch_mean": _mean(row["best_epochs"]),
                "actual_epochs": sum(row["actual_epochs"].values()),
            })
    _write_csv(RESULT_ROOT / "complexity_table.csv", complexity)

    p0_status, p0_reason = _status_p0(p0_pair_map["p0_residual - p0_relation_only"])
    primary_h1 = h1_pair_map["dual_functional_static - dual_agg"]
    h1_status, h1_reason = _status_h1(primary_h1, [
        row for row in diagnostics if row["variant"] == "h1_dual_functional_static"
    ])
    job_index = json.loads((OUTPUT_ROOT / "formal_jobs.json").read_text(encoding="utf-8"))
    completed = sum(1 for record in records.values() if record.get("checkpoint_validation") == "passed")
    total_runs = sum(int(row["num_runs"]) for row in records.values())
    failures = job_index.get("failed", []) + job_index.get("not_started", [])
    summary = {
        "study": "S4.3 Hypothesis Validation Pilots Phase 1: P0 + H1",
        "protocol_version": "unified_full_graph_nc_v1", "datasets": list(DATASETS),
        "variants": list(VARIANTS), "test_evaluation": False,
        "formal_jobs_expected": 24, "formal_jobs_checkpoint_validated": completed,
        "formal_runs_expected": 72, "formal_runs_present": total_runs,
        "failed_or_not_started_jobs": failures,
        "p0": {"status": p0_status, "reason": p0_reason,
               "primary_contrast": "p0_residual - p0_relation_only",
               "concat_interpretation": "CAPACITY_DIFFERENT_SECONDARY_CONTROL"},
        "h1": {"status": h1_status, "reason": h1_reason,
               "primary_contrast": "h1_dual_functional_static - h1_dual_agg",
               "interpretation_boundary": "Functional structural basis diversity is tested against repeated aggregative capacity; this does not establish two relation types because U_D=sH-U_A is linearly related to H and U_A."},
        "global_coefficient_interpretation": "learned coefficient != causal contribution; interpret with FORCE_A/FORCE_D frozen sensitivity",
        "interventions_interpretation": "frozen sensitivity; no retraining; not a causal retrained ablation",
        "mean_epoch_seconds_note": "job wall time divided by observed completed epoch log lines; includes setup and validation overhead",
        "results_root": str(RESULT_ROOT.relative_to(ROOT)),
    }
    (RESULT_ROOT / "s43_p0_h1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# S4.3 P0/H1 pilot report", "",
        f"- Formal jobs: {completed}/24 checkpoint-validated; formal runs: {total_runs}/72.",
        f"- Failed or not-started jobs: {len(failures)}.",
        "- Test evaluation was disabled for every formal job. Selection used best validation accuracy.",
        "- Protocol: unified_full_graph_nc_v1; seed command 42; internal seeds 42/43/44; 300 epochs maximum.",
        "", "## P0: Intrinsic Evidence Preservation", "",
        f"Status: **{p0_status}**. {p0_reason}",
        "Primary comparison: p0_residual minus p0_relation_only, which has matched trainable parameter count.",
        "p0_concat comparisons are marked CAPACITY_DIFFERENT_SECONDARY_CONTROL and cannot alone establish intrinsic preservation.",
        "See p0_table.csv and p0_paired_contrasts.csv for validation accuracy and macro-F1 by dataset and paired seed.",
        "", "## H1: Functional Relational Basis Diversity", "",
        f"Status: **{h1_status}**. {h1_reason}",
        "Primary comparison: h1_dual_functional_static minus h1_dual_agg. Both use two transforms and have matched trainable parameter counts; only structural basis inputs differ.",
        "This tests functional structural basis diversity against repeated aggregative capacity. It does not establish two relation types: U_D=sH-U_A is linearly related to H and U_A.",
        "See h1_table.csv, h1_paired_contrasts.csv, and h1_basis_diagnostics.csv.",
        "", "## Frozen interventions and global coefficients", "",
        "FORCE_A, FORCE_D, and SWAP_MODALITY_ALPHA were evaluated on the selected checkpoints without retraining. These are frozen sensitivity checks, not causal retrained ablations.",
        "Learned global coefficients are not causal contributions and must be interpreted alongside the FORCE_A/FORCE_D results.",
        "See h1_forced_basis_interventions.csv and h1_global_coefficients.csv.",
        "", "## Complexity and QA", "",
        "complexity_table.csv records parameter counts, sampled peak CUDA memory, job wall time, best epoch, and job-wall-time/observed-epochs as a mean-epoch proxy.",
        "Peak CUDA memory uses nvidia-smi sampling on a physical GPU with at most one active job per GPU.",
        "The epoch-time value includes startup and validation overhead; it is a proxy, not a per-epoch timer.",
        "Formal job records validate checkpoint task/protocol/seed/selection, validation metrics, and absence of test metrics.",
        "", "## Provenance", "",
        f"Branch/commit and source hashes are recorded in outputs/s43_p0_h1_v1/provenance.json. Results: `{RESULT_ROOT.relative_to(ROOT)}`.",
        "",
    ]
    (RESULT_ROOT / "s43_p0_h1_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"jobs": completed, "runs": total_runs, "p0_status": p0_status,
                      "h1_status": h1_status, "results": str(RESULT_ROOT)}, indent=2))


if __name__ == "__main__":
    main()
