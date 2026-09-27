from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from scripts import analyze_s45_relcal_statepres as base

ROOT = base.ROOT
DATASETS, SEEDS, VARIANTS = base.DATASETS, base.SEEDS, base.VARIANTS
OUTPUT_ROOT, RESULT_ROOT, G0_ROOT = base.OUTPUT_ROOT, base.RESULT_ROOT, base.G0_ROOT
_load_data, _load_model = base._load_data, base._load_model
_metrics, _eval_model, _delta_row = base._metrics, base._eval_model, base._delta_row
_gate_summary, _row_operator_audit = base._gate_summary, base._row_operator_audit
_shuffle_by_target, _geometry_rows = base._shuffle_by_target, base._geometry_rows
_paired_contrasts, _interaction_rows, _statuses = base._paired_contrasts, base._interaction_rows, base._statuses
_write_csv, _file_sha256 = base._write_csv, base._file_sha256
run_granularity_audit = base.run_granularity_audit


def _dataset_hashes(dataset: str) -> dict[str, str]:
    from src.data.loaders import resolve_path
    cfg = base._config(dataset, "relcal_statepres_pilot", VARIANTS[0])
    keys = ("graph_path", "text_feat_path", "image_feat_path", "joint_feat_path",
            "edge_path", "label_path", "node_split_path", "nc_split_path")
    result = {}
    for key in keys:
        value = cfg.dataset.get(key)
        if value is None:
            continue
        path = resolve_path(str(value))
        if path.is_file():
            result[str(path)] = _file_sha256(path)
    return result


def _historical_mob_timing(dataset: str) -> float:
    path = ROOT / "outputs/mob_factorial_nc_v1" / dataset / "mob_uniform_plain" / "complete.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    main_log = (ROOT / record["results"]).parent / "main.log"
    if not main_log.is_file():
        return math.nan
    epochs = sum("Epoch " in line and "| Train Loss" in line
                 for line in main_log.read_text(encoding="utf-8", errors="replace").splitlines())
    return float(record["seconds"]) / epochs if epochs else math.nan


def _all_contrast_rows(rows: list[dict[str, Any]], datasets: tuple[str, ...]) -> list[dict[str, Any]]:
    names = sorted({str(row["contrast"]) for row in rows})
    for name in names:
        paired = [row for row in rows if row["contrast"] == name and row.get("seed") != "ALL"]
        summary = {"contrast": name, "dataset": "ALL", "seed": "ALL"}
        for metric in ("val_acc", "val_macro_f1", "true_label_ce"):
            vals = [float(row[f"delta_{metric}"]) for row in paired]
            summary[f"mean_delta_{metric}"] = statistics.fmean(vals)
            summary[f"population_std_delta_{metric}"] = statistics.pstdev(vals)
            summary[f"positive_seed_pairs_{metric}"] = sum(value > 0 for value in vals)
            summary[f"positive_dataset_means_{metric}"] = sum(
                statistics.fmean(float(row[f"delta_{metric}"]) for row in paired
                                 if row["dataset"] == dataset) > 0 for dataset in datasets)
        rows.append(summary)
    return rows


def _all_interaction_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"dataset": "ALL", "seed": "ALL"}
    paired = [row for row in rows if row.get("seed") != "ALL"]
    for metric in ("val_acc", "val_macro_f1", "true_label_ce"):
        vals = [float(row[f"interaction_{metric}"]) for row in paired]
        summary[f"mean_interaction_{metric}"] = statistics.fmean(vals)
        summary[f"population_std_interaction_{metric}"] = statistics.pstdev(vals)
        summary[f"positive_seed_pairs_{metric}"] = sum(value > 0 for value in vals)
    rows.append(summary)
    return summary


def _historical_identity_crosscheck(table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reference_map = {
        "s45_identity_terminal": "mob_terminal_plain",
        "s45_identity_uniform": "mob_uniform_plain",
    }
    refs = {}
    with (ROOT / "results/nc_benchmark_v1/nc_per_run.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("model") == "multi_order_bank":
                refs[(row["dataset"], int(row["seed"]), row["variant"])] = row
    output = []
    for row in table:
        variant = row["variant"]
        if variant not in reference_map:
            continue
        key = (row["dataset"], int(row["seed"]), reference_map[variant])
        old = refs.get(key)
        if old is None:
            continue
        da = float(row["val_acc"]) - float(old["val_acc"])
        df = float(row["val_macro_f1"]) - float(old["val_macro_f1"])
        output.append({
            "dataset": row["dataset"], "seed": row["seed"], "variant": variant,
            "historical_variant": reference_map[variant], "val_acc": row["val_acc"],
            "historical_val_acc": float(old["val_acc"]), "delta_val_acc": da,
            "val_macro_f1": row["val_macro_f1"], "historical_val_macro_f1": float(old["val_macro_f1"]),
            "delta_val_macro_f1": df, "max_abs_metric_delta": max(abs(da), abs(df)),
            "reference": "historical MOB plain checkpoint result table; descriptive cross-check",
        })
    return output


def _report(summary: dict[str, Any]) -> str:
    c = summary["primary_contrasts"]
    i = summary["factorial_interaction"]
    s = summary["interpretation_statuses"]
    lines = [
        "# S4.5 Relation-Calibrated State-Preserving Propagation", "",
        f"Training commit: `{summary['training_commit']}`; analysis commit: `{summary['analysis_commit']}`; source branch/SHA: `s44_relational_transform` / `{base.SOURCE_SHA}`.",
        "Scope: unified full-graph node classification on Movies, Grocery, ele-fashion and Reddit-S; seeds 42–44; best-validation-accuracy selection; test disabled. Toys remains an architecture holdout. No LP results are included.", "",
        "## Findings", "",
        f"- ControllerGranularity: **{s['ControllerGranularity']['status']}**. Overall mean within-target fraction {s['ControllerGranularity']['mean_within_ratio']:.4f}; family means: " + "; ".join(
            f"{name.removeprefix('s44_')} {values['mean_within_ratio']:.4f} (SD {values['population_std_within_ratio']:.4f})"
            for name, values in s['ControllerGranularity']['mean_within_ratio_by_variant'].items()) + ". Matched scalar advantage: " + "; ".join(
            f"vs {name.removeprefix('scalar_minus_')}: {values['mean_delta_within_ratio']:.4f}, positive {values['positive_dataset_seed_modality_pairs']}/{values['n']}"
            for name, values in s['ControllerGranularity']['matched_scalar_comparisons'].items()) + ". This describes controller assignment, not task utility.",
        f"- RelationRedistribution: **{s['RelationRedistribution']['status']}**. Mass-preserving entry uniform minus identity uniform: {c['masspres_entry_uniform_minus_identity_uniform']}.",
        f"- StatePreservation: **{s['StatePreservation']['status']}**. Identity uniform minus terminal: {c['identity_uniform_minus_identity_terminal']}; calibrated uniform minus terminal: {c['masspres_entry_uniform_minus_masspres_entry_terminal']}.",
        f"- CalibrationStateInteraction: **{s['CalibrationStateInteraction']['status']}**. Factorial interaction in accuracy: mean {i['mean_interaction_val_acc']:.6f}, population SD {i['population_std_interaction_val_acc']:.6f}, positive seed pairs {i['positive_seed_pairs_val_acc']}/12.",
        f"- CalibrationPlacement: **{s['CalibrationPlacement']['status']}**. Persistent minus entry-only uniform: {c['masspres_persistent_minus_masspres_entry_uniform']}.", "",
        "## Interpretation boundaries", "",
        "Row-mass preservation guarantees only each target's one-step off-diagonal mass. It does not guarantee symmetry, spectral equivalence, the same smoothing spectrum, or the same stationary distribution. Frozen interventions are checkpoint sensitivities, not retrained causal ablations. R1 is a fixed carrier, not a proven superior or novel module. Gate magnitude is not causal relation utility. A positive calibrated score does not establish synergy; inspect I. These NC development results do not generalize to LP. Toys was not used.", "",
        "Historical MOB terminal/uniform metrics are cross-checked in `historical_mob_crosscheck.csv`; identity discrepancies above 1e-4 stop interpretation pending audit. Identity propagated-uniform equivalence is covered by the exact unit test because the historical aggregate table has no propagated-uniform run.", "",
        "This pilot does not by itself freeze the paper backbone. Review the raw paired results, interventions, state diagnostics, identity checks, and factorial interaction before freezing relation → propagation → state composition.", "",
        "All detailed tables preserve dataset-seed rows; interpretation labels do not replace the raw evidence.",
    ]
    return "\n".join(lines) + "\n"


def run_analysis(datasets: tuple[str, ...] = DATASETS, device_name: str | None = None) -> dict[str, Any]:
    from src.tasks.nc import _resolve_nc_eval_labels

    requested = tuple(dataset for dataset in DATASETS if dataset in datasets)
    if not requested or requested != datasets:
        raise ValueError(f"datasets must be a nonempty canonical subset of {DATASETS}")
    if not (G0_ROOT / "granularity_summary.json").is_file():
        raise FileNotFoundError("run phase=granularity before phase=analyze")
    for dataset in requested:
        for variant in VARIANTS:
            path = OUTPUT_ROOT / "formal" / dataset / variant / "complete.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            complete = json.loads(path.read_text(encoding="utf-8"))
            if complete.get("phase") != "formal" or complete.get("test_evaluation") is not False:
                raise ValueError(f"invalid formal record: {path}")
            if complete.get("training_branch") != "s45_relcal_statepres":
                raise ValueError(f"wrong training branch: {path}")

    device = torch.device(device_name or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    table, gate_rows, row_rows, asym_rows = [], [], [], []
    intervention_rows, change_rows, geometry_rows, complexity_rows = [], [], [], []
    metrics_by_run: dict[tuple[str, int, str], dict[str, float]] = {}
    checkpoint_sources, training_commits = {}, set()
    dataset_hashes = {dataset: _dataset_hashes(dataset) for dataset in requested}

    for dataset in requested:
        data = _load_data(dataset)
        labels = _resolve_nc_eval_labels(data)
        x, edge = data.x.to(device), data.edge_index.to(device)
        mob_epoch_proxy = _historical_mob_timing(dataset)
        for seed in SEEDS:
            for variant in VARIANTS:
                model, head, payload, checkpoint = _load_model(dataset, seed, variant, data, device)
                normal, metrics, pred = _eval_model(model, head, data, x, edge, labels)
                for metric in ("val_acc", "val_macro_f1"):
                    if abs(metrics[metric] - float(payload["metrics"][metric])) > 1e-6:
                        raise RuntimeError(f"checkpoint metric mismatch {dataset}/{seed}/{variant}/{metric}")
                complete = json.loads((OUTPUT_ROOT / "formal" / dataset / variant / "complete.json").read_text())
                training_commits.add(complete["training_commit"])
                checkpoint_sources[f"{dataset}/{seed}/{variant}"] = {
                    "path": str(checkpoint), "sha256": _file_sha256(checkpoint),
                    "training_commit": complete["training_commit"],
                }
                metrics_by_run[(dataset, seed, variant)] = metrics
                table.append({"dataset": dataset, "seed": seed, "variant": variant, **metrics,
                              "checkpoint_best_epoch": int(payload["epoch"]),
                              "trainable_params": complete["trainable_params"], "test_evaluation": False})

                for modality in ("text", "visual"):
                    geometry_rows.extend(base._geometry_rows(dataset, seed, variant, modality,
                                                              normal[f"S_{modality}"], data.edge_index))

                if model.calibrated:
                    p, p_self, p_rel = normal["P"], normal["P_self"], normal["P_rel"]
                    rows, weights = p_rel.indices()[0], p_rel.values()
                    for modality in ("text", "visual"):
                        raw, used = normal["raw_gates"][modality], normal["normalized_gates"][modality]
                        gate_rows.append(_gate_summary(dataset, seed, variant, modality, raw, used,
                                                       weights, rows, data.num_nodes))
                        audit = _row_operator_audit(p, p_self, p_rel, modality,
                                                    normal["operators"][modality], dataset, seed, variant)
                        row_rows.append(audit)
                        asym_rows.append({key: audit[key] for key in (
                            "dataset", "seed", "variant", "modality", "operator_asymmetry_frobenius_ratio")})

                    if model.mass_preserving:
                        off, off_metrics, off_pred = _eval_model(model, head, data, x, edge, labels,
                                                                 gate_override="off")
                        intervention_rows.append(_delta_row(
                            dataset, seed, variant, "CALIBRATION_OFF", off_metrics, metrics, off_pred, pred,
                            frozen_sensitivity_not_retrained_causal_ablation=True))
                        for hop in range(1, 4):
                            for modality in ("text", "visual"):
                                calibrated_state, original_state = normal[f"S_{modality}"][hop], off[f"S_{modality}"][hop]
                                cosine = F.cosine_similarity(calibrated_state.float(), original_state.float(), dim=-1).mean()
                                change_rows.append({
                                    "dataset": dataset, "seed": seed, "variant": variant,
                                    "modality": modality, "hop": hop,
                                    "mean_cosine": float(cosine.item()),
                                    "mean_one_minus_cosine": float((1 - cosine).item()),
                                    "mean_normalized_l2": float((calibrated_state - original_state).norm() / (original_state.norm() + 1e-12)),
                                    "interpretation": "descriptive state change, not task utility",
                                })
                        del off, off_metrics, off_pred
                        for replicate in range(10):
                            override = {
                                modality: _shuffle_by_target(
                                    normal["raw_gates"][modality], rows,
                                    450000 + SEEDS.index(seed) * 10000 + VARIANTS.index(variant) * 100 + replicate,
                                ) for modality in ("text", "visual")
                            }
                            shuffled, sm, sp = _eval_model(model, head, data, x, edge, labels,
                                                           gate_override=override)
                            intervention_rows.append(_delta_row(
                                dataset, seed, variant, "EDGE_SHUFFLE_RENORM", sm, metrics, sp, pred,
                                shuffle_replicate=replicate + 1,
                                row_mass_preserved=True, edge_support_preserved=True))
                            del shuffled, sm, sp
                    else:
                        renorm, rm, rp = _eval_model(model, head, data, x, edge, labels,
                                                     preserve_mass_override=True)
                        intervention_rows.append(_delta_row(
                            dataset, seed, variant, "ROW_MASS_NORMALIZE_AT_INFERENCE", rm, metrics, rp, pred,
                            frozen_sensitivity_not_retrained_causal_ablation=True))
                        dosage = {}
                        for modality in ("text", "visual"):
                            means, _ = base._weighted_target_means(normal["raw_gates"][modality], weights,
                                                                   rows, data.num_nodes)
                            dosage[modality] = means[rows]
                        dosage_result, dm, dp = _eval_model(model, head, data, x, edge, labels,
                                                            gate_override=dosage)
                        intervention_rows.append(_delta_row(
                            dataset, seed, variant, "TARGET_MEAN_DOSAGE_ONLY", dm, metrics, dp, pred,
                            frozen_sensitivity_not_retrained_causal_ablation=True))
                        del renorm, rm, rp, dosage, dosage_result, dm, dp

                epoch_proxy = float(complete["mean_epoch_seconds_proxy"])
                ratio = epoch_proxy / mob_epoch_proxy if math.isfinite(mob_epoch_proxy) and mob_epoch_proxy > 0 else math.nan
                peak = complete.get("peak_cuda_memory_mb")
                heavy = (peak is not None and int(peak) >= 20 * 1024) or (math.isfinite(ratio) and ratio >= 2.5)
                complexity_rows.append({
                    "dataset": dataset, "seed": seed, "variant": variant,
                    "trainable_params": complete["trainable_params"],
                    "peak_cuda_memory_mb": peak, "wall_clock_seconds": complete["wall_clock_seconds"],
                    "mean_epoch_seconds_proxy": epoch_proxy,
                    "historical_mob_uniform_epoch_seconds_proxy": mob_epoch_proxy,
                    "historical_mob_uniform_epoch_ratio": ratio,
                    "best_epoch": int(payload["epoch"]),
                    "complexity_status": "COMPUTE_HEAVY" if heavy else ("WITHIN_REFERENCE" if math.isfinite(ratio) else "RATIO_UNAVAILABLE"),
                })
                del model, head, normal, metrics, pred, payload, checkpoint, complete
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        del x, edge, data

    if len(training_commits) != 1:
        raise ValueError(f"formal checkpoints span multiple training commits: {sorted(training_commits)}")
    identity_crosscheck = _historical_identity_crosscheck(table)
    _write_csv(RESULT_ROOT / "historical_mob_crosscheck.csv", identity_crosscheck)
    identity_delta = max((float(row["max_abs_metric_delta"]) for row in identity_crosscheck), default=math.nan)
    if math.isfinite(identity_delta) and identity_delta > 1e-4:
        raise RuntimeError(f"identity/MOB discrepancy {identity_delta:g}; audit before interpreting S4.5")

    contrasts = _all_contrast_rows(_paired_contrasts(metrics_by_run, requested), requested)
    interaction = _interaction_rows(metrics_by_run, requested)
    all_interaction = _all_interaction_rows(interaction)
    interpretation = _statuses(contrasts, interaction, intervention_rows)
    training_commit = next(iter(training_commits))
    analysis_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dataset_config_hashes = {dataset: _file_sha256(ROOT / "configs/dataset" / f"{dataset}.yaml")
                             for dataset in requested}
    summary = {
        "experiment": "S4.5 Relation-Calibrated State-Preserving Propagation",
        "protocol": {"task": "NC", "protocol_version": base.PROTOCOL, "training_mode": "full_graph",
                     "selection": "best_val_accuracy", "base_seed": 42, "run_seeds": list(SEEDS),
                     "test_evaluation": False, "lp_evaluation": False,
                     "datasets": list(requested), "variants": list(VARIANTS),
                     "completed_runs": len(table), "expected_runs": len(requested) * len(SEEDS) * len(VARIANTS)},
        "training_commit": training_commit, "analysis_commit": analysis_commit,
        "identity_mob_crosscheck": {"rows": len(identity_crosscheck),
                                    "max_abs_metric_delta": identity_delta,
                                    "audit_required": False},
        "primary_contrasts": {row["contrast"]: row for row in contrasts if row.get("dataset") == "ALL"},
        "factorial_interaction": all_interaction,
        "interpretation_boundaries": [
            "Per-target row-mass preservation guarantees one-step off-diagonal mass only; it does not imply symmetry or spectral equivalence.",
            "Frozen interventions are checkpoint sensitivities, not retrained causal ablations.",
            "R1 is a fixed experimental carrier, not a proven superior or novel module.",
            "Gate magnitude is not causal relation utility.",
            "Positive calibrated performance does not establish calibration-by-preservation synergy; inspect factorial interaction I.",
            "These are development-set NC results and do not generalize to LP.",
            "Toys was not run and remains an architecture holdout.",
        ],
        "provenance": {
            "source_branch": "s44_relational_transform", "source_sha": base.SOURCE_SHA,
            "training_branch": "s45_relcal_statepres", "training_commit": training_commit,
            "checkpoint_sources_and_sha256": checkpoint_sources,
            "dataset_file_sha256": dataset_hashes,
            "historical_evidence_commits": {
                "multi_order_bank": "31468c5a1439b3561487936bf5d83749b502f61e",
                "mechanism_discovery": "d31b095faa4f98538e60cfac293a831138dde3db",
                "problem_deep_dive": "e535ad91911557790658ddc87bfaa516da531aad",
                "s44_training": json.loads((G0_ROOT / "granularity_summary.json").read_text())["source_training_sha"],
            },
            "config_sha256": {"model": _file_sha256(ROOT / "configs/model/relcal_statepres_pilot.yaml"),
                              "task": _file_sha256(ROOT / "configs/task/nc.yaml"),
                              **{f"dataset/{ds}": dataset_config_hashes[ds] for ds in requested}},
            "code_sha256": {
                "src/models/relcal_statepres_pilot.py": _file_sha256(ROOT / "src/models/relcal_statepres_pilot.py"),
                "scripts/run_s45_relcal_statepres_nc.py": _file_sha256(ROOT / "scripts/run_s45_relcal_statepres_nc.py"),
                "scripts/analyze_s45_relcal_statepres.py": _file_sha256(ROOT / "scripts/analyze_s45_relcal_statepres.py"),
                "scripts/analyze_s45_relcal_statepres_v2.py": _file_sha256(ROOT / "scripts/analyze_s45_relcal_statepres_v2.py"),
                "tests/test_relcal_statepres_pilot.py": _file_sha256(ROOT / "tests/test_relcal_statepres_pilot.py"),
            },
        },
    }
    summary["interpretation_statuses"] = interpretation
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("s45_table.csv", table), ("s45_paired_contrasts.csv", contrasts),
        ("s45_factorial_interaction.csv", interaction), ("gate_statistics.csv", gate_rows),
        ("row_mass_audit.csv", row_rows), ("operator_asymmetry.csv", asym_rows),
        ("calibration_interventions.csv", intervention_rows),
        ("state_change_diagnostics.csv", change_rows), ("state_geometry.csv", geometry_rows),
        ("complexity_table.csv", complexity_rows),
    ):
        _write_csv(RESULT_ROOT / filename, rows)
    (RESULT_ROOT / "s45_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    (RESULT_ROOT / "s45_report.md").write_text(_report(summary), encoding="utf-8")
    return {"completed_runs": len(table), "expected_runs": len(requested) * len(SEEDS) * len(VARIANTS),
            "training_commit": training_commit, "analysis_commit": analysis_commit,
            "interpretation_statuses": interpretation,
            "result_directory": str(RESULT_ROOT)}
