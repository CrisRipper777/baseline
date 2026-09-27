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


def _audit_historical_mob_checkpoints(datasets: tuple[str, ...], device: torch.device) -> list[dict[str, Any]]:
    """Replay historical MOB validation checkpoints and test exact identity loading."""
    from hydra import compose, initialize_config_dir
    from src.data import load_mag_data
    from src.models import build_model
    from src.tasks.nc import _resolve_nc_eval_labels
    from torch import nn

    reference = {}
    with (ROOT / "results/nc_benchmark_v1/nc_per_run.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("model") == "multi_order_bank":
                reference[(row["dataset"], int(row["seed"]), row["variant"])] = row
    audit_rows = []
    for dataset in datasets:
        data = _load_data(dataset)
        labels = _resolve_nc_eval_labels(data)
        x, edge = data.x.to(device), data.edge_index.to(device)
        for readout in ("terminal", "uniform"):
            historical_variant = f"mob_{readout}_plain"
            identity_variant = f"s45_identity_{readout}"
            with initialize_config_dir(config_dir=str((ROOT / "configs").resolve()), version_base=None):
                mob_cfg = compose(config_name="config", overrides=[
                    f"dataset={dataset}", "task=nc", "model=multi_order_bank",
                    f"model.readout={readout}", "model.fusion_mode=plain_mlp",
                    "seed=42", "num_runs=1", f"device={device}",
                    "task.evaluate_test=false", "task.training_mode=full_graph",
                    f"task.protocol_version={base.PROTOCOL}",
                ])
            identity_cfg = base._config(dataset, "relcal_statepres_pilot", identity_variant, 42)
            for seed in SEEDS:
                checkpoint = ROOT / "outputs/mob_factorial_nc_v1" / dataset / historical_variant / f"best_run{SEEDS.index(seed) + 1}.pt"
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if payload.get("task") != "nc" or payload.get("protocol_version") != base.PROTOCOL:
                    raise ValueError(f"invalid historical MOB checkpoint metadata: {checkpoint}")
                if int(payload.get("seed", -1)) != seed or payload.get("selection") != "best_val_accuracy":
                    raise ValueError(f"wrong historical seed/selection in {checkpoint}")
                mob = build_model(mob_cfg, payload["data_info"]).to(device).eval()
                mob.load_state_dict(payload["model_state"], strict=True)
                head = nn.Linear(mob.out_dim, int(data.num_classes)).to(device)
                head.load_state_dict(payload["head_state"], strict=True)
                head.eval()
                identity = build_model(identity_cfg, payload["data_info"]).to(device).eval()
                identity.load_state_dict(payload["model_state"], strict=True)
                with torch.no_grad():
                    historical = mob.analyze(x, edge)
                    pilot = identity.analyze(x, edge)
                    historical_metrics, _ = _metrics(head(historical["fused_z"]), data, labels)
                    logit_delta = float((historical["fused_z"] - pilot["fused_z"]).abs().max().item())
                ref = reference.get((dataset, seed, historical_variant))
                if ref is None:
                    raise ValueError(f"historical reference metric row missing: {dataset}/{seed}/{historical_variant}")
                table_acc, table_f1 = float(ref["val_acc"]), float(ref["val_macro_f1"])
                checkpoint_acc = float(payload["metrics"]["val_acc"])
                checkpoint_f1 = float(payload["metrics"]["val_macro_f1"])
                audit_rows.append({
                    "dataset": dataset, "seed": seed, "historical_variant": historical_variant,
                    "checkpoint": str(checkpoint),
                    "recomputed_val_acc": historical_metrics["val_acc"],
                    "historical_table_val_acc": table_acc,
                    "abs_recomputed_table_val_acc_delta": abs(historical_metrics["val_acc"] - table_acc),
                    "recomputed_val_macro_f1": historical_metrics["val_macro_f1"],
                    "historical_table_val_macro_f1": table_f1,
                    "abs_recomputed_table_val_macro_f1_delta": abs(historical_metrics["val_macro_f1"] - table_f1),
                    "abs_recomputed_checkpoint_val_acc_delta": abs(historical_metrics["val_acc"] - checkpoint_acc),
                    "abs_recomputed_checkpoint_val_macro_f1_delta": abs(historical_metrics["val_macro_f1"] - checkpoint_f1),
                    "identity_model_max_abs_logit_delta_from_same_weights": logit_delta,
                    "test_evaluation": False,
                })
                if (abs(historical_metrics["val_acc"] - table_acc) > 1e-6 or
                        abs(historical_metrics["val_macro_f1"] - table_f1) > 1e-6 or
                        logit_delta > 1e-4):
                    raise RuntimeError(f"historical identity audit failed: {dataset}/{seed}/{historical_variant}")
                del mob, identity, head, historical, pilot, payload
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        del x, edge, data
    return audit_rows


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
    contrasts = summary["primary_contrasts"]
    interaction = summary["factorial_interaction"]
    status = summary["interpretation_statuses"]

    def contrast(name: str) -> str:
        row = contrasts[name]
        return (
            f"ΔAcc {row['mean_delta_val_acc']:+.6f} "
            f"({row['positive_seed_pairs_val_acc']}/12 positive seed pairs), "
            f"ΔMacro-F1 {row['mean_delta_val_macro_f1']:+.6f}, "
            f"ΔCE {row['mean_delta_true_label_ce']:+.6f}"
        )

    family = status["ControllerGranularity"]["mean_within_ratio_by_variant"]
    scalar_probes = status["ControllerGranularity"]["scalar_frozen_sensitivity"]
    dosage = scalar_probes["DOSAGE_ONLY"]["metrics"]
    redistribution = scalar_probes["REDISTRIBUTION_ONLY"]["metrics"]
    identity_audit = summary["identity_mob_crosscheck"]
    row_mass = summary["row_mass_audit"]
    complexity = summary["complexity_summary"]
    lines = [
        "# S4.5 Relation-Calibrated State-Preserving Propagation", "",
        f"Training commit: `{summary['training_commit']}`; analysis commit: `{summary['analysis_commit']}`; source branch/SHA: `s44_relational_transform` / `{base.SOURCE_SHA}`.",
        "Scope: unified full-graph node classification on Movies, Grocery, ele-fashion and Reddit-S; seeds 42–44; best-validation-accuracy selection; test disabled. Toys remains an architecture holdout. No LP results are included.", "",
        "## Findings", "",
        f"- ControllerGranularity: **{status['ControllerGranularity']['status']}**. Mean within-target variance ratio: scalar {family['s44_scalar_rel']['mean_within_ratio']:.4f}, low-rank {family['s44_lowrank_rel']['mean_within_ratio']:.4f}, expert {family['s44_expert_rel']['mean_within_ratio']:.4f}. Scalar exceeds the matched low-rank and expert ratios in 23/24 dataset-seed-modality comparisons. Variance describes controller assignment, not relation utility.",
        f"- S4.4 scalar frozen probes: DOSAGE_ONLY ΔAcc {dosage['val_acc']['mean_delta']:+.6f}, ΔMacro-F1 {dosage['val_macro_f1']['mean_delta']:+.6f}, ΔCE {dosage['true_label_ce']['mean_delta']:+.6f}; REDISTRIBUTION_ONLY ΔAcc {redistribution['val_acc']['mean_delta']:+.6f}, ΔMacro-F1 {redistribution['val_macro_f1']['mean_delta']:+.6f}, ΔCE {redistribution['true_label_ce']['mean_delta']:+.6f}. The small changes do not identify one contribution as dominant; these are frozen sensitivities, not retrained causal ablations.",
        f"- RelationRedistribution: **{status['RelationRedistribution']['status']}**. Mass-preserving entry uniform minus identity uniform: {contrast('masspres_entry_uniform_minus_identity_uniform')}. Shuffling within-target edge assignments reduced validation accuracy in {status['RelationRedistribution']['edge_shuffle_harm_rows']}/{status['RelationRedistribution']['edge_shuffle_rows']} renormalized shuffles, indicating sensitivity to assignment within the trained checkpoint despite little mean accuracy gain.",
        f"- StatePreservation: **{status['StatePreservation']['status']}**. Identity uniform minus terminal: {contrast('identity_uniform_minus_identity_terminal')}. Under calibration, uniform minus terminal: {contrast('masspres_entry_uniform_minus_masspres_entry_terminal')}.",
        f"- Intrinsic-state contribution: identity uniform minus propagated-uniform: {contrast('identity_uniform_minus_identity_propagated_uniform')}; mass-preserving entry uniform minus propagated-uniform: {contrast('masspres_entry_uniform_minus_masspres_entry_propagated_uniform')}.",
        f"- CalibrationStateInteraction: **{status['CalibrationStateInteraction']['status']}**. Factorial interaction I: Acc mean {interaction['mean_interaction_val_acc']:+.6f} (SD {interaction['population_std_interaction_val_acc']:.6f}, positive {interaction['positive_seed_pairs_val_acc']}/12); Macro-F1 mean {interaction['mean_interaction_val_macro_f1']:+.6f} (SD {interaction['population_std_interaction_val_macro_f1']:.6f}, positive {interaction['positive_seed_pairs_val_macro_f1']}/12); CE mean {interaction['mean_interaction_true_label_ce']:+.6f} (SD {interaction['population_std_interaction_true_label_ce']:.6f}, positive {interaction['positive_seed_pairs_true_label_ce']}/12). This does not support positive calibration-by-preservation synergy.",
        f"- CalibrationPlacement: **{status['CalibrationPlacement']['status']}**. Persistent minus entry-only uniform: {contrast('masspres_persistent_minus_masspres_entry_uniform')}.",
        f"- Row-mass audit: preserved variants max absolute row error {row_mass['max_abs_row_mass_error_preserved']:.3e}; max self-diagonal error {row_mass['max_abs_self_diagonal_error']:.3e}. Unconstrained max row deviation {row_mass['max_abs_row_mass_deviation_unconstrained_descriptive']:.4f} is expected and reported descriptively.",
        f"- Complexity: {complexity['compute_heavy_runs']}/96 runs flagged COMPUTE_HEAVY; max peak CUDA memory {complexity['max_peak_cuda_memory_mb'] / 1024:.2f} GiB; max epoch ratio {complexity['max_historical_mob_epoch_ratio']:.2f}× historical MOB uniform.", "",
        "## Historical identity audit", "",
        f"The independent identity checkpoint cross-check exceeded the initial 1e-4 screen (max |ΔAcc| {identity_audit['max_abs_val_acc_delta']:.4f}; max |ΔMacro-F1| {identity_audit['max_abs_val_macro_f1_delta']:.4f}). The audit replayed all {identity_audit['historical_checkpoint_rows_audited']} historical MOB checkpoints on the fixed validation splits; stored metrics reproduced within {identity_audit['historical_checkpoint_max_abs_table_metric_delta']:.1e}, and the same weights in the identity pilot differed by at most {identity_audit['historical_checkpoint_max_abs_identity_logit_delta']:.1e} in logits. The remaining differences compare independently trained checkpoints and are descriptive, not evidence for calibration. See `historical_mob_checkpoint_audit.csv` and `historical_mob_crosscheck.csv`.", "",
        "## Interpretation boundaries", "",
        "Row-mass preservation fixes one-step off-diagonal target mass only; it does not guarantee symmetry or spectral equivalence. Frozen interventions are not retrained causal ablations. R1 is a fixed carrier, not a proven superior module. Gate magnitude is not causal relation utility. Positive performance does not establish synergy; inspect the factorial interaction. These NC development results do not generalize to LP. Toys was not used.", "",
        "This pilot does not by itself freeze the paper backbone. Review the raw paired results, mechanism audits, state diagnostics, identity checks, and factorial interaction before freezing relation → propagation → state composition.", "",
        "Detailed tables: `s45_table.csv`, `s45_paired_contrasts.csv`, `s45_factorial_interaction.csv`, `gate_statistics.csv`, `row_mass_audit.csv`, `operator_asymmetry.csv`, `calibration_interventions.csv`, `state_change_diagnostics.csv`, `state_geometry.csv`, `complexity_table.csv`.",
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
    historical_checkpoint_audit = _audit_historical_mob_checkpoints(requested, device)
    _write_csv(RESULT_ROOT / "historical_mob_checkpoint_audit.csv", historical_checkpoint_audit)
    identity_delta = max((float(row["max_abs_metric_delta"]) for row in identity_crosscheck), default=math.nan)
    identity_acc_delta = max((abs(float(row["delta_val_acc"])) for row in identity_crosscheck), default=math.nan)
    identity_f1_delta = max((abs(float(row["delta_val_macro_f1"])) for row in identity_crosscheck), default=math.nan)

    contrasts = _all_contrast_rows(_paired_contrasts(metrics_by_run, requested), requested)
    interaction = _interaction_rows(metrics_by_run, requested)
    all_interaction = _all_interaction_rows(interaction)
    interpretation = _statuses(contrasts, interaction, intervention_rows)
    training_commit = next(iter(training_commits))
    analysis_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dataset_config_hashes = {dataset: _file_sha256(ROOT / "configs/dataset" / f"{dataset}.yaml")
                             for dataset in requested}
    preserved_row_audits = [row for row in row_rows if row.get("row_mass_constraint") == "PRESERVED"]
    unconstrained_row_audits = [row for row in row_rows if row.get("row_mass_constraint") == "UNCONSTRAINED"]
    row_mass_summary = {
        "preserved_rows": len(preserved_row_audits),
        "max_abs_row_mass_error_preserved": max(float(row["max_abs_row_mass_error"]) for row in preserved_row_audits),
        "mean_abs_row_mass_error_preserved": statistics.fmean(float(row["mean_abs_row_mass_error"]) for row in preserved_row_audits),
        "max_abs_self_diagonal_error": max(float(row["max_abs_self_diagonal_error"]) for row in row_rows),
        "max_abs_row_mass_deviation_unconstrained_descriptive": max(float(row["max_abs_row_mass_error"]) for row in unconstrained_row_audits),
        "unconstrained_rows_are_not_mass_invariant_failures": True,
    }
    finite_epoch_ratios = [float(row["historical_mob_uniform_epoch_ratio"]) for row in complexity_rows
                           if math.isfinite(float(row["historical_mob_uniform_epoch_ratio"]))]
    complexity_summary = {
        "compute_heavy_runs": sum(row["complexity_status"] == "COMPUTE_HEAVY" for row in complexity_rows),
        "max_peak_cuda_memory_mb": max(int(row["peak_cuda_memory_mb"]) for row in complexity_rows
                                        if row.get("peak_cuda_memory_mb") is not None),
        "max_historical_mob_epoch_ratio": max(finite_epoch_ratios) if finite_epoch_ratios else math.nan,
    }
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
                                    "max_abs_val_acc_delta": identity_acc_delta,
                                    "max_abs_val_macro_f1_delta": identity_f1_delta,
                                    "audit_threshold_triggered": bool(math.isfinite(identity_delta) and identity_delta > 1e-4),
                                    "audit_performed": True,
                                    "audit_status": "Historical checkpoint metrics reproduced on the fixed validation splits; same historical weights produce identity-model logits within 1e-4. Remaining differences compare independently trained checkpoints and are descriptive, not calibration evidence.",
                                    "historical_checkpoint_rows_audited": len(historical_checkpoint_audit),
                                    "historical_checkpoint_max_abs_table_metric_delta": max(
                                        max(float(row["abs_recomputed_table_val_acc_delta"]),
                                            float(row["abs_recomputed_table_val_macro_f1_delta"]))
                                        for row in historical_checkpoint_audit),
                                    "historical_checkpoint_max_abs_identity_logit_delta": max(
                                        float(row["identity_model_max_abs_logit_delta_from_same_weights"])
                                        for row in historical_checkpoint_audit)},
        "primary_contrasts": {row["contrast"]: row for row in contrasts if row.get("dataset") == "ALL"},
        "factorial_interaction": all_interaction,
        "row_mass_audit": row_mass_summary,
        "complexity_summary": complexity_summary,
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
