from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import torch
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("Movies", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
VARIANTS = (
    "s45_identity_terminal", "s45_identity_uniform", "s45_identity_propagated_uniform",
    "s45_unconstrained_entry_uniform", "s45_masspres_entry_terminal",
    "s45_masspres_entry_uniform", "s45_masspres_entry_propagated_uniform",
    "s45_masspres_persistent_uniform",
)
G0_VARIANTS = ("s44_scalar_rel", "s44_lowrank_rel", "s44_expert_rel")
PROTOCOL = "unified_full_graph_nc_v1"
SOURCE_SHA = "24c3eec6325a6606d6a4c222ab418b4e616148c1"
OUTPUT_ROOT = ROOT / "outputs/s45_relcal_statepres_v1"
RESULT_ROOT = ROOT / "results/s45_relcal_statepres_v1"
G0_ROOT = RESULT_ROOT / "granularity_audit"
_DIRICHLET_CACHE: dict[tuple[int, int, int, tuple[int, ...]], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _config(dataset: str, model_name: str, variant: str, seed: int = 42):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str((ROOT / "configs").resolve()), version_base=None):
        return compose(config_name="config", overrides=[
            f"dataset={dataset}", "task=nc", f"model={model_name}", f"model.variant={variant}",
            f"seed={seed}", "num_runs=1", "device=cpu", "task.evaluate_test=false",
            "task.training_mode=full_graph", f"task.protocol_version={PROTOCOL}",
        ])


@lru_cache(maxsize=4)
def _load_data(dataset: str):
    from src.data import load_mag_data
    return load_mag_data(_config(dataset, "relcal_statepres_pilot", VARIANTS[0]), "nc", 42)


def _data_info(data) -> dict[str, int]:
    return {"input_dim": data.input_dim, "num_nodes": data.num_nodes,
            "num_classes": data.num_classes, "text_dim": int(data.x_t.size(1)),
            "visual_dim": int(data.x_i.size(1))}


def _checkpoint_meta(path: Path, dataset: str, seed: int, training_branch: str):
    import torch
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("task") != "nc" or payload.get("protocol_version") != PROTOCOL:
        raise ValueError(f"wrong task/protocol in {path}")
    if int(payload.get("seed", -1)) != seed or payload.get("selection") != "best_val_accuracy":
        raise ValueError(f"wrong seed/selection in {path}")
    metrics = payload.get("metrics", {})
    if not {"val_acc", "val_macro_f1"}.issubset(metrics):
        raise ValueError(f"validation metrics missing in {path}")
    if any(key.startswith("test_") for key in metrics):
        raise ValueError(f"test metrics found in {path}")
    return payload


def _load_model(dataset: str, seed: int, variant: str, data, device: torch.device,
                family: str = "s45"):
    import torch
    from src.models import build_model
    from torch import nn
    model_name = "relcal_statepres_pilot" if family == "s45" else "relational_transform_pilot"
    root = OUTPUT_ROOT if family == "s45" else ROOT / "outputs/s44_relational_transform_v1"
    checkpoint = root / "formal" / dataset / variant / f"best_run{SEEDS.index(seed) + 1}.pt"
    payload = _checkpoint_meta(checkpoint, dataset, seed, "s45_relcal_statepres" if family == "s45" else "s44_relational_transform")
    cfg = _config(dataset, model_name, variant, seed)
    model = build_model(cfg, payload["data_info"]).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    head = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    head.load_state_dict(payload["head_state"], strict=True)
    model.eval()
    head.eval()
    return model, head, payload, checkpoint


def _metrics(logits, data, labels: list[int]):
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import f1_score
    val_cpu = data.val_idx.detach().cpu().long()
    val = val_cpu.to(logits.device)
    target = data.y[val_cpu].to(logits.device, dtype=torch.long)
    val_logits = logits[val]
    pred = val_logits.argmax(dim=-1)
    metrics = {
        "val_acc": float((pred == target).float().mean().item()),
        "val_macro_f1": float(f1_score(target.cpu().numpy(), pred.cpu().numpy(), labels=labels,
                                       average="macro", zero_division=0)),
        "true_label_ce": float(F.cross_entropy(val_logits, target).item()),
    }
    return metrics, pred.detach().cpu()


def _eval_model(model, head, data, x, edge, labels, gate_override=None,
                preserve_mass_override=None):
    import torch
    with torch.no_grad():
        result = model.analyze(x, edge, gate_override=gate_override,
                               preserve_mass_override=preserve_mass_override)
        metrics, pred = _metrics(head(result["fused_z"]), data, labels)
    return result, metrics, pred


def _delta_row(dataset, seed, variant, intervention, metrics, normal, pred, normal_pred, **extra):
    return {
        "dataset": dataset, "seed": seed, "variant": variant, "intervention": intervention,
        **metrics,
        "delta_val_acc": metrics["val_acc"] - normal["val_acc"],
        "delta_val_macro_f1": metrics["val_macro_f1"] - normal["val_macro_f1"],
        "delta_true_label_ce": metrics["true_label_ce"] - normal["true_label_ce"],
        "prediction_flip_rate": float((pred != normal_pred).float().mean().item()),
        **extra,
    }


def _weighted_target_means(values, weights, rows, num_nodes: int):
    width = values.size(-1) if values.ndim > 1 else 1
    val = values.reshape(-1, width)
    w = weights.reshape(-1, 1).to(val.dtype)
    mass = weights.new_zeros(num_nodes).index_add_(0, rows, weights)
    sums = val.new_zeros((num_nodes, width)).index_add_(0, rows, val * w)
    means = sums / mass.clamp_min(1e-30).unsqueeze(-1)
    return means, means[rows]


def _weighted_pair_l2(values: torch.Tensor, weights: torch.Tensor, rows: torch.Tensor) -> float:
    values, weights, rows = values.double(), weights.double(), rows.long()
    _, counts = torch.unique_consecutive(rows, return_counts=True)
    target_means, target_masses = [], []
    offset = 0
    for count_value in counts.tolist():
        count = int(count_value)
        end = offset + count
        if count >= 2:
            x, w = values[offset:end], weights[offset:end]
            denom = w.sum().square() - w.square().sum()
            if denom > 0:
                numerator = x.new_zeros(())
                for start in range(0, count, 256):
                    stop = min(start + 256, count)
                    distances = torch.cdist(x[start:stop], x, p=2)
                    numerator += (distances * w[start:stop, None] * w[None, :]).sum()
                target_means.append(numerator / denom)
                target_masses.append(w.sum())
        offset = end
    if not target_means:
        return 0.0
    means, masses = torch.stack(target_means), torch.stack(target_masses)
    return float((means * masses).sum().div(masses.sum()).item())

def _weighted_decomposition(dataset: str, seed: int, variant: str, modality: str,
                            controller: torch.Tensor, rows: torch.Tensor, weights: torch.Tensor):
    pair_l2 = _weighted_pair_l2(controller.detach().double().cpu().reshape(controller.size(0), -1), weights.detach().double().cpu(), rows.detach().long().cpu())
    c, w, row = controller.detach().double().cpu().reshape(controller.size(0), -1), weights.detach().double().cpu(), rows.detach().long().cpu()
    num_nodes = int(row.max().item()) + 1 if row.numel() else 0
    mass = w.new_zeros(num_nodes).index_add_(0, row, w)
    total_weight = w.sum().clamp_min(1e-30)
    mean = (c * w[:, None]).sum(0) / total_weight
    sums = c.new_zeros((num_nodes, c.size(1))).index_add_(0, row, c * w[:, None])
    target_mean = sums / mass.clamp_min(1e-30)[:, None]
    centered = c - target_mean[row]
    within = (centered.square() * w[:, None]).sum(0) / total_weight
    global_centered = c - mean
    total = (global_centered.square() * w[:, None]).sum(0) / total_weight
    between = ((target_mean - mean).square() * mass[:, None]).sum(0) / total_weight
    rows_out = []
    for dim in range(c.size(1)):
        vt, vw, vb = float(total[dim]), float(within[dim]), float(between[dim])
        rows_out.append({
            "dataset": dataset, "seed": seed, "variant": variant, "modality": modality,
            "controller_dim": dim, "mean": float(mean[dim]), "V_total": vt,
            "V_within": vw, "V_between": vb,
            "within_ratio": vw / vt if vt > 0 else math.nan,
            "between_ratio": vb / vt if vt > 0 else math.nan,
            "decomposition_error": vt - vw - vb,
            "weighted_mean_within_target_pairwise_L2": pair_l2,
        })
    return rows_out, target_mean, mass


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _summary_stats(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    values = [float(row[metric]) for row in rows]
    return {"mean": statistics.fmean(values) if values else math.nan,
            "population_std": statistics.pstdev(values) if values else math.nan,
            "positive_seed_pairs": sum(value > 0 for value in values), "n": len(values)}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_granularity_audit(datasets: tuple[str, ...] = DATASETS) -> dict[str, Any]:
    import torch
    from src.data import load_mag_data
    from src.models import build_model
    from src.tasks.nc import _resolve_nc_eval_labels
    from torch import nn

    requested = tuple(dataset for dataset in DATASETS if dataset in datasets)
    if not requested or requested != datasets:
        raise ValueError(f"datasets must be a nonempty canonical subset of {DATASETS}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    variance_rows, intervention_rows, scalar_rows = [], [], []
    training_sha_by_checkpoint = {}
    for dataset in requested:
        data = _load_data(dataset)
        labels = _resolve_nc_eval_labels(data)
        x, edge = data.x.to(device), data.edge_index.to(device)
        for seed in SEEDS:
            for variant in G0_VARIANTS:
                complete_path = ROOT / "outputs/s44_relational_transform_v1/formal" / dataset / variant / "complete.json"
                complete = json.loads(complete_path.read_text(encoding="utf-8"))
                if complete.get("training_branch") != "s44_relational_transform" or complete.get("test_evaluation") is not False:
                    raise ValueError(f"invalid historical S4.4 provenance: {complete_path}")
                training_sha = str(complete.get("training_commit", ""))
                if len(training_sha) != 40:
                    raise ValueError(f"missing S4.4 training SHA in {complete_path}")
                training_sha_by_checkpoint[f"{dataset}/{seed}/{variant}"] = training_sha
                checkpoint = Path(complete["checkpoint_paths"][SEEDS.index(seed)])
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if payload.get("task") != "nc" or payload.get("protocol_version") != PROTOCOL:
                    raise ValueError(f"invalid historical checkpoint metadata: {checkpoint}")
                if int(payload.get("seed", -1)) != seed or payload.get("selection") != "best_val_accuracy":
                    raise ValueError(f"wrong seed/selection in {checkpoint}")
                if any(key.startswith("test_") for key in payload.get("metrics", {})):
                    raise ValueError(f"test metrics in historical checkpoint: {checkpoint}")
                cfg = _config(dataset, "relational_transform_pilot", variant, seed)
                model = build_model(cfg, payload["data_info"]).to(device)
                model.load_state_dict(payload["model_state"], strict=True)
                head = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
                head.load_state_dict(payload["head_state"], strict=True)
                model.eval(); head.eval()
                with torch.no_grad():
                    normal_analysis = model.analyze(x, edge, return_edge_state=True)
                    normal_metrics, normal_pred = _metrics(head(normal_analysis["fused_z"]), data, labels)
                weights = normal_analysis["edge_weight"].detach()
                rows = normal_analysis["edge_row"].detach()
                controllers = normal_analysis["controllers"]
                for modality in ("text", "visual"):
                    ctr = controllers[modality].detach()
                    decomp, target_mean, mass = _weighted_decomposition(
                        dataset, seed, variant, modality, ctr, rows, weights
                    )
                    variance_rows.extend(decomp)
                    both_means = {}
                    for other in ("text", "visual"):
                        c = controllers[other].detach()
                        means, _ = _weighted_target_means(c, weights, rows, data.num_nodes)
                        both_means[other] = means[rows]
                    with torch.no_grad():
                        replaced = model.analyze(x, edge, controller_override=both_means)
                        mean_metrics, mean_pred = _metrics(head(replaced["fused_z"]), data, labels)
                    if modality == "text":
                        intervention_rows.append(_delta_row(
                            dataset, seed, variant, "TARGET_MEAN_REPLACE", mean_metrics,
                            normal_metrics, mean_pred, normal_pred,
                            note="both modalities replaced by P_rel-weighted target means"))

                    if variant == "s44_scalar_rel":
                        scalar = ctr.reshape(-1, 1)
                        d_i, dosage = _weighted_target_means(scalar, weights, rows, data.num_nodes)
                        dosage_only = {mod: (d_i[rows] if mod == modality else controllers[mod].detach())
                                       for mod in ("text", "visual")}
                        redistribution = {
                            mod: (torch.where(d_i[rows] > 1e-12, scalar / d_i[rows].clamp_min(1e-12),
                                              torch.ones_like(scalar)) if mod == modality else controllers[mod].detach())
                            for mod in ("text", "visual")
                        }
                        for name, override in (("DOSAGE_ONLY", dosage_only),
                                               ("REDISTRIBUTION_ONLY", redistribution)):
                            with torch.no_grad():
                                changed = model.analyze(x, edge, controller_override=override)
                                changed_metrics, changed_pred = _metrics(head(changed["fused_z"]), data, labels)
                            intervention_rows.append(_delta_row(
                                dataset, seed, variant, name, changed_metrics, normal_metrics,
                                changed_pred, normal_pred,
                                frozen_sensitivity_not_retrained_causal_ablation=True,
                            ))
                        mass_cpu = mass.detach().cpu().double()
                        g_cpu = scalar.detach().cpu().flatten().double()
                        w_cpu, row_cpu = weights.detach().cpu().double(), rows.detach().cpu()
                        d_cpu = d_i.detach().cpu().flatten().double()
                        dosage_error = (torch.zeros_like(mass_cpu).index_add_(0, row_cpu, w_cpu * g_cpu)
                                        - torch.zeros_like(mass_cpu).index_add_(0, row_cpu, w_cpu * d_cpu[row_cpu])).abs().max()
                        redistributed = g_cpu / d_cpu[row_cpu].clamp_min(1e-12)
                        redistribution_error = (torch.zeros_like(mass_cpu).index_add_(0, row_cpu, w_cpu * redistributed)
                                                 - mass_cpu).abs().max()
                        scalar_rows.append({
                            "dataset": dataset, "seed": seed, "modality": modality,
                            "val_acc": normal_metrics["val_acc"], "val_macro_f1": normal_metrics["val_macro_f1"],
                            "true_label_ce": normal_metrics["true_label_ce"],
                            "mean_target_dosage": float(d_cpu[d_cpu > 0].mean()) if bool((d_cpu > 0).any()) else 0.0,
                            "mean_abs_target_dosage_deviation_from_one": float((d_cpu[d_cpu > 0] - 1).abs().mean()) if bool((d_cpu > 0).any()) else 0.0,
                            "dosage_weighted_mass_error": float(dosage_error),
                            "redistribution_row_mass_error": float(redistribution_error),
                            "interpretation": "frozen sensitivity; not retrained causal ablation",
                        })
                del model, head, normal_analysis
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        del x, edge, data

    scalar_sensitivity = {}
    for intervention_name in ("DOSAGE_ONLY", "REDISTRIBUTION_ONLY"):
        group = [row for row in intervention_rows if row.get("intervention") == intervention_name]
        scalar_sensitivity[intervention_name] = {"rows": len(group), "metrics": {}}
        for metric in ("val_acc", "val_macro_f1", "true_label_ce"):
            deltas = [float(row[f"delta_{metric}"]) for row in group]
            scalar_sensitivity[intervention_name]["metrics"][metric] = {
                "mean_delta": statistics.fmean(deltas),
                "population_std_delta": statistics.pstdev(deltas),
                "positive_rows": sum(value > 0 for value in deltas),
            }
    training_shas = sorted(set(training_sha_by_checkpoint.values()))
    if len(training_shas) != 1:
        raise ValueError(f"S4.4 G0 checkpoints span multiple training SHAs: {training_shas}")
    G0_ROOT.mkdir(parents=True, exist_ok=True)
    _write_csv(G0_ROOT / "controller_variance_decomposition.csv", variance_rows)
    _write_csv(G0_ROOT / "target_mean_interventions.csv", intervention_rows)
    _write_csv(G0_ROOT / "scalar_dosage_redistribution.csv", scalar_rows)
    ratios = [row["within_ratio"] for row in variance_rows if math.isfinite(row["within_ratio"])]
    by_variant = {}
    for variant in G0_VARIANTS:
        group = [row for row in variance_rows if row["variant"] == variant]
        family_ratios = [float(row["within_ratio"]) for row in group if math.isfinite(row["within_ratio"])]
        pair_l2 = [float(row["weighted_mean_within_target_pairwise_L2"]) for row in group]
        by_variant[variant] = {
            "mean_within_ratio": statistics.fmean(family_ratios),
            "population_std_within_ratio": statistics.pstdev(family_ratios),
            "mean_weighted_within_target_pairwise_L2": statistics.fmean(pair_l2),
            "rows": len(group),
        }
    matched = {}
    keys = sorted({(row["dataset"], int(row["seed"]), row["modality"]) for row in variance_rows})
    for other in ("s44_lowrank_rel", "s44_expert_rel"):
        deltas = []
        for key in keys:
            def ratio_for(variant):
                values = [float(row["within_ratio"]) for row in variance_rows
                          if row["variant"] == variant and
                          (row["dataset"], int(row["seed"]), row["modality"]) == key]
                return statistics.fmean(values)
            deltas.append(ratio_for("s44_scalar_rel") - ratio_for(other))
        matched[f"scalar_minus_{other.removeprefix('s44_')}"] = {
            "mean_delta_within_ratio": statistics.fmean(deltas),
            "population_std_delta_within_ratio": statistics.pstdev(deltas),
            "positive_dataset_seed_modality_pairs": sum(value > 0 for value in deltas),
            "n": len(deltas),
        }
    summary = {
        "source_training_branch": "s44_relational_transform", "source_training_sha": training_shas[0],
        "source_commit_for_s45": SOURCE_SHA,
        "scope": {"datasets": list(requested), "seeds": list(SEEDS), "variants": list(G0_VARIANTS),
                  "test_evaluation": False, "retraining": False},
        "checkpoint_metadata_checks": {"count": len(training_sha_by_checkpoint), "training_shas": training_sha_by_checkpoint,
                                       "all_task_protocol_seed_best_val_test_disabled": True},
        "controller_variance": {"mean_within_ratio": statistics.fmean(ratios) if ratios else math.nan,
                                "population_std_within_ratio": statistics.pstdev(ratios) if ratios else math.nan,
                                "by_variant": by_variant, "matched_scalar_comparisons": matched,
                                "rows": len(variance_rows)},
        "scalar_frozen_sensitivity": scalar_sensitivity,
        "interpretation": "Weighted variance decomposition and frozen checkpoint sensitivities. Controller variance or gate size alone does not establish causal relation utility.",
    }
    (G0_ROOT / "granularity_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    report = [
        "# S4.5 G0: Frozen controller granularity audit", "",
        f"S4.4 training SHA: `{training_shas[0]}`. Checkpoints read: {len(training_sha_by_checkpoint)}; all were validation-selected NC checkpoints with test evaluation disabled.", "",
        f"Mean weighted within-target variance fraction across all controller dimensions: {summary['controller_variance']['mean_within_ratio']:.6f} (population SD {summary['controller_variance']['population_std_within_ratio']:.6f}).", "",
        "Controller-family means (population SD): " + "; ".join(
            f"{name.removeprefix('s44_')}: {values['mean_within_ratio']:.4f} ({values['population_std_within_ratio']:.4f})"
            for name, values in summary["controller_variance"]["by_variant"].items()) + ".", "",
        "Matched scalar-minus-family within-ratio differences: " + "; ".join(
            f"{name.removeprefix('scalar_minus_')}: mean {values['mean_delta_within_ratio']:.4f}, positive in {values['positive_dataset_seed_modality_pairs']}/{values['n']} pairs"
            for name, values in summary["controller_variance"]["matched_scalar_comparisons"].items()) + ".", "",
        "The CSV retains modality, controller dimension, total/within/between variance, within ratio, decomposition error, and weighted within-target pairwise L2. Frozen target-mean, dosage-only, and redistribution-only results are checkpoint sensitivity probes, not retrained causal ablations.", "",
        "Scalar frozen sensitivity, mean delta vs NORMAL: " + "; ".join(
            f"{name}: " + ", ".join(f"{metric} {values['mean_delta']:+.6f}" for metric, values in stats["metrics"].items())
            for name, stats in summary["scalar_frozen_sensitivity"].items()) + ". No retraining was performed, so these probes do not separate retrained causal contributions.", "",
        "A high within-target variance fraction indicates edge-specific controller assignment at the checkpoint; it does not establish that the assigned magnitudes are useful. See the intervention CSV for validation accuracy, macro-F1, true-label CE, and prediction flips.", "",
        "Row-mass preservation in the later S4.5 model constrains only one-step off-diagonal target mass. It does not imply symmetry or spectral equivalence.",
    ]
    (G0_ROOT / "granularity_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"source_training_sha": training_shas[0], "checkpoints": len(training_sha_by_checkpoint),
            "variance_rows": len(variance_rows), "target_mean_rows": len(intervention_rows),
            "scalar_rows": len(scalar_rows), "result_directory": str(G0_ROOT)}


def _gate_summary(dataset: str, seed: int, variant: str, modality: str,
                  raw: torch.Tensor, normalized: torch.Tensor, weights: torch.Tensor,
                  rows: torch.Tensor, num_nodes: int):
    raw_cpu = raw.detach().cpu().double().flatten()
    norm_cpu = normalized.detach().cpu().double().flatten()
    weights_cpu, rows_cpu = weights.detach().cpu().double(), rows.detach().cpu().long()
    dosage, dosage_by_edge = _weighted_target_means(raw_cpu, weights_cpu, rows_cpu, num_nodes)
    target_mass = weights_cpu.new_zeros(num_nodes).index_add_(0, rows_cpu, weights_cpu)
    normalized_by_edge = norm_cpu
    dvals, dweights = dosage.flatten(), target_mass
    eligible = dweights > 0
    centered = raw_cpu - dosage_by_edge.flatten()
    within_var = float((weights_cpu * centered.square()).sum() / weights_cpu.sum().clamp_min(1e-30))
    row = {"dataset": dataset, "seed": seed, "variant": variant, "modality": modality,
           "edge_count": raw_cpu.numel(), "target_count_with_edges": int(eligible.sum())}
    for label, values in (("raw", raw_cpu), ("target_dosage", dvals[eligible]), ("normalized", normalized_by_edge)):
        if values.numel() == 0:
            continue
        row[f"{label}_mean"] = float(values.mean())
        row[f"{label}_std"] = float(values.std(unbiased=False))
        for quantile in (0.1, 0.25, 0.5, 0.75, 0.9):
            row[f"{label}_q{int(quantile * 100)}"] = float(torch.quantile(values, quantile))
        row[f"{label}_min"], row[f"{label}_max"] = float(values.min()), float(values.max())
    row["weighted_within_target_variance"] = within_var
    return row


def _row_operator_audit(p, p_self, p_rel, modality: str, operator, dataset, seed, variant):
    import torch
    base_i, base_v = p_rel.indices(), p_rel.values().double()
    n = p_rel.size(0)
    base_mass = base_v.new_zeros(n).index_add_(0, base_i[0], base_v)
    idx, val = operator.indices(), operator.values().double()
    off = idx[0] != idx[1]
    mass = val.new_zeros(n).index_add_(0, idx[0, off], val[off])
    mass_error = (mass - base_mass).abs()
    diag_self = p_self.values().double()
    diag_self_nodes = p_self.indices()[0]
    diag = val.new_zeros(n).index_add_(0, idx[0, ~off], val[~off])
    expected_diag = val.new_zeros(n).index_add_(0, diag_self_nodes, diag_self)
    diag_error = (diag - expected_diag).abs()
    asym = (operator - operator.transpose(0, 1)).coalesce()
    asym_norm = float(torch.linalg.vector_norm(asym.values()).item())
    op_norm = float(torch.linalg.vector_norm(operator.values()).item())
    return {
        "dataset": dataset, "seed": seed, "variant": variant, "modality": modality,
        "max_abs_row_mass_error": float(mass_error.max().item()) if n else 0.0,
        "mean_abs_row_mass_error": float(mass_error.mean().item()) if n else 0.0,
        "max_abs_self_diagonal_error": float(diag_error.max().item()) if n else 0.0,
        "operator_asymmetry_frobenius_ratio": asym_norm / max(op_norm, 1e-30),
        "edge_support_matches_original": bool(torch.equal(operator.indices()[:, off], p_rel.indices())),
        "row_mass_constraint": "PRESERVED" if variant in {
            "s45_masspres_entry_terminal", "s45_masspres_entry_uniform",
            "s45_masspres_entry_propagated_uniform", "s45_masspres_persistent_uniform",
        } else "UNCONSTRAINED",
        "row_mass_invariant_pass": (float(mass_error.max().item()) <= 1e-6) if variant in {
            "s45_masspres_entry_terminal", "s45_masspres_entry_uniform",
            "s45_masspres_entry_propagated_uniform", "s45_masspres_persistent_uniform",
        } else "NOT_APPLICABLE",
        "one_step_row_mass_only": True,
    }


def _shuffle_by_target(raw: torch.Tensor, rows: torch.Tensor, seed: int):
    rows_cpu, raw_cpu = rows.detach().cpu().long(), raw.detach().cpu()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    n_edges = rows_cpu.numel()
    random_rank = torch.randperm(n_edges, generator=generator)
    key = rows_cpu * max(n_edges, 1) + random_rank
    source_order = torch.argsort(key)
    destination_order = torch.argsort(rows_cpu, stable=True)
    shuffled = torch.empty_like(raw_cpu)
    shuffled[destination_order] = raw_cpu[source_order]
    return shuffled.to(raw.device)


def _normalized_dirichlet_energy_cached(state: torch.Tensor, edge_index: torch.Tensor) -> float:
    """Historical normalized A+I Dirichlet energy with graph structure cached."""
    x = state.detach().float().cpu()
    key = (edge_index.data_ptr(), int(getattr(edge_index, "_version", 0)), x.size(0), tuple(edge_index.shape))
    structure = _DIRICHLET_CACHE.get(key)
    if structure is None:
        edge = edge_index.long().cpu()
        edge = torch.cat((edge, edge.flip(0)), dim=1)
        edge = edge[:, edge[0] != edge[1]]
        edge = torch.unique(edge.T, dim=0).T
        loops = torch.arange(x.size(0), dtype=torch.long).repeat(2, 1)
        edge = torch.cat((edge, loops), dim=1)
        row, col = edge
        degree = torch.zeros(x.size(0)).index_add_(0, row, torch.ones(row.numel()))
        inv = degree.clamp_min(1).pow(-0.5)
        structure = (row, col, inv[row] * inv[col])
        _DIRICHLET_CACHE[key] = structure
    row, col, weights = structure
    px = torch.zeros_like(x)
    px.index_add_(0, row, weights[:, None] * x[col])
    energy = (x * (x - px)).sum().item()
    return float(energy / max(float(x.square().sum().item()), 1e-20))


def _geometry_rows(dataset, seed, variant, modality, states, edge_index):
    from src.analysis.mechanism_discovery import geometry_summary
    rows = []
    for row in geometry_summary(states):
        order = int(row["order"])
        row.update({"dataset": dataset, "seed": seed, "variant": variant,
                    "modality": modality,
                    "normalized_dirichlet_energy": _normalized_dirichlet_energy_cached(
                        states[order], edge_index)})
        rows.append(row)
    return rows


def _paired_contrasts(metric_map, datasets):
    definitions = (
        ("identity_uniform_minus_identity_terminal", "s45_identity_uniform", "s45_identity_terminal"),
        ("masspres_entry_uniform_minus_identity_uniform", "s45_masspres_entry_uniform", "s45_identity_uniform"),
        ("unconstrained_minus_masspres_entry_uniform", "s45_unconstrained_entry_uniform", "s45_masspres_entry_uniform"),
        ("masspres_entry_uniform_minus_masspres_entry_terminal", "s45_masspres_entry_uniform", "s45_masspres_entry_terminal"),
        ("masspres_propagated_uniform_minus_identity_propagated_uniform", "s45_masspres_entry_propagated_uniform", "s45_identity_propagated_uniform"),
        ("masspres_persistent_minus_masspres_entry_uniform", "s45_masspres_persistent_uniform", "s45_masspres_entry_uniform"),
    )
    rows = []
    metrics = ("val_acc", "val_macro_f1", "true_label_ce")
    for name, left, right in definitions:
        for dataset in datasets:
            group = []
            for seed in SEEDS:
                a, b = metric_map[(dataset, seed, left)], metric_map[(dataset, seed, right)]
                row = {"contrast": name, "dataset": dataset, "seed": seed,
                       "left_variant": left, "right_variant": right}
                for metric in metrics:
                    row[f"delta_{metric}"] = a[metric] - b[metric]
                rows.append(row); group.append(row)
            summary = {"contrast": name, "dataset": dataset, "seed": "ALL",
                       "left_variant": left, "right_variant": right}
            for metric in metrics:
                values = [float(row[f"delta_{metric}"]) for row in group]
                summary[f"mean_delta_{metric}"] = statistics.fmean(values)
                summary[f"population_std_delta_{metric}"] = statistics.pstdev(values)
                summary[f"positive_seed_pairs_{metric}"] = sum(v > 0 for v in values)
            rows.append(summary)
    return rows


def _interaction_rows(metric_map, datasets):
    rows = []
    for dataset in datasets:
        group = []
        for seed in SEEDS:
            vals = metric_map
            identity = vals[(dataset, seed, "s45_identity_uniform")]
            identity_term = vals[(dataset, seed, "s45_identity_terminal")]
            mass = vals[(dataset, seed, "s45_masspres_entry_uniform")]
            mass_term = vals[(dataset, seed, "s45_masspres_entry_terminal")]
            row = {"dataset": dataset, "seed": seed}
            for metric in ("val_acc", "val_macro_f1", "true_label_ce"):
                row[f"interaction_{metric}"] = (mass[metric] - mass_term[metric]) - (identity[metric] - identity_term[metric])
            rows.append(row); group.append(row)
        summary = {"dataset": dataset, "seed": "ALL"}
        for metric in ("val_acc", "val_macro_f1", "true_label_ce"):
            values = [float(row[f"interaction_{metric}"]) for row in group]
            summary[f"mean_interaction_{metric}"] = statistics.fmean(values)
            summary[f"population_std_interaction_{metric}"] = statistics.pstdev(values)
            summary[f"positive_seed_pairs_{metric}"] = sum(value > 0 for value in values)
        rows.append(summary)
    return rows


def _statuses(contrasts, interaction, interventions):
    def contrast(name, metric="val_acc"):
        return next(row for row in contrasts if row.get("contrast") == name and row.get("dataset") == "ALL")
    def label(row, mean_key="mean_delta_val_acc", positive_key="positive_seed_pairs_val_acc"):
        mean = float(row[mean_key]); positive = int(row[positive_key])
        total = 3 * 4
        if mean > 0 and positive > total / 2:
            return "STRONG_SUPPORT"
        if mean > 0 or positive > total / 2:
            return "MECHANISM_SUPPORT"
        if mean < 0 and positive == 0:
            return "UNSUPPORTED"
        return "MIXED"
    state = label(contrast("identity_uniform_minus_identity_terminal"))
    redistribution = label(contrast("masspres_entry_uniform_minus_identity_uniform"))
    interaction_mean = statistics.fmean(float(r["interaction_val_acc"]) for r in interaction if r.get("seed") != "ALL")
    interaction_pos = sum(float(r["interaction_val_acc"]) > 0 for r in interaction if r.get("seed") != "ALL")
    interaction_status = ("STRONG_SUPPORT" if interaction_mean > 0 and interaction_pos > 6 else
                          "MECHANISM_SUPPORT" if interaction_mean > 0 or interaction_pos > 6 else
                          "UNSUPPORTED" if interaction_mean < 0 and interaction_pos == 0 else "MIXED")
    placement = label(contrast("masspres_persistent_minus_masspres_entry_uniform"))
    shuffle_rows = [r for r in interventions if r.get("intervention") == "EDGE_SHUFFLE_RENORM"]
    shuffle_positive = sum(float(r["delta_val_acc"]) < 0 for r in shuffle_rows)
    gran = json.loads((G0_ROOT / "granularity_summary.json").read_text(encoding="utf-8"))
    variance = gran["controller_variance"]
    ratio = float(variance["mean_within_ratio"])
    by_variant = variance.get("by_variant", {})
    matched = variance.get("matched_scalar_comparisons", {})
    scalar_ratio = float(by_variant.get("s44_scalar_rel", {}).get("mean_within_ratio", math.nan))
    lowrank_ratio = float(by_variant.get("s44_lowrank_rel", {}).get("mean_within_ratio", math.nan))
    expert_ratio = float(by_variant.get("s44_expert_rel", {}).get("mean_within_ratio", math.nan))
    scalar_lowrank = matched.get("scalar_minus_lowrank_rel", {})
    scalar_expert = matched.get("scalar_minus_expert_rel", {})
    beats_lowrank = int(scalar_lowrank.get("positive_dataset_seed_modality_pairs", 0))
    beats_expert = int(scalar_expert.get("positive_dataset_seed_modality_pairs", 0))
    if scalar_ratio > lowrank_ratio and scalar_ratio > expert_ratio and beats_lowrank > 12 and beats_expert > 12:
        gran_status = "STRONG_SUPPORT"
    elif scalar_ratio > lowrank_ratio or scalar_ratio > expert_ratio:
        gran_status = "MECHANISM_SUPPORT"
    elif scalar_ratio < lowrank_ratio and scalar_ratio < expert_ratio:
        gran_status = "UNSUPPORTED"
    else:
        gran_status = "MIXED"
    return {
        "ControllerGranularity": {"status": gran_status, "mean_within_ratio": ratio,
                                   "mean_within_ratio_by_variant": by_variant,
                                   "matched_scalar_comparisons": matched,
                                   "scalar_frozen_sensitivity": gran.get("scalar_frozen_sensitivity", {}),
                                   "evidence_boundary": "variance indicates edge-specific assignment, not task utility"},
        "RelationRedistribution": {"status": redistribution, "masspres_uniform_vs_identity_uniform": contrast("masspres_entry_uniform_minus_identity_uniform"),
                                   "edge_shuffle_harm_rows": shuffle_positive, "edge_shuffle_rows": len(shuffle_rows)},
        "StatePreservation": {"status": state, "identity_uniform_minus_terminal": contrast("identity_uniform_minus_identity_terminal"),
                              "calibrated_uniform_minus_terminal": contrast("masspres_entry_uniform_minus_masspres_entry_terminal")},
        "CalibrationStateInteraction": {"status": interaction_status, "mean_factorial_interaction_val_acc": interaction_mean,
                                        "positive_seed_pairs": interaction_pos, "pairs": 12},
        "CalibrationPlacement": {"status": placement, "persistent_minus_entry_uniform": contrast("masspres_persistent_minus_masspres_entry_uniform")},
    }


def _report(summary):
    contrasts = summary["primary_contrasts"]
    interaction = summary["factorial_interaction"]
    status = summary["interpretation_statuses"]
    lines = [
        "# S4.5 Relation-Calibrated State-Preserving Propagation", "",
        f"Training commit: `{summary['provenance']['training_commit']}`; source branch/SHA: `{summary['provenance']['source_branch']}` / `{summary['provenance']['source_sha']}`.",
        "Scope: unified full-graph node classification on Movies, Grocery, ele-fashion and Reddit-S; seeds 42–44; best validation accuracy; test disabled. Toys remains an architecture holdout. No LP conclusions are made.", "",
        "## Findings", "",
        f"- ControllerGranularity: **{status['ControllerGranularity']['status']}**. Mean weighted within-target variance fraction {status['ControllerGranularity']['mean_within_ratio']:.4f}; this describes controller variation and does not establish task utility.",
        f"- RelationRedistribution: **{status['RelationRedistribution']['status']}**. The primary validation contrast is mass-preserving entry uniform minus identity uniform: {contrasts['masspres_entry_uniform_minus_identity_uniform']}.",
        f"- StatePreservation: **{status['StatePreservation']['status']}**. Identity uniform minus terminal: {contrasts['identity_uniform_minus_identity_terminal']}; calibrated uniform minus terminal: {contrasts['masspres_entry_uniform_minus_masspres_entry_terminal']}.",
        f"- CalibrationStateInteraction: **{status['CalibrationStateInteraction']['status']}**. Mean factorial interaction in validation accuracy is {interaction['mean_interaction_val_acc']:.6f} (population SD {interaction['population_std_interaction_val_acc']:.6f}); positive seed pairs {interaction['positive_seed_pairs_val_acc']}/12.",
        f"- CalibrationPlacement: **{status['CalibrationPlacement']['status']}**. Persistent minus entry-only uniform: {contrasts['masspres_persistent_minus_masspres_entry_uniform']}.", "",
        "## Required interpretation boundaries", "",
        "Per-target row-mass preservation fixes only one-step off-diagonal mass. It does not preserve symmetry, spectrum, stationary distribution, or smoothing behavior. Frozen interventions are sensitivity analyses, not retrained causal ablations. R1 is a fixed experimental carrier, not a proven innovation. Gate magnitude is not causal relation utility. Positive calibrated uniform performance does not imply synergy; the factorial interaction is reported separately. These NC development results do not generalize to LP. Toys was not run.", "",
        "## Backbone decision", "",
        "This pilot alone does not freeze a paper backbone. Freeze relation → propagation → state composition only after reviewing the raw paired results, mechanism audits, identity equivalence, and the factorial interaction; no single threshold determines that decision.", "",
        "Detailed raw tables: `s45_table.csv`, `s45_paired_contrasts.csv`, `s45_factorial_interaction.csv`, `gate_statistics.csv`, `row_mass_audit.csv`, `operator_asymmetry.csv`, `calibration_interventions.csv`, `state_change_diagnostics.csv`, `state_geometry.csv`, and `complexity_table.csv`.",
    ]
    return "\n".join(lines) + "\n"
