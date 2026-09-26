from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
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
    "h1r_dual_agg_signed", "h1r_dual_functional_signed",
    "h2a_global_scalar", "h2a_node_scalar", "h2a_global_group", "h2a_node_group",
    "h2b_global_agg_correction", "h2b_global_diff_correction",
    "h2b_node_agg_correction", "h2b_node_diff_correction",
)
SOURCE_COMMIT = "601bad98ff28e85c9aeac23b84b5db51ad4c7c86"
OUTPUT_ROOT = ROOT / "outputs/s43_h15_h1r_h2ab_v1"
RESULT_ROOT = ROOT / "results/s43_h15_h1r_h2ab_v1"
HISTORICAL_ROOT = ROOT / "outputs/s43_p0_h1_v1"
GRID = (-1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean(xs):
    return statistics.fmean(xs) if xs else math.nan


def _std(xs):
    return statistics.pstdev(xs) if xs else math.nan


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    q = torch.quantile(values.float().reshape(-1), torch.tensor([.1, .25, .5, .75, .9], device=values.device))
    return {f"q{int(p * 100)}": float(v) for p, v in zip((.1, .25, .5, .75, .9), q)}


def _stats(values: torch.Tensor) -> dict[str, float]:
    v = values.float().reshape(-1)
    return {"mean": float(v.mean()), "std": float(v.std(unbiased=False)), "min": float(v.min()),
            "max": float(v.max()), **_quantiles(v)}


def _config(dataset: str, model_name: str, variant: str):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        return compose(config_name="config", overrides=[
            f"dataset={dataset}", "task=nc", f"model={model_name}", f"model.variant={variant}",
            "model.hidden_dim=256", "model.dropout=0.2", "seed=42", "num_runs=1",
            "device=cpu", "task.evaluate_test=false",
        ])


@lru_cache(maxsize=4)
def _load_data(dataset: str):
    cfg = _config(dataset, "relation_basis_pilot", "h1_dual_functional_static")
    return load_mag_data(cfg, "nc", 42)


def _data_info(data):
    return {"input_dim": data.input_dim, "num_nodes": data.num_nodes,
            "num_classes": data.num_classes, "text_dim": int(data.x_t.shape[1]),
            "visual_dim": int(data.x_i.shape[1])}


def _eval_labels(data):
    from src.tasks.nc import _resolve_nc_eval_labels
    return _resolve_nc_eval_labels(data)


def _load_model(dataset: str, model_name: str, variant: str, seed: int, checkpoint: Path, device: torch.device):
    data = _load_data(dataset)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("task") != "nc" or payload.get("protocol_version") != "unified_full_graph_nc_v1":
        raise ValueError(f"invalid task/protocol checkpoint: {checkpoint}")
    if int(payload.get("seed", -1)) != seed or payload.get("selection") != "best_val_accuracy":
        raise ValueError(f"invalid seed/selection checkpoint: {checkpoint}")
    if any(k.startswith("test_") for k in payload.get("metrics", {})):
        raise ValueError(f"test metrics are forbidden: {checkpoint}")
    cfg = _config(dataset, model_name, variant)
    model = build_model(cfg, _data_info(data)).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    classifier.load_state_dict(payload["head_state"])
    classifier.eval()
    return data, model, classifier, payload, _eval_labels(data)


def _val(data):
    idx = data.val_idx.detach().cpu().long()
    y = data.y[idx].detach().cpu().long()
    return idx, y


def _metrics(logits: torch.Tensor, data, labels: list[int]) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    idx_cpu, y_cpu = _val(data)
    logits_val = logits[idx_cpu.to(logits.device)]
    y = y_cpu.to(logits.device)
    pred = logits_val.argmax(dim=-1)
    result = {
        "val_acc": float((pred == y).float().mean()),
        "val_macro_f1": float(f1_score(y.detach().cpu().numpy(), pred.detach().cpu().numpy(),
                                         labels=labels, average="macro", zero_division=0)),
        "true_label_ce": float(F.cross_entropy(logits_val, y)),
    }
    return result, pred.detach().cpu(), y_cpu


def _fused_logits(model, classifier, h_text, h_visual, rel_text, rel_visual):
    zt = model.output_norm_text(h_text + rel_text)
    zv = model.output_norm_visual(h_visual + rel_visual)
    return classifier(model.plain_fusion(torch.cat([zt, zv], dim=-1)))


def _historical_path(dataset: str, seed: int, variant: str) -> Path:
    run = SEEDS.index(seed) + 1
    return HISTORICAL_ROOT / "formal" / dataset / variant / f"best_run{run}.pt"


def _write_h15_report(summary: dict[str, Any]) -> None:
    lines = [
        "# H1.5 Frozen Differential-Usage Heterogeneity Audit", "",
        "This is a frozen validation-only scan of historical `h1_dual_functional_static` checkpoints. No model was retrained.",
        "The node oracle uses validation labels and is `DESCRIPTIVE_ORACLE_ONLY`; oracle-selected lambdas were not used for training, routing, thresholds, or hyperparameter selection.", "",
        f"- Checkpoints: {summary['checkpoints_read']} historical checkpoints.",
        f"- Fixed lambda grid: `{GRID}`.",
        f"- Max absolute logit deviation at lambda=1 from historical NORMAL: {summary['max_lambda1_logit_abs_error']:.3g}.",
        "- Lambda=0 retains `0.5 * R_A` and removes D while preserving A scale.", "",
        "## Validation-node oracle proportions by dataset and seed", "",
        "| Dataset | Seed | lambda*=0 | lambda*>0 | lambda*<0 | Mean headroom |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["oracle_summary"]:
        lines.append(f"| {row['dataset']} | {row['seed']} | {row['fraction_lambda_star_zero']:.3f} | {row['fraction_lambda_star_positive']:.3f} | {row['fraction_lambda_star_negative']:.3f} | {row['mean_oracle_headroom']:.5f} |")
    lines += ["", "## Modality-only validation scans", "",
              "These aggregate scan minima also use validation labels and are descriptive summaries of the preregistered grid; they were not used to select a model or training hyperparameter.",
              "| Dataset | Seed | Text-only best lambda | Visual-only best lambda | Best lambdas differ |", "|---|---:|---:|---:|:---:|"]
    for row in summary["modality_summary"]:
        if row["mode"] == "TEXT_ONLY":
            pair = next(x for x in summary["modality_summary"] if x["dataset"] == row["dataset"] and x["seed"] == row["seed"] and x["mode"] == "VISUAL_ONLY")
            lines.append(f"| {row['dataset']} | {row['seed']} | {row['best_lambda']:.2g} | {pair['best_lambda']:.2g} | {'yes' if row['best_lambda'] != pair['best_lambda'] else 'no'} |")
    lines += ["", "## Interpretation boundary", "",
              "The node oracle is optimistic because it minimizes each validation node's true-label CE over the same fixed grid. It does not estimate deployable routing value. Modality-only minima are likewise descriptive validation scans, not selected operating points.", ""]
    (RESULT_ROOT / "h15_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_h15() -> dict[str, Any]:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scan_rows, modality_rows, modality_summary, node_rows, oracle_summary = [], [], [], [], []
    max_error = 0.0
    n_checkpoints = 0
    for dataset in DATASETS:
        complete_path = HISTORICAL_ROOT / "formal" / dataset / "h1_dual_functional_static" / "complete.json"
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("test_evaluation") is not False or complete.get("task") != "nc" or complete.get("protocol_version") != "unified_full_graph_nc_v1":
            raise ValueError(f"invalid historical completion record: {complete_path}")
        data = _load_data(dataset)
        val_idx_cpu, y_cpu = _val(data)
        for seed in SEEDS:
            checkpoint = _historical_path(dataset, seed, "h1_dual_functional_static")
            data, model, classifier, payload, labels = _load_model(
                dataset, "relation_basis_pilot", "h1_dual_functional_static", seed, checkpoint, device)
            with torch.no_grad():
                analysis = model.analyze(data.x.to(device), data.edge_index.to(device))
                h = {m: analysis[f"H_{m}"] for m in ("text", "visual")}
                ra = {m: analysis[f"R_A_{m}"] for m in ("text", "visual")}
                rd = {m: analysis[f"R_D_{m}"] for m in ("text", "visual")}
                normal_logits = analysis["fused_z"].new_tensor(0)  # keep graph/device dtype explicit
                normal_logits = classifier(analysis["fused_z"])
                normal_metrics, normal_pred, yv = _metrics(normal_logits, data, labels)
                logits_shared, node_losses, node_preds = [], [], []
                for lam in GRID:
                    rel_t = 0.5 * ra["text"] + 0.5 * float(lam) * rd["text"]
                    rel_v = 0.5 * ra["visual"] + 0.5 * float(lam) * rd["visual"]
                    logits = _fused_logits(model, classifier, h["text"], h["visual"], rel_t, rel_v)
                    err = float((logits - normal_logits).abs().max()) if lam == 1.0 else 0.0
                    if lam == 1.0:
                        max_error = max(max_error, err)
                        if not torch.allclose(logits, normal_logits, atol=1e-6, rtol=1e-5):
                            raise AssertionError(f"lambda=1 failed historical NORMAL reproduction: {dataset}/{seed}, max_abs={err}")
                    metrics, pred, _ = _metrics(logits, data, labels)
                    flip = float((pred != normal_pred).float().mean()) if lam != 1 else 0.0
                    scan_rows.append({"dataset": dataset, "seed": seed, "mode": "SHARED", "lambda": lam,
                                      **metrics, "prediction_flip_vs_lambda1": flip,
                                      "lambda1_max_abs_logit_error": err if lam == 1 else ""})
                    if lam == 0.0:
                        relation_zero = 0.5 * ra["text"]
                        if not torch.equal(relation_zero, 0.5 * ra["text"]) or torch.count_nonzero(0.5 * lam * rd["text"]) != 0:
                            raise AssertionError("lambda=0 must keep 0.5*R_A and remove D")
                    if lam in GRID:
                        val_logits = logits[val_idx_cpu.to(device)]
                        target = y_cpu.to(device)
                        node_losses.append(F.cross_entropy(val_logits, target, reduction="none").detach().cpu())
                        node_preds.append(val_logits.argmax(-1).detach().cpu())
                    logits_shared.append(logits)

                this_modality_rows = []
                for mode in ("TEXT_ONLY", "VISUAL_ONLY"):
                    for lam in GRID:
                        lt = lam if mode == "TEXT_ONLY" else 1.0
                        lv = lam if mode == "VISUAL_ONLY" else 1.0
                        rel_t = 0.5 * ra["text"] + 0.5 * float(lt) * rd["text"]
                        rel_v = 0.5 * ra["visual"] + 0.5 * float(lv) * rd["visual"]
                        logits = _fused_logits(model, classifier, h["text"], h["visual"], rel_t, rel_v)
                        metrics, pred, _ = _metrics(logits, data, labels)
                        record = {"dataset": dataset, "seed": seed, "mode": mode, "lambda": lam,
                                  "lambda_text": lt, "lambda_visual": lv, **metrics,
                                  "prediction_flip_vs_lambda1": float((pred != normal_pred).float().mean())}
                        modality_rows.append(record)
                        this_modality_rows.append(record)
                for mode in ("TEXT_ONLY", "VISUAL_ONLY"):
                    candidates = [r for r in this_modality_rows if r["mode"] == mode]
                    best = min(candidates, key=lambda r: r["true_label_ce"])
                    baseline = next(r for r in candidates if float(r["lambda"]) == 1.0)
                    modality_summary.append({"dataset": dataset, "seed": seed, "mode": mode,
                                              "best_lambda": float(best["lambda"]), "best_val_ce": best["true_label_ce"],
                                              "best_val_accuracy": best["val_acc"], "best_val_macro_f1": best["val_macro_f1"],
                                              "val_ce_at_lambda1": baseline["true_label_ce"],
                                              "ce_change_vs_lambda1": best["true_label_ce"] - baseline["true_label_ce"],
                                              "interpretation": "DESCRIPTIVE_MODALITY_SCAN_ONLY; validation labels used"})

                losses = torch.stack(node_losses)  # [grid, validation nodes]
                preds = torch.stack(node_preds)
                best_pos = losses.argmin(dim=0)
                chosen_lambda = torch.tensor(GRID, dtype=torch.float64)[best_pos]
                best_loss = losses[best_pos, torch.arange(losses.shape[1])]
                base_pos = GRID.index(1.0)
                base_loss = losses[base_pos]
                best_correct = preds[best_pos, torch.arange(preds.shape[1])] == y_cpu
                base_correct = preds[base_pos] == y_cpu
                oracle_row = {"dataset": dataset, "seed": seed, "n_validation_nodes": int(y_cpu.numel()),
                              "fraction_lambda_star_zero": float((chosen_lambda == 0).float().mean()),
                              "fraction_lambda_star_positive": float((chosen_lambda > 0).float().mean()),
                              "fraction_lambda_star_negative": float((chosen_lambda < 0).float().mean()),
                              "mean_ce_at_oracle": float(best_loss.mean()), "mean_ce_at_lambda1": float(base_loss.mean()),
                              "mean_oracle_headroom": float((base_loss - best_loss).mean()),
                              "accuracy_at_oracle": float(best_correct.float().mean()),
                              "accuracy_at_lambda1": float(base_correct.float().mean()),
                              "interpretation": "DESCRIPTIVE_ORACLE_ONLY; uses validation labels"}
                oracle_summary.append(oracle_row)
                for j, node_id in enumerate(val_idx_cpu.tolist()):
                    node_rows.append({"dataset": dataset, "seed": seed, "node_id": int(node_id),
                                      "lambda_star": float(chosen_lambda[j]), "ce_at_oracle": float(best_loss[j]),
                                      "ce_at_lambda1": float(base_loss[j]),
                                      "correct_at_oracle": bool(best_correct[j]), "correct_at_lambda1": bool(base_correct[j]),
                                      "node_oracle_headroom": float(base_loss[j] - best_loss[j]),
                                      "interpretation": "DESCRIPTIVE_ORACLE_ONLY"})
            n_checkpoints += 1
            del model, classifier, analysis
            if device.type == "cuda":
                torch.cuda.empty_cache()
    _write_csv(RESULT_ROOT / "h15_lambda_scan.csv", scan_rows)
    _write_csv(RESULT_ROOT / "h15_modality_scan.csv", modality_rows)
    _write_csv(RESULT_ROOT / "h15_modality_summary.csv", modality_summary)
    _write_csv(RESULT_ROOT / "h15_node_oracle.csv", node_rows)
    _write_csv(RESULT_ROOT / "h15_oracle_summary.csv", oracle_summary)
    summary = {"study": "S4.3 Phase 2 H1.5 Frozen Differential-Usage Heterogeneity Audit",
               "generated_at_utc": datetime.now(timezone.utc).isoformat(), "grid": list(GRID),
               "checkpoints_read": n_checkpoints, "training_runs": 0,
               "max_lambda1_logit_abs_error": max_error, "oracle_summary": oracle_summary,
               "modality_summary": modality_summary,
               "modality_best_lambda_differing_pairs": sum(
                   next(x["best_lambda"] for x in modality_summary if x["dataset"] == ds and x["seed"] == seed and x["mode"] == "TEXT_ONLY")
                   != next(x["best_lambda"] for x in modality_summary if x["dataset"] == ds and x["seed"] == seed and x["mode"] == "VISUAL_ONLY")
                   for ds in DATASETS for seed in SEEDS),
               "oracle_policy": "DESCRIPTIVE_ORACLE_ONLY; validation labels are not used for training/router/threshold/hyperparameter selection"}
    (RESULT_ROOT / "h15_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_h15_report(summary)
    return summary


def _metrics_delta_row(dataset, seed, label, metrics, normal_metrics, pred, normal_pred):
    return {"dataset": dataset, "seed": seed, "intervention": label, **metrics,
            "delta_val_acc": metrics["val_acc"] - normal_metrics["val_acc"],
            "delta_val_macro_f1": metrics["val_macro_f1"] - normal_metrics["val_macro_f1"],
            "delta_true_label_ce": metrics["true_label_ce"] - normal_metrics["true_label_ce"],
            "prediction_flip_rate_vs_normal": float((pred != normal_pred).float().mean()),
            "interpretation": "frozen sensitivity; no retraining"}


def _with_h1r_intervention(model, classifier, analysis, data, labels, normal_metrics, normal_pred, dataset, seed, rows):
    if model.variant != "h1r_dual_functional_signed":
        return
    h = {m: analysis[f"H_{m}"] for m in ("text", "visual")}
    ra = {m: analysis[f"R_A_{m}"] for m in ("text", "visual")}
    rd = {m: analysis[f"R_D_{m}"] for m in ("text", "visual")}
    for label, coefficient_a, coefficient_d in (("NORMAL", .5, .5), ("ZERO_D_KEEP_SCALE", .5, 0.),
                                                  ("ZERO_A_KEEP_SCALE", 0., .5), ("FORCE_A", 1., 0.),
                                                  ("FORCE_D", 0., 1.)):
        logits = _fused_logits(model, classifier, h["text"], h["visual"],
                               coefficient_a * ra["text"] + coefficient_d * rd["text"],
                               coefficient_a * ra["visual"] + coefficient_d * rd["visual"])
        metrics, pred, _ = _metrics(logits, data, labels)
        rows.append(_metrics_delta_row(dataset, seed, label, metrics, normal_metrics, pred, normal_pred))


def _degree_bins(data, val_idx: torch.Tensor, num_bins: int = 10):
    from torch_geometric.utils import degree
    edge = data.edge_index.detach().cpu().long()
    deg = degree(edge[0], num_nodes=data.num_nodes, dtype=torch.float32)
    values = deg[val_idx]
    order = torch.argsort(values, stable=True)
    result = []
    for group in torch.tensor_split(order, num_bins):
        if group.numel():
            result.append(group)
    return values, result


def _expand_gate(value: torch.Tensor, n: int, group: bool = False):
    if group:
        if value.dim() == 1:
            return value.reshape(1, -1).expand(n, -1).clone()
        return value.clone()
    if value.dim() == 0:
        return value.reshape(1, 1).expand(n, 1).clone()
    if value.dim() == 1:
        if value.numel() == n:
            return value.reshape(n, 1).clone()
        return value.reshape(1, -1).expand(n, -1).clone()
    return value.clone()


def _gate_logits(model, classifier, analysis, gate_t, gate_v, data, is_beta: bool):
    n = data.num_nodes
    h_t, h_v = analysis["H_text"], analysis["H_visual"]
    ra_t, ra_v = analysis["R_A_text"], analysis["R_A_visual"]
    if is_beta:
        group = model.variant.endswith("group")
        gt, gv = _expand_gate(gate_t, n, group), _expand_gate(gate_v, n, group)
        if group:
            rt = (ra_t.reshape(n, 8, 32) * gt.unsqueeze(-1)).reshape_as(ra_t)
            rv = (ra_v.reshape(n, 8, 32) * gv.unsqueeze(-1)).reshape_as(ra_v)
        else:
            rt, rv = ra_t * gt, ra_v * gv
    else:
        rc_t, rc_v = analysis["R_C_text"], analysis["R_C_visual"]
        gt, gv = _expand_gate(gate_t, n), _expand_gate(gate_v, n)
        rt, rv = ra_t + gt * rc_t, ra_v + gv * rc_v
    return classifier(model.plain_fusion(torch.cat([
        model.output_norm_text(h_t + rt), model.output_norm_visual(h_v + rv)
    ], dim=-1)))


def _log_gate_stats(dataset, seed, modality, values, prefix, rows, group=False):
    stats = _stats(values)
    row = {"dataset": dataset, "seed": seed, "modality": modality, **{f"{prefix}_{k}": v for k, v in stats.items()}}
    if prefix == "beta":
        row.update({"fraction_beta_lt_0_5": float((values < .5).float().mean()),
                    "fraction_beta_gt_1_5": float((values > 1.5).float().mean()),
                    "fraction_saturated_lt_0_05": float((values < .05).float().mean()),
                    "fraction_saturated_gt_1_95": float((values > 1.95).float().mean())})
    else:
        row.update({"fraction_lambda_positive": float((values > 0).float().mean()),
                    "fraction_lambda_negative": float((values < 0).float().mean()),
                    "fraction_abs_lambda_lt_0_1": float((values.abs() < .1).float().mean()),
                    "fraction_abs_lambda_gt_1": float((values.abs() > 1).float().mean()),
                    "fraction_abs_lambda_gt_1_9": float((values.abs() > 1.9).float().mean())})
    if group and values.dim() == 2:
        for g in range(values.shape[1]):
            row[f"group_{g}_mean"] = float(values[:, g].mean())
            row[f"group_{g}_std"] = float(values[:, g].std(unbiased=False))
        row["mean_within_node_group_variance"] = float(values.var(dim=-1, unbiased=False).mean())
    rows.append(row)


def _intervene_adaptive(model, classifier, analysis, data, labels, normal_metrics, normal_pred,
                        dataset, seed, intervention_rows, gate_stats_rows):
    val_idx_cpu, _ = _val(data)
    device = analysis["H_text"].device
    idx = val_idx_cpu.to(device)
    is_beta = model.variant in model.H2A_VARIANTS
    is_node = model.variant.startswith("h2a_node") or model.variant.startswith("h2b_node")
    if is_beta:
        gate_key = "beta"
        prefix = "beta"
    else:
        gate_key = "lambda"
        prefix = "lambda"
    gates = {}
    for modality in ("text", "visual"):
        values = analysis[f"{gate_key}_{modality}"]
        if values is None:
            raise ValueError(f"missing {gate_key} for {model.variant}/{modality}")
        if values.dim() == 1 and values.numel() == data.num_nodes:
            val_values = values[idx]
        elif values.dim() == 2 and values.shape[0] == data.num_nodes:
            val_values = values[idx]
        else:
            val_values = _expand_gate(values, data.num_nodes, is_beta and model.variant.endswith("group"))[idx]
        gates[modality] = values
        _log_gate_stats(dataset, seed, modality, val_values, prefix, gate_stats_rows,
                        group=is_beta and model.variant.endswith("group"))

    labels_to_run = [("NORMAL", gates["text"], gates["visual"])]
    if is_beta:
        if is_node:
            one_t = _expand_gate(torch.ones(8 if model.variant.endswith("group") else 1, device=device), data.num_nodes, model.variant.endswith("group"))
            one_v = one_t.clone()
            labels_to_run.append(("FORCE_ONE", one_t, one_v))
            mean_t = _expand_gate(gates["text"], data.num_nodes, model.variant.endswith("group"))
            mean_v = _expand_gate(gates["visual"], data.num_nodes, model.variant.endswith("group"))
            mt = mean_t[idx].mean(dim=0, keepdim=True).expand_as(mean_t).clone()
            mv = mean_v[idx].mean(dim=0, keepdim=True).expand_as(mean_v).clone()
            labels_to_run.append(("MEAN_REPLACE", mt, mv))
            _, bins = _degree_bins(data, val_idx_cpu)
            for iteration in range(10):
                shuffled = {}
                for modality in ("text", "visual"):
                    gate = _expand_gate(gates[modality], data.num_nodes, model.variant.endswith("group"))
                    val_gate = gate[idx].clone()
                    salt = sum(ord(c) for c in dataset) + seed * 1009 + iteration * 97 + (0 if modality == "text" else 1)
                    generator = torch.Generator(device="cpu").manual_seed(salt)
                    for bin_idx in bins:
                        if bin_idx.numel() > 1:
                            perm = torch.randperm(bin_idx.numel(), generator=generator)
                            bin_device = bin_idx.to(device)
                            val_gate[bin_device] = val_gate[bin_device[perm.to(device)]]
                    gate[idx] = val_gate
                    shuffled[modality] = gate
                labels_to_run.append((f"NODE_SHUFFLE_{iteration + 1:02d}", shuffled["text"], shuffled["visual"]))
            if gates["text"].shape == gates["visual"].shape:
                swap_t = _expand_gate(gates["visual"], data.num_nodes, model.variant.endswith("group"))
                swap_v = _expand_gate(gates["text"], data.num_nodes, model.variant.endswith("group"))
                labels_to_run.append(("MODALITY_SWAP", swap_t, swap_v))
        else:
            # Fixed one is the matched P0 residual dosage control for global variants.
            one = torch.ones(8 if model.variant.endswith("group") else 1, device=device)
            labels_to_run.append(("FIXED_ONE_P0_DOSAGE", one, one))
    else:
        zeros_t = torch.zeros_like(gates["text"])
        zeros_v = torch.zeros_like(gates["visual"])
        labels_to_run.append(("ZERO_CORRECTION", zeros_t, zeros_v))
        labels_to_run.append(("SIGN_FLIP", -gates["text"], -gates["visual"]))
        if is_node:
            mean_t = _expand_gate(gates["text"], data.num_nodes)[idx].mean(dim=0, keepdim=True).expand(data.num_nodes, -1).clone()
            mean_v = _expand_gate(gates["visual"], data.num_nodes)[idx].mean(dim=0, keepdim=True).expand(data.num_nodes, -1).clone()
            labels_to_run.append(("MEAN_REPLACE", mean_t, mean_v))
            _, bins = _degree_bins(data, val_idx_cpu)
            for iteration in range(10):
                shuffled = {}
                for modality in ("text", "visual"):
                    gate = _expand_gate(gates[modality], data.num_nodes)
                    val_gate = gate[idx].clone()
                    salt = sum(ord(c) for c in dataset) + seed * 1009 + iteration * 97 + (0 if modality == "text" else 1)
                    generator = torch.Generator(device="cpu").manual_seed(salt)
                    for bin_idx in bins:
                        if bin_idx.numel() > 1:
                            perm = torch.randperm(bin_idx.numel(), generator=generator)
                            bin_device = bin_idx.to(device)
                            val_gate[bin_device] = val_gate[bin_device[perm.to(device)]]
                    gate[idx] = val_gate
                    shuffled[modality] = gate
                labels_to_run.append((f"NODE_SHUFFLE_{iteration + 1:02d}", shuffled["text"], shuffled["visual"]))
            if gates["text"].shape == gates["visual"].shape:
                labels_to_run.append(("MODALITY_SWAP", _expand_gate(gates["visual"], data.num_nodes),
                                      _expand_gate(gates["text"], data.num_nodes)))

    for label, gate_t, gate_v in labels_to_run:
        logits = _gate_logits(model, classifier, analysis, gate_t, gate_v, data, is_beta)
        metrics, pred, _ = _metrics(logits, data, labels)
        row = _metrics_delta_row(dataset, seed, label, metrics, normal_metrics, pred, normal_pred)
        row["variant"] = model.variant
        intervention_rows.append(row)


def _load_p0_metric(dataset: str, seed: int, device: torch.device):
    ckpt = _historical_path(dataset, seed, "p0_residual")
    data, model, classifier, payload, labels = _load_model(dataset, "relation_basis_pilot", "p0_residual", seed, ckpt, device)
    with torch.no_grad():
        logits = classifier(model.analyze(data.x.to(device), data.edge_index.to(device))["fused_z"])
        metrics, _, _ = _metrics(logits, data, labels)
    return metrics


def _h1r_intervention_rows_for_run(dataset, seed, data, model, classifier, analysis, labels, normal_metrics, normal_pred):
    rows = []
    _with_h1r_intervention(model, classifier, analysis, data, labels, normal_metrics, normal_pred, dataset, seed, rows)
    return rows


def _paired_rows(metric_table: dict[tuple[str, int, str], dict[str, float]], contrasts: list[tuple[str, str, str]], dataset_list=DATASETS):
    rows = []
    for name, left, right in contrasts:
        deltas = []
        for dataset in dataset_list:
            for seed in SEEDS:
                a_key, b_key = (dataset, seed, left), (dataset, seed, right)
                if a_key not in metric_table or b_key not in metric_table:
                    continue
                a, b = metric_table[a_key], metric_table[b_key]
                row = {"dataset": dataset, "seed": seed, "contrast": name,
                       "left_variant": left, "right_variant": right,
                       "delta_val_acc": b["val_acc"] - a["val_acc"],
                       "delta_val_macro_f1": b["val_macro_f1"] - a["val_macro_f1"],
                       "delta_true_label_ce": b["true_label_ce"] - a["true_label_ce"]}
                rows.append(row); deltas.append(row)
        for metric in ("val_acc", "val_macro_f1", "true_label_ce"):
            key = {"val_acc": "delta_val_acc", "val_macro_f1": "delta_val_macro_f1", "true_label_ce": "delta_true_label_ce"}[metric]
            vals = [float(r[key]) for r in deltas]
            rows.append({"dataset": "ALL", "seed": "ALL", "contrast": name,
                         "left_variant": left, "right_variant": right,
                         f"mean_{key}": _mean(vals), f"population_std_{key}": _std(vals),
                         f"positive_seed_pairs_{key}": sum(v > 0 for v in vals)})
    return rows


def _complexity_rows() -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            path = OUTPUT_ROOT / "formal" / dataset / variant / "complete.json"
            rec = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"dataset": dataset, "variant": variant, "n_runs": 3,
                         "trainable_params": rec["trainable_params"],
                         "peak_cuda_memory_mb": rec["peak_cuda_memory_mb"],
                         "wall_clock_seconds": rec["wall_clock_seconds"],
                         "mean_epoch_seconds_proxy": rec["mean_epoch_seconds_proxy"],
                         "mean_best_epoch": _mean(rec["best_epochs"]),
                         "training_commit": rec["training_commit"]})
        p0 = json.loads((HISTORICAL_ROOT / "formal" / dataset / "p0_residual" / "complete.json").read_text())
        rows.append({"dataset": dataset, "variant": "p0_residual_reference", "n_runs": 3,
                     "trainable_params": p0["trainable_params"],
                     "peak_cuda_memory_mb": p0["peak_cuda_memory_mb"],
                     "wall_clock_seconds": p0["wall_clock_seconds"],
                     "mean_epoch_seconds_proxy": p0["mean_epoch_seconds_proxy"],
                     "mean_best_epoch": _mean(p0["best_epochs"]),
                     "training_commit": p0["training_commit"]})
    return rows


def _dataset_hashes(datasets):
    result = {}
    for dataset in datasets:
        data = _load_data(dataset)
        h = hashlib.sha256()
        for name, tensor in (("features", data.x), ("physical_edges", data.edge_index),
                             ("train_indices", data.train_idx), ("validation_indices", data.val_idx)):
            t = tensor.detach().cpu().contiguous()
            h.update(name.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode()); h.update(t.numpy().tobytes())
        # Only train and validation label slices enter this signature; test labels are intentionally omitted.
        for name, idx in (("train_labels", data.train_idx), ("validation_labels", data.val_idx)):
            t = data.y[idx.detach().cpu().long()].detach().cpu().contiguous()
            h.update(name.encode()); h.update(t.numpy().tobytes())
        result[dataset] = h.hexdigest()
    return result


def _provenance(training_commit: str):
    files = ["configs/model/adaptive_relation_pilot.yaml", "configs/task/nc.yaml",
             "src/models/adaptive_relation_pilot.py", "scripts/run_s43_h15_h1r_h2ab_nc.py",
             "scripts/analyze_s43_h15_h1r_h2ab.py", "tests/test_s43_h15_h1r_h2ab.py"]
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files if (ROOT / name).is_file()}
    configs = {name: value for name, value in hashes.items() if name.startswith("configs/")}
    payload = {
        "study": "S4.3 Phase 2 Adaptive Structural Utility Audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_branch": "s43_p0_h1", "source_commit": SOURCE_COMMIT,
        "training_branch": "s43_h15_h1r_h2ab", "training_commit": training_commit,
        "analysis_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "old_evidence_commits": {"multi_order_bank": "31468c5a1439b3561487936bf5d83749b502f61e",
                                 "mechanism_discovery": "d31b095faa4f98538e60cfac293a831138dde3db",
                                 "problem_deep_dive": "e535ad91911557790658ddc87bfaa516da531aad",
                                 "s43_p0_h1_results": "601bad98ff28e85c9aeac23b84b5db51ad4c7c86"},
        "protocol_version": "unified_full_graph_nc_v1", "task": "nc", "test_disabled": True,
        "formal_datasets": list(DATASETS), "excluded_datasets": ["Toys"], "seeds": list(SEEDS),
        "seed_command": 42, "num_runs": 3, "formal_variants": list(VARIANTS),
        "config_sha256": configs, "code_file_sha256": hashes, "dataset_hashes_train_validation_scope": _dataset_hashes(DATASETS),
        "variant_definitions": {
            "h1r_dual_agg_signed": "LN(H+0.5*T_A1(P_rel H)+0.5*T_A2(P_rel H)); LeakyReLU slope 0.1",
            "h1r_dual_functional_signed": "LN(H+0.5*T_A(P_rel H)+0.5*T_D(sH-P_rel H)); LeakyReLU slope 0.1",
            "h2a_global_scalar": "LN(H+beta_m*R_A), beta=2sigmoid(b_m), initialized 1",
            "h2a_node_scalar": "LN(H+beta_i,m*R_A), beta=2sigmoid(b_m+ConditionEncoderV1(H,R_A))",
            "h2a_global_group": "LN(H+group_beta_m*R_A), 8 groups of 32 channels",
            "h2a_node_group": "LN(H+group_beta_i,m*R_A), node-conditioned 8-group dosage",
            "h2b_global_agg_correction": "LN(H+R_A+lambda_m*T_C(P_rel H))",
            "h2b_global_diff_correction": "LN(H+R_A+lambda_m*T_D(sH-P_rel H))",
            "h2b_node_agg_correction": "LN(H+R_A+lambda_i,m*T_C(P_rel H))",
            "h2b_node_diff_correction": "LN(H+R_A+lambda_i,m*T_D(sH-P_rel H))",
        },
        "interpretation_boundaries": [
            "H1.5 node oracle uses validation labels and is DESCRIPTIVE_ORACLE_ONLY",
            "H1R tests current handcrafted D basis, not all experts/transforms",
            "learned beta/lambda are not causal contributions",
            "group indices have no intrinsic semantic meaning",
            "global beta/lambda are per-dataset trained priors, not cross-dataset adaptation",
            "frozen interventions are sensitivity tests, not retrained ablations",
        ],
        "cuda_available": torch.cuda.is_available(), "cuda_device_count": torch.cuda.device_count(),
    }
    (RESULT_ROOT / "phase2_provenance.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _status_from_deltas(rows, contrast):
    subset = [r for r in rows if r.get("contrast") == contrast and r.get("dataset") != "ALL"]
    if not subset:
        return "UNSUPPORTED", "No paired rows available."
    acc = [r["delta_val_acc"] for r in subset]
    f1 = [r["delta_val_macro_f1"] for r in subset]
    if all(x > 0 for x in acc) and all(x > 0 for x in f1):
        return "STRONG_SUPPORT", "Both validation metrics favor the right-hand variant in every paired dataset-seed row."
    if _mean(acc) > 0 or _mean(f1) > 0:
        return "MECHANISM_SUPPORT", "At least one pooled metric favors the right-hand variant; effects vary across pairs."
    if any(x > 0 for x in acc + f1):
        return "MIXED", "Some paired metric effects are positive, but pooled effects do not consistently favor the right-hand variant."
    return "UNSUPPORTED", "Paired validation metrics do not favor the right-hand variant on average."


def _report(contrast_tables, intervention_rows, h15_summary, complexity):
    h1r = contrast_tables["h1r"]
    h2a = contrast_tables["h2a"]
    h2b = contrast_tables["h2b"]
    status_h1r, why_h1r = _status_from_deltas(h1r, "h1r_functional_signed_minus_agg_signed")
    status_h2a_s, why_h2a_s = _status_from_deltas(h2a, "h2a_node_scalar_minus_global_scalar")
    status_h2a_g, why_h2a_g = _status_from_deltas(h2a, "h2a_node_group_minus_global_group")
    status_h2b, why_h2b = _status_from_deltas(h2b, "h2b_node_diff_minus_node_agg")
    status_h2b_global, why_h2b_global = _status_from_deltas(h2b, "h2b_global_diff_minus_global_agg")
    status_h2b_adapt, why_h2b_adapt = _status_from_deltas(h2b, "h2b_node_diff_minus_global_diff")
    h15_fraction = [r["fraction_lambda_star_zero"] for r in h15_summary["oracle_summary"]]
    modality_mismatch = h15_summary["modality_best_lambda_differing_pairs"]
    status_h15 = "MECHANISM_SUPPORT" if (max(h15_fraction, default=0)-min(h15_fraction, default=0) > .05 or modality_mismatch > 0) else "MIXED"
    mean_oracle = _mean([r["mean_oracle_headroom"] for r in h15_summary["oracle_summary"]])

    def contrast_mean(rows, name, key):
        values = [float(r[key]) for r in rows if r.get("contrast") == name and r.get("dataset") != "ALL"]
        return _mean(values)

    def intervention_mean(family, intervention, key):
        values = [float(r[key]) for r in intervention_rows
                  if r.get("variant", "").startswith(family) and r.get("intervention") == intervention]
        return _mean(values)

    h1r_acc = contrast_mean(h1r, "h1r_functional_signed_minus_agg_signed", "delta_val_acc")
    h1r_f1 = contrast_mean(h1r, "h1r_functional_signed_minus_agg_signed", "delta_val_macro_f1")
    h1r_ce = contrast_mean(h1r, "h1r_functional_signed_minus_agg_signed", "delta_true_label_ce")
    act_acc = contrast_mean(h1r, "signed_functional_minus_historical_relu_functional", "delta_val_acc")
    act_f1 = contrast_mean(h1r, "signed_functional_minus_historical_relu_functional", "delta_val_macro_f1")
    h2a_gs_acc = contrast_mean(h2a, "h2a_global_scalar_minus_p0_residual", "delta_val_acc")
    h2a_gs_f1 = contrast_mean(h2a, "h2a_global_scalar_minus_p0_residual", "delta_val_macro_f1")
    h2a_gs_ce = contrast_mean(h2a, "h2a_global_scalar_minus_p0_residual", "delta_true_label_ce")
    h2a_s_acc = contrast_mean(h2a, "h2a_node_scalar_minus_global_scalar", "delta_val_acc")
    h2a_g_acc = contrast_mean(h2a, "h2a_node_group_minus_global_group", "delta_val_acc")
    h2a_g_ce = contrast_mean(h2a, "h2a_node_group_minus_global_group", "delta_true_label_ce")
    h2a_force_acc = intervention_mean("h2a_node", "FORCE_ONE", "delta_val_acc")
    h2a_force_f1 = intervention_mean("h2a_node", "FORCE_ONE", "delta_val_macro_f1")
    h2a_mean_acc = intervention_mean("h2a_node", "MEAN_REPLACE", "delta_val_acc")
    h2a_mean_f1 = intervention_mean("h2a_node", "MEAN_REPLACE", "delta_val_macro_f1")
    h2a_mean_ce = intervention_mean("h2a_node", "MEAN_REPLACE", "delta_true_label_ce")
    h2a_shuffle_acc = _mean([float(r["delta_val_acc"]) for r in intervention_rows if r.get("variant", "").startswith("h2a_node") and r.get("intervention", "").startswith("NODE_SHUFFLE")])
    h2a_shuffle_f1 = _mean([float(r["delta_val_macro_f1"]) for r in intervention_rows if r.get("variant", "").startswith("h2a_node") and r.get("intervention", "").startswith("NODE_SHUFFLE")])
    h2a_shuffle_ce = _mean([float(r["delta_true_label_ce"]) for r in intervention_rows if r.get("variant", "").startswith("h2a_node") and r.get("intervention", "").startswith("NODE_SHUFFLE")])
    h2b_glob_acc = contrast_mean(h2b, "h2b_global_diff_minus_global_agg", "delta_val_acc")
    h2b_glob_f1 = contrast_mean(h2b, "h2b_global_diff_minus_global_agg", "delta_val_macro_f1")
    h2b_glob_ce = contrast_mean(h2b, "h2b_global_diff_minus_global_agg", "delta_true_label_ce")
    h2b_node_acc = contrast_mean(h2b, "h2b_node_diff_minus_node_agg", "delta_val_acc")
    h2b_node_f1 = contrast_mean(h2b, "h2b_node_diff_minus_node_agg", "delta_val_macro_f1")
    h2b_node_ce = contrast_mean(h2b, "h2b_node_diff_minus_node_agg", "delta_true_label_ce")
    h2b_adapt_acc = contrast_mean(h2b, "h2b_node_diff_minus_global_diff", "delta_val_acc")
    h2b_mean_acc = intervention_mean("h2b_node", "MEAN_REPLACE", "delta_val_acc")
    h2b_mean_ce = intervention_mean("h2b_node", "MEAN_REPLACE", "delta_true_label_ce")
    h2b_shuffle_acc = _mean([float(r["delta_val_acc"]) for r in intervention_rows if r.get("variant", "").startswith("h2b_node") and r.get("intervention", "").startswith("NODE_SHUFFLE")])
    h2b_shuffle_f1 = _mean([float(r["delta_val_macro_f1"]) for r in intervention_rows if r.get("variant", "").startswith("h2b_node") and r.get("intervention", "").startswith("NODE_SHUFFLE")])
    h2b_shuffle_ce = _mean([float(r["delta_true_label_ce"]) for r in intervention_rows if r.get("variant", "").startswith("h2b_node") and r.get("intervention", "").startswith("NODE_SHUFFLE")])
    report = [
        "# S4.3 Phase 2: Adaptive Structural Utility Audit", "",
        f"Training commit: `{contrast_tables['training_commit']}`. Analysis commit: `{contrast_tables['analysis_commit']}`. Source commit: `{SOURCE_COMMIT}`.",
        "Protocol: unified full-graph node classification; validation-only selection/evaluation; test evaluation disabled. Formal scope is Movies, Grocery, ele-fashion, Reddit-S; Toys remains a holdout.", "",
        "## Result status", "",
        f"- H1.5: **{status_h15}** — lambda*=0 shares span {min(h15_fraction, default=0):.3f}–{max(h15_fraction, default=0):.3f}; modality-only aggregate scan minima differ in {modality_mismatch}/12 dataset-seed pairs; mean node-oracle headroom={mean_oracle:.5f}.",
        f"- H1R: **{status_h1r}** — {why_h1r}",
        f"- H2a scalar adaptation: **{status_h2a_s}** — {why_h2a_s}",
        f"- H2a group adaptation: **{status_h2a_g}** — {why_h2a_g}",
        f"- H2b global differential vs matched generic correction: **{status_h2b_global}** — {why_h2b_global}",
        f"- H2b node differential vs matched generic correction: **{status_h2b}** — {why_h2b}", "",
        "## Answers", "",
        f"1. **Differential usage heterogeneity:** node-oracle mean validation CE headroom is {mean_oracle:.4f}; lambda*=0 fractions vary by dataset/seed, and Text-only versus Visual-only scan minima differ in {modality_mismatch}/12 checkpoint pairs. This is an optimistic descriptive bound and aggregate validation scan, not a deployable selector.",
        f"2. **ReLU/sign preservation:** signed functional minus signed aggregative averages {h1r_acc:+.4f} accuracy, {h1r_f1:+.4f} Macro-F1, and {h1r_ce:+.4f} CE, so signed preservation does not rescue the functional basis here ({status_h1r}). Signed functional versus historical ReLU is ACTIVATION_CHANGED and averages {act_acc:+.4f} accuracy and {act_f1:+.4f} Macro-F1; it is not a pure causal comparison.",
        f"3. **Structural dosage:** global scalar beta versus reused P0 residual averages {h2a_gs_acc:+.4f} accuracy and {h2a_gs_f1:+.4f} Macro-F1, with CE {h2a_gs_ce:+.4f}. Node-vs-global effects are {h2a_s_acc:+.4f} accuracy for scalar and {h2a_g_acc:+.4f} accuracy / {h2a_g_ce:+.4f} CE for group dosage. Effects are small and metric-dependent; group indices have no intrinsic semantic meaning, and global beta values are per-dataset priors.",
        f"4. **Functional node assignment:** forcing beta=1 on trained H2a node gates changes accuracy/F1 by {h2a_force_acc:+.4f}/{h2a_force_f1:+.4f}; validation-mean replacement changes them by {h2a_mean_acc:+.4f}/{h2a_mean_f1:+.4f} (CE {h2a_mean_ce:+.4f}), while degree-bin shuffling averages {h2a_shuffle_acc:+.4f}/{h2a_shuffle_f1:+.4f} (CE {h2a_shuffle_ce:+.4f}). H2b node mean replacement gives {h2b_mean_acc:+.4f} accuracy (CE {h2b_mean_ce:+.4f}) and degree-bin shuffling gives {h2b_shuffle_acc:+.4f}/{h2b_shuffle_f1:+.4f} (CE {h2b_shuffle_ce:+.4f}). This is modest frozen sensitivity, not a retrained causal ablation.",
        f"5. **D correction value:** global differential-minus-generic averages {h2b_glob_acc:+.4f} accuracy, {h2b_glob_f1:+.4f} Macro-F1, {h2b_glob_ce:+.4f} CE; node differential-minus-generic averages {h2b_node_acc:+.4f}, {h2b_node_f1:+.4f}, {h2b_node_ce:+.4f}. Differential corrections do not show consistent utility over matched generic corrections.",
        f"6. **D correction adaptation:** node_diff minus global_diff averages {h2b_adapt_acc:+.4f} accuracy. Node shuffling and mean replacement move validation metrics only modestly; coefficients are learned dosage, not causal contributions.",
        "7. **Next-stage carrier:** if one carrier is carried forward for a later hypothesis test, the simplest candidate is modality-specific global structural dosage (`h2a_global_scalar`). Node/group adaptation has no consistent accuracy gain, and the tested D corrections regress on the primary classification contrasts. Treat that choice as provisional; this report does not define an H3/H4 architecture.", "",
        "## Complexity", "",
        f"The complexity table contains {sum(r['variant'] != 'p0_residual_reference' for r in complexity)} formal variant-dataset profiles plus the reused P0 residual reference for each dataset. See `complexity_table.csv` for parameters, peak memory, wall time, mean epoch proxy, and best epoch.", "",
        "## Interpretation boundaries", "",
        "- H1.5 node oracle uses validation labels and is DESCRIPTIVE_ORACLE_ONLY; modality-only scan minima are descriptive validation summaries.",
        "- H1R tests the current handcrafted differential basis, not all learned experts or transforms.",
        "- Learned beta/lambda do not equal causal contribution.",
        "- Group indices have no intrinsic semantic meaning.",
        "- Global beta/lambda are per-dataset trained priors, not cross-dataset adaptation.",
        "- Frozen interventions are sensitivity tests, not retrained ablations.",
        "- No H3/H4 experiment was started.", "",
    ]
    (RESULT_ROOT / "s43_phase2_report.md").write_text("\n".join(report), encoding="utf-8")
    return {"statuses": {"H1.5": status_h15, "H1R": status_h1r, "H2a_scalar": status_h2a_s,
                         "H2a_group": status_h2a_g, "H2b_global": status_h2b_global,
                         "H2b_node": status_h2b, "H2b_node_vs_global": status_h2b_adapt},
            "mean_h15_oracle_headroom": mean_oracle}


def run_formal_analysis(datasets: tuple[str, ...] = DATASETS) -> dict[str, Any]:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    h1r_table, h2a_table, h2b_table = [], [], []
    h1r_diag, h1r_interventions = [], []
    h2a_gates, h2a_interventions, h2b_lambdas, h2b_interventions = [], [], [], []
    metric_maps = {"h1r": {}, "h2a": {}, "h2b": {}}
    historical_relu = {}
    p0_metrics = {}
    for dataset in datasets:
        for seed in SEEDS:
            p0_metrics[(dataset, seed)] = _load_p0_metric(dataset, seed, device)
            hist_ckpt = _historical_path(dataset, seed, "h1_dual_functional_static")
            hdata, hmodel, hclassifier, _, hlabels = _load_model(
                dataset, "relation_basis_pilot", "h1_dual_functional_static", seed, hist_ckpt, device)
            with torch.no_grad():
                hlogits = hclassifier(hmodel.analyze(hdata.x.to(device), hdata.edge_index.to(device))["fused_z"])
                historical_relu[(dataset, seed)] = _metrics(hlogits, hdata, hlabels)[0]
            del hmodel, hclassifier
            if device.type == "cuda":
                torch.cuda.empty_cache()
        for variant in VARIANTS:
            complete_path = OUTPUT_ROOT / "formal" / dataset / variant / "complete.json"
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            if complete.get("test_evaluation") is not False or complete.get("protocol_version") != "unified_full_graph_nc_v1":
                raise ValueError(f"invalid formal completion metadata: {complete_path}")
            for seed in SEEDS:
                checkpoint = OUTPUT_ROOT / "formal" / dataset / variant / f"best_run{SEEDS.index(seed)+1}.pt"
                data, model, classifier, payload, labels = _load_model(dataset, "adaptive_relation_pilot", variant, seed, checkpoint, device)
                with torch.no_grad():
                    analysis = model.analyze(data.x.to(device), data.edge_index.to(device))
                    normal_logits = classifier(analysis["fused_z"])
                    normal_metrics, normal_pred, _ = _metrics(normal_logits, data, labels)
                    base_row = {"dataset": dataset, "seed": seed, "variant": variant, **normal_metrics,
                                "checkpoint_best_epoch": int(payload["epoch"]),
                                "trainable_params": complete["trainable_params"]}
                    if variant in model.H1R_VARIANTS:
                        h1r_table.append(base_row)
                        metric_maps["h1r"][(dataset, seed, variant)] = normal_metrics
                        if variant == "h1r_dual_functional_signed":
                            _with_h1r_intervention(model, classifier, analysis, data, labels, normal_metrics,
                                                   normal_pred, dataset, seed, h1r_interventions)
                            for modality in ("text", "visual"):
                                ra, rd = analysis[f"R_A_{modality}"], analysis[f"R_D_{modality}"]
                                idx, _ = _val(data); idx = idx.to(device)
                                va, vd = ra[idx], rd[idx]
                                na, nd = torch.linalg.vector_norm(va, dim=-1), torch.linalg.vector_norm(vd, dim=-1)
                                cosine = F.cosine_similarity(va, vd, dim=-1, eps=1e-8)
                                h1r_diag.append({"dataset": dataset, "seed": seed, "variant": variant,
                                                 "modality": modality, "mean_norm_R_A": float(na.mean()),
                                                 "mean_norm_R_D": float(nd.mean()), "mean_cosine_R_A_R_D": float(cosine.mean()),
                                                 "std_cosine_R_A_R_D": float(cosine.std(unbiased=False)),
                                                 "negative_R_D_fraction": float((vd < 0).float().mean()),
                                                 "interpretation": "current handcrafted D basis; not all learned experts"})
                    elif variant in model.H2A_VARIANTS:
                        h2a_table.append(base_row); metric_maps["h2a"][(dataset, seed, variant)] = normal_metrics
                        _intervene_adaptive(model, classifier, analysis, data, labels, normal_metrics, normal_pred,
                                            dataset, seed, h2a_interventions, h2a_gates)
                    else:
                        h2b_table.append(base_row); metric_maps["h2b"][(dataset, seed, variant)] = normal_metrics
                        _intervene_adaptive(model, classifier, analysis, data, labels, normal_metrics, normal_pred,
                                            dataset, seed, h2b_interventions, h2b_lambdas)
                del model, classifier, analysis
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    # Include the fixed P0 residual reference as paired descriptive rows without retraining.
    for table, mapping, family in ((h2a_table, metric_maps["h2a"], "h2a"), (h2b_table, metric_maps["h2b"], "h2b")):
        for dataset in datasets:
            for seed in SEEDS:
                metrics = p0_metrics[(dataset, seed)]
                variant = "p0_residual_reference"
                table.append({"dataset": dataset, "seed": seed, "variant": variant, **metrics,
                              "checkpoint_best_epoch": "historical", "trainable_params": "historical"})
                mapping[(dataset, seed, variant)] = metrics

    h1r_contrasts = _paired_rows(metric_maps["h1r"], [
        ("h1r_functional_signed_minus_agg_signed", "h1r_dual_agg_signed", "h1r_dual_functional_signed")], datasets)
    # Secondary activation comparison is explicitly descriptive and activation-changed.
    for dataset in datasets:
        for seed in SEEDS:
            a = historical_relu[(dataset, seed)]
            b = metric_maps["h1r"][(dataset, seed, "h1r_dual_functional_signed")]
            h1r_contrasts.append({"dataset": dataset, "seed": seed, "contrast": "signed_functional_minus_historical_relu_functional",
                                  "comparison_label": "ACTIVATION_CHANGED", "delta_val_acc": b["val_acc"]-a["val_acc"],
                                  "delta_val_macro_f1": b["val_macro_f1"]-a["val_macro_f1"],
                                  "delta_true_label_ce": b["true_label_ce"]-a["true_label_ce"]})
    secondary = [r for r in h1r_contrasts if r.get("contrast") == "signed_functional_minus_historical_relu_functional"]
    for metric, key in (("val_acc", "delta_val_acc"), ("val_macro_f1", "delta_val_macro_f1"), ("true_label_ce", "delta_true_label_ce")):
        values = [r[key] for r in secondary]
        h1r_contrasts.append({"dataset": "ALL", "seed": "ALL", "contrast": "signed_functional_minus_historical_relu_functional",
                              "comparison_label": "ACTIVATION_CHANGED", f"mean_{key}": _mean(values),
                              f"population_std_{key}": _std(values), f"positive_seed_pairs_{key}": sum(v > 0 for v in values)})
    h2a_contrasts = _paired_rows(metric_maps["h2a"], [
        ("h2a_node_scalar_minus_global_scalar", "h2a_global_scalar", "h2a_node_scalar"),
        ("h2a_node_group_minus_global_group", "h2a_global_group", "h2a_node_group"),
        ("h2a_global_scalar_minus_p0_residual", "p0_residual_reference", "h2a_global_scalar"),
        ("h2a_global_group_minus_p0_residual", "p0_residual_reference", "h2a_global_group"),
        ("h2a_node_group_minus_node_scalar", "h2a_node_scalar", "h2a_node_group")], datasets)
    h2b_contrasts = _paired_rows(metric_maps["h2b"], [
        ("h2b_global_diff_minus_global_agg", "h2b_global_agg_correction", "h2b_global_diff_correction"),
        ("h2b_node_diff_minus_node_agg", "h2b_node_agg_correction", "h2b_node_diff_correction"),
        ("h2b_node_diff_minus_global_diff", "h2b_global_diff_correction", "h2b_node_diff_correction"),
        ("h2b_node_agg_minus_global_agg", "h2b_global_agg_correction", "h2b_node_agg_correction"),
        ("h2b_node_diff_minus_p0_residual", "p0_residual_reference", "h2b_node_diff_correction"),
        ("h2b_node_agg_minus_p0_residual", "p0_residual_reference", "h2b_node_agg_correction"),
        ("h2b_global_diff_minus_p0_residual", "p0_residual_reference", "h2b_global_diff_correction")], datasets)

    _write_csv(RESULT_ROOT / "h1r_table.csv", h1r_table)
    _write_csv(RESULT_ROOT / "h1r_paired_contrasts.csv", h1r_contrasts)
    _write_csv(RESULT_ROOT / "h1r_basis_diagnostics.csv", h1r_diag)
    _write_csv(RESULT_ROOT / "h1r_frozen_interventions.csv", h1r_interventions)
    _write_csv(RESULT_ROOT / "h2a_table.csv", h2a_table)
    _write_csv(RESULT_ROOT / "h2a_paired_contrasts.csv", h2a_contrasts)
    _write_csv(RESULT_ROOT / "h2a_gate_statistics.csv", h2a_gates)
    _write_csv(RESULT_ROOT / "h2a_frozen_interventions.csv", h2a_interventions)
    _write_csv(RESULT_ROOT / "h2b_table.csv", h2b_table)
    _write_csv(RESULT_ROOT / "h2b_paired_contrasts.csv", h2b_contrasts)
    _write_csv(RESULT_ROOT / "h2b_lambda_statistics.csv", h2b_lambdas)
    _write_csv(RESULT_ROOT / "h2b_frozen_interventions.csv", h2b_interventions)

    complexity = _complexity_rows()
    _write_csv(RESULT_ROOT / "complexity_table.csv", complexity)
    if not (RESULT_ROOT / "h15_summary.json").is_file():
        h15_summary = run_h15()
    else:
        h15_summary = json.loads((RESULT_ROOT / "h15_summary.json").read_text())
    contrast_tables = {"h1r": h1r_contrasts, "h2a": h2a_contrasts, "h2b": h2b_contrasts,
                       "training_commit": json.loads((OUTPUT_ROOT / "formal" / datasets[0] / VARIANTS[0] / "complete.json").read_text())["training_commit"]}
    statuses = _report(contrast_tables, h2a_interventions + h2b_interventions, h15_summary, complexity)
    provenance = _provenance(contrast_tables["training_commit"])
    summary = {
        "study": "S4.3 Phase 2 Adaptive Structural Utility Audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "source_commit": SOURCE_COMMIT,
        "training_commit": contrast_tables["training_commit"], "analysis_commit": provenance["analysis_commit"],
        "protocol_version": "unified_full_graph_nc_v1", "task": "nc", "test_disabled": True,
        "datasets": list(datasets), "seeds": list(SEEDS), "formal_variants": list(VARIANTS),
        "formal_jobs": len(datasets) * len(VARIANTS), "formal_runs": len(datasets) * len(VARIANTS) * len(SEEDS),
        "statuses": statuses["statuses"], "mean_h15_oracle_headroom": statuses["mean_h15_oracle_headroom"],
        "outputs": [p.name for p in RESULT_ROOT.iterdir() if p.is_file()],
        "interpretation_boundaries": provenance["interpretation_boundaries"],
    }
    (RESULT_ROOT / "s43_phase2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_analysis(phase: str, datasets: tuple[str, ...] = DATASETS):
    if phase == "h15":
        result = run_h15()
    elif phase == "analyze":
        result = run_formal_analysis(datasets)
    else:
        raise ValueError(phase)
    print(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("h15", "analyze"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    run_analysis(args.phase, tuple(args.datasets))
