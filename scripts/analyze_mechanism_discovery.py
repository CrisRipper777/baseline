from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scipy.sparse.linalg import ArpackNoConvergence, eigsh
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.analysis.mechanism_discovery import (
    FrozenProbeProtocolV1,
    PROBE_CONFIG,
    canonical_undirected_pairs,
    cca_spectrum_summary,
    classification_metrics,
    classwise_metrics,
    cosine_flat,
    edge_percentile_compatibility,
    geometry_summary,
    js_divergence,
    linear_cka,
    node_cross_entropy,
    normalized_dirichlet_energy,
    orthogonal_procrustes_fit,
    quantile_bins,
    remove_random_same_size,
    remove_undirected_pairs,
    safe_spearman,
    train_median_threshold,
)
from src.data import load_mag_data
from src.models import build_model

DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
MOB_ROOT = ROOT / "outputs/mob_factorial_nc_v1"
DISCOVERY_ROOT = ROOT / "outputs/mechanism_discovery_v1"
RESULTS_ROOT = ROOT / "results/mechanism_discovery_v1"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PROBE = FrozenProbeProtocolV1()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=True), encoding="utf-8")


def _compose(dataset: str, seed: int, readout: str, mode: str = "both"):
    with initialize_config_dir(version_base=None, config_dir=str((ROOT / "configs").resolve())):
        cfg = compose(config_name="config", overrides=[
            f"dataset={dataset}", "task=nc", "model=multi_order_bank",
            f"seed={seed}", f"model.readout={readout}",
            f"model.modality_mode={mode}", "model.fusion_mode=plain_mlp",
        ])
    return cfg


def _checkpoint_path(dataset: str, seed: int, name: str, group: str | None = None) -> Path:
    run_id = seed - 41
    if name in {"terminal", "uniform", "gpr"}:
        variant = f"mob_{name}_plain"
        path = MOB_ROOT / dataset / variant / f"best_run{run_id}.pt"
    elif name in {"text_self", "text_uniform", "visual_self", "visual_uniform"}:
        path = DISCOVERY_ROOT / "unimodal" / dataset / name / f"best_run{run_id}.pt"
    else:
        path = DISCOVERY_ROOT / "order_source" / dataset / f"{name}_plain" / f"best_run{run_id}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen checkpoint: {path}")
    return path


def _known_labels(data) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    train = data.train_idx.detach().long().cpu()
    val = data.val_idx.detach().long().cpu()
    train_val = torch.cat([train, val])
    observed = data.y[train_val].detach().long().cpu()
    valid = observed[(observed >= 0) & (observed < int(data.num_classes))]
    classes = sorted(set(map(int, valid.tolist())))
    if not classes:
        raise ValueError(f"{data.name}: no labels in train+validation")
    return train, val, classes


def _load_model_data(dataset: str, seed: int, readout: str, mode: str = "both", checkpoint: Path | None = None):
    cfg = _compose(dataset, seed, readout, mode)
    data = load_mag_data(cfg, "nc", seed)
    train_idx, val_idx, class_ids = _known_labels(data)
    path = checkpoint or _checkpoint_path(dataset, seed, readout)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("task") != "nc" or payload.get("protocol_version") != "unified_full_graph_nc_v1":
        raise ValueError(f"{path}: unexpected protocol")
    if int(payload.get("seed", -1)) != seed:
        raise ValueError(f"{path}: seed mismatch")
    info = {"input_dim": data.input_dim, "num_nodes": data.num_nodes,
            "num_classes": data.num_classes,
            "text_dim": int(data.x_t.size(1)) if data.x_t is not None else 0,
            "visual_dim": int(data.x_i.size(1)) if data.x_i is not None else 0}
    model = build_model(cfg, info).to(DEVICE)
    model.load_state_dict(payload["model_state"], strict=True)
    head = nn.Linear(model.out_dim, int(data.num_classes)).to(DEVICE)
    head.load_state_dict(payload["head_state"], strict=True)
    model.eval(); head.eval()
    return cfg, data, payload, model, head, train_idx, val_idx, class_ids


@torch.no_grad()
def _forward_context(model, head, data, edge_index: torch.Tensor | None = None, x: torch.Tensor | None = None):
    x_gpu = (data.x if x is None else x).to(DEVICE)
    edge_gpu = (data.edge_index if edge_index is None else edge_index).to(DEVICE)
    analysis = model.analyze(x_gpu, edge_gpu)
    logits = head(analysis["fused_z"])
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Nonfinite logits in frozen model analysis")
    result: dict[str, Any] = {}
    for key in ("S_text", "S_visual"):
        value = analysis.get(key)
        result[key] = [item.detach().cpu() for item in value] if value is not None else None
    # Reuse S0 storage instead of copying the two projected matrices a second time.
    result["H0_text"] = result["S_text"][0] if result["S_text"] is not None else None
    result["H0_visual"] = result["S_visual"][0] if result["S_visual"] is not None else None
    for key in ("Z_text", "Z_visual", "fused_z", "gamma_text", "gamma_visual"):
        value = analysis.get(key)
        result[key] = value.detach().cpu() if isinstance(value, torch.Tensor) else value
    result["logits"] = logits.detach().cpu()
    return result


def _val_metrics(logits: torch.Tensor, data, val_idx: torch.Tensor, class_ids: list[int]) -> dict[str, float]:
    return classification_metrics(logits[val_idx], data.y[val_idx], class_ids)


def _probe_row(dataset: str, seed: int, mode: str, order: str, feature: torch.Tensor,
               labels: torch.Tensor, train_idx: torch.Tensor, val_idx: torch.Tensor,
               class_ids: list[int]) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    result = PROBE.fit(feature, labels, train_idx, val_idx, class_ids, DEVICE)
    row = {"dataset": dataset, "encoder_seed": seed, "modality_mode": mode,
           "order": order, "val_accuracy": result["val_metrics"]["accuracy"],
           "val_macro_f1": result["val_metrics"]["macro_f1"],
           "train_accuracy": result["train_metrics"]["accuracy"],
           "trainable_parameters": result["trainable_parameters"],
           "probe_protocol": PROBE.config.name, "probe_config_sha256": result["config_sha256"]}
    return row, {"train_logits": result["train_logits"], "val_logits": result["val_logits"]}


def _step_a() -> dict[str, Any]:
    exp_root = RESULTS_ROOT / "experiment1"
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    spectral = _spectral_audit()
    for dataset in DATASETS:
        for seed in SEEDS:
            print(f"STEP_A {dataset} seed={seed}", flush=True)
            cfg, data, payload, model, head, train, val, classes = _load_model_data(dataset, seed, "uniform")
            out = _forward_context(model, head, data)
            # Mask all out-of-split labels before any probe sees a label tensor.
            known_y = torch.full_like(data.y, -1)
            known_indices = torch.cat([train, val])
            known_y[known_indices] = data.y[known_indices]
            # Never index test labels: probes receive only explicit train/validation indices.
            for readout in ("uniform", "gpr"):
                if readout == "uniform":
                    context = out
                    local_model = model
                else:
                    gcfg, gdata, gp, gm, gh, _, _, _ = _load_model_data(dataset, seed, "gpr")
                    context = _forward_context(gm, gh, gdata)
                    local_model = gm
                for modality in ("text", "visual"):
                    states = context[f"S_{modality}"]
                    geom = geometry_summary(states)
                    for item in geom:
                        item.update({"dataset": dataset, "encoder_seed": seed,
                                     "readout_source": readout, "modality": modality,
                                     "dirichlet_energy": normalized_dirichlet_energy(states[item["order"]], data.edge_index)})
                        for other in range(4):
                            item[f"cosine_to_order_{other}"] = cosine_flat(states[item["order"]], states[other])
                        rows["representation_geometry"].append(item)
                    by_order = {entry["order"]: entry for entry in geom}
                    start, end = by_order[0], by_order[3]
                    variance_drop = end["node_variance"] < start["node_variance"]
                    rank_drop = end["effective_rank"] < start["effective_rank"]
                    energy_drop = end["dirichlet_energy"] < start["dirichlet_energy"]
                    similarity_rise = end["mean_pairwise_node_cosine"] > start["mean_pairwise_node_cosine"]
                    for entry in geom:
                        entry["variance_decreases_S0_to_S3"] = variance_drop
                        entry["effective_rank_decreases_S0_to_S3"] = rank_drop
                        entry["dirichlet_energy_decreases_S0_to_S3"] = energy_drop
                        entry["node_similarity_increases_S0_to_S3"] = similarity_rise
                        entry["joint_smoothing_collapse_evidence"] = bool(variance_drop and rank_drop and energy_drop and similarity_rise)
                if readout == "gpr":
                    gamma_t = [float(x) for x in gp["model_state"]["gamma_text"].tolist()]
                    gamma_v = [float(x) for x in gp["model_state"]["gamma_visual"].tolist()]
                    del gcfg, gdata, gp, gm, gh
                else:
                    del context, local_model
            # Frozen hop and propagation information probes use uniform_plain only.
            states_t, states_v = out["S_text"], out["S_visual"]
            for order in range(4):
                for mode, feature in (("text", states_t[order]), ("visual", states_v[order]),
                                      ("concat", torch.cat([states_t[order], states_v[order]], -1))):
                    row, _ = _probe_row(dataset, seed, mode, str(order), feature,
                                        known_y, train, val, classes)
                    rows["hop_task_probe"].append(row)
            innovations_t = [states_t[k] - states_t[k - 1] for k in range(1, 4)]
            innovations_v = [states_v[k] - states_v[k - 1] for k in range(1, 4)]
            eps = 1e-12
            for k, (dt, dv) in enumerate(zip(innovations_t, innovations_v, strict=True), start=1):
                for modality, delta, prev, initial in (("text", dt, states_t[k-1], states_t[0]),
                                                       ("visual", dv, states_v[k-1], states_v[0])):
                    rows["propagation_innovation"].append({
                        "dataset": dataset, "encoder_seed": seed, "modality": modality, "order": k,
                        "innovation_energy_ratio": float(delta.norm() / (prev.norm() + eps)),
                        "cosine_previous": cosine_flat(delta, prev), "cosine_s0": cosine_flat(delta, initial),
                    })
                for mode, feature in (("text", dt), ("visual", dv), ("concat", torch.cat([dt, dv], -1))):
                    row, _ = _probe_row(dataset, seed, mode, str(k), feature,
                                        known_y, train, val, classes)
                    rows["innovation_probe"].append(row)
            for order in range(1, 4):
                seq_t = torch.cat([states_t[0], *innovations_t[:order]], dim=-1)
                seq_v = torch.cat([states_v[0], *innovations_v[:order]], dim=-1)
                for mode, feature in (("text", seq_t), ("visual", seq_v),
                                      ("concat", torch.cat([seq_t, seq_v], -1))):
                    row, _ = _probe_row(dataset, seed, mode, f"S0_plus_Delta1_to_{order}", feature,
                                        known_y, train, val, classes)
                    rows["incremental_probe"].append(row)
            # Stability diagnostics on the three immutable plain checkpoints.
            for readout in ("terminal", "uniform", "gpr"):
                stab = _stability_for_checkpoint(dataset, seed, readout, train, val, classes)
                rows["stability_audit"].extend(stab)
            del out, model, head, data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    groups: dict[tuple[str, int, str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for item in rows["representation_geometry"]:
        groups[(item["dataset"], item["encoder_seed"], item["readout_source"], item["modality"])][item["order"]] = item
    for (dataset, seed, readout, modality), states in sorted(groups.items()):
        start, end = states[0], states[3]
        rows["representation_smoothing"].append({
            "dataset": dataset, "encoder_seed": seed, "readout_source": readout, "modality": modality,
            "variance_decrease": start["node_variance"] > end["node_variance"],
            "effective_rank_decrease": start["effective_rank"] > end["effective_rank"],
            "dirichlet_energy_decrease": start["dirichlet_energy"] > end["dirichlet_energy"],
            "node_similarity_increase": start["mean_pairwise_node_cosine"] < end["mean_pairwise_node_cosine"],
            "joint_smoothing_collapse_evidence": bool(start["node_variance"] > end["node_variance"] and
                start["effective_rank"] > end["effective_rank"] and
                start["dirichlet_energy"] > end["dirichlet_energy"] and
                start["mean_pairwise_node_cosine"] < end["mean_pairwise_node_cosine"]),
        })
    for name, items in rows.items():
        _write_csv(exp_root / f"{name}.csv", items)
    _write_csv(exp_root / "spectral_response.csv", spectral["response_rows"])
    _write_json(exp_root / "experiment1_summary.json", {
        "stage": "A_existing_checkpoint_zero_training_diagnostics",
        "datasets": list(DATASETS), "seeds": list(SEEDS),
        "probe_config": asdict(PROBE_CONFIG), "probe_config_sha256": PROBE_CONFIG.sha256(),
        "spectral_summary": spectral["summary"],
        "spectral_response_summary": spectral["response_summary"],
        "joint_smoothing_collapse_evidence_count": sum(bool(row["joint_smoothing_collapse_evidence"]) for row in rows["representation_smoothing"]),
        "representation_smoothing_rows": rows["representation_smoothing"],
        "data_boundary": "train+validation only; no test labels read by discovery analysis",
    })
    _write_text(exp_root / "spectral_summary.md", spectral["markdown"])
    _write_text(exp_root / "experiment1_report.md", _render_step_a_report(rows, spectral))
    print("STEP_A_COMPLETE", flush=True)
    return {key: len(value) for key, value in rows.items()}


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _render_step_a_report(rows, spectral) -> str:
    return "\n".join([
        "# Experiment 1: frozen mechanism audit (Step A)", "",
        "This report uses existing terminal/uniform/GPR plain checkpoints. No model was retrained.",
        "All probe fitting and validation reporting use train and validation labels only; test labels are not used.", "",
        f"Frozen probes: AdamW, lr={PROBE_CONFIG.lr}, weight_decay={PROBE_CONFIG.weight_decay}, epochs={PROBE_CONFIG.epochs}, seed={PROBE_CONFIG.seed}.",
        "A probe score is task information evidence, not a causal contribution.", "",
        "The original preregistered `Uniform - Terminal` result remains frozen in `results/nc_benchmark_v1/`; this analysis writes to the separate mechanism_discovery_v1 directory.", "",
        f"Spectral eigensolver records: {len(spectral['summary'])} datasets; see `spectral_summary.md`.", "",
        "## Generated validation-only tables", "",
        *[f"- `{key}.csv`: {len(value)} rows" for key, value in sorted(rows.items())], "",
    ])


def _spectral_audit() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    response_summaries: list[dict[str, Any]] = []
    for dataset in DATASETS:
        cfg = _compose(dataset, 42, "uniform")
        data = load_mag_data(cfg, "nc", 42)
        info = {"input_dim": data.input_dim, "num_nodes": data.num_nodes,
                "num_classes": data.num_classes,
                "text_dim": int(data.x_t.size(1)) if data.x_t is not None else 0,
                "visual_dim": int(data.x_i.size(1)) if data.x_i is not None else 0}
        model = build_model(cfg, info)
        P = model._build_propagation_operator(data.edge_index, data.num_nodes, torch.float32).coalesce()
        ij = P.indices().numpy(); vals = P.values().numpy()
        matrix = sp.coo_matrix((vals, ij), shape=(data.num_nodes, data.num_nodes)).tocsr()
        row: dict[str, Any] = {"dataset": dataset, "num_nodes": data.num_nodes,
                               "num_undirected_edges": int(canonical_undirected_pairs(data.edge_index).size(0)),
                               "operator": "D_tilde^-1/2 (A+I) D_tilde^-1/2"}
        eigenvalues: list[tuple[str, float]] = []
        for label, which in (("largest", "LA"), ("smallest", "SA")):
            try:
                values = eigsh(matrix, k=min(32, data.num_nodes - 2), which=which,
                               return_eigenvectors=False, tol=1e-6, maxiter=max(1000, data.num_nodes * 2))
                row[f"{label}_converged"] = True
                row[f"{label}_count"] = len(values)
                row[f"{label}_failure"] = ""
            except ArpackNoConvergence as exc:
                values = exc.eigenvalues if exc.eigenvalues is not None else np.array([])
                row[f"{label}_converged"] = False
                row[f"{label}_count"] = len(values)
                row[f"{label}_failure"] = str(exc)[:1000]
            except Exception as exc:
                values = np.array([])
                row[f"{label}_converged"] = False
                row[f"{label}_count"] = 0
                row[f"{label}_failure"] = repr(exc)
            for value in values:
                eigenvalues.append((label, float(value)))
        row["eigenvalues"] = [value for _, value in eigenvalues]
        summaries.append(row)
        # Response curves are emitted for each existing GPR seed, preserving the actual learned coefficients.
        for seed in SEEDS:
            gpr_path = _checkpoint_path(dataset, seed, "gpr")
            payload = torch.load(gpr_path, map_location="cpu", weights_only=False)
            gamma_t = payload["model_state"]["gamma_text"].numpy()
            gamma_v = payload["model_state"]["gamma_visual"].numpy()
            seed_responses = []
            for tag, lam in eigenvalues:
                response = {
                    "dataset": dataset, "encoder_seed": seed, "eigenvalue_group": tag,
                    "lambda": lam, "h_terminal": lam ** 3,
                    "h_uniform": sum(lam ** k for k in range(4)) / 4,
                    "h_gpr_text": sum(float(gamma_t[k]) * lam ** k for k in range(4)),
                    "h_gpr_visual": sum(float(gamma_v[k]) * lam ** k for k in range(4)),
                }
                response["mid_frequency"] = int(0.25 <= abs(lam) <= 0.75)
                response["negative_frequency"] = int(lam < 0)
                rows.append(response)
                seed_responses.append(response)
            mid = [item for item in seed_responses if item["mid_frequency"]]
            negative = [item for item in seed_responses if item["negative_frequency"]]
            response_summaries.append({
                "dataset": dataset, "seed": seed,
                "mid_frequency_retention_uniform_mean_abs": float(np.mean([abs(x["h_uniform"]) for x in mid])) if mid else float("nan"),
                "mid_frequency_retention_terminal_mean_abs": float(np.mean([abs(x["h_terminal"]) for x in mid])) if mid else float("nan"),
                "mid_frequency_retention_gpr_text_mean_abs": float(np.mean([abs(x["h_gpr_text"]) for x in mid])) if mid else float("nan"),
                "mid_frequency_retention_gpr_visual_mean_abs": float(np.mean([abs(x["h_gpr_visual"]) for x in mid])) if mid else float("nan"),
                "negative_frequency_uniform_mean_abs_response": float(np.mean([abs(x["h_uniform"]) for x in negative])) if negative else float("nan"),
                "negative_frequency_terminal_mean_abs_response": float(np.mean([abs(x["h_terminal"]) for x in negative])) if negative else float("nan"),
                "negative_frequency_gpr_text_mean_abs_response": float(np.mean([abs(x["h_gpr_text"]) for x in negative])) if negative else float("nan"),
                "negative_frequency_gpr_visual_mean_abs_response": float(np.mean([abs(x["h_gpr_visual"]) for x in negative])) if negative else float("nan"),
                "gpr_text_vs_uniform_response_rmse": float(np.mean([(x["h_gpr_text"] - x["h_uniform"]) ** 2 for x in seed_responses]) ** .5) if seed_responses else float("nan"),
                "gpr_visual_vs_uniform_response_rmse": float(np.mean([(x["h_gpr_visual"] - x["h_uniform"]) ** 2 for x in seed_responses]) ** .5) if seed_responses else float("nan"),
                "mid_frequency_eigenvalue_count": len(mid), "negative_eigenvalue_count": len(negative),
            })
    summary_lines = ["# Sparse spectral audit", "", "Uses the exact model normalized operator and `scipy.sparse.linalg.eigsh`; no dense eigendecomposition or graph changes were used.", ""]
    for row in summaries:
        summary_lines.extend([
            f"## {row['dataset']}", "",
            f"- Nodes: {row['num_nodes']}; undirected edge pairs: {row['num_undirected_edges']}.",
            f"- Largest 32 convergence: {row['largest_converged']} ({row['largest_count']} values). Failure: {row['largest_failure'] or 'none'}.",
            f"- Smallest 32 convergence: {row['smallest_converged']} ({row['smallest_count']} values). Failure: {row['smallest_failure'] or 'none'}.",
        ])
        if row["eigenvalues"]:
            eig = np.asarray(row["eigenvalues"])
            summary_lines.append(f"- Observed spectral span: [{eig.min():.6f}, {eig.max():.6f}].")
        summary_lines.append("")
    summary_lines.extend(["## Response summary by dataset and GPR seed", "",
                          "Mid-frequency retention is mean absolute response over observed eigenvalues with 0.25 <= |lambda| <= 0.75. Negative-frequency attenuation is reported as mean absolute response over observed negative eigenvalues. GPR distance is RMSE to Uniform over the audited eigenvalues.", "",
                          "| Dataset | Seed | Uniform mid | Terminal mid | GPR text mid | GPR visual mid | GPR text vs Uniform RMSE | GPR visual vs Uniform RMSE |",
                          "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for item in response_summaries:
        summary_lines.append(f"| {item['dataset']} | {item['seed']} | {item['mid_frequency_retention_uniform_mean_abs']:.5f} | {item['mid_frequency_retention_terminal_mean_abs']:.5f} | {item['mid_frequency_retention_gpr_text_mean_abs']:.5f} | {item['mid_frequency_retention_gpr_visual_mean_abs']:.5f} | {item['gpr_text_vs_uniform_response_rmse']:.5f} | {item['gpr_visual_vs_uniform_response_rmse']:.5f} |")
    return {"response_rows": rows, "summary": summaries, "response_summary": response_summaries, "markdown": "\n".join(summary_lines)}


def _stability_for_checkpoint(dataset: str, seed: int, readout: str,
                              train: torch.Tensor, val: torch.Tensor,
                              classes: list[int]) -> list[dict[str, Any]]:
    cfg, data, payload, model, head, train, val, classes = _load_model_data(dataset, seed, readout)
    base = _forward_context(model, head, data)
    base_metrics = _val_metrics(base["logits"], data, val, classes)
    base_pred = base["logits"][val].argmax(-1)
    n_pairs = int(canonical_undirected_pairs(data.edge_index).size(0))
    rows: list[dict[str, Any]] = []
    def record(condition: str, level: float, modality: str, logits_list: list[torch.Tensor]):
        stack = torch.stack([v[val] for v in logits_list])
        metrics = [classification_metrics(v, data.y[val], classes) for v in stack]
        flips = [float((v.argmax(-1) != base_pred).float().mean()) for v in stack]
        rows.append({"dataset": dataset, "seed": seed, "readout": readout,
                     "perturbation": condition, "level": level, "modality": modality,
                     "perturbation_seeds": 10,
                     "val_acc_drop_mean": base_metrics["accuracy"] - float(np.mean([m["accuracy"] for m in metrics])),
                     "val_macro_f1_drop_mean": base_metrics["macro_f1"] - float(np.mean([m["macro_f1"] for m in metrics])),
                     "prediction_flip_rate_mean": float(np.mean(flips)),
                     "mean_logit_variance": float(stack.var(dim=0, unbiased=False).mean().item())})
    for rate in (0.05, 0.10, 0.20):
        outputs = []
        for replicate in range(10):
            _, chosen = remove_random_same_size(data.edge_index, round(rate * n_pairs),
                                                seed=seed * 1_000_003 + replicate + int(rate * 1000))
            altered = remove_undirected_pairs(data.edge_index, chosen)
            ctx = _forward_context(model, head, data, edge_index=altered)
            outputs.append(ctx["logits"])
        record("edge_dropout", rate, "both", outputs)
    x = data.x.to(DEVICE)
    for modality, begin, end in (("text", 0, model.text_dim),
                                 ("visual", model.text_dim, model.text_dim + model.visual_dim)):
        scale = x[:, begin:end].std(dim=0, unbiased=False).clamp_min(1e-12)
        for noise_scale in (0.05, 0.10, 0.20):
            outputs = []
            for replicate in range(10):
                generator = torch.Generator(device=DEVICE).manual_seed(seed * 1_000_003 + replicate + begin * 17 + int(noise_scale * 1000))
                noise = torch.randn(x[:, begin:end].shape, generator=generator, device=DEVICE) * scale * noise_scale
                perturbed = x.clone()
                perturbed[:, begin:end] += noise
                ctx = _forward_context(model, head, data, x=perturbed)
                outputs.append(ctx["logits"])
            record("feature_noise", noise_scale, modality, outputs)
    del cfg, data, payload, model, head, base
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return rows


def _metric_record(path: Path, dataset: str, seed: int, variant: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metrics = payload["metrics"]
    return {"dataset": dataset, "seed": seed, "variant": variant,
            "best_epoch": int(payload["epoch"]), "val_accuracy": float(metrics["val_acc"]),
            "val_macro_f1": float(metrics["val_macro_f1"]),
            "test_accuracy_descriptive": float(metrics["test_acc"]),
            "test_macro_f1_descriptive": float(metrics["test_macro_f1"])}


def _order_source_tables() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    table: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    variant_names = ["terminal", "self_only", "self25_terminal75", "self50_terminal50",
                     "self75_terminal25", "uniform", "propagated_uniform"]
    for dataset in DATASETS:
        by_seed: dict[int, dict[str, dict[str, Any]]] = {}
        for seed in SEEDS:
            by_seed[seed] = {}
            for variant in variant_names:
                path = _checkpoint_path(dataset, seed, variant)
                row = _metric_record(path, dataset, seed, variant)
                by_seed[seed][variant] = row
                table.append(row)
            for variant in variant_names:
                values = [by_seed[seed][variant] for seed in SEEDS]
                table.append({
                    "dataset": dataset, "seed": "mean±population_std", "variant": variant,
                    "val_accuracy": float(np.mean([r["val_accuracy"] for r in values])),
                    "val_accuracy_std": float(np.std([r["val_accuracy"] for r in values], ddof=0)),
                    "val_macro_f1": float(np.mean([r["val_macro_f1"] for r in values])),
                    "val_macro_f1_std": float(np.std([r["val_macro_f1"] for r in values], ddof=0)),
                    "test_accuracy_descriptive": float(np.mean([r["test_accuracy_descriptive"] for r in values])),
                    "test_accuracy_descriptive_std": float(np.std([r["test_accuracy_descriptive"] for r in values], ddof=0)),
                    "test_macro_f1_descriptive": float(np.mean([r["test_macro_f1_descriptive"] for r in values])),
                    "test_macro_f1_descriptive_std": float(np.std([r["test_macro_f1_descriptive"] for r in values], ddof=0)),
                    "aggregation": "mean ± population std (ddof=0)",
                })
        contrasts = [
            ("G_self_25", "self25_terminal75", "terminal"),
            ("G_mid", "uniform", "self25_terminal75"),
            ("G_prop", "propagated_uniform", "terminal"),
            ("self50_minus_terminal", "self50_terminal50", "terminal"),
            ("self75_minus_terminal", "self75_terminal25", "terminal"),
            ("self_only_minus_terminal", "self_only", "terminal"),
        ]
        for label, left, right in contrasts:
            for seed in SEEDS:
                a, b = by_seed[seed][left], by_seed[seed][right]
                paired.append({"dataset": dataset, "seed": seed, "contrast": label,
                               "left_variant": left, "right_variant": right,
                               "val_accuracy_difference": a["val_accuracy"] - b["val_accuracy"],
                               "val_macro_f1_difference": a["val_macro_f1"] - b["val_macro_f1"],
                               "test_accuracy_difference_descriptive": a["test_accuracy_descriptive"] - b["test_accuracy_descriptive"]})
        for alpha, variant in ((0.0, "terminal"), (0.25, "self25_terminal75"),
                               (0.5, "self50_terminal50"), (0.75, "self75_terminal25"),
                               (1.0, "self_only")):
            for seed in SEEDS:
                src = by_seed[seed][variant]
                table.append({"dataset": dataset, "seed": seed,
                              "variant": f"alpha_scan_{alpha:g}", "alpha_self": alpha,
                              "val_accuracy": src["val_accuracy"], "val_macro_f1": src["val_macro_f1"],
                              "test_accuracy_descriptive": src["test_accuracy_descriptive"],
                              "test_macro_f1_descriptive": src["test_macro_f1_descriptive"],
                              "source_variant": variant})
    return table, paired


def _order_source_summary(table: list[dict[str, Any]], paired: list[dict[str, Any]]) -> dict[str, Any]:
    names = sorted({row["contrast"] for row in paired})
    contrasts = {}
    for name in names:
        values = [row["val_accuracy_difference"] for row in paired if row["contrast"] == name]
        contrasts[name] = {"mean_pp": float(np.mean(values) * 100),
                           "population_std_pp": float(np.std(values, ddof=0) * 100),
                           "positive_paired_seed_count": int(sum(x > 0 for x in values)),
                           "paired_seed_count": len(values)}
    alpha_summary = {}
    for alpha in (0, .25, .5, .75, 1):
        records = [row for row in table if row.get("variant") == f"alpha_scan_{alpha:g}"]
        alpha_summary[str(alpha)] = {
            "val_accuracy_mean": float(np.mean([r["val_accuracy"] for r in records])),
            "val_macro_f1_mean": float(np.mean([r["val_macro_f1"] for r in records])),
        }
    return {"new_order_source_contrasts": contrasts,
            "alpha_scan_descriptive_only": alpha_summary,
            "preserved_prior_frozen_facts": {
                "uniform_minus_terminal_val_accuracy_mean_pp": 2.447,
                "uniform_minus_terminal_positive_pairs": "15/15 across 5/5 datasets",
                "gpr_minus_uniform_val_accuracy_mean_pp": 0.13,
                "source": "pre-existing results/nc_benchmark_v1; not recomputed or overwritten"},
            "note": "Test metrics are included descriptively only and excluded from all contrasts used for discovery."}


def _spearman_quantile_summary(x: torch.Tensor, y: torch.Tensor, bins: int = 4) -> tuple[float, list[dict[str, Any]]]:
    corr = safe_spearman(x, y)
    edges = torch.quantile(x[torch.isfinite(x)], torch.linspace(0, 1, bins + 1))
    rows: list[dict[str, Any]] = []
    for index in range(bins):
        mask = torch.isfinite(x) & torch.isfinite(y)
        if index == bins - 1:
            mask &= (x >= edges[index]) & (x <= edges[index + 1])
        else:
            mask &= (x >= edges[index]) & (x < edges[index + 1])
        if int(mask.sum()):
            rows.append({"bin": index, "n": int(mask.sum()), "mean_x": float(x[mask].mean()),
                         "mean_y": float(y[mask].mean())})
    return corr, rows


def _regression_row(dataset: str, seed: int, outcome: torch.Tensor,
                    a: torch.Tensor, c: torch.Tensor, m: torch.Tensor) -> dict[str, Any]:
    valid = torch.isfinite(outcome) & torch.isfinite(a) & torch.isfinite(c) & torch.isfinite(m)
    x = torch.stack([a[valid], c[valid], m[valid]], -1).numpy()
    y = outcome[valid].numpy()
    if len(y) < 10:
        return {"dataset": dataset, "seed": seed, "n": len(y), "status": "insufficient_rows"}
    scale = StandardScaler()
    z = scale.fit_transform(x)
    design = np.column_stack([np.ones(len(z)), z[:, 0], z[:, 1], z[:, 2],
                              z[:, 0]*z[:, 1], z[:, 0]*z[:, 2], z[:, 1]*z[:, 2],
                              z[:, 0]*z[:, 1]*z[:, 2]])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    fit = design @ coef
    ss_res = float(np.square(y - fit).sum())
    ss_tot = float(np.square(y - y.mean()).sum())
    return {"dataset": dataset, "seed": seed, "n": len(y), "status": "descriptive_only",
            **{name: float(value) for name, value in zip(
                ("beta0", "betaA", "betaC", "betaM", "betaAC", "betaAM", "betaCM", "betaACM"), coef, strict=True)},
            "r_squared_descriptive": 1 - ss_res / max(ss_tot, 1e-20),
            "p_values": "not computed; graph nodes are not independent"}


def _safe_percentile_bins(values: torch.Tensor, n_bins: int = 4) -> torch.Tensor:
    finite = torch.isfinite(values)
    out = torch.full(values.shape, -1, dtype=torch.long)
    if finite.any():
        qs = torch.quantile(values[finite], torch.arange(1, n_bins) / n_bins)
        out[finite] = torch.bucketize(values[finite].contiguous(), qs.contiguous(), right=False)
    return out


def _load_on_existing_data(dataset: str, seed: int, readout: str, mode: str, data):
    cfg = _compose(dataset, seed, readout, mode)
    path = _checkpoint_path(dataset, seed, readout)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    info = {"input_dim": data.input_dim, "num_nodes": data.num_nodes,
            "num_classes": data.num_classes,
            "text_dim": int(data.x_t.size(1)) if data.x_t is not None else 0,
            "visual_dim": int(data.x_i.size(1)) if data.x_i is not None else 0}
    model = build_model(cfg, info).to(DEVICE)
    model.load_state_dict(payload["model_state"], strict=True)
    head = nn.Linear(model.out_dim, int(data.num_classes)).to(DEVICE)
    head.load_state_dict(payload["head_state"], strict=True)
    model.eval(); head.eval()
    return cfg, payload, model, head


def _prediction_splits(logits: torch.Tensor, data, train: torch.Tensor, val: torch.Tensor):
    # Access only the supervised train and validation labels.
    y_train, y_val = data.y[train], data.y[val]
    train_logits = logits[train]
    val_logits = logits[val]
    return {
        "train_logits": train_logits,
        "val_logits": val_logits,
        "train_probs": train_logits.softmax(-1),
        "val_probs": val_logits.softmax(-1),
        "train_loss": node_cross_entropy(train_logits, y_train),
        "val_loss": node_cross_entropy(val_logits, y_val),
        "train_metrics": classification_metrics(train_logits, y_train, _known_labels(data)[2]),
        "val_metrics": classification_metrics(val_logits, y_val, _known_labels(data)[2]),
    }


@torch.no_grad()
def _run_validation_intervention(model, head, data, val, classes, edge_index):
    context = _forward_context(model, head, data, edge_index=edge_index)
    logits = context["logits"][val]
    metrics = classification_metrics(logits, data.y[val], classes)
    top2 = logits.topk(min(2, logits.size(-1)), dim=-1).values
    margin = top2[:, 0] - (top2[:, 1] if top2.size(1) > 1 else 0)
    return metrics, float(margin.mean()), logits


def _intervention_group(model, head, data, val, classes, group_pairs: torch.Tensor,
                        baseline_metrics: dict[str, float], baseline_margin: float,
                        seed: int, repeats: int = 10) -> dict[str, Any]:
    targeted_edges = remove_undirected_pairs(data.edge_index, group_pairs)
    target_metrics, target_margin, _ = _run_validation_intervention(
        model, head, data, val, classes, targeted_edges)
    random_metrics, random_margins = [], []
    for replicate in range(repeats):
        _, chosen = remove_random_same_size(
            data.edge_index, int(group_pairs.size(0)), seed + replicate * 7919)
        altered = remove_undirected_pairs(data.edge_index, chosen)
        metrics, margin, _ = _run_validation_intervention(model, head, data, val, classes, altered)
        random_metrics.append(metrics); random_margins.append(margin)
    rand_acc = float(np.mean([x["accuracy"] for x in random_metrics]))
    rand_f1 = float(np.mean([x["macro_f1"] for x in random_metrics]))
    rand_margin = float(np.mean(random_margins))
    return {
        "removed_undirected_pairs": int(group_pairs.size(0)),
        "target_val_accuracy_change": target_metrics["accuracy"] - baseline_metrics["accuracy"],
        "target_macro_f1_change": target_metrics["macro_f1"] - baseline_metrics["macro_f1"],
        "target_margin_change": target_margin - baseline_margin,
        "random_val_accuracy_change_mean": rand_acc - baseline_metrics["accuracy"],
        "random_macro_f1_change_mean": rand_f1 - baseline_metrics["macro_f1"],
        "random_margin_change_mean": rand_margin - baseline_margin,
        "targeted_effect_minus_random_mean": target_metrics["accuracy"] - rand_acc,
        "targeted_macro_f1_minus_random_mean": target_metrics["macro_f1"] - rand_f1,
        "targeted_margin_minus_random_mean": target_margin - rand_margin,
        "random_repeats": repeats,
        "intervention_type": "FROZEN_INTERVENTION",
    }


def _edge_roles(dataset: str, seed: int, edge_t: torch.Tensor, pct_t: torch.Tensor,
                edge_v: torch.Tensor, pct_v: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not torch.equal(edge_t, edge_v):
        raise ValueError("Text and visual compatibility edge sets differ")
    t_high, t_low = pct_t >= .75, pct_t <= .25
    v_high, v_low = pct_v >= .75, pct_v <= .25
    masks = {
        "shared_supportive": t_high & v_high,
        "text_specific": t_high & v_low,
        "visual_specific": t_low & v_high,
        "weak_conflicting": t_low & v_low,
    }
    # Sensitivity uses a non-overlapping median split; ties are assigned to high.
    th, vh = pct_t >= .5, pct_v >= .5
    masks_50 = {
        "shared_supportive": th & vh,
        "text_specific": th & ~vh,
        "visual_specific": ~th & vh,
        "weak_conflicting": ~th & ~vh,
    }
    rows = []
    total = int(edge_t.size(1))
    for role, mask in masks.items():
        rows.append({"dataset": dataset, "seed": seed, "role": role,
                     "threshold": "0.75/0.25", "edge_pairs": int(mask.sum()),
                     "sampled_edge_pairs": total,
                     "prevalence": float(mask.float().mean()) if total else float("nan")})
    for role, mask in masks_50.items():
        rows.append({"dataset": dataset, "seed": seed, "role": role,
                     "threshold": "0.5_sensitivity", "edge_pairs": int(mask.sum()),
                     "sampled_edge_pairs": total,
                     "prevalence": float(mask.float().mean()) if total else float("nan")})
    return {**masks, **{f"50_{k}": v for k, v in masks_50.items()}}, {
        "rows": rows,
        "avg_similarity_percentile": float(((pct_t + pct_v) / 2).mean()) if total else float("nan"),
    }


def _node_edge_summaries(edge_index: torch.Tensor, pct_t: torch.Tensor, pct_v: torch.Tensor,
                         roles: dict[str, torch.Tensor], num_nodes: int):
    edge = edge_index.long().cpu()
    t = torch.zeros(num_nodes); v = torch.zeros(num_nodes); cross = torch.zeros(num_nodes)
    counts = torch.zeros(num_nodes)
    role_counts = {name: torch.zeros(num_nodes) for name in
                   ("shared_supportive", "text_specific", "visual_specific", "weak_conflicting")}
    for col in range(edge.size(1)):
        a, b = map(int, edge[:, col])
        for node in (a, b):
            t[node] += pct_t[col]; v[node] += pct_v[col]
            cross[node] += abs(float(pct_t[col] - pct_v[col])); counts[node] += 1
            for role, mask in roles.items():
                if not role.startswith("50_") and bool(mask[col]):
                    role_counts[role][node] += 1
    valid = counts > 0
    t[valid] /= counts[valid]; v[valid] /= counts[valid]; cross[valid] /= counts[valid]
    for value in role_counts.values():
        value[valid] /= counts[valid]
        value[~valid] = float("nan")
    t[~valid] = float("nan"); v[~valid] = float("nan"); cross[~valid] = float("nan")
    return t, v, cross, role_counts


def _local_homophily(data, train: torch.Tensor, val: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = data.num_nodes
    known = torch.zeros(n, dtype=torch.bool)
    known[torch.cat([train, val])] = True
    labels = torch.full((n,), -1, dtype=torch.long)
    known_idx = torch.cat([train, val])
    labels[known_idx] = data.y[known_idx].long()
    edge = data.edge_index.detach().long().cpu()
    src, dst = edge
    keep = (src != dst) & known[src] & known[dst]
    src, dst = src[keep], dst[keep]
    totals = torch.zeros(n).index_add_(0, src, torch.ones(src.numel()))
    matches = torch.zeros(n).index_add_(0, src, (labels[src] == labels[dst]).float())
    result = torch.full((n,), float("nan"))
    enough = totals >= 2
    result[enough] = matches[enough] / totals[enough]
    return result[val], torch.bincount(edge[0], minlength=n).float()[val]


def _fit_probe(name: str, dataset: str, seed: int, features: torch.Tensor, labels: torch.Tensor,
               train: torch.Tensor, val: torch.Tensor, classes: list[int], rows: list[dict[str, Any]]):
    result = PROBE.fit(features, labels, train, val, classes, DEVICE)
    rows.append({"dataset": dataset, "seed": seed, "probe": name,
                 "val_accuracy": result["val_metrics"]["accuracy"],
                 "val_macro_f1": result["val_metrics"]["macro_f1"],
                 "train_accuracy": result["train_metrics"]["accuracy"],
                 "trainable_parameters": result["trainable_parameters"],
                 "protocol": PROBE.config.name, "config_sha256": result["config_sha256"]})
    return result["train_logits"], result["val_logits"]


def _ridge_predict(source: torch.Tensor, target: torch.Tensor, train: torch.Tensor,
                   alpha: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    x = source.float(); y = target.float()
    mu_x = x[train].mean(0, keepdim=True)
    mu_y = y[train].mean(0, keepdim=True)
    xc = x[train] - mu_x
    yc = y[train] - mu_y
    eye = torch.eye(x.size(1), dtype=torch.float32)
    weight = torch.linalg.solve(xc.T @ xc + alpha * eye, xc.T @ yc)
    pred = (x - mu_x) @ weight + mu_y
    return pred, y - pred


def _alignment(dataset: str, seed: int, states_t: list[torch.Tensor], states_v: list[torch.Tensor],
               train: torch.Tensor, val: torch.Tensor, retrieval_seed: int) -> tuple[list[dict[str, Any]], torch.Tensor]:
    rows: list[dict[str, Any]] = []
    alignment_distance = torch.zeros(val.numel())
    gen = torch.Generator().manual_seed(retrieval_seed)
    retrieval_ids = val[torch.randperm(val.numel(), generator=gen)[:min(5000, val.numel())]].sort().values
    sample_hash = hashlib.sha256(retrieval_ids.numpy().tobytes()).hexdigest()
    for order, (text, visual) in enumerate(zip(states_t, states_v, strict=True)):
        train_t, train_v = text[train].float(), visual[train].float()
        val_t, val_v = text[val].float(), visual[val].float()
        # CKA/CCA are reported on train and validation partitions separately.
        cka_train, cka_val = linear_cka(train_t, train_v), linear_cka(val_t, val_v)
        cca_train, cca_val = cca_spectrum_summary(train_t, train_v), cca_spectrum_summary(val_t, val_v)
        mu_t, mu_v = train_t.mean(0, keepdim=True), train_v.mean(0, keepdim=True)
        u, _, vh = torch.linalg.svd((train_t - mu_t).T @ (train_v - mu_v), full_matrices=False)
        rotation = u @ vh
        aligned_val = (val_t - mu_t) @ rotation + mu_v
        distance = (aligned_val - val_v).norm(dim=-1) / math.sqrt(text.size(1))
        if order == 3:
            alignment_distance = distance
        sample_pos = torch.searchsorted(val, retrieval_ids)
        retrieval_t = ((text[retrieval_ids] - mu_t) @ rotation + mu_v)
        retrieval_v = visual[retrieval_ids]
        similarity = torch.nn.functional.normalize(retrieval_t, dim=-1) @ torch.nn.functional.normalize(retrieval_v, dim=-1).T
        ranks = (similarity.argsort(dim=-1, descending=True) == torch.arange(similarity.size(0))[:, None]).float().argmax(-1) + 1
        rows.append({"dataset": dataset, "seed": seed, "order": order,
                     "linear_cka_train": cka_train, "linear_cka_validation": cka_val,
                     **{f"{k}_train": v for k, v in cca_train.items()},
                     **{f"{k}_validation": v for k, v in cca_val.items()},
                     "procrustes_mean_distance_validation": float(distance.mean()),
                     "procrustes_median_distance_validation": float(distance.median()),
                     "retrieval_R_at_1": float((ranks <= 1).float().mean()),
                     "retrieval_R_at_10": float((ranks <= 10).float().mean()),
                     "retrieval_n": int(retrieval_ids.numel()), "retrieval_seed": retrieval_seed,
                     "retrieval_node_ids_sha256": sample_hash,
                     "procrustes_fit_split": "train", "alignment_eval_split": "validation"})
    return rows, alignment_distance


def _validate_formal_outputs() -> dict[str, int]:
    expected = {"order_source": 25, "unimodal": 20}
    validated: dict[str, int] = {}
    for phase, expected_jobs in expected.items():
        root = DISCOVERY_ROOT / phase
        complete_files = list(root.glob("*/*/complete.json"))
        if len(complete_files) != expected_jobs:
            raise RuntimeError(f"{phase}: expected {expected_jobs} completed dataset/variant jobs, found {len(complete_files)}")
        for path in complete_files:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("protocol_version") != "unified_full_graph_nc_v1":
                raise ValueError(f"{path}: wrong protocol")
            if record.get("seed") != 42 or record.get("num_runs") != 3 or record.get("run_seeds") != list(SEEDS):
                raise ValueError(f"{path}: expected one seed=42,num_runs=3 job for 42/43/44")
            run_metrics = Path(record["results"])
            if not run_metrics.is_file():
                raise FileNotFoundError(f"{path}: missing run metrics {run_metrics}")
            metrics_payload = json.loads(run_metrics.read_text(encoding="utf-8"))
            if metrics_payload.get("run_seeds") != list(SEEDS) or len(metrics_payload.get("runs", [])) != 3:
                raise ValueError(f"{path}: invalid internal multi-run aggregation metadata")
            for index, seed in enumerate(SEEDS, start=1):
                checkpoint = root / record["dataset"] / record["variant"] / f"best_run{index}.pt"
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if payload.get("seed") != seed or payload.get("selection") != "best_val_accuracy":
                    raise ValueError(f"{checkpoint}: invalid seed or checkpoint selection")
                if payload.get("protocol_version") != "unified_full_graph_nc_v1":
                    raise ValueError(f"{checkpoint}: protocol mismatch")
        validated[phase] = len(complete_files)
    return validated


def _final_analysis() -> dict[str, Any]:
    validated_jobs = _validate_formal_outputs()
    e1_root = RESULTS_ROOT / "experiment1"
    e2_root = RESULTS_ROOT / "experiment2"
    order_table, order_paired = _order_source_tables()
    _write_csv(e1_root / "order_source_table.csv", order_table)
    _write_csv(e1_root / "order_source_paired.csv", order_paired)
    e1_summary = _order_source_summary(order_table, order_paired)
    e1_summary["existing_checkpoint_diagnostics"] = _summarize_existing_e1()
    _write_json(e1_root / "experiment1_summary.json", {
        **e1_summary, "probe_config": asdict(PROBE_CONFIG),
        "probe_config_sha256": PROBE_CONFIG.sha256(),
        "data_boundary": "Train+Validation only for discovery; test metrics descriptive only.",
    })
    _write_text(e1_root / "experiment1_report.md", _render_final_e1(e1_summary))

    result_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    node_manifest: list[dict[str, Any]] = []
    gamma_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            print(f"FINAL {dataset} seed={seed}", flush=True)
            cfg, data, up, uniform_model, uniform_head, train, val, classes = _load_model_data(dataset, seed, "uniform")
            known_y = torch.full_like(data.y, -1)
            known = torch.cat([train, val])
            known_y[known] = data.y[known]
            joint = _forward_context(uniform_model, uniform_head, data)
            model_outputs: dict[str, dict[str, Any]] = {}
            model_outputs["uniform"] = _prediction_splits(joint["logits"], data, train, val)
            model_outputs["uniform"]["val_probs"] = model_outputs["uniform"]["val_logits"].softmax(-1)
            model_outputs["uniform"]["train_probs"] = model_outputs["uniform"]["train_logits"].softmax(-1)
            for name, readout, mode in (
                ("self", "self_only", "both"), ("terminal", "terminal", "both"),
                ("gpr", "gpr", "both"), ("text_self", "self_only", "text"),
                ("text_uniform", "uniform", "text"), ("visual_self", "self_only", "visual"),
                ("visual_uniform", "uniform", "visual"),
            ):
                local_cfg, payload, model, head = _load_on_existing_data(dataset, seed, readout, mode, data)
                ctx = _forward_context(model, head, data)
                model_outputs[name] = _prediction_splits(ctx["logits"], data, train, val)
                if name == "gpr":
                    gamma_rows.append({"dataset": dataset, "seed": seed,
                                       "gamma_text": json.dumps(payload["model_state"]["gamma_text"].tolist()),
                                       "gamma_visual": json.dumps(payload["model_state"]["gamma_visual"].tolist()),
                                       "text_visual_l2_difference": float(torch.dist(payload["model_state"]["gamma_text"], payload["model_state"]["gamma_visual"]))})
                if name in {"text_uniform", "visual_uniform"}:
                    # Preserve no full graph states from independently trained unimodal controls.
                    pass
                del local_cfg, payload, ctx
                del model, head
            # Model checkpoint metric records are reported for sufficiency and validation regimes only.
            p_self, p_context, p_uniform = {}, {}, {}
            for probe_name, feature in (
                ("self", torch.cat([joint["S_text"][0], joint["S_visual"][0]], dim=-1)),
                ("context", torch.cat([torch.stack(joint["S_text"][1:]).mean(0),
                                       torch.stack(joint["S_visual"][1:]).mean(0)], dim=-1)),
                ("uniform", joint["fused_z"]),
            ):
                train_logits, val_logits = _fit_probe(probe_name, dataset, seed, feature, known_y,
                                                       train, val, classes, result_rows["structure_attribute_conflict"])
                p_train = train_logits.softmax(-1); p_val = val_logits.softmax(-1)
                if probe_name == "self": p_self = {"train": p_train, "val": p_val, "train_logits": train_logits, "val_logits": val_logits}
                elif probe_name == "context": p_context = {"train": p_train, "val": p_val, "train_logits": train_logits, "val_logits": val_logits}
                else: p_uniform = {"train": p_train, "val": p_val, "train_logits": train_logits, "val_logits": val_logits}
            d_sc = js_divergence(p_self["val"], p_context["val"])
            self_native = model_outputs["self"]
            uniform_native = model_outputs["uniform"]
            g_struct_model = self_native["val_loss"] - uniform_native["val_loss"]
            g_struct_probe = node_cross_entropy(p_self["val_logits"], data.y[val]) - node_cross_entropy(p_uniform["val_logits"], data.y[val])
            for outcome_name, outcome in (("model", g_struct_model), ("probe", g_struct_probe)):
                corr, bins = _spearman_quantile_summary(d_sc, outcome)
                result_rows["structure_attribute_conflict"].append({
                    "dataset": dataset, "seed": seed, "analysis": "D_SC_vs_G_struct",
                    "outcome": outcome_name, "spearman": corr,
                    "quantile_bins_json": json.dumps(bins), "n_validation": int(val.numel()),
                    "split": "validation", "probe_sha256": PROBE.config.sha256()})

            # Independent modality sufficiency and node-level correctness regimes.
            t = model_outputs["text_uniform"]; v = model_outputs["visual_uniform"]
            a_t, a_v = t["val_metrics"]["accuracy"], v["val_metrics"]["accuracy"]
            a_tv = uniform_native["val_metrics"]["accuracy"]
            result_rows["modality_sufficiency"].append({
                "dataset": dataset, "seed": seed,
                "A_T_text_uniform": a_t, "A_V_visual_uniform": a_v,
                "A_TV_multimodal_uniform": a_tv,
                "modality_dominance_abs_accuracy_gap": abs(a_t - a_v),
                "multimodal_complementarity_accuracy": a_tv - max(a_t, a_v),
                "A_T_text_self": model_outputs["text_self"]["val_metrics"]["accuracy"],
                "A_V_visual_self": model_outputs["visual_self"]["val_metrics"]["accuracy"],
                "A_TV_multimodal_self": self_native["val_metrics"]["accuracy"],
                "validation_only": True,
            })
            pt, pv, pm = t["val_probs"], v["val_probs"], uniform_native["val_probs"]
            yv = data.y[val]
            pred_t, pred_v = pt.argmax(-1), pv.argmax(-1)
            correct_t, correct_v = pred_t == yv, pred_v == yv
            regime_names = []
            for ct, cv in zip(correct_t.tolist(), correct_v.tolist(), strict=True):
                regime_names.append("TT" if ct and cv else "T-only" if ct else "V-only" if cv else "Neither")
            deg_val = torch.bincount(data.edge_index[0].cpu(), minlength=data.num_nodes)[val]
            for regime in ("TT", "T-only", "V-only", "Neither"):
                mask_list = [item == regime for item in regime_names]
                mask = torch.tensor(mask_list, dtype=torch.bool)
                if not mask.any():
                    continue
                classes_in = Counter(map(int, yv[mask].tolist()))
                result_rows["modality_regimes"].append({
                    "dataset": dataset, "seed": seed, "regime": regime,
                    "population": int(mask.sum()), "population_fraction": float(mask.float().mean()),
                    "multimodal_uniform_accuracy": float((pm[mask].argmax(-1) == yv[mask]).float().mean()),
                    "mean_multimodal_confidence": float(pm[mask].max(-1).values.mean()),
                    "class_distribution_json": json.dumps(dict(classes_in), sort_keys=True),
                    "mean_degree": float(deg_val[mask].float().mean()),
                })

            # Joint branch probes and independently trained modality checkpoints.
            branch_suppression: dict[str, torch.Tensor] = {}
            for modality in ("text", "visual"):
                tr_logits, va_logits = _fit_probe(f"joint_{modality}_branch", dataset, seed,
                                                  joint[f"Z_{modality}"], known_y,
                                                  train, val, classes,
                                                  result_rows["joint_branch_suppression"])
                independent = model_outputs[f"{modality}_uniform"]["val_metrics"]["accuracy"]
                branch_acc = classification_metrics(va_logits, data.y[val], classes)["accuracy"]
                branch_suppression[modality] = torch.tensor(independent - branch_acc)
                result_rows["joint_branch_suppression"].append({
                    "dataset": dataset, "seed": seed, "modality": modality,
                    "independent_model_val_accuracy": independent,
                    "joint_branch_probe_val_accuracy": branch_acc,
                    "branch_suppression_descriptive": independent - branch_acc,
                    "interpretation_boundary": "probe comparison; not causal branch intervention"})

            # Modality-specific topology utility from paired self/uniform models.
            loss_ts = model_outputs["text_self"]["val_loss"]
            loss_tu = model_outputs["text_uniform"]["val_loss"]
            loss_vs = model_outputs["visual_self"]["val_loss"]
            loss_vu = model_outputs["visual_uniform"]["val_loss"]
            g_t, g_v = loss_ts - loss_tu, loss_vs - loss_vu
            d_ms = g_t - g_v
            result_rows["modality_structure_utility"].append({
                "dataset": dataset, "seed": seed,
                "mean_G_T": float(g_t.mean()), "std_G_T": float(g_t.std(unbiased=False)),
                "mean_G_V": float(g_v.mean()), "std_G_V": float(g_v.std(unbiased=False)),
                "mean_D_MS": float(d_ms.mean()), "std_D_MS": float(d_ms.std(unbiased=False)),
                "D_MS_q10": float(torch.quantile(d_ms, .1)), "D_MS_q50": float(torch.quantile(d_ms, .5)),
                "D_MS_q90": float(torch.quantile(d_ms, .9)),
                "fraction_G_T_positive_G_V_negative": float(((g_t > 0) & (g_v < 0)).float().mean()),
                "fraction_G_T_negative_G_V_positive": float(((g_t < 0) & (g_v > 0)).float().mean()),
                "fraction_both_positive": float(((g_t > 0) & (g_v > 0)).float().mean()),
                "fraction_both_negative": float(((g_t < 0) & (g_v < 0)).float().mean()),
                "validation_nodes": int(val.numel()),
            })

            # Source-matched edge compatibility, unique edge roles, and frozen interventions.
            edge_t, pct_t, sample_t = edge_percentile_compatibility(
                joint["H0_text"], data.edge_index, seed * 101 + 7)
            edge_v, pct_v, sample_v = edge_percentile_compatibility(
                joint["H0_visual"], data.edge_index, seed * 101 + 7)
            role_masks, role_summary = _edge_roles(dataset, seed, edge_t, pct_t, edge_v, pct_v)
            result_rows["edge_interaction_prevalence"].extend([
                {**row, "sampled_source_nodes": sample_t["sampled_source_nodes"],
                 "nonedge_reference_scores_text": sample_t["nonedge_reference_scores"],
                 "nonedge_reference_scores_visual": sample_v["nonedge_reference_scores"],
                 "compatibility_method": "source_matched_nonedge_percentile"}
                for row in role_summary["rows"]])
            node_ct, node_cv, node_edge_dis, node_role = _node_edge_summaries(
                edge_t, pct_t, pct_v, role_masks, data.num_nodes)
            compatibility = (node_ct[val] + node_cv[val]) / 2
            edge_disagreement = node_edge_dis[val]
            baseline_metrics = uniform_native["val_metrics"]
            base_logits = uniform_native["val_logits"]
            base_margin = float((base_logits.topk(2, dim=-1).values[:, 0] - base_logits.topk(2, dim=-1).values[:, 1]).mean())
            for role_idx, role in enumerate(("shared_supportive", "text_specific", "visual_specific", "weak_conflicting")):
                pairs = edge_t[:, role_masks[role]].T.contiguous()
                outcome = _intervention_group(uniform_model, uniform_head, data, val, classes, pairs,
                                              baseline_metrics, base_margin,
                                              seed * 1000003 + role_idx * 10007)
                result_rows["edge_role_intervention"].append({
                    "dataset": dataset, "seed": seed, "edge_role": role,
                    "threshold": "high>=.75_low<=.25", **outcome})
            semantic_score = (pct_t + pct_v) / 2
            semantic_bin = _safe_percentile_bins(semantic_score, 4)
            for bin_id in range(4):
                mask = semantic_bin == bin_id
                pairs = edge_t[:, mask].T.contiguous()
                outcome = _intervention_group(uniform_model, uniform_head, data, val, classes, pairs,
                                              baseline_metrics, base_margin,
                                              seed * 2000003 + bin_id * 30011)
                result_rows["semantic_utility_relation"].append({
                    "dataset": dataset, "seed": seed, "semantic_percentile_quantile": bin_id,
                    "mean_text_compatibility_percentile": float(pct_t[mask].mean()) if mask.any() else float("nan"),
                    "mean_visual_compatibility_percentile": float(pct_v[mask].mean()) if mask.any() else float("nan"),
                    "mean_joint_compatibility_percentile": float(semantic_score[mask].mean()) if mask.any() else float("nan"),
                    **outcome})

            # Cross-modal disagreement and interaction probe with a fixed, label-independent projection.
            dtv = js_divergence(pt, pv)
            multimodal_gain = torch.maximum(t["val_loss"], v["val_loss"]) - uniform_native["val_loss"]
            loss_multimodal = uniform_native["val_loss"]
            val_ce = node_cross_entropy(uniform_native["val_logits"], yv)
            corr_fields = {
                "multimodal_gain": multimodal_gain,
                "classification_loss": val_ce,
                "structure_benefit": g_struct_model,
                "G_T_minus_G_V": d_ms,
            }
            dtv_bins = quantile_bins(dtv, 4)
            for name, value in corr_fields.items():
                result_rows["crossmodal_disagreement"].append({
                    "dataset": dataset, "seed": seed, "analysis": "D_TV_association",
                    "outcome": name, "spearman": safe_spearman(dtv, value),
                    "D_TV_mean": float(dtv.mean()), "D_TV_q25": float(torch.quantile(dtv, .25)),
                    "D_TV_q50": float(torch.quantile(dtv, .5)), "D_TV_q75": float(torch.quantile(dtv, .75)),
                    "validation_nodes": int(val.numel()),
                })
            for class_id in classes:
                mask = yv == class_id
                if mask.any():
                    result_rows["crossmodal_disagreement"].append({
                        "dataset": dataset, "seed": seed, "analysis": "per_class_D_TV",
                        "class_id": class_id, "population": int(mask.sum()),
                        "D_TV_mean": float(dtv[mask].mean()),
                    })
            t_branch, v_branch = joint["Z_text"], joint["Z_visual"]
            plain_features = torch.cat([t_branch, v_branch], -1)
            full_interaction = torch.cat([t_branch, v_branch, t_branch * v_branch,
                                          (t_branch - v_branch).abs()], -1)
            torch.manual_seed(0)
            projection = torch.randn(full_interaction.size(1), plain_features.size(1)) / math.sqrt(plain_features.size(1))
            interaction_features = full_interaction @ projection
            plain_train, plain_val = _fit_probe("plain_concat", dataset, seed, plain_features,
                                                 known_y, train, val, classes,
                                                 result_rows["crossmodal_interaction_probe"])
            int_train, int_val = _fit_probe("interaction_fixed_projection", dataset, seed,
                                             interaction_features, known_y, train, val, classes,
                                             result_rows["crossmodal_interaction_probe"])
            plain_acc = classification_metrics(plain_val, yv, classes)["accuracy"]
            int_acc = classification_metrics(int_val, yv, classes)["accuracy"]
            plain_params = int(plain_features.size(1) * (max(classes) + 1) + (max(classes) + 1))
            int_params = int(interaction_features.size(1) * (max(classes) + 1) + (max(classes) + 1))
            capacity_flag = "CAPACITY_CONFOUNDED_DIAGNOSTIC" if abs(int_params - plain_params) / max(plain_params, 1) > .1 else "PARAMETER_MATCHED_LINEAR_DIAGNOSTIC"
            int_gain = (int_val.argmax(-1) == yv).float() - (plain_val.argmax(-1) == yv).float()
            for bin_id in range(4):
                mask = dtv_bins == bin_id
                if mask.any():
                    result_rows["crossmodal_interaction_probe"].append({
                        "dataset": dataset, "seed": seed, "analysis": "interaction_gain_by_D_TV_quantile",
                        "D_TV_quantile": bin_id, "population": int(mask.sum()),
                        "plain_val_accuracy": float((plain_val[mask].argmax(-1) == yv[mask]).float().mean()),
                        "interaction_val_accuracy": float((int_val[mask].argmax(-1) == yv[mask]).float().mean()),
                        "accuracy_gain": float(int_gain[mask].mean()), "capacity_status": capacity_flag,
                        "trainable_params_plain": plain_params, "trainable_params_interaction": int_params,
                        "feature_definition": "[T,V,T*V,abs(T-V)] then fixed seeded label-independent Gaussian projection"})
            result_rows["crossmodal_interaction_probe"].append({
                "dataset": dataset, "seed": seed, "analysis": "overall_interaction_gain",
                "plain_val_accuracy": plain_acc, "interaction_val_accuracy": int_acc,
                "accuracy_gain": int_acc - plain_acc,
                "capacity_status": capacity_flag, "trainable_params_plain": plain_params,
                "trainable_params_interaction": int_params,
                "feature_definition": "[T,V,T*V,abs(T-V)] then fixed seeded label-independent Gaussian projection"})

            # Alignment trajectories and task relevance use only train-fitted transforms and validation outcomes.
            alignment_rows, alignment_distance = _alignment(
                dataset, seed, joint["S_text"], joint["S_visual"], train, val,
                retrieval_seed=seed * 101 + 19)
            result_rows["crossmodal_alignment_trajectory"].extend(alignment_rows)
            for outcome_name, outcome in (("classification_loss", val_ce),
                                          ("multimodal_gain", multimodal_gain),
                                          ("structure_benefit", g_struct_model),
                                          ("D_TV", dtv)):
                result_rows["crossmodal_alignment_trajectory"].append({
                    "dataset": dataset, "seed": seed, "order": 3,
                    "analysis": "aligned_distance_task_relevance",
                    "outcome": outcome_name,
                    "spearman_aligned_distance": safe_spearman(alignment_distance, outcome),
                    "validation_nodes": int(val.numel()),
                    "status": "TASK_RELEVANT_DIVERGENCE" if abs(safe_spearman(alignment_distance, outcome)) >= .2 else "DESCRIPTIVE_ONLY",
                })

            # Linear predictable/residual proxy: ridge fit uses train nodes only.
            zt, zv = joint["Z_text"], joint["Z_visual"]
            pred_v, resid_v = _ridge_predict(zt, zv, train)
            pred_t, resid_t = _ridge_predict(zv, zt, train)
            proxy_features = {
                "visual_predictable_from_text": pred_v,
                "text_predictable_from_visual": pred_t,
                "text_linear_residual": resid_t,
                "visual_linear_residual": resid_v,
                "combined_predictable": torch.cat([pred_t, pred_v], -1),
                "combined_residual": torch.cat([resid_t, resid_v], -1),
                "predictable_plus_residual": torch.cat([pred_t, resid_t, pred_v, resid_v], -1),
            }
            for name, feature in proxy_features.items():
                _fit_probe(name, dataset, seed, feature, known_y, train, val, classes,
                           result_rows["shared_private_proxy"])

            # 2x2x2 train-median cube. Validation values never set thresholds.
            self_conf_train = p_self["train"].max(-1).values
            self_conf_val = p_self["val"].max(-1).values
            agreement_train = 1 - js_divergence(t["train_probs"], v["train_probs"])
            agreement_val = 1 - dtv
            compat_all = (node_ct + node_cv) / 2
            thresholds = {
                "intrinsic_sufficiency_median_train": train_median_threshold(self_conf_train),
                "structural_compatibility_median_train": train_median_threshold(compat_all[train]),
                "modal_agreement_median_train": train_median_threshold(agreement_train),
            }
            A_hi = self_conf_val >= thresholds["intrinsic_sufficiency_median_train"]
            C_hi = compatibility >= thresholds["structural_compatibility_median_train"]
            M_hi = agreement_val >= thresholds["modal_agreement_median_train"]
            C_valid = torch.isfinite(compatibility)
            for a_bit in (0, 1):
                for c_bit in (0, 1):
                    for m_bit in (0, 1):
                        mask = C_valid & (A_hi == bool(a_bit)) & (C_hi == bool(c_bit)) & (M_hi == bool(m_bit))
                        def group_accuracy(name: str) -> float:
                            logits = model_outputs[name]["val_logits"][mask]
                            return float((logits.argmax(-1) == yv[mask]).float().mean()) if mask.any() else float("nan")
                        result_rows["interaction_cube"].append({
                            "dataset": dataset, "seed": seed,
                            "intrinsic_sufficiency_high": a_bit,
                            "structural_compatibility_high": c_bit,
                            "modal_agreement_high": m_bit,
                            "population": int(mask.sum()),
                            "self_accuracy": group_accuracy("self"),
                            "uniform_accuracy": group_accuracy("uniform"),
                            "gpr_accuracy": group_accuracy("gpr"),
                            "text_accuracy": group_accuracy("text_uniform"),
                            "visual_accuracy": group_accuracy("visual_uniform"),
                            "mean_graph_benefit": float(g_struct_model[mask].mean()) if mask.any() else float("nan"),
                            "mean_multimodal_gain": float(multimodal_gain[mask].mean()) if mask.any() else float("nan"),
                            "mean_modality_specific_graph_utility_gap": float(d_ms[mask].mean()) if mask.any() else float("nan"),
                            "mean_loss": float(loss_multimodal[mask].mean()) if mask.any() else float("nan"),
                            **thresholds,
                            "threshold_source": "train_nodes_only",
                        })
            result_rows["interaction_regression"].append(_regression_row(
                dataset, seed, g_struct_model, self_conf_val, compatibility, dtv))

            # Validation class metrics, with class frequency defined only on training labels.
            train_freq = Counter(map(int, data.y[train].tolist()))
            classwise_by_variant = {}
            val_logits_by_variant = {name: model_outputs[name]["val_logits"]
                                     for name in ("self", "terminal", "uniform", "gpr", "text_uniform", "visual_uniform")}
            for name, logits in val_logits_by_variant.items():
                classwise_by_variant[name] = classwise_metrics(logits, yv, classes)
            by_variant_class = {
                name: {int(row["class_id"]): row for row in values}
                for name, values in classwise_by_variant.items()
            }
            for class_id in classes:
                self_acc = by_variant_class["self"][class_id]["accuracy"]
                terminal_acc = by_variant_class["terminal"][class_id]["accuracy"]
                uni_acc = by_variant_class["uniform"][class_id]["accuracy"]
                best_uni = max(by_variant_class["text_uniform"][class_id]["accuracy"],
                               by_variant_class["visual_uniform"][class_id]["accuracy"])
                for variant in ("self", "terminal", "uniform", "gpr", "text_uniform", "visual_uniform"):
                    item = by_variant_class[variant][class_id]
                    result_rows["classwise_gain"].append({
                        "dataset": dataset, "seed": seed, "variant": variant,
                        "class_id": class_id, "support_validation": item["support"],
                        "class_frequency_train": train_freq.get(class_id, 0),
                        "accuracy": item["accuracy"], "precision": item["precision"],
                        "recall": item["recall"], "f1": item["f1"],
                        "self_to_uniform_accuracy_gain": uni_acc - self_acc,
                        "terminal_to_uniform_accuracy_gain": uni_acc - terminal_acc,
                        "multimodal_to_best_unimodal_accuracy_gain": uni_acc - best_uni,
                        "validation_only": True,
                    })

            # Node table: exactly one row per validation node, with labels masked outside train+validation.
            local_homophily, degree = _local_homophily(data, train, val)
            probs_by_name = {name: value["val_probs"] for name, value in model_outputs.items()}
            losses_by_name = {name: value["val_loss"] for name, value in model_outputs.items()}
            correct_by_name = {name: value["val_logits"].argmax(-1) == yv for name, value in model_outputs.items()}
            class_freq_val_node = [train_freq.get(int(label), 0) for label in yv.tolist()]
            node_rows = []
            for i, node_id in enumerate(val.tolist()):
                node = {
                    "dataset": dataset, "seed": seed, "node_id": int(node_id),
                    "label": int(yv[i]), "class_frequency": int(class_freq_val_node[i]),
                    "degree": float(degree[i]), "local_homophily": float(local_homophily[i]),
                    "self_loss": float(losses_by_name["self"][i]),
                    "self_confidence": float(probs_by_name["self"][i].max()),
                    "uniform_loss": float(losses_by_name["uniform"][i]),
                    "uniform_confidence": float(probs_by_name["uniform"][i].max()),
                    "graph_benefit_model": float(g_struct_model[i]),
                    "graph_benefit_probe": float(g_struct_probe[i]),
                    "text_loss": float(losses_by_name["text_uniform"][i]),
                    "text_confidence": float(probs_by_name["text_uniform"][i].max()),
                    "visual_loss": float(losses_by_name["visual_uniform"][i]),
                    "visual_confidence": float(probs_by_name["visual_uniform"][i].max()),
                    "multimodal_loss": float(losses_by_name["uniform"][i]),
                    "text_graph_benefit": float(g_t[i]),
                    "visual_graph_benefit": float(g_v[i]),
                    "tv_js": float(dtv[i]), "structure_self_js": float(d_sc[i]),
                    "text_edge_compatibility": float(node_ct[node_id]),
                    "visual_edge_compatibility": float(node_cv[node_id]),
                    "edge_crossmodal_disagreement": float(node_edge_dis[node_id]),
                    "alignment_distance": float(alignment_distance[i]),
                    "terminal_correct": bool(correct_by_name["terminal"][i]),
                    "uniform_correct": bool(correct_by_name["uniform"][i]),
                    "gpr_correct": bool(correct_by_name["gpr"][i]),
                    "multimodal_correct": bool(correct_by_name["uniform"][i]),
                    "multimodal_gain": float(multimodal_gain[i]),
                }
                for role, values in node_role.items():
                    node[f"{role}_ratio"] = float(values[node_id])
                node_rows.append(node)
            node_dir = DISCOVERY_ROOT / "node_tables"
            node_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{dataset}_seed{seed}"
            plain_path = node_dir / f"{stem}.csv"
            _write_csv(plain_path, node_rows)
            final_path = plain_path
            if plain_path.stat().st_size > 20 * 1024 * 1024:
                import gzip
                gz_path = node_dir / f"{stem}.csv.gz"
                with plain_path.open("rb") as source, gz_path.open("wb") as raw:
                    import gzip
                    with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, mtime=0) as target:
                        while chunk := source.read(1024 * 1024):
                            target.write(chunk)
                plain_path.unlink()
                final_path = gz_path
            node_manifest.append({"dataset": dataset, "seed": seed,
                                  "path": str(final_path.relative_to(ROOT)),
                                  "rows": len(node_rows), "sha256": _sha256(final_path),
                                  "compressed": final_path.suffix == ".gz",
                                  "size_bytes": final_path.stat().st_size})

            # Collect trained GPR coefficients without implying a research mechanism.
            # (The coefficient rows are saved after all dataset/seed runs.)
            del cfg, data, up, uniform_model, uniform_head, joint
            del model_outputs
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    # Persist every requested analysis table in the separate discovery result tree.
    e2_files = {
        "modality_sufficiency": "modality_sufficiency.csv",
        "modality_regimes": "modality_regimes.csv",
        "joint_branch_suppression": "joint_branch_suppression.csv",
        "structure_attribute_conflict": "structure_attribute_conflict.csv",
        "modality_structure_utility": "modality_structure_utility.csv",
        "edge_interaction_prevalence": "edge_interaction_prevalence.csv",
        "edge_role_intervention": "edge_role_intervention.csv",
        "semantic_utility_relation": "semantic_utility_relation.csv",
        "crossmodal_disagreement": "crossmodal_disagreement.csv",
        "crossmodal_interaction_probe": "crossmodal_interaction_probe.csv",
        "crossmodal_alignment_trajectory": "crossmodal_alignment_trajectory.csv",
        "shared_private_proxy": "shared_private_proxy.csv",
        "interaction_cube": "interaction_cube.csv",
        "interaction_regression": "interaction_regression.csv",
        "classwise_gain": "classwise_gain.csv",
    }
    for key, filename in e2_files.items():
        _write_csv(e2_root / filename, result_rows[key])
    _write_csv(RESULTS_ROOT / "gpr_gamma.csv", gamma_rows)
    _write_csv(DISCOVERY_ROOT / "node_tables_manifest.csv", node_manifest)
    matrix = _problem_discovery_matrix(result_rows)
    _write_csv(RESULTS_ROOT / "problem_discovery_matrix.csv", matrix)
    summary = {
        "phase": "D_final_dependent_analyses",
        "datasets": list(DATASETS), "seeds": list(SEEDS),
        "formal_new_training_runs_expected": 135,
        "order_source_runs_expected": 75, "unimodal_runs_expected": 60,
        "formal_jobs_validated": validated_jobs,
        "formal_runs_validated": sum(validated_jobs.values()) * 3,
        "test_usage": "No test labels used for discovery. Test metrics on frozen formal controls are descriptive only.",
        "probe_config": asdict(PROBE_CONFIG), "probe_config_sha256": PROBE_CONFIG.sha256(),
        "node_table_files": len(node_manifest),
        "node_table_rows": int(sum(item["rows"] for item in node_manifest)),
        "problem_status_counts": dict(Counter(row["status"] for row in matrix)),
        "report_boundaries": [
            "DESCRIPTIVE_ONLY, TASK_RELEVANT, and INTERVENTION_SUPPORTED are evidence levels, not paper claims.",
            "No node-independence p-values are reported.",
            "No topology rewiring training was run.",
            "The original results/nc_benchmark_v1/ and outputs/nc_benchmark_v1/ were read-only inputs.",
        ],
    }
    _write_json(e2_root / "experiment2_summary.json", summary)
    _write_json(RESULTS_ROOT / "experiment2_summary.json", summary)
    _write_text(e2_root / "experiment2_report.md", _render_e2_report(result_rows, matrix, summary))
    _write_text(e1_root / "experiment1_report.md", _render_final_e1(e1_summary))
    _write_json(e1_root / "experiment1_summary.json", {
        **e1_summary, "probe_config": asdict(PROBE_CONFIG),
        "probe_config_sha256": PROBE_CONFIG.sha256(),
        "representation_geometry_source": "uniform_plain best validation checkpoints; gpr_plain robustness only",
        "test_usage": "descriptive only",
    })
    print("FINAL_ANALYSIS_COMPLETE", flush=True)
    return summary



def _summarize_existing_e1() -> dict[str, Any]:
    root = RESULTS_ROOT / "experiment1"
    def read(name: str) -> list[dict[str, str]]:
        path = root / name
        return list(csv.DictReader(path.open(encoding="utf-8"))) if path.is_file() else []
    hop, incremental = read("hop_task_probe.csv"), read("incremental_probe.csv")
    innovation = read("innovation_probe.csv")
    smoothing, stability = read("representation_smoothing.csv"), read("stability_audit.csv")
    hop_summary = {}
    for mode in ("text", "visual", "concat"):
        hop_summary[mode] = {}
        for order in range(4):
            vals = [float(row["val_accuracy"]) for row in hop if row["modality_mode"] == mode and int(row["order"]) == order]
            f1s = [float(row["val_macro_f1"]) for row in hop if row["modality_mode"] == mode and int(row["order"]) == order]
            if vals:
                hop_summary[mode][str(order)] = {"val_accuracy_mean": float(np.mean(vals)), "val_macro_f1_mean": float(np.mean(f1s)), "n": len(vals)}
    base = {(r["dataset"], int(r["encoder_seed"]), r["modality_mode"]): float(r["val_accuracy"])
            for r in hop if r["order"] == "0"}
    incremental_gain = {}
    for order in (1, 2, 3):
        gains = []
        for row in incremental:
            if row["order"] != f"S0_plus_Delta1_to_{order}":
                continue
            key = (row["dataset"], int(row["encoder_seed"]), row["modality_mode"])
            if key in base:
                gains.append(float(row["val_accuracy"]) - base[key])
        incremental_gain[str(order)] = {"mean_accuracy_gain_over_S0": float(np.mean(gains)) if gains else float("nan"),
                                        "positive_comparisons": int(sum(x > 0 for x in gains)),
                                        "comparisons": len(gains)}
    innovation_summary = {}
    for mode in ("text", "visual", "concat"):
        innovation_summary[mode] = {}
        for order in ("1", "2", "3"):
            vals = [float(r["val_accuracy"]) for r in innovation if r["modality_mode"] == mode and r["order"] == order]
            if vals:
                innovation_summary[mode][order] = {"val_accuracy_mean": float(np.mean(vals)), "n": len(vals)}
    smooth_count = sum(r.get("joint_smoothing_collapse_evidence", "").lower() == "true" for r in smoothing)
    stability_summary: dict[str, Any] = {}
    for readout in ("terminal", "uniform", "gpr"):
        selected = [r for r in stability if r["readout"] == readout]
        stability_summary[readout] = {}
        for kind in ("edge_dropout", "feature_noise"):
            vals = [float(r["val_acc_drop_mean"]) for r in selected if r["perturbation"] == kind]
            if vals:
                stability_summary[readout][kind] = {"mean_validation_accuracy_drop": float(np.mean(vals)), "rows": len(vals)}
    spectral = read("spectral_response.csv")
    stage_a_summary_path = root / "experiment1_summary.json"
    stage_a_summary = json.loads(stage_a_summary_path.read_text(encoding="utf-8")) if stage_a_summary_path.is_file() else {}
    return {
        "checkpoint_diagnostic_rows": {"hop_task_probe": len(hop), "incremental_probe": len(incremental),
            "innovation_probe": len(innovation), "representation_smoothing": len(smoothing),
            "stability_audit": len(stability), "spectral_response": len(spectral)},
        "hop_probe_mean_by_order": hop_summary,
        "incremental_probe_gain_over_S0": incremental_gain,
        "innovation_probe_mean_by_order": innovation_summary,
        "four_signal_smoothing_rows": {"supported": smooth_count, "total": len(smoothing)},
        "stability_mean_validation_accuracy_drop": stability_summary,
        "spectral_dataset_eigensolver": stage_a_summary.get("spectral_summary", []),
        "spectral_response_summary": stage_a_summary.get("spectral_response_summary", []),
        "stability_audit_correction": stage_a_summary.get("stability_audit_rerun", {}),
    }

def _render_final_e1(summary: dict[str, Any]) -> str:
    lines = [
        "# Experiment 1: Multi-Order mechanism audit", "",
        "The archived uniform/terminal result remains frozen as reported in the original benchmark outputs: +2.447 percentage points Validation Accuracy, 5/5 datasets and 15/15 paired seeds. GPR over Uniform remains +0.13 percentage points. This report does not recompute or overwrite those facts.", "",
        "The new fixed readout controls are analyzed separately. Test metrics are descriptive only.", "",
        "## New paired validation contrasts", "",
        "| Contrast | Mean (pp) | Population SD (pp) | Positive seed pairs |", "|---|---:|---:|---:|",
    ]
    for name, row in summary["new_order_source_contrasts"].items():
        lines.append(f"| {name} | {row['mean_pp']:.3f} | {row['population_std_pp']:.3f} | {row['positive_paired_seed_count']}/{row['paired_seed_count']} |")
    lines.extend(["", "## Alpha scan", "", "Alpha is the fixed coefficient on S0 in alpha*S0 + (1-alpha)*S3. Values are validation summaries across the fixed controls; no alpha was tuned.", "", "| alpha | Val Accuracy mean | Val Macro-F1 mean |", "|---:|---:|---:|"])
    for alpha, row in summary["alpha_scan_descriptive_only"].items():
        lines.append(f"| {alpha} | {row['val_accuracy_mean']:.4f} | {row['val_macro_f1_mean']:.4f} |")
    diagnostics = summary.get("existing_checkpoint_diagnostics", {})
    lines.extend(["", "## Existing-checkpoint diagnostics", "",
                  "Hop-probe mean Validation Accuracy by modality and order:", "",
                  "| Modality | S0 | S1 | S2 | S3 |", "|---|---:|---:|---:|---:|"])
    for mode, orders in diagnostics.get("hop_probe_mean_by_order", {}).items():
        lines.append("| " + mode + " | " + " | ".join(
            f"{orders.get(str(order), {}).get('val_accuracy_mean', float('nan')):.4f}" for order in range(4)) + " |")
    lines.extend(["", "Cumulative innovation-probe Accuracy gain over S0:", "",
                  "| Highest added order | Mean gain | Positive comparisons |", "|---:|---:|---:|"])
    for order, value in diagnostics.get("incremental_probe_gain_over_S0", {}).items():
        lines.append(f"| {order} | {value['mean_accuracy_gain_over_S0']:.4f} | {value['positive_comparisons']}/{value['comparisons']} |")
    smooth = diagnostics.get("four_signal_smoothing_rows", {})
    lines.extend(["", f"Four-signal progressive smoothing/collapse checks met: {smooth.get('supported', 0)}/{smooth.get('total', 0)} rows.",
                  "This requires variance decrease, effective-rank decrease, Dirichlet-energy decrease, and sampled node-cosine increase together.", "",
                  f"The frozen stability perturbations were rerun after the sparse operator cache was fixed: {bool(diagnostics.get('stability_audit_correction'))}.", "",
                  "See representation_geometry.csv, hop_task_probe.csv, propagation_innovation.csv, innovation_probe.csv, incremental_probe.csv, spectral_response.csv and stability_audit.csv for full details. Probe accuracy is evidence of task information, not a causal contribution.", "",
                  "Original checkpoint outputs and benchmark summaries were not modified.", ""])
    return "\n".join(lines)


def _problem_discovery_matrix(result_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    suff = result_rows["modality_sufficiency"]
    branch = [r for r in result_rows["joint_branch_suppression"] if "branch_suppression_descriptive" in r]
    utility = result_rows["modality_structure_utility"]
    conflict = [r for r in result_rows["structure_attribute_conflict"] if r.get("analysis") == "D_SC_vs_G_struct"]
    role_inter = result_rows["edge_role_intervention"]
    semantic = result_rows["semantic_utility_relation"]
    interaction = [r for r in result_rows["crossmodal_interaction_probe"] if r.get("analysis") == "interaction_gain_by_D_TV_quantile"]
    alignment = [r for r in result_rows["crossmodal_alignment_trajectory"] if r.get("analysis") == "aligned_distance_task_relevance"]
    residuals = [r for r in result_rows["shared_private_proxy"] if r.get("probe") in {"text_linear_residual", "visual_linear_residual"}]
    cube = result_rows["interaction_cube"]
    regressions = result_rows["interaction_regression"]
    matrix = []

    def evidence_rows(rows, pred):
        selected = [r for r in rows if pred(r)]
        datasets = sorted({r["dataset"] for r in selected})
        seeds = {(r.get("dataset"), r.get("seed")) for r in selected}
        return selected, datasets, len(seeds)

    candidates = [
        ("P1", "intrinsic semantic loss during propagation",
         "DESCRIPTIVE_ONLY: aligned variance, effective-rank, Dirichlet-energy and node-cosine trends.",
         "TASK_RELEVANT: frozen S0 versus S3 probe accuracy/F1 by modality.",
         "No order-state intervention isolated intrinsic loss; readout controls are trained comparisons.", "C_DESCRIPTIVE_ONLY", False),
        ("P2", "task-relevant propagation innovation",
         "DESCRIPTIVE_ONLY: Delta order energy and cosine relative to prior state and S0.",
         "TASK_RELEVANT: frozen innovation and cumulative S0+Delta probes on validation.",
         "Probe evidence is observational and is not a causal contribution test.", "C_DESCRIPTIVE_ONLY", True),
        ("P3", "modality imbalance or branch suppression",
         "Independent modality performance and joint branch probe differences.",
         "Validation accuracy/loss and branch suppression consistency across seeds.",
         "Unimodal retrained controls test modality sufficiency; branch probes are diagnostic.", "B_PLAUSIBLE_CANDIDATE", True),
        ("P4", "modality complementarity",
         "Multimodal versus best unimodal validation performance.",
         "Complementarity = A_TV - max(A_T,A_V), paired by dataset and seed.",
         "Retrained modality controls are the direct sufficiency comparison; no test selection.", "B_PLAUSIBLE_CANDIDATE", True),
        ("P5", "structure-attribute predictive conflict",
         "D_SC = JS(p_self,p_context) prevalence and quantile distribution.",
         "Spearman association with model and probe graph benefit on validation nodes.",
         "Node-level D_SC has no isolated causal intervention in this project.", "B_PLAUSIBLE_CANDIDATE", True),
        ("P6", "modality-specific topology utility",
         "Paired G_T and G_V distributions from self/uniform unimodal controls.",
         "Task losses and sign-discordance fractions on validation nodes.",
         "Fixed readout controls compare propagation within each modality; edge roles add context.", "B_PLAUSIBLE_CANDIDATE", True),
        ("P7", "neighbor relation heterogeneity",
         "DESCRIPTIVE_ONLY: high-confidence edge-role prevalence at 0.75/0.25 cutoffs.",
         "TASK_RELEVANT: validation effects for edge-role groups.",
         "INTERVENTION_SUPPORTED only when frozen targeted removal differs from same-size random removal.", "C_DESCRIPTIVE_ONLY", True),
        ("P8", "semantic compatibility differs from task utility",
         "DESCRIPTIVE_ONLY: source-matched physical-edge compatibility percentiles and quantiles.",
         "TASK_RELEVANT: validation utility across compatibility bins.",
         "INTERVENTION_SUPPORTED only for group removal effects beyond matched random controls.", "C_DESCRIPTIVE_ONLY", True),
        ("P9", "conditional need for cross-modal interaction",
         "D_TV quantile prevalence and probe gains by disagreement quartile.",
         "Plain versus interaction probe validation accuracy in each fixed D_TV quantile.",
         "Frozen interaction probe does not intervene on the trained backbone.", "C_DESCRIPTIVE_ONLY", True),
        ("P10", "task-relevant cross-modal divergence",
         "Alignment trajectory and validation aligned-distance distribution.",
         "Association of train-fitted alignment distance with loss, gain and D_TV.",
         "No alignment intervention or alignment-trained model is included.", "B_PLAUSIBLE_CANDIDATE", True),
        ("P11", "task-relevant modality-private residual proxy",
         "Ridge predictable/residual feature variance on train-fitted mappings.",
         "Frozen validation probes on linear residual components.",
         "This is a linear proxy; it does not prove information-theoretic privacy.", "B_PLAUSIBLE_CANDIDATE", True),
        ("P12", "structure x attribute x modality interaction",
         "Train-median 2x2x2 validation cell prevalence and cell outcomes.",
         "Per-dataset/seed standardized descriptive 3-way regression directions.",
         "No significance claim or graph-node independent p-value is used.", "C_DESCRIPTIVE_ONLY", True),
    ]
    for pid, name, prevalence, relevance, intervention, default_status, mag_specific in candidates:
        datasets = sorted({row["dataset"] for rows in result_rows.values() for row in rows if row.get("dataset")})
        selected = []
        if pid == "P1":
            geom_path = RESULTS_ROOT / "experiment1/representation_smoothing.csv"
            hop_path = RESULTS_ROOT / "experiment1/hop_task_probe.csv"
            geom_rows = list(csv.DictReader(geom_path.open())) if geom_path.is_file() else []
            hop_rows = list(csv.DictReader(hop_path.open())) if hop_path.is_file() else []
            by_hop = {(r["dataset"], int(r["encoder_seed"]), r["modality_mode"], int(r["order"])): float(r["val_accuracy"]) for r in hop_rows}
            selected = []
            for key, acc0 in by_hop.items():
                dataset, seed, mode, order = key
                if order == 0 and (dataset, seed, mode, 3) in by_hop and by_hop[(dataset, seed, mode, 3)] < acc0:
                    selected.append({"dataset": dataset, "seed": seed, "mode": mode,
                                     "probe_drop": acc0 - by_hop[(dataset, seed, mode, 3)]})
            supported = sorted({r["dataset"] for r in selected})
            nseed = len({(r["dataset"], r["seed"]) for r in selected})
            smooth_n = sum(r.get("joint_smoothing_collapse_evidence", "False").lower() == "true" for r in geom_rows)
            prevalence = f"DESCRIPTIVE_ONLY: {smooth_n}/{len(geom_rows)} dataset-seed-modality-readout rows meet all four smoothing trends."
            relevance = f"TASK_RELEVANT: S0 validation accuracy exceeds S3 in {nseed} dataset-seed-mode comparisons."
        elif pid == "P2":
            inc_path = RESULTS_ROOT / "experiment1/incremental_probe.csv"
            hop_path = RESULTS_ROOT / "experiment1/hop_task_probe.csv"
            inc_rows = list(csv.DictReader(inc_path.open())) if inc_path.is_file() else []
            hop_rows = list(csv.DictReader(hop_path.open())) if hop_path.is_file() else []
            baseline = {(r["dataset"], int(r["encoder_seed"]), r["modality_mode"]): float(r["val_accuracy"])
                        for r in hop_rows if r["order"] == "0"}
            selected = []
            for row in inc_rows:
                key = (row["dataset"], int(row["encoder_seed"]), row["modality_mode"])
                if row.get("order") == "S0_plus_Delta1_to_3" and key in baseline and float(row["val_accuracy"]) > baseline[key]:
                    selected.append(row)
            supported = sorted({r["dataset"] for r in selected})
            nseed = len({(r["dataset"], int(r["encoder_seed"])) for r in selected})
            relevance = f"TASK_RELEVANT: cumulative S0+Delta1..3 probe improves over S0 in {nseed} dataset-seed-mode comparisons."
        elif pid == "P3": selected, supported, nseed = evidence_rows(branch, lambda r: r.get("branch_suppression_descriptive", 0) > 0)
        elif pid == "P4": selected, supported, nseed = evidence_rows(suff, lambda r: r.get("multimodal_complementarity_accuracy", 0) > 0)
        elif pid == "P5": selected, supported, nseed = evidence_rows(conflict, lambda r: abs(r.get("spearman", 0)) >= .2)
        elif pid == "P6": selected, supported, nseed = evidence_rows(utility, lambda r: r.get("fraction_G_T_positive_G_V_negative", 0) + r.get("fraction_G_T_negative_G_V_positive", 0) > .05)
        elif pid in {"P7", "P8"}:
            candidates_rows = role_inter if pid == "P7" else semantic
            group_field = "edge_role" if pid == "P7" else "semantic_percentile_quantile"
            grouped: dict[tuple[str, Any], list[dict[str, Any]]] = defaultdict(list)
            for item in candidates_rows:
                grouped[(item["dataset"], item[group_field])].append(item)
            selected = []
            for _, group in grouped.items():
                eligible = [item for item in group if abs(float(item.get("targeted_effect_minus_random_mean", 0))) >= .005]
                signs = {int(np.sign(float(item["targeted_effect_minus_random_mean"]))) for item in eligible}
                if len(eligible) >= 2 and len(signs) == 1:
                    selected.extend(eligible)
            supported = sorted({item["dataset"] for item in selected})
            nseed = len({(item["dataset"], item["seed"]) for item in selected})
        elif pid == "P9": selected, supported, nseed = evidence_rows(interaction, lambda r: r.get("accuracy_gain", 0) > 0)
        elif pid == "P10": selected, supported, nseed = evidence_rows(alignment, lambda r: abs(r.get("spearman_aligned_distance", 0)) >= .2)
        elif pid == "P11": selected, supported, nseed = evidence_rows(residuals, lambda r: r.get("val_accuracy", 0) > .1)
        elif pid == "P12": selected, supported, nseed = evidence_rows(regressions, lambda r: abs(r.get("betaACM", 0)) > .01)
        else:
            supported, nseed = [], 0
        dataset_n = len(set(supported))
        consistent = nseed >= max(1, dataset_n * 2)
        status = default_status
        if pid in {"P7", "P8"} and dataset_n >= 3 and consistent:
            status = "A_STRONG_CANDIDATE"
        elif pid in {"P3", "P4", "P5", "P6", "P9", "P10", "P11", "P12"} and dataset_n >= 2 and consistent:
            status = "B_PLAUSIBLE_CANDIDATE"
        elif pid in {"P1", "P2"} and dataset_n >= 2 and consistent:
            status = "B_PLAUSIBLE_CANDIDATE"
        elif pid in {"P1", "P2"}:
            status = "C_DESCRIPTIVE_ONLY"
        elif not selected and pid not in {"P1", "P2"}:
            status = "D_NOT_SUPPORTED"
        matrix.append({
            "problem_id": pid, "problem_name": name,
            "prevalence_evidence": prevalence,
            "task_relevance_evidence": relevance,
            "intervention_evidence": intervention,
            "datasets_supported": json.dumps(supported),
            "seed_consistency": f"{nseed} dataset-seed units across {dataset_n} datasets; descriptive threshold only",
            "simple_MOB_unresolved": "yes; requires follow-up comparison" if pid not in {"P1", "P2"} else "not resolved by diagnostics",
            "MAG_specificity": "candidate; requires cross-dataset consistency" if mag_specific else "not established",
            "status": status,
            "status_basis": "rule-based evidence hierarchy; intervention candidates require >=0.5 percentage-point target-minus-random effect with same direction in at least two seeds per dataset; no cosine/gamma/prevalence alone promoted to task relevance",
        })
    return matrix


def _render_e2_report(rows, matrix, summary):
    lines = [
        "# Experiment 2: MAG interaction and problem discovery", "",
        "All problem discovery uses train and validation labels only. Test labels and test results do not enter thresholds, features, probes, correlations, interventions or ranking.", "",
        f"Validation node tables: {summary['node_table_files']} files, {summary['node_table_rows']} rows; hashes are in `outputs/mechanism_discovery_v1/node_tables_manifest.csv`.", "",
        "## Evidence hierarchy", "",
        "- `DESCRIPTIVE_ONLY`: prevalence or representation statistics without task relevance.",
        "- `TASK_RELEVANT`: consistent validation probe, loss, or prediction evidence.",
        "- `INTERVENTION_SUPPORTED`: frozen targeted intervention effect beyond a matched same-size random control.",
        "- Frozen edge removal is an intervention on an existing checkpoint, not retrained ablation.",
        "- No topology rewiring training or paper model was run.", "",
        "## Problem Discovery Matrix", "",
        "| ID | Candidate problem | Status | Supported datasets |", "|---|---|---|---|",
    ]
    for row in matrix:
        lines.append(f"| {row['problem_id']} | {row['problem_name']} | {row['status']} | {row['datasets_supported']} |")
    lines.extend(["", "## Evidence files", ""])
    for key, records in rows.items():
        lines.append(f"- `{key}.csv`: {len(records)} records")
    lines.extend(["", "Statuses summarize evidence hierarchy, not a research claim. Candidate ranking remains provisional and should be reviewed before choosing the next research problem.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/validation-only frozen mechanism diagnostics.")
    parser.add_argument("phase", choices=("step-a", "stability", "final"))
    args = parser.parse_args()
    if args.phase == "step-a":
        _step_a()
    elif args.phase == "stability":
        rows = []
        for dataset in DATASETS:
            for seed in SEEDS:
                cfg, data, payload, model, head, train, val, classes = _load_model_data(dataset, seed, "uniform")
                for readout in ("terminal", "uniform", "gpr"):
                    rows.extend(_stability_for_checkpoint(dataset, seed, readout, train, val, classes))
                print(f"STABILITY {dataset} seed={seed}", flush=True)
                del cfg, data, payload, model, head
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        _write_csv(RESULTS_ROOT / "experiment1/stability_audit.csv", rows)
        summary_path = RESULTS_ROOT / "experiment1/experiment1_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["stability_audit_rerun"] = {
            "reason": "operator cache now retains the exact edge tensor to prevent data_ptr reuse across equal-size edge interventions",
            "rows": len(rows), "seeded_perturbation_outputs": "recomputed for all existing checkpoints",
        }
        _write_json(summary_path, summary)
        print(f"STABILITY_COMPLETE rows={len(rows)}", flush=True)
    else:
        _final_analysis()


if __name__ == "__main__":
    main()
