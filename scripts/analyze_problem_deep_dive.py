from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.analyze_mechanism_discovery as d2
from src.analysis.problem_deep_dive import (
    canonical_pairs, context_novelty, fixed_norm_mask, four_regimes,
    matched_random_edge_sets, normalized_physical_operator,
    source_matched_compatibility, structural_edge_descriptors,
    train_edge_quantile_bins, train_edge_quartile_thresholds,
)
from src.data import load_mag_data
from src.models import build_model

DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
OUT = ROOT / "outputs/problem_deep_dive_v1"
RESULTS = ROOT / "results/problem_deep_dive_v1"
CACHE = OUT / "checkpoint_cache"
NODE_DIR = OUT / "node_intervention_effects"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=True), encoding="utf-8")


def _compat(states: torch.Tensor, edge_index: torch.Tensor, seed: int, pairs: torch.Tensor):
    scored_pairs, percentile, meta = source_matched_compatibility(states, edge_index, seed)
    if not torch.equal(scored_pairs.T.contiguous(), pairs):
        raise RuntimeError("source-matched edge compatibility changed the canonical edge set")
    return percentile.float(), meta


def _node_average(values: torch.Tensor, pairs: torch.Tensor, n: int) -> torch.Tensor:
    sums = torch.zeros(n); counts = torch.zeros(n)
    valid = torch.isfinite(values)
    p = pairs[valid]; v = values[valid]
    sums.index_add_(0, p[:, 0], v); sums.index_add_(0, p[:, 1], v)
    counts.index_add_(0, p[:, 0], torch.ones(len(p))); counts.index_add_(0, p[:, 1], torch.ones(len(p)))
    out = torch.full((n,), float("nan")); keep = counts > 0
    out[keep] = sums[keep] / counts[keep]
    return out


def _quantile_bin(values: torch.Tensor, bins: int = 4) -> torch.Tensor:
    thresholds = torch.quantile(values.float(), torch.arange(1, bins) / bins)
    return torch.bucketize(values.float().contiguous(), thresholds.contiguous(), right=False)


def _cached_contexts(dataset: str, seed: int) -> dict[str, Any]:
    cfg, data, payload, model, head, train, val, classes = d2._load_model_data(dataset, seed, "uniform")
    known_idx = torch.cat([train, val]).long().cpu()
    n = data.num_nodes
    pairs = canonical_pairs(data.edge_index)
    descriptor = structural_edge_descriptors(data.edge_index, n, data.y, known_idx)
    if not torch.equal(descriptor["pairs"], pairs):
        raise RuntimeError("edge descriptor ordering mismatch")
    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[train] = True
    train_edges = train_mask[pairs[:, 0]] & train_mask[pairs[:, 1]]
    degree_q_pair = torch.floor(descriptor["degree_quantile"][pairs] * 4).clamp(max=3).long()
    degree_pair_bin = torch.sort(degree_q_pair, dim=1).values
    tilde_degree = descriptor["degree"][pairs] + 1.0
    norm_weight = torch.rsqrt(tilde_degree[:, 0] * tilde_degree[:, 1])
    weight_bin = _quantile_bin(norm_weight, 4)

    logits: dict[str, torch.Tensor] = {}
    compat: dict[str, torch.Tensor] = {}
    novelty: dict[str, torch.Tensor] = {}
    source_meta: dict[str, Any] = {}
    cache_names = (
        ("uniform", "uniform", "both", "uniform"),
        ("gpr", "gpr", "both", "gpr"),
        ("self_only", "self_only", "both", "self_only"),
        ("propagated_uniform", "propagated_uniform", "both", "propagated_uniform"),
        ("text_self", "self_only", "text", "text_self"),
        ("text_uniform", "uniform", "text", "text_uniform"),
        ("visual_self", "self_only", "visual", "visual_self"),
        ("visual_uniform", "uniform", "visual", "visual_uniform"),
    )
    for output_name, readout, mode, ckpt_name in cache_names:
        if output_name == "uniform":
            local_model, local_head, local_payload = model, head, payload
            ctx = d2._forward_context(model, head, data)
            local_cfg = cfg
        else:
            local_cfg, local_payload, local_model, local_head = d2._load_on_existing_data(
                dataset, seed, readout, mode, data, checkpoint_name=ckpt_name)
            ctx = d2._forward_context(local_model, local_head, data)
        logits[output_name] = {
            "train": ctx["logits"][train].float().cpu(),
            "val": ctx["logits"][val].float().cpu(),
        }
        if output_name in {"uniform", "gpr"}:
            readout_compat = {}
            for modality in ("text", "visual"):
                p, meta = _compat(ctx[f"H0_{modality}"], data.edge_index, seed * 101 + 7, pairs)
                compat[f"{output_name}_{modality}"] = p
                readout_compat[modality] = meta
                novelty[f"{output_name}_{modality}"] = context_novelty(ctx[f"S_{modality}"][3], pairs)
            source_meta[output_name] = readout_compat
        elif output_name == "text_uniform":
            p, meta = _compat(ctx["H0_text"], data.edge_index, seed * 101 + 7, pairs)
            compat[output_name] = p
            novelty[output_name] = context_novelty(ctx["S_text"][3], pairs)
            source_meta[output_name] = meta
        elif output_name == "visual_uniform":
            p, meta = _compat(ctx["H0_visual"], data.edge_index, seed * 101 + 7, pairs)
            compat[output_name] = p
            novelty[output_name] = context_novelty(ctx["S_visual"][3], pairs)
            source_meta[output_name] = meta
        del ctx
        if output_name != "uniform":
            del local_cfg, local_payload, local_model, local_head
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    c_u_t, c_u_v = compat["uniform_text"], compat["uniform_visual"]
    c_g_t, c_g_v = compat["gpr_text"], compat["gpr_visual"]
    n_u_t, n_u_v = novelty["uniform_text"], novelty["uniform_visual"]
    n_g_t, n_g_v = novelty["gpr_text"], novelty["gpr_visual"]
    c_joint = (c_u_t + c_u_v) / 2
    n_joint = torch.stack([n_u_t, n_u_v]).nanmean(0)
    c_train = train_edge_quartile_thresholds(c_joint, pairs, train)
    n_train = train_edge_quartile_thresholds(n_joint, pairs, train)
    embedded_train = train_edge_quartile_thresholds(descriptor["edge_embeddedness"], pairs, train)
    sim_novelty_group = four_regimes(c_joint, n_joint, c_train, n_train)
    semantic_q, semantic_q_thresholds = train_edge_quantile_bins(c_joint, pairs, train)
    gpr_joint = (c_g_t + c_g_v) / 2
    gpr_novelty_joint = torch.stack([n_g_t, n_g_v]).nanmean(0)
    gpr_q, gpr_q_thresholds = train_edge_quantile_bins(gpr_joint, pairs, train)
    gpr_similarity_thresholds = train_edge_quartile_thresholds(gpr_joint, pairs, train)
    gpr_novelty_thresholds = train_edge_quartile_thresholds(gpr_novelty_joint, pairs, train)
    gpr_regimes = four_regimes(gpr_joint, gpr_novelty_joint, gpr_similarity_thresholds, gpr_novelty_thresholds)
    text_c_train = train_edge_quartile_thresholds(compat["text_uniform"], pairs, train)
    text_n_train = train_edge_quartile_thresholds(novelty["text_uniform"], pairs, train)
    visual_c_train = train_edge_quartile_thresholds(compat["visual_uniform"], pairs, train)
    visual_n_train = train_edge_quartile_thresholds(novelty["visual_uniform"], pairs, train)
    text_groups = four_regimes(compat["text_uniform"], novelty["text_uniform"], text_c_train, text_n_train)
    visual_groups = four_regimes(compat["visual_uniform"], novelty["visual_uniform"], visual_c_train, visual_n_train)
    text_semantic_q, text_semantic_q_thresholds = train_edge_quantile_bins(compat["text_uniform"], pairs, train)
    visual_semantic_q, visual_semantic_q_thresholds = train_edge_quantile_bins(compat["visual_uniform"], pairs, train)

    edge_cache = {
        "pairs": pairs, "degree_pair_bin": degree_pair_bin, "weight_bin": weight_bin,
        "norm_weight": norm_weight, "descriptor": descriptor,
        "compat": compat, "novelty": novelty,
        "joint_compatibility": c_joint, "mean_novelty": n_joint,
        "gpr_joint_compatibility": gpr_joint, "gpr_mean_novelty": gpr_novelty_joint,
        "semantic_q_uniform": semantic_q, "semantic_q_gpr": gpr_q,
        "gpr_sim_novelty_group": gpr_regimes,
        "text_semantic_q": text_semantic_q, "visual_semantic_q": visual_semantic_q,
        "sim_novelty_group": sim_novelty_group,
        "text_sim_novelty_group": text_groups, "visual_sim_novelty_group": visual_groups,
        "train_thresholds": {"uniform_similarity": c_train, "uniform_novelty": n_train,
                              "embeddedness": embedded_train,
                              "uniform_semantic": semantic_q_thresholds, "gpr_semantic": gpr_q_thresholds,
                              "gpr_similarity": gpr_similarity_thresholds, "gpr_novelty": gpr_novelty_thresholds,
                              "text_similarity": text_c_train, "text_novelty": text_n_train,
                              "text_semantic": text_semantic_q_thresholds,
                              "visual_similarity": visual_c_train, "visual_novelty": visual_n_train,
                              "visual_semantic": visual_semantic_q_thresholds},
        "source_matching_metadata": source_meta,
    }
    def node_feature(values, name):
        return _node_average(values, pairs, n)
    node_features = {
        "degree": descriptor["degree"],
        "log_degree": torch.log1p(descriptor["degree"]),
        "degree_quantile": descriptor["degree_quantile"],
        "text_edge_compatibility": node_feature(c_u_t, "text compatibility"),
        "visual_edge_compatibility": node_feature(c_u_v, "visual compatibility"),
        "structural_compatibility": node_feature(c_joint, "joint compatibility"),
        "text_novelty_mean": node_feature(n_u_t, "text novelty"),
        "visual_novelty_mean": node_feature(n_u_v, "visual novelty"),
        "node_mean_novelty": node_feature(n_joint, "mean novelty"),
        "edge_embeddedness_mean": node_feature(descriptor["edge_embeddedness"], "embeddedness"),
    }
    known_y = torch.full_like(data.y.detach().cpu(), -1)
    known_y[known_idx] = data.y.detach().cpu()[known_idx]
    def _put_split_feature(name, split, values):
        full = torch.full((n,), float("nan"), dtype=torch.float32)
        full[split] = values.detach().float().cpu()
        node_features[name] = full
    probability_cache = {}
    for name, split_name in (("self_only", "both"), ("propagated_uniform", "both"),
                             ("text_uniform", "text"), ("visual_uniform", "visual")):
        probability_cache[name] = {}
        for split_name in ("train", "val"):
            p = logits[name][split_name].softmax(-1)
            probability_cache[name][split_name] = p
            confidence = p.max(-1).values
            entropy = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1)
            _put_split_feature(f"{name}_confidence", train if split_name == "train" else val, confidence)
            _put_split_feature(f"{name}_entropy", train if split_name == "train" else val, entropy)
    for split_name in ("train", "val"):
        split = train if split_name == "train" else val
        js_sc = d2.js_divergence(probability_cache["self_only"][split_name],
                                 probability_cache["propagated_uniform"][split_name])
        js_tv = d2.js_divergence(probability_cache["text_uniform"][split_name],
                                 probability_cache["visual_uniform"][split_name])
        _put_split_feature("self_context_js", split, js_sc)
        _put_split_feature("D_TV", split, js_tv)
        _put_split_feature("abs_confidence_gap_TV", split,
                           (probability_cache["text_uniform"][split_name].max(-1).values -
                            probability_cache["visual_uniform"][split_name].max(-1).values).abs())
        _put_split_feature("abs_entropy_gap_TV", split,
                           (-(probability_cache["text_uniform"][split_name].clamp_min(1e-12) *
                              probability_cache["text_uniform"][split_name].clamp_min(1e-12).log()).sum(-1) +
                             (probability_cache["visual_uniform"][split_name].clamp_min(1e-12) *
                              probability_cache["visual_uniform"][split_name].clamp_min(1e-12).log()).sum(-1)).abs())
    cache = {
        "cache_version": 2, "dataset": dataset, "seed": seed, "num_nodes": n,
        "edge_index": data.edge_index.detach().long().cpu(), "train_idx": train,
        "val_idx": val, "known_idx": known_idx, "known_y": known_y,
        "num_classes": int(data.num_classes), "class_ids": classes,
        "logits": logits, "edge_cache": edge_cache, "node_features": node_features,
        "compat_sampling": source_meta,
    }
    del cfg, payload, model, head, data
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return cache


def _edge_diagnostic_rows(cache: dict[str, Any]) -> list[dict[str, Any]]:
    e = cache["edge_cache"]; d = e["descriptor"]; p = e["pairs"]
    n = cache["num_nodes"]
    rows = []
    for idx, (i, j) in enumerate(p.tolist()):
        rows.append({
            "dataset": cache["dataset"], "seed": cache["seed"], "node_i": i, "node_j": j,
            "degree_i": float(d["degree_i"][idx]), "degree_j": float(d["degree_j"][idx]),
            "degree_quantile_i": float(d["degree_quantile_i"][idx]),
            "degree_quantile_j": float(d["degree_quantile_j"][idx]),
            "common_neighbors": float(d["common_neighbors"][idx]),
            "neighbor_jaccard": float(d["neighbor_jaccard"][idx]),
            "adamic_adar": float(d["adamic_adar"][idx]),
            "edge_embeddedness": float(d["edge_embeddedness"][idx]),
            "normalized_edge_weight": float(e["norm_weight"][idx]),
            "same_label_edge_descriptive_only": float(d["same_label_edge"][idx]),
            "uniform_text_compatibility": float(e["compat"]["uniform_text"][idx]),
            "uniform_visual_compatibility": float(e["compat"]["uniform_visual"][idx]),
            "uniform_joint_compatibility": float(e["joint_compatibility"][idx]),
            "uniform_text_novelty": float(e["novelty"]["uniform_text"][idx]),
            "uniform_visual_novelty": float(e["novelty"]["uniform_visual"][idx]),
            "uniform_mean_novelty": float(e["mean_novelty"][idx]),
            "gpr_text_compatibility_robustness": float(e["compat"]["gpr_text"][idx]),
            "gpr_visual_compatibility_robustness": float(e["compat"]["gpr_visual"][idx]),
            "gpr_text_novelty_robustness": float(e["novelty"]["gpr_text"][idx]),
            "gpr_visual_novelty_robustness": float(e["novelty"]["gpr_visual"][idx]),
            "gpr_joint_compatibility": float(e["gpr_joint_compatibility"][idx]),
            "gpr_mean_novelty": float(e["gpr_mean_novelty"][idx]),
            "gpr_similarity_novelty_regime": int(e["gpr_sim_novelty_group"][idx]),
            "text_only_compatibility": float(e["compat"]["text_uniform"][idx]),
            "text_only_novelty": float(e["novelty"]["text_uniform"][idx]),
            "visual_only_compatibility": float(e["compat"]["visual_uniform"][idx]),
            "visual_only_novelty": float(e["novelty"]["visual_uniform"][idx]),
            "similarity_novelty_regime": int(e["sim_novelty_group"][idx]),
            "text_similarity_novelty_regime": int(e["text_sim_novelty_group"][idx]),
            "visual_similarity_novelty_regime": int(e["visual_sim_novelty_group"][idx]),
            "text_semantic_q": int(e["text_semantic_q"][idx]),
            "visual_semantic_q": int(e["visual_semantic_q"][idx]),
            "semantic_q_uniform": int(e["semantic_q_uniform"][idx]),
            "semantic_q_gpr": int(e["semantic_q_gpr"][idx]),
        })
    return rows


def _cache_phase(datasets: tuple[str, ...], seeds: tuple[int, ...] = SEEDS, output_root: Path = OUT) -> None:
    cache_root = output_root / "checkpoint_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    edge_manifest = []
    cache_manifest = []
    for dataset in datasets:
        for seed in seeds:
            path = cache_root / f"{dataset}_seed{seed}.pt"
            if path.is_file():
                try:
                    existing = torch.load(path, map_location="cpu", weights_only=False)
                    if existing.get("cache_version") == 2:
                        cache = existing
                    else:
                        raise ValueError("stale cache")
                except Exception:
                    cache = _cached_contexts(dataset, seed)
                    torch.save(cache, path)
            else:
                print(f"CACHE {dataset} seed={seed}", flush=True)
                cache = _cached_contexts(dataset, seed)
                torch.save(cache, path)
            edge_path = output_root / "edge_diagnostics" / dataset / f"seed{seed}.csv.gz"
            edge_path.parent.mkdir(parents=True, exist_ok=True)
            if not edge_path.is_file():
                rows = _edge_diagnostic_rows(cache)
                with gzip.open(edge_path, "wt", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [], lineterminator="\n")
                    if rows:
                        w.writeheader(); w.writerows(rows)
            edge_manifest.append({"dataset": dataset, "seed": seed, "path": str(edge_path),
                                  "rows": int(cache["edge_cache"]["pairs"].size(0)), "sha256": _sha256(edge_path),
                                  "size_bytes": edge_path.stat().st_size})
            cache_manifest.append({"dataset": dataset, "seed": seed, "path": str(path),
                                   "sha256": _sha256(path), "size_bytes": path.stat().st_size})
            print(f"CACHE_COMPLETE {dataset} seed={seed} edges={cache['edge_cache']['pairs'].size(0)}", flush=True)
    _write_csv(output_root / "edge_diagnostics_manifest.csv", edge_manifest)
    _write_csv(output_root / "checkpoint_cache_manifest.csv", cache_manifest)


def _read_run_metrics(root: Path, dataset: str, variant: str) -> list[dict[str, Any]]:
    directory = root / dataset / variant
    path = directory / "run_metrics.json"
    if path.is_file():
        payload = json.loads(path.read_text())
        if payload.get("run_seeds") != list(SEEDS) or len(payload.get("runs", [])) != 3:
            raise ValueError(f"{path}: expected internal runs 42/43/44")
        return payload["runs"]
    # Immutable earlier D2 runs stored metrics in each selected checkpoint and only aggregate results.json.
    complete = directory / "complete.json"
    if not complete.is_file():
        raise FileNotFoundError(f"missing run_metrics.json and complete.json under {directory}")
    metadata = json.loads(complete.read_text())
    if metadata.get("run_seeds") != list(SEEDS):
        raise ValueError(f"{complete}: expected internal runs 42/43/44")
    runs = []
    for checkpoint_path in metadata.get("checkpoints", []):
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metrics = payload.get("metrics", {})
        runs.append({"seed": int(payload.get("seed", -1)), "metrics": metrics})
    if [int(run["seed"]) for run in runs] != list(SEEDS):
        raise ValueError(f"{complete}: checkpoint seeds do not equal 42/43/44")
    return runs


def _operator_transfer_analysis() -> dict[str, Any]:
    root = OUT / "operator_transfer"
    rows, paired = [], []
    for dataset in DATASETS:
        for operator, model in (("gcn", "operator_control_gcn"), ("sage", "operator_control_sage")):
            deep = _read_run_metrics(root, dataset, f"{model}_deep_only")
            anchor = _read_run_metrics(root, dataset, f"{model}_anchored25")
            by_seed = {}
            for run in deep:
                by_seed[int(run["seed"])] = {"deep": run["metrics"]}
            for run in anchor:
                by_seed[int(run["seed"])]["anchored25"] = run["metrics"]
            for seed in SEEDS:
                d, a = by_seed[seed]["deep"], by_seed[seed]["anchored25"]
                paired.append({"dataset": dataset, "operator": operator, "seed": seed,
                               "deep_val_accuracy": d["val_acc"], "anchored25_val_accuracy": a["val_acc"],
                               "val_accuracy_gain": a["val_acc"] - d["val_acc"],
                               "deep_val_macro_f1": d["val_macro_f1"], "anchored25_val_macro_f1": a["val_macro_f1"],
                               "val_macro_f1_gain": a["val_macro_f1"] - d["val_macro_f1"],
                               "test_accuracy_descriptive_gain": a["test_acc"] - d["test_acc"]})
            gains = [by_seed[seed]["anchored25"]["val_acc"] - by_seed[seed]["deep"]["val_acc"] for seed in SEEDS]
            rows.append({"dataset": dataset, "operator": operator,
                         "deep_val_accuracy_mean": float(np.mean([by_seed[s]["deep"]["val_acc"] for s in SEEDS])),
                         "deep_val_accuracy_population_std": float(np.std([by_seed[s]["deep"]["val_acc"] for s in SEEDS], ddof=0)),
                         "anchored25_val_accuracy_mean": float(np.mean([by_seed[s]["anchored25"]["val_acc"] for s in SEEDS])),
                         "anchored25_val_accuracy_population_std": float(np.std([by_seed[s]["anchored25"]["val_acc"] for s in SEEDS], ddof=0)),
                         "deep_val_macro_f1_mean": float(np.mean([by_seed[s]["deep"]["val_macro_f1"] for s in SEEDS])),
                         "anchored25_val_macro_f1_mean": float(np.mean([by_seed[s]["anchored25"]["val_macro_f1"] for s in SEEDS])),
                         "anchored25_minus_deep_mean": float(np.mean(gains)),
                         "anchored25_minus_deep_population_std": float(np.std(gains, ddof=0)),
                         "positive_seed_pairs": int(sum(g > 0 for g in gains)), "seed_pairs": len(gains)})
    operator_summary = {}
    for operator in ("gcn", "sage"):
        selected = [r for r in rows if r["operator"] == operator]
        operator_summary[operator] = {
            "mean_gain": float(np.mean([r["anchored25_minus_deep_mean"] for r in selected])),
            "positive_datasets": int(sum(r["anchored25_minus_deep_mean"] > 0 for r in selected)),
        }
    supported = all(v["mean_gain"] > 0 and v["positive_datasets"] >= 3 for v in operator_summary.values())
    gate = "ANCHORING_PRINCIPLE_TRANSFER_SUPPORTED" if supported else "OPERATOR_DEPENDENT"
    _write_csv(RESULTS / "operator_transfer.csv", rows)
    _write_csv(RESULTS / "operator_transfer_paired.csv", paired)
    lines = ["# Operator-transfer control", "", f"Pre-registered decision: **{gate}**.", "",
             "Validation is primary. Test differences are descriptive only.", "",
             "| Operator | Mean anchored25 − deep_only (pp) | Positive datasets | Positive seed pairs |", "|---|---:|---:|---:|"]
    for op in ("gcn", "sage"):
        selected = [r for r in rows if r["operator"] == op]
        mean = operator_summary[op]["mean_gain"] * 100
        pos_ds = operator_summary[op]["positive_datasets"]
        pos_seed = sum(r["positive_seed_pairs"] for r in selected)
        lines.append(f"| {op} | {mean:.3f} | {pos_ds}/5 | {pos_seed}/15 |")
    lines.extend(["", "This is a fixed-readout operator comparison, not a paper claim or new backbone proposal.", ""])
    (RESULTS / "operator_transfer_report.md").parent.mkdir(parents=True, exist_ok=True)
    (RESULTS / "operator_transfer_report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"gate": gate, "operators": operator_summary, "rows": rows, "paired_rows": paired}


def _topology_analysis() -> list[dict[str, Any]]:
    rewired_root = OUT / "rewired_topology"
    real_root = ROOT / "outputs/mob_factorial_nc_v1"
    rows = []
    for dataset in DATASETS:
        real = _read_run_metrics(real_root, dataset, "mob_uniform_plain")
        rewired = _read_run_metrics(rewired_root, dataset, "multi_order_bank_uniform")
        rew_by_seed = {int(r["seed"]): r["metrics"] for r in rewired}
        real_by_seed = {int(r["seed"]): r["metrics"] for r in real}
        for seed in SEEDS:
            a, b = real_by_seed[seed], rew_by_seed[seed]
            rows.append({"dataset": dataset, "seed": seed,
                         "real_val_accuracy": a["val_acc"], "rewired_val_accuracy": b["val_acc"],
                         "real_minus_rewired_val_accuracy": a["val_acc"] - b["val_acc"],
                         "real_val_macro_f1": a["val_macro_f1"], "rewired_val_macro_f1": b["val_macro_f1"],
                         "real_minus_rewired_val_macro_f1": a["val_macro_f1"] - b["val_macro_f1"],
                         "real_test_accuracy_descriptive": a["test_acc"],
                         "rewired_test_accuracy_descriptive": b["test_acc"]})
    _write_csv(RESULTS / "topology_reality_check.csv", rows)
    return rows


def _known_eval_metrics(logits: torch.Tensor, y: torch.Tensor, idx: torch.Tensor, classes: list[int]):
    return d2.classification_metrics(logits[idx], y[idx], classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D3 train/validation-only analysis and formal control evaluation.")
    parser.add_argument("phase", choices=("cache", "operator", "topology", "selectors", "interventions", "intervention-one", "finalize", "all"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpus", default="0,1", help="physical GPU IDs for parallel intervention jobs")
    args = parser.parse_args()
    selected = tuple(args.datasets)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.phase in {"cache", "all"}:
        _cache_phase(selected, tuple(args.seeds))
    if args.phase in {"operator", "all"}:
        _operator_transfer_analysis()
    if args.phase in {"topology", "all"}:
        _topology_analysis()
    if args.phase in {"selectors", "all"}:
        from scripts.problem_deep_dive_analysis import run_analysis_phase
        run_analysis_phase("selectors", selected, tuple(args.seeds), args.device)
    if args.phase in {"interventions", "all"}:
        from scripts.problem_deep_dive_analysis import run_analysis_phase
        intervention_device = args.device if args.device == "cpu" else f"gpu_ids:{args.gpus}"
        run_analysis_phase("interventions", selected, tuple(args.seeds), intervention_device)
    if args.phase == "intervention-one":
        from scripts.problem_deep_dive_analysis import run_analysis_phase
        run_analysis_phase("intervention-one", selected, tuple(args.seeds), args.device)
    if args.phase == "finalize":
        from scripts.problem_deep_dive_analysis import run_analysis_phase
        run_analysis_phase("finalize", selected, tuple(args.seeds), args.device)
    if args.phase == "all":
        from scripts.problem_deep_dive_analysis import run_analysis_phase
        run_analysis_phase("finalize", selected, tuple(args.seeds), args.device)
