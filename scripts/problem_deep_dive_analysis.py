from __future__ import annotations

import csv
import gzip
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scripts.gpu_scheduler import GPUJobOutcome, run_gpu_jobs

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.analyze_mechanism_discovery as d2
import scripts.analyze_problem_deep_dive as d3
from src.analysis.problem_deep_dive import (
    affected_within_hops,
    build_csr_adjacency,
    canonical_pairs,
    fit_train_only_logistic,
    fixed_norm_mask,
    matched_random_edge_sets,
    normalized_physical_operator,
    remove_pairs_vectorized,
)

DATASETS = d3.DATASETS
SEEDS = d3.SEEDS
RESULTS = d3.RESULTS
OUT = d3.OUT

SELF_STRUCTURE_SELECTOR_FEATURES = ["self_only_confidence", "propagated_uniform_confidence",
                                   "self_only_entropy", "propagated_uniform_entropy", "self_context_js",
                                   "log_degree", "text_edge_compatibility", "visual_edge_compatibility",
                                   "node_mean_novelty", "D_TV"]
MODALITY_SELECTOR_FEATURES = ["text_uniform_confidence", "visual_uniform_confidence",
                              "text_uniform_entropy", "visual_uniform_entropy", "D_TV",
                              "text_edge_compatibility", "visual_edge_compatibility",
                              "text_novelty_mean", "visual_novelty_mean", "log_degree"]

UTILITY_FEATURE_GROUPS = {
    "SIMILARITY_ONLY": ["text_edge_compatibility", "visual_edge_compatibility", "structural_compatibility"],
    "TOPOLOGY_ONLY": ["log_degree", "degree_quantile", "edge_embeddedness_mean"],
    "CONFIDENCE_ONLY": ["self_only_confidence", "propagated_uniform_confidence", "self_only_entropy", "propagated_uniform_entropy", "self_context_js"],
    "NOVELTY_ONLY": ["text_novelty_mean", "visual_novelty_mean", "node_mean_novelty"],
    "MODAL_DISAGREEMENT_ONLY": ["D_TV", "abs_confidence_gap_TV", "abs_entropy_gap_TV"],
}
UTILITY_FEATURE_GROUPS["ALL"] = list(dict.fromkeys(x for group in UTILITY_FEATURE_GROUPS.values() for x in group))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    d3._write_csv(path, rows)


def _read_cache(dataset: str, seed: int) -> dict[str, Any]:
    path = d3.CACHE / f"{dataset}_seed{seed}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing train/validation cache: {path}; run cache phase first")
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if cache.get("cache_version") != 2:
        raise ValueError(f"{path}: expected cache_version=2, got {cache.get('cache_version')}")
    return cache


def _split_labels(cache: dict[str, Any], split: str) -> torch.Tensor:
    return cache["known_y"][cache[f"{split}_idx"]]


def _loss(cache: dict[str, Any], name: str, split: str) -> torch.Tensor:
    return d2.node_cross_entropy(cache["logits"][name][split], _split_labels(cache, split))


def _node_x(cache: dict[str, Any], feature_names: list[str]) -> torch.Tensor:
    return torch.stack([cache["node_features"][name] for name in feature_names], dim=-1)


def _soft_metrics(logits: torch.Tensor, labels: torch.Tensor, classes: list[int]) -> dict[str, float]:
    return d2.classification_metrics(logits.detach().cpu(), labels.detach().cpu(), classes)


def _selected_logits(a: torch.Tensor, b: torch.Tensor, choose_a: torch.Tensor) -> torch.Tensor:
    return torch.where(choose_a.to(dtype=torch.bool).unsqueeze(-1), a, b)


def _selection_row(dataset: str, seed: int, family: str, method: str,
                   train_pred: torch.Tensor, val_pred: torch.Tensor,
                   oracle_train: torch.Tensor, oracle_val: torch.Tensor,
                   logits_a: dict[str, torch.Tensor], logits_b: dict[str, torch.Tensor],
                   labels_train: torch.Tensor, labels_val: torch.Tensor,
                   classes: list[int], static_choice_a: bool,
                   reference_logits_val: torch.Tensor) -> dict[str, Any]:
    selected_val = _selected_logits(logits_a["val"], logits_b["val"], val_pred)
    metrics = _soft_metrics(selected_val, labels_val, classes)
    static_val = _soft_metrics(logits_a["val"] if static_choice_a else logits_b["val"], labels_val, classes)
    ref_val = _soft_metrics(reference_logits_val, labels_val, classes)
    train_selected = _soft_metrics(_selected_logits(logits_a["train"], logits_b["train"], train_pred), labels_train, classes)
    oracle_logits = _selected_logits(logits_a["val"], logits_b["val"], oracle_val)
    oracle_acc = _soft_metrics(oracle_logits, labels_val, classes)["accuracy"]
    return {
        "dataset": dataset, "seed": seed, "family": family, "method": method,
        "val_selection_accuracy": float((val_pred == oracle_val).float().mean()),
        "val_selected_accuracy": metrics["accuracy"], "val_selected_macro_f1": metrics["macro_f1"],
        "val_best_static_accuracy": static_val["accuracy"],
        "val_gain_vs_best_static": metrics["accuracy"] - static_val["accuracy"],
        "val_reference_accuracy": ref_val["accuracy"],
        "val_gain_vs_reference": metrics["accuracy"] - ref_val["accuracy"],
        "val_oracle_accuracy": oracle_acc,
        "val_oracle_headroom_vs_static": oracle_acc - static_val["accuracy"],
        "train_selection_accuracy": float((train_pred == oracle_train).float().mean()),
        "train_selected_accuracy": train_selected["accuracy"],
    }


def _selector_analysis_one(cache: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset, seed = cache["dataset"], cache["seed"]
    train, val = cache["train_idx"], cache["val_idx"]
    ytr, yva = _split_labels(cache, "train"), _split_labels(cache, "val")
    classes = cache["class_ids"]
    f = cache["node_features"]
    oracle_rows, learned_rows = [], []

    def run_family(family: str, expert_a: str, expert_b: str, features: list[str],
                   reference: str, conf_a: str, conf_b: str,
                   entropy_a: str, entropy_b: str):
        la = cache["logits"][expert_a]
        lb = cache["logits"][expert_b]
        loss_a_tr = d2.node_cross_entropy(la["train"], ytr)
        loss_b_tr = d2.node_cross_entropy(lb["train"], ytr)
        loss_a_va = d2.node_cross_entropy(la["val"], yva)
        loss_b_va = d2.node_cross_entropy(lb["val"], yva)
        oracle_tr = loss_a_tr < loss_b_tr
        oracle_va = loss_a_va < loss_b_va
        acc_a_tr = _soft_metrics(la["train"], ytr, classes)["accuracy"]
        acc_b_tr = _soft_metrics(lb["train"], ytr, classes)["accuracy"]
        static_a = acc_a_tr >= acc_b_tr
        ref_logits = cache["logits"][reference]["val"]
        ref_train_logits = cache["logits"][reference]["train"]
        ref_train_accuracy = _soft_metrics(ref_train_logits, ytr, classes)["accuracy"]
        conf_choice = f[conf_a][val] >= f[conf_b][val]
        conf_choice_tr = f[conf_a][train] >= f[conf_b][train]
        ent_choice = f[entropy_a][val] <= f[entropy_b][val]
        ent_choice_tr = f[entropy_a][train] <= f[entropy_b][train]
        methods = [
            ("oracle", oracle_tr, oracle_va),
            ("confidence", conf_choice_tr, conf_choice),
            ("entropy", ent_choice_tr, ent_choice),
            ("best_static_train_selected", torch.full_like(oracle_tr, static_a), torch.full_like(oracle_va, static_a)),
        ]
        simple_candidates = []
        for method, train_pred, val_pred in methods:
            row = _selection_row(dataset, seed, family, method, train_pred, val_pred,
                                 oracle_tr, oracle_va, la, lb, ytr, yva, classes,
                                 static_a, ref_logits)
            row["train_reference_accuracy"] = ref_train_accuracy
            oracle_rows.append(row)
            if method in {"confidence", "entropy", "best_static_train_selected"}:
                simple_candidates.append((row["train_selected_accuracy"], row["val_selected_accuracy"], method))
        simple_candidates.append((ref_train_accuracy, _soft_metrics(ref_logits, yva, classes)["accuracy"], reference))
        _, best_simple_val, best_simple_name = max(simple_candidates, key=lambda item: item[0])
        # Logistic target labels are filled only on train rows; validation target values are not read by fit.
        target = torch.full((cache["num_nodes"],), -1, dtype=torch.long)
        target[train] = oracle_tr.long()
        valid_train = (loss_a_tr - loss_b_tr).abs() > 1e-10
        train_for_fit = train[valid_train]
        target[:] = -1
        target[train_for_fit] = oracle_tr[valid_train].long()
        x = _node_x(cache, features)
        try:
            fit = fit_train_only_logistic(x, target, train_for_fit, val, random_state=seed)
        except ValueError as exc:
            if "requires both target classes" not in str(exc):
                raise
            learned_rows.append({"dataset": dataset, "seed": seed, "family": family,
                                 "method": "learned_logistic_train_only", "selector_status": "NO_TRAIN_TARGET_VARIATION",
                                 "train_rows_fit": int(valid_train.sum()), "feature_names": ",".join(features),
                                 "best_simple_reference_method": best_simple_name,
                                 "val_selected_accuracy": float("nan"),
                                 "val_gain_vs_best_simple_reference": float("nan"),
                                 "val_oracle_headroom_vs_best_simple": float("nan")})
            return
        learned_pred = torch.as_tensor(fit["val_pred"], dtype=torch.bool)
        # Training predictions are only used as a diagnostic for fitted training rows.
        medians = fit["medians"]
        xtr = x[train_for_fit].numpy()
        xtr = np.where(np.isfinite(xtr), xtr, medians)
        learned_train_pred = torch.as_tensor(
            fit["model"].predict(fit["scaler"].transform(xtr)), dtype=torch.bool)
        la_fit = {"train": la["train"][valid_train], "val": la["val"]}
        lb_fit = {"train": lb["train"][valid_train], "val": lb["val"]}
        selected = _selection_row(dataset, seed, family, "learned_logistic_train_only",
                                  learned_train_pred, learned_pred, oracle_tr[valid_train], oracle_va,
                                  la_fit, lb_fit, ytr[valid_train], yva, classes, static_a, ref_logits)
        selected["selector_status"] = "FITTED"
        selected["train_rows_fit"] = int(valid_train.sum())
        selected["feature_names"] = ",".join(features)
        selected["best_simple_reference_method"] = best_simple_name
        selected["val_best_simple_reference_accuracy"] = best_simple_val
        selected["val_gain_vs_best_simple_reference"] = selected["val_selected_accuracy"] - best_simple_val
        selected["val_oracle_headroom_vs_best_simple"] = selected["val_oracle_accuracy"] - best_simple_val
        learned_rows.append(selected)

    run_family(
        "self_structure", "self_only", "propagated_uniform",
        SELF_STRUCTURE_SELECTOR_FEATURES,
        "uniform", "self_only_confidence", "propagated_uniform_confidence",
        "self_only_entropy", "propagated_uniform_entropy")
    run_family(
        "modality", "text_uniform", "visual_uniform",
        MODALITY_SELECTOR_FEATURES,
        "uniform", "text_uniform_confidence", "visual_uniform_confidence",
        "text_uniform_entropy", "visual_uniform_entropy")
    return oracle_rows, learned_rows


def _selector_analysis(datasets: tuple[str, ...], seeds: tuple[int, ...] = SEEDS) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    oracle, learned = [], []
    for dataset in datasets:
        for seed in seeds:
            print(f"SELECTOR {dataset} seed={seed}", flush=True)
            a, b = _selector_analysis_one(_read_cache(dataset, seed))
            oracle.extend(a); learned.extend(b)
    _write_csv(RESULTS / "self_structure_oracle.csv", [r for r in oracle if r["family"] == "self_structure"])
    _write_csv(RESULTS / "modality_oracle.csv", [r for r in oracle if r["family"] == "modality"])
    _write_csv(RESULTS / "self_structure_selector.csv", [r for r in learned if r["family"] == "self_structure"])
    _write_csv(RESULTS / "modality_selector.csv", [r for r in learned if r["family"] == "modality"])
    return oracle, learned


def _utility_predictability_one(cache: dict[str, Any]) -> list[dict[str, Any]]:
    dataset, seed = cache["dataset"], cache["seed"]
    train, val = cache["train_idx"], cache["val_idx"]
    ytr, yva = _split_labels(cache, "train"), _split_labels(cache, "val")
    targets = {
        "G_struct": (_loss(cache, "self_only", "train") - _loss(cache, "uniform", "train"),
                     _loss(cache, "self_only", "val") - _loss(cache, "uniform", "val")),
        "G_T": (_loss(cache, "text_self", "train") - _loss(cache, "text_uniform", "train"),
                _loss(cache, "text_self", "val") - _loss(cache, "text_uniform", "val")),
        "G_V": (_loss(cache, "visual_self", "train") - _loss(cache, "visual_uniform", "train"),
                _loss(cache, "visual_self", "val") - _loss(cache, "visual_uniform", "val")),
    }
    rows = []
    for target_name, (target_tr, target_va) in targets.items():
        for group_name, names in UTILITY_FEATURE_GROUPS.items():
            x_all = _node_x(cache, names)
            xtr = x_all[train].numpy(); xva = x_all[val].numpy()
            medians = np.zeros(xtr.shape[1], dtype=np.float64)
            for col in range(xtr.shape[1]):
                good = np.isfinite(xtr[:, col])
                medians[col] = np.median(xtr[good, col]) if good.any() else 0.0
            xtr = np.where(np.isfinite(xtr), xtr, medians)
            xva = np.where(np.isfinite(xva), xva, medians)
            scaler = StandardScaler().fit(xtr)
            model = Ridge(alpha=1.0).fit(scaler.transform(xtr), target_tr.numpy())
            pred = model.predict(scaler.transform(xva))
            pred_t = torch.as_tensor(pred, dtype=torch.float32)
            continuous_spearman = d2.safe_spearman(pred_t, target_va)
            try:
                r2 = float(r2_score(target_va.numpy(), pred)) if torch.var(target_va, unbiased=False) > 0 else float("nan")
            except Exception:
                r2 = float("nan")
            ysign_tr = (target_tr > 0).long().numpy()
            ysign_va = (target_va > 0).long().numpy()
            auroc, bacc = float("nan"), float("nan")
            if len(np.unique(ysign_tr)) == 2:
                clf = LogisticRegression(max_iter=1000, random_state=seed)
                clf.fit(scaler.transform(xtr), ysign_tr)
                sign_pred = clf.predict(scaler.transform(xva))
                sign_prob = clf.predict_proba(scaler.transform(xva))[:, 1]
                bacc = float(balanced_accuracy_score(ysign_va, sign_pred))
                if len(np.unique(ysign_va)) == 2:
                    auroc = float(roc_auc_score(ysign_va, sign_prob))
            rows.append({"dataset": dataset, "seed": seed, "target": target_name,
                         "feature_group": group_name, "feature_names": ",".join(names),
                         "val_spearman": continuous_spearman, "val_r2": r2,
                         "val_sign_auroc": auroc, "val_sign_balanced_accuracy": bacc,
                         "train_nodes": int(train.numel()), "validation_nodes": int(val.numel()),
                         "fit_split": "train", "evaluation_split": "validation"})
    return rows


def _utility_predictability(datasets: tuple[str, ...], seeds: tuple[int, ...] = SEEDS) -> list[dict[str, Any]]:
    rows = []
    for dataset in datasets:
        for seed in seeds:
            print(f"UTILITY_PROBE {dataset} seed={seed}", flush=True)
            rows.extend(_utility_predictability_one(_read_cache(dataset, seed)))
    _write_csv(RESULTS / "utility_predictability.csv", rows)
    return rows


def _selector_and_probe_phase(datasets: tuple[str, ...], seeds: tuple[int, ...] = SEEDS) -> None:
    _selector_analysis(datasets, seeds)
    _utility_predictability(datasets, seeds)


def _margin(logits: torch.Tensor) -> torch.Tensor:
    top = logits.topk(min(2, logits.size(-1)), dim=-1).values
    return top[:, 0] - (top[:, 1] if top.size(1) > 1 else 0)


def _model_logits_from_operator(model, head, x: torch.Tensor, operator: torch.Tensor,
                                val_idx: torch.Tensor) -> torch.Tensor:
    x = x.to(next(model.parameters()).device)
    operator = operator.to(device=x.device, dtype=x.dtype).coalesce()
    text_on = model.modality_mode in {"both", "text"}
    visual_on = model.modality_mode in {"both", "visual"}
    z_t = z_v = None
    if text_on:
        h0_t = model.text_projector(x[:, :model.text_dim])
        states_t = model._propagate(h0_t, operator, model.max_order)
        z_t = model._readout(states_t, getattr(model, "gamma_text", None))
    if visual_on:
        h0_v = model.visual_projector(x[:, model.text_dim:model.text_dim + model.visual_dim])
        states_v = model._propagate(h0_v, operator, model.max_order)
        z_v = model._readout(states_v, getattr(model, "gamma_visual", None))
    if model.modality_mode == "both":
        fused = model._fuse_modalities(z_t, z_v)[-1]
    else:
        fused = z_t if model.modality_mode == "text" else z_v
    logits = head(fused)
    if not torch.isfinite(logits).all():
        raise FloatingPointError("nonfinite logits in edge intervention")
    return logits[val_idx.to(logits.device)].detach().cpu()


def _effect(logits_val: torch.Tensor, y_val: torch.Tensor, classes: list[int]) -> dict[str, Any]:
    metrics = _soft_metrics(logits_val, y_val, classes)
    losses = d2.node_cross_entropy(logits_val, y_val)
    margins = _margin(logits_val)
    return {"val_accuracy": metrics["accuracy"], "val_macro_f1": metrics["macro_f1"],
            "val_mean_loss": float(losses.mean()), "val_mean_margin": float(margins.mean()),
            "node_loss": losses, "node_margin": margins, "pred": logits_val.argmax(-1)}


def _intervention_groups(cache: dict[str, Any], model_name: str) -> list[tuple[str, torch.Tensor]]:
    e = cache["edge_cache"]
    groups: list[tuple[str, torch.Tensor]] = []
    if model_name in {"uniform", "gpr"}:
        qkey = "semantic_q_uniform" if model_name == "uniform" else "semantic_q_gpr"
        for q in range(4):
            groups.append((f"semantic_q{q}", torch.where(e[qkey] == q)[0]))
        rkey = "sim_novelty_group" if model_name == "uniform" else "gpr_sim_novelty_group"
        prefix = "" if model_name == "uniform" else "gpr_"
        for rid, name in ((0, "high_similarity_low_novelty"), (1, "high_similarity_high_novelty"),
                          (2, "low_similarity_high_novelty"), (3, "low_similarity_low_novelty")):
            groups.append((f"{prefix}{name}", torch.where(e[rkey] == rid)[0]))
        if model_name == "uniform":
            q0 = e["semantic_q_uniform"] == 0
            embedded = e["descriptor"]["edge_embeddedness"]
            train_cut = e["train_thresholds"]["embeddedness"]
            groups.append(("low_similarity_low_embeddedness", torch.where(q0 & (embedded <= train_cut["q25"]))[0]))
            groups.append(("low_similarity_high_embeddedness", torch.where(q0 & (embedded >= train_cut["q75"]))[0]))
    elif model_name == "text_uniform":
        for q in (0, 3):
            groups.append((f"text_semantic_q{q}", torch.where(e["text_semantic_q"] == q)[0]))
        for rid, name in ((2, "text_low_similarity_high_novelty"), (0, "text_high_similarity_low_novelty")):
            groups.append((name, torch.where(e["text_sim_novelty_group"] == rid)[0]))
        for rid, name in ((2, "joint_low_similarity_high_novelty"), (0, "joint_high_similarity_low_novelty")):
            groups.append((name, torch.where(e["sim_novelty_group"] == rid)[0]))
    elif model_name == "visual_uniform":
        for q in (0, 3):
            groups.append((f"visual_semantic_q{q}", torch.where(e["visual_semantic_q"] == q)[0]))
        for rid, name in ((2, "visual_low_similarity_high_novelty"), (0, "visual_high_similarity_low_novelty")):
            groups.append((name, torch.where(e["visual_sim_novelty_group"] == rid)[0]))
        for rid, name in ((2, "joint_low_similarity_high_novelty"), (0, "joint_high_similarity_low_novelty")):
            groups.append((name, torch.where(e["sim_novelty_group"] == rid)[0]))
    return groups


def _load_intervention_model(dataset: str, seed: int, model_name: str, data):
    if model_name == "gpr":
        readout, mode, checkpoint = "gpr", "both", "gpr"
    elif model_name == "text_uniform":
        readout, mode, checkpoint = "uniform", "text", "text_uniform"
    elif model_name == "visual_uniform":
        readout, mode, checkpoint = "uniform", "visual", "visual_uniform"
    else:
        readout, mode, checkpoint = "uniform", "both", "uniform"
    cfg, payload, model, head = d2._load_on_existing_data(
        dataset, seed, readout, mode, data, checkpoint_name=checkpoint)
    model.eval(); head.eval()
    return model, head


def _intervention_summary_row(cache, model_name, group_name, mode, target_pairs,
                              effect, baseline, control_effects, matching):
    edge_utility = effect["val_mean_loss"] - baseline["val_mean_loss"]
    control_utilities = [x["val_mean_loss"] - baseline["val_mean_loss"] for x in control_effects]
    control_acc_delta = [x["val_accuracy"] - baseline["val_accuracy"] for x in control_effects]
    return {
        "dataset": cache["dataset"], "seed": cache["seed"], "readout": model_name,
        "group": group_name, "mode": mode, "target_edge_count": int(target_pairs.size(0)),
        "val_accuracy_delta": effect["val_accuracy"] - baseline["val_accuracy"],
        "val_macro_f1_delta": effect["val_macro_f1"] - baseline["val_macro_f1"],
        "edge_utility_val_loss_increase": edge_utility,
        "val_margin_delta": effect["val_mean_margin"] - baseline["val_mean_margin"],
        "matched_control_count": len(control_effects),
        "matched_control_edge_utility_mean": float(np.mean(control_utilities)) if control_utilities else float("nan"),
        "matched_control_edge_utility_population_std": float(np.std(control_utilities)) if control_utilities else float("nan"),
        "target_minus_matched_utility": edge_utility - float(np.mean(control_utilities)) if control_utilities else float("nan"),
        "matched_control_accuracy_delta_mean": float(np.mean(control_acc_delta)) if control_acc_delta else float("nan"),
        "matching_status": matching.get("status"),
        "matching_diagnostics_json": json.dumps(matching, sort_keys=True, allow_nan=True),
        "interpretation_boundary": "frozen graph intervention; FIXED_NORM_MASK is not retrained valid-operator performance",
    }


def _node_delta_rows(cache, model_name, group_name, mode, target_pairs,
                     baseline, effect, data, adjacency=None):
    if target_pairs.numel() == 0:
        return []
    val = cache["val_idx"]
    affected = affected_within_hops(data.edge_index, target_pairs.flatten(), data.num_nodes,
                                    hops=3, adjacency=adjacency)
    keep = affected[val]
    selected_positions = torch.where(keep)[0]
    if selected_positions.numel() == 0:
        return []
    y_val = cache["known_y"][val][selected_positions]
    after_loss = d2.node_cross_entropy(
        # Reconstruct per-node logits from effect was not retained; caller attaches below.
        effect["logits_val"][selected_positions], y_val)
    before_loss = effect["baseline_logits_val"]
    before_loss = d2.node_cross_entropy(before_loss[selected_positions], y_val)
    before_margin = _margin(effect["baseline_logits_val"])[selected_positions]
    after_margin = _margin(effect["logits_val"])[selected_positions]
    before_pred = effect["baseline_logits_val"][selected_positions].argmax(-1)
    after_pred = effect["logits_val"][selected_positions].argmax(-1)
    rows = []
    features = cache["node_features"]
    for local, position in enumerate(selected_positions.tolist()):
        node = int(val[position])
        row = {"dataset": cache["dataset"], "seed": cache["seed"], "node_id": node,
               "readout": model_name, "group": group_name, "mode": mode,
               "delta_loss": float(after_loss[local] - before_loss[local]),
               "delta_margin": float(after_margin[local] - before_margin[local]),
               "prediction_flip": int(before_pred[local] != after_pred[local])}
        row.update({f"feature_{key}": float(value[node]) for key, value in features.items()})
        rows.append(row)
    return rows


def _run_intervention_seed(dataset: str, seed: int, device_name: str = "cuda:0") -> None:
    cache = _read_cache(dataset, seed)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    d2.DEVICE = device
    d3.DEVICE = device
    cfg, data, payload, _, _, train, val, classes = d2._load_model_data(dataset, seed, "uniform")
    x = data.x.to(device)
    edge_index = data.edge_index.detach().long().cpu()
    y_val = cache["known_y"][val]
    all_summaries, node_rows = [], []
    pairs = cache["edge_cache"]["pairs"]
    adjacency = build_csr_adjacency(edge_index, data.num_nodes)
    for model_name in ("uniform", "gpr", "text_uniform", "visual_uniform"):
        print(f"INTERVENTION {dataset} seed={seed} model={model_name} device={device}", flush=True)
        model, head = _load_intervention_model(dataset, seed, model_name, data)
        model = model.to(device); head = head.to(device)
        baseline_operator = normalized_physical_operator(edge_index, data.num_nodes, device=device)
        baseline_logits_val = _model_logits_from_operator(model, head, x, baseline_operator, val)
        cached_baseline = cache["logits"][model_name]["val"]
        if not torch.allclose(baseline_logits_val, cached_baseline, atol=2e-4, rtol=2e-4):
            raise RuntimeError(f"intervention baseline mismatch for {dataset}/{seed}/{model_name}")
        baseline = _effect(baseline_logits_val, y_val, classes)
        model_groups = _intervention_groups(cache, model_name)
        for group_index, (group_name, target_indices) in enumerate(model_groups):
            target_indices = target_indices.detach().long().cpu().unique(sorted=True)
            target_pairs = pairs[target_indices]
            controls, matching = matched_random_edge_sets(
                pairs, target_indices, cache["edge_cache"]["degree_pair_bin"],
                cache["edge_cache"]["weight_bin"], repeats=20,
                seed=seed * 1000003 + group_index * 7919 + sum(map(ord, model_name)))
            for mode in ("FIXED_NORM_MASK", "RENORMALIZED_DELETE"):
                if mode == "FIXED_NORM_MASK":
                    target_op = fixed_norm_mask(baseline_operator, target_pairs)
                else:
                    kept = remove_pairs_vectorized(edge_index, target_pairs, data.num_nodes)
                    target_op = normalized_physical_operator(kept, data.num_nodes, device=device)
                logits_val = _model_logits_from_operator(model, head, x, target_op, val)
                effect = _effect(logits_val, y_val, classes)
                effect["logits_val"] = logits_val
                effect["baseline_logits_val"] = baseline_logits_val
                control_effects = []
                for control_pairs in controls:
                    if mode == "FIXED_NORM_MASK":
                        control_op = fixed_norm_mask(baseline_operator, control_pairs)
                    else:
                        kept = remove_pairs_vectorized(edge_index, control_pairs, data.num_nodes)
                        control_op = normalized_physical_operator(kept, data.num_nodes, device=device)
                    control_logits = _model_logits_from_operator(model, head, x, control_op, val)
                    control_effects.append(_effect(control_logits, y_val, classes))
                all_summaries.append(_intervention_summary_row(
                    cache, model_name, group_name, mode, target_pairs, effect, baseline,
                    control_effects, matching))
                if target_pairs.numel():
                    node_rows.extend(_node_delta_rows(cache, model_name, group_name, mode,
                                                      target_pairs, baseline, effect, data, adjacency=adjacency))
        del model, head
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary_path = OUT / "intervention_summaries" / f"{dataset}_seed{seed}.csv"
    _write_csv(summary_path, all_summaries)
    raw_path = OUT / "node_intervention_effects" / f"{dataset}_seed{seed}.csv.gz"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if node_rows:
        with gzip.open(raw_path, "wt", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(node_rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(node_rows)
    else:
        with gzip.open(raw_path, "wt", encoding="utf-8", newline="") as f:
            f.write("")
    print(f"INTERVENTION_COMPLETE {dataset} seed={seed} groups={len(all_summaries)} node_rows={len(node_rows)}", flush=True)


def _write_node_manifest(datasets: tuple[str, ...], seeds: tuple[int, ...]) -> None:
    rows = []
    for dataset in datasets:
        for seed in seeds:
            path = OUT / "node_intervention_effects" / f"{dataset}_seed{seed}.csv.gz"
            if not path.is_file():
                continue
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                count = max(sum(1 for _ in handle) - 1, 0)
            rows.append({"dataset": dataset, "seed": seed, "path": str(path),
                         "sha256": d3._sha256(path), "size_bytes": path.stat().st_size,
                         "rows": count})
    _write_csv(OUT / "node_intervention_effects" / "manifest.csv", rows)


def _intervention_phase(datasets: tuple[str, ...], seeds: tuple[int, ...], device_name: str) -> None:
    if device_name.startswith("gpu_ids:"):
        gpu_ids = tuple(x.strip() for x in device_name.split(":", 1)[1].split(",") if x.strip())
        jobs = [(dataset, seed) for dataset in datasets for seed in seeds]
        def run_one(job, gpu):
            dataset, seed = job
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = gpu
            command = [sys.executable, "scripts/analyze_problem_deep_dive.py", "intervention-one",
                       "--datasets", dataset, "--seeds", str(seed), "--device", "cuda:0"]
            result = subprocess.run(command, cwd=ROOT, env=env, check=False)
            if result.returncode:
                raise RuntimeError(f"intervention failed for {dataset} seed={seed} on GPU {gpu}")
            return {"dataset": dataset, "seed": seed, "gpu": gpu}
        def report(outcome: GPUJobOutcome[Any, Any]):
            print(json.dumps(outcome.result) if outcome.error is None else
                  f"FAILED {outcome.job} on GPU {outcome.gpu}: {outcome.error}", flush=True)
        summary = run_gpu_jobs(jobs, gpu_ids, run_one, report)
        if summary.failures or summary.not_started:
            raise RuntimeError(f"D3 intervention jobs failed={len(summary.failures)} not_started={len(summary.not_started)}")
    else:
        for dataset in datasets:
            for seed in seeds:
                _run_intervention_seed(dataset, seed, device_name)
    _write_node_manifest(datasets, seeds)
    _finalize_intervention_tables(datasets, seeds)


def _finalize_intervention_tables(datasets: tuple[str, ...], seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    rows = []
    for dataset in datasets:
        for seed in seeds:
            path = OUT / "intervention_summaries" / f"{dataset}_seed{seed}.csv"
            if not path.is_file():
                continue
            with path.open(newline="", encoding="utf-8") as f:
                rows.extend(dict(row) for row in csv.DictReader(f))
    semantic = [r for r in rows if r["group"].startswith("semantic_q")]
    regimes = [r for r in rows if r["group"].endswith((
        "high_similarity_low_novelty", "high_similarity_high_novelty",
        "low_similarity_high_novelty", "low_similarity_low_novelty"))]
    low_explain = [r for r in rows if "low_similarity_" in r["group"] and
                   ("embeddedness" in r["group"] or "novelty" in r["group"])]
    modality = [r for r in rows if r["readout"] in {"text_uniform", "visual_uniform"}]
    _write_csv(RESULTS / "semantic_utility_deconfounded.csv", semantic)
    _write_csv(RESULTS / "similarity_novelty_intervention.csv", regimes)
    _write_csv(RESULTS / "low_similarity_edge_explanation.csv", low_explain)
    _write_csv(RESULTS / "modality_specific_edge_utility.csv", modality)
    # Conditional utility pairs high-similarity/low-novelty against low-similarity/high-novelty.
    by_key: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for r in regimes:
        if r["readout"] not in {"uniform", "gpr"} or r["mode"] != "FIXED_NORM_MASK":
            continue
        key = (r["dataset"], r["seed"], r["readout"])
        by_key.setdefault(key, {})[r["group"]] = r
    conditional = []
    for (dataset, seed, readout), items in sorted(by_key.items()):
        c = next((v for k, v in items.items() if k.endswith("low_similarity_high_novelty")), None)
        a = next((v for k, v in items.items() if k.endswith("high_similarity_low_novelty")), None)
        if c and a:
            conditional.append({"dataset": dataset, "seed": seed, "readout": readout,
                                "complementary_candidate_edge_utility": float(c["edge_utility_val_loss_increase"]),
                                "redundant_candidate_edge_utility": float(a["edge_utility_val_loss_increase"]),
                                "candidate_utility_difference": float(c["edge_utility_val_loss_increase"]) - float(a["edge_utility_val_loss_increase"]),
                                "low_similarity_matched_control_utility": c["matched_control_edge_utility_mean"],
                                "high_similarity_matched_control_utility": a["matched_control_edge_utility_mean"],
                                "low_similarity_matching_status": c["matching_status"],
                                "high_similarity_matching_status": a["matching_status"],
                                "validation_only": True})
    _write_csv(RESULTS / "conditional_relation_utility.csv", conditional)
    return rows


def _summarize_gate_a(intervention_rows: list[dict[str, Any]], predictability: list[dict[str, Any]]) -> dict[str, Any]:
    primary_all = [r for r in intervention_rows if r["readout"] == "uniform" and
                   r["group"] == "low_similarity_high_novelty" and r["mode"] == "FIXED_NORM_MASK"]
    primary_by_ds: dict[str, list[dict[str, Any]]] = {}
    for r in primary_all:
        primary_by_ds.setdefault(r["dataset"], []).append(r)
    stable, matching_failed = [], []
    for dataset, values in primary_by_ds.items():
        if len(values) != len(SEEDS) or any(x.get("matching_status") != "MATCHED" or
                                            int(x.get("matched_control_count", 0)) != 20 for x in values):
            matching_failed.append(dataset)
            continue
        target = float(np.mean([float(x["edge_utility_val_loss_increase"]) for x in values]))
        matched = float(np.mean([float(x["matched_control_edge_utility_mean"]) for x in values]))
        contrast = float(np.mean([float(x["target_minus_matched_utility"]) for x in values]))
        if target > 0 and contrast > 0:
            stable.append(dataset)
    probe_rows = [r for r in predictability if r["target"] == "G_struct" and
                  r["feature_group"] in {"SIMILARITY_ONLY", "NOVELTY_ONLY", "ALL"}]
    by_probe: dict[str, dict[str, list[float]]] = {}
    for r in probe_rows:
        by_probe.setdefault(r["dataset"], {}).setdefault(r["feature_group"], []).append(abs(float(r["val_spearman"])))
    novelty_better = [d for d, groups in by_probe.items()
                      if len(groups.get("NOVELTY_ONLY", [])) == len(SEEDS) and
                         len(groups.get("SIMILARITY_ONLY", [])) == len(SEEDS) and
                         np.isfinite(groups["NOVELTY_ONLY"]).all() and
                         np.isfinite(groups["SIMILARITY_ONLY"]).all() and
                         np.mean(groups["NOVELTY_ONLY"]) > np.mean(groups["SIMILARITY_ONLY"])]
    mod_rows = [r for r in intervention_rows if r["readout"] in {"text_uniform", "visual_uniform"} and
                r["group"] == "joint_low_similarity_high_novelty" and
                r["mode"] == "FIXED_NORM_MASK"]
    mod_accum: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for r in mod_rows:
        group = "text" if r["readout"] == "text_uniform" else "visual"
        mod_accum.setdefault(r["dataset"], {}).setdefault(group, []).append(r)
    modality_diff, modality_matching_failed = [], []
    for dataset, x in mod_accum.items():
        text_rows, visual_rows = x.get("text", []), x.get("visual", [])
        if (len(text_rows) != len(SEEDS) or len(visual_rows) != len(SEEDS) or
                any(r.get("matching_status") != "MATCHED" or int(r.get("matched_control_count", 0)) != 20
                    for r in text_rows + visual_rows)):
            modality_matching_failed.append(dataset)
            continue
        text_utility = np.mean([float(r["edge_utility_val_loss_increase"]) for r in text_rows])
        visual_utility = np.mean([float(r["edge_utility_val_loss_increase"]) for r in visual_rows])
        if abs(text_utility - visual_utility) > 1e-4:
            modality_diff.append(dataset)
    checks = {"fixed_mask_degree_weight_matched_utility": len(stable) >= 3,
              "novelty_predictability_exceeds_similarity": len(novelty_better) >= 3,
              "modality_specific_utility_differs": len(modality_diff) >= 3}
    status = "STRONG_SUPPORT" if all(checks.values()) else "MIXED" if any(checks.values()) or matching_failed or modality_matching_failed else "UNSUPPORTED"
    return {"status": status, "checks": checks, "datasets_low_similarity_high_novelty_positive_vs_matched": stable,
            "datasets_matching_failed_or_incomplete": matching_failed,
            "datasets_novelty_predictability_gt_similarity": novelty_better,
            "datasets_text_visual_utility_different": modality_diff,
            "datasets_modality_matching_failed_or_incomplete": modality_matching_failed}


def _aggregate_selector_gate(rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    learned = [r for r in rows if r["family"] == family and r["method"] == "learned_logistic_train_only" and
               r.get("selector_status", "FITTED") == "FITTED"]
    gains, headrooms = [], []
    per_ds = {}
    for dataset in sorted({r["dataset"] for r in learned}):
        selected = [r for r in learned if r["dataset"] == dataset]
        g = [float(r["val_gain_vs_best_simple_reference"]) for r in selected]
        h = [float(r["val_oracle_headroom_vs_best_simple"]) for r in selected]
        if g:
            gain = float(np.mean(g)); gains.append(gain)
            per_ds[dataset] = {
                "learned_minus_best_train_selected_simple_reference": gain,
                "positive": gain > 0,
                "best_simple_reference_methods": sorted({r["best_simple_reference_method"] for r in selected}),
                "oracle_headroom": float(np.mean(h)) if h else float("nan"),
            }
            headrooms.extend(h)
    mean_gain = float(np.mean(gains)) if gains else float("nan")
    positive_ds = int(sum(value["positive"] for value in per_ds.values()))
    oracle_headroom = float(np.mean(headrooms)) if headrooms else float("nan")
    clear_headroom = bool(np.isfinite(oracle_headroom) and oracle_headroom >= mean_gain + 0.003)
    strong = bool(np.isfinite(mean_gain) and mean_gain >= 0.003 and positive_ds >= 3 and clear_headroom)
    return {"status": "STRONG_SUPPORT" if strong else "MIXED" if mean_gain > 0 or oracle_headroom > 0 else "UNSUPPORTED",
            "mean_gain_over_best_train_selected_simple_reference": mean_gain,
            "positive_datasets": positive_ds, "oracle_headroom_mean": oracle_headroom,
            "oracle_headroom_rule": "oracle headroom >= learned gain + 0.003 accuracy",
            "per_dataset": per_ds}


def _all_intervention_rows() -> list[dict[str, Any]]:
    rows = []
    root = OUT / "intervention_summaries"
    if root.is_dir():
        for path in sorted(root.glob("*_seed*.csv")):
            with path.open(newline="", encoding="utf-8") as f:
                rows.extend(dict(row) for row in csv.DictReader(f))
    # parse numeric columns written as CSV strings
    for row in rows:
        for key in ("edge_utility_val_loss_increase", "matched_control_edge_utility_mean", "target_minus_matched_utility"):
            try: row[key] = float(row[key])
            except (TypeError, ValueError): row[key] = float("nan")
    return rows


def _final_report(operator: dict[str, Any], topology: list[dict[str, Any]],
                  selectors: list[dict[str, Any]], predictability: list[dict[str, Any]]) -> dict[str, Any]:
    intervention = _all_intervention_rows()
    gate_a = _summarize_gate_a(intervention, predictability)
    gate_b = _aggregate_selector_gate(selectors, "self_structure")
    gate_c = _aggregate_selector_gate(selectors, "modality")
    topo_by_ds = {}
    for dataset in DATASETS:
        values = [float(r["real_minus_rewired_val_accuracy"]) for r in topology if r["dataset"] == dataset]
        if values:
            topo_by_ds[dataset] = {"mean": float(np.mean(values)),
                                   "population_std": float(np.std(values, ddof=0)),
                                   "runs": len(values)}

    p8 = {}
    for dataset in DATASETS:
        values = [r for r in intervention if r["dataset"] == dataset and r["readout"] == "uniform" and
                  r["group"] == "low_similarity_high_novelty" and r["mode"] == "FIXED_NORM_MASK"]
        if values:
            p8[dataset] = {
                "target_loss_increase_mean": float(np.mean([float(r["edge_utility_val_loss_increase"]) for r in values])),
                "matched_control_loss_increase_mean": float(np.mean([float(r["matched_control_edge_utility_mean"]) for r in values])),
                "target_minus_control_mean": float(np.mean([float(r["target_minus_matched_utility"]) for r in values])),
                "matched_runs": sum(r.get("matching_status") == "MATCHED" and int(r.get("matched_control_count", 0)) == 20 for r in values),
            }
    modality_utility = {}
    for dataset in DATASETS:
        modality_utility[dataset] = {}
        for name in ("text_uniform", "visual_uniform"):
            values = [r for r in intervention if r["dataset"] == dataset and r["readout"] == name and
                      r["group"] == "joint_low_similarity_high_novelty" and r["mode"] == "FIXED_NORM_MASK" and
                      r.get("matching_status") == "MATCHED"]
            if values:
                modality_utility[dataset][name] = float(np.mean([float(r["edge_utility_val_loss_increase"]) for r in values]))
        if modality_utility[dataset]:
            vals = list(modality_utility[dataset].values())
            modality_utility[dataset]["text_minus_visual"] = (
                modality_utility[dataset]["text_uniform"] - modality_utility[dataset]["visual_uniform"]
                if "text_uniform" in modality_utility[dataset] and "visual_uniform" in modality_utility[dataset] else float("nan"))

    utility_scores = {}
    for group in ("SIMILARITY_ONLY", "TOPOLOGY_ONLY", "CONFIDENCE_ONLY", "NOVELTY_ONLY", "MODAL_DISAGREEMENT_ONLY", "ALL"):
        values = [r for r in predictability if r["target"] == "G_struct" and r["feature_group"] == group]
        def finite_mean(field: str):
            items = [float(r[field]) for r in values if np.isfinite(float(r.get(field, float("nan"))))]
            return float(np.mean(items)) if items else float("nan")
        utility_scores[group] = {"validation_spearman_mean": finite_mean("val_spearman"),
                                 "validation_r2_mean": finite_mean("val_r2"),
                                 "sign_auroc_mean": finite_mean("val_sign_auroc"),
                                 "finite_spearman_runs": int(sum(bool(np.isfinite(float(r.get("val_spearman", float("nan")))) ) for r in values))}

    roots = {
        "preflight_jobs": len(list((OUT / "preflight").glob("*/*/complete.json"))),
        "operator_transfer_jobs": len(list((OUT / "operator_transfer").glob("*/*/complete.json"))),
        "rewired_uniform_jobs": len(list((OUT / "rewired_topology").glob("*/*/complete.json"))),
    }
    raw_manifest = OUT / "node_intervention_effects" / "manifest.csv"
    RESULTS.mkdir(parents=True, exist_ok=True)
    committed_manifest = RESULTS / "node_intervention_effects_manifest.csv"
    committed_manifest.write_bytes(raw_manifest.read_bytes())
    payload = {
        "study": "D3 Principle Generalization & Problem Causal Deep-Dive",
        "source_commit": "d31b095faa4f98538e60cfac293a831138dde3db",
        "datasets": list(DATASETS), "task": "NC only",
        "analysis_splits": ["train", "validation"],
        "test_usage": "descriptive metrics only for newly retrained formal controls; no research decision uses test",
        "formal_training": {**roots, "formal_runs": roots["operator_transfer_jobs"] * 3 + roots["rewired_uniform_jobs"] * 3,
                            "internal_seeds": list(SEEDS), "completed": roots["operator_transfer_jobs"] == 20 and roots["rewired_uniform_jobs"] == 5},
        "gate_A_conditional_relational_utility": gate_a,
        "gate_B_adaptive_self_structure": gate_b,
        "gate_C_adaptive_modality_arbitration": gate_c,
        "anchoring_transfer": operator,
        "P8_fixed_mask_low_similarity_high_novelty": p8,
        "modality_specific_fixed_mask_utility": modality_utility,
        "utility_predictability_G_struct": utility_scores,
        "real_minus_rewired_val_accuracy_by_dataset": topo_by_ds,
        "intervention_semantics": {
            "FIXED_NORM_MASK": "frozen normalized operator with target message weights zeroed; no renormalization and no retraining",
            "RENORMALIZED_DELETE": "remove target undirected pairs and rebuild the symmetric normalized operator; frozen checkpoint",
        },
        "thresholds": "semantic, novelty, and embeddedness thresholds are derived from edges whose endpoints are both in train split",
        "raw_node_effect_manifest": str(committed_manifest.relative_to(ROOT)),
        "raw_node_effect_data_root": str((OUT / "node_intervention_effects").relative_to(ROOT)),
    }
    d3._write_json(RESULTS / "problem_deep_dive_summary.json", payload)

    gcn = operator.get("operators", {}).get("gcn", {})
    sage = operator.get("operators", {}).get("sage", {})
    operator_lines = []
    for name, item in (("GCN", gcn), ("SAGE", sage)):
        operator_lines.append(f"| {name} | {100*float(item.get('mean_gain', float('nan'))):+.3f} pp | {item.get('positive_datasets', 0)}/5 |")
    p8_lines = []
    for dataset in DATASETS:
        item = p8.get(dataset)
        if item:
            p8_lines.append(f"| {dataset} | {item['target_loss_increase_mean']:+.4f} | {item['matched_control_loss_increase_mean']:+.4f} | {item['target_minus_control_mean']:+.4f} | {item['matched_runs']}/3 |")
    modality_lines = []
    for dataset in DATASETS:
        item = modality_utility.get(dataset, {})
        if "text_uniform" in item and "visual_uniform" in item:
            modality_lines.append(f"| {dataset} | {item['text_uniform']:+.4f} | {item['visual_uniform']:+.4f} | {item['text_minus_visual']:+.4f} |")
    topology_lines = []
    for dataset, item in topo_by_ds.items():
        topology_lines.append(f"| {dataset} | {100*item['mean']:+.3f} pp | {100*item['population_std']:.3f} pp |")

    novelty_ds = ", ".join(gate_a["datasets_novelty_predictability_gt_similarity"]) or "none"
    modality_ds = ", ".join(gate_a["datasets_text_visual_utility_different"]) or "none"
    supported_list = gate_a["datasets_low_similarity_high_novelty_positive_vs_matched"]
    supported_ds = ", ".join(supported_list) or "none"
    lines = [
        "# D3 problem deep-dive report", "",
        "Scope: Movies, Toys, Grocery, ele-fashion, Reddit-S; node classification only.",
        "All selector fits, thresholds, edge grouping, intervention ranking, and gates use Train/Validation only.",
        "Test metrics appear only as descriptive fields for the formal retrained controls; no research decision uses Test.",
        f"Formal controls completed: {payload['formal_training']['operator_transfer_jobs']} operator jobs × 3 seeds + {payload['formal_training']['rewired_uniform_jobs']} rewired jobs × 3 seeds = {payload['formal_training']['formal_runs']} runs.", "",
        "## Frozen questions", "",
        "1. **Does attribute anchoring transfer beyond fixed GPR-style diffusion?**",
        f"   No under the registered transfer gate: `{operator.get('gate', 'NOT_RUN')}`. The mean anchored25−deep_only validation gain was GCN {100*float(gcn.get('mean_gain', float('nan'))):+.3f} pp (positive in {gcn.get('positive_datasets',0)}/5) and SAGE {100*float(sage.get('mean_gain', float('nan'))):+.3f} pp (positive in {sage.get('positive_datasets',0)}/5).",
        "   | Operator | Mean paired gain | Positive datasets |", "   |---|---:|---:|", *operator_lines,
        "2. **Does conditional relation utility remain after degree and normalized-weight matching?**",
        f"   Registered Gate A: `{gate_a['status']}`. Under FIXED_NORM_MASK, the low-similarity/high-novelty group had positive validation cross-entropy increase beyond 20 exact matched controls in {len(supported_list)}/5 datasets ({supported_ds}); ele-fashion did not. This is a frozen functional diagnostic, not retrained valid-operator performance.",
        "   | Dataset | Target ΔCE | Matched-control ΔCE | Target−control ΔCE | Matched seeds |", "   |---|---:|---:|---:|---:|", *p8_lines,
        f"   Novelty-only validation utility prediction exceeded similarity-only in {len(gate_a['datasets_novelty_predictability_gt_similarity'])}/5 datasets ({novelty_ds}); the average correlations remain small. Gate A also required a modality difference.",
        "3. **What best describes low-similarity useful edges?**",
        "   The fixed-mask low-similarity/high-novelty category is the strongest consistent candidate: it was positive against matched controls in four datasets. The separate embeddedness splits do not show a consistent cross-dataset ordering, so these data do not identify structural bridges or establish semantic complementarity. See `low_similarity_edge_explanation.csv` and `conditional_relation_utility.csv`.",
        "4. **Is relation utility modality-dependent?**",
        f"   Text and Visual fixed-mask utility means differed for the same joint low-similarity/high-novelty edges in {len(modality_ds.split(', '))}/5 datasets ({modality_ds}); the stronger modality changed by dataset.",
        "   | Dataset | U_text ΔCE | U_visual ΔCE | U_text−U_visual ΔCE |", "   |---|---:|---:|---:|", *modality_lines,
        "5. **Can self-versus-structure utility be predicted from observable node signals?**",
        f"   Gate B: `{gate_b['status']}`. The train-only logistic selector averaged {100*float(gate_b.get('mean_gain_over_best_train_selected_simple_reference', float('nan'))):+.3f} pp against the strongest Train-selected simple reference, with positive dataset means in {gate_b.get('positive_datasets',0)}/5. Oracle headroom averaged {100*float(gate_b.get('oracle_headroom_mean', float('nan'))):.3f} pp, so a simple selector did not realize the available oracle gain.",
        "6. **Can modality reliability be predicted?**",
        f"   Gate C: `{gate_c['status']}`. The modality selector averaged {100*float(gate_c.get('mean_gain_over_best_train_selected_simple_reference', float('nan'))):+.3f} pp against its strongest simple reference, positive in {gate_c.get('positive_datasets',0)}/5. See `modality_selector.csv` and `modality_oracle.csv`.",
        f"   Across G_struct utility probes, mean validation Spearman was SIMILARITY_ONLY {utility_scores['SIMILARITY_ONLY']['validation_spearman_mean']:.3f}, NOVELTY_ONLY {utility_scores['NOVELTY_ONLY']['validation_spearman_mean']:.3f}, TOPOLOGY_ONLY {utility_scores['TOPOLOGY_ONLY']['validation_spearman_mean']:.3f}; mean R² values were negative. These are weak predictive signals, not evidence for a usable routing rule.",
        "7. **Does the real topology outperform degree-preserving rewiring?**",
        "   Yes on Validation in all five datasets; real−rewired paired accuracy differences are:",
        "   | Dataset | Mean real−rewired | Population std |", "   |---|---:|---:|", *topology_lines,
        "8. **Which areas reached their preregistered gate?**",
        f"   Conditional relation utility: `{gate_a['status']}`; adaptive self/structure: `{gate_b['status']}`; adaptive modality arbitration: `{gate_c['status']}`; anchoring transfer: `{operator.get('gate', 'NOT_RUN')}`. The evidence does not automatically freeze a model design or paper claim.",
        "", "## Boundaries", "",
        "Gate A requires 3/5 datasets with positive low-similarity/high-novelty utility beyond exact controls under FIXED_NORM_MASK, novelty-only validation predictability above similarity-only in 3/5, and text/visual utility differences in 3/5. Gate B/C require at least +0.30 pp mean accuracy over the strongest non-learned simple reference selected by Train accuracy, positive gains in 3/5 datasets, and oracle headroom at least 0.30 pp larger than learned gain.",
        "`FIXED_NORM_MASK` and `RENORMALIZED_DELETE` are frozen-checkpoint interventions; neither is a trained model comparison. No node-level p-values are used. Same-label edge values are descriptive only and are excluded from selectors, features, and group construction.",
        "", "## Output map", "",
        "Operator: `operator_transfer.csv`, `operator_transfer_paired.csv`, `operator_transfer_report.md`; interventions: `semantic_utility_deconfounded.csv`, `similarity_novelty_intervention.csv`, `low_similarity_edge_explanation.csv`, `conditional_relation_utility.csv`, `modality_specific_edge_utility.csv`; selectors: `self_structure_oracle.csv`, `self_structure_selector.csv`, `modality_oracle.csv`, `modality_selector.csv`, `utility_predictability.csv`; topology: `topology_reality_check.csv`, `topology_reality_check_summary.csv`.",
        "Raw per-node deltas are gzip-compressed under `outputs/problem_deep_dive_v1/node_intervention_effects/`; their SHA-256 manifest is committed as `node_intervention_effects_manifest.csv`. No raw node table is committed.", "",
    ]
    (RESULTS / "problem_deep_dive_report.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def run_analysis_phase(phase: str, datasets: tuple[str, ...], seeds: tuple[int, ...], device: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    if phase == "selectors":
        _selector_and_probe_phase(datasets, seeds)
        return
    if phase == "interventions":
        _intervention_phase(datasets, seeds, device)
        return
    if phase == "intervention-one":
        if len(datasets) != 1 or len(seeds) != 1:
            raise ValueError("intervention-one requires one dataset and one seed")
        _run_intervention_seed(datasets[0], seeds[0], device)
        return
    if phase == "finalize":
        _finalize_intervention_tables(datasets, seeds)
        operator = d3._operator_transfer_analysis()
        topology = d3._topology_analysis()
        selectors, learned = [], []
        for family_file in (RESULTS / "self_structure_oracle.csv", RESULTS / "modality_oracle.csv"):
            if family_file.is_file():
                with family_file.open(newline="", encoding="utf-8") as f:
                    selectors.extend(dict(r) for r in csv.DictReader(f))
        for family_file in (RESULTS / "self_structure_selector.csv", RESULTS / "modality_selector.csv"):
            if family_file.is_file():
                with family_file.open(newline="", encoding="utf-8") as f:
                    learned.extend(dict(r) for r in csv.DictReader(f))
        selectors.extend(learned)
        for row in selectors:
            for key in ("val_selected_accuracy", "val_oracle_headroom_vs_static",
                        "val_gain_vs_best_simple_reference", "val_oracle_headroom_vs_best_simple"):
                if key in row:
                    row[key] = float(row[key])
        predictability = []
        path = RESULTS / "utility_predictability.csv"
        if path.is_file():
            with path.open(newline="", encoding="utf-8") as f:
                predictability = list(csv.DictReader(f))
            for row in predictability:
                for key in ("val_spearman", "val_r2"):
                    row[key] = float(row[key])
        _final_report(operator, topology, selectors, predictability)
        return
    raise ValueError(phase)
