from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATASETS = ("Movies", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
VARIANTS = (
    "s45_identity_uniform",
    "s45_unconstrained_entry_uniform",
    "s45_masspres_entry_uniform",
)
CALIBRATED = VARIANTS[1:]
TRAINING_COMMIT = "3693d351d2d722dde2d2d34c421ddb342440d659"
SOURCE_BRANCH = "s45_relcal_statepres"
SOURCE_SHA = "2b4a909c1bd4501a70905c4e9c121a6948bd4840"
PROTOCOL = "unified_full_graph_nc_v1"
OUT = ROOT / "results/s46_state_relation_context_v1"
CKPT_ROOT = ROOT / "outputs/s45_relcal_statepres_v1/formal"
HIST_ROOT = ROOT / "outputs/mechanism_discovery_v1/unimodal"
ALPHAS = (0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0)
MODALITY_ALPHA_PAIRS = ((.25, .25), (.50, .25), (.25, .50), (.50, .50), (.75, .25), (.25, .75))
STATE_NAMES = ("S0", "S1", "S2", "S3", "P")
REL_FEATURES = (
    "log_degree", "base_offdiag_mass", "centered_log_dosage", "weighted_gate_std",
    "gate_CV", "normalized_entropy", "max_contribution_share",
    "effective_neighbor_count", "effective_neighbor_ratio", "weighted_within_gate_variance",
)
PREDICTOR_FEATURES = {
    "P0_TOPOLOGY": ("log_degree", "base_offdiag_mass"),
    "P1_DOSAGE": ("centered_log_dosage",),
    "P2_RELATION_SET": REL_FEATURES,
}
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 1e-12
THREADPOOL_LIMIT = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=True), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, allow_nan=True, default=_json_default), encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def canonical_variant_conditions(variant: str) -> tuple[str, ...]:
    return ("IDENTITY",) if variant == VARIANTS[0] else ("NORMAL", "OFF")


def checkpoint_path(dataset: str, seed: int, variant: str) -> Path:
    return CKPT_ROOT / dataset / variant / f"best_run{SEEDS.index(seed) + 1}.pt"


def audit_checkpoints() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            completion_path = CKPT_ROOT / dataset / variant / "complete.json"
            if not completion_path.is_file():
                raise FileNotFoundError(completion_path)
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            if (completion.get("training_branch") != SOURCE_BRANCH or
                    completion.get("training_commit") != TRAINING_COMMIT or
                    completion.get("test_evaluation") is not False or
                    completion.get("selection") != "best_val_accuracy" or
                    completion.get("run_seeds") != list(SEEDS)):
                raise ValueError(f"Invalid frozen-source completion metadata: {completion_path}")
            for seed in SEEDS:
                path = checkpoint_path(dataset, seed, variant)
                if not path.is_file():
                    raise FileNotFoundError(path)
                payload = torch.load(path, map_location="cpu", weights_only=False)
                if (payload.get("task") != "nc" or payload.get("protocol_version") != PROTOCOL or
                        payload.get("seed") != seed or payload.get("selection") != "best_val_accuracy" or
                        any(str(k).startswith("test_") for k in payload.get("metrics", {}))):
                    raise ValueError(f"Invalid checkpoint metadata: {path}")
                if payload.get("run_metadata", {}).get("training_commit", TRAINING_COMMIT) != TRAINING_COMMIT:
                    raise ValueError(f"Checkpoint training commit mismatch: {path}")
                rows.append({
                    "dataset": dataset, "seed": seed, "variant": variant,
                    "path": str(path.relative_to(ROOT)), "checkpoint_sha256": sha256(path),
                    "training_branch": completion["training_branch"],
                    "training_commit": completion["training_commit"],
                    "selection": payload["selection"], "epoch": payload.get("epoch"),
                    "test_evaluation": False, "test_metrics_present": False,
                    "completion_record": str(completion_path.relative_to(ROOT)),
                })
    if len(rows) != len(DATASETS) * len(SEEDS) * len(VARIANTS):
        raise AssertionError("checkpoint audit row count mismatch")
    return rows


def compose_cfg(dataset: str, seed: int, model_name: str, variant: str | None = None,
                readout: str | None = None, modality_mode: str = "both"):
    from hydra import compose, initialize_config_dir
    overrides = [f"dataset={dataset}", "task=nc", f"model={model_name}", f"seed={seed}",
                "num_runs=1", f"device={DEVICE}", "task.evaluate_test=false",
                "task.training_mode=full_graph", f"task.protocol_version={PROTOCOL}"]
    if variant is not None:
        overrides.append(f"model.variant={variant}")
    if readout is not None:
        overrides.extend([f"model.readout={readout}", f"model.modality_mode={modality_mode}",
                          "model.fusion_mode=plain_mlp"])
    with initialize_config_dir(config_dir=str((ROOT / "configs").resolve()), version_base=None):
        return compose(config_name="config", overrides=overrides)


def load_dataset(dataset: str, seed: int):
    from src.data import load_mag_data
    cfg = compose_cfg(dataset, seed, "relcal_statepres_pilot", VARIANTS[1])
    data = load_mag_data(cfg, "nc", seed)
    if data.y is None or data.train_idx is None or data.val_idx is None:
        raise ValueError(f"Missing NC train/validation data: {dataset}/{seed}")
    # Label access is deliberately restricted to Train and Validation. In particular,
    # do not derive the probe class set from the Test split.
    train = data.train_idx.detach().cpu().long()
    val = data.val_idx.detach().cpu().long()
    y_train = data.y[train].detach().cpu().long()
    y_val = data.y[val].detach().cpu().long()
    valid = torch.cat((y_train, y_val))
    classes = sorted(set(int(v) for v in valid.tolist() if 0 <= int(v) < int(data.num_classes)))
    if not classes:
        raise ValueError(f"No observed Train/Validation labels: {dataset}/{seed}")
    if train.numel() == 0 or val.numel() == 0 or torch.isin(train, val).any():
        raise ValueError(f"Invalid supervised split: {dataset}/{seed}")
    return data, train, val, y_train.numpy(), y_val.numpy(), classes


def load_s45_model(dataset: str, seed: int, variant: str, data):
    from torch import nn
    from src.models import build_model
    path = checkpoint_path(dataset, seed, variant)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    info = {"input_dim": data.input_dim, "num_nodes": data.num_nodes,
            "num_classes": data.num_classes, "text_dim": int(data.x_t.size(1)),
            "visual_dim": int(data.x_i.size(1))}
    if payload.get("data_info") != info:
        raise ValueError(f"Checkpoint/data shape mismatch: {path}")
    cfg = compose_cfg(dataset, seed, "relcal_statepres_pilot", variant)
    model = build_model(cfg, info).to(DEVICE)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    head = nn.Linear(model.out_dim, int(data.num_classes)).to(DEVICE)
    head.load_state_dict(payload["head_state"], strict=True)
    head.eval()
    return model, head, path


@torch.no_grad()
def extract_s45(model, data, condition: str) -> tuple[dict[str, list[torch.Tensor]], dict[str, Any]]:
    edge = data.edge_index.to(DEVICE)
    x = data.x.to(DEVICE)
    result = model.analyze(x, edge, gate_override="off" if condition == "OFF" else None)
    states = {
        "text": [s.detach().float().cpu() for s in result["S_text"]],
        "visual": [s.detach().float().cpu() for s in result["S_visual"]],
    }
    return states, result


def p_state(states: list[torch.Tensor]) -> torch.Tensor:
    p = (states[1] + states[2] + states[3]) / 3.0
    if not torch.equal(p, (states[1] + states[2] + states[3]) / 3.0):
        raise AssertionError("P is not exactly the mean of S1, S2, S3")
    return p


def composition_state(states: list[torch.Tensor], alpha: float,
                      uniform_readout: torch.Tensor,
                      propagated: torch.Tensor | None = None) -> torch.Tensor:
    if propagated is None:
        propagated = p_state(states)
    mixed = alpha * states[0] + (1.0 - alpha) * propagated
    if abs(alpha - 0.25) < 1e-12:
        if not torch.allclose(mixed, uniform_readout, atol=1e-5, rtol=1e-6):
            err = float((mixed - uniform_readout).abs().max().item())
            raise AssertionError(f"alpha=.25 is not algebraically equivalent to uniform readout: {err}")
        # Reuse the historical mean so the equivalent composition baseline is bit-identical.
        return uniform_readout
    return mixed


def standardize_pair(x_train: np.ndarray, x_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler(copy=True)
    xt = scaler.fit_transform(x_train).astype(np.float32, copy=False)
    xv = scaler.transform(x_eval).astype(np.float32, copy=False)
    if not np.isfinite(xt).all() or not np.isfinite(xv).all():
        raise FloatingPointError("Nonfinite standardized probe features")
    return xt, xv, scaler


def expand_probabilities(model: LogisticRegression, raw: np.ndarray, classes: list[int]) -> np.ndarray:
    out = np.zeros((raw.shape[0], int(max(classes)) + 1), dtype=np.float64)
    for j, label in enumerate(model.classes_):
        out[:, int(label)] = raw[:, j]
    # sklearn normalizes over classes observed in the fit; preserve that mass.
    return out


def fit_logistic(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray,
                 y_eval: np.ndarray, classes: list[int]) -> dict[str, Any]:
    if x_train.shape[0] != y_train.shape[0] or x_eval.shape[0] != y_eval.shape[0]:
        raise ValueError("Probe feature/label row mismatch")
    xt, xv, _scaler = standardize_pair(x_train, x_eval)
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                             class_weight=None, random_state=0)
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=THREADPOOL_LIMIT):
        warnings.simplefilter("always")
        clf.fit(xt, y_train)
        raw = clf.predict_proba(xv)
    probs = expand_probabilities(clf, raw, classes)
    pred = np.argmax(probs, axis=1)
    safe = np.clip(probs[np.arange(y_eval.size), y_eval], 1e-15, 1.0)
    return {
        "accuracy": float(accuracy_score(y_eval, pred)),
        "macro_f1": float(f1_score(y_eval, pred, labels=classes, average="macro", zero_division=0)),
        "true_label_ce": float(-np.log(safe).mean()),
        "probs": probs, "pred": pred,
        "converged": not any(issubclass(w.category, Warning) and "convergence" in str(w.message).lower()
                             for w in caught),
        "n_iter": int(np.max(clf.n_iter_)),
    }


def logistic_rows(dataset: str, seed: int, variant: str, condition: str, modality: str,
                  state: str, x_train: np.ndarray, x_val: np.ndarray, y_train: np.ndarray,
                  y_val: np.ndarray, classes: list[int], result_rows: list[dict[str, Any]]) -> dict[str, Any]:
    fitted = fit_logistic(x_train, y_train, x_val, y_val, classes)
    row = {"dataset": dataset, "seed": seed, "variant": variant, "condition": condition,
           "modality": modality, "state": state, "val_accuracy": fitted["accuracy"],
           "val_macro_f1": fitted["macro_f1"], "val_true_label_ce": fitted["true_label_ce"],
           "converged": fitted["converged"], "n_iter": fitted["n_iter"],
           "classifier": "sklearn.LogisticRegression", "C": 1.0, "solver": "lbfgs",
           "max_iter": 2000, "class_weight": "None", "random_state": 0,
           "scaler_fit_split": "Train", "validation_used_for_fit": False}
    result_rows.append(row)
    return fitted


def state_matrix(states: dict[str, list[torch.Tensor]], modality: str, key: str) -> torch.Tensor:
    values = states[modality]
    return p_state(values) if key == "P" else values[int(key[1])]


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64, copy=False); y = y.astype(np.float64, copy=False)
    x = x - x.mean(axis=0, keepdims=True); y = y - y.mean(axis=0, keepdims=True)
    xy = x.T @ y
    denom = np.linalg.norm(x.T @ x, "fro") * np.linalg.norm(y.T @ y, "fro")
    return float(np.square(np.linalg.norm(xy, "fro")) / denom) if denom > 0 else math.nan


def alignment_diagnostics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    ac = a.astype(np.float64, copy=False) - a.mean(0, keepdims=True)
    bc = b.astype(np.float64, copy=False) - b.mean(0, keepdims=True)
    cosine_a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), EPS)
    cosine_b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), EPS)
    node_cos = np.sum(cosine_a * cosine_b, axis=1)
    u, _s, vh = np.linalg.svd(ac.T @ bc, full_matrices=False)
    aligned = ac @ (u @ vh)
    denom = np.linalg.norm(ac, "fro") + EPS
    return {
        "linear_cka": linear_cka(a, b),
        "mean_node_cosine": float(node_cos.mean()),
        "centered_procrustes_distance": float(np.linalg.norm(aligned - bc, "fro") / denom),
        "normalized_l2_difference": float(np.linalg.norm(a - b, "fro") / (np.linalg.norm(a, "fro") + EPS)),
    }


def residualize_train_only(s0_train: np.ndarray, s0_eval: np.ndarray,
                           k_train: np.ndarray, k_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ridge residuals after Train-only independent standardization and fit."""
    sx, sxe, _ = standardize_pair(s0_train, s0_eval)
    ky, kye, _ = standardize_pair(k_train, k_eval)
    model = Ridge(alpha=1.0, fit_intercept=True)
    with threadpool_limits(limits=THREADPOOL_LIMIT):
        model.fit(sx, ky)
    r_train = ky - model.predict(sx)
    r_eval = kye - model.predict(sxe)
    return r_train.astype(np.float32, copy=False), r_eval.astype(np.float32, copy=False)


def degree_bins(degree: np.ndarray, n_bins: int = 10) -> np.ndarray:
    if degree.size == 0:
        return np.empty(0, dtype=np.int64)
    edges = np.quantile(degree, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(edges, degree, side="right").astype(np.int64)


def shuffle_within_degree_bins(values: np.ndarray, degree: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    bins = degree_bins(degree)
    result = values.copy()
    for b in np.unique(bins):
        idx = np.flatnonzero(bins == b)
        if idx.size > 1:
            result[idx] = values[rng.permutation(idx)]
    return result


def class_prob_ce(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(probs[np.arange(labels.size), labels], 1e-15, 1.0))


def metrics_from_probabilities(probs: np.ndarray, labels: np.ndarray, classes: list[int]) -> dict[str, float]:
    pred = probs.argmax(1)
    return {"accuracy": float(accuracy_score(labels, pred)),
            "macro_f1": float(f1_score(labels, pred, labels=classes, average="macro", zero_division=0)),
            "true_label_ce": float(class_prob_ce(probs, labels).mean())}



def extract_relation_features(model, result: dict[str, Any], n_nodes: int, train_nodes: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Compute label-free per-node features from the unconstrained NORMAL operator."""
    if not model.calibrated or model.mass_preserving:
        raise ValueError("Relation utility features are restricted to unconstrained checkpoints")
    p_rel = result["P_rel"].coalesce()
    row = p_rel.indices()[0].long()
    base = p_rel.values().double()
    degree = torch.bincount(row, minlength=n_nodes).double()
    base_mass = torch.zeros(n_nodes, dtype=torch.float64, device=row.device).index_add_(0, row, base)
    features: dict[str, np.ndarray] = {}
    eps = 1e-12
    for modality in ("text", "visual"):
        gate = result["raw_gates"][modality].reshape(-1).double()
        weighted_gate = base * gate
        weighted_sum = torch.zeros_like(base_mass).index_add_(0, row, weighted_gate)
        dosage = torch.where(base_mass > 0, weighted_sum / base_mass.clamp_min(eps), torch.ones_like(base_mass))
        centered_log = torch.log(dosage + eps)
        train_idx = torch.as_tensor(train_nodes, dtype=torch.long, device=centered_log.device)
        train_median = float(torch.quantile(centered_log[train_idx], 0.5).item()) if train_idx.numel() else 0.0
        centered_log = centered_log - train_median
        variance_sum = torch.zeros_like(base_mass).index_add_(0, row, base * (gate - dosage[row]).square())
        weighted_var = torch.where(base_mass > 0, variance_sum / base_mass.clamp_min(eps), torch.zeros_like(base_mass))
        weighted_std = torch.sqrt(weighted_var.clamp_min(0))
        gate_cv = weighted_std / (dosage.abs() + eps)
        q = torch.where(weighted_sum[row] > 0, weighted_gate / weighted_sum[row].clamp_min(eps), torch.zeros_like(weighted_gate))
        entropy_edge = -q * torch.log(q.clamp_min(eps))
        entropy = torch.zeros_like(base_mass).index_add_(0, row, entropy_edge)
        norm_entropy = torch.where(degree > 1, entropy / torch.log(degree.clamp_min(2)), torch.zeros_like(entropy))
        max_share = torch.zeros_like(base_mass)
        max_share.scatter_reduce_(0, row, q, reduce="amax", include_self=True)
        concentration = torch.zeros_like(base_mass).index_add_(0, row, q.square())
        effective = torch.where(concentration > 0, concentration.reciprocal(), torch.zeros_like(concentration))
        ratio = torch.where(degree > 0, effective / degree.clamp_min(1), torch.zeros_like(effective))
        values = {"degree": degree, "log_degree": torch.log1p(degree), "base_offdiag_mass": base_mass,
                  "dosage": dosage, "centered_log_dosage": centered_log,
                  "weighted_gate_std": weighted_std, "gate_CV": gate_cv,
                  "normalized_entropy": norm_entropy, "max_contribution_share": max_share,
                  "effective_neighbor_count": effective, "effective_neighbor_ratio": ratio,
                  "weighted_within_gate_variance": weighted_var}
        for name, value in values.items():
            arr = value.detach().cpu().numpy().astype(np.float64, copy=False)
            if not np.isfinite(arr).all():
                raise FloatingPointError(f"Nonfinite relation feature {modality}/{name}")
            features[f"{modality}_{name}"] = arr
    return features, degree.detach().cpu().numpy().astype(np.int64)


def fit_utility_oof(features_self: np.ndarray, features_struct: np.ndarray,
                    y_train: np.ndarray, y_val: np.ndarray,
                    classes: list[int], train_nodes: np.ndarray,
                    val_nodes: np.ndarray, dataset: str, seed: int, modality: str
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(np.unique(y_train)) < 2:
        raise ValueError(f"Cannot fit utility probes on a single-class train split: {dataset}/{seed}/{modality}")
    counts = np.unique(y_train, return_counts=True)[1]
    if counts.min() < 5:
        raise ValueError(f"5-fold stratified OOF impossible for rare class: {dataset}/{seed}/{modality}")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof_ce = {"self": np.full(y_train.size, np.nan), "struct": np.full(y_train.size, np.nan)}
    seen = np.zeros(y_train.size, dtype=np.int8)
    for fit_idx, held_idx in splitter.split(np.zeros(y_train.size), y_train):
        if np.intersect1d(train_nodes[fit_idx], train_nodes[held_idx]).size:
            raise AssertionError("OOF fold model sees a held-out training node")
        seen[held_idx] += 1
        for name, values in (("self", features_self), ("struct", features_struct)):
            xt, xv, _ = standardize_pair(values[train_nodes[fit_idx]], values[train_nodes[held_idx]])
            model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                       class_weight=None, random_state=0)
            with warnings.catch_warnings(), threadpool_limits(limits=THREADPOOL_LIMIT):
                warnings.simplefilter("ignore")
                model.fit(xt, y_train[fit_idx])
                probs = expand_probabilities(model, model.predict_proba(xv), classes)
            oof_ce[name][held_idx] = class_prob_ce(probs, y_train[held_idx])
    if not np.all(seen == 1) or np.isnan(oof_ce["self"]).any() or np.isnan(oof_ce["struct"]).any():
        raise AssertionError("Every Train node must receive exactly one out-of-fold prediction")
    val_ce: dict[str, np.ndarray] = {}
    for name, values in (("self", features_self), ("struct", features_struct)):
        xt, xv, _ = standardize_pair(values[train_nodes], values[val_nodes])
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                   class_weight=None, random_state=0)
        with warnings.catch_warnings(), threadpool_limits(limits=THREADPOOL_LIMIT):
            warnings.simplefilter("ignore")
            model.fit(xt, y_train)
            probs = expand_probabilities(model, model.predict_proba(xv), classes)
        val_ce[name] = class_prob_ce(probs, y_val)
    return (oof_ce["self"] - oof_ce["struct"], val_ce["self"] - val_ce["struct"],
            oof_ce["self"], oof_ce["struct"])


def composition_suite(dataset: str, seed: int, variant: str, condition: str,
                      states: dict[str, list[torch.Tensor]], train: torch.Tensor, val: torch.Tensor,
                      y_train: np.ndarray, y_val: np.ndarray, classes: list[int],
                      uniform_readouts: dict[str, torch.Tensor], landscape_rows: list[dict[str, Any]],
                      modality_rows: list[dict[str, Any]], oracle_rows: list[dict[str, Any]]) -> None:
    nodes_tr, nodes_va = train.numpy(), val.numpy()
    propagated = {m: p_state(states[m]) for m in ("text", "visual")}
    results: dict[str, list[dict[str, Any]]] = {"text": [], "visual": [], "multimodal": []}
    for alpha in ALPHAS:
        z = {m: composition_state(states[m], alpha, uniform_readouts[m], propagated[m])
             for m in ("text", "visual")}
        views = {"text": z["text"], "visual": z["visual"],
                 "multimodal": torch.cat((z["text"], z["visual"]), dim=-1)}
        for modality, feature in views.items():
            fit = fit_logistic(feature[nodes_tr].numpy(), y_train, feature[nodes_va].numpy(), y_val, classes)
            results[modality].append(fit)
            landscape_rows.append({
                "dataset": dataset, "seed": seed, "variant": variant, "condition": condition,
                "modality": modality, "alpha": alpha, "val_accuracy": fit["accuracy"],
                "val_macro_f1": fit["macro_f1"], "val_true_label_ce": fit["true_label_ce"],
                "converged": fit["converged"], "n_iter": fit["n_iter"],
                "best_alpha_descriptive_only": False,
            })
    for modality, fits in results.items():
        all_ce = np.stack([class_prob_ce(fit["probs"], y_val) for fit in fits])
        star = all_ce.argmin(axis=0)
        pick = np.stack([fit["pred"] for fit in fits])[star, np.arange(y_val.size)]
        base_index = ALPHAS.index(.25)
        base_ce = all_ce[base_index]
        oracle_ce = all_ce[star, np.arange(y_val.size)]
        base_pred = fits[base_index]["pred"]
        oracle_acc = float((pick == y_val).mean())
        base_acc = float((base_pred == y_val).mean())
        counts = np.bincount(star, minlength=len(ALPHAS))
        for ix, alpha in enumerate(ALPHAS):
            oracle_rows.append({
                "dataset": dataset, "seed": seed, "variant": variant, "condition": condition,
                "modality": modality, "alpha": alpha, "alpha_star_count": int(counts[ix]),
                "alpha_star_fraction": float(counts[ix] / y_val.size),
                "validation_label_oracle": True, "descriptive_only": True,
                "oracle_val_ce": float(oracle_ce.mean()), "alpha_0_25_val_ce": float(base_ce.mean()),
                "oracle_ce_headroom": float((base_ce - oracle_ce).mean()),
                "oracle_val_accuracy": oracle_acc, "alpha_0_25_val_accuracy": base_acc,
                "oracle_accuracy_headroom": oracle_acc - base_acc,
            })
    for alpha_t, alpha_v in MODALITY_ALPHA_PAIRS:
        zt = alpha_t * states["text"][0] + (1.0 - alpha_t) * propagated["text"]
        zv = alpha_v * states["visual"][0] + (1.0 - alpha_v) * propagated["visual"]
        fit = fit_logistic(torch.cat((zt[nodes_tr], zv[nodes_tr]), -1).numpy(), y_train,
                           torch.cat((zt[nodes_va], zv[nodes_va]), -1).numpy(), y_val, classes)
        modality_rows.append({"dataset": dataset, "seed": seed, "variant": variant, "condition": condition,
                              "alpha_text": alpha_t, "alpha_visual": alpha_v,
                              "val_accuracy": fit["accuracy"], "val_macro_f1": fit["macro_f1"],
                              "val_true_label_ce": fit["true_label_ce"], "converged": fit["converged"],
                              "n_iter": fit["n_iter"],
                              "interpretation": "Text/Visual composition heterogeneity diagnostic only"})


def process_states(dataset: str, seed: int, variant: str, condition: str,
                   states: dict[str, list[torch.Tensor]], uniform_readouts: dict[str, torch.Tensor],
                   data, train: torch.Tensor, val: torch.Tensor, y_train: np.ndarray, y_val: np.ndarray,
                   classes: list[int], alignment_rows: list[dict[str, Any]], probe_rows: list[dict[str, Any]],
                   residual_rows: list[dict[str, Any]], conditional_rows: list[dict[str, Any]],
                   landscape_rows: list[dict[str, Any]], modality_rows: list[dict[str, Any]],
                   oracle_rows: list[dict[str, Any]]) -> None:
    nodes_tr, nodes_va = train.numpy(), val.numpy()
    five_states = {m: [*states[m], p_state(states[m])] for m in ("text", "visual")}
    for modality in ("text", "visual"):
        s0 = five_states[modality][0].numpy()
        for ix, state_name in enumerate(("S1", "S2", "S3", "P"), start=1):
            structural = five_states[modality][ix].numpy()
            for split, ids in (("Train", nodes_tr), ("Validation", nodes_va)):
                alignment_rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                                       "condition": condition, "modality": modality,
                                       "pair": f"S0_vs_{state_name}", "split": split,
                                       **alignment_diagnostics(s0[ids], structural[ids]),
                                       "interpretation": "alignment/redundancy is not task utility"})
    for ix, state_name in enumerate(STATE_NAMES):
        t, v = five_states["text"][ix], five_states["visual"][ix]
        for modality, feature in (("Text", t), ("Visual", v), ("Multimodal", torch.cat((t, v), -1))):
            logistic_rows(dataset, seed, variant, condition, modality, state_name,
                          feature[nodes_tr].numpy(), feature[nodes_va].numpy(), y_train, y_val,
                          classes, probe_rows)
    composition_suite(dataset, seed, variant, condition, states, train, val, y_train, y_val,
                      classes, uniform_readouts, landscape_rows, modality_rows, oracle_rows)

    if condition not in ("NORMAL", "OFF"):
        return
    residuals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for modality in ("text", "visual"):
        s0 = five_states[modality][0].numpy()
        for state_index, state_name in enumerate(("S1", "S2", "S3", "P"), start=1):
            target = five_states[modality][state_index].numpy()
            residuals[f"{modality}_{state_name}"] = residualize_train_only(
                s0[nodes_tr], s0[nodes_va], target[nodes_tr], target[nodes_va])
    for state_name in ("S1", "S2", "S3", "P"):
        for modality, train_x, val_x in (
            ("Text", *residuals[f"text_{state_name}"]),
            ("Visual", *residuals[f"visual_{state_name}"]),
            ("Multimodal", np.concatenate((residuals[f"text_{state_name}"][0], residuals[f"visual_{state_name}"][0]), 1),
             np.concatenate((residuals[f"text_{state_name}"][1], residuals[f"visual_{state_name}"][1]), 1)),
        ):
            logistic_rows(dataset, seed, variant, condition, modality,
                          f"R_{state_name}_linearly_S0_unpredictable", train_x, val_x,
                          y_train, y_val, classes, residual_rows)
    degree = physical_degree(data)
    rtr = np.concatenate((residuals["text_P"][0], residuals["visual_P"][0]), axis=1)
    rval = np.concatenate((residuals["text_P"][1], residuals["visual_P"][1]), axis=1)
    s0tr = np.concatenate((five_states["text"][0][nodes_tr].numpy(), five_states["visual"][0][nodes_tr].numpy()), 1)
    s0val = np.concatenate((five_states["text"][0][nodes_va].numpy(), five_states["visual"][0][nodes_va].numpy()), 1)
    real = fit_logistic(np.concatenate((s0tr, rtr), 1), y_train,
                        np.concatenate((s0val, rval), 1), y_val, classes)
    shuffled = []
    for repeat in range(10):
        seed_base = 20260927 + seed * 101 + repeat * 7
        rtr_shuf = shuffle_within_degree_bins(rtr, degree[nodes_tr], np.random.default_rng(seed_base))
        rval_shuf = shuffle_within_degree_bins(rval, degree[nodes_va], np.random.default_rng(seed_base + 1))
        shuffled.append(fit_logistic(np.concatenate((s0tr, rtr_shuf), 1), y_train,
                                     np.concatenate((s0val, rval_shuf), 1), y_val, classes))
    avg = lambda key: statistics.fmean(float(item[key]) for item in shuffled)
    conditional_rows.append({
        "dataset": dataset, "seed": seed, "variant": variant, "condition": condition,
        "real_val_accuracy": real["accuracy"], "real_val_macro_f1": real["macro_f1"],
        "real_val_true_label_ce": real["true_label_ce"],
        "shuffle_mean_val_accuracy": avg("accuracy"), "shuffle_mean_val_macro_f1": avg("macro_f1"),
        "shuffle_mean_val_true_label_ce": avg("true_label_ce"),
        "conditional_gain_accuracy": real["accuracy"] - avg("accuracy"),
        "conditional_gain_macro_f1": real["macro_f1"] - avg("macro_f1"),
        "conditional_gain_true_label_ce": real["true_label_ce"] - avg("true_label_ce"),
        "n_deterministic_shuffles": 10,
        "shuffle_unit": "node assignment within physical-degree quantile bins",
        "train_and_validation_shuffled_independently": True,
        "residual_name": "linearly-S0-unpredictable residual",
    })



def physical_degree(data) -> np.ndarray:
    edge = data.edge_index.detach().cpu().long().t().numpy()
    if edge.size == 0:
        return np.zeros(data.num_nodes, dtype=np.int64)
    edge = edge[edge[:, 0] != edge[:, 1]]
    undirected = np.sort(edge, axis=1)
    undirected = np.unique(undirected, axis=0)
    degree = np.zeros(data.num_nodes, dtype=np.int64)
    np.add.at(degree, undirected[:, 0], 1)
    np.add.at(degree, undirected[:, 1], 1)
    return degree


def historical_checkpoint_path(dataset: str, seed: int, name: str) -> Path:
    path = HIST_ROOT / dataset / name / f"best_run{seed - 41}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


@torch.no_grad()
def historical_p6_logits(dataset: str, seed: int, data) -> dict[str, torch.Tensor]:
    from torch import nn
    from src.models import build_model
    result = {}
    info = {"input_dim": data.input_dim, "num_nodes": data.num_nodes,
            "num_classes": data.num_classes, "text_dim": int(data.x_t.size(1)),
            "visual_dim": int(data.x_i.size(1))}
    for modality in ("text", "visual"):
        for state, readout in (("self", "self_only"), ("uniform", "uniform")):
            name = f"{modality}_{state}"
            path = historical_checkpoint_path(dataset, seed, name)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if (payload.get("seed") != seed or payload.get("selection") != "best_val_accuracy" or
                    payload.get("task") != "nc" or payload.get("protocol_version") != PROTOCOL):
                raise ValueError(f"Invalid historical P6 checkpoint metadata: {path}")
            cfg = compose_cfg(dataset, seed, "multi_order_bank", readout=readout, modality_mode=modality)
            model = build_model(cfg, info).to(DEVICE)
            model.load_state_dict(payload["model_state"], strict=True)
            head = nn.Linear(model.out_dim, int(data.num_classes)).to(DEVICE)
            head.load_state_dict(payload["head_state"], strict=True)
            model.eval(); head.eval()
            analysis = model.analyze(data.x.to(DEVICE), data.edge_index.to(DEVICE))
            logits = head(analysis["fused_z"]).detach().float().cpu()
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"Nonfinite historical P6 logits: {path}")
            result[name] = logits
            del model, head, analysis, payload
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return result


def node_ce(logits: torch.Tensor, labels: np.ndarray, node_ids: np.ndarray) -> np.ndarray:
    import torch.nn.functional as F
    ids = torch.as_tensor(node_ids, dtype=torch.long)
    y = torch.as_tensor(labels, dtype=torch.long)
    selected = logits[ids]
    return F.cross_entropy(selected, y, reduction="none").numpy()


def historical_p6_rows(dataset: str, seed: int, data, train: torch.Tensor, val: torch.Tensor,
                       y_train: np.ndarray, y_val: np.ndarray,
                       new_g: dict[str, tuple[np.ndarray, np.ndarray]],
                       rows: list[dict[str, Any]]) -> None:
    logits = historical_p6_logits(dataset, seed, data)
    split_data = (("Train", train.numpy(), y_train), ("Validation", val.numpy(), y_val))
    historical: dict[str, dict[str, np.ndarray]] = {"text": {}, "visual": {}}
    for modality in ("text", "visual"):
        for split, nodes, labels in split_data:
            self_ce = node_ce(logits[f"{modality}_self"], labels, nodes)
            struct_ce = node_ce(logits[f"{modality}_uniform"], labels, nodes)
            historical[modality][split] = self_ce - struct_ce
            g_values = new_g[modality][0 if split == "Train" else 1]
            for node, gn, gh in zip(nodes, g_values, historical[modality][split], strict=True):
                rows.append({"dataset": dataset, "seed": seed, "modality": modality,
                             "split": split, "node_id": int(node),
                             "new_probe_defined_G": float(gn), "historical_p6_G": float(gh),
                             "historical_source": "Experiment 2 frozen modality self/uniform checkpoints",
                             "comparison_boundary": "different representation/model family; external consistency only"})
    del logits


def relation_feature_rows(dataset: str, seed: int, feature_dict: dict[str, np.ndarray],
                          degree: np.ndarray, train: torch.Tensor, val: torch.Tensor,
                          rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    nodes_tr, nodes_va = train.numpy(), val.numpy()
    features = dict(feature_dict)
    for modality in ("text", "visual"):
        for name in REL_FEATURES:
            if name not in {key.split("_", 1)[1] for key in features if key.startswith(f"{modality}_")}:
                raise AssertionError(f"Missing relation-set feature: {modality}_{name}")
    for split, nodes in (("Train", nodes_tr), ("Validation", nodes_va)):
        for modality in ("text", "visual"):
            for i, node in enumerate(nodes):
                row: dict[str, Any] = {"dataset": dataset, "seed": seed, "node_id": int(node),
                                       "modality": modality, "split": split,
                                       "physical_degree": int(degree[node]),
                                       "degree_zero_or_one_convention": "degree0: dosage=1, moments/shares/counts=0; degree1: entropy/variance=0, share=1, effective_count=1"}
                for name in REL_FEATURES:
                    row[name] = float(features[f"{modality}_{name}"][node])
                row["dosage"] = float(features[f"{modality}_dosage"][node])
                row["weighted_within_gate_variance"] = float(features[f"{modality}_weighted_within_gate_variance"][node])
                rows.append(row)
    return features


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return math.nan
    value = spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else math.nan


def relation_utility_rows(dataset: str, seed: int, features: dict[str, np.ndarray],
                          train: torch.Tensor, val: torch.Tensor,
                          targets: dict[str, tuple[np.ndarray, np.ndarray]],
                          association_rows: list[dict[str, Any]],
                          crossmodal_rows: list[dict[str, Any]],
                          predictor_rows: list[dict[str, Any]],
                          predictor_contrasts: list[dict[str, Any]]) -> None:
    ids = {"Train": train.numpy(), "Validation": val.numpy()}
    targets_by_modality: dict[str, dict[str, np.ndarray]] = {"text": {}, "visual": {}}
    for modality in ("text", "visual"):
        targets_by_modality[modality] = {"Train": targets[modality][0], "Validation": targets[modality][1]}
        for split in ("Train", "Validation"):
            idx = ids[split]
            g = targets_by_modality[modality][split]
            signs = (g > 0).astype(np.int8)
            for name in REL_FEATURES:
                x = features[f"{modality}_{name}"][idx]
                rho = safe_spearman(x, g)
                sign_source = "Train_OOF" if split == "Validation" else "Train_OOF"
                train_g = targets_by_modality[modality]["Train"]
                train_x = features[f"{modality}_{name}"][ids["Train"]]
                train_rho = safe_spearman(train_x, train_g)
                direction = -1 if np.isfinite(train_rho) and train_rho < 0 else 1
                score = direction * x
                auc = float(roc_auc_score(signs, score)) if len(np.unique(signs)) == 2 else math.nan
                association_rows.append({"dataset": dataset, "seed": seed, "modality": modality,
                                         "feature": name, "split": split, "spearman_feature_vs_G": rho,
                                         "train_oof_spearman_for_direction": train_rho,
                                         "auc_direction": direction, "binary_G_positive_auroc": auc,
                                         "auc_direction_source": sign_source,
                                         "binary_G_positive_fraction": float(signs.mean()),
                                         "n_nodes": int(g.size),
                                         "target": "probe-defined structural-state utility proxy; not causal utility"})
    for split in ("Train", "Validation"):
        gt = targets_by_modality["text"][split]
        gv = targets_by_modality["visual"][split]
        dctx = features["text_centered_log_dosage"][ids[split]] - features["visual_centered_log_dosage"][ids[split]]
        dut = gt - gv
        crossmodal_rows.append({"dataset": dataset, "seed": seed, "split": split,
                                "spearman_D_context_vs_D_utility": safe_spearman(dctx, dut),
                                "sign_agreement": float((np.sign(dctx) == np.sign(dut)).mean()),
                                "n_nodes": int(dut.size), "D_context": "centered_log_dosage_text - centered_log_dosage_visual",
                                "D_utility": "G_text - G_visual"})

    per_target_metrics: dict[str, dict[str, float]] = {}
    for target_modality in ("text", "visual"):
        ytr_g = targets_by_modality[target_modality]["Train"]
        yva_g = targets_by_modality[target_modality]["Validation"]
        ytr = (ytr_g > 0).astype(np.int64)
        yva = (yva_g > 0).astype(np.int64)
        names = ["P0_TOPOLOGY", "P1_DOSAGE", "P2_RELATION_SET", "P3_CROSS_MODAL_RELATION_SET"]
        local: dict[str, dict[str, float]] = {}
        for predictor in names:
            if predictor == "P3_CROSS_MODAL_RELATION_SET":
                other = "visual" if target_modality == "text" else "text"
                cols = [f"{target_modality}_{n}" for n in REL_FEATURES] + [f"{other}_{n}" for n in REL_FEATURES]
            else:
                cols = [f"{target_modality}_{n}" for n in PREDICTOR_FEATURES[predictor]]
            xtr = np.column_stack([features[c][ids["Train"]] for c in cols]).astype(np.float32)
            xva = np.column_stack([features[c][ids["Validation"]] for c in cols]).astype(np.float32)
            scaler = StandardScaler()
            xtr_s = scaler.fit_transform(xtr).astype(np.float32, copy=False)
            xva_s = scaler.transform(xva).astype(np.float32, copy=False)
            continuous = Ridge(alpha=1.0, fit_intercept=True)
            with threadpool_limits(limits=THREADPOOL_LIMIT):
                continuous.fit(xtr_s, ytr_g)
                ghat = continuous.predict(xva_s)
            cont_spearman = safe_spearman(ghat, yva_g)
            cont_r2 = float(r2_score(yva_g, ghat)) if yva_g.size > 1 else math.nan
            if len(np.unique(ytr)) < 2:
                status = "INVALID_SINGLE_CLASS"
                auc = balacc = acc = math.nan
            else:
                classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                                class_weight=None, random_state=0)
                with warnings.catch_warnings(), threadpool_limits(limits=THREADPOOL_LIMIT):
                    warnings.simplefilter("ignore")
                    classifier.fit(xtr_s, ytr)
                    pred = classifier.predict(xva_s)
                    prob = classifier.predict_proba(xva_s)[:, list(classifier.classes_).index(1)] if 1 in classifier.classes_ else np.zeros_like(yva, dtype=float)
                status = "VALID"
                auc = float(roc_auc_score(yva, prob)) if len(np.unique(yva)) == 2 else math.nan
                balacc = float(balanced_accuracy_score(yva, pred)) if len(np.unique(yva)) == 2 else math.nan
                acc = float(accuracy_score(yva, pred))
            local[predictor] = {"auroc": auc, "balanced_accuracy": balacc,
                                "accuracy": acc, "spearman_pred_G": cont_spearman,
                                "r2_pred_G": cont_r2}
            predictor_rows.append({"dataset": dataset, "seed": seed, "target_modality": target_modality,
                                   "predictor": predictor, "status": status,
                                   "val_auroc": auc, "val_balanced_accuracy": balacc, "val_accuracy": acc,
                                   "val_spearman_pred_G": cont_spearman, "val_r2_pred_G": cont_r2,
                                   "scaler_fit_split": "Train OOF utility rows only",
                                   "logistic_hyperparameters": "C=1.0;solver=lbfgs;max_iter=2000;class_weight=None;random_state=0",
                                   "ridge_alpha": 1.0, "validation_used_for_fit": False,
                                   "binary_target": "1[G>0]; Train target uses 5-fold OOF probe-derived G"})
        per_target_metrics[target_modality] = local
        for left, right in (("P1_DOSAGE", "P0_TOPOLOGY"), ("P2_RELATION_SET", "P0_TOPOLOGY"),
                            ("P2_RELATION_SET", "P1_DOSAGE"),
                            ("P3_CROSS_MODAL_RELATION_SET", "P2_RELATION_SET")):
            l, r = local[left], local[right]
            predictor_contrasts.append({"dataset": dataset, "seed": seed,
                                        "target_modality": target_modality, "contrast": f"{left}-{right}",
                                        **{f"delta_{metric}": (l[metric] - r[metric])
                                           if np.isfinite(l[metric]) and np.isfinite(r[metric]) else math.nan
                                           for metric in ("auroc", "balanced_accuracy", "accuracy", "spearman_pred_G", "r2_pred_G")}})


def historical_consistency_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], int(row["seed"]), row["modality"], row["split"])].append(row)
    summary = []
    for (dataset, seed, modality, split), items in sorted(grouped.items()):
        new = np.array([float(r["new_probe_defined_G"]) for r in items])
        old = np.array([float(r["historical_p6_G"]) for r in items])
        summary.append({"dataset": dataset, "seed": seed, "modality": modality, "split": split,
                        "spearman_new_vs_historical_G": safe_spearman(new, old),
                        "sign_agreement": float((np.sign(new) == np.sign(old)).mean()),
                        "mean_new_G": float(new.mean()), "mean_historical_G": float(old.mean()),
                        "n_nodes": int(new.size),
                        "external_check_only": "different representation/model families; descriptive consistency"})
    return summary


HISTORICAL_CHECKPOINT_AUDIT: list[dict[str, Any]] = []


@torch.no_grad()
def historical_p6_logits(dataset: str, seed: int, data) -> dict[str, torch.Tensor]:
    from torch import nn
    from src.models import build_model
    result = {}
    info = {"input_dim": data.input_dim, "num_nodes": data.num_nodes,
            "num_classes": data.num_classes, "text_dim": int(data.x_t.size(1)),
            "visual_dim": int(data.x_i.size(1))}
    for modality in ("text", "visual"):
        for state, readout in (("self", "self_only"), ("uniform", "uniform")):
            name = f"{modality}_{state}"
            path = historical_checkpoint_path(dataset, seed, name)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if (payload.get("seed") != seed or payload.get("selection") != "best_val_accuracy" or
                    payload.get("task") != "nc" or payload.get("protocol_version") != PROTOCOL):
                raise ValueError(f"Invalid historical P6 checkpoint metadata: {path}")
            HISTORICAL_CHECKPOINT_AUDIT.append({
                "dataset": dataset, "seed": seed, "variant": name, "source": "historical_Experiment2_P6",
                "path": str(path.relative_to(ROOT)), "checkpoint_sha256": sha256(path),
                "training_branch": payload.get("run_metadata", {}).get("training_branch", "historical"),
                "training_commit": payload.get("run_metadata", {}).get("training_commit", "historical provenance"),
                "selection": payload["selection"], "epoch": payload.get("epoch"),
                "test_metrics_used": False, "historical_test_fields_ignored": True,
            })
            cfg = compose_cfg(dataset, seed, "multi_order_bank", readout=readout, modality_mode=modality)
            model = build_model(cfg, info).to(DEVICE)
            model.load_state_dict(payload["model_state"], strict=True)
            head = nn.Linear(model.out_dim, int(data.num_classes)).to(DEVICE)
            head.load_state_dict(payload["head_state"], strict=True)
            model.eval(); head.eval()
            analysis = model.analyze(data.x.to(DEVICE), data.edge_index.to(DEVICE))
            logits = head(analysis["fused_z"]).detach().float().cpu()
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"Nonfinite historical P6 logits: {path}")
            result[name] = logits
            del model, head, analysis, payload
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return result


def summarize_paired_probes(probe_rows: list[dict[str, Any]], state_name: str) -> list[dict[str, Any]]:
    by_key = {(r["dataset"], int(r["seed"]), r["variant"], r["condition"], r["modality"], r["state"]): r
              for r in probe_rows}
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for variant in CALIBRATED:
                for modality in ("Text", "Visual", "Multimodal"):
                    normal = by_key[(dataset, seed, variant, "NORMAL", modality, state_name)]
                    off = by_key[(dataset, seed, variant, "OFF", modality, state_name)]
                    rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                                 "modality": modality, "state": state_name,
                                 "contrast": "NORMAL_minus_OFF_same_checkpoint",
                                 "delta_val_accuracy": float(normal["val_accuracy"]) - float(off["val_accuracy"]),
                                 "delta_val_macro_f1": float(normal["val_macro_f1"]) - float(off["val_macro_f1"]),
                                 "delta_val_true_label_ce": float(normal["val_true_label_ce"]) - float(off["val_true_label_ce"])})
    return rows


def paired_residual_contrasts(residual_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(r["dataset"], int(r["seed"]), r["variant"], r["condition"], r["modality"], r["state"]): r
              for r in residual_rows}
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for variant in CALIBRATED:
                for modality in ("Text", "Visual", "Multimodal"):
                    name = "R_P_linearly_S0_unpredictable"
                    normal = by_key[(dataset, seed, variant, "NORMAL", modality, name)]
                    off = by_key[(dataset, seed, variant, "OFF", modality, name)]
                    rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                                 "modality": modality, "state": name,
                                 "contrast": "NORMAL_minus_OFF_same_checkpoint",
                                 "delta_val_accuracy": float(normal["val_accuracy"]) - float(off["val_accuracy"]),
                                 "delta_val_macro_f1": float(normal["val_macro_f1"]) - float(off["val_macro_f1"]),
                                 "delta_val_true_label_ce": float(normal["val_true_label_ce"]) - float(off["val_true_label_ce"])})
    return rows


def add_composition_best_alpha(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], int(row["seed"]), row["variant"], row["condition"], row["modality"])].append(row)
    for items in groups.values():
        by_ce = min(items, key=lambda r: float(r["val_true_label_ce"]))
        by_acc = max(items, key=lambda r: float(r["val_accuracy"]))
        for row in items:
            row["best_alpha_by_val_ce_descriptive"] = float(by_ce["alpha"])
            row["best_alpha_by_val_accuracy_descriptive"] = float(by_acc["alpha"])
            row["best_alpha_descriptive_only"] = True


def mean_key(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows if np.isfinite(float(r[key]))]
    return statistics.fmean(vals) if vals else math.nan


def build_hypothesis_summary(state_contrasts, residual_contrasts, conditional_rows,
                             composition_rows, modality_rows, predictor_contrasts,
                             utility_assoc, crossmodal_assoc) -> dict[str, Any]:
    state_p = [r for r in state_contrasts if r["state"] == "P"]
    residual_p = [r for r in residual_contrasts if r["state"] == "R_P_linearly_S0_unpredictable"]
    def effect(rows):
        result = {}
        for metric in ("accuracy", "macro_f1", "true_label_ce"):
            vals = [float(r[f"delta_val_{metric}"]) for r in rows]
            result[metric] = {"mean": statistics.fmean(vals) if vals else math.nan,
                              "positive_pairs": sum(v > 0 for v in vals), "n": len(vals)}
        return result
    state_e, residual_e = effect(state_p), effect(residual_p)
    cond_acc = [float(r["conditional_gain_accuracy"]) for r in conditional_rows]
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in composition_rows:
        groups[(row["dataset"], row["seed"], row["variant"], row["condition"], row["modality"])].append(row)
    unique_best = len({min(v, key=lambda x: float(x["val_true_label_ce"]))["alpha"] for v in groups.values()})
    p2 = [r for r in predictor_contrasts if r["contrast"] == "P2_RELATION_SET-P0_TOPOLOGY"]
    p3 = [r for r in predictor_contrasts if r["contrast"] == "P3_CROSS_MODAL_RELATION_SET-P2_RELATION_SET"]
    p2mean, p3mean = mean_key(p2, "delta_auroc"), mean_key(p3, "delta_auroc")
    p2positive = sum(float(r["delta_auroc"]) > 0 for r in p2 if np.isfinite(float(r["delta_auroc"])))
    p3positive = sum(float(r["delta_auroc"]) > 0 for r in p3 if np.isfinite(float(r["delta_auroc"])))
    h51 = state_e["accuracy"]["mean"] > 0 and state_e["accuracy"]["positive_pairs"] >= 18
    h52 = residual_e["accuracy"]["mean"] > 0 and residual_e["accuracy"]["positive_pairs"] >= 18
    return {
        "H5.1 StructuralStateInformation": {
            "status": "STRONG_SUPPORT" if h51 else ("MIXED" if state_e["accuracy"]["positive_pairs"] else "UNSUPPORTED"),
            "effect_sizes": state_e,
            "definition": "same-checkpoint NORMAL minus OFF P probes; positive Acc/F1 and negative CE favor NORMAL"},
        "H5.2 StateComplementarityBeyondS0": {
            "status": "STRONG_SUPPORT" if h52 else ("MIXED" if residual_e["accuracy"]["positive_pairs"] else "UNSUPPORTED"),
            "effect_sizes": residual_e,
            "conditional_shuffle_mean_accuracy_gain": mean_key(conditional_rows, "conditional_gain_accuracy"),
            "conditional_shuffle_mean_macro_f1_gain": mean_key(conditional_rows, "conditional_gain_macro_f1"),
            "conditional_shuffle_mean_ce_delta": mean_key(conditional_rows, "conditional_gain_true_label_ce"),
            "conditional_shuffle_positive_accuracy_pairs": sum(v > 0 for v in cond_acc),
            "conditional_shuffle_pairs": len(cond_acc),
            "definition": "linearly-S0-unpredictable residual; not statistically or causally independent"},
        "H5.3 CompositionHeterogeneity": {
            "status": "MECHANISM_SUPPORT" if unique_best > 1 else "UNSUPPORTED",
            "distinct_descriptive_best_alpha_values": unique_best, "composition_rows": len(composition_rows),
            "fixed_modality_composition_rows": len(modality_rows),
            "validation_label_oracle_descriptive_only": True},
        "H5.4 RelationContextPredictability": {
            "status": "MECHANISM_SUPPORT" if p2mean > 0 and p2positive >= 18 else ("MIXED" if p2positive else "UNSUPPORTED"),
            "P2_minus_P0_mean_validation_AUROC": p2mean, "P2_minus_P0_positive_pairs": p2positive,
            "P2_minus_P0_pairs": len(p2), "association_rows": len(utility_assoc)},
        "H5.5 CrossModalContextPredictability": {
            "status": "MECHANISM_SUPPORT" if p3mean > 0 and p3positive >= 18 else ("MIXED" if p3positive else "UNSUPPORTED"),
            "P3_minus_P2_mean_validation_AUROC": p3mean, "P3_minus_P2_positive_pairs": p3positive,
            "P3_minus_P2_pairs": len(p3), "crossmodal_association_rows": len(crossmodal_assoc),
            "boundary": "incremental prediction does not establish model benefit"},
    }


def render_report(summary: dict[str, Any]) -> str:
    h = summary["hypotheses"]; p = summary["provenance"]
    s1, s2, s3, s4, s5 = [h[k] for k in h]
    lines = [
        "# S4.6 State Complementarity & Relation-Context Utility Audit", "",
        f"- Source: {p['source_branch']} at {p['source_sha']}; analysis branch {p['analysis_branch']}.",
        f"- Frozen checkpoint training commit: {p['checkpoint_training_commit']}.",
        f"- Datasets/seeds: {', '.join(p['datasets'])}; {', '.join(map(str, p['seeds']))}.",
        "- Frozen inference only; no GNN training. Test disabled. No Toys, Test-label reads, or LP.",
        "- Probe: Train-only StandardScaler + sklearn LogisticRegression, C=1, lbfgs, max_iter=2000, class_weight=None, random_state=0.",
        "- Utility Train target: 5-fold stratified OOF, shuffle=True, random_state=0; Validation probes fit on Train only.", "",
        "## Main hypothesis assessment", "",
        "| Hypothesis | Status | Observed effects |", "|---|---|---|",
        f"| H5.1 StructuralStateInformation | {s1['status']} | P NORMAL−OFF: ΔAcc {s1['effect_sizes']['accuracy']['mean']:+.4f} ({s1['effect_sizes']['accuracy']['positive_pairs']}/{s1['effect_sizes']['accuracy']['n']} positive); ΔF1 {s1['effect_sizes']['macro_f1']['mean']:+.4f}; ΔCE {s1['effect_sizes']['true_label_ce']['mean']:+.4f} |",
        f"| H5.2 StateComplementarityBeyondS0 | {s2['status']} | Residual P ΔAcc {s2['effect_sizes']['accuracy']['mean']:+.4f}; real−shuffle ΔAcc {s2['conditional_shuffle_mean_accuracy_gain']:+.4f} ({s2['conditional_shuffle_positive_accuracy_pairs']}/{s2['conditional_shuffle_pairs']}) |",
        f"| H5.3 CompositionHeterogeneity | {s3['status']} | {s3['distinct_descriptive_best_alpha_values']} distinct best-CE alpha values; Validation-label oracle is descriptive only |",
        f"| H5.4 RelationContextPredictability | {s4['status']} | P2−P0 mean Val AUROC {s4['P2_minus_P0_mean_validation_AUROC']:+.4f} ({s4['P2_minus_P0_positive_pairs']}/{s4['P2_minus_P0_pairs']} positive pairs) |",
        f"| H5.5 CrossModalContextPredictability | {s5['status']} | P3−P2 mean Val AUROC {s5['P3_minus_P2_mean_validation_AUROC']:+.4f} ({s5['P3_minus_P2_positive_pairs']}/{s5['P3_minus_P2_pairs']} positive pairs) |", "",
        "## Stage questions", "",
        "1. Same-checkpoint NORMAL/OFF isolates the gate intervention while preserving projector, fusion, and weights. Linear-probe performance is not end-to-end NC performance.",
        "2. Residual probes and degree-bin shuffles test whether a linearly-S0-unpredictable component remains useful. Ridge residuals are not statistically or causally independent.",
        "3. The alpha landscape, six fixed modality pairs, and node oracle describe possible composition heterogeneity. Validation best alpha and oracle never set a future hyperparameter.",
        "4. G is a probe-defined structural-state utility proxy, not true counterfactual relation utility. OOF avoids a Train node's in-sample probe leakage but does not make G causal.",
        "5. Compare P2−P0, P2−P1, and feature associations. The relation features come from supervised trained checkpoints and may contain co-adapted label-related signals.",
        "6. P3−P2 measures incremental prediction only. It does not show that cross-modal interaction improves a final model.",
        "7. The status table is a diagnostic gate; weak correlations are not mechanism success.", "",
        "## Boundaries", "",
        "- Composition alpha oracle uses Validation labels and is descriptive only.",
        "- No Test-label access, Toys, LP, or new GNN training was used.",
        "- Historical P6 consistency compares distinct representation/model families and is an external descriptive check.",
        "- Relation-set dosage comes only from unconstrained NORMAL; mass-preserving raw gate magnitudes are excluded.", "",
        "All required outputs and checkpoint hashes are listed in s46_summary.json and CSVs in this result directory.", ""
    ]
    return "\n".join(lines)



def run(args) -> None:
    global DEVICE
    DEVICE = torch.device(args.device)
    if DEVICE.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested CUDA device is unavailable: {DEVICE}")
        torch.cuda.set_device(DEVICE)
    if git_branch() != "s46_state_relation_context":
        raise RuntimeError("S4.6 analysis must run on branch s46_state_relation_context")
    if git_head() != SOURCE_SHA:
        raise RuntimeError(f"S4.6 source HEAD mismatch: expected {SOURCE_SHA}, got {git_head()}")
    checkpoint_audit = audit_checkpoints()
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_OK", "branch": git_branch(), "head": git_head(),
                          "checkpoints": len(checkpoint_audit), "datasets": DATASETS, "seeds": SEEDS,
                          "device": str(DEVICE), "test_disabled": True, "formal_gnn_training": False,
                          "toys": False, "lp": False}, indent=2))
        return

    OUT.mkdir(parents=True, exist_ok=True)
    cache_dir = OUT / ".resume"
    cache_dir.mkdir(parents=True, exist_ok=True)
    analysis_sha256 = sha256(Path(__file__).resolve())
    tables = {name: [] for name in (
        "state_alignment", "state_linear_probes", "state_residual_probes",
        "conditional_residual_probe", "composition_landscape", "modality_composition_probe",
        "composition_node_oracle", "state_utility_targets", "relation_set_features",
        "relation_utility_association", "crossmodal_utility_association",
        "utility_predictor_results", "utility_predictor_contrasts", "historical_p6_consistency",
    )}
    same_checkpoint_qa: dict[tuple[str, int, str], dict[str, float]] = {}
    for dataset in DATASETS:
        for seed in SEEDS:
            cache_path = cache_dir / f"{dataset}_seed{seed}.json"
            checkpoint_hashes = {
                row["variant"]: row["checkpoint_sha256"] for row in checkpoint_audit
                if row["dataset"] == dataset and row["seed"] == seed
            }
            cache_metadata = {
                "schema": 1, "source_sha": SOURCE_SHA, "training_commit": TRAINING_COMMIT,
                "analysis_sha256": analysis_sha256, "dataset": dataset, "seed": seed,
                "checkpoint_sha256": checkpoint_hashes,
            }
            if cache_path.is_file():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("metadata") != cache_metadata:
                    raise ValueError(f"Resume cache provenance mismatch: {cache_path}; remove stale cache explicitly")
                for name in tables:
                    tables[name].extend(cached["tables"][name])
                for variant, qa in cached["same_checkpoint_qa"].items():
                    same_checkpoint_qa[(dataset, seed, variant)] = qa
                    audit = next(r for r in checkpoint_audit
                                 if r["dataset"] == dataset and r["seed"] == seed and r["variant"] == variant)
                    audit.update(qa)
                    audit["P_definition"] = "(S1+S2+S3)/3"
                    audit["alpha_0_25_uniform_max_abs_error"] = 0.0
                    audit["relation_gate_used_for_utility"] = variant == "s45_unconstrained_entry_uniform"
                print(f"S46 {dataset} seed={seed} resumed from atomic cache", flush=True)
                continue

            row_starts = {name: len(rows) for name, rows in tables.items()}
            print(f"S46 {dataset} seed={seed} device={DEVICE}", flush=True)
            data, train, val, y_train, y_val, classes = load_dataset(dataset, seed)
            physical_deg = physical_degree(data)

            # The identity source has no learned relation gates.
            variant = VARIANTS[0]
            model, head, _ = load_s45_model(dataset, seed, variant, data)
            id_states, _ = extract_s45(model, data, "IDENTITY")
            id_uniform = {m: sum(id_states[m]) / 4.0 for m in ("text", "visual")}
            process_states(dataset, seed, variant, "IDENTITY", id_states, id_uniform, data,
                           train, val, y_train, y_val, classes, tables["state_alignment"],
                           tables["state_linear_probes"], tables["state_residual_probes"],
                           tables["conditional_residual_probe"], tables["composition_landscape"],
                           tables["modality_composition_probe"], tables["composition_node_oracle"])
            del id_states, id_uniform, model, head
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for variant in CALIBRATED:
                model, head, _ = load_s45_model(dataset, seed, variant, data)
                normal_states, normal_result = extract_s45(model, data, "NORMAL")
                off_states, off_result = extract_s45(model, data, "OFF")

                # QA A: same checkpoint, same projector/fusion/weights; only relation gate changes.
                s0_error = max(float((normal_states[m][0] - off_states[m][0]).abs().max().item())
                               for m in ("text", "visual"))
                if s0_error > 1e-7:
                    raise AssertionError(f"S0 NORMAL/OFF mismatch {dataset}/{seed}/{variant}: {s0_error}")

                # QA B: g_ij=1 reconstructs the historical uncalibrated operator exactly.
                p_error = 0.0
                for modality in ("text", "visual"):
                    base_p = normal_result["P"].coalesce()
                    off_p = off_result["operators"][modality].coalesce()
                    if not torch.equal(base_p.indices(), off_p.indices()):
                        raise AssertionError(f"OFF operator index mismatch {dataset}/{seed}/{variant}/{modality}")
                    p_error = max(p_error, float((base_p.values() - off_p.values()).abs().max().item()))
                if p_error > 1e-7:
                    raise AssertionError(f"OFF operator differs from P {dataset}/{seed}/{variant}: {p_error}")
                same_checkpoint_qa[(dataset, seed, variant)] = {
                    "s0_normal_off_max_abs": s0_error,
                    "off_operator_vs_historical_p_max_abs": p_error,
                }

                if variant == "s45_unconstrained_entry_uniform":
                    # Only the unconstrained NORMAL gate is eligible for relation-set utility features.
                    relation_features, gate_degree = extract_relation_features(
                        model, normal_result, data.num_nodes, train.numpy())
                    if not np.array_equal(gate_degree, physical_deg):
                        raise AssertionError("P_rel degree must match physical undirected graph degree")
                    relation_feature_rows(dataset, seed, relation_features, gate_degree, train, val,
                                          tables["relation_set_features"])
                    ids_train, ids_val = train.numpy(), val.numpy()
                    new_g: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                    for modality in ("text", "visual"):
                        self_state = normal_states[modality][0].numpy()
                        struct_state = p_state(normal_states[modality]).numpy()
                        gtr, gval, ce_self_tr, ce_struct_tr = fit_utility_oof(
                            self_state, struct_state, y_train, y_val, classes,
                            ids_train, ids_val, dataset, seed, modality)
                        # Validation targets use full-Train fitted probes; scaler and model never see Val labels.
                        xts, xvs, _ = standardize_pair(self_state[ids_train], self_state[ids_val])
                        xtr, xvr, _ = standardize_pair(struct_state[ids_train], struct_state[ids_val])
                        probe_self = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                                        class_weight=None, random_state=0)
                        probe_struct = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                                          class_weight=None, random_state=0)
                        with warnings.catch_warnings(), threadpool_limits(limits=THREADPOOL_LIMIT):
                            warnings.simplefilter("ignore")
                            probe_self.fit(xts, y_train); probe_struct.fit(xtr, y_train)
                            prob_self = expand_probabilities(probe_self, probe_self.predict_proba(xvs), classes)
                            prob_struct = expand_probabilities(probe_struct, probe_struct.predict_proba(xvr), classes)
                        ce_self_val = class_prob_ce(prob_self, y_val)
                        ce_struct_val = class_prob_ce(prob_struct, y_val)
                        if not np.allclose(ce_self_val - ce_struct_val, gval, atol=1e-10, rtol=1e-10):
                            raise AssertionError("G_val does not equal same-capacity probe CE difference")
                        new_g[modality] = (gtr, gval)
                        for split, nodes, gain, self_ce, struct_ce in (
                            ("Train", ids_train, gtr, ce_self_tr, ce_struct_tr),
                            ("Validation", ids_val, gval, ce_self_val, ce_struct_val),
                        ):
                            for node, g, cs, cr in zip(nodes, gain, self_ce, struct_ce, strict=True):
                                tables["state_utility_targets"].append({
                                    "node_id": int(node), "dataset": dataset, "seed": seed,
                                    "modality": modality, "split": split, "G": float(g),
                                    "binary_G_positive": bool(g > 0), "self_CE": float(cs), "struct_CE": float(cr),
                                    "train_target_oof": split == "Train",
                                    "target_definition": "probe-defined structural-state utility proxy; not causal utility",
                                })
                    relation_utility_rows(dataset, seed, relation_features, train, val, new_g,
                                          tables["relation_utility_association"],
                                          tables["crossmodal_utility_association"],
                                          tables["utility_predictor_results"],
                                          tables["utility_predictor_contrasts"])
                    historical_p6_rows(dataset, seed, data, train, val, y_train, y_val, new_g,
                                       tables["historical_p6_consistency"])

                uniform_normal = {m: sum(normal_states[m]) / 4.0 for m in ("text", "visual")}
                uniform_off = {m: sum(off_states[m]) / 4.0 for m in ("text", "visual")}
                process_states(dataset, seed, variant, "NORMAL", normal_states, uniform_normal,
                               data, train, val, y_train, y_val, classes, tables["state_alignment"],
                               tables["state_linear_probes"], tables["state_residual_probes"],
                               tables["conditional_residual_probe"], tables["composition_landscape"],
                               tables["modality_composition_probe"], tables["composition_node_oracle"])
                process_states(dataset, seed, variant, "OFF", off_states, uniform_off, data,
                               train, val, y_train, y_val, classes, tables["state_alignment"],
                               tables["state_linear_probes"], tables["state_residual_probes"],
                               tables["conditional_residual_probe"], tables["composition_landscape"],
                               tables["modality_composition_probe"], tables["composition_node_oracle"])

                audit = next(r for r in checkpoint_audit
                             if r["dataset"] == dataset and r["seed"] == seed and r["variant"] == variant)
                audit.update(same_checkpoint_qa[(dataset, seed, variant)])
                audit["P_definition"] = "(S1+S2+S3)/3"
                audit["alpha_0_25_uniform_max_abs_error"] = 0.0
                audit["relation_gate_used_for_utility"] = variant == "s45_unconstrained_entry_uniform"
                del normal_result, off_result, normal_states, off_states, uniform_normal, uniform_off, model, head
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del data
            cache_payload = {
                "metadata": cache_metadata,
                "tables": {name: rows[row_starts[name]:] for name, rows in tables.items()},
                "same_checkpoint_qa": {
                    variant: same_checkpoint_qa[(dataset, seed, variant)] for variant in CALIBRATED
                },
            }
            write_json_atomic(cache_path, cache_payload)
            print(f"S46 {dataset} seed={seed} cached", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    state_contrasts = summarize_paired_probes(tables["state_linear_probes"], "P")
    residual_contrasts = paired_residual_contrasts(tables["state_residual_probes"])
    add_composition_best_alpha(tables["composition_landscape"])
    historical_summary = historical_consistency_summary(tables["historical_p6_consistency"])
    checkpoint_audit.extend(HISTORICAL_CHECKPOINT_AUDIT)
    tables["checkpoint_audit"] = checkpoint_audit
    tables["state_probe_contrasts"] = state_contrasts
    tables["state_residual_contrasts"] = residual_contrasts
    tables["historical_p6_consistency_summary"] = historical_summary

    files = {
        "checkpoint_audit": "checkpoint_audit.csv",
        "state_alignment": "state_alignment.csv",
        "state_linear_probes": "state_linear_probes.csv",
        "state_probe_contrasts": "state_probe_contrasts.csv",
        "state_residual_probes": "state_residual_probes.csv",
        "state_residual_contrasts": "state_residual_contrasts.csv",
        "conditional_residual_probe": "conditional_residual_probe.csv",
        "composition_landscape": "composition_landscape.csv",
        "modality_composition_probe": "modality_composition_probe.csv",
        "composition_node_oracle": "composition_node_oracle.csv",
        "state_utility_targets": "state_utility_targets.csv",
        "relation_set_features": "relation_set_features.csv",
        "relation_utility_association": "relation_utility_association.csv",
        "crossmodal_utility_association": "crossmodal_utility_association.csv",
        "utility_predictor_results": "utility_predictor_results.csv",
        "utility_predictor_contrasts": "utility_predictor_contrasts.csv",
        "historical_p6_consistency": "historical_p6_consistency.csv",
        "historical_p6_consistency_summary": "historical_p6_consistency_summary.csv",
    }
    for key, filename in files.items():
        write_csv(OUT / filename, tables[key])

    hypotheses = build_hypothesis_summary(state_contrasts, residual_contrasts,
                                          tables["conditional_residual_probe"],
                                          tables["composition_landscape"],
                                          tables["modality_composition_probe"],
                                          tables["utility_predictor_contrasts"],
                                          tables["relation_utility_association"],
                                          tables["crossmodal_utility_association"])
    summary = {
        "stage": "S4.6_State_Complementarity_Relation_Context_Utility_Audit",
        "provenance": {
            "source_branch": SOURCE_BRANCH, "source_sha": SOURCE_SHA,
            "analysis_branch": "s46_state_relation_context",
            "checkpoint_training_commit": TRAINING_COMMIT,
            "datasets": list(DATASETS), "seeds": list(SEEDS),
            "s45_checkpoint_count": len([r for r in checkpoint_audit if r.get("source") != "historical_Experiment2_P6"]),
            "historical_p6_checkpoint_count": len(HISTORICAL_CHECKPOINT_AUDIT),
            "checkpoint_sha256": {f"{r['dataset']}/{r['seed']}/{r['variant']}": r["checkpoint_sha256"]
                                  for r in checkpoint_audit if r.get("source") != "historical_Experiment2_P6"},
            "historical_checkpoint_sha256": {f"{r['dataset']}/{r['seed']}/{r['variant']}": r["checkpoint_sha256"]
                                             for r in HISTORICAL_CHECKPOINT_AUDIT},
        },
        "protocol": {
            "test_disabled": True, "test_label_access": False, "toys_included": False,
            "lp_included": False, "formal_gnn_training": False,
            "inference_device": str(DEVICE),
            "probe": {"classifier": "sklearn LogisticRegression", "C": 1.0, "solver": "lbfgs",
                      "max_iter": 2000, "class_weight": None, "random_state": 0,
                      "scaler": "StandardScaler fit on Train only",
                      "threadpool_limit": THREADPOOL_LIMIT},
            "residual_ridge_alpha": 1.0,
            "utility_oof": {"folds": 5, "stratified": True, "shuffle": True, "random_state": 0},
            "utility_predictor_ridge_alpha": 1.0, "conditional_shuffle_repeats": 10,
            "alpha_grid": list(ALPHAS), "modality_alpha_pairs": [list(p) for p in MODALITY_ALPHA_PAIRS],
            "macro_f1_class_set": "classes observed from Train and Validation only; no Test labels accessed",
        },
        "qa": {
            "same_checkpoint_s0_normal_off_max_abs": max(v["s0_normal_off_max_abs"] for v in same_checkpoint_qa.values()),
            "off_operator_vs_historical_p_max_abs": max(v["off_operator_vs_historical_p_max_abs"] for v in same_checkpoint_qa.values()),
            "P_definition_assertions": "passed; P=(S1+S2+S3)/3",
            "alpha_0_25_uniform_assertions": "algebraic equivalence checked within float32 tolerance; uniform baseline reused bit-identically",
            "residualization_fit_split": "Train only",
            "scaler_fit_split": "Train only",
            "oof_exactly_once": "asserted for each Train node in both modalities",
            "validation_used_for_probe_or_predictor_fit": False,
            "test_labels_or_test_targets_used": False,
            "toys_excluded": True,
            "finite_degree_0_1_relation_features": True,
            "masspres_raw_gate_used_for_utility": False,
            "historical_files_modified": False,
            "preexisting_historical_checkpoint_test_metrics_used": False,
        },
        "row_counts": {key: len(value) for key, value in tables.items()},
        "hypotheses": hypotheses,
        "interpretation_boundaries": [
            "Linear-probe performance is not end-to-end model performance.",
            "Ridge residuals mean only linearly-S0-unpredictable; they are not independent information.",
            "G_i is a probe-defined utility proxy, not true counterfactual relation utility.",
            "Trained relation-set features may contain co-adapted label-related signal.",
            "OOF prevents train-node in-sample probe leakage but does not make G causal.",
            "Composition alpha oracle uses Validation labels and is descriptive only.",
            "P3>P2 is incremental prediction only, not proof that cross-modal interaction improves a model.",
            "Weak correlations are not mechanism success.", "No Toys, Test, or LP.",
        ],
    }
    write_json(OUT / "s46_summary.json", summary)
    write_text(OUT / "s46_report.md", render_report(summary))
    shutil.rmtree(cache_dir, ignore_errors=True)
    print(json.dumps({"status": "S46_COMPLETE", "out": str(OUT),
                      "rows": summary["row_counts"], "hypotheses": hypotheses}, indent=2))


def git_branch() -> str:
    import subprocess
    return subprocess.check_output(["git", "-C", str(ROOT), "branch", "--show-current"], text=True).strip()


def git_head() -> str:
    import subprocess
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen S4.6 state and relation-context audit")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default=str(DEVICE))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

