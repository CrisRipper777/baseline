from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCREEN_DATASETS = ("Movies", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
VARIANTS = (
    "p0_identity", "p0_masspres_scalar", "p0_operator_uniform",
    "p0_operator_global", "p0_operator_routed",
)
PROTOCOL = "unified_full_graph_nc_v1"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/risa_v04_p0_v1"
DEFAULT_REPORT_ROOT = ROOT / "results/risa_v04_p0_v1/formal_analysis"
ROUTED_INTERVENTIONS = (
    "normal", "router_mean", "router_uniform", "router_matched_shuffle",
    "delta_off", "delta_parallel_only", "delta_orthogonal_only",
)


def _config(dataset: str, variant: str, seed: int, device: str):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str((ROOT / "configs").resolve()), version_base=None):
        return compose(config_name="config", overrides=[
            f"dataset={dataset}", "task=nc", "model=risa_v04",
            f"model.variant={variant}", f"seed={seed}", "num_runs=1", f"device={device}",
            "task.evaluate_test=false", "task.training_mode=full_graph",
            f"task.protocol_version={PROTOCOL}",
        ])


def _data_info(data) -> dict[str, int]:
    return {"input_dim": int(data.input_dim), "num_nodes": int(data.num_nodes),
            "num_classes": int(data.num_classes), "text_dim": int(data.x_t.size(1)),
            "visual_dim": int(data.x_i.size(1))}


def _load_data(dataset: str):
    from src.data import load_mag_data
    return load_mag_data(_config(dataset, "p0_operator_routed", 42, "cpu"), "nc", 42)


def _checkpoint_path(root: Path, phase: str, dataset: str, variant: str, seed: int) -> Path:
    run_id = SEEDS.index(seed) + 1
    if phase == "preflight":
        return root / "preflight" / dataset / variant / f"best_run{run_id}.pt"
    return root / "formal" / dataset / variant / f"best_run{run_id}.pt"


def _validate_payload(payload: dict[str, Any], seed: int, checkpoint: Path) -> None:
    if payload.get("task") != "nc" or payload.get("protocol_version") != PROTOCOL:
        raise ValueError(f"{checkpoint} does not use unified full-graph NC")
    if int(payload.get("seed", -1)) != seed or payload.get("selection") != "best_val_accuracy":
        raise ValueError(f"{checkpoint} is not the requested validation-accuracy checkpoint")
    metrics = payload.get("metrics", {})
    if not {"val_acc", "val_macro_f1"}.issubset(metrics):
        raise ValueError(f"{checkpoint} is missing validation selection metrics")
    if any(key.startswith("test_") for key in metrics):
        raise ValueError(f"test metrics are present in {checkpoint}")


def _evaluate_val(classifier, z: torch.Tensor, data, device: torch.device) -> dict[str, float]:
    val_idx = data.val_idx.detach().to(device=device, dtype=torch.long)
    labels_all = data.y.to(device=device, dtype=torch.long)
    labels = labels_all[val_idx]
    logits = classifier(z[val_idx])
    preds = logits.argmax(dim=-1)
    class_ids = list(range(int(data.num_classes)))
    return {
        "val_acc": float((preds == labels).float().mean().item()),
        "val_macro_f1": float(f1_score(labels.cpu().numpy(), preds.cpu().numpy(), labels=class_ids,
                                        average="macro", zero_division=0)),
        "val_cross_entropy": float(F.cross_entropy(logits, labels, reduction="mean").item()),
    }


def _quantile_codes(values: np.ndarray, n_bins: int = 4) -> tuple[np.ndarray, list[float]]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.empty((0,), dtype=np.int64), []
    cuts = np.quantile(values, np.arange(1, n_bins, dtype=np.float64) / n_bins, method="linear")
    # Searchsorted handles repeated quantiles deterministically (including tied degrees).
    return np.searchsorted(cuts, values, side="right").astype(np.int64), [float(v) for v in cuts]


def matched_structural_shuffle(p_rel: torch.Tensor, n_nodes: int, n_bins: int = 4) -> tuple[torch.Tensor, dict[str, Any]]:
    """Deterministic cyclic routing shuffle within structural quantile bins."""
    indices = p_rel.indices().detach().cpu().numpy()
    rows, cols = indices[0], indices[1]
    weights = p_rel.values().detach().cpu().numpy().astype(np.float64, copy=False)
    target_degree = np.bincount(rows, minlength=n_nodes)
    source_degree = np.bincount(cols, minlength=n_nodes)
    features = {
        "target_degree": target_degree[rows],
        "source_degree": source_degree[cols],
        "normalized_weight": weights,
    }
    codes = []
    quantiles = {}
    for name, values in features.items():
        code, cuts = _quantile_codes(values, n_bins)
        codes.append(code)
        quantiles[name] = cuts
    if rows.size:
        bin_codes = np.stack(codes, axis=1)
        order = np.lexsort((bin_codes[:, 2], bin_codes[:, 1], bin_codes[:, 0]))
        sorted_codes = bin_codes[order]
        boundaries = np.flatnonzero(np.any(sorted_codes[1:] != sorted_codes[:-1], axis=1)) + 1
        groups = np.split(order, boundaries)
    else:
        bin_codes = np.empty((0, 3), dtype=np.int64)
        groups = []
    permutation = np.arange(rows.size, dtype=np.int64)
    non_singleton_bins = 0
    for group in groups:
        if group.size > 1:
            permutation[group] = np.roll(group, 1)
            non_singleton_bins += 1
    if rows.size and not np.array_equal(bin_codes, bin_codes[permutation]):
        raise RuntimeError("matched shuffle crossed a structural quantile bin")
    report = {
        "method": "deterministic cyclic shift within target-degree/source-degree/normalized-weight quantile bins",
        "quantile_bins_per_feature": n_bins,
        "quantile_cutpoints": quantiles,
        "structural_bin_count": len(groups),
        "non_singleton_bin_count": non_singleton_bins,
        "singleton_edge_count": int(sum(group.size == 1 for group in groups)),
        "edge_count": int(rows.size),
        "shuffled_edge_count": int(np.count_nonzero(permutation != np.arange(rows.size))),
        "preserves_edge_marginal_within_each_bin": True,
    }
    device = p_rel.device
    return torch.as_tensor(permutation, dtype=torch.long, device=device), report


def _as_json(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _as_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _mechanism_summary(result: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for modality in ("text", "visual"):
        summary[modality] = {
            "per_operator_mean_routing_probability": _as_json(result["operator_usage"][modality]),
            "per_operator_across_edge_routing_std": _as_json(result["router_probability_std"][modality]),
            "mean_router_entropy": float(result["router_entropy"][modality].detach().cpu()),
            "mean_kl_pi_edge_to_mean_pi": result["mean_kl_to_mean_router"][modality],
            "operator_output_norm": _as_json(result["operator_output_norm"][modality]),
            "pairwise_operator_output_cosine": _as_json(result["operator_output_pairwise_cosine"][modality]),
            "correction_norm_ratio": result["correction_norm_ratio"][modality],
            "cos_base_base_plus_delta": result["cos_base_corrected"][modality],
            "parallel_norm_ratio": result["parallel_orthogonal"][modality]["parallel_norm_ratio"],
            "orthogonal_norm_ratio": result["parallel_orthogonal"][modality]["orthogonal_norm_ratio"],
            "orthogonal_fraction": result["parallel_orthogonal"][modality]["orthogonal_fraction"],
        }
    return summary


def _parameter_count(model: torch.nn.Module, classifier: torch.nn.Module) -> dict[str, int]:
    groups = {"s45_backbone": 0, "relation_encoder": 0, "operator_adapters": 0,
              "router": 0, "global_mixture_logits": 0, "dynamic_conditioner": 0}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            key = "s45_backbone"
        elif name.startswith("relation_encoder_"):
            key = "relation_encoder"
        elif name.startswith(("operators_", "dynamic_")):
            key = "operator_adapters"
        elif name.startswith("router_"):
            key = "router"
        elif name.startswith("theta_"):
            key = "global_mixture_logits"
        elif name.startswith("condition_"):
            key = "dynamic_conditioner"
        else:
            continue
        groups[key] += parameter.numel()
    groups["model_total"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    groups["classifier"] = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    groups["total_with_classifier"] = groups["model_total"] + groups["classifier"]
    return groups


def _load_model(payload, dataset: str, variant: str, seed: int, device: torch.device, data):
    from src.models import build_model
    from torch import nn

    cfg = _config(dataset, variant, seed, str(device))
    model = build_model(cfg, payload["data_info"]).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    classifier.load_state_dict(payload["head_state"], strict=True)
    model.eval()
    classifier.eval()
    return model, classifier


def analyze_checkpoint(checkpoint: Path, dataset: str, variant: str, seed: int,
                       device_name: str) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _validate_payload(payload, seed, checkpoint)
    device = torch.device(device_name if not device_name.startswith("cuda") or torch.cuda.is_available() else "cpu")
    data = _load_data(dataset)
    model, classifier = _load_model(payload, dataset, variant, seed, device, data)
    parameter_audit = _parameter_count(model, classifier)
    x = data.x.to(device)
    edge = data.edge_index.to(device)
    p_rel = model.backbone._get_operators(edge, int(x.size(0)), x.dtype)[2]
    p_rel_indices_before = p_rel.indices().clone()
    p_rel_values_before = p_rel.values().clone()

    with torch.no_grad():
        normal = model.analyze(x, edge, return_edge_state=False)
        normal_metrics = _evaluate_val(classifier, normal["fused_z"], data, device)
        official = payload["metrics"]
        if abs(normal_metrics["val_acc"] - float(official["val_acc"])) > 1e-5:
            raise RuntimeError(f"post-hoc normal Val Acc differs from selected checkpoint: {normal_metrics} vs {official}")
        if abs(normal_metrics["val_macro_f1"] - float(official["val_macro_f1"])) > 1e-5:
            raise RuntimeError(f"post-hoc normal Val Macro-F1 differs from selected checkpoint: {normal_metrics} vs {official}")
        mechanisms = _mechanism_summary(normal)
        intervention_metrics = {"normal": normal_metrics}
        matched_report = None
        if variant == "p0_operator_routed":
            permutation, matched_report = matched_structural_shuffle(p_rel, int(x.size(0)))
            interventions = (
                ("router_mean", "router_mean", None),
                ("router_uniform", "router_uniform", None),
                ("router_matched_shuffle", "router_shuffle", permutation),
                ("delta_off", "delta_off", None),
                ("delta_parallel_only", "delta_parallel_only", None),
                ("delta_orthogonal_only", "delta_orthogonal_only", None),
            )
            for report_name, intervention, router_permutation in interventions:
                result = model.analyze(x, edge, intervention=intervention,
                                       router_shuffle_permutation=router_permutation,
                                       return_edge_state=False)
                intervention_metrics[report_name] = _evaluate_val(classifier, result["fused_z"], data, device)
                del result

    p_rel_unchanged = (torch.equal(p_rel.indices(), p_rel_indices_before) and
                       torch.equal(p_rel.values(), p_rel_values_before))
    if not p_rel_unchanged:
        raise RuntimeError("formal analyzer observed a mutation of physical P_rel")
    return {
        "dataset": dataset, "seed": seed, "variant": variant,
        "phase": "preflight" if "/preflight/" in str(checkpoint) else "formal",
        "checkpoint": str(checkpoint), "selection": "best_val_accuracy",
        "selected_checkpoint_metrics": official,
        "posthoc_validation_metrics": intervention_metrics,
        "routed_mechanism_diagnostics": mechanisms if variant == "p0_operator_routed" else None,
        "router_matched_shuffle": matched_report,
        "parameter_audit": parameter_audit,
        "physical_P_rel_unchanged": p_rel_unchanged,
        "test_evaluated": False, "lp_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc validation analyzer for RISA P0 checkpoints")
    parser.add_argument("--phase", choices=("formal", "preflight"), default="formal")
    parser.add_argument("--datasets", nargs="+", choices=SCREEN_DATASETS, default=None)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    datasets = tuple(args.datasets or (("Movies",) if args.phase == "preflight" else SCREEN_DATASETS))
    variants = tuple(args.variants or (("p0_operator_routed",) if args.phase == "preflight" else VARIANTS))
    if args.phase == "preflight" and (datasets != ("Movies",) or variants != ("p0_operator_routed",)
                                      or tuple(args.seeds) != SEEDS):
        parser.error("preflight analysis is fixed to Movies × p0_operator_routed × seeds 42/43/44")
    results = []
    for dataset in datasets:
        for variant in variants:
            for seed in args.seeds:
                checkpoint = _checkpoint_path(args.output_root, args.phase, dataset, variant, seed)
                record = analyze_checkpoint(checkpoint, dataset, variant, seed, args.device)
                results.append(record)
                print(json.dumps({"dataset": dataset, "seed": seed, "variant": variant,
                                  "status": "analyzed", "checkpoint": str(checkpoint)}), flush=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": args.phase, "protocol_version": PROTOCOL,
        "datasets": list(datasets), "variants": list(variants), "seeds": list(args.seeds),
        "checkpoint_selection": "validation accuracy only; CE is post-hoc and does not select checkpoints",
        "test_evaluated": False, "lp_evaluated": False, "records": results,
    }
    output_path = args.report_root / f"{args.phase}_risa_p0_formal_analysis.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_path = args.report_root / f"{args.phase}_risa_p0_formal_analysis.md"
    lines = [f"# RISA P0 {args.phase} validation analysis", "",
             "Checkpoints were selected by validation accuracy. Val CE is post-hoc only.",
             "Test and LP evaluation were disabled.", ""]
    for record in results:
        metrics = record["posthoc_validation_metrics"]
        lines.append(f"## {record['dataset']} / seed {record['seed']} / {record['variant']}")
        lines.append("")
        for name, values in metrics.items():
            lines.append(f"- {name}: Acc={values['val_acc']:.8f}, Macro-F1={values['val_macro_f1']:.8f}, CE={values['val_cross_entropy']:.8f}")
        if record["routed_mechanism_diagnostics"] is not None:
            lines.append(f"- Routed diagnostics: `{json.dumps(record['routed_mechanism_diagnostics'], sort_keys=True)}`")
            lines.append(f"- Matched shuffle: `{json.dumps(record['router_matched_shuffle'], sort_keys=True)}`")
        lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report_json": str(output_path), "report_markdown": str(summary_path),
                      "records": len(results), "test_evaluated": False}, indent=2))


if __name__ == "__main__":
    main()
