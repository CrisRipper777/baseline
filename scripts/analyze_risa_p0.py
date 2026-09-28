from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASETS = ("Movies", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
VARIANTS = (
    "p0_identity", "p0_masspres_scalar", "p0_single_dynamic_transform",
    "p0_operator_uniform", "p0_operator_global", "p0_operator_routed",
)
PROTOCOL = "unified_full_graph_nc_v1"
DEFAULT_CHECKPOINT = ROOT / "outputs/risa_v04_p0_v1/smoke/Movies/p0_operator_routed/best.pt"
RESULT_ROOT = ROOT / "results/risa_v04_p0_v1/diagnostics"


def _config(dataset: str, variant: str, device: str):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str((ROOT / "configs").resolve()), version_base=None):
        return compose(config_name="config", overrides=[
            f"dataset={dataset}", "task=nc", "model=risa_v04",
            f"model.variant={variant}", "seed=42", "num_runs=1", f"device={device}",
            "task.evaluate_test=false", "task.training_mode=full_graph",
            f"task.protocol_version={PROTOCOL}",
        ])


def _data_info(data) -> dict[str, int]:
    return {"input_dim": int(data.input_dim), "num_nodes": int(data.num_nodes),
            "num_classes": int(data.num_classes), "text_dim": int(data.x_t.size(1)),
            "visual_dim": int(data.x_i.size(1))}


def _parameter_groups(model: torch.nn.Module) -> dict[str, int]:
    prefixes = {
        "s45_backbone": ("backbone.",),
        "relation_encoder": ("relation_encoder_",),
        "operator_adapters": ("operators_", "dynamic_"),
        "router": ("router_",),
        "global_mixture_logits": ("theta_",),
        "dynamic_conditioner": ("condition_",),
    }
    result = {key: 0 for key in prefixes}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        for key, starts in prefixes.items():
            if name.startswith(starts) or (key == "s45_backbone" and name.startswith("backbone.")):
                result[key] += parameter.numel()
    result["total_model"] = sum(parameter.numel() for parameter in model.parameters()
                                 if parameter.requires_grad)
    return result


def _load_data(dataset: str):
    from src.data import load_mag_data
    return load_mag_data(_config(dataset, VARIANTS[-1], "cpu"), "nc", 42)


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0:
        return 0.0
    return float((left - right).abs().max().detach().cpu())


def analyze(checkpoint: Path, dataset: str, device_name: str) -> dict[str, Any]:
    from src.models import build_model
    from torch import nn

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("task") != "nc" or payload.get("protocol_version") != PROTOCOL:
        raise ValueError("checkpoint does not use the unified full-graph NC protocol")
    if int(payload.get("seed", -1)) != 42 or payload.get("selection") != "best_val_accuracy":
        raise ValueError("diagnostics require the Movies seed-42 validation-selected checkpoint")
    if any(key.startswith("test_") for key in payload.get("metrics", {})):
        raise ValueError("test metrics are forbidden in this diagnostic workflow")
    if payload.get("model_state", {}).get("backbone.plain_fusion.0.weight") is None:
        raise ValueError("checkpoint is not a RISA P0 model")

    device = torch.device(device_name if not device_name.startswith("cuda") or torch.cuda.is_available() else "cpu")
    data = _load_data(dataset)
    info = _data_info(data)
    cfg = _config(dataset, VARIANTS[-1], str(device))
    model = build_model(cfg, payload["data_info"]).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    classifier.load_state_dict(payload["head_state"], strict=True)
    model.eval(); classifier.eval()

    parameter_audit = {}
    for variant in VARIANTS:
        variant_cfg = _config(dataset, variant, "cpu")
        variant_model = build_model(variant_cfg, info)
        group_counts = _parameter_groups(variant_model)
        classifier_count = (model.out_dim + 1) * int(data.num_classes)
        parameter_audit[variant] = {**group_counts,
                                    "classifier": classifier_count,
                                    "total_with_classifier": group_counts["total_model"] + classifier_count}
        del variant_model

    x = data.x.to(device)
    edge = data.edge_index.to(device)
    val_nodes = data.val_idx.detach().cpu().long()[:32]
    if val_nodes.numel() == 0:
        raise RuntimeError("Movies validation split has no target nodes")
    p_rel = model.backbone._get_operators(edge, int(x.size(0)), x.dtype)[2]
    rows = p_rel.indices()[0]
    targets_gpu = val_nodes.to(device)
    target_mask = torch.zeros(x.size(0), dtype=torch.bool, device=device)
    target_mask[targets_gpu] = True
    candidate = torch.nonzero(target_mask[rows], as_tuple=False).flatten()
    edge_subset = candidate[:512]
    physical_indices_before = p_rel.indices().clone()
    physical_values_before = p_rel.values().clone()
    with torch.no_grad():
        normal = model.analyze(x, edge, target_nodes=targets_gpu, edge_subset=edge_subset,
                               return_edge_state=True)
        off = model.analyze(x, edge, intervention="delta_off", return_edge_state=False)
        reference = model.backbone.analyze(x, edge)
    unchanged = (torch.equal(p_rel.indices(), physical_indices_before) and
                 torch.equal(p_rel.values(), physical_values_before))
    route_stats = {}
    decomposition = {}
    for modality in ("text", "visual"):
        probabilities = normal["router_probabilities"][modality]
        if probabilities.numel():
            route_stats[modality] = {
                "selected_edge_count": int(probabilities.size(0)),
                "max_probability_sum_error": float((probabilities.sum(-1) - 1.0).abs().max().cpu()),
                "mean_probability": probabilities.mean(0).detach().cpu().tolist(),
                "entropy": float(normal["router_entropy"][modality].detach().cpu()),
                "operator_usage": normal["operator_usage"][modality].detach().cpu().tolist(),
                "pairwise_output_cosine": normal["operator_output_pairwise_cosine"][modality].detach().cpu().tolist(),
            }
        else:
            route_stats[modality] = {"selected_edge_count": int(normal["edge_row"].numel()),
                                     "router": "not present for this control variant"}
        base = normal["base_edge_message"][modality]
        delta = normal["operator_correction_delta"][modality]
        if base.numel():
            denominator = base.square().sum(dim=-1, keepdim=True).clamp_min(1e-12)
            parallel = (delta * base).sum(dim=-1, keepdim=True) / denominator * base
            orthogonal = delta - parallel
            decomposition[modality] = {
                "max_abs_delta_minus_parallel_plus_orthogonal": _max_abs(delta, parallel + orthogonal),
                "max_abs_parallel_dot_orthogonal": float((parallel * orthogonal).sum(-1).abs().max().cpu()),
                "selected_edges": int(base.size(0)),
            }
        else:
            decomposition[modality] = {"selected_edges": 0}

    identity_error = {}
    for modality in ("text", "visual"):
        identity_error[f"H0_{modality}"] = _max_abs(off[f"H0_{modality}"], reference[f"H0_{modality}"])
        for order in range(1, 4):
            identity_error[f"C{order}_{modality}"] = _max_abs(off[f"C_{modality}"][order],
                                                               reference[f"S_{modality}"][order])
    identity_error["fused_z"] = _max_abs(off["fused_z"], reference["fused_z"])
    identity_tolerance = 2e-6
    passed = (unchanged and max(identity_error.values(), default=0.0) <= identity_tolerance and
              all(value.get("selected_edge_count", 0) == 0 or
                  value.get("max_probability_sum_error", 0.0) < 1e-6 for value in route_stats.values()) and
              all(value.get("max_abs_delta_minus_parallel_plus_orthogonal", 0.0) < 1e-6
                  for value in decomposition.values()))
    report = {
        "dataset": dataset, "seed": 42, "variant": VARIANTS[-1],
        "protocol_version": PROTOCOL, "test_evaluated": False, "lp_evaluated": False,
        "checkpoint": str(checkpoint), "checkpoint_metrics": payload.get("metrics", {}),
        "device": str(device), "parameter_audit": parameter_audit,
        "diagnostic_subset": {"target_nodes": val_nodes.tolist(),
                              "selected_edges": int(edge_subset.numel()),
                              "available_target_edges": int(candidate.numel())},
        "physical_p_rel_unchanged": unchanged,
        "delta_off_max_abs_against_s45_identity_uniform": identity_error,
        "delta_off_cuda_tolerance": identity_tolerance,
        "correction_norm_ratio": normal["correction_norm_ratio"],
        "cos_base_corrected": normal["cos_base_corrected"],
        "parallel_orthogonal": normal["parallel_orthogonal"],
        "operator_router": route_stats,
        "decomposition_check": decomposition,
        "sanity_passed": bool(passed),
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}_seed42_diagnostics"
    (RESULT_ROOT / f"{stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_lines = [
        f"# RISA v0.4 P0 diagnostics: {dataset} seed 42",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Smoke validation metrics: `{json.dumps(payload.get('metrics', {}), sort_keys=True)}`",
        f"- Physical P_rel unchanged: `{unchanged}`",
        f"- delta_off versus S45 identity-uniform max absolute errors (tolerance `{identity_tolerance}`): `{json.dumps(identity_error, sort_keys=True)}`",
        f"- Selected diagnostic edges: `{int(edge_subset.numel())}` of `{int(candidate.numel())}` edges incident to the selected validation nodes",
        f"- Correction norm ratios: `{json.dumps(normal['correction_norm_ratio'], sort_keys=True)}`",
        f"- cos(base, base + delta): `{json.dumps(normal['cos_base_corrected'], sort_keys=True)}`",
        f"- Router and operator summary: `{json.dumps(route_stats, sort_keys=True)}`",
        f"- Parallel/orthogonal decomposition: `{json.dumps(decomposition, sort_keys=True)}`",
        f"- Sanity checks: `{'PASS' if passed else 'FAIL'}`",
        "- Test and LP evaluation: disabled.",
        "",
    ]
    (RESULT_ROOT / f"{stem}.md").write_text("\n".join(summary_lines), encoding="utf-8")
    if not passed:
        raise RuntimeError(f"diagnostic sanity checks failed; report saved in {RESULT_ROOT / (stem + '.json')}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit RISA v0.4 P0 checkpoint and diagnostics")
    parser.add_argument("--dataset", choices=DATASETS, default="Movies")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = analyze(args.checkpoint.resolve(), args.dataset, args.device)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
