from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCREEN_DATASETS = ("Movies", "Grocery", "ele-fashion")
PROTOCOL = "unified_full_graph_nc_v1"


def _config(dataset: str, device: str, variant: str, model_name: str = "risa_v06"):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str((ROOT / "configs").resolve()), version_base=None):
        return compose(config_name="config", overrides=[
            f"dataset={dataset}", "task=nc", f"model={model_name}",
            f"model.variant={variant}", "seed=42", "num_runs=1", f"device={device}",
            "task.evaluate_test=false", "task.training_mode=full_graph",
            f"task.protocol_version={PROTOCOL}",
        ])


def _data_info(data) -> dict[str, int]:
    return {
        "input_dim": int(data.input_dim), "num_nodes": int(data.num_nodes),
        "num_classes": int(data.num_classes), "text_dim": int(data.x_t.size(1)),
        "visual_dim": int(data.x_i.size(1)),
    }


def _parameter_audit(model: torch.nn.Module, num_classes: int) -> dict[str, int]:
    groups = {
        "backbone": ("backbone.",),
        "relation_encoders": ("relation_encoder_",),
        "shared_relation": ("shared_relation_encoder.", "shared_to_", "shared_norm_"),
        "same_evidence": ("evidence_extractor_",),
        "cross_evidence": ("evidence_extractor_",),
        "rho_projection": ("W_rho_",),
        "absorption": ("imci_",),
    }
    counts = {key: 0 for key in groups}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        for group, prefixes in groups.items():
            if name.startswith(prefixes):
                if group == "same_evidence" and "." in name and not any(
                        f".{branch}." in name for branch in ("V_same", "G_same", "U_same")):
                    continue
                if group == "cross_evidence" and "." in name and not any(
                        f".{branch}." in name for branch in ("V_cross", "G_cross", "U_cross")):
                    continue
                counts[group] += parameter.numel()
    counts["total_model"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    counts["classifier"] = (model.out_dim + 1) * int(num_classes)
    counts["total_with_classifier"] = counts["total_model"] + counts["classifier"]
    return counts


def _gradient_audit(model: torch.nn.Module, classifier: torch.nn.Module,
                    x: torch.Tensor, edge: torch.Tensor,
                    labels: torch.Tensor, train_idx: torch.Tensor,
                    device: torch.device) -> tuple[dict[str, Any], dict[str, int | None]]:
    prefixes = {
        "relation_encoders": ("relation_encoder_",),
        "shared_relation": ("shared_relation_encoder.", "shared_to_", "shared_norm_"),
        "same_evidence": ("evidence_extractor_",),
        "cross_evidence": ("evidence_extractor_",),
        "W_rho": ("W_rho_",),
        "MHA": ("imci_text.attention.", "imci_visual.attention."),
        "FFN": ("imci_text.ffn.", "imci_visual.ffn."),
    }
    model.train()
    classifier.train()
    model.zero_grad(set_to_none=True)
    classifier.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    z, _, _, _, _ = model(x, edge)
    logits = classifier(z.index_select(0, train_idx))
    loss = F.cross_entropy(logits, labels.index_select(0, train_idx))
    loss.backward()
    audit: dict[str, Any] = {}
    for group, starts in prefixes.items():
        parameters = [(name, parameter) for name, parameter in model.named_parameters()
                      if name.startswith(starts)]
        if group in {"same_evidence", "cross_evidence"}:
            branch = "same" if group == "same_evidence" else "cross"
            parameters = [(name, parameter) for name, parameter in parameters
                          if f"_{branch}." in name]
        present = [(name, parameter.grad) for name, parameter in parameters
                   if parameter.grad is not None]
        nonzero = [(name, grad) for name, grad in present if bool(torch.count_nonzero(grad))]
        norm = torch.sqrt(sum((grad.detach().float().square().sum() for _, grad in present),
                              torch.zeros((), device=device)))
        audit[group] = {
            "parameters": len(parameters), "parameters_with_grad": len(present),
            "parameters_with_nonzero_grad": len(nonzero),
            "global_grad_l2_norm": float(norm.detach().cpu()),
            "nonzero": bool(nonzero),
            "status": "audited" if parameters else "not_applicable_for_variant",
        }
    memory = {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
    }
    return audit, memory


def _ratio(numerator: torch.Tensor, anchor: torch.Tensor) -> float:
    values = numerator.norm(dim=-1) / (anchor.norm(dim=-1) + 1e-12)
    return float(values.mean().detach().cpu())


def _modality_diagnostics(result: dict[str, Any], modality: str) -> dict[str, Any]:
    h0 = result[f"H0_{modality}"]
    evidence = result[f"E_{modality}"]
    s1 = result[f"S1_{modality}"]
    attention = result["attention_weights"][modality]
    weights = attention.squeeze(-2) if attention is not None else None
    return {
        "relational_evidence_strength_E_over_H0": _ratio(evidence, h0),
        "same_evidence_ratio_E_same_over_H0": _ratio(result[f"E_same_{modality}"], h0),
        "cross_evidence_ratio_E_cross_over_H0": _ratio(result[f"E_cross_{modality}"], h0),
        "relation_context_query_shift_over_H0": _ratio(result[f"rho_shift_{modality}"], h0),
        "total_query_shift_over_H0": _ratio(result[f"query_shift_{modality}"], h0),
        "mean_cosine_E_S1": float(F.cosine_similarity(
            evidence, s1, dim=-1, eps=1e-12,
        ).mean().detach().cpu()),
        "mean_cosine_E_H0": float(F.cosine_similarity(
            evidence, h0, dim=-1, eps=1e-12,
        ).mean().detach().cpu()),
        "mean_hop_attention": result["mean_hop_attention"][modality].detach().cpu().tolist()
        if result["mean_hop_attention"][modality] is not None else None,
        "hop_attention_entropy": float(result["hop_attention_entropy"][modality].detach().cpu())
        if result["hop_attention_entropy"][modality] is not None else None,
        "per_hop_attention_std_across_nodes_and_heads": result["hop_attention_std"][modality]
        .detach().cpu().tolist() if result["hop_attention_std"][modality] is not None else None,
        "attention_shape": list(attention.shape) if attention is not None else None,
        "max_attention_sum_error": float((weights.sum(-1) - 1.0).abs().max().detach().cpu())
        if weights is not None and weights.numel() else 0.0,
    }


@torch.no_grad()
def _equivalence_audit(model: torch.nn.Module, x: torch.Tensor,
                       edge: torch.Tensor, data_info: dict[str, int],
                       dataset: str, device: torch.device) -> dict[str, Any]:
    from src.models.risa_v05 import Model as V05Model

    v05_config = _config(dataset, str(device), "v05_absorb_only", "risa_v05")
    v05 = V05Model(v05_config, data_info).to(device)
    current = model.state_dict()
    target = v05.state_dict()
    missing_from_v06 = sorted(set(target) - set(current))
    extra_v05 = sorted(set(current) - set(target))
    if missing_from_v06:
        return {"parameter_mapping": "failed", "missing_from_v06": missing_from_v06,
                "unexpected_v06_keys": extra_v05}
    v05.load_state_dict({key: current[key] for key in target}, strict=True)
    model.eval()
    v05.eval()
    result_v06 = model.analyze(x, edge, collect_attention=False)
    result_v05 = v05.analyze(x, edge, collect_edge_state=False, collect_attention=False)
    stage_pairs = {
        "H0_text": (result_v06["H0_text"], result_v05["H0_text"]),
        "H0_visual": (result_v06["H0_visual"], result_v05["H0_visual"]),
        "S1_text_vs_C1_text": (result_v06["S1_text"], result_v05["C1_text"]),
        "S1_visual_vs_C1_visual": (result_v06["S1_visual"], result_v05["C1_visual"]),
        "S2_text_vs_C2_text": (result_v06["S2_text"], result_v05["C2_text"]),
        "S2_visual_vs_C2_visual": (result_v06["S2_visual"], result_v05["C2_visual"]),
        "S3_text_vs_C3_text": (result_v06["S3_text"], result_v05["C3_text"]),
        "S3_visual_vs_C3_visual": (result_v06["S3_visual"], result_v05["C3_visual"]),
        "Z_text": (result_v06["Z_text"], result_v05["Z_text"]),
        "Z_visual": (result_v06["Z_visual"], result_v05["Z_visual"]),
        "fused_z": (result_v06["fused_z"], result_v05["fused_z"]),
    }
    stage_differences = {}
    for stage, (left, right) in stage_pairs.items():
        absolute = (left - right).abs()
        stage_differences[stage] = {
            "bitwise_equal": bool(torch.equal(left, right)),
            "max_abs_difference": float(absolute.max().detach().cpu()) if absolute.numel() else 0.0,
            "mean_abs_difference": float(absolute.mean().detach().cpu()) if absolute.numel() else 0.0,
        }
    first_divergence = next((stage for stage, values in stage_differences.items()
                             if not values["bitwise_equal"]), None)
    maximum = max((values["max_abs_difference"] for values in stage_differences.values()), default=0.0)
    return {
        "parameter_mapping": "passed",
        "mapped_parameter_tensors": len(target),
        "unexpected_v06_relation_parameters": extra_v05,
        "forward_bitwise_equal": stage_differences["fused_z"]["bitwise_equal"],
        "forward_max_abs_difference": stage_differences["fused_z"]["max_abs_difference"],
        "forward_mean_abs_difference": stage_differences["fused_z"]["mean_abs_difference"],
        "stage_differences": stage_differences,
        "first_divergence": first_divergence,
        "difference_source": (
            "CUDA COO sparse.mm accumulation on the first plain physical propagation hop; "
            "projected H0 is bitwise equal and later hop/readout differences inherit the "
            "sub-micro-scale sparse reduction rounding. CPU toy-graph compatibility is "
            "checked separately for bitwise equality."
            if first_divergence is not None and first_divergence.startswith("S1_")
            and stage_differences["H0_text"]["bitwise_equal"]
            and stage_differences["H0_visual"]["bitwise_equal"]
            else "See stage_differences for the first differing intermediate."
        ),
        "numeric_tolerance": 1e-5,
        "numerically_equivalent_within_tolerance": maximum <= 1e-5,
        "max_abs_difference_over_all_stages": maximum,
    }


def analyze(checkpoint: Path, dataset: str, variant: str,
            device_name: str) -> dict[str, Any]:
    from torch import nn
    from src.data import load_mag_data
    from src.models import build_model

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("task") != "nc" or payload.get("protocol_version") != PROTOCOL:
        raise ValueError("checkpoint does not use the unified full-graph NC protocol")
    if int(payload.get("seed", -1)) != 42 or payload.get("selection") != "best_val_accuracy":
        raise ValueError("smoke analysis requires a seed-42 Val-Accuracy-selected checkpoint")
    if any(key.startswith("test_") for key in payload.get("metrics", {})):
        raise ValueError("test metrics are forbidden in this workflow")
    if variant not in ("v06_full", "v06_no_relation_condition"):
        raise ValueError(f"unknown v0.6 variant {variant!r}")

    device = torch.device(device_name if not device_name.startswith("cuda") or torch.cuda.is_available()
                          else "cpu")
    config = _config(dataset, str(device), variant)
    data = load_mag_data(config, "nc", 42)
    info = _data_info(data)
    model = build_model(config, payload["data_info"]).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    classifier.load_state_dict(payload["head_state"], strict=True)
    model.eval()
    classifier.eval()

    x = data.x.to(device)
    edge = data.edge_index.to(device)
    p, p_self, p_rel = model.backbone._get_operators(edge, int(x.size(0)), x.dtype)
    operator_snapshots = [(op.indices().clone(), op.values().clone()) for op in (p, p_self, p_rel)]
    with torch.no_grad():
        diagnostic = model.analyze(x, edge, collect_attention=True)
    operators_unchanged = all(
        torch.equal(op.indices(), indices) and torch.equal(op.values(), values)
        for op, (indices, values) in zip((diagnostic["P"], diagnostic["P_self"],
                                          diagnostic["P_rel"]), operator_snapshots, strict=True)
    )
    structural_errors = {}
    for modality in ("text", "visual"):
        expected1 = torch.sparse.mm(diagnostic["P"], diagnostic[f"H0_{modality}"])
        expected2 = torch.sparse.mm(diagnostic["P"], expected1)
        expected3 = torch.sparse.mm(diagnostic["P"], expected2)
        structural_errors[modality] = max(
            float((diagnostic[f"S1_{modality}"] - expected1).abs().max().detach().cpu()),
            float((diagnostic[f"S2_{modality}"] - expected2).abs().max().detach().cpu()),
            float((diagnostic[f"S3_{modality}"] - expected3).abs().max().detach().cpu()),
        )
    modalities = {name: _modality_diagnostics(diagnostic, name) for name in ("text", "visual")}

    train_idx = data.train_idx.to(device=device, dtype=torch.long)
    labels = data.y.to(device=device, dtype=torch.long)
    gradient_audit, memory = _gradient_audit(
        model, classifier, x, edge, labels, train_idx, device,
    )
    required = ["MHA", "FFN"]
    if variant == "v06_full":
        required += ["relation_encoders", "shared_relation", "same_evidence",
                     "cross_evidence", "W_rho"]
    all_embeddings = [diagnostic[key] for key in (
        "H0_text", "H0_visual", "S1_text", "S2_text", "S3_text",
        "S1_visual", "S2_visual", "S3_visual", "Z_text", "Z_visual",
        "E_text", "E_same_text", "E_cross_text", "E_visual", "E_same_visual",
        "E_cross_visual", "rho_text", "rho_visual", "Q_text", "Q_visual", "fused_z",
    )]
    finite = all(bool(torch.isfinite(tensor).all()) for tensor in all_embeddings)
    attention_valid = all(
        modalities[name]["max_attention_sum_error"] < 1e-5
        for name in modalities
    )
    query_shift_finite = all(bool(torch.isfinite(diagnostic[f"query_shift_{m}"]).all())
                             for m in modalities)
    if variant == "v06_no_relation_condition":
        query_is_intrinsic = all(torch.equal(diagnostic[f"Q_{m}"], diagnostic[f"H0_{m}"])
                                 for m in modalities)
    else:
        query_is_intrinsic = True
    gradient_ok = all(gradient_audit[group]["nonzero"] for group in required)
    equivalence = (_equivalence_audit(model, x, edge, info, dataset, device)
                   if variant == "v06_no_relation_condition" else None)
    equivalence_ok = equivalence is None or (
        equivalence.get("parameter_mapping") == "passed"
        and equivalence.get("numerically_equivalent_within_tolerance") is True
    )
    sanity = {
        "finite_embeddings_and_evidence": finite,
        "P_Pself_Prel_unchanged": operators_unchanged,
        "structural_path_matches_three_plain_P_hops": all(value <= 1e-5
                                                            for value in structural_errors.values()),
        "query_shift_finite": query_shift_finite,
        "no_relation_variant_Q_equals_H0": query_is_intrinsic,
        "attention_probabilities_sum_to_one": attention_valid,
        "required_gradients_nonzero": gradient_ok,
        "v05_absorb_only_equivalence": equivalence_ok,
    }
    report = {
        "dataset": dataset, "seed": 42, "variant": variant,
        "protocol_version": PROTOCOL, "checkpoint": str(checkpoint),
        "checkpoint_selection": payload["selection"], "checkpoint_metrics": payload.get("metrics", {}),
        "device": str(device), "parameter_count": _parameter_audit(model, int(data.num_classes)),
        "diagnostics": modalities, "structural_max_abs_errors": structural_errors,
        "gradient_audit": gradient_audit,
        "gradient_audit_loss": "train-split cross-entropy; no optimizer step",
        "gpu_memory": memory,
        "equivalence_v05_absorb_only": equivalence,
        "sanity": sanity, "sanity_passed": all(sanity.values()),
        "test_evaluated": False, "lp_evaluated": False,
    }
    result_dir = ROOT / "results/risa_v06/smoke"
    result_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}_seed42_{variant}"
    (result_dir / f"{stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        f"# RISA v0.6 smoke diagnostics: {dataset}, seed 42, {variant}", "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Validation-selected metrics: `{json.dumps(payload.get('metrics', {}), sort_keys=True)}`",
        f"- Parameter count: `{json.dumps(report['parameter_count'], sort_keys=True)}`",
        f"- Modality diagnostics: `{json.dumps(modalities, sort_keys=True)}`",
        f"- Gradient audit: `{json.dumps(gradient_audit, sort_keys=True)}`",
        f"- Peak PyTorch GPU allocated: `{memory['peak_allocated_bytes']}` bytes",
        f"- v05 absorb-only equivalence: `{json.dumps(equivalence, sort_keys=True)}`",
        f"- Structural max errors: `{json.dumps(structural_errors, sort_keys=True)}`",
        f"- Sanity: `{'PASS' if report['sanity_passed'] else 'FAIL'}`",
        "- Test and LP evaluation: disabled.", "",
    ]
    (result_dir / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")
    if not report["sanity_passed"]:
        raise RuntimeError(f"v0.6 smoke diagnostics failed; see {result_dir / (stem + '.json')}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a RISA v0.6 smoke checkpoint")
    parser.add_argument("--dataset", choices=SCREEN_DATASETS, default="Movies")
    parser.add_argument("--variant", choices=("v06_full", "v06_no_relation_condition"),
                        default="v06_full")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(analyze(args.checkpoint.resolve(), args.dataset, args.variant,
                             args.device), indent=2))


if __name__ == "__main__":
    main()
