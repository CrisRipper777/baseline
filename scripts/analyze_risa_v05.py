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

SCREEN_DATASETS = ("Movies", "ele-fashion", "Reddit-S")
PROTOCOL = "unified_full_graph_nc_v1"


def _config(dataset: str, device: str, variant: str = "v05_full"):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str((ROOT / "configs").resolve()), version_base=None):
        return compose(config_name="config", overrides=[
            f"dataset={dataset}", "task=nc", "model=risa_v05",
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
        "s45_backbone": ("backbone.",),
        "modality_relation_encoders": ("relation_encoder_",),
        "shared_relation_encoder": ("shared_relation_encoder.", "shared_to_", "shared_norm_"),
        "rotation_angle_heads": ("rotation_",),
        "imci": ("imci_",),
    }
    counts = {key: 0 for key in groups}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        for group, prefixes in groups.items():
            if name.startswith(prefixes):
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
        "modality_relation_encoders": ("relation_encoder_",),
        "shared_relation_encoder": ("shared_relation_encoder.", "shared_to_", "shared_norm_"),
        "angle_heads": ("rotation_",),
        "multihead_attention": ("imci_text.attention.", "imci_visual.attention."),
        "imci_ffn": ("imci_text.ffn.", "imci_visual.ffn."),
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
        }
    memory = {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
    }
    return audit, memory


def analyze(checkpoint: Path, dataset: str, device_name: str) -> dict[str, Any]:
    from torch import nn
    from src.data import load_mag_data
    from src.models import build_model

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("task") != "nc" or payload.get("protocol_version") != PROTOCOL:
        raise ValueError("checkpoint does not use the unified full-graph NC protocol")
    if int(payload.get("seed", -1)) != 42 or payload.get("selection") != "best_val_accuracy":
        raise ValueError("v0.5 smoke analysis requires the seed-42 Val-Accuracy-selected checkpoint")
    if any(key.startswith("test_") for key in payload.get("metrics", {})):
        raise ValueError("test metrics are forbidden in this workflow")
    if not any(key.startswith("imci_text.") for key in payload.get("model_state", {})):
        raise ValueError("checkpoint is not a v05_full model")

    device = torch.device(device_name if not device_name.startswith("cuda") or torch.cuda.is_available()
                          else "cpu")
    config = _config(dataset, str(device))
    data = load_mag_data(config, "nc", 42)
    info = _data_info(data)
    model = build_model(config, payload["data_info"]).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    classifier.load_state_dict(payload["head_state"], strict=True)
    model.eval()
    classifier.eval()

    parameter_audit = {}
    for variant in model.VARIANTS:
        variant_config = _config(dataset, "cpu", variant)
        variant_model = build_model(variant_config, info)
        parameter_audit[variant] = _parameter_audit(variant_model, int(data.num_classes))
        del variant_model

    x = data.x.to(device)
    edge = data.edge_index.to(device)
    p_rel = model.backbone._get_operators(edge, int(x.size(0)), x.dtype)[2]
    p_rel_indices = p_rel.indices().clone()
    p_rel_values = p_rel.values().clone()
    with torch.no_grad():
        diagnostic = model.analyze(
            x, edge, sample_seed=42042, max_diagnostic_edges=512,
            collect_edge_state=True, collect_attention=True,
        )
    p_rel_unchanged = torch.equal(p_rel.indices(), p_rel_indices) and torch.equal(
        p_rel.values(), p_rel_values
    )
    attention_sanity = {}
    for modality in ("text", "visual"):
        weights = diagnostic["attention_weights"][modality]
        mean_hop = diagnostic["mean_hop_attention"][modality]
        entropy = diagnostic["hop_attention_entropy"][modality]
        attention_sanity[modality] = {
            "attention_shape": list(weights.shape) if weights is not None else None,
            "max_hop_probability_sum_error": float(
                (weights.sum(dim=-1) - 1.0).abs().max().cpu()
            ) if weights is not None and weights.numel() else 0.0,
            "mean_hop_attention": mean_hop.detach().cpu().tolist() if mean_hop is not None else None,
            "mean_hop_attention_entropy": float(entropy.detach().cpu()) if entropy is not None else None,
        }
    crst_sanity = {
        modality: {
            "mean_abs_theta_radians": diagnostic["mean_abs_angle"][modality],
            "p95_abs_theta_radians": diagnostic["p95_abs_angle"][modality],
            "norm_preservation_max_abs_error": diagnostic[
                "norm_preservation_max_abs_error"][modality],
            "mean_cos_base_rotated": diagnostic["cos_base_rotated"][modality],
            "valid_context_edge_ratio": diagnostic["valid_context_edge_ratio"][modality],
        }
        for modality in ("text", "visual")
    }

    train_idx = data.train_idx.to(device=device, dtype=torch.long)
    labels = data.y.to(device=device, dtype=torch.long)
    gradient_audit, memory = _gradient_audit(
        model, classifier, x, edge, labels, train_idx, device,
    )
    finite = all(torch.isfinite(diagnostic[key]).all().item()
                 for key in ("H0_text", "H0_visual", "C1_text", "C2_text", "C3_text",
                             "C1_visual", "C2_visual", "C3_visual", "Z_text", "Z_visual",
                             "fused_z"))
    sanity = {
        "finite_embeddings": bool(finite),
        "physical_p_rel_unchanged": bool(p_rel_unchanged),
        "nonzero_rotation_angles": all(value["mean_abs_theta_radians"] > 0.0
                                        for value in crst_sanity.values()),
        "rotation_norm_preserved": all(value["norm_preservation_max_abs_error"] < 1e-4
                                        for value in crst_sanity.values()),
        "attention_probabilities_sum_to_one": all(
            value["max_hop_probability_sum_error"] < 1e-5
            for value in attention_sanity.values()
        ),
        "required_gradients_nonzero": all(
            gradient_audit[name]["nonzero"] for name in (
                "modality_relation_encoders", "shared_relation_encoder", "angle_heads",
                "multihead_attention", "imci_ffn",
            )
        ),
    }
    report = {
        "dataset": dataset, "seed": 42, "variant": "v05_full",
        "protocol_version": PROTOCOL, "checkpoint": str(checkpoint),
        "checkpoint_metrics": payload.get("metrics", {}),
        "device": str(device), "parameter_audit": parameter_audit,
        "smoke_gradient_audit": gradient_audit,
        "smoke_gradient_audit_loss": "train-split cross-entropy; no optimizer step",
        "gpu_memory": memory, "crst_sanity": crst_sanity,
        "imci_sanity": attention_sanity,
        "diagnostic_sample": {
            "seed": 42042, "edge_count": int(diagnostic["edge_indices"].numel()),
            "sampling": "deterministic random sample over P_rel edges",
        },
        "p_rel_unchanged": bool(p_rel_unchanged),
        "sanity": sanity, "sanity_passed": all(sanity.values()),
        "test_evaluated": False, "lp_evaluated": False,
    }
    result_dir = ROOT / "results/risa_v05/smoke"
    result_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}_seed42_v05_full"
    (result_dir / f"{stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        f"# RISA v0.5 smoke diagnostics: {dataset} seed 42", "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Checkpoint validation metrics: `{json.dumps(payload.get('metrics', {}), sort_keys=True)}`",
        f"- CRST sanity: `{json.dumps(crst_sanity, sort_keys=True)}`",
        f"- IMCI sanity: `{json.dumps(attention_sanity, sort_keys=True)}`",
        f"- Gradient audit: `{json.dumps(gradient_audit, sort_keys=True)}`",
        f"- Peak GPU allocation: `{memory['peak_allocated_bytes']}` bytes",
        f"- Physical P_rel unchanged: `{p_rel_unchanged}`",
        f"- Sanity checks: `{'PASS' if report['sanity_passed'] else 'FAIL'}`",
        "- Test and LP evaluation: disabled.", "",
    ]
    (result_dir / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")
    if not report["sanity_passed"]:
        raise RuntimeError(f"v0.5 smoke diagnostics failed; report saved to {result_dir / (stem + '.json')}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a RISA v0.5 validation-selected checkpoint")
    parser.add_argument("--dataset", choices=SCREEN_DATASETS, default="Movies")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "outputs/risa_v05_v1/smoke/Movies/v05_full/best.pt")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = analyze(args.checkpoint.resolve(), args.dataset, args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
