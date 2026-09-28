from __future__ import annotations

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from src.models.risa_v04 import Model as RisaP0
from src.models.relcal_statepres_pilot import Model as S45Model


def _cfg(variant: str, chunk: int = 3):
    return SimpleNamespace(model=OmegaConf.create({
        "name": "risa_v04", "hidden_dim": 256, "dropout": 0.2,
        "max_order": 3, "relation_dim": 32, "bottleneck_dim": 32,
        "num_operators": 4, "edge_chunk_size": chunk, "variant": variant,
    }))


def _info():
    return {"input_dim": 12, "text_dim": 5, "visual_dim": 7,
            "num_nodes": 7, "num_classes": 3}


def _graph():
    torch.manual_seed(1307)
    x = torch.randn(7, 12)
    edge_index = torch.tensor([
        [0, 0, 1, 2, 2, 3, 4, 5, 6, 1],
        [1, 2, 2, 0, 3, 4, 5, 4, 6, 5],
    ])
    return x, edge_index


def _s45_identity():
    cfg = SimpleNamespace(model=OmegaConf.create({
        "hidden_dim": 256, "dropout": 0.2, "max_order": 3,
        "relation_dim": 32, "edge_chunk_size": 3,
        "variant": "s45_identity_uniform",
    }))
    return S45Model(cfg, _info()).eval()


def test_delta_off_is_numerically_s45_identity_uniform():
    x, edge = _graph()
    torch.manual_seed(19)
    model = RisaP0(_cfg("p0_operator_routed"), _info()).eval()
    reference = _s45_identity()
    reference.load_state_dict(model.backbone.state_dict(), strict=True)
    actual = model.analyze(x, edge, intervention="delta_off", return_edge_state=False)
    expected = reference.analyze(x, edge)
    for key in ("H0_text", "H0_visual", "fused_z"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    for modality in ("text", "visual"):
        for left, right in zip(actual[f"C_{modality}"], expected[f"S_{modality}"], strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_router_never_changes_physical_p_rel_values():
    x, edge = _graph()
    model = RisaP0(_cfg("p0_operator_routed"), _info()).eval()
    p_rel = model.backbone._get_operators(edge, x.size(0), x.dtype)[2]
    indices_before, values_before = p_rel.indices().clone(), p_rel.values().clone()
    result = model.analyze(x, edge)
    torch.testing.assert_close(result["P_rel"].indices(), indices_before, rtol=0, atol=0)
    torch.testing.assert_close(result["P_rel"].values(), values_before, rtol=0, atol=0)
    torch.testing.assert_close(result["P_rel"].values(), p_rel.values(), rtol=0, atol=0)


def test_uniform_global_and_routed_match_at_initialization():
    x, edge = _graph()
    models = {}
    for variant in ("p0_operator_uniform", "p0_operator_global", "p0_operator_routed"):
        torch.manual_seed(71)
        models[variant] = RisaP0(_cfg(variant), _info()).eval()
    uniform = models["p0_operator_uniform"]
    global_model = models["p0_operator_global"]
    routed = models["p0_operator_routed"]
    for modality in ("text", "visual"):
        banks = [getattr(model, f"operators_{modality}") for model in models.values()]
        for bank in banks[1:]:
            for left, right in zip(banks[0], bank, strict=True):
                for a, b in zip(left.parameters(), right.parameters(), strict=True):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(getattr(global_model, f"theta_{modality}"),
                                   torch.zeros(4), rtol=0, atol=0)
        assert torch.count_nonzero(getattr(routed, f"router_{modality}").weight) == 0
        assert torch.count_nonzero(getattr(routed, f"router_{modality}").bias) == 0
        assert not hasattr(global_model, f"relation_encoder_{modality}")
    outputs = [model.analyze(x, edge)["fused_z"] for model in models.values()]
    for output in outputs[1:]:
        torch.testing.assert_close(outputs[0], output, rtol=0, atol=0)
    for modality in ("text", "visual"):
        for model in (global_model, routed):
            probs = model.analyze(x, edge)["router_probabilities"][modality]
            torch.testing.assert_close(probs, torch.full_like(probs, 0.25), rtol=0, atol=0)


def test_global_operator_mixture_is_identical_across_all_edges():
    x, edge = _graph()
    model = RisaP0(_cfg("p0_operator_global"), _info()).eval()
    with torch.no_grad():
        model.theta_text.copy_(torch.tensor([-2.0, -0.5, 1.0, 2.5]))
        model.theta_visual.copy_(torch.tensor([2.0, 0.0, -1.0, -3.0]))
    result = model.analyze(x, edge, return_edge_state=True)
    for modality in ("text", "visual"):
        expected = torch.softmax(getattr(model, f"theta_{modality}"), dim=0)
        probs = result["router_probabilities"][modality]
        torch.testing.assert_close(probs, expected.expand_as(probs), rtol=0, atol=0)
        torch.testing.assert_close(probs.std(dim=0, unbiased=False), torch.zeros(4), rtol=0, atol=0)


def test_delta_parallel_orthogonal_decomposition():
    x, edge = _graph()
    model = RisaP0(_cfg("p0_operator_routed"), _info()).eval()
    with torch.no_grad():
        model.operators_text[0].up.weight.normal_(std=0.02)
    result = model.analyze(x, edge, return_edge_state=True)
    for modality in ("text", "visual"):
        base = result["base_edge_message"][modality]
        delta = result["operator_correction_delta"][modality]
        denom = base.square().sum(dim=-1, keepdim=True).clamp_min(1e-12)
        parallel = (delta * base).sum(dim=-1, keepdim=True) / denom * base
        orthogonal = delta - parallel
        torch.testing.assert_close(delta, parallel + orthogonal, rtol=1e-6, atol=1e-7)
        assert result["parallel_orthogonal"][modality]["orthogonal_norm_ratio"] >= 0.0


def test_chunked_edge_computation_matches():
    x, edge = _graph()
    torch.manual_seed(83)
    small = RisaP0(_cfg("p0_operator_routed", chunk=2), _info()).eval()
    large = RisaP0(_cfg("p0_operator_routed", chunk=19), _info()).eval()
    large.load_state_dict(small.state_dict(), strict=True)
    left = small.analyze(x, edge, return_edge_state=True)
    right = large.analyze(x, edge, return_edge_state=True)
    torch.testing.assert_close(left["fused_z"], right["fused_z"], rtol=1e-6, atol=1e-6)
    for modality in ("text", "visual"):
        for key in ("router_probabilities", "base_edge_message", "operator_correction_delta",
                    "corrected_edge_message"):
            torch.testing.assert_close(left[key][modality], right[key][modality], rtol=1e-6, atol=1e-6)


def test_relation_and_operator_parameters_receive_gradients():
    x, edge = _graph()
    model = RisaP0(_cfg("p0_operator_routed"), _info())
    with torch.no_grad():
        model.router_text.weight.normal_(std=0.03)
        model.router_visual.weight.normal_(std=0.03)
    output, *_ = model(x, edge)
    output.square().mean().backward()
    for modality in ("text", "visual"):
        for name, parameter in getattr(model, f"relation_encoder_{modality}").named_parameters():
            assert parameter.grad is not None, f"missing relation gradient: {modality}.{name}"
        for operator_index, operator in enumerate(getattr(model, f"operators_{modality}")):
            for name, parameter in operator.named_parameters():
                assert parameter.grad is not None, f"missing operator gradient: {modality}.{operator_index}.{name}"
    assert sum(float(p.grad.abs().sum()) for n, p in model.named_parameters()
               if "relation_encoder" in n and p.grad is not None) > 0.0
    assert sum(float(p.grad.abs().sum()) for n, p in model.named_parameters()
               if "operators_" in n and p.grad is not None) > 0.0


def test_targeted_interventions_keep_selected_edge_alignment():
    x, edge = _graph()
    model = RisaP0(_cfg("p0_operator_routed"), _info()).eval()
    targets = torch.tensor([1, 3, 6])
    normal = model.analyze(x, edge, target_nodes=targets, return_edge_state=True)
    uniform = model.analyze(x, edge, target_nodes=targets, intervention="router_uniform",
                            return_edge_state=True)
    assert normal["edge_row"].numel() == normal["base_edge_message"]["text"].size(0)
    torch.testing.assert_close(uniform["router_probabilities"]["text"],
                               torch.full_like(uniform["router_probabilities"]["text"], 0.25))
    shuffled = model.analyze(x, edge, target_nodes=targets, intervention="router_shuffle",
                             shuffle_seed=4, return_edge_state=True)
    assert shuffled["router_probabilities"]["text"].shape == normal["router_probabilities"]["text"].shape


def test_variant_surface_is_fixed():
    assert RisaP0.VARIANTS == (
        "p0_identity", "p0_masspres_scalar", "p0_single_dynamic_transform",
        "p0_operator_uniform", "p0_operator_global", "p0_operator_routed",
    )


def test_all_variants_run_and_delta_off_is_defined():
    x, edge = _graph()
    for variant in RisaP0.VARIANTS:
        model = RisaP0(_cfg(variant), _info()).eval()
        result = model.analyze(x, edge, return_edge_state=True)
        assert result["fused_z"].shape == (x.size(0), 256)
        assert torch.isfinite(result["fused_z"]).all()
        assert result["C1_text"].shape == (x.size(0), 256)
        off = model.analyze(x, edge, intervention="delta_off", return_edge_state=False)
        identity = model.backbone.analyze(x, edge, gate_override=("off" if variant == "p0_masspres_scalar" else None))
        torch.testing.assert_close(off["fused_z"], identity["fused_z"], rtol=0, atol=0)


def test_router_mean_uses_full_graph_edge_marginal():
    x, edge = _graph()
    model = RisaP0(_cfg("p0_operator_routed"), _info()).eval()
    with torch.no_grad():
        model.router_text.weight.normal_(std=0.2)
        model.router_visual.weight.normal_(std=0.2)
    raw = model.router_probabilities_for_edges(x, edge)
    result = model.analyze(x, edge, intervention="router_mean", return_edge_state=True)
    for modality in ("text", "visual"):
        expected = raw[modality].mean(dim=0)
        torch.testing.assert_close(result["router_probabilities"][modality],
                                   expected.expand_as(result["router_probabilities"][modality]),
                                   rtol=1e-6, atol=1e-7)
        stats = result["parallel_orthogonal"][modality]
        assert "orthogonal_fraction" in stats
        assert result["router_probability_std"][modality].abs().max().item() == 0.0
        assert abs(result["mean_kl_to_mean_router"][modality]) < 1e-7


def test_disabling_training_diagnostics_does_not_change_embeddings():
    x, edge = _graph()
    model = RisaP0(_cfg("p0_operator_routed"), _info()).eval()
    normal = model.analyze(x, edge, return_edge_state=False)
    lean = model.analyze(x, edge, return_edge_state=False, collect_diagnostics=False)
    torch.testing.assert_close(normal["fused_z"], lean["fused_z"], rtol=0, atol=0)
