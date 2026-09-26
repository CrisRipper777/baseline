from __future__ import annotations

from types import SimpleNamespace

import torch

from src.models.relation_basis_pilot import Model as P0Model
from src.models.relational_transform_pilot import Model


def _cfg(variant: str, chunk: int = 3):
    return SimpleNamespace(model={"hidden_dim": 256, "dropout": 0.2, "relation_dim": 32,
                                 "lowrank_rank": 8, "edge_chunk_size": chunk, "variant": variant})


def _graph():
    torch.manual_seed(18)
    x = torch.randn(7, 12)
    edges = torch.tensor([[0, 0, 1, 2, 2, 3, 4, 5, 6, 6, 1],
                          [1, 2, 2, 0, 3, 4, 5, 6, 0, 4, 5]])
    return x, edges


def _copy_shared(source: torch.nn.Module, target: torch.nn.Module):
    for name in ("text_projector", "visual_projector", "transform_a_text", "transform_a_visual",
                 "output_norm_text", "output_norm_visual", "plain_fusion"):
        getattr(target, name).load_state_dict(getattr(source, name).state_dict())


def test_variants_forward_backward_and_controller_initialization():
    x, edges = _graph()
    for variant in Model.VARIANTS:
        model = Model(_cfg(variant), {"text_dim": 5, "visual_dim": 7, "input_dim": 12})
        model.eval()
        output = model.analyze(x, edges, return_edge_state=True)
        assert output["fused_z"].shape == (7, 256)
        assert torch.isfinite(output["fused_z"]).all()
        if variant in Model.SCALAR_EDGE:
            for controller in output["controllers"].values():
                torch.testing.assert_close(controller, torch.ones_like(controller))
        if variant in Model.LOWRANK:
            for controller in output["controllers"].values():
                torch.testing.assert_close(controller, torch.zeros_like(controller))
        if variant in Model.EXPERT:
            for controller in output["controllers"].values():
                torch.testing.assert_close(controller, torch.full_like(controller, 0.5))
        loss = output["fused_z"].square().mean()
        loss.backward()


def test_lowrank_zero_dynamic_matches_historical_p0_residual_carrier():
    x, edges = _graph()
    torch.manual_seed(2026)
    dynamic = Model(_cfg("s44_lowrank_rel"), {"text_dim": 5, "visual_dim": 7, "input_dim": 12})
    reference = P0Model(_cfg("p0_residual"), {"text_dim": 5, "visual_dim": 7, "input_dim": 12})
    _copy_shared(dynamic, reference)
    dynamic.eval()
    reference.eval()
    e = dynamic._get_p_rel(edges, x.size(0), x.dtype)._nnz()
    zero = {"text": torch.zeros(e, 8), "visual": torch.zeros(e, 8)}
    actual = dynamic.analyze(x, edges, controller_override=zero)["fused_z"]
    expected = reference.analyze(x, edges)["fused_z"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_chunked_edge_computation_matches_larger_chunks():
    x, edges = _graph()
    for variant in ("s44_scalar_rel", "s44_lowrank_rel", "s44_expert_rel", "s44_scalar_rel_multi"):
        small = Model(_cfg(variant, 2), {"text_dim": 5, "visual_dim": 7, "input_dim": 12})
        large = Model(_cfg(variant, 9), {"text_dim": 5, "visual_dim": 7, "input_dim": 12})
        large.load_state_dict(small.state_dict())
        small.eval()
        large.eval()
        a = small.analyze(x, edges)["fused_z"]
        b = large.analyze(x, edges)["fused_z"]
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)


def test_multi_relation_remove_and_swap_are_defined():
    x, edges = _graph()
    model = Model(_cfg("s44_lowrank_rel_multi"), {"text_dim": 5, "visual_dim": 7, "input_dim": 12})
    torch.nn.init.normal_(model.lowrank_ctrl_text.weight, std=0.1)
    torch.nn.init.normal_(model.lowrank_ctrl_visual.weight, std=0.1)
    model.eval()
    normal = model.analyze(x, edges, return_edge_state=True)
    remove = model.analyze(x, edges, cross_modal_mode="remove", return_edge_state=True)
    swap = model.analyze(x, edges, cross_modal_mode="swap", return_edge_state=True)
    assert normal["controllers"]["text"].shape == remove["controllers"]["text"].shape
    assert normal["controllers"]["text"].shape == swap["controllers"]["text"].shape
    assert not torch.allclose(normal["controllers"]["text"], remove["controllers"]["text"])


def test_target_node_analysis_preserves_validation_logits_and_edge_order():
    x, edges = _graph()
    targets = torch.tensor([1, 3, 6])
    for variant in ("s44_scalar_rel", "s44_lowrank_rel_multi", "s44_expert_rel"):
        model = Model(_cfg(variant), {"text_dim": 5, "visual_dim": 7, "input_dim": 12})
        model.eval()
        full = model.analyze(x, edges, return_edge_state=True)
        targeted = model.analyze(x, edges, target_nodes=targets, return_edge_state=True)
        torch.testing.assert_close(full["fused_z"][targets], targeted["fused_z"][targets], rtol=2e-6, atol=2e-6)
        selected = torch.isin(full["edge_row"], targets)
        assert torch.equal(full["edge_row"][selected], targeted["edge_row"])
        for modality in ("text", "visual"):
            torch.testing.assert_close(full["controllers"][modality][selected],
                                       targeted["controllers"][modality], rtol=0, atol=0)


def test_paired_dynamic_variants_share_adapter_initialization():
    info = {"text_dim": 5, "visual_dim": 7, "input_dim": 12}
    pairs = (("s44_lowrank_global", "s44_lowrank_rel", ("v_text", "u_text", "v_visual", "u_visual")),
             ("s44_expert_uniform", "s44_expert_rel",
              ("expert_v1_text", "expert_v2_text", "expert_u1_text", "expert_u2_text",
               "expert_v1_visual", "expert_v2_visual", "expert_u1_visual", "expert_u2_visual")))
    for left, right, names in pairs:
        torch.manual_seed(715)
        model_left = Model(_cfg(left), info)
        torch.manual_seed(715)
        model_right = Model(_cfg(right), info)
        for name in names:
            torch.testing.assert_close(getattr(model_left, name).weight,
                                       getattr(model_right, name).weight, rtol=0, atol=0)
