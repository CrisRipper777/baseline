from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.models.multi_order_bank import Model
from src.tasks.common import resolve_num_neighbors
from src.tasks.lp import _resolve_lp_num_neighbors
from src.tasks.nc import _resolve_training_mode


NUM_NODES = 4
TEXT_DIM = 2
VISUAL_DIM = 2
HIDDEN = 4


def _cfg(readout: str, fusion_mode: str = "plain_mlp"):
    return OmegaConf.create(
        {
            "model": {
                "name": "multi_order_bank",
                "hidden_dim": HIDDEN,
                "max_order": 3,
                "num_layers": 3,
                "dropout": 0.0,
                "readout": readout,
                "fusion_mode": fusion_mode,
            }
        }
    )


def _inputs():
    x = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.5, 0.5],
            [0.5, 0.5, 1.0, 1.0],
        ]
    )
    # Intentionally provide one orientation plus a self-loop; the operator
    # must produce the standard undirected A+I exactly once.
    edge_index = torch.tensor([[0, 1, 2, 2], [1, 2, 2, 3]], dtype=torch.long)
    return x, edge_index


def _model(readout: str, fusion_mode: str = "plain_mlp") -> Model:
    return Model(
        _cfg(readout, fusion_mode),
        {
            "input_dim": TEXT_DIM + VISUAL_DIM,
            "num_nodes": NUM_NODES,
            "num_classes": 2,
            "text_dim": TEXT_DIM,
            "visual_dim": VISUAL_DIM,
        },
    ).eval()


def test_mult_order_bank_fixes_max_order_three_and_keeps_s0() -> None:
    model = _model("terminal")
    x, edge_index = _inputs()
    analysis = model.analyze(x, edge_index)

    assert model.max_order == 3
    assert len(analysis["S_text"]) == 4
    assert len(analysis["S_visual"]) == 4
    assert torch.equal(analysis["S_text"][0], analysis["H0_text"])
    assert torch.equal(analysis["S_visual"][0], analysis["H0_visual"])


def test_propagation_states_are_exact_repeated_physical_graph_products() -> None:
    model = _model("uniform")
    x, edge_index = _inputs()
    analysis = model.analyze(x, edge_index)
    operator = model._get_propagation_operator(edge_index, NUM_NODES, x.dtype)

    for modality in ("text", "visual"):
        states = analysis[f"S_{modality}"]
        assert torch.allclose(states[1], torch.sparse.mm(operator, states[0]))
        assert torch.allclose(states[2], torch.sparse.mm(operator, states[1]))
        assert torch.allclose(states[3], torch.sparse.mm(operator, states[2]))


def test_physical_operator_is_symmetric_normalized_a_plus_identity() -> None:
    model = _model("terminal")
    _, edge_index = _inputs()
    operator = model._build_propagation_operator(edge_index, NUM_NODES, torch.float32)

    adjacency = torch.eye(NUM_NODES)
    adjacency[0, 1] = adjacency[1, 0] = 1.0
    adjacency[1, 2] = adjacency[2, 1] = 1.0
    adjacency[2, 3] = adjacency[3, 2] = 1.0
    degree = adjacency.sum(dim=1)
    inv_sqrt = degree.pow(-0.5)
    expected = inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    assert torch.allclose(operator.to_dense(), expected)


def test_terminal_and_uniform_readouts_are_exact() -> None:
    x, edge_index = _inputs()
    terminal = _model("terminal").analyze(x, edge_index)
    uniform = _model("uniform").analyze(x, edge_index)

    for modality in ("text", "visual"):
        assert torch.equal(terminal[f"Z_{modality}"], terminal[f"S_{modality}"][3])
        expected = sum(uniform[f"S_{modality}"]) / 4.0
        assert torch.allclose(uniform[f"Z_{modality}"], expected)


def test_gpr_is_an_exact_signed_weighted_sum_without_softmax() -> None:
    model = _model("gpr")
    with torch.no_grad():
        model.gamma_text.copy_(torch.tensor([-0.5, 0.1, 0.2, 1.2]))
        model.gamma_visual.copy_(torch.tensor([0.7, -0.3, 0.4, 0.2]))
    x, edge_index = _inputs()
    analysis = model.analyze(x, edge_index)

    assert torch.equal(model.gamma_text, torch.tensor([-0.5, 0.1, 0.2, 1.2]))
    for modality in ("text", "visual"):
        expected = sum(
            analysis[f"gamma_{modality}"][k] * analysis[f"S_{modality}"][k]
            for k in range(4)
        )
        assert torch.allclose(analysis[f"Z_{modality}"], expected)


def test_gpr_gamma_initialization_and_independent_parameters() -> None:
    model = _model("gpr")
    expected = torch.full((4,), 0.25)
    assert torch.equal(model.gamma_text.detach(), expected)
    assert torch.equal(model.gamma_visual.detach(), expected)
    assert model.gamma_text is not model.gamma_visual
    assert model.gamma_text.data_ptr() != model.gamma_visual.data_ptr()


def test_changing_text_gamma_does_not_change_visual_pre_fusion_readout() -> None:
    model = _model("gpr")
    x, edge_index = _inputs()
    before = model.analyze(x, edge_index)
    with torch.no_grad():
        model.gamma_text.add_(torch.tensor([0.4, -0.2, 0.1, 0.3]))
    after = model.analyze(x, edge_index)
    assert torch.equal(before["Z_visual"], after["Z_visual"])
    assert not torch.equal(before["Z_text"], after["Z_text"])


def test_changing_visual_gamma_does_not_change_text_pre_fusion_readout() -> None:
    model = _model("gpr")
    x, edge_index = _inputs()
    before = model.analyze(x, edge_index)
    with torch.no_grad():
        model.gamma_visual.add_(torch.tensor([-0.2, 0.4, 0.3, -0.1]))
    after = model.analyze(x, edge_index)
    assert torch.equal(before["Z_text"], after["Z_text"])
    assert not torch.equal(before["Z_visual"], after["Z_visual"])


def test_modality_states_do_not_cross_propagate() -> None:
    model = _model("gpr")
    x, edge_index = _inputs()
    before = model.analyze(x, edge_index)
    changed_text = x.clone()
    changed_text[:, :TEXT_DIM] *= -3.0
    after_text_change = model.analyze(changed_text, edge_index)
    assert torch.equal(before["H0_visual"], after_text_change["H0_visual"])
    assert all(
        torch.equal(a, b)
        for a, b in zip(before["S_visual"], after_text_change["S_visual"], strict=True)
    )
    assert torch.equal(before["Z_visual"], after_text_change["Z_visual"])

    changed_visual = x.clone()
    changed_visual[:, TEXT_DIM:] *= 2.0
    after_visual_change = model.analyze(changed_visual, edge_index)
    assert torch.equal(before["H0_text"], after_visual_change["H0_text"])
    assert all(
        torch.equal(a, b)
        for a, b in zip(before["S_text"], after_visual_change["S_text"], strict=True)
    )


def test_readouts_share_projector_and_fusion_parameter_shapes() -> None:
    for fusion_mode in ("plain_mlp", "residual"):
        models = [_model(name, fusion_mode) for name in ("terminal", "uniform", "gpr")]
        layouts = [
            {
                name: tuple(value.shape)
                for name, value in model.state_dict().items()
                if not name.startswith("gamma_")
            }
            for model in models
        ]
        assert layouts[0] == layouts[1] == layouts[2]
        for model in models:
            assert isinstance(model.text_projector[0], nn.Linear)
            assert isinstance(model.text_projector[1], nn.LayerNorm)
            assert isinstance(model.visual_projector[0], nn.Linear)
            assert isinstance(model.visual_projector[1], nn.LayerNorm)
        if fusion_mode == "plain_mlp":
            assert isinstance(models[0].plain_fusion[0], nn.Linear)
            assert isinstance(models[0].plain_fusion[1], nn.ReLU)
            assert isinstance(models[0].plain_fusion[2], nn.Dropout)
            assert isinstance(models[0].plain_fusion[3], nn.Linear)
            assert not hasattr(models[0], "output_norm")
        else:
            assert isinstance(models[0].fusion_skip, nn.Linear)
            assert isinstance(models[0].output_norm, nn.LayerNorm)
            assert models[0].text_refine_mlp is not models[0].visual_refine_mlp
            assert models[0].text_refine_norm is not models[0].visual_refine_norm


def test_plain_mlp_fusion_matches_exact_concat_mlp_formula() -> None:
    model = _model("terminal", "plain_mlp")
    x, edge_index = _inputs()
    result = model.analyze(x, edge_index)
    u = torch.cat([result["Z_text"], result["Z_visual"]], dim=-1)
    layer1, _, dropout, layer2 = model.plain_fusion
    expected = layer2(dropout(torch.relu(layer1(u))))
    assert torch.equal(result["refined_text"], result["Z_text"])
    assert torch.equal(result["refined_visual"], result["Z_visual"])
    assert torch.equal(result["fusion_input"], u)
    assert torch.allclose(result["fused_z"], expected)


def test_residual_fusion_matches_exact_late_fusion_formula() -> None:
    model = _model("uniform", "residual")
    x, edge_index = _inputs()
    result = model.analyze(x, edge_index)
    expected_t = model.text_refine_norm(
        result["Z_text"] + model.text_refine_mlp(result["Z_text"])
    )
    expected_v = model.visual_refine_norm(
        result["Z_visual"] + model.visual_refine_mlp(result["Z_visual"])
    )
    u = torch.cat([expected_t, expected_v], dim=-1)
    expected_z = model.output_norm(
        model.fusion_skip(u) + model.fusion_mlp(u)
    )
    assert torch.allclose(result["refined_text"], expected_t)
    assert torch.allclose(result["refined_visual"], expected_v)
    assert torch.allclose(result["fusion_input"], u)
    assert torch.allclose(result["fused_z"], expected_z)


def test_residual_modality_refinement_is_independent_until_concat() -> None:
    model = _model("gpr", "residual")
    text = torch.randn(NUM_NODES, HIDDEN)
    visual = torch.randn(NUM_NODES, HIDDEN)
    _, visual_before, _, _ = model._fuse_modalities(text, visual)
    _, visual_after, _, _ = model._fuse_modalities(text + 3.0, visual)
    assert torch.equal(visual_before, visual_after)


def test_readout_and_fusion_choices_do_not_change_propagation_states() -> None:
    x, edge_index = _inputs()
    analyses = []
    for readout, fusion_mode in (
        ("terminal", "plain_mlp"),
        ("uniform", "residual"),
        ("gpr", "plain_mlp"),
        ("terminal", "residual"),
    ):
        torch.manual_seed(90210)
        model = _model(readout, fusion_mode)
        analyses.append(model.analyze(x, edge_index))
    for key in ("H0_text", "H0_visual", "S_text", "S_visual"):
        first = analyses[0][key]
        if isinstance(first, list):
            assert all(
                torch.equal(a, b)
                for other in analyses[1:]
                for a, b in zip(first, other[key], strict=True)
            )
        else:
            assert all(torch.equal(first, other[key]) for other in analyses[1:])


def test_no_attention_or_auxiliary_objective_and_forward_is_finite() -> None:
    model = _model("gpr")
    x, edge_index = _inputs()
    z, aux_a, aux_b, aux_loss, aux_info = model(x, edge_index)
    assert z.shape == (NUM_NODES, HIDDEN)
    assert torch.isfinite(z).all()
    assert aux_a is None and aux_b is None
    assert aux_loss.item() == 0.0
    assert aux_info == {}
    assert not hasattr(model, "attention")
    assert not any(isinstance(module, nn.MultiheadAttention) for module in model.modules())


def test_linear_debug_fusion_remains_available() -> None:
    model = _model("terminal", "linear")
    x, edge_index = _inputs()
    analysis = model.analyze(x, edge_index)
    expected = model.fusion(torch.cat([analysis["Z_text"], analysis["Z_visual"]], dim=-1))
    assert torch.allclose(analysis["fused_z"], expected)


def test_forward_backward_gradients_are_finite() -> None:
    model = Model(_cfg("gpr"), {
        "input_dim": TEXT_DIM + VISUAL_DIM,
        "num_nodes": NUM_NODES,
        "num_classes": 2,
        "text_dim": TEXT_DIM,
        "visual_dim": VISUAL_DIM,
    })
    x, edge_index = _inputs()
    z, _, _, aux_loss, _ = model(x, edge_index)
    (z.square().mean() + aux_loss).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_full_graph_nc_and_sampled_lp_interfaces() -> None:
    cfg = _cfg("terminal")
    model = _model("terminal")
    cfg.task = OmegaConf.create({"training_mode": "full_graph"})
    assert _resolve_training_mode(cfg, model) == "full_graph"
    cfg.task = OmegaConf.create({"num_neighbors": [5, 5, 5]})
    assert _resolve_lp_num_neighbors(cfg, model) == [5, 5, 5]

    x, edge_index = _inputs()
    sampled_z, _, _, _, _ = model(x[:3], edge_index[:, (edge_index[0] < 3) & (edge_index[1] < 3)])
    assert sampled_z.shape == (3, HIDDEN)
    assert torch.isfinite(sampled_z).all()
    assert resolve_num_neighbors(
        OmegaConf.create({"model": {"num_layers": 3}, "task": {"num_neighbors": [5, 5, 5]}})
    ) == [5, 5, 5]


def test_inference_returns_finite_cpu_embeddings() -> None:
    model = _model("uniform")
    x, edge_index = _inputs()
    inferred = model.inference(x, edge_index, device=torch.device("cpu"), batch_size=2)
    assert inferred.shape == (NUM_NODES, HIDDEN)
    assert torch.isfinite(inferred).all()


def test_non_three_order_config_is_rejected() -> None:
    cfg = _cfg("terminal")
    cfg.model.max_order = 2
    try:
        Model(
            cfg,
            {
                "input_dim": TEXT_DIM + VISUAL_DIM,
                "text_dim": TEXT_DIM,
                "visual_dim": VISUAL_DIM,
            },
        )
    except ValueError as exc:
        assert "fixes max_order=3" in str(exc)
    else:
        raise AssertionError("max_order other than 3 must be rejected")
