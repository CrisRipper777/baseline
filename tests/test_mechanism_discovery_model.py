from __future__ import annotations

import torch
import pytest
from omegaconf import OmegaConf

from src.models.multi_order_bank import Model


def _model(readout: str, modality: str = "both", fusion: str = "plain_mlp") -> Model:
    cfg = OmegaConf.create({"model": {
        "name": "multi_order_bank", "hidden_dim": 5, "max_order": 3,
        "num_layers": 3, "dropout": 0.0, "readout": readout,
        "modality_mode": modality, "fusion_mode": fusion,
    }})
    return Model(cfg, {"input_dim": 7, "text_dim": 3, "visual_dim": 4,
                       "num_nodes": 6, "num_classes": 3}).eval()


def _data():
    torch.manual_seed(4)
    x = torch.randn(6, 7)
    edges = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
    return x, edges


@pytest.mark.parametrize("name,weights", [
    ("self_only", (1, 0, 0, 0)),
    ("self25_terminal75", (.25, 0, 0, .75)),
    ("self50_terminal50", (.5, 0, 0, .5)),
    ("self75_terminal25", (.75, 0, 0, .25)),
    ("propagated_uniform", (0, 1/3, 1/3, 1/3)),
])
def test_fixed_order_source_readouts_are_exact(name, weights):
    model = _model(name)
    x, edges = _data()
    result = model.analyze(x, edges)
    for modality in ("text", "visual"):
        states = result[f"S_{modality}"]
        if name == "propagated_uniform":
            expected = (states[1] + states[2] + states[3]) / 3.0
        else:
            expected = sum(weight * state for weight, state in zip(weights, states, strict=True))
        assert torch.equal(result[f"Z_{modality}"], expected)


@pytest.mark.parametrize("mode,selected,ignored", [
    ("text", "text", "visual"), ("visual", "visual", "text")])
def test_unimodal_mode_output_uses_selected_branch_only(mode, selected, ignored):
    model = _model("uniform", modality=mode)
    x, edges = _data()
    before = model.analyze(x, edges)
    changed = x.clone()
    split = 3
    changed[:, split:] += 9 if mode == "text" else 0
    changed[:, :split] += 0 if mode == "text" else 9
    after = model.analyze(changed, edges)
    assert torch.equal(before["fused_z"], before[f"Z_{selected}"])
    assert before["fusion_input"] is None
    assert torch.equal(before[f"Z_{selected}"], after[f"Z_{selected}"])
    assert before[f"Z_{ignored}"] is None
    assert before[f"S_{ignored}"] is None


def test_existing_both_mode_readouts_keep_identical_parameter_layout_and_load_strictly():
    layouts = []
    states = []
    for readout in ("terminal", "uniform", "gpr"):
        model = _model(readout)
        layouts.append(tuple(model.state_dict().keys()))
        states.append({key: value.clone() for key, value in model.state_dict().items()})
    assert layouts[0] == layouts[1]
    assert set(layouts[2]) - {"gamma_text", "gamma_visual"} == set(layouts[0])
    legacy = _model("terminal")
    assert legacy.load_state_dict(states[0], strict=True).missing_keys == []
    assert legacy.modality_mode == "both"


def test_fixed_readout_variants_share_same_fusion_parameter_layout():
    expected = tuple(_model("terminal").state_dict().keys())
    for readout in ("self_only", "self25_terminal75", "self50_terminal50", "self75_terminal25", "propagated_uniform"):
        assert tuple(_model(readout).state_dict().keys()) == expected


def test_multimodal_and_unimodal_modes_have_finite_forward_backward_and_sampled_graph_interface():
    x, edges = _data()
    for mode in ("both", "text", "visual"):
        model = _model("uniform", modality=mode, fusion="plain_mlp")
        z, _, _, aux_loss, _ = model(x, edges)
        loss = z.square().mean() + aux_loss
        loss.backward()
        assert z.shape == (x.size(0), 5)
        assert torch.isfinite(z).all() and torch.isfinite(loss)
        # A LinkNeighborLoader subgraph presents the same [features, edge_index] API.
        sub_z, *_ = model(x[:4], edges[:, :4].clamp_max(3))
        assert sub_z.shape == (4, 5) and torch.isfinite(sub_z).all()


def test_existing_saved_uniform_checkpoint_state_loads_strictly_when_available():
    from pathlib import Path
    import pytest
    checkpoint_path = Path("outputs/mob_factorial_nc_v1/Movies/mob_uniform_plain/best_run1.pt")
    if not checkpoint_path.is_file():
        pytest.skip("immutable benchmark checkpoint is not present in this checkout")
    from omegaconf import OmegaConf
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create({"model": OmegaConf.load("configs/model/multi_order_bank.yaml")})
    cfg.model.readout = "uniform"
    model = Model(cfg, payload["data_info"])
    model.load_state_dict(payload["model_state"], strict=True)
    head = torch.nn.Linear(model.out_dim, int(payload["data_info"]["num_classes"]))
    head.load_state_dict(payload["head_state"], strict=True)


def test_operator_cache_retains_edge_tensor_and_rebuilds_for_an_equal_shape_graph():
    model = _model("uniform")
    x, edges = _data()
    first = model._get_propagation_operator(edges, x.size(0), x.dtype)
    cached_input = model._operator_cache_edge_index
    alternate = edges.clone()
    alternate[:, 0] = torch.tensor([0, 5])
    second = model._get_propagation_operator(alternate, x.size(0), x.dtype)
    assert cached_input is edges
    assert model._operator_cache_edge_index is alternate
    assert not torch.equal(first.to_dense(), second.to_dense())
