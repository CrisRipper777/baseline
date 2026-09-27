from __future__ import annotations

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from src.models.multi_order_bank import Model as MultiOrderBank
from src.models.relcal_statepres_pilot import Model as RelCalStatePres, _SparseValueMatmul


def _cfg(variant: str):
    return SimpleNamespace(model={"name": "relcal_statepres_pilot", "hidden_dim": 256,
                                  "dropout": 0.2, "max_order": 3, "relation_dim": 32,
                                  "edge_chunk_size": 3, "variant": variant})


def _info():
    return {"input_dim": 12, "text_dim": 5, "visual_dim": 7, "num_nodes": 7, "num_classes": 3}


def _graph():
    torch.manual_seed(1307)
    x = torch.randn(7, 12)
    # Directed input, duplicated pair, raw self-loop, and isolated node 6.
    edge_index = torch.tensor([[0, 0, 1, 2, 2, 3, 4, 5, 6, 1],
                               [1, 2, 2, 0, 3, 4, 5, 4, 6, 5]])
    return x, edge_index


def _mob(readout: str):
    cfg = SimpleNamespace(model=OmegaConf.create({"name": "multi_order_bank", "hidden_dim": 256,
                                 "max_order": 3, "dropout": 0.2, "readout": readout,
                                 "fusion_mode": "plain_mlp", "modality_mode": "both"}))
    return MultiOrderBank(cfg, _info()).eval()


def test_identity_readouts_exactly_match_historical_mob():
    x, edge_index = _graph()
    for variant, readout in (("s45_identity_terminal", "terminal"),
                             ("s45_identity_uniform", "uniform"),
                             ("s45_identity_propagated_uniform", "propagated_uniform")):
        torch.manual_seed(24)
        pilot = RelCalStatePres(_cfg(variant), _info()).eval()
        torch.manual_seed(24)
        historical = _mob(readout)
        historical.load_state_dict(pilot.state_dict(), strict=True)
        actual = pilot.analyze(x, edge_index)
        expected = historical.analyze(x, edge_index)
        for key in ("H0_text", "H0_visual", "Z_text", "Z_visual", "fused_z"):
            historical_key = "H0_text" if key == "H0_text" else key
            torch.testing.assert_close(actual[key], expected[historical_key], rtol=0, atol=0)
        for modality in ("text", "visual"):
            for a, b in zip(actual[f"S_{modality}"], expected[f"S_{modality}"], strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert set(pilot.state_dict()) == set(historical.state_dict())


def test_original_operator_split_and_diagonal_support():
    x, edge_index = _graph()
    model = RelCalStatePres(_cfg("s45_identity_uniform"), _info()).eval()
    result = model.analyze(x, edge_index)
    torch.testing.assert_close((result["P_self"] + result["P_rel"]).to_dense(),
                               result["P"].to_dense(), rtol=0, atol=0)
    idx = result["P_self"].indices()
    assert torch.equal(idx[0], idx[1])
    assert not bool((result["P_rel"].indices()[0] == result["P_rel"].indices()[1]).any())


def test_gate_initialization_and_independent_modalities():
    x, edge_index = _graph()
    model = RelCalStatePres(_cfg("s45_masspres_entry_uniform"), _info()).eval()
    output = model.analyze(x, edge_index)
    for name in ("text", "visual"):
        torch.testing.assert_close(output["raw_gates"][name],
                                   torch.ones_like(output["raw_gates"][name]), rtol=0, atol=0)
    assert model.q_text is not model.q_visual
    assert model.k_text is not model.k_visual
    assert model.pair_text is not model.pair_visual
    assert model.gate_text is not model.gate_visual
    before_text, before_visual = output["Z_text"].clone(), output["Z_visual"].clone()
    with torch.no_grad():
        model.gate_text.weight[0, 0] = 0.7
        model.gate_text.bias.add_(0.4)
    after = model.analyze(x, edge_index)
    assert not torch.equal(before_text, after["Z_text"])
    torch.testing.assert_close(before_visual, after["Z_visual"], rtol=0, atol=0)


def test_mass_preservation_keeps_each_target_mass_and_self_diagonal():
    x, edge_index = _graph()
    model = RelCalStatePres(_cfg("s45_masspres_entry_uniform"), _info()).eval()
    with torch.no_grad():
        model.gate_text.bias.fill_(0.7)
        model.gate_visual.bias.fill_(-0.3)
    result = model.analyze(x, edge_index)
    p_rel = result["P_rel"]
    rows = p_rel.indices()[0]
    base = p_rel.values()
    for modality in ("text", "visual"):
        op = result["operators"][modality]
        rel_idx = op.indices()[:, op.indices()[0] != op.indices()[1]]
        rel_val = op.values()[op.indices()[0] != op.indices()[1]]
        actual = base.new_zeros(7).index_add_(0, rel_idx[0], rel_val)
        expected = base.new_zeros(7).index_add_(0, rows, base)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
        diag = op.to_dense().diagonal()
        torch.testing.assert_close(diag, result["P_self"].to_dense().diagonal(), rtol=0, atol=0)
        torch.testing.assert_close(rel_idx, p_rel.indices(), rtol=0, atol=0)


def test_entry_and_persistent_hop_placement_use_one_frozen_gate():
    x, edge_index = _graph()
    for variant in ("s45_masspres_entry_uniform", "s45_masspres_persistent_uniform"):
        model = RelCalStatePres(_cfg(variant), _info()).eval()
        call_count = {"text": 0, "visual": 0}
        hooks = [model.pair_text.register_forward_hook(lambda *args: call_count.__setitem__("text", call_count["text"] + 1)),
                 model.pair_visual.register_forward_hook(lambda *args: call_count.__setitem__("visual", call_count["visual"] + 1))]
        result = model.analyze(x, edge_index)
        for hook in hooks:
            hook.remove()
        expected_calls = (result["P_rel"]._nnz() + model.edge_chunk_size - 1) // model.edge_chunk_size
        assert call_count == {"text": expected_calls, "visual": expected_calls}
        p = result["P"]
        for modality in ("text", "visual"):
            states = result[f"S_{modality}"]
            first = result["operators"][modality]
            torch.testing.assert_close(states[1], torch.sparse.mm(first, states[0]))
            if variant == "s45_masspres_entry_uniform":
                torch.testing.assert_close(states[2], torch.sparse.mm(p, states[1]))
                torch.testing.assert_close(states[3], torch.sparse.mm(p, states[2]))
            else:
                torch.testing.assert_close(states[2], torch.sparse.mm(first, states[1]))
                torch.testing.assert_close(states[3], torch.sparse.mm(first, states[2]))


def test_calibration_off_recovers_original_operator_and_outputs():
    x, edge_index = _graph()
    model = RelCalStatePres(_cfg("s45_masspres_entry_uniform"), _info()).eval()
    normal = model.analyze(x, edge_index)
    off = model.analyze(x, edge_index, gate_override="off")
    for modality in ("text", "visual"):
        torch.testing.assert_close(off["operators"][modality].to_dense(), normal["P"].to_dense(),
                                   rtol=0, atol=0)
        manual = [off[f"H0_{modality}"]]
        for _ in range(3):
            manual.append(torch.sparse.mm(off["P"], manual[-1]))
        for a, b in zip(off[f"S_{modality}"],
                        manual, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_edge_shuffle_renormalization_preserves_support_multiset_and_mass():
    x, edge_index = _graph()
    model = RelCalStatePres(_cfg("s45_masspres_entry_uniform"), _info()).eval()
    result = model.analyze(x, edge_index)
    rows = result["P_rel"].indices()[0]
    raw = result["raw_gates"]["text"].flatten()
    shuffled = raw.clone()
    generator = torch.Generator().manual_seed(441)
    for target in torch.unique_consecutive(rows):
        index = (rows == target).nonzero().flatten()
        if index.numel() > 1:
            shuffled[index] = raw[index[torch.randperm(index.numel(), generator=generator)]]
        torch.testing.assert_close(torch.sort(shuffled[index]).values,
                                   torch.sort(raw[index]).values, rtol=0, atol=0)
    original = result["operators"]["text"]
    shuffled_analysis = model.analyze(x, edge_index, gate_override={"text": shuffled})
    reordered = shuffled_analysis["operators"]["text"]
    assert torch.equal(original.indices(), reordered.indices())
    for op in (original, reordered):
        idx = op.indices()
        rel = idx[0] != idx[1]
        total = op.values().new_zeros(7).index_add_(0, idx[0, rel], op.values()[rel])
        base_idx = result["P_rel"].indices()
        base_mass = result["P_rel"].values().new_zeros(7).index_add_(0, base_idx[0], result["P_rel"].values())
        torch.testing.assert_close(total, base_mass, rtol=0, atol=1e-6)


def test_scope_constants_exclude_toys_lp_test_and_fix_seeds():
    from scripts.run_s45_relcal_statepres_nc import DATASETS, SEEDS, VARIANTS

    assert DATASETS == ("Movies", "Grocery", "ele-fashion", "Reddit-S")
    assert SEEDS == (42, 43, 44)
    assert len(VARIANTS) == 8


def test_sparse_value_matmul_matches_dense_forward_and_backward():
    indices = torch.tensor([[0, 0, 1, 2], [0, 2, 1, 0]])
    initial = torch.tensor([0.3, -0.2, 0.7, 0.5])
    dense_initial = torch.randn(3, 4)
    custom_values = initial.clone().requires_grad_()
    custom_dense = dense_initial.clone().requires_grad_()
    custom = _SparseValueMatmul.apply(indices, custom_values, (3, 3), custom_dense)
    (custom.square().sum()).backward()

    reference_values = initial.clone().requires_grad_()
    reference_dense = dense_initial.clone().requires_grad_()
    reference_sparse = torch.sparse_coo_tensor(indices, reference_values, (3, 3)).coalesce()
    reference = reference_sparse.to_dense() @ reference_dense
    reference.square().sum().backward()

    torch.testing.assert_close(custom, reference, rtol=0, atol=0)
    torch.testing.assert_close(custom_values.grad, reference_values.grad, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(custom_dense.grad, reference_dense.grad, rtol=1e-6, atol=1e-7)
