from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.models.multi_order_bank import Model as MultiOrderBankModel
from src.models.relation_basis_pilot import Model


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    "p0_relation_only", "p0_residual", "p0_concat", "h1_dual_agg",
    "h1_dual_functional_static", "h1_dual_functional_global",
)


def build(variant: str) -> Model:
    cfg = OmegaConf.create({"model": {"hidden_dim": 256, "dropout": 0.2, "variant": variant}})
    return Model(cfg, {"input_dim": 7, "text_dim": 3, "visual_dim": 4})


def nparams(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def test_relation_operator_preserves_exact_off_diagonal_weights():
    edges = torch.tensor([[0, 1, 1, 2, 2, 0, 2], [1, 0, 2, 1, 0, 0, 2]])
    p, p_rel, row_sum = Model._build_operators(edges, 3, torch.float64)
    dense_p, dense_rel = p.to_dense(), p_rel.to_dense()
    offdiag = ~torch.eye(3, dtype=torch.bool)
    assert torch.equal(dense_rel.diag(), torch.zeros(3, dtype=torch.float64))
    assert torch.equal(dense_rel[offdiag], dense_p[offdiag])
    assert torch.equal(dense_rel.sum(dim=1, keepdim=True), row_sum)


def test_p0_relation_only_and_residual_match_parameter_count_and_equations():
    relation, residual = build("p0_relation_only"), build("p0_residual")
    residual.load_state_dict(relation.state_dict())
    assert nparams(relation) == nparams(residual)
    relation.eval(); residual.eval()
    x = torch.randn(5, 7)
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    only = relation.analyze(x, edges)
    keep = residual.analyze(x, edges)
    expected_only = relation.output_norm_text(only["R_A_text"])
    expected_keep = residual.output_norm_text(keep["H_text"] + keep["R_A_text"])
    torch.testing.assert_close(only["Z_text"], expected_only)
    torch.testing.assert_close(keep["Z_text"], expected_keep)
    torch.testing.assert_close(only["R_A_text"], keep["R_A_text"])


def test_h1_static_dual_matches_aggregative_capacity_and_differential_identity():
    agg, static = build("h1_dual_agg"), build("h1_dual_functional_static")
    assert nparams(agg) == nparams(static)
    static.eval()
    x = torch.randn(4, 7)
    edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    result = static.analyze(x, edges)
    torch.testing.assert_close(result["U_D_text"], result["s"] * result["H_text"] - result["U_A_text"])
    p_rel = result["P_rel"].to_dense()
    h = result["H_text"]
    edge_sum = torch.zeros_like(h)
    rel_indices = result["P_rel"].coalesce().indices()
    rel_values = result["P_rel"].coalesce().values()
    for edge_id in range(rel_values.numel()):
        i, j = rel_indices[:, edge_id].tolist()
        edge_sum[i] += rel_values[edge_id] * (h[i] - h[j])
    torch.testing.assert_close(result["U_D_text"], edge_sum)


def test_global_alpha_starts_uniform_and_modality_parameters_are_independent():
    model = build("h1_dual_functional_global")
    result = model.analyze(torch.randn(4, 7), torch.tensor([[0, 1], [1, 0]]))
    torch.testing.assert_close(result["alpha_text"], torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(result["alpha_visual"], torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(result["alpha_visual"].sum(), torch.tensor(1.0))
    assert model.theta_text is not model.theta_visual
    assert model.theta_text.data_ptr() != model.theta_visual.data_ptr()


def test_plain_fusion_matches_multi_order_bank_semantics():
    model = build("p0_relation_only")
    bank_cfg = OmegaConf.create({"model": {"hidden_dim": 256, "dropout": 0.2,
        "max_order": 3, "readout": "terminal", "modality_mode": "both", "fusion_mode": "plain_mlp"}})
    bank = MultiOrderBankModel(bank_cfg, {"input_dim": 7, "text_dim": 3, "visual_dim": 4})
    bank.plain_fusion.load_state_dict(model.plain_fusion.state_dict())
    model.eval(); bank.eval()
    x, edge = torch.randn(4, 7), torch.tensor([[0, 1], [1, 0]])
    analysis = model.analyze(x, edge)
    fused_input = torch.cat([analysis["Z_text"], analysis["Z_visual"]], dim=-1)
    torch.testing.assert_close(analysis["fused_z"], bank.plain_fusion(fused_input))


def test_formal_launcher_fixes_seed_test_and_dataset_scope():
    launcher = (ROOT / "scripts/run_s43_p0_h1_nc.py").read_text(encoding="utf-8")
    assert "SEEDS = (42, 43, 44)" in launcher
    assert "evaluate_test=false" in launcher
    assert "DATASETS = (\"Movies\", \"Grocery\", \"ele-fashion\", \"Reddit-S\")" in launcher
    assert "Toys" not in launcher
    assert "num_runs, epochs = (1, 1) if is_preflight else (3, 300)" in launcher
    assert "f" + '"num_runs={num_runs}"' in launcher


def test_variant_set_is_exact():
    assert set(VARIANTS) == Model.VARIANTS
