from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.models.adaptive_relation_pilot import (
    ConditionEncoderV1,
    Model,
    SignedRelationTransform,
    apply_correction,
    apply_group_gate,
)
from src.models.relation_basis_pilot import Model as HistoricalModel

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    "h1r_dual_agg_signed", "h1r_dual_functional_signed",
    "h2a_global_scalar", "h2a_node_scalar", "h2a_global_group", "h2a_node_group",
    "h2b_global_agg_correction", "h2b_global_diff_correction",
    "h2b_node_agg_correction", "h2b_node_diff_correction",
)


def _model(variant: str):
    cfg = OmegaConf.create({"model": {"hidden_dim": 256, "dropout": 0.2, "variant": variant}})
    return Model(cfg, {"text_dim": 12, "visual_dim": 9, "input_dim": 21, "num_nodes": 5, "num_classes": 3})


def _historical():
    cfg = OmegaConf.create({"model": {"hidden_dim": 256, "dropout": 0.2, "variant": "h1_dual_functional_static"}})
    return HistoricalModel(cfg, {"text_dim": 12, "visual_dim": 9, "input_dim": 21})


def _count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_signed_transform_retains_negative_responses():
    transform = SignedRelationTransform().eval()
    with torch.no_grad():
        transform.linear.weight.copy_(torch.eye(256))
        transform.linear.bias.zero_()
        transform.norm.weight.fill_(1.0)
        transform.norm.bias.zero_()
    x = torch.zeros(2, 256)
    x[0, 0] = -2
    x[1, 0] = 2
    out = transform(x)
    assert out[0, 0] < 0
    assert out[1, 0] > 0
    preactivation = transform.norm(transform.linear(x))
    torch.testing.assert_close(out[0, 0], 0.1 * preactivation[0, 0], atol=1e-6, rtol=1e-6)


def test_h1r_signed_variants_have_matched_trainable_parameters():
    assert _count(_model("h1r_dual_agg_signed")) == _count(_model("h1r_dual_functional_signed"))


def test_projector_plain_mlp_and_operator_semantics_match_historical_model():
    torch.manual_seed(22)
    old = _historical().eval()
    new = _model("h2a_global_scalar").eval()
    new.text_projector.load_state_dict(old.text_projector.state_dict())
    new.visual_projector.load_state_dict(old.visual_projector.state_dict())
    new.plain_fusion.load_state_dict(old.plain_fusion.state_dict())
    x = torch.randn(6, 21)
    torch.testing.assert_close(new.text_projector(x[:, :12]), old.text_projector(x[:, :12]))
    torch.testing.assert_close(new.visual_projector(x[:, 12:]), old.visual_projector(x[:, 12:]))
    edge = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    _, old_rel, old_sum = old._build_operators(edge, 6, x.dtype)
    new_rel, new_sum = new._get_operators(edge, 6, x.dtype)
    torch.testing.assert_close(new_rel.to_dense(), old_rel.to_dense(), atol=0, rtol=0)
    torch.testing.assert_close(new_sum, old_sum, atol=0, rtol=0)
    assert new.plain_fusion[0].in_features == 512 and new.plain_fusion[-1].out_features == 256


def test_h15_fixed_lambda_scale_contract():
    ra = torch.tensor([[2.0, -4.0]])
    rd = torch.tensor([[8.0, 6.0]])
    normal = 0.5 * ra + 0.5 * rd
    at_one = 0.5 * ra + 0.5 * 1.0 * rd
    at_zero = 0.5 * ra + 0.5 * 0.0 * rd
    torch.testing.assert_close(at_one, normal, atol=0, rtol=0)
    torch.testing.assert_close(at_zero, 0.5 * ra, atol=0, rtol=0)


def test_h2a_beta_init_range_and_independent_modalities():
    for variant in ("h2a_global_scalar", "h2a_node_scalar", "h2a_global_group", "h2a_node_group"):
        model = _model(variant)
        for modality in ("text", "visual"):
            assert torch.equal(model.base_bias[modality], torch.zeros_like(model.base_bias[modality]))
        assert model.base_bias["text"] is not model.base_bias["visual"]
        if variant.startswith("h2a_node"):
            assert model.controllers["text"] is not model.controllers["visual"]
            assert torch.count_nonzero(model.controllers["text"].mlp[-1].weight) == 0
            assert torch.count_nonzero(model.controllers["text"].mlp[-1].bias) == 0
            assert model.controllers["text"].mlp[0].in_features == 64
        raw = torch.linspace(-8, 8, 9)
        beta = 2 * torch.sigmoid(raw)
        assert torch.all(beta > 0) and torch.all(beta < 2)
        init_beta = 2 * torch.sigmoid(model.base_bias["text"])
        torch.testing.assert_close(init_beta, torch.ones_like(init_beta), atol=0, rtol=0)


def test_h2a_group_reshape_multiply_is_exact():
    relation = torch.arange(2 * 256, dtype=torch.float32).reshape(2, 256)
    beta = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                         [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
    expected = (relation.reshape(2, 8, 32) * beta.unsqueeze(-1)).reshape(2, 256)
    torch.testing.assert_close(apply_group_gate(relation, beta), expected, atol=0, rtol=0)


def test_h2b_generic_and_differential_variants_match_parameters():
    for a, b in (("h2b_global_agg_correction", "h2b_global_diff_correction"),
                 ("h2b_node_agg_correction", "h2b_node_diff_correction")):
        assert _count(_model(a)) == _count(_model(b))
    for variant in ("h2b_global_agg_correction", "h2b_node_agg_correction"):
        model = _model(variant)
        for modality in ("text", "visual"):
            lam = 2 * torch.tanh(model.base_bias[modality])
            torch.testing.assert_close(lam, torch.tensor([0.1]), atol=1e-7, rtol=1e-6)
        assert model.base_bias["text"] is not model.base_bias["visual"]
        if variant.startswith("h2b_node"):
            ctl = model.controllers["text"]
            assert ctl.q_h.in_features == 256 and ctl.q_relation.in_features == 256
            assert ctl.q_correction.in_features == 256
            assert ctl.mlp[0].in_features == 96
            assert torch.count_nonzero(ctl.mlp[-1].weight) == 0
            assert torch.count_nonzero(ctl.mlp[-1].bias) == 0


def test_h2b_frozen_correction_interventions_exact():
    ra = torch.randn(5, 256)
    rc = torch.randn(5, 256)
    lam = torch.full((5, 1), 0.2)
    torch.testing.assert_close(apply_correction(ra, rc, lam, "zero_correction"), ra, atol=0, rtol=0)
    torch.testing.assert_close(apply_correction(ra, rc, lam, "sign_flip"), ra - lam * rc, atol=0, rtol=0)


def test_all_variants_and_formal_protocol_scope_are_closed():
    assert set(VARIANTS) == Model.VARIANTS
    launcher = (ROOT / "scripts/run_s43_h15_h1r_h2ab_nc.py").read_text()
    assert '"task.evaluate_test=false"' in launcher
    assert '"Movies", "Grocery", "ele-fashion", "Reddit-S"' in launcher
    assert "(42, 43, 44)" in launcher
    assert "Toys" not in launcher


def test_historical_model_source_and_checkpoints_are_unchanged():
    source_commit = "601bad98ff28e85c9aeac23b84b5db51ad4c7c86"
    frozen = subprocess.check_output(["git", "show", f"{source_commit}:src/models/relation_basis_pilot.py"], cwd=ROOT)
    assert frozen == (ROOT / "src/models/relation_basis_pilot.py").read_bytes()
    for dataset in ("Movies", "Grocery", "ele-fashion", "Reddit-S"):
        complete = ROOT / "outputs/s43_p0_h1_v1/formal" / dataset / "h1_dual_functional_static" / "complete.json"
        assert complete.is_file()
        for run in (1, 2, 3):
            checkpoint = complete.parent / f"best_run{run}.pt"
            assert checkpoint.is_file()
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert payload["task"] == "nc" and payload["protocol_version"] == "unified_full_graph_nc_v1"
            assert payload["seed"] == 41 + run
            assert not any(k.startswith("test_") for k in payload["metrics"])


def test_analysis_csv_sparse_cells_are_explicit_and_diff_clean(tmp_path):
    import csv
    from scripts.analyze_s43_h15_h1r_h2ab import _write_csv

    target = tmp_path / "sparse.csv"
    _write_csv(target, [{"metric": 1.0}, {"metric": 2.0, "summary": "all"}])
    lines = target.read_text().splitlines()
    assert all(not line.endswith(",") for line in lines)
    records = list(csv.DictReader(target.open()))
    assert records[0]["summary"] == "NA"
