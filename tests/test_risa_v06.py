from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_risa_v06 import build_command, formal_dry_run
from src.models.risa_v05 import Model as V05Model
from src.models.risa_v06 import Model, RelationalEvidenceExtractor


def _cfg(variant: str = "v06_full", edge_chunk_size: int = 16384,
         node_chunk_size: int = 32768):
    return SimpleNamespace(model=OmegaConf.create({
        "name": "risa_v06", "hidden_dim": 256, "dropout": 0.2,
        "max_order": 3, "relation_dim": 32, "evidence_dim": 32,
        "edge_chunk_size": edge_chunk_size, "node_chunk_size": node_chunk_size,
        "num_heads": 4, "ff_mult": 2, "variant": variant,
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


def test_structural_path_is_exactly_three_plain_physical_p_hops():
    x, edge = _graph()
    result = Model(_cfg(), _info()).eval().analyze(x, edge)
    for modality in ("text", "visual"):
        s1 = torch.sparse.mm(result["P"], result[f"H0_{modality}"])
        s2 = torch.sparse.mm(result["P"], s1)
        s3 = torch.sparse.mm(result["P"], s2)
        torch.testing.assert_close(result[f"S1_{modality}"], s1, rtol=0, atol=0)
        torch.testing.assert_close(result[f"S2_{modality}"], s2, rtol=0, atol=0)
        torch.testing.assert_close(result[f"S3_{modality}"], s3, rtol=0, atol=0)


def test_relation_module_does_not_mutate_p_p_self_or_p_rel():
    x, edge = _graph()
    model = Model(_cfg(), _info()).eval()
    operators = model.backbone._get_operators(edge, x.size(0), x.dtype)
    before = [(op.indices().clone(), op.values().clone()) for op in operators]
    result = model.analyze(x, edge)
    for op, (indices, values) in zip(
            (result["P"], result["P_self"], result["P_rel"]), before, strict=True):
        assert torch.equal(op.indices(), indices)
        assert torch.equal(op.values(), values)


def test_leave_one_edge_out_matches_weighted_toy_graph():
    h = torch.tensor([[0.0], [2.0], [6.0]])
    row = torch.tensor([0, 0, 1])
    col = torch.tensor([1, 2, 0])
    weight = torch.tensor([0.25, 0.75, 1.0])
    deviation, valid = Model.leave_one_edge_out_context(h, row, col, weight)
    torch.testing.assert_close(deviation, torch.tensor([[-4.0], [4.0], [0.0]]), rtol=0, atol=0)
    assert valid.tolist() == [True, True, False]


def test_degree_one_context_is_zero_and_finite():
    h = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    row, col, weight = torch.tensor([0, 1]), torch.tensor([1, 0]), torch.tensor([1.0, 1.0])
    deviation, valid = Model.leave_one_edge_out_context(h, row, col, weight)
    assert not valid.any()
    assert torch.equal(deviation, torch.zeros_like(deviation))
    assert torch.isfinite(deviation).all()


def test_evidence_bottleneck_and_edge_shapes_are_32_then_256():
    torch.manual_seed(9)
    extractor = RelationalEvidenceExtractor()
    result = extractor(torch.randn(4, 32), torch.randn(4, 256), torch.randn(4, 256))
    for key in ("u_same", "u_cross", "g_same", "g_cross"):
        assert result[key].shape == (4, 32)
    for key in ("delta_same", "delta_cross"):
        assert result[key].shape == (4, 256)


def test_same_and_cross_evidence_use_the_correct_neighbor_modalities():
    x, edge = _graph()
    model = Model(_cfg(edge_chunk_size=10000), _info()).eval()
    captured: dict[str, list[torch.Tensor]] = {"text": [], "visual": []}
    handles = []
    for modality in ("text", "visual"):
        extractor = getattr(model, f"evidence_extractor_{modality}")
        handles.append(extractor.register_forward_pre_hook(
            lambda _module, args, name=modality: captured[name].append(
                (args[1].detach().clone(), args[2].detach().clone())
            )
        ))
    result = model.analyze(x, edge, collect_attention=False)
    for handle in handles:
        handle.remove()
    row, col = result["P_rel"].indices()
    assert len(captured["text"]) == len(captured["visual"]) == 1
    same_text, cross_text = captured["text"][0]
    same_visual, cross_visual = captured["visual"][0]
    torch.testing.assert_close(same_text, result["H0_text"][col], rtol=0, atol=0)
    torch.testing.assert_close(cross_text, result["H0_visual"][col], rtol=0, atol=0)
    torch.testing.assert_close(same_visual, result["H0_visual"][col], rtol=0, atol=0)
    torch.testing.assert_close(cross_visual, result["H0_text"][col], rtol=0, atol=0)


def test_edge_evidence_aggregation_matches_manual_weighted_sum():
    row = torch.tensor([0, 0, 2])
    weight = torch.tensor([0.25, 0.75, 0.5])
    delta = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    actual = Model.aggregate_edge_evidence(3, row, weight, delta)
    expected = torch.tensor([[2.5, 3.5], [0.0, 0.0], [2.5, 3.0]])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_relation_context_pooling_matches_manual_values_and_zero_degree():
    row = torch.tensor([0, 0, 2])
    weight = torch.tensor([0.25, 0.75, 0.5])
    relation = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 4.0]])
    row_mass = torch.tensor([1.0, 0.0, 0.5])
    actual = Model.pool_relation_context(3, row, weight, relation, row_mass)
    expected = torch.tensor([[0.25, 0.75], [0.0, 0.0], [2.0, 4.0]])
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-11)


def test_evidence_is_not_sent_through_additional_graph_propagation(monkeypatch):
    x, edge = _graph()
    model = Model(_cfg(), _info()).eval()
    calls = []
    sparse_mm = torch.sparse.mm

    def record(operator, dense):
        calls.append((operator, dense))
        return sparse_mm(operator, dense)

    monkeypatch.setattr(torch.sparse, "mm", record)
    result = model.analyze(x, edge, collect_attention=False)
    assert len(calls) == 6
    assert all(torch.equal(operator.indices(), result["P"].indices()) for operator, _ in calls)
    assert all(dense.shape[-1] == 256 for _, dense in calls)
    assert not any(key.startswith("edge_") for key in result)


def test_no_relation_variant_has_no_relation_modules_and_query_is_h0():
    x, edge = _graph()
    model = Model(_cfg("v06_no_relation_condition"), _info()).eval()
    assert not hasattr(model, "relation_encoder_text")
    assert not hasattr(model, "evidence_extractor_text")
    result = model.analyze(x, edge)
    for modality in ("text", "visual"):
        assert torch.equal(result[f"Q_{modality}"], result[f"H0_{modality}"])


def test_no_relation_variant_is_bitwise_equal_to_v05_absorb_only():
    x, edge = _graph()
    torch.manual_seed(917)
    v06 = Model(_cfg("v06_no_relation_condition"), _info()).eval()
    v05_cfg = SimpleNamespace(model=OmegaConf.create({
        "name": "risa_v05", "hidden_dim": 256, "dropout": 0.2,
        "max_order": 3, "relation_dim": 32, "edge_chunk_size": 16384,
        "rotation_group_size": 2, "max_rotation_angle": 1.57079632679,
        "iamr_num_heads": 4, "iamr_ff_mult": 2, "node_chunk_size": 32768,
        "variant": "v05_absorb_only",
    }))
    v05 = V05Model(v05_cfg, _info()).eval()
    v05.load_state_dict(v06.state_dict(), strict=True)
    output06 = v06(x, edge)[0]
    output05 = v05(x, edge)[0]
    assert torch.equal(output06, output05)


def test_full_initialization_has_small_finite_query_shift_and_nonzero_evidence():
    x, edge = _graph()
    torch.manual_seed(2401)
    model = Model(_cfg(), _info()).eval()
    result = model.analyze(x, edge, collect_attention=False)
    for modality in ("text", "visual"):
        assert torch.isfinite(result[f"query_shift_{modality}"]).all()
        assert torch.isfinite(result[f"E_{modality}"]).all()
        assert float(result[f"query_shift_{modality}"].norm(dim=-1).mean()) < 0.1
        assert float(result[f"E_{modality}"].norm()) > 0.0


def test_full_model_required_relation_and_absorption_gradients_are_nonzero():
    x, edge = _graph()
    torch.manual_seed(771)
    model = Model(_cfg(), _info())
    classifier = torch.nn.Linear(256, 3)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0])
    logits = classifier(model(x, edge)[0])
    torch.nn.functional.cross_entropy(logits, labels).backward()
    groups = {
        "relation encoder": ("relation_encoder_",),
        "shared relation": ("shared_relation_encoder.", "shared_to_", "shared_norm_"),
        "same evidence": ("evidence_extractor_",),
        "cross evidence": ("evidence_extractor_",),
        "W_rho": ("W_rho_",),
        "MHA": ("imci_text.attention.", "imci_visual.attention."),
        "FFN": ("imci_text.ffn.", "imci_visual.ffn."),
    }
    for group, prefixes in groups.items():
        params = [(name, p) for name, p in model.named_parameters() if name.startswith(prefixes)]
        if group in {"same evidence", "cross evidence"}:
            branch = "same" if group == "same evidence" else "cross"
            params = [(name, p) for name, p in params if f"_{branch}." in name]
        grads = [p.grad for _, p in params if p.grad is not None]
        assert grads and sum(float(g.abs().sum()) for g in grads) > 0, group


def test_v06_smoke_and_formal_dry_run_respect_requested_protocol():
    output = Path("/tmp/risa-v06-test-output")
    command = build_command("Movies", "v06_full", "0", output, output / "hydra")
    assert "task=nc" in command
    assert "task.evaluate_test=false" in command
    assert "task.training_mode=full_graph" in command
    assert "task.epochs=1" in command
    assert not any("test=true" in item.lower() for item in command)
    dry = formal_dry_run()
    assert dry["runs_total"] == 9
    assert [(job["dataset"], job["variant"]) for job in dry["jobs"]] == [
        (dataset, "v06_full") for dataset in ("Movies", "Grocery", "ele-fashion")
    ]
    assert all(job["seeds"] == [42, 43, 44] for job in dry["jobs"])
    assert dry["launched"] is False
