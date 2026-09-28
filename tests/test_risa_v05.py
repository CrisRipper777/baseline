from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_risa_v05 import build_command
from src.models.risa_v05 import GroupedRotationTransform, Model


def _cfg(variant: str = "v05_full", edge_chunk_size: int = 3, node_chunk_size: int = 3):
    return SimpleNamespace(model=OmegaConf.create({
        "name": "risa_v05", "hidden_dim": 256, "dropout": 0.2,
        "max_order": 3, "relation_dim": 32,
        "edge_chunk_size": edge_chunk_size, "rotation_group_size": 2,
        "max_rotation_angle": 1.57079632679, "iamr_num_heads": 4,
        "iamr_ff_mult": 2, "node_chunk_size": node_chunk_size,
        "variant": variant,
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


def test_zero_theta_returns_exact_base_message():
    transform = GroupedRotationTransform()
    base = torch.randn(11, 256)
    rotated = transform.rotate(base, torch.zeros(11, 128))
    torch.testing.assert_close(rotated, base, rtol=0, atol=0)


def test_grouped_rotation_preserves_norm():
    transform = GroupedRotationTransform()
    base = torch.randn(29, 256)
    angles = torch.randn(29, 128) * 1.2
    rotated = transform.rotate(base, angles)
    torch.testing.assert_close(rotated.norm(dim=-1), base.norm(dim=-1), rtol=1e-6, atol=2e-6)


def test_no_crst_first_order_state_is_plain_p_times_h0():
    x, edge = _graph()
    model = Model(_cfg("v05_no_crst"), _info()).eval()
    result = model.analyze(x, edge, collect_edge_state=False, collect_attention=False)
    for modality in ("text", "visual"):
        expected = torch.sparse.mm(result["P"], result[f"H0_{modality}"])
        torch.testing.assert_close(result[f"C1_{modality}"], expected, rtol=0, atol=0)


def test_crst_never_mutates_physical_p_rel():
    x, edge = _graph()
    model = Model(_cfg(), _info()).eval()
    p_rel = model.backbone._get_operators(edge, x.size(0), x.dtype)[2]
    indices, values = p_rel.indices().clone(), p_rel.values().clone()
    result = model.analyze(x, edge)
    torch.testing.assert_close(result["P_rel"].indices(), indices, rtol=0, atol=0)
    torch.testing.assert_close(result["P_rel"].values(), values, rtol=0, atol=0)


def test_leave_one_edge_out_context_matches_toy_graph():
    h = torch.tensor([[0.0], [2.0], [6.0]])
    row = torch.tensor([0, 0, 1])
    col = torch.tensor([1, 2, 0])
    weight = torch.tensor([0.25, 0.75, 1.0])
    deviation, valid = Model.leave_one_edge_out_context(h, row, col, weight)
    expected = torch.tensor([[-4.0], [4.0], [0.0]])
    torch.testing.assert_close(deviation, expected, rtol=0, atol=0)
    assert valid.tolist() == [True, True, False]


def test_degree_one_context_is_zero_and_finite():
    h = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    row, col, weight = torch.tensor([0, 1]), torch.tensor([1, 0]), torch.tensor([1.0, 1.0])
    deviation, valid = Model.leave_one_edge_out_context(h, row, col, weight)
    assert not valid.any()
    assert torch.equal(deviation, torch.zeros_like(deviation))
    assert torch.isfinite(deviation).all()


def test_chunked_crst_matches_large_chunk():
    x, edge = _graph()
    torch.manual_seed(93)
    chunked = Model(_cfg(edge_chunk_size=2), _info()).eval()
    large = Model(_cfg(edge_chunk_size=1000), _info()).eval()
    large.load_state_dict(chunked.state_dict(), strict=True)
    left = chunked.analyze(x, edge, collect_edge_state=True)
    right = large.analyze(x, edge, collect_edge_state=True)
    for key in ("C1_text", "C2_text", "C3_text", "C1_visual", "C2_visual",
                "C3_visual", "Z_text", "Z_visual", "fused_z"):
        torch.testing.assert_close(left[key], right[key], rtol=2e-6, atol=2e-6)
    for modality in ("text", "visual"):
        torch.testing.assert_close(left["rotation_angles"][modality],
                                   right["rotation_angles"][modality], rtol=2e-6, atol=2e-6)


def test_imci_attention_probabilities_sum_to_one_over_three_hops():
    x, edge = _graph()
    result = Model(_cfg(), _info()).eval().analyze(x, edge)
    for modality in ("text", "visual"):
        weights = result["attention_weights"][modality]
        assert weights.shape == (x.size(0), 4, 1, 3)
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)),
                                   rtol=1e-6, atol=1e-6)


def test_no_imci_uses_uniform_four_state_readout():
    x, edge = _graph()
    result = Model(_cfg("v05_no_imci"), _info()).eval().analyze(
        x, edge, collect_attention=False,
    )
    for modality in ("text", "visual"):
        expected = sum(result[f"C_{modality}"]) / 4.0
        torch.testing.assert_close(result[f"Z_{modality}"], expected, rtol=0, atol=0)


def test_full_model_required_parameter_groups_receive_nonzero_gradients():
    x, edge = _graph()
    torch.manual_seed(771)
    model = Model(_cfg(), _info())
    classifier = torch.nn.Linear(256, 3)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0])
    logits = classifier(model(x, edge)[0])
    torch.nn.functional.cross_entropy(logits, labels).backward()
    required_prefixes = {
        "modality relation": ("relation_encoder_",),
        "shared relation": ("shared_relation_encoder.", "shared_to_", "shared_norm_"),
        "angle heads": ("rotation_",),
        "MHA": ("imci_text.attention.", "imci_visual.attention."),
        "IMCI FFN": ("imci_text.ffn.", "imci_visual.ffn."),
    }
    for group, prefixes in required_prefixes.items():
        grads = [p.grad for name, p in model.named_parameters()
                 if name.startswith(prefixes) and p.grad is not None]
        assert grads, f"no gradients in {group}"
        assert sum(float(grad.abs().sum()) for grad in grads) > 0.0, f"zero gradients in {group}"


def test_nc_forward_and_inference_api_compatibility():
    x, edge = _graph()
    model = Model(_cfg(), _info()).eval()
    output = model(x, edge)
    assert len(output) == 5
    assert output[0].shape == (x.size(0), 256)
    embeddings = model.inference(x, edge, device="cpu")
    assert embeddings.shape == (x.size(0), 256)
    assert torch.isfinite(embeddings).all()


def test_smoke_command_disables_test_and_uses_only_requested_nc_path(tmp_path):
    command = build_command("Movies", "v05_full", "smoke", "0", tmp_path,
                            tmp_path / "hydra")
    assert "task=nc" in command
    assert "task.evaluate_test=false" in command
    assert "task.epochs=1" in command
    assert not any("test=true" in item.lower() for item in command)


def test_variant_surface_and_screening_dataset_staging():
    from scripts.run_risa_v05 import CONFIRM_DATASETS, SCREEN_DATASETS, SEEDS

    assert Model.VARIANTS == ("v05_full", "v05_no_crst", "v05_no_relation_context", "v05_no_imci")
    assert SCREEN_DATASETS == ("Movies", "ele-fashion", "Reddit-S")
    assert CONFIRM_DATASETS == ("Grocery",)
    assert SEEDS == (42, 43, 44)
