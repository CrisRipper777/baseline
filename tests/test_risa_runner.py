from pathlib import Path

from scripts.run_risa_p0 import (
    CONFIRM_DATASETS,
    SCREEN_DATASETS,
    VARIANTS,
    checkpoint_base_path,
    expected_checkpoint_paths,
)


def test_formal_checkpoint_base_generates_unified_nc_run_names(tmp_path: Path):
    base = checkpoint_base_path(tmp_path)
    assert base.name == "best.pt"
    assert [p.name for p in expected_checkpoint_paths(tmp_path, 3)] == [
        "best_run1.pt", "best_run2.pt", "best_run3.pt"
    ]
    assert expected_checkpoint_paths(tmp_path, 1) == [base]


def test_default_formal_screening_excludes_confirmation_and_dynamic_control():
    assert SCREEN_DATASETS == ("Movies", "ele-fashion", "Reddit-S")
    assert CONFIRM_DATASETS == ("Grocery",)
    assert VARIANTS == (
        "p0_identity", "p0_masspres_scalar", "p0_operator_uniform",
        "p0_operator_global", "p0_operator_routed",
    )


def test_matched_shuffle_is_deterministic_and_stays_inside_structural_bins():
    import torch

    from scripts.analyze_risa_p0_formal import matched_structural_shuffle
    from src.models.relcal_statepres_pilot import Model as S45Model
    from types import SimpleNamespace
    from omegaconf import OmegaConf

    edge = torch.tensor([[0, 0, 1, 2, 2, 3, 4, 5, 6, 1],
                         [1, 2, 2, 0, 3, 4, 5, 4, 6, 5]])
    cfg = SimpleNamespace(model=OmegaConf.create({
        "hidden_dim": 256, "dropout": 0.2, "max_order": 3,
        "relation_dim": 32, "edge_chunk_size": 3, "variant": "s45_identity_uniform",
    }))
    backbone = S45Model(cfg, {"input_dim": 12, "text_dim": 5, "visual_dim": 7,
                              "num_nodes": 7, "num_classes": 3})
    p_rel = backbone._get_operators(edge, 7, torch.float32)[2]
    first, report = matched_structural_shuffle(p_rel, 7)
    second, report2 = matched_structural_shuffle(p_rel, 7)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert report == report2
    assert sorted(first.tolist()) == list(range(p_rel._nnz()))
    assert report["preserves_edge_marginal_within_each_bin"] is True
