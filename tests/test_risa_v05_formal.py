from __future__ import annotations

from pathlib import Path

import torch

from scripts import run_risa_v05 as runner
from scripts import analyze_risa_v05_formal as analyzer


def test_nc_dataset_order_is_exact():
    assert runner.NC_DATASETS == ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")


def test_compatibility_defaults_are_four_non_movies_seed42_one_epoch():
    jobs = runner.jobs_for_phase("compatibility")
    assert jobs == [
        ("Toys", "v05_full"),
        ("Grocery", "v05_full"),
        ("ele-fashion", "v05_full"),
        ("Reddit-S", "v05_full"),
    ]
    assert runner.phase_seeds("compatibility") == (42,)
    command = runner.build_command(
        "Toys", "v05_full", "compatibility", "0", Path("/tmp/out"), Path("/tmp/hydra"),
    )
    assert "task.epochs=1" in command
    assert "task.early_stop_min_epoch=1" in command
    assert "task.evaluate_test=false" in command
    assert "task.training_mode=full_graph" in command
    assert "task.protocol_version=unified_full_graph_nc_v1" in command


def test_formal_defaults_are_five_jobs_one_variant_and_fifteen_runs():
    jobs = runner.jobs_for_phase("formal")
    assert jobs == [(dataset, "v05_full") for dataset in runner.NC_DATASETS]
    assert runner.phase_seeds("formal") == (42, 43, 44)
    assert len(jobs) == 5
    assert len(jobs) * len(runner.phase_seeds("formal")) == 15
    command = runner.build_command(
        "Movies", "v05_full", "formal", "0", Path("/tmp/Movies"), Path("/tmp/hydra"),
    )
    assert "num_runs=3" in command
    assert "task.epochs=300" in command
    assert "task.patience=30" in command
    assert "task.early_stop_min_epoch=30" in command
    assert "task.lr=1e-3" in command
    assert "task.weight_decay=1e-4" in command
    assert "task.grad_clip=1.0" in command
    assert "task.evaluate_test=false" in command


def test_formal_cli_allows_explicit_ablation_but_default_is_full_only():
    assert runner.jobs_for_phase("formal") == [(dataset, "v05_full") for dataset in runner.NC_DATASETS]
    assert runner.jobs_for_phase(
        "formal", ("Movies",), ("v05_no_crst", "v05_no_imci"),
    ) == [("Movies", "v05_no_crst"), ("Movies", "v05_no_imci")]


def test_formal_checkpoint_base_and_run_names():
    out = Path("/tmp/risa_formal")
    assert runner.expected_checkpoint_paths(out, 3) == [
        out / "best_run1.pt", out / "best_run2.pt", out / "best_run3.pt",
    ]
    cmd = runner.build_command(
        "Movies", "v05_full", "formal", "0", out, out / "hydra",
    )
    assert f"task.save_ckpt_path={out / 'best.pt'}" in cmd


def test_formal_checkpoint_seed_protocol_and_test_metric_audit(tmp_path):
    valid = {
        "task": "nc", "protocol_version": runner.PROTOCOL, "seed": 43,
        "selection": "best_val_accuracy", "metrics": {"val_acc": 0.5, "val_macro_f1": 0.4},
        "model_state": {}, "head_state": {},
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(valid, path)
    assert runner._validate_checkpoint(path, 43)["seed"] == 43
    valid["metrics"]["test_acc"] = 0.9
    torch.save(valid, path)
    try:
        runner._validate_checkpoint(path, 43)
    except RuntimeError as error:
        assert "test metric" in str(error)
    else:
        raise AssertionError("checkpoint audit accepted test metrics")


def test_validation_view_never_reads_test_split():
    class Data:
        val_idx = torch.tensor([1, 3])
        y = torch.tensor([4, 7, 8, 2, 9])

        @property
        def test_idx(self):
            raise AssertionError("formal analyzer read test_idx")

    indices, labels = analyzer.validation_view(Data(), torch.device("cpu"))
    assert indices.tolist() == [1, 3]
    assert labels.tolist() == [7, 2]


def test_formal_task_metrics_emit_only_validation_fields():
    metrics = analyzer.validation_task_metrics(
        {"val_acc": 0.75, "val_macro_f1": 0.7},
        torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
        torch.tensor([0, 1]),
    )
    assert set(metrics) == {"val_acc", "val_macro_f1", "val_cross_entropy"}
    assert metrics["val_acc"] == 0.75
    assert metrics["val_macro_f1"] == 0.7
    assert metrics["val_cross_entropy"] < 0.2


def test_crst_statistics_include_abs_angle_spread_and_saturation():
    theta = torch.tensor([[0.0, 0.5], [1.5, -1.2]])
    diagnostic = {
        "rotation_angles": {"text": theta},
        "mean_abs_angle": {"text": 0.775},
        "cos_base_rotated": {"text": 0.9},
        "norm_preservation_max_abs_error": {"text": 1e-6},
        "valid_context_edge_ratio": {"text": 0.8},
    }
    streamed = {
        "mean_abs_theta": 0.775,
        "std_abs_theta": 0.3,
        "angle_saturation_ratio": 0.25,
    }
    result = analyzer._crst_stats(diagnostic, "text", streamed)
    assert result["std_abs_theta"] > 0
    assert result["p95_abs_theta"] > 1.2
    assert 0 < result["angle_saturation_ratio"] < 1
    assert result["mean_abs_theta"] == 0.775


def test_imci_statistics_include_mean_entropy_and_per_hop_std():
    weights = torch.tensor([
        [[[0.8, 0.1, 0.1]], [[0.2, 0.3, 0.5]]],
        [[[0.2, 0.7, 0.1]], [[0.4, 0.4, 0.2]]],
    ])
    diagnostic = {
        "attention_weights": {"visual": weights},
        "hop_attention_entropy": {"visual": torch.tensor(0.8)},
    }
    result = analyzer._imci_stats(diagnostic, "visual")
    assert len(result["mean_hop_attention"]) == 3
    assert abs(result["mean_hop_attention_entropy"] - 0.8) < 1e-6
    assert all(value > 0 for value in result["per_hop_attention_std_across_nodes_heads"])


def test_formal_analyzer_has_no_test_metric_output_fields():
    assert "test_idx" not in analyzer.validation_view.__code__.co_names
    result = analyzer.validation_task_metrics(
        {"val_acc": 0.1, "val_macro_f1": 0.2},
        torch.tensor([[0.0, 1.0]]), torch.tensor([1]),
    )
    assert not any(key.startswith("test_") for key in result)



def test_streamed_full_graph_angle_moments_match_analyze_mean():
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from src.models.risa_v05 import Model

    cfg = SimpleNamespace(model=OmegaConf.create({
        "hidden_dim": 256, "dropout": 0.2, "max_order": 3,
        "relation_dim": 32, "edge_chunk_size": 2, "rotation_group_size": 2,
        "max_rotation_angle": 1.57079632679, "iamr_num_heads": 4,
        "iamr_ff_mult": 2, "node_chunk_size": 2, "variant": "v05_full",
    }))
    info = {"input_dim": 6, "text_dim": 3, "visual_dim": 3,
            "num_nodes": 4, "num_classes": 2}
    model = Model(cfg, info).eval()
    x = torch.randn(4, 6)
    edge = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    with torch.no_grad():
        diagnostic = model.analyze(x, edge, max_diagnostic_edges=32)
        streamed = analyzer._stream_angle_statistics(
            model, diagnostic, model.max_rotation_angle,
        )
    for modality in ("text", "visual"):
        assert abs(streamed[modality]["mean_abs_theta"] -
                   diagnostic["mean_abs_angle"][modality]) < 1e-6
        assert streamed[modality]["std_abs_theta"] >= 0
        assert 0 <= streamed[modality]["angle_saturation_ratio"] <= 1

