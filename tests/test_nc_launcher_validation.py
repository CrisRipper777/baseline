from __future__ import annotations

import json

import torch

from scripts.preflight_nc_baselines import DATASETS, _select_heavy_datasets
from scripts.run_mob_factorial_nc import _checkpoint_errors
from scripts.run_nc_baselines import _can_resume, _valid_run_metrics


def _run_metrics_payload() -> dict:
    runs = []
    for seed in (42, 43, 44):
        runs.append({
            "seed": seed,
            "metrics": {"val_acc": 0.5, "val_macro_f1": 0.4, "test_acc": 0.45, "test_macro_f1": 0.35},
            "metadata": {
                "best_epoch": 3,
                "model_parameters": 1000,
                "optimizer": "AdamW",
                "optimizer_groups": [{"lr": 0.001, "weight_decay": 0.0001}],
            },
        })
    return {
        "protocol_version": "unified_full_graph_nc_v1",
        "run_seeds": [42, 43, 44],
        "runs": runs,
    }


def test_heavy_dataset_union_can_select_one_or_two_datasets() -> None:
    same = {dataset: {"num_nodes": 10, "num_edges": 20, "input_dim": 4} for dataset in DATASETS}
    same["Toys"]["num_nodes"] = 11
    same["Toys"]["num_edges"] = 21
    assert _select_heavy_datasets(same) == ("Toys",)

    distinct = {dataset: {"num_nodes": 10, "num_edges": 20, "input_dim": 4} for dataset in DATASETS}
    distinct["Toys"]["num_nodes"] = 11
    distinct["Reddit-S"]["num_edges"] = 21
    assert _select_heavy_datasets(distinct) == ("Toys", "Reddit-S")


def test_factorial_checkpoint_validation_checks_each_seed_and_protocol(tmp_path) -> None:
    paths = [tmp_path / f"best_run{run_id}.pt" for run_id in (1, 2, 3)]
    for seed, path in zip((42, 43, 44), paths):
        torch.save({
            "task": "nc",
            "protocol_version": "unified_full_graph_nc_v1",
            "seed": seed,
            "selection": "best_val_accuracy",
            "epoch": 2,
            "metrics": {"val_acc": 0.5, "val_macro_f1": 0.4, "test_acc": 0.45, "test_macro_f1": 0.35},
        }, path)
    assert _checkpoint_errors(paths, (42, 43, 44)) == []

    bad = torch.load(paths[1], map_location="cpu", weights_only=False)
    bad["seed"] = 999
    bad["metrics"].pop("test_macro_f1")
    torch.save(bad, paths[1])
    errors = _checkpoint_errors(paths, (42, 43, 44))
    assert any("expected seed 43" in error for error in errors)
    assert any("test_macro_f1" in error for error in errors)


def test_baseline_resume_requires_protocol_metrics_and_metadata(tmp_path) -> None:
    metrics_path = tmp_path / "per_run_metrics.json"
    results_path = tmp_path / "results.json"
    marker_path = tmp_path / "complete.json"
    metrics = _run_metrics_payload()
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    results_path.write_text("{}", encoding="utf-8")
    marker_path.write_text(json.dumps({
        "dataset": "Movies",
        "model": "gcn",
        "protocol_version": "unified_full_graph_nc_v1",
        "run_seeds": [42, 43, 44],
        "num_runs": 3,
        "results": str(results_path),
    }), encoding="utf-8")
    assert _valid_run_metrics(metrics)
    assert _can_resume(marker_path, metrics_path, "Movies", "gcn")

    metrics["runs"][2]["metadata"].pop("optimizer_groups")
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    assert not _valid_run_metrics(metrics)
    assert not _can_resume(marker_path, metrics_path, "Movies", "gcn")


def test_mechanism_discovery_launcher_job_counts_keep_three_internal_seeds():
    from scripts.run_mechanism_discovery_nc import _jobs
    order = list(_jobs("order", ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")))
    unimodal = list(_jobs("unimodal", ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")))
    preflight = list(_jobs("preflight", ()))
    assert len(order) == 25 and len(unimodal) == 20 and len(preflight) == 15
    assert all(job[0] == "order" for job in order)
    assert {job[1] for job in order + unimodal} == {"Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"}


def test_preflight_checkpoint_validation_does_not_require_test_metrics(tmp_path):
    import torch
    from scripts.run_mechanism_discovery_nc import _checkpoint_errors
    path = tmp_path / "best.pt"
    torch.save({"task": "nc", "protocol_version": "unified_full_graph_nc_v1",
                "seed": 42, "selection": "best_val_accuracy", "epoch": 1,
                "metrics": {"val_acc": .5, "val_macro_f1": .4}}, path)
    assert _checkpoint_errors([path], (42,), require_test_metrics=False) == []
    assert _checkpoint_errors([path], (42,), require_test_metrics=True)
