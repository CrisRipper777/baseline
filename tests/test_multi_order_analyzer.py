from __future__ import annotations

import pytest
import torch

from scripts.summarize_multi_order_bank_nc import analyze, _render_markdown


def test_analyzer_computes_paired_population_stats_and_exports_gammas(tmp_path) -> None:
    dataset = "Movies"
    for readout_idx, readout in enumerate(("terminal", "uniform", "gpr")):
        folder = tmp_path / dataset / readout
        folder.mkdir(parents=True)
        for run_id, seed in enumerate((42, 43, 44), start=1):
            base = float(run_id) / 10.0
            metrics = {
                "val_acc": base + readout_idx * 0.01,
                "val_macro_f1": base + readout_idx * 0.02,
                "test_acc": base + readout_idx * 0.03,
                "test_macro_f1": base + readout_idx * 0.04,
            }
            payload = {
                "task": "nc",
                "protocol_version": "unified_full_graph_nc_v1",
                "seed": seed,
                "selection": "best_val_accuracy",
                "epoch": run_id + 4,
                "metrics": metrics,
                "model_state": {},
            }
            if readout == "gpr":
                payload["model_state"] = {
                    "gamma_text": torch.full((4,), 0.25 + run_id / 100.0),
                    "gamma_visual": torch.full((4,), 0.15 + run_id / 100.0),
                }
            torch.save(payload, folder / f"best_run{run_id}.pt")

    result = analyze(tmp_path, datasets=(dataset,))
    assert result["run_seeds"] == [42, 43, 44]
    paired = result["paired_comparisons"][dataset]["uniform - terminal"]
    assert paired["val_acc"]["per_seed"] == pytest.approx([0.01, 0.01, 0.01])
    assert paired["val_acc"]["mean"] == pytest.approx(0.01)
    assert paired["val_acc"]["std"] == pytest.approx(0.0, abs=1e-16)
    assert result["per_dataset_readout"][dataset]["terminal"]["metrics"]["val_acc"]["std"] > 0.0
    assert result["gpr_gammas"][dataset][0]["gamma_text"] == pytest.approx([0.26] * 4)
    assert "Test Macro-F1" in _render_markdown(result)
