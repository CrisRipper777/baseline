from __future__ import annotations

import pytest
import torch

from scripts.summarize_multi_order_bank_nc import analyze, render_report


DATASET = "Movies"
VARIANTS = (
    "mob_terminal_plain", "mob_uniform_plain", "mob_gpr_plain",
    "mob_terminal_residual", "mob_uniform_residual", "mob_gpr_residual",
)


def test_analyzer_computes_factorial_pairs_population_stats_and_gammas(tmp_path) -> None:
    for variant_idx, variant in enumerate(VARIANTS):
        folder = tmp_path / DATASET / variant
        folder.mkdir(parents=True)
        for run_id, seed in enumerate((42, 43, 44), start=1):
            base = float(run_id) / 10.0
            is_residual = variant.endswith("residual")
            readout_idx = {"terminal": 0, "uniform": 1, "gpr": 2}[variant.split("_")[1]]
            metrics = {
                "val_acc": base + readout_idx * 0.01 + int(is_residual) * 0.005,
                "val_macro_f1": base + readout_idx * 0.02 + int(is_residual) * 0.006,
                "test_acc": base + readout_idx * 0.03 + int(is_residual) * 0.007,
                "test_macro_f1": base + readout_idx * 0.04 + int(is_residual) * 0.008,
            }
            payload = {
                "task": "nc",
                "protocol_version": "unified_full_graph_nc_v1",
                "seed": seed,
                "selection": "best_val_accuracy",
                "epoch": run_id + 4,
                "metrics": metrics,
                "model_state": {},
                "data_info": {},
            }
            if "_gpr_" in variant:
                payload["model_state"] = {
                    "gamma_text": torch.full((4,), 0.25 + run_id / 100.0),
                    "gamma_visual": torch.full((4,), 0.15 + run_id / 100.0),
                }
            torch.save(payload, folder / f"best_run{run_id}.pt")

    result = analyze(tmp_path, datasets=(DATASET,))
    assert result["seeds"] == [42, 43, 44]
    paired = result["paired_contrasts"]["uniform_plain - terminal_plain"]
    assert paired["per_dataset"][DATASET]["val_acc"]["per_seed"] == pytest.approx([0.01] * 3)
    assert paired["val_acc_summary"]["mean_pp"] == pytest.approx(1.0)
    assert paired["val_acc_summary"]["population_std_across_seed_pairs"] == pytest.approx(0.0, abs=1e-15)
    assert result["fusion_attribution_required"] is False
    assert result["factorial"][DATASET]["mob_terminal_plain"]["metrics"]["val_acc"]["std"] > 0.0
    assert result["per_run_rows"][0]["seed"] == 42
    assert "Fusion attribution trigger" in render_report(result)
