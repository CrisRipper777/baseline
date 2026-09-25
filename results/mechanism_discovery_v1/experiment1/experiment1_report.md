# Experiment 1: frozen mechanism audit (Step A)

This report uses existing terminal/uniform/GPR plain checkpoints. No model was retrained.
All probe fitting and validation reporting use train and validation labels only; test labels are not used.

Frozen probes: AdamW, lr=0.01, weight_decay=0.0001, epochs=200, seed=0.
A probe score is task information evidence, not a causal contribution.

The original preregistered `Uniform - Terminal` result remains frozen in `results/nc_benchmark_v1/`; this analysis writes to the separate mechanism_discovery_v1 directory.

Spectral eigensolver records: 5 datasets; see `spectral_summary.md`.

## Generated validation-only tables

- `hop_task_probe.csv`: 180 rows
- `incremental_probe.csv`: 135 rows
- `innovation_probe.csv`: 135 rows
- `propagation_innovation.csv`: 90 rows
- `representation_geometry.csv`: 240 rows
- `representation_smoothing.csv`: 60 rows
- `stability_audit.csv`: 405 rows
