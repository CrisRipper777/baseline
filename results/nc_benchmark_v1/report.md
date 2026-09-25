# NC factorial and baseline benchmark report

NC checkpoints are selected only by validation accuracy. Test metrics are descriptive.
All means and standard deviations use seeds 42, 43, and 44; standard deviation is population SD.

## Multi-Order factorial metrics

| Dataset | Variant | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 |
|---|---|---:|---:|---:|---:|
| Movies | mob_terminal_plain | 54.84 ± 0.01 | 45.37 ± 0.19 | 54.78 ± 0.33 | 45.67 ± 0.31 |
| Movies | mob_uniform_plain | 57.54 ± 0.24 | 48.84 ± 0.36 | 56.47 ± 0.29 | 49.61 ± 0.84 |
| Movies | mob_gpr_plain | 57.72 ± 0.41 | 49.50 ± 0.61 | 56.39 ± 0.31 | 49.42 ± 0.51 |
| Movies | mob_terminal_residual | 54.65 ± 0.21 | 46.16 ± 1.02 | 54.43 ± 0.20 | 46.07 ± 1.36 |
| Movies | mob_uniform_residual | 57.36 ± 0.56 | 50.62 ± 0.65 | 56.53 ± 0.40 | 50.56 ± 0.78 |
| Movies | mob_gpr_residual | 57.60 ± 0.44 | 49.92 ± 1.18 | 56.73 ± 0.11 | 50.56 ± 0.26 |
| Toys | mob_terminal_plain | 79.83 ± 0.02 | 76.76 ± 0.28 | 78.96 ± 0.07 | 75.90 ± 0.14 |
| Toys | mob_uniform_plain | 80.67 ± 0.10 | 77.93 ± 0.09 | 79.70 ± 0.37 | 77.13 ± 0.08 |
| Toys | mob_gpr_plain | 80.61 ± 0.13 | 77.90 ± 0.20 | 79.61 ± 0.48 | 76.92 ± 0.30 |
| Toys | mob_terminal_residual | 79.40 ± 0.06 | 76.25 ± 0.21 | 78.63 ± 0.15 | 75.52 ± 0.26 |
| Toys | mob_uniform_residual | 80.56 ± 0.17 | 77.73 ± 0.36 | 79.36 ± 0.22 | 76.57 ± 0.18 |
| Toys | mob_gpr_residual | 80.56 ± 0.24 | 77.71 ± 0.34 | 79.66 ± 0.23 | 77.07 ± 0.37 |
| Grocery | mob_terminal_plain | 80.04 ± 0.17 | 70.74 ± 0.98 | 80.34 ± 0.04 | 69.32 ± 0.52 |
| Grocery | mob_uniform_plain | 82.99 ± 0.21 | 75.30 ± 1.51 | 83.35 ± 0.08 | 74.60 ± 1.02 |
| Grocery | mob_gpr_plain | 83.00 ± 0.14 | 75.61 ± 1.06 | 83.23 ± 0.28 | 74.95 ± 0.98 |
| Grocery | mob_terminal_residual | 80.88 ± 0.08 | 72.70 ± 0.74 | 80.76 ± 0.29 | 71.72 ± 0.40 |
| Grocery | mob_uniform_residual | 83.69 ± 0.16 | 77.03 ± 0.93 | 83.51 ± 0.38 | 75.80 ± 0.56 |
| Grocery | mob_gpr_residual | 83.80 ± 0.28 | 78.19 ± 0.28 | 83.69 ± 0.19 | 76.50 ± 0.31 |
| ele-fashion | mob_terminal_plain | 83.31 ± 0.14 | 67.95 ± 0.75 | 82.99 ± 0.19 | 67.90 ± 0.76 |
| ele-fashion | mob_uniform_plain | 87.56 ± 0.04 | 75.23 ± 0.65 | 87.68 ± 0.09 | 76.88 ± 0.35 |
| ele-fashion | mob_gpr_plain | 87.97 ± 0.07 | 76.38 ± 0.28 | 88.10 ± 0.10 | 77.69 ± 0.33 |
| ele-fashion | mob_terminal_residual | 83.95 ± 0.13 | 69.70 ± 0.51 | 83.75 ± 0.09 | 70.15 ± 0.69 |
| ele-fashion | mob_uniform_residual | 87.80 ± 0.17 | 75.94 ± 0.41 | 87.88 ± 0.12 | 77.22 ± 0.53 |
| ele-fashion | mob_gpr_residual | 88.04 ± 0.14 | 76.22 ± 0.74 | 88.02 ± 0.16 | 76.94 ± 0.47 |
| Reddit-S | mob_terminal_plain | 94.46 ± 0.03 | 90.47 ± 0.19 | 93.67 ± 0.14 | 88.61 ± 0.19 |
| Reddit-S | mob_uniform_plain | 95.95 ± 0.04 | 92.33 ± 0.41 | 95.99 ± 0.22 | 91.17 ± 0.73 |
| Reddit-S | mob_gpr_plain | 96.04 ± 0.09 | 92.48 ± 0.51 | 96.01 ± 0.14 | 91.31 ± 0.60 |
| Reddit-S | mob_terminal_residual | 94.40 ± 0.08 | 90.28 ± 0.16 | 93.53 ± 0.17 | 88.62 ± 0.40 |
| Reddit-S | mob_uniform_residual | 96.10 ± 0.04 | 92.39 ± 0.22 | 96.20 ± 0.13 | 91.90 ± 0.30 |
| Reddit-S | mob_gpr_residual | 96.20 ± 0.10 | 92.82 ± 0.52 | 96.21 ± 0.15 | 92.00 ± 0.29 |

## Preregistered paired contrasts

Positive differences favor the left-hand variant.

| Contrast | Mean Val Acc difference (pp) | Positive datasets | Positive seed pairs | Support |
|---|---:|---:|---:|---|
| uniform_plain - terminal_plain | 2.447 | 5/5 | 15/15 | STRONG_SUPPORT |
| uniform_residual - terminal_residual | 2.447 | 5/5 | 15/15 | STRONG_SUPPORT |
| gpr_plain - uniform_plain | 0.124 | 4/5 | 8/15 | MODERATE_SUPPORT |
| gpr_residual - uniform_residual | 0.137 | 4/5 | 11/15 | MODERATE_SUPPORT |
| terminal_residual - terminal_plain | 0.160 | 2/5 | 7/15 | UNSUPPORTED_OR_MIXED |
| uniform_residual - uniform_plain | 0.160 | 3/5 | 11/15 | MODERATE_SUPPORT |
| gpr_residual - gpr_plain | 0.174 | 3/5 | 10/15 | MODERATE_SUPPORT |

## Factorial decisions

- Fusion attribution trigger: not triggered.
- Fusion-robust direction for uniform_plain - terminal_plain / uniform_residual - terminal_residual: True.
- Fusion-robust direction for gpr_plain - uniform_plain / gpr_residual - uniform_residual: True.
- Practical support labels are descriptive preregistered rules, not statistical significance claims.
- Text/Visual GPR profiles, actual order contributions, hop cosine matrices, and frozen sensitivity are in the companion CSV files.
- Frozen sensitivity sets a selected gamma coefficient to zero without retraining; it is not a retrained ablation or a causal necessity test.
