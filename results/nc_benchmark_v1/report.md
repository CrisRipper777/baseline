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

## NC baseline summary

| Dataset | Model | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 |
|---|---|---:|---:|---:|---:|
| Movies | mlp | 52.06 ± 0.15 | 38.75 ± 1.43 | 51.80 ± 0.13 | 40.26 ± 1.21 |
| Movies | gcn | 52.75 ± 0.13 | 43.75 ± 0.62 | 53.20 ± 0.20 | 46.17 ± 0.22 |
| Movies | sage | 52.84 ± 2.28 | 40.97 ± 10.74 | 52.80 ± 2.19 | 40.92 ± 11.67 |
| Movies | mmgcn | 54.87 ± 0.29 | 39.98 ± 2.99 | 55.46 ± 0.12 | 42.28 ± 3.30 |
| Movies | mgat | 53.54 ± 0.20 | 35.09 ± 1.96 | 53.16 ± 0.60 | 37.17 ± 1.35 |
| Movies | dip | 56.49 ± 0.22 | 47.79 ± 1.44 | 55.28 ± 0.19 | 47.25 ± 0.65 |
| Movies | dgf | 51.80 ± 0.04 | 26.11 ± 0.85 | 51.76 ± 0.27 | 28.99 ± 0.10 |
| Movies | dmgc | 47.03 ± 1.15 | 22.12 ± 2.55 | 47.70 ± 0.98 | 23.47 ± 2.47 |
| Movies | lgmrec | 55.92 ± 0.12 | 48.50 ± 0.77 | 55.59 ± 0.30 | 49.20 ± 0.09 |
| Toys | mlp | 76.48 ± 0.14 | 74.03 ± 0.58 | 74.78 ± 0.17 | 71.85 ± 0.17 |
| Toys | gcn | 79.45 ± 0.16 | 76.17 ± 0.29 | 78.58 ± 0.12 | 75.11 ± 0.44 |
| Toys | sage | 79.69 ± 0.25 | 77.13 ± 0.55 | 78.52 ± 0.31 | 76.08 ± 0.50 |
| Toys | mmgcn | 79.43 ± 0.15 | 76.07 ± 0.17 | 78.29 ± 0.52 | 75.21 ± 0.71 |
| Toys | mgat | 78.30 ± 0.23 | 73.78 ± 0.79 | 77.82 ± 0.10 | 74.02 ± 0.27 |
| Toys | dip | 80.74 ± 0.05 | 77.48 ± 0.18 | 79.23 ± 0.35 | 76.09 ± 0.60 |
| Toys | dgf | 79.28 ± 0.05 | 75.79 ± 0.20 | 78.44 ± 0.10 | 75.11 ± 0.37 |
| Toys | dmgc | 72.78 ± 1.51 | 69.47 ± 0.72 | 71.26 ± 1.33 | 67.45 ± 0.27 |
| Toys | lgmrec | 80.30 ± 0.10 | 77.19 ± 0.11 | 79.18 ± 0.02 | 76.52 ± 0.13 |
| Grocery | mlp | 78.76 ± 0.24 | 71.58 ± 0.73 | 78.46 ± 0.19 | 70.39 ± 0.60 |
| Grocery | gcn | 80.62 ± 0.08 | 71.95 ± 0.26 | 81.18 ± 0.18 | 71.53 ± 0.22 |
| Grocery | sage | 83.29 ± 0.12 | 76.86 ± 0.29 | 82.81 ± 0.35 | 74.58 ± 0.34 |
| Grocery | mmgcn | 81.84 ± 0.28 | 74.12 ± 0.42 | 82.18 ± 0.20 | 72.51 ± 0.69 |
| Grocery | mgat | 80.71 ± 0.11 | 68.52 ± 0.90 | 80.60 ± 0.38 | 67.83 ± 1.37 |
| Grocery | dip | 83.71 ± 0.27 | 77.44 ± 0.58 | 83.72 ± 0.30 | 75.43 ± 0.43 |
| Grocery | dgf | 80.76 ± 0.11 | 68.58 ± 0.76 | 81.31 ± 0.38 | 68.88 ± 0.93 |
| Grocery | dmgc | 73.15 ± 2.92 | 61.43 ± 2.17 | 71.79 ± 3.80 | 59.55 ± 3.52 |
| Grocery | lgmrec | 83.18 ± 0.11 | 77.24 ± 0.30 | 82.88 ± 0.08 | 75.33 ± 0.21 |
| ele-fashion | mlp | 88.10 ± 0.04 | 77.40 ± 0.58 | 88.12 ± 0.07 | 77.82 ± 0.32 |
| ele-fashion | gcn | 85.40 ± 0.09 | 72.42 ± 0.71 | 85.11 ± 0.11 | 72.95 ± 0.28 |
| ele-fashion | sage | 87.74 ± 0.11 | 76.24 ± 0.48 | 87.87 ± 0.13 | 77.28 ± 0.35 |
| ele-fashion | mmgcn | 87.68 ± 0.08 | 75.45 ± 0.61 | 87.57 ± 0.12 | 75.85 ± 0.80 |
| ele-fashion | mgat | 86.62 ± 0.05 | 70.72 ± 0.64 | 86.72 ± 0.13 | 71.18 ± 1.28 |
| ele-fashion | dip | 88.01 ± 0.14 | 75.47 ± 0.98 | 87.97 ± 0.16 | 76.44 ± 0.80 |
| ele-fashion | dgf | 86.89 ± 0.02 | 71.94 ± 0.27 | 86.83 ± 0.03 | 72.52 ± 0.38 |
| ele-fashion | dmgc | 86.63 ± 0.05 | 70.67 ± 1.31 | 86.59 ± 0.11 | 71.45 ± 0.92 |
| ele-fashion | lgmrec | 85.82 ± 0.18 | 70.40 ± 1.05 | 85.68 ± 0.31 | 71.29 ± 1.52 |
| Reddit-S | mlp | 92.67 ± 0.13 | 87.72 ± 0.12 | 92.74 ± 0.05 | 87.23 ± 0.29 |
| Reddit-S | gcn | 94.25 ± 0.15 | 90.60 ± 0.07 | 93.56 ± 0.04 | 89.02 ± 0.10 |
| Reddit-S | sage | 95.33 ± 0.08 | 91.75 ± 0.13 | 94.86 ± 0.06 | 90.54 ± 0.09 |
| Reddit-S | mmgcn | 95.84 ± 0.23 | 92.31 ± 0.43 | 95.65 ± 0.06 | 91.17 ± 0.21 |
| Reddit-S | mgat | 95.81 ± 0.17 | 91.90 ± 0.33 | 95.71 ± 0.17 | 90.61 ± 0.18 |
| Reddit-S | dip | 96.25 ± 0.07 | 92.63 ± 0.11 | 96.43 ± 0.12 | 92.38 ± 0.53 |
| Reddit-S | dgf | 96.04 ± 0.08 | 91.34 ± 0.42 | 95.99 ± 0.16 | 90.75 ± 0.82 |
| Reddit-S | dmgc | 92.79 ± 0.65 | 86.95 ± 2.14 | 92.83 ± 0.83 | 86.24 ± 2.52 |
| Reddit-S | lgmrec | 95.91 ± 0.13 | 92.43 ± 0.35 | 95.99 ± 0.04 | 91.89 ± 0.13 |
