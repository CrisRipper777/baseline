# S4.4 Phase 2 analysis patch

This analysis-only patch restores the omitted `variant` column in copies of the S4.3 H2a/H2b statistics. Historical CSVs and model outputs were read only; no training was run. Variant labels are assigned using the historical writer order (dataset, variant, seed, modality), and each source row's dataset/seed/modality is checked against that order before writing.

| Table | Historical rows | Patched rows |
|---|---:|---:|
| `h2a_gate_statistics_v2.csv` | 96 | 96 |
| `h2b_lambda_statistics_v2.csv` | 96 | 96 |

Scope: Movies, Grocery, ele-fashion, Reddit-S; seeds 42–44; modalities text and visual.
