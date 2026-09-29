# RISA v0.6 formal screening summary

- Protocol: `unified_full_graph_nc_v1`; variant: `v06_full`.
- Runs: 9/9 validated; seeds 42, 43, 44.
- Checkpoints selected by Val Accuracy; test and LP evaluation were disabled.
- Statistics are mean ± population SD (ddof=0) over three seeds.
- Peak sampled process GPU memory: 18534 MiB.

## Validation metrics

| Dataset | Val Accuracy | Val Macro-F1 | Best epoch | Peak GPU (MiB) |
|---|---:|---:|---:|---:|
| Movies | 0.5583 ± 0.0029 | 0.4541 ± 0.0161 | 73.7 (57–94) | 5516 |
| Grocery | 0.8303 ± 0.0017 | 0.7586 ± 0.0046 | 111.3 (98–120) | 5338 |
| ele-fashion | 0.8816 ± 0.0004 | 0.7682 ± 0.0009 | 152.7 (146–161) | 18534 |

## Paired comparison with v0.5

Δ is v0.6 minus v0.5 in percentage points, paired by dataset and seed. These are validation metrics, not test-set estimates.

| Dataset | Baseline | Δ Val Accuracy (pp) | Δ Val Macro-F1 (pp) |
|---|---|---:|---:|
| Movies | v05_full | -0.37 | -0.70 |
| Movies | v05_absorb_only | -0.73 | -1.86 |
| Grocery | v05_full | +0.01 | -0.61 |
| Grocery | v05_absorb_only | +0.18 | -0.49 |
| ele-fashion | v05_full | +0.24 | +0.53 |
| ele-fashion | v05_absorb_only | +0.13 | +1.01 |

## Seed-level results

| Dataset | Seed | Best epoch | Val Accuracy | Val Macro-F1 |
|---|---:|---:|---:|---:|
| Movies | 42 | 94 | 0.5621 | 0.4767 |
| Movies | 43 | 57 | 0.5576 | 0.4407 |
| Movies | 44 | 70 | 0.5552 | 0.4449 |
| Grocery | 42 | 98 | 0.8322 | 0.7644 |
| Grocery | 43 | 120 | 0.8305 | 0.7582 |
| Grocery | 44 | 116 | 0.8281 | 0.7532 |
| ele-fashion | 42 | 146 | 0.8810 | 0.7672 |
| ele-fashion | 43 | 161 | 0.8818 | 0.7680 |
| ele-fashion | 44 | 151 | 0.8821 | 0.7693 |

## Interpretation

- All nine validation-selected checkpoints passed seed, protocol, finiteness, and run-metrics consistency checks.
- Against v0.5 Full, v0.6 is dataset-dependent: both validation metrics decrease on Movies, Grocery accuracy is nearly unchanged while Macro-F1 is lower, and both metrics improve on ele-fashion.
- Val Accuracy has low seed spread (SD 0.0004–0.0029); Val Macro-F1 varies more on Movies (SD 0.0161) than Grocery (0.0046) or ele-fashion (0.0009).
- These are paired validation-set comparisons over three seeds. No test set was evaluated, so they do not establish test-set generalization or statistical significance.
- v0.5 comparison sources: `results/risa_v05/formal_analysis/formal_risa_v05.json` and `results/risa_v05/main_ablation/risa_v05_main_ablation.json`.
- Full per-checkpoint provenance, metrics, and paired seed deltas are in `formal_risa_v06.json`; row-level values are in `formal_risa_v06.csv`.
