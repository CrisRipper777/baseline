# S4.3 P0/H1 pilot report

- Formal jobs: 24/24 checkpoint-validated; formal runs: 72/72.
- Failed or not-started jobs: 0.
- Test evaluation was disabled for every formal job. Selection used best validation accuracy.
- Protocol: unified_full_graph_nc_v1; seed command 42; internal seeds 42/43/44; 300 epochs maximum.

## P0: Intrinsic Evidence Preservation

Status: **STRONG_SUPPORT**. Primary residual-minus-relation-only validation deltas are positive for both metrics on all four datasets.
Primary paired contrast: p0_residual minus p0_relation_only; trainable parameter counts match exactly.
| Contrast | Δ Val Accuracy (pp; mean ± population SD) | Positive pairs; dataset means | Δ Val Macro-F1 (pp; mean ± population SD) | Positive pairs; dataset means |
|---|---:|---:|---:|---:|
| p0_residual - p0_relation_only | +6.68 ± 1.54 pp | 12/12 seed pairs; 4/4 dataset means | +10.41 ± 3.19 pp | 12/12 seed pairs; 4/4 dataset means |
| p0_concat - p0_relation_only | +6.32 ± 1.61 pp | 12/12 seed pairs; 4/4 dataset means | +9.72 ± 3.52 pp | 12/12 seed pairs; 4/4 dataset means |
| p0_concat - p0_residual | -0.36 ± 0.47 pp | 4/12 seed pairs; 2/4 dataset means | -0.69 ± 1.60 pp | 7/12 seed pairs; 2/4 dataset means |

The primary contrast is +6.68 pp Val Accuracy and +10.41 pp Val Macro-F1, positive in all 12 paired seeds and all four dataset means.
p0_concat is marked **CAPACITY_DIFFERENT_SECONDARY_CONTROL**. Its comparison against p0_relation_only cannot alone establish intrinsic preservation.
See p0_table.csv and p0_paired_contrasts.csv for per-dataset and per-seed validation results.

## H1: Functional Relational Basis Diversity

Status: **MIXED**. Performance direction and/or functional diagnostics are mixed.
Primary comparison: h1_dual_functional_static minus h1_dual_agg. Both use two transforms and have matched trainable parameter counts; only structural basis inputs differ.
| Contrast | Δ Val Accuracy (pp; mean ± population SD) | Positive pairs; dataset means | Δ Val Macro-F1 (pp; mean ± population SD) | Positive pairs; dataset means |
|---|---:|---:|---:|---:|
| dual_agg - single_agg | +0.09 ± 0.32 pp | 7/12 seed pairs; 3/4 dataset means | +0.20 ± 1.32 pp | 8/12 seed pairs; 2/4 dataset means |
| dual_functional_static - dual_agg | -1.28 ± 1.17 pp | 0/12 seed pairs; 0/4 dataset means | -2.94 ± 3.90 pp | 3/12 seed pairs; 1/4 dataset means |
| dual_functional_global - dual_functional_static | +0.02 ± 0.27 pp | 4/12 seed pairs; 1/4 dataset means | -0.14 ± 1.79 pp | 5/12 seed pairs; 1/4 dataset means |
| dual_functional_global - single_agg | -1.17 ± 1.18 pp | 0/12 seed pairs; 0/4 dataset means | -2.88 ± 4.17 pp | 3/12 seed pairs; 0/4 dataset means |

For the primary capacity-matched contrast, static functional basis is lower by 1.28 pp Val Accuracy (0/12 positive seed pairs; 0/4 positive dataset means) and 2.94 pp Val Macro-F1 (3/12; 1/4). Functional diagnostics are non-degenerate, so performance and representation evidence point in different directions.
This tests functional structural basis diversity against repeated aggregative capacity. It does not establish two relation types: U_D=sH-U_A is linearly related to H and U_A.
See h1_table.csv and h1_paired_contrasts.csv for per-dataset and per-seed results.

## Basis functionality

Validation-node means at best-validation checkpoints; no NaN, zero-basis, or near-unit-cosine flags were raised.
| Variant | Modality | mean ||R_A|| | mean ||R_D|| | mean cosine(R_A,R_D) | cosine SD | norm ratio A/D |
|---|---|---:|---:|---:|---:|---:|
| h1_dual_functional_static | text | 11.080 | 11.149 | 0.340 | 0.059 | 0.994 |
| h1_dual_functional_static | visual | 11.260 | 11.131 | 0.328 | 0.060 | 1.013 |
| h1_dual_functional_global | text | 11.125 | 11.157 | 0.341 | 0.058 | 0.997 |
| h1_dual_functional_global | visual | 11.395 | 11.138 | 0.327 | 0.060 | 1.024 |

## Frozen basis interventions and global mixture

These are frozen checkpoint sensitivity checks without retraining, not causal retrained ablations.
| Variant | Intervention | Δ Val Acc (pp) | Δ Val Macro-F1 (pp) | Δ true-label CE | prediction flip rate |
|---|---|---:|---:|---:|---:|
| h1_dual_functional_static | FORCE_A | -0.32 | -1.56 | +0.0550 | 0.0873 |
| h1_dual_functional_static | FORCE_D | -8.89 | -9.08 | +0.2228 | 0.1728 |
| h1_dual_functional_global | FORCE_A | -0.07 | -0.96 | +0.0513 | 0.0839 |
| h1_dual_functional_global | FORCE_D | -9.40 | -9.45 | +0.2563 | 0.1770 |
| h1_dual_functional_global | SWAP_MODALITY_ALPHA | -0.04 | -0.03 | +0.0001 | 0.0011 |

Mean global coefficients were Text A/D=0.5033/0.4967 and Visual A/D=0.5011/0.4989; all remain close to the initialized 0.5/0.5 mixture.
SWAP_MODALITY_ALPHA had near-zero average validation change and a 0.0011 mean prediction flip rate. Learned coefficients are not causal contributions; interpret them with FORCE_A/FORCE_D.
Across both functional variants, FORCE_A changed accuracy by less than 0.33 pp on average, while FORCE_D reduced accuracy by about 8.9–9.4 pp and increased true-label CE. This is frozen sensitivity only.
See h1_forced_basis_interventions.csv and h1_global_coefficients.csv for dataset/seed results.

## Complexity

Trainable parameters below include the classifier head. Peak CUDA memory is sampled nvidia-smi device memory above the pre-job baseline.
| Variant | Trainable params across input shapes | Peak CUDA memory range (MB) | Mean job wall time (s) |
|---|---:|---:|---:|
| p0_relation_only | 597,516, 730,644 | 1033–3635 | 50.9 |
| p0_residual | 597,516, 730,644 | 1015–3835 | 53.2 |
| p0_concat | 860,172, 993,300 | 1141–4215 | 74.9 |
| h1_dual_agg | 730,124, 863,252 | 1179–4407 | 62.4 |
| h1_dual_functional_static | 730,124, 863,252 | 1213–4599 | 56.5 |
| h1_dual_functional_global | 730,128, 863,256 | 1215–5694 | 63.2 |

H1 dual-aggregative and static-functional variants have exactly matched parameter counts within each dataset. Mean epoch time is a job-wall-time/observed-epochs proxy that includes setup and validation overhead; see complexity_table.csv.

## QA and provenance

- QA: full pytest 154 passed; compileall, git diff --check, both dry-runs passed; ele-fashion preflight 6/6 passed.
- Source: problem_deep_dive at e535ad91911557790658ddc87bfaa516da531aad.
- Training: s43_p0_h1 at e5c39c14315fcdaaf1e5e9d085f6538e4380a795.
- Analysis: s43_p0_h1 at cb45fc4cb27ebae9e0811a660e49b5d194283850.
- Result files: results/s43_p0_h1_v1; run logs/checkpoints/completion records: outputs/s43_p0_h1_v1/formal/.
- Detailed provenance and configuration/code hashes: outputs/s43_p0_h1_v1/provenance.json.

## Research boundary

This report covers only P0 and H1. No H2/H3/H4 or next-stage model was implemented or launched.
