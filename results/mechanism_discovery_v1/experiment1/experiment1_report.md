# Experiment 1: Multi-Order mechanism audit

The archived uniform/terminal result remains frozen as reported in the original benchmark outputs: +2.447 percentage points Validation Accuracy, 5/5 datasets and 15/15 paired seeds. GPR over Uniform remains +0.13 percentage points. This report does not recompute or overwrite those facts.

The new fixed readout controls are analyzed separately. Test metrics are descriptive only.

## New paired validation contrasts

| Contrast | Mean (pp) | Population SD (pp) | Positive seed pairs |
|---|---:|---:|---:|
| G_mid | 0.058 | 0.275 | 7/15 |
| G_prop | 1.094 | 0.931 | 12/15 |
| G_self_25 | 2.389 | 1.058 | 15/15 |
| self50_minus_terminal | 1.855 | 1.574 | 14/15 |
| self75_minus_terminal | 0.481 | 2.240 | 9/15 |
| self_only_minus_terminal | -1.195 | 3.008 | 3/15 |

## Alpha scan

Alpha is the fixed coefficient on S0 in alpha*S0 + (1-alpha)*S3. Values are validation summaries across the fixed controls; no alpha was tuned.

| alpha | Val Accuracy mean | Val Macro-F1 mean |
|---:|---:|---:|
| 0 | 0.7850 | 0.7026 |
| 0.25 | 0.8088 | 0.7370 |
| 0.5 | 0.8035 | 0.7327 |
| 0.75 | 0.7898 | 0.7170 |
| 1 | 0.7730 | 0.6959 |

## Existing-checkpoint diagnostics

Hop-probe mean Validation Accuracy by modality and order:

| Modality | S0 | S1 | S2 | S3 |
|---|---:|---:|---:|---:|
| text | 0.6879 | 0.7659 | 0.7697 | 0.7592 |
| visual | 0.7212 | 0.7750 | 0.7838 | 0.7752 |
| concat | 0.7970 | 0.8122 | 0.8115 | 0.7961 |

Cumulative innovation-probe Accuracy gain over S0:

| Highest added order | Mean gain | Positive comparisons |
|---:|---:|---:|
| 1 | 0.0307 | 42/45 |
| 2 | 0.0311 | 37/45 |
| 3 | 0.0306 | 38/45 |

Four-signal progressive smoothing/collapse checks met: 60/60 rows.
This requires variance decrease, effective-rank decrease, Dirichlet-energy decrease, and sampled node-cosine increase together.

The frozen stability perturbations were rerun after the sparse operator cache was fixed: True.

See representation_geometry.csv, hop_task_probe.csv, propagation_innovation.csv, innovation_probe.csv, incremental_probe.csv, spectral_response.csv and stability_audit.csv for full details. Probe accuracy is evidence of task information, not a causal contribution.

Original checkpoint outputs and benchmark summaries were not modified.
