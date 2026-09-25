# Mechanism & Problem Discovery v1

## Purpose

This stage audits the existing Multi-Order Bank and measures candidate failure modes on Movies, Toys, Grocery, ele-fashion, and Reddit-S node classification. It is an empirical diagnosis stage only; it does not introduce a research model or claim a contribution.

Frozen questions:

1. Does retaining multiple propagation orders outperform terminal-only propagation?
2. Does a learnable order response improve over uniform averaging?
3. Do Text and Visual learn different order responses?

## Evidence boundary

Problem selection, thresholds, correlations, probes, node regimes, edge roles, alignment, and frozen interventions use train and validation data only. Frozen NC checkpoints may report their already-defined test metrics as descriptive benchmark outputs; those values are never used in discovery tables, thresholds, probes, model selection, or status assignment. `scripts/analyze_mechanism_discovery.py` masks labels outside train and validation before probe fitting.

Evidence levels remain distinct:

- `DESCRIPTIVE_ONLY`: representation or prevalence statistics.
- `TASK_RELEVANT`: validation prediction, loss, or frozen probe evidence.
- `INTERVENTION_SUPPORTED`: a frozen targeted graph perturbation differs from a same-size random edge-removal control.

Cosine similarity, CKA, gamma values, or prevalence alone never establish a problem. No independent-node significance tests are reported.

## Fixed order-source controls

The existing `terminal`, `uniform`, and `gpr` readouts retain their original code paths. The new controls use the same independent Text and Visual propagation states and the same projector and plain fusion modules:

- `self_only`: `S0`
- `self25_terminal75`: `0.25*S0 + 0.75*S3`
- `self50_terminal50`: `0.50*S0 + 0.50*S3`
- `self75_terminal25`: `0.75*S0 + 0.25*S3`
- `propagated_uniform`: `(S1 + S2 + S3)/3`

All controls use `hidden_dim=256`, `max_order=3`, `dropout=0.2`, and `fusion_mode=plain_mlp` under `unified_full_graph_nc_v1`. The three alpha-scan interior points reuse the corresponding fixed controls; terminal and self-only are the endpoints. Alpha is not tuned.

## Unimodal controls

`modality_mode=both` is the backward-compatible default. `text` and `visual` execute only that projector and propagation branch and return its readout directly, without multimodal fusion. Controls are `text_self`, `text_uniform`, `visual_self`, and `visual_uniform`. Existing both-mode checkpoint parameter names and tensor shapes remain unchanged.

## Frozen probe protocol

`FrozenProbeProtocolV1` is a single `Linear(in_dim, num_classes)` head. It is fit on train nodes with AdamW (`lr=1e-2`, `weight_decay=1e-4`) for 200 epochs, fixed seed 0, no scheduler, and no early stopping. Validation is the reporting split. Nonfinite optimization raises an error; the optimizer settings are never auto-adjusted. The serialized protocol SHA256 is included in probe tables.

## Diagnostics

Experiment 1 uses `uniform_plain` best-validation checkpoints as its primary representation source and `gpr_plain` only as a robustness source. It records order geometry, hop and innovation probes, sparse spectral response, and frozen edge/feature perturbation stability. Oversmoothing/collapse wording is allowed only when node variance and effective rank decrease, normalized Dirichlet energy decreases, and sampled pairwise node cosine increases together.

Experiment 2 uses `uniform_plain` as its primary reference. It compares modality sufficiency, prediction regimes, branch probes, structure-attribute conflict, modality-specific topology utility, source-matched edge semantic compatibility, high-confidence edge-role removals against random controls, semantic-utility bins, modal disagreement, interaction probes, train-fitted cross-modal alignment, linear predictable/residual proxies, train-median interaction cells, and class-wise validation metrics. The residual analysis is named a linear proxy and is not an information-theoretic decomposition.

Edge removals operate on unique undirected pairs, remove both orientations, preserve model-added self-loops, and do not retrain. There is no per-edge deletion. No rewired-graph training is performed.

## Frozen evidence retained

The original `results/nc_benchmark_v1/`, `outputs/mob_factorial_nc_v1/`, and `outputs/nc_benchmark_v1/` are immutable inputs. The previously reported Uniform-minus-Terminal and GPR-minus-Uniform benchmark facts remain sourced from the original results tree and are not recalculated into or written over that tree.
