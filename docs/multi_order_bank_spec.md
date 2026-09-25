# Multi-Order Bank specification

## Scope

Multi-Order Bank is a simple experimental backbone, not a paper model. It measures propagation-order and late-fusion effects without making a novelty, contribution, or MAG-specific mechanism claim. The fixed capacity for this round is `hidden_dim=256`, `max_order=3`, `num_layers=3`, and `dropout=0.2`. These values are frozen before the formal factorial runs.

## Independent encoders and physical propagation

Text and Visual use separate projectors and remain independent through propagation and modality readout:

- `H0_m = projector_m(X_m)`
- `projector_m = Linear(input_dim_m, 256) → LayerNorm → ReLU → Dropout(0.2)`
- `P = D_tilde^(-1/2) A_tilde D_tilde^(-1/2)`
- `A_tilde` is the physical graph made undirected, with one self-loop per node.
- `S0_m=H0_m`, then `S1_m=P S0_m`, `S2_m=P S1_m`, and `S3_m=P S2_m`.

The same physical operator is used for both modalities. Feature states never cross between modalities before fusion. There is no semantic edge weighting, restart, attention, MoE, node-adaptive routing, graph rewiring, or auxiliary loss.

## Readouts

`model.readout` selects one of three exact formulas:

- `terminal`: `Z_m = S3_m`
- `uniform`: `Z_m = (S0_m + S1_m + S2_m + S3_m) / 4`
- `gpr`: `Z_m = sum(k=0..3) gamma_m[k] * S_k_m`

GPR has separate learnable Text and Visual vectors initialized to `[0.25, 0.25, 0.25, 0.25]`. Coefficients are signed and used directly, without softmax.

## Fusion endpoints

Fusion happens only after both modality readouts. The six formal variants are the 3 readouts crossed with the 2 fusion modes.

`plain_mlp` uses `U=concat(Z_text,Z_visual)` and exactly:

`Linear(512,256) → ReLU → Dropout(0.2) → Linear(256,256)`.

`residual` independently refines each modality:

`Zbar_m = LayerNorm(Z_m + Linear(256,256) → ReLU → Dropout(0.2) → Linear(256,256)(Z_m))`.

It then concatenates `U=concat(Zbar_text,Zbar_visual)` and computes:

`Z = LayerNorm(Linear_skip(512,256)(U) + MLP_fuse(U))`,

where `MLP_fuse = Linear(512,256) → ReLU → Dropout(0.2) → Linear(256,256)`.

This specifies only the requested late-fusion equations. No other V3 research mechanism is included. The old single-linear fusion remains available as `fusion_mode=linear` for debugging; it is not a factorial endpoint.

All variants expose `out_dim=256`. Fusion choice does not alter propagation states. The task head is the same NC linear classifier for every variant.

## Analysis API and tests

`Model.analyze(x, edge_index)` returns `H0_text`, `H0_visual`, `S_text`, `S_visual`, `Z_text`, `Z_visual`, `refined_text`, `refined_visual`, `fusion_input`, `fused_z`, and the GPR coefficients when present.

`tests/test_multi_order_bank.py` verifies the physical operator and `S0`–`S3` recurrence, exact readouts, signed GPR behavior and initialization, independent modality parameters/states, exact `plain_mlp` and residual equations, branch independence before concatenation, unchanged propagation across readout/fusion variants, finite backward/inference, and NC/LP interface compatibility.

## Formal questions and analysis

- Q1: Does uniform multi-order retention improve over terminal-only propagation, in each fusion setting?
- Q2: Does GPR improve over uniform averaging, in each fusion setting?
- Q3: Does the residual fusion endpoint improve over `plain_mlp`?
- Q4: Do Text and Visual have different GPR order profiles across fusion settings?
- Q5: Are the propagation states complementary or highly redundant?

The analyzer exports signed and absolute gamma profiles, effective orders, Text/Visual profile distance and cosine, actual gamma-weighted order contribution, `S0`–`S3` pairwise mean nodewise cosine matrices, and frozen validation sensitivity. Sensitivity zeros gamma coefficients without retraining; it is not a retrained ablation or causal-necessity test.

Paired support labels use validation accuracy only: strong support requires at least +0.30 percentage points mean across five datasets, positive means on at least 4/5 datasets, and at least 10/15 same-seed pairs positive. Moderate support requires positive mean and at least 3/5 positive dataset means. These are practical evidence labels, not significance claims. Test metrics are descriptive only.
