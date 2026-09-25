# Multi-Order Bank specification

## Scope

Multi-Order Bank is a small experimental backbone in this repository. It is not a paper model and carries no novelty, contribution, or MAG-specific mechanism claim. Its purpose is to measure whether explicit retention and readout of several physical-graph propagation orders behaves differently from terminal-only propagation.

The three frozen empirical questions are:

- Q1: Does retaining multiple propagation orders outperform terminal-only propagation? Compare terminal and uniform.
- Q2: Does learnable multi-order response outperform uniform averaging? Compare gpr and uniform.
- Q3: Do Text and Visual learn different order responses? Compare gamma_text and gamma_visual.

## Encoder

Text and Visual paths are independent from input projection through their per-modality readout.

For modality m in {text, visual}:

- H0_m = projector_m(X_m)
- Projector_m = Linear(input_dim_m, hidden_dim), LayerNorm, ReLU, Dropout.
- The two projectors have independent parameters and matching structure.
- Both modalities use the same physical graph operator, while their feature states remain separate.

The physical operator is the standard symmetric-normalized GCN operator:

P = D_tilde^(-1/2) A_tilde D_tilde^(-1/2)

A_tilde is the original physical adjacency after making its undirected representation explicit and adding exactly one self-loop per node. No semantic edges or learned graph weights are added.

Propagation order is fixed at three:

- S0_m = H0_m
- Sk_m = P S(k-1)_m for k in {1, 2, 3}

All four states remain available through the analysis API.

## Readouts

The only model-config difference among the three variants is model.readout.

- terminal: Z_m = S3_m
- uniform: Z_m = (S0_m + S1_m + S2_m + S3_m) / 4
- gpr: Z_m = sum over k=0..3 of gamma_m[k] * Sk_m

GPR has independent learnable gamma_text and gamma_visual vectors of length four. Both initialize to [0.25, 0.25, 0.25, 0.25]. They are direct signed coefficients; no softmax, node-dependent routing, or attention is applied.

## Fusion and task interface

After per-modality readout, concatenate Z_text and Z_visual and apply one Linear(2 * hidden_dim, hidden_dim) fusion projection. The formal config uses hidden_dim=128 and dropout=0.2. There is no cross-modal attention, early fusion, mixture of experts, semantic edge weighting, restart, prototype, auxiliary loss, or graph rewiring.

The model exposes out_dim=hidden_dim and the common five-field task forward tuple with a zero scalar auxiliary loss. NC uses the unified_full_graph_nc_v1 full-graph training protocol. LP can use the unified_sampled_lp_v1 LinkNeighborLoader interface; the model opts into depth resolution and has num_layers=3, matching the fixed [5, 5, 5] fanouts.

## Analysis API

Model.analyze(x, edge_index) returns:

- H0_text and H0_visual
- S_text and S_visual, each ordered [S0, S1, S2, S3]
- Z_text and Z_visual
- fused_z
- gamma_text and gamma_visual for gpr (None for terminal/uniform)

No attention, semantic similarity gate, expert score, or prototype score is computed.

## Formula-level checks

tests/test_multi_order_bank.py verifies state recurrence, the explicit normalized operator, all three exact readouts, signed unnormalized GPR coefficients, gamma initialization and modality independence, physical-graph-only propagation, finite forward/backward values, inference, and NC/LP interface compatibility.

## Current smoke status

Movies NC with seed 42, num_runs=1, and epochs=2 completed for terminal, uniform, and gpr. Each run trained, selected and saved a validation checkpoint, evaluated validation and test metrics, and returned finite values. These are interface smokes only; no results are interpreted as research evidence.
