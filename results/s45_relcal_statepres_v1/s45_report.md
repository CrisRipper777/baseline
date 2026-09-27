# S4.5 Relation-Calibrated State-Preserving Propagation

Training commit: `3693d351d2d722dde2d2d34c421ddb342440d659`; analysis commit: `8460c1a0368f944ffd76f6ef7b7195d609f42fdc`; source branch/SHA: `s44_relational_transform` / `24c3eec6325a6606d6a4c222ab418b4e616148c1`.
Scope: unified full-graph node classification on Movies, Grocery, ele-fashion and Reddit-S; seeds 42–44; best-validation-accuracy selection; test disabled. Toys remains an architecture holdout. No LP results are included.

## Findings

- ControllerGranularity: **STRONG_SUPPORT**. Mean within-target variance ratio: scalar 0.2615, low-rank 0.0786, expert 0.0875. Scalar exceeds the matched low-rank and expert ratios in 23/24 dataset-seed-modality comparisons. Variance describes controller assignment, not relation utility.
- S4.4 scalar frozen probes: DOSAGE_ONLY ΔAcc -0.000388, ΔMacro-F1 -0.001052, ΔCE -0.000010; REDISTRIBUTION_ONLY ΔAcc -0.000726, ΔMacro-F1 -0.001605, ΔCE +0.000977. The small changes do not identify one contribution as dominant; these are frozen sensitivities, not retrained causal ablations.
- RelationRedistribution: **MIXED**. Mass-preserving entry uniform minus identity uniform: ΔAcc -0.000197 (5/12 positive seed pairs), ΔMacro-F1 -0.002339, ΔCE +0.016936. Shuffling within-target edge assignments reduced validation accuracy in 428/480 renormalized shuffles, indicating sensitivity to assignment within the trained checkpoint despite little mean accuracy gain.
- StatePreservation: **STRONG_SUPPORT**. Identity uniform minus terminal: ΔAcc +0.029062 (12/12 positive seed pairs), ΔMacro-F1 +0.044539, ΔCE -0.077365. Under calibration, uniform minus terminal: ΔAcc +0.028012 (12/12 positive seed pairs), ΔMacro-F1 +0.040276, ΔCE -0.073745.
- Intrinsic-state contribution: identity uniform minus propagated-uniform: ΔAcc +0.015966 (12/12 positive seed pairs), ΔMacro-F1 +0.030607, ΔCE -0.042226; mass-preserving entry uniform minus propagated-uniform: ΔAcc +0.012886 (12/12 positive seed pairs), ΔMacro-F1 +0.018533, ΔCE -0.066609.
- CalibrationStateInteraction: **MIXED**. Factorial interaction I: Acc mean -0.001049 (SD 0.003820, positive 5/12); Macro-F1 mean -0.004263 (SD 0.018868, positive 6/12); CE mean +0.003619 (SD 0.045464, positive 6/12). This does not support positive calibration-by-preservation synergy.
- CalibrationPlacement: **MIXED**. Persistent minus entry-only uniform: ΔAcc -0.001183 (3/12 positive seed pairs), ΔMacro-F1 +0.005898, ΔCE +0.025722.
- Row-mass audit: preserved variants max absolute row error 6.655e-07; max self-diagonal error 0.000e+00. Unconstrained max row deviation 15.0752 is expected and reported descriptively.
- Complexity: 0/96 runs flagged COMPUTE_HEAVY; max peak CUDA memory 9.56 GiB; max epoch ratio 2.03× historical MOB uniform.

## Historical identity audit

The independent identity checkpoint cross-check exceeded the initial 1e-4 screen (max |ΔAcc| 0.0063; max |ΔMacro-F1| 0.0310). The audit replayed all 24 historical MOB checkpoints on the fixed validation splits; stored metrics reproduced within 6.0e-08, and the same weights in the identity pilot differed by at most 7.6e-06 in logits. The remaining differences compare independently trained checkpoints and are descriptive, not evidence for calibration. See `historical_mob_checkpoint_audit.csv` and `historical_mob_crosscheck.csv`.

## Interpretation boundaries

Row-mass preservation fixes one-step off-diagonal target mass only; it does not guarantee symmetry or spectral equivalence. Frozen interventions are not retrained causal ablations. R1 is a fixed carrier, not a proven superior module. Gate magnitude is not causal relation utility. Positive performance does not establish synergy; inspect the factorial interaction. These NC development results do not generalize to LP. Toys was not used.

This pilot does not by itself freeze the paper backbone. Review the raw paired results, mechanism audits, state diagnostics, identity checks, and factorial interaction before freezing relation → propagation → state composition.

Detailed tables: `s45_table.csv`, `s45_paired_contrasts.csv`, `s45_factorial_interaction.csv`, `gate_statistics.csv`, `row_mass_audit.csv`, `operator_asymmetry.csv`, `calibration_interventions.csv`, `state_change_diagnostics.csv`, `state_geometry.csv`, `complexity_table.csv`.
