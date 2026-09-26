# D3 problem deep-dive report

Scope: Movies, Toys, Grocery, ele-fashion, Reddit-S; node classification only.
All selector fits, thresholds, edge grouping, intervention ranking, and gates use Train/Validation only.
Test metrics appear only as descriptive fields for the formal retrained controls; no research decision uses Test.
Formal controls completed: 20 operator jobs × 3 seeds + 5 rewired jobs × 3 seeds = 75 runs.

## Frozen questions

1. **Does attribute anchoring transfer beyond fixed GPR-style diffusion?**
   No under the registered transfer gate: `OPERATOR_DEPENDENT`. The mean anchored25−deep_only validation gain was GCN +0.084 pp (positive in 3/5) and SAGE +0.048 pp (positive in 2/5).
   | Operator | Mean paired gain | Positive datasets |
   |---|---:|---:|
| GCN | +0.084 pp | 3/5 |
| SAGE | +0.048 pp | 2/5 |
2. **Does conditional relation utility remain after degree and normalized-weight matching?**
   Registered Gate A: `STRONG_SUPPORT`. Under FIXED_NORM_MASK, the low-similarity/high-novelty group had positive validation cross-entropy increase beyond 20 exact matched controls in 4/5 datasets (Grocery, Movies, Reddit-S, Toys); ele-fashion did not. This is a frozen functional diagnostic, not retrained valid-operator performance.
   | Dataset | Target ΔCE | Matched-control ΔCE | Target−control ΔCE | Matched seeds |
   |---|---:|---:|---:|---:|
| Movies | +0.0043 | +0.0024 | +0.0019 | 3/3 |
| Toys | +0.0200 | +0.0104 | +0.0096 | 3/3 |
| Grocery | +0.0104 | +0.0038 | +0.0066 | 3/3 |
| ele-fashion | -0.0054 | -0.0042 | -0.0012 | 3/3 |
| Reddit-S | +0.0120 | -0.0007 | +0.0128 | 3/3 |
   Novelty-only validation utility prediction exceeded similarity-only in 3/5 datasets (Movies, Toys, Reddit-S); the average correlations remain small. Gate A also required a modality difference.
3. **What best describes low-similarity useful edges?**
   The fixed-mask low-similarity/high-novelty category is the strongest consistent candidate: it was positive against matched controls in four datasets. The separate embeddedness splits do not show a consistent cross-dataset ordering, so these data do not identify structural bridges or establish semantic complementarity. See `low_similarity_edge_explanation.csv` and `conditional_relation_utility.csv`.
4. **Is relation utility modality-dependent?**
   Text and Visual fixed-mask utility means differed for the same joint low-similarity/high-novelty edges in 5/5 datasets (Grocery, Movies, Reddit-S, Toys, ele-fashion); the stronger modality changed by dataset.
   | Dataset | U_text ΔCE | U_visual ΔCE | U_text−U_visual ΔCE |
   |---|---:|---:|---:|
| Movies | +0.0210 | +0.0229 | -0.0019 |
| Toys | +0.0299 | +0.0350 | -0.0050 |
| Grocery | +0.0133 | +0.0259 | -0.0126 |
| ele-fashion | +0.0001 | +0.0092 | -0.0091 |
| Reddit-S | +0.0654 | +0.0380 | +0.0275 |
5. **Can self-versus-structure utility be predicted from observable node signals?**
   Gate B: `MIXED`. The train-only logistic selector averaged -0.711 pp against the strongest Train-selected simple reference, with positive dataset means in 0/5. Oracle headroom averaged 5.284 pp, so a simple selector did not realize the available oracle gain.
6. **Can modality reliability be predicted?**
   Gate C: `MIXED`. The modality selector averaged -1.738 pp against its strongest simple reference, positive in 0/5. See `modality_selector.csv` and `modality_oracle.csv`.
   Across G_struct utility probes, mean validation Spearman was SIMILARITY_ONLY 0.006, NOVELTY_ONLY 0.036, TOPOLOGY_ONLY -0.044; mean R² values were negative. These are weak predictive signals, not evidence for a usable routing rule.
7. **Does the real topology outperform degree-preserving rewiring?**
   Yes on Validation in all five datasets; real−rewired paired accuracy differences are:
   | Dataset | Mean real−rewired | Population std |
   |---|---:|---:|
| Movies | +8.828 pp | 0.135 pp |
| Toys | +8.601 pp | 0.347 pp |
| Grocery | +8.072 pp | 0.527 pp |
| ele-fashion | +1.967 pp | 0.138 pp |
| Reddit-S | +3.733 pp | 0.015 pp |
8. **Which areas reached their preregistered gate?**
   Conditional relation utility: `STRONG_SUPPORT`; adaptive self/structure: `MIXED`; adaptive modality arbitration: `MIXED`; anchoring transfer: `OPERATOR_DEPENDENT`. The evidence does not automatically freeze a model design or paper claim.

## Boundaries

Gate A requires 3/5 datasets with positive low-similarity/high-novelty utility beyond exact controls under FIXED_NORM_MASK, novelty-only validation predictability above similarity-only in 3/5, and text/visual utility differences in 3/5. Gate B/C require at least +0.30 pp mean accuracy over the strongest non-learned simple reference selected by Train accuracy, positive gains in 3/5 datasets, and oracle headroom at least 0.30 pp larger than learned gain.
`FIXED_NORM_MASK` and `RENORMALIZED_DELETE` are frozen-checkpoint interventions; neither is a trained model comparison. No node-level p-values are used. Same-label edge values are descriptive only and are excluded from selectors, features, and group construction.

## Output map

Operator: `operator_transfer.csv`, `operator_transfer_paired.csv`, `operator_transfer_report.md`; interventions: `semantic_utility_deconfounded.csv`, `similarity_novelty_intervention.csv`, `low_similarity_edge_explanation.csv`, `conditional_relation_utility.csv`, `modality_specific_edge_utility.csv`; selectors: `self_structure_oracle.csv`, `self_structure_selector.csv`, `modality_oracle.csv`, `modality_selector.csv`, `utility_predictability.csv`; topology: `topology_reality_check.csv`, `topology_reality_check_summary.csv`.
Raw per-node deltas are gzip-compressed under `outputs/problem_deep_dive_v1/node_intervention_effects/`; their SHA-256 manifest is committed as `node_intervention_effects_manifest.csv`. No raw node table is committed.
