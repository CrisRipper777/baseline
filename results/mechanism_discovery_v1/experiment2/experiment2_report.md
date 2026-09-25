# Experiment 2: MAG interaction and problem discovery

All problem discovery uses train and validation labels only. Test labels and test results do not enter thresholds, features, probes, correlations, interventions or ranking.

Validation node tables: 15 files, 71532 rows; hashes are in `outputs/mechanism_discovery_v1/node_tables_manifest.csv`.

## Evidence hierarchy

- `DESCRIPTIVE_ONLY`: prevalence or representation statistics without task relevance.
- `TASK_RELEVANT`: consistent validation probe, loss, or prediction evidence.
- `INTERVENTION_SUPPORTED`: frozen targeted intervention effect beyond a matched same-size random control.
- Frozen edge removal is an intervention on an existing checkpoint, not retrained ablation.
- No topology rewiring training or paper model was run.

## Problem Discovery Matrix

| ID | Candidate problem | Status | Supported datasets |
|---|---|---|---|
| P1 | intrinsic semantic loss during propagation | B_PLAUSIBLE_CANDIDATE | ["Grocery", "Movies", "ele-fashion"] |
| P2 | task-relevant propagation innovation | B_PLAUSIBLE_CANDIDATE | ["Grocery", "Movies", "Reddit-S", "Toys", "ele-fashion"] |
| P3 | modality imbalance or branch suppression | B_PLAUSIBLE_CANDIDATE | ["Grocery", "Movies", "Reddit-S", "Toys", "ele-fashion"] |
| P4 | modality complementarity | B_PLAUSIBLE_CANDIDATE | ["Grocery", "Movies", "Reddit-S", "Toys", "ele-fashion"] |
| P5 | structure-attribute predictive conflict | B_PLAUSIBLE_CANDIDATE | ["Grocery", "Movies", "Reddit-S", "Toys", "ele-fashion"] |
| P6 | modality-specific topology utility | B_PLAUSIBLE_CANDIDATE | ["Grocery", "Movies", "Reddit-S", "Toys", "ele-fashion"] |
| P7 | neighbor relation heterogeneity | A_STRONG_CANDIDATE | ["Grocery", "Movies", "ele-fashion"] |
| P8 | semantic compatibility differs from task utility | A_STRONG_CANDIDATE | ["Grocery", "Reddit-S", "Toys"] |
| P9 | conditional need for cross-modal interaction | C_DESCRIPTIVE_ONLY | ["Grocery", "ele-fashion"] |
| P10 | task-relevant cross-modal divergence | B_PLAUSIBLE_CANDIDATE | ["Grocery", "Reddit-S", "Toys", "ele-fashion"] |
| P11 | task-relevant modality-private residual proxy | B_PLAUSIBLE_CANDIDATE | ["Grocery", "Movies", "Reddit-S", "Toys", "ele-fashion"] |
| P12 | structure x attribute x modality interaction | C_DESCRIPTIVE_ONLY | ["Grocery", "Movies", "Reddit-S", "Toys"] |

## Evidence files

- `structure_attribute_conflict.csv`: 75 records
- `modality_sufficiency.csv`: 15 records
- `modality_regimes.csv`: 60 records
- `joint_branch_suppression.csv`: 60 records
- `modality_structure_utility.csv`: 15 records
- `edge_interaction_prevalence.csv`: 120 records
- `edge_role_intervention.csv`: 60 records
- `semantic_utility_relation.csv`: 60 records
- `crossmodal_disagreement.csv`: 327 records
- `crossmodal_interaction_probe.csv`: 105 records
- `crossmodal_alignment_trajectory.csv`: 120 records
- `shared_private_proxy.csv`: 105 records
- `interaction_cube.csv`: 120 records
- `interaction_regression.csv`: 15 records
- `classwise_gain.csv`: 1602 records

Statuses summarize evidence hierarchy, not a research claim. Candidate ranking remains provisional and should be reviewed before choosing the next research problem.
