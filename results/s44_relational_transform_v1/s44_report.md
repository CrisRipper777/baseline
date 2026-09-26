# S4.4 Relational Transformation Pilot

Scope: full-graph node classification on Movies, Grocery, ele-fashion and Reddit-S; seeds 42–44; test evaluation disabled. No LP, Toys, H3, H4 or final-model claim is included.

- Training commit: `8e53b1b8f1eed4aac6a61082222327f4dca17da3`
- Analysis commit: `92cb223f102d43a9b32810891f11cacebf72afbc`
- Source branch/commit: `s43_h15_h1r_h2ab` / `7b98b31a78fb978fadd96448bbcfd3ffa295d82f`
- Formal runs: 120 / 120
- Selection: best validation accuracy; no test metrics were evaluated.

## Primary paired contrasts

Positive accuracy and macro-F1 deltas favor the right-hand variant; negative CE favors the right-hand variant.
| Contrast | Mean Δ val acc | Positive pairs | Mean Δ macro-F1 | Mean Δ CE | Label |
|---|---:|---:|---:|---:|---|
| scalar_rel_minus_raw | 0.00054 | 8/12 | -0.00065 | 0.01062 | PRIMARY_PAIRED |
| scalar_rel_minus_global | 0.00103 | 8/12 | 0.00047 | -0.00625 | PRIMARY_PAIRED |
| scalar_rel_multi_minus_rel | -0.00071 | 4/12 | 0.00077 | 0.00262 | PRIMARY_PAIRED |
| lowrank_rel_minus_global | 0.00059 | 4/12 | -0.00046 | -0.01756 | PRIMARY_PAIRED |
| lowrank_rel_multi_minus_rel | 0.00017 | 7/12 | 0.00262 | 0.12642 | PRIMARY_PAIRED |
| expert_rel_minus_uniform | 0.00090 | 6/12 | 0.00179 | 0.03703 | PRIMARY_PAIRED |
| expert_rel_multi_minus_rel | -0.00049 | 6/12 | -0.00500 | -0.01879 | PRIMARY_PAIRED |

## Interpretation statuses

- **RelationState: STRONG_SUPPORT** — scalar relation minus raw mean Δaccuracy=0.00054; positive paired seeds=8/12
- **EdgeConditionality: STRONG_SUPPORT** — across edge-conditioned variants, mean loss under target-wise edge shuffle=0.00039; affected variant-seed rows=49/96
- **DynamicTransformation: MECHANISM_SUPPORT** — low-rank relation minus global mean Δaccuracy=0.00059; ZERO_DYNAMIC=P0 branch within 1e-5 absolute tolerance=True
- **MultimodalRelationContext: MECHANISM_SUPPORT** — scalar multi-relation minus relation mean Δaccuracy=-0.00071; remove/swap interventions lower accuracy in 37/72 rows
- **ExpertRouting: MIXED** — routed minus uniform mean Δaccuracy=0.00090; routing shuffle harms 8/24 rows; mean edge fraction pi_max>0.90=0.434

## Functional audits

Frozen validation interventions measure sensitivity of these trained checkpoints. They do not establish retrained causal utility. Router weights are not causal effects. Expert-routing evidence requires routed-vs-uniform benefit, edge-shuffle sensitivity and no severe collapse together. Multimodal-context evidence requires multi-vs-single paired results and remove/swap interventions.

- Frozen intervention rows: 1476 (replicate-level edge shuffles and their mean/std are included).
- Multimodal intervention rows: 72.
- Expert diagnostic rows: 72.
- Low semantic correlation alone is not evidence of task utility; inspect the paired edge-shuffle results alongside the semantic audit.

## Complexity

COMPUTE_HEAVY marks peak process GPU use ≥20 GiB or a mean epoch proxy >2.5× historical P0 for that dataset. Epoch proxy includes setup and validation overhead.
| Dataset | Variant | Params | Peak job GPU MiB | Epoch proxy s | P0 ratio | Status |
|---|---|---:|---:|---:|---:|---|
| Movies | s44_scalar_global | 730646 | 1105 | 0.075 | 0.68 | WITHIN_REFERENCE |
| Movies | s44_scalar_raw | 773974 | 1985 | 0.159 | 1.44 | WITHIN_REFERENCE |
| Movies | s44_scalar_rel | 773974 | 2087 | 0.127 | 1.15 | WITHIN_REFERENCE |
| Movies | s44_scalar_rel_multi | 778326 | 2641 | 0.196 | 1.78 | WITHIN_REFERENCE |
| Movies | s44_lowrank_global | 738852 | 1773 | 0.108 | 0.98 | WITHIN_REFERENCE |
| Movies | s44_lowrank_rel | 782628 | 2125 | 0.132 | 1.20 | WITHIN_REFERENCE |
| Movies | s44_lowrank_rel_multi | 786980 | 2687 | 0.203 | 1.84 | WITHIN_REFERENCE |
| Movies | s44_expert_uniform | 747028 | 1847 | 0.122 | 1.11 | WITHIN_REFERENCE |
| Movies | s44_expert_rel | 790424 | 2783 | 0.160 | 1.45 | WITHIN_REFERENCE |
| Movies | s44_expert_rel_multi | 794776 | 3339 | 0.238 | 2.15 | WITHIN_REFERENCE |
| Grocery | s44_scalar_global | 730646 | 1115 | 0.075 | 0.70 | WITHIN_REFERENCE |
| Grocery | s44_scalar_raw | 773974 | 1899 | 0.145 | 1.36 | WITHIN_REFERENCE |
| Grocery | s44_scalar_rel | 773974 | 1997 | 0.119 | 1.11 | WITHIN_REFERENCE |
| Grocery | s44_scalar_rel_multi | 778326 | 2479 | 0.156 | 1.46 | WITHIN_REFERENCE |
| Grocery | s44_lowrank_global | 738852 | 1709 | 0.130 | 1.21 | WITHIN_REFERENCE |
| Grocery | s44_lowrank_rel | 782628 | 2035 | 0.154 | 1.44 | WITHIN_REFERENCE |
| Grocery | s44_lowrank_rel_multi | 786980 | 3622 | 0.165 | 1.54 | WITHIN_REFERENCE |
| Grocery | s44_expert_uniform | 747028 | 1771 | 0.143 | 1.34 | WITHIN_REFERENCE |
| Grocery | s44_expert_rel | 790424 | 2623 | 0.150 | 1.40 | WITHIN_REFERENCE |
| Grocery | s44_expert_rel_multi | 794776 | 3113 | 0.219 | 2.05 | WITHIN_REFERENCE |
| ele-fashion | s44_scalar_global | 597518 | 5088 | 0.271 | 1.05 | WITHIN_REFERENCE |
| ele-fashion | s44_scalar_raw | 640846 | 6127 | 0.491 | 1.90 | WITHIN_REFERENCE |
| ele-fashion | s44_scalar_rel | 640846 | 6712 | 0.411 | 1.59 | WITHIN_REFERENCE |
| ele-fashion | s44_scalar_rel_multi | 645198 | 7789 | 0.626 | 2.42 | WITHIN_REFERENCE |
| ele-fashion | s44_lowrank_global | 605724 | 6720 | 0.388 | 1.50 | WITHIN_REFERENCE |
| ele-fashion | s44_lowrank_rel | 649500 | 10810 | 0.475 | 1.84 | WITHIN_REFERENCE |
| ele-fashion | s44_lowrank_rel_multi | 653852 | 7969 | 0.661 | 2.56 | COMPUTE_HEAVY |
| ele-fashion | s44_expert_uniform | 613900 | 5741 | 0.436 | 1.68 | WITHIN_REFERENCE |
| ele-fashion | s44_expert_rel | 657296 | 8159 | 0.628 | 2.43 | WITHIN_REFERENCE |
| ele-fashion | s44_expert_rel_multi | 661648 | 9537 | 0.680 | 2.63 | COMPUTE_HEAVY |
| Reddit-S | s44_scalar_global | 730646 | 1045 | 0.101 | 1.31 | WITHIN_REFERENCE |
| Reddit-S | s44_scalar_raw | 773974 | 2579 | 0.182 | 2.35 | WITHIN_REFERENCE |
| Reddit-S | s44_scalar_rel | 773974 | 2781 | 0.163 | 2.10 | WITHIN_REFERENCE |
| Reddit-S | s44_scalar_rel_multi | 778326 | 3767 | 0.270 | 3.49 | COMPUTE_HEAVY |
| Reddit-S | s44_lowrank_global | 738852 | 2201 | 0.133 | 1.72 | WITHIN_REFERENCE |
| Reddit-S | s44_lowrank_rel | 782628 | 2859 | 0.177 | 2.29 | WITHIN_REFERENCE |
| Reddit-S | s44_lowrank_rel_multi | 786980 | 3837 | 0.279 | 3.62 | COMPUTE_HEAVY |
| Reddit-S | s44_expert_uniform | 747028 | 7393 | 0.164 | 2.13 | WITHIN_REFERENCE |
| Reddit-S | s44_expert_rel | 790424 | 3949 | 0.261 | 3.38 | COMPUTE_HEAVY |
| Reddit-S | s44_expert_rel_multi | 794776 | 4927 | 0.310 | 4.01 | COMPUTE_HEAVY |

## Interpretation boundary

These are validation-selected, frozen-checkpoint mechanism probes in the stated NC protocol. Frozen interventions are sensitivity analyses, not retrained counterfactuals. The model outputs do not imply a universal architecture winner or a final recommendation.
