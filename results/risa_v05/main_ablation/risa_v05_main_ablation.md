# RISA v0.5 main ablation validation analysis

- Protocol: unified_full_graph_nc_v1
- Test and LP evaluation: disabled.
- Val Acc / Macro-F1: read from validation-selected checkpoints.
- Val cross-entropy: post-hoc calculation on val_idx only; it does not select checkpoints.
- CRST p95/std/saturation: deterministic random edge sample; means and norm/context summaries stream over edges.
- IMCI attention summaries aggregate over nodes and heads.

## Movies

### v05_plain

| Metric | Mean ± population std |
|---|---:|
| val_acc | 0.545291 ± 0.00365715 |
| val_macro_f1 | 0.443157 ± 0.0145929 |
| val_cross_entropy | 1.4371 ± 0.0083077 |
| crst_text_mean_abs_theta | 0 ± 0 |
| crst_text_std_abs_theta | 0 ± 0 |
| crst_text_p95_abs_theta | 0 ± 0 |
| crst_text_mean_cos_base_rotated | 1 ± 0 |
| crst_text_norm_preservation_max_abs_error | 0 ± 0 |
| crst_text_valid_context_edge_ratio | 0 ± 0 |
| crst_text_angle_saturation_ratio | 0 ± 0 |
| crst_visual_mean_abs_theta | 0 ± 0 |
| crst_visual_std_abs_theta | 0 ± 0 |
| crst_visual_p95_abs_theta | 0 ± 0 |
| crst_visual_mean_cos_base_rotated | 1 ± 0 |
| crst_visual_norm_preservation_max_abs_error | 0 ± 0 |
| crst_visual_valid_context_edge_ratio | 0 ± 0 |
| crst_visual_angle_saturation_ratio | 0 ± 0 |

### v05_crst_only

| Metric | Mean ± population std |
|---|---:|
| val_acc | 0.545991 ± 0.00490616 |
| val_macro_f1 | 0.433448 ± 0.00937489 |
| val_cross_entropy | 1.43542 ± 0.0102449 |
| crst_text_mean_abs_theta | 0.45626 ± 0.00762067 |
| crst_text_std_abs_theta | 0.286919 ± 0.00591195 |
| crst_text_p95_abs_theta | 0.971232 ± 0.0324227 |
| crst_text_mean_cos_base_rotated | 0.829293 ± 0.0105424 |
| crst_text_norm_preservation_max_abs_error | 1.90735e-06 ± 0 |
| crst_text_valid_context_edge_ratio | 0.978825 ± 0 |
| crst_text_angle_saturation_ratio | 0 ± 0 |
| crst_visual_mean_abs_theta | 0.393927 ± 0.0505846 |
| crst_visual_std_abs_theta | 0.279505 ± 0.024595 |
| crst_visual_p95_abs_theta | 0.938938 ± 0.0851515 |
| crst_visual_mean_cos_base_rotated | 0.867257 ± 0.0275978 |
| crst_visual_norm_preservation_max_abs_error | 1.90735e-06 ± 0 |
| crst_visual_valid_context_edge_ratio | 0.978825 ± 0 |
| crst_visual_angle_saturation_ratio | 0 ± 0 |

### v05_absorb_only

| Metric | Mean ± population std |
|---|---:|
| val_acc | 0.565587 ± 0.00306207 |
| val_macro_f1 | 0.472709 ± 0.00752314 |
| val_cross_entropy | 1.62153 ± 0.0647548 |
| crst_text_mean_abs_theta | 0 ± 0 |
| crst_text_std_abs_theta | 0 ± 0 |
| crst_text_p95_abs_theta | 0 ± 0 |
| crst_text_mean_cos_base_rotated | 1 ± 0 |
| crst_text_norm_preservation_max_abs_error | 0 ± 0 |
| crst_text_valid_context_edge_ratio | 0 ± 0 |
| crst_text_angle_saturation_ratio | 0 ± 0 |
| imci_text_mean_hop_attention_entropy | 1.05334 ± 0.00600792 |
| imci_text_mean_hop_attention_1 | 0.311443 ± 0.0322968 |
| imci_text_mean_hop_attention_2 | 0.321507 ± 0.010896 |
| imci_text_mean_hop_attention_3 | 0.36705 ± 0.0359626 |
| imci_text_per_hop_attention_std_across_nodes_heads_1 | 0.117058 ± 0.00710115 |
| imci_text_per_hop_attention_std_across_nodes_heads_2 | 0.0520253 ± 0.00431308 |
| imci_text_per_hop_attention_std_across_nodes_heads_3 | 0.0931965 ± 0.00272942 |
| crst_visual_mean_abs_theta | 0 ± 0 |
| crst_visual_std_abs_theta | 0 ± 0 |
| crst_visual_p95_abs_theta | 0 ± 0 |
| crst_visual_mean_cos_base_rotated | 1 ± 0 |
| crst_visual_norm_preservation_max_abs_error | 0 ± 0 |
| crst_visual_valid_context_edge_ratio | 0 ± 0 |
| crst_visual_angle_saturation_ratio | 0 ± 0 |
| imci_visual_mean_hop_attention_entropy | 0.992289 ± 0.00367976 |
| imci_visual_mean_hop_attention_1 | 0.432029 ± 0.0149402 |
| imci_visual_mean_hop_attention_2 | 0.31678 ± 0.0055723 |
| imci_visual_mean_hop_attention_3 | 0.251192 ± 0.0177277 |
| imci_visual_per_hop_attention_std_across_nodes_heads_1 | 0.177904 ± 0.0109601 |
| imci_visual_per_hop_attention_std_across_nodes_heads_2 | 0.0925264 ± 0.00509682 |
| imci_visual_per_hop_attention_std_across_nodes_heads_3 | 0.116557 ± 0.0117879 |

### v05_full

| Metric | Mean ± population std |
|---|---:|
| val_acc | 0.561988 ± 0.00275264 |
| val_macro_f1 | 0.461045 ± 0.0098049 |
| val_cross_entropy | 1.45425 ± 0.00880131 |
| crst_text_mean_abs_theta | 0.298367 ± 0.0478941 |
| crst_text_std_abs_theta | 0.217793 ± 0.0323081 |
| crst_text_p95_abs_theta | 0.71911 ± 0.113851 |
| crst_text_mean_cos_base_rotated | 0.929035 ± 0.0224111 |
| crst_text_norm_preservation_max_abs_error | 1.90735e-06 ± 0 |
| crst_text_valid_context_edge_ratio | 0.978825 ± 0 |
| crst_text_angle_saturation_ratio | 0 ± 0 |
| imci_text_mean_hop_attention_entropy | 1.07036 ± 0.0126146 |
| imci_text_mean_hop_attention_1 | 0.310645 ± 0.00346707 |
| imci_text_mean_hop_attention_2 | 0.319749 ± 0.00635802 |
| imci_text_mean_hop_attention_3 | 0.369606 ± 0.00799819 |
| imci_text_per_hop_attention_std_across_nodes_heads_1 | 0.089809 ± 0.0213319 |
| imci_text_per_hop_attention_std_across_nodes_heads_2 | 0.0427969 ± 0.015139 |
| imci_text_per_hop_attention_std_across_nodes_heads_3 | 0.0746064 ± 0.0181594 |
| crst_visual_mean_abs_theta | 0.169766 ± 0.0134569 |
| crst_visual_std_abs_theta | 0.136549 ± 0.0115716 |
| crst_visual_p95_abs_theta | 0.442619 ± 0.0373825 |
| crst_visual_mean_cos_base_rotated | 0.976433 ± 0.00339693 |
| crst_visual_norm_preservation_max_abs_error | 1.90735e-06 ± 0 |
| crst_visual_valid_context_edge_ratio | 0.978825 ± 0 |
| crst_visual_angle_saturation_ratio | 0 ± 0 |
| imci_visual_mean_hop_attention_entropy | 1.01679 ± 0.0113891 |
| imci_visual_mean_hop_attention_1 | 0.352583 ± 0.0218212 |
| imci_visual_mean_hop_attention_2 | 0.325829 ± 0.00977779 |
| imci_visual_mean_hop_attention_3 | 0.321588 ± 0.0126775 |
| imci_visual_per_hop_attention_std_across_nodes_heads_1 | 0.169883 ± 0.00997043 |
| imci_visual_per_hop_attention_std_across_nodes_heads_2 | 0.0851318 ± 0.00869629 |
| imci_visual_per_hop_attention_std_across_nodes_heads_3 | 0.126591 ± 0.00922922 |

## Toys

### v05_plain

| Metric | Mean ± population std |
|---|---:|
| val_acc | 0.798985 ± 0.00123194 |
| val_macro_f1 | 0.770044 ± 0.00189571 |
| val_cross_entropy | 0.759602 ± 0.0040842 |
| crst_text_mean_abs_theta | 0 ± 0 |
| crst_text_std_abs_theta | 0 ± 0 |
| crst_text_p95_abs_theta | 0 ± 0 |
| crst_text_mean_cos_base_rotated | 1 ± 0 |
| crst_text_norm_preservation_max_abs_error | 0 ± 0 |
| crst_text_valid_context_edge_ratio | 0 ± 0 |
| crst_text_angle_saturation_ratio | 0 ± 0 |
| crst_visual_mean_abs_theta | 0 ± 0 |
| crst_visual_std_abs_theta | 0 ± 0 |
| crst_visual_p95_abs_theta | 0 ± 0 |
| crst_visual_mean_cos_base_rotated | 1 ± 0 |
| crst_visual_norm_preservation_max_abs_error | 0 ± 0 |
| crst_visual_valid_context_edge_ratio | 0 ± 0 |
| crst_visual_angle_saturation_ratio | 0 ± 0 |

### v05_crst_only

| Metric | Mean ± population std |
|---|---:|
| val_acc | 0.799227 ± 0.00104384 |
| val_macro_f1 | 0.770718 ± 0.00257427 |
| val_cross_entropy | 0.759763 ± 0.00513584 |
| crst_text_mean_abs_theta | 0.347468 ± 0.0140058 |
| crst_text_std_abs_theta | 0.258248 ± 0.0105734 |
| crst_text_p95_abs_theta | 0.856268 ± 0.0358048 |
| crst_text_mean_cos_base_rotated | 0.8995 ± 0.00340855 |
| crst_text_norm_preservation_max_abs_error | 1.90735e-06 ± 0 |
| crst_text_valid_context_edge_ratio | 0.949631 ± 0 |
| crst_text_angle_saturation_ratio | 0 ± 0 |
| crst_visual_mean_abs_theta | 0.356786 ± 0.0222198 |
| crst_visual_std_abs_theta | 0.255332 ± 0.00222763 |
| crst_visual_p95_abs_theta | 0.837624 ± 0.00976434 |
| crst_visual_mean_cos_base_rotated | 0.892712 ± 0.00682052 |
| crst_visual_norm_preservation_max_abs_error | 1.90735e-06 ± 0 |
| crst_visual_valid_context_edge_ratio | 0.949631 ± 0 |
| crst_visual_angle_saturation_ratio | 0 ± 0 |

### v05_absorb_only

| Metric | Mean ± population std |
|---|---:|
| val_acc | 0.799871 ± 0.00186451 |
| val_macro_f1 | 0.765303 ± 0.00282671 |
| val_cross_entropy | 0.771571 ± 0.0133657 |
| crst_text_mean_abs_theta | 0 ± 0 |
| crst_text_std_abs_theta | 0 ± 0 |
| crst_text_p95_abs_theta | 0 ± 0 |
| crst_text_mean_cos_base_rotated | 1 ± 0 |
| crst_text_norm_preservation_max_abs_error | 0 ± 0 |
| crst_text_valid_context_edge_ratio | 0 ± 0 |
| crst_text_angle_saturation_ratio | 0 ± 0 |
| imci_text_mean_hop_attention_entropy | 1.01412 ± 0.0090548 |
| imci_text_mean_hop_attention_1 | 0.279691 ± 0.0346052 |
| imci_text_mean_hop_attention_2 | 0.329221 ± 0.00323281 |
| imci_text_mean_hop_attention_3 | 0.391088 ± 0.031379 |
| imci_text_per_hop_attention_std_across_nodes_heads_1 | 0.149806 ± 0.014188 |
| imci_text_per_hop_attention_std_across_nodes_heads_2 | 0.0889938 ± 0.00415972 |
| imci_text_per_hop_attention_std_across_nodes_heads_3 | 0.121796 ± 0.0062328 |
| crst_visual_mean_abs_theta | 0 ± 0 |
| crst_visual_std_abs_theta | 0 ± 0 |
| crst_visual_p95_abs_theta | 0 ± 0 |
| crst_visual_mean_cos_base_rotated | 1 ± 0 |
| crst_visual_norm_preservation_max_abs_error | 0 ± 0 |
| crst_visual_valid_context_edge_ratio | 0 ± 0 |
| crst_visual_angle_saturation_ratio | 0 ± 0 |
| imci_visual_mean_hop_attention_entropy | 0.915784 ± 0.021928 |
| imci_visual_mean_hop_attention_1 | 0.143339 ± 0.00921229 |
| imci_visual_mean_hop_attention_2 | 0.399351 ± 0.00241013 |
| imci_visual_mean_hop_attention_3 | 0.457309 ± 0.0110142 |
| imci_visual_per_hop_attention_std_across_nodes_heads_1 | 0.132444 ± 0.00740155 |
| imci_visual_per_hop_attention_std_across_nodes_heads_2 | 0.118152 ± 0.00951413 |
| imci_visual_per_hop_attention_std_across_nodes_heads_3 | 0.128162 ± 0.0102853 |

### v05_full

| Metric | Mean ± population std |
|---|---:|
| val_acc | 0.798905 ± 0.00228638 |
| val_macro_f1 | 0.765196 ± 0.00209991 |
| val_cross_entropy | 0.776181 ± 0.025965 |
| crst_text_mean_abs_theta | 0.264434 ± 0.0325934 |
| crst_text_std_abs_theta | 0.197075 ± 0.0195895 |
| crst_text_p95_abs_theta | 0.657395 ± 0.0682687 |
| crst_text_mean_cos_base_rotated | 0.944334 ± 0.0100162 |
| crst_text_norm_preservation_max_abs_error | 1.90735e-06 ± 0 |
| crst_text_valid_context_edge_ratio | 0.949631 ± 0 |
| crst_text_angle_saturation_ratio | 0 ± 0 |
| imci_text_mean_hop_attention_entropy | 0.995145 ± 0.0276668 |
| imci_text_mean_hop_attention_1 | 0.282719 ± 0.0551951 |
| imci_text_mean_hop_attention_2 | 0.321604 ± 0.0175023 |
| imci_text_mean_hop_attention_3 | 0.395678 ± 0.0413005 |
| imci_text_per_hop_attention_std_across_nodes_heads_1 | 0.160643 ± 0.0422986 |
| imci_text_per_hop_attention_std_across_nodes_heads_2 | 0.0853813 ± 0.0180731 |
| imci_text_per_hop_attention_std_across_nodes_heads_3 | 0.13444 ± 0.0298622 |
| crst_visual_mean_abs_theta | 0.206016 ± 0.0166094 |
| crst_visual_std_abs_theta | 0.160327 ± 0.0222269 |
| crst_visual_p95_abs_theta | 0.523839 ± 0.0687394 |
| crst_visual_mean_cos_base_rotated | 0.964688 ± 0.00747014 |
| crst_visual_norm_preservation_max_abs_error | 1.90735e-06 ± 0 |
| crst_visual_valid_context_edge_ratio | 0.949631 ± 0 |
| crst_visual_angle_saturation_ratio | 0 ± 0 |
| imci_visual_mean_hop_attention_entropy | 0.924412 ± 0.0274731 |
| imci_visual_mean_hop_attention_1 | 0.182768 ± 0.0340184 |
| imci_visual_mean_hop_attention_2 | 0.364133 ± 0.0294531 |
| imci_visual_mean_hop_attention_3 | 0.453099 ± 0.0598453 |
| imci_visual_per_hop_attention_std_across_nodes_heads_1 | 0.159355 ± 0.022799 |
| imci_visual_per_hop_attention_std_across_nodes_heads_2 | 0.112984 ± 0.00799799 |
| imci_visual_per_hop_attention_std_across_nodes_heads_3 | 0.145843 ± 0.00695418 |

## Grocery

### v05_plain

| Metric | Mean ± population std |
|---|---:|
| val_acc | 0.798536 ± 0.00442803 |
| val_macro_f1 | 0.699768 ± 0.0165154 |
| val_cross_entropy | 0.804069 ± 0.00757359 |
| crst_text_mean_abs_theta | 0 ± 0 |
| crst_text_std_abs_theta | 0 ± 0 |
| crst_text_p95_abs_theta | 0 ± 0 |
| crst_text_mean_cos_base_rotated | 1 ± 0 |
| crst_text_norm_preservation_max_abs_error | 0 ± 0 |
| crst_text_valid_context_edge_ratio | 0 ± 0 |
| crst_text_angle_saturation_ratio | 0 ± 0 |
| crst_visual_mean_abs_theta | 0 ± 0 |
| crst_visual_std_abs_theta | 0 ± 0 |
| crst_visual_p95_abs_theta | 0 ± 0 |
| crst_visual_mean_cos_base_rotated | 1 ± 0 |
| crst_visual_norm_preservation_max_abs_error | 0 ± 0 |
| crst_visual_valid_context_edge_ratio | 0 ± 0 |
| crst_visual_angle_saturation_ratio | 0 ± 0 |

### v05_crst_only

| Metric | Mean ± population std |
|---|---:|
| val_acc | 0.801367 ± 0.00131682 |
| val_macro_f1 | 0.710128 ± 0.0105326 |
| val_cross_entropy | 0.815029 ± 0.00604108 |
| crst_text_mean_abs_theta | 0.621057 ± 0.0271735 |
| crst_text_std_abs_theta | 0.390182 ± 0.0218655 |
| crst_text_p95_abs_theta | 1.26983 ± 0.0583177 |
| crst_text_mean_cos_base_rotated | 0.710116 ± 0.026355 |
| crst_text_norm_preservation_max_abs_error | 1.90735e-06 ± 0 |
| crst_text_valid_context_edge_ratio | 0.970027 ± 0 |
| crst_text_angle_saturation_ratio | 0.00793183 ± 0.00647134 |
| crst_visual_mean_abs_theta | 0.608552 ± 0.032188 |
| crst_visual_std_abs_theta | 0.359668 ± 0.00697555 |
| crst_visual_p95_abs_theta | 1.20537 ± 0.0259378 |
| crst_visual_mean_cos_base_rotated | 0.745877 ± 0.01759 |
| crst_visual_norm_preservation_max_abs_error | 1.90735e-06 ± 0 |
| crst_visual_valid_context_edge_ratio | 0.970027 ± 0 |
| crst_visual_angle_saturation_ratio | 0.00122148 ± 0.000330515 |

### v05_absorb_only

| Metric | Mean ± population std |
|---|---:|
| val_acc | 0.828502 ± 0.000768574 |
| val_macro_f1 | 0.763441 ± 0.00351272 |
| val_cross_entropy | 0.900047 ± 0.0850852 |
| crst_text_mean_abs_theta | 0 ± 0 |
| crst_text_std_abs_theta | 0 ± 0 |
| crst_text_p95_abs_theta | 0 ± 0 |
| crst_text_mean_cos_base_rotated | 1 ± 0 |
| crst_text_norm_preservation_max_abs_error | 0 ± 0 |
| crst_text_valid_context_edge_ratio | 0 ± 0 |
| crst_text_angle_saturation_ratio | 0 ± 0 |
| imci_text_mean_hop_attention_entropy | 0.70537 ± 0.0368669 |
| imci_text_mean_hop_attention_1 | 0.486875 ± 0.00761907 |
| imci_text_mean_hop_attention_2 | 0.201498 ± 0.01747 |
| imci_text_mean_hop_attention_3 | 0.311628 ± 0.020503 |
| imci_text_per_hop_attention_std_across_nodes_heads_1 | 0.33283 ± 0.0123315 |
| imci_text_per_hop_attention_std_across_nodes_heads_2 | 0.177356 ± 0.00882272 |
| imci_text_per_hop_attention_std_across_nodes_heads_3 | 0.265958 ± 0.0174278 |
| crst_visual_mean_abs_theta | 0 ± 0 |
| crst_visual_std_abs_theta | 0 ± 0 |
| crst_visual_p95_abs_theta | 0 ± 0 |
| crst_visual_mean_cos_base_rotated | 1 ± 0 |
| crst_visual_norm_preservation_max_abs_error | 0 ± 0 |
| crst_visual_valid_context_edge_ratio | 0 ± 0 |
| crst_visual_angle_saturation_ratio | 0 ± 0 |
| imci_visual_mean_hop_attention_entropy | 0.945847 ± 0.0179114 |
| imci_visual_mean_hop_attention_1 | 0.297485 ± 0.0174316 |
| imci_visual_mean_hop_attention_2 | 0.337256 ± 0.00727385 |
| imci_visual_mean_hop_attention_3 | 0.365259 ± 0.0105487 |
| imci_visual_per_hop_attention_std_across_nodes_heads_1 | 0.21728 ± 0.0180604 |
| imci_visual_per_hop_attention_std_across_nodes_heads_2 | 0.125837 ± 0.0123212 |
| imci_visual_per_hop_attention_std_across_nodes_heads_3 | 0.172974 ± 0.0107292 |

### v05_full

| Metric | Mean ± population std |
|---|---:|
| val_acc | 0.830161 ± 0.00204281 |
| val_macro_f1 | 0.764728 ± 0.0106594 |
| val_cross_entropy | 0.998384 ± 0.285739 |
| crst_text_mean_abs_theta | 0.309889 ± 0.101155 |
| crst_text_std_abs_theta | 0.226818 ± 0.0570608 |
| crst_text_p95_abs_theta | 0.747453 ± 0.191972 |
| crst_text_mean_cos_base_rotated | 0.920451 ± 0.0460454 |
| crst_text_norm_preservation_max_abs_error | 1.90735e-06 ± 0 |
| crst_text_valid_context_edge_ratio | 0.970027 ± 0 |
| crst_text_angle_saturation_ratio | 1.82139e-05 ± 2.57583e-05 |
| imci_text_mean_hop_attention_entropy | 0.607219 ± 0.177979 |
| imci_text_mean_hop_attention_1 | 0.569886 ± 0.102314 |
| imci_text_mean_hop_attention_2 | 0.156783 ± 0.0826046 |
| imci_text_mean_hop_attention_3 | 0.273331 ± 0.0201562 |
| imci_text_per_hop_attention_std_across_nodes_heads_1 | 0.330962 ± 0.0153095 |
| imci_text_per_hop_attention_std_across_nodes_heads_2 | 0.146601 ± 0.0258173 |
| imci_text_per_hop_attention_std_across_nodes_heads_3 | 0.273114 ± 0.0371975 |
| crst_visual_mean_abs_theta | 0.25647 ± 0.0646696 |
| crst_visual_std_abs_theta | 0.195468 ± 0.0387847 |
| crst_visual_p95_abs_theta | 0.641752 ± 0.131124 |
| crst_visual_mean_cos_base_rotated | 0.946471 ± 0.0237437 |
| crst_visual_norm_preservation_max_abs_error | 1.90735e-06 ± 0 |
| crst_visual_valid_context_edge_ratio | 0.970027 ± 0 |
| crst_visual_angle_saturation_ratio | 1.57427e-06 ± 2.22635e-06 |
| imci_visual_mean_hop_attention_entropy | 0.857181 ± 0.0912742 |
| imci_visual_mean_hop_attention_1 | 0.350103 ± 0.127209 |
| imci_visual_mean_hop_attention_2 | 0.334002 ± 0.0641036 |
| imci_visual_mean_hop_attention_3 | 0.315895 ± 0.0632259 |
| imci_visual_per_hop_attention_std_across_nodes_heads_1 | 0.256787 ± 0.046698 |
| imci_visual_per_hop_attention_std_across_nodes_heads_2 | 0.147606 ± 0.0188802 |
| imci_visual_per_hop_attention_std_across_nodes_heads_3 | 0.190417 ± 0.0217894 |

## ele-fashion

### v05_plain

| Metric | Mean ± population std |
|---|---:|
| val_acc | 0.832941 ± 0.00160492 |
| val_macro_f1 | 0.684909 ± 0.00772701 |
| val_cross_entropy | 0.532906 ± 0.00345155 |
| crst_text_mean_abs_theta | 0 ± 0 |
| crst_text_std_abs_theta | 0 ± 0 |
| crst_text_p95_abs_theta | 0 ± 0 |
| crst_text_mean_cos_base_rotated | 1 ± 0 |
| crst_text_norm_preservation_max_abs_error | 0 ± 0 |
| crst_text_valid_context_edge_ratio | 0 ± 0 |
| crst_text_angle_saturation_ratio | 0 ± 0 |
| crst_visual_mean_abs_theta | 0 ± 0 |
| crst_visual_std_abs_theta | 0 ± 0 |
| crst_visual_p95_abs_theta | 0 ± 0 |
| crst_visual_mean_cos_base_rotated | 1 ± 0 |
| crst_visual_norm_preservation_max_abs_error | 0 ± 0 |
| crst_visual_valid_context_edge_ratio | 0 ± 0 |
| crst_visual_angle_saturation_ratio | 0 ± 0 |

### v05_crst_only

| Metric | Mean ± population std |
|---|---:|
| val_acc | 0.852238 ± 0.00108994 |
| val_macro_f1 | 0.704703 ± 0.00165904 |
| val_cross_entropy | 0.475278 ± 0.00527768 |
| crst_text_mean_abs_theta | 0.924531 ± 0.05638 |
| crst_text_std_abs_theta | 0.441611 ± 0.00755936 |
| crst_text_p95_abs_theta | 1.48207 ± 0.0227479 |
| crst_text_mean_cos_base_rotated | 0.47737 ± 0.0547984 |
| crst_text_norm_preservation_max_abs_error | 1.90735e-06 ± 0 |
| crst_text_valid_context_edge_ratio | 0.886199 ± 0 |
| crst_text_angle_saturation_ratio | 0.143831 ± 0.0480431 |
| crst_visual_mean_abs_theta | 0.721619 ± 0.0678749 |
| crst_visual_std_abs_theta | 0.42867 ± 0.0203034 |
| crst_visual_p95_abs_theta | 1.3792 ± 0.0489025 |
| crst_visual_mean_cos_base_rotated | 0.578129 ± 0.0536126 |
| crst_visual_norm_preservation_max_abs_error | 1.90735e-06 ± 0 |
| crst_visual_valid_context_edge_ratio | 0.886199 ± 0 |
| crst_visual_angle_saturation_ratio | 0.0394963 ± 0.0240165 |

### v05_absorb_only

| Metric | Mean ± population std |
|---|---:|
| val_acc | 0.880331 ± 0.00129646 |
| val_macro_f1 | 0.758026 ± 0.00419486 |
| val_cross_entropy | 0.43058 ± 0.0180028 |
| crst_text_mean_abs_theta | 0 ± 0 |
| crst_text_std_abs_theta | 0 ± 0 |
| crst_text_p95_abs_theta | 0 ± 0 |
| crst_text_mean_cos_base_rotated | 1 ± 0 |
| crst_text_norm_preservation_max_abs_error | 0 ± 0 |
| crst_text_valid_context_edge_ratio | 0 ± 0 |
| crst_text_angle_saturation_ratio | 0 ± 0 |
| imci_text_mean_hop_attention_entropy | 0.940904 ± 0.0232172 |
| imci_text_mean_hop_attention_1 | 0.296412 ± 0.03143 |
| imci_text_mean_hop_attention_2 | 0.335836 ± 0.0155721 |
| imci_text_mean_hop_attention_3 | 0.367752 ± 0.0172488 |
| imci_text_per_hop_attention_std_across_nodes_heads_1 | 0.205579 ± 0.0154599 |
| imci_text_per_hop_attention_std_across_nodes_heads_2 | 0.174993 ± 0.0116119 |
| imci_text_per_hop_attention_std_across_nodes_heads_3 | 0.149191 ± 0.0150319 |
| crst_visual_mean_abs_theta | 0 ± 0 |
| crst_visual_std_abs_theta | 0 ± 0 |
| crst_visual_p95_abs_theta | 0 ± 0 |
| crst_visual_mean_cos_base_rotated | 1 ± 0 |
| crst_visual_norm_preservation_max_abs_error | 0 ± 0 |
| crst_visual_valid_context_edge_ratio | 0 ± 0 |
| crst_visual_angle_saturation_ratio | 0 ± 0 |
| imci_visual_mean_hop_attention_entropy | 0.860723 ± 0.0103259 |
| imci_visual_mean_hop_attention_1 | 0.372244 ± 0.055786 |
| imci_visual_mean_hop_attention_2 | 0.253348 ± 0.0168454 |
| imci_visual_mean_hop_attention_3 | 0.374408 ± 0.0389802 |
| imci_visual_per_hop_attention_std_across_nodes_heads_1 | 0.260909 ± 0.00326708 |
| imci_visual_per_hop_attention_std_across_nodes_heads_2 | 0.178296 ± 0.0013133 |
| imci_visual_per_hop_attention_std_across_nodes_heads_3 | 0.195561 ± 0.00532693 |

### v05_full

| Metric | Mean ± population std |
|---|---:|
| val_acc | 0.879274 ± 0.00138322 |
| val_macro_f1 | 0.762828 ± 0.00222096 |
| val_cross_entropy | 0.44944 ± 0.0302646 |
| crst_text_mean_abs_theta | 0.376522 ± 0.0135025 |
| crst_text_std_abs_theta | 0.268452 ± 0.0129041 |
| crst_text_p95_abs_theta | 0.893334 ± 0.0417066 |
| crst_text_mean_cos_base_rotated | 0.895637 ± 0.00781285 |
| crst_text_norm_preservation_max_abs_error | 1.90735e-06 ± 0 |
| crst_text_valid_context_edge_ratio | 0.886199 ± 0 |
| crst_text_angle_saturation_ratio | 0 ± 0 |
| imci_text_mean_hop_attention_entropy | 0.924274 ± 0.0101927 |
| imci_text_mean_hop_attention_1 | 0.387018 ± 0.026935 |
| imci_text_mean_hop_attention_2 | 0.336522 ± 0.0212637 |
| imci_text_mean_hop_attention_3 | 0.27646 ± 0.0249532 |
| imci_text_per_hop_attention_std_across_nodes_heads_1 | 0.234939 ± 0.00554699 |
| imci_text_per_hop_attention_std_across_nodes_heads_2 | 0.18512 ± 0.00205084 |
| imci_text_per_hop_attention_std_across_nodes_heads_3 | 0.133471 ± 0.00496368 |
| crst_visual_mean_abs_theta | 0.425895 ± 0.0553476 |
| crst_visual_std_abs_theta | 0.297168 ± 0.0234323 |
| crst_visual_p95_abs_theta | 0.981404 ± 0.0917629 |
| crst_visual_mean_cos_base_rotated | 0.863077 ± 0.0299021 |
| crst_visual_norm_preservation_max_abs_error | 1.90735e-06 ± 0 |
| crst_visual_valid_context_edge_ratio | 0.886199 ± 0 |
| crst_visual_angle_saturation_ratio | 6.09595e-05 ± 4.64376e-05 |
| imci_visual_mean_hop_attention_entropy | 0.876041 ± 0.0352239 |
| imci_visual_mean_hop_attention_1 | 0.424302 ± 0.0650008 |
| imci_visual_mean_hop_attention_2 | 0.284052 ± 0.00975116 |
| imci_visual_mean_hop_attention_3 | 0.291646 ± 0.0555899 |
| imci_visual_per_hop_attention_std_across_nodes_heads_1 | 0.251389 ± 0.0181297 |
| imci_visual_per_hop_attention_std_across_nodes_heads_2 | 0.187325 ± 0.0185088 |
| imci_visual_per_hop_attention_std_across_nodes_heads_3 | 0.166113 ± 0.0109032 |

## Reddit-S

### v05_plain

| Metric | Mean ± population std |
|---|---:|
| val_acc | 0.944637 ± 0.00025684 |
| val_macro_f1 | 0.902862 ± 0.00124308 |
| val_cross_entropy | 0.228557 ± 0.00225393 |
| crst_text_mean_abs_theta | 0 ± 0 |
| crst_text_std_abs_theta | 0 ± 0 |
| crst_text_p95_abs_theta | 0 ± 0 |
| crst_text_mean_cos_base_rotated | 1 ± 0 |
| crst_text_norm_preservation_max_abs_error | 0 ± 0 |
| crst_text_valid_context_edge_ratio | 0 ± 0 |
| crst_text_angle_saturation_ratio | 0 ± 0 |
| crst_visual_mean_abs_theta | 0 ± 0 |
| crst_visual_std_abs_theta | 0 ± 0 |
| crst_visual_p95_abs_theta | 0 ± 0 |
| crst_visual_mean_cos_base_rotated | 1 ± 0 |
| crst_visual_norm_preservation_max_abs_error | 0 ± 0 |
| crst_visual_valid_context_edge_ratio | 0 ± 0 |
| crst_visual_angle_saturation_ratio | 0 ± 0 |

### v05_crst_only

| Metric | Mean ± population std |
|---|---:|
| val_acc | 0.943378 ± 0.000679526 |
| val_macro_f1 | 0.902649 ± 0.00088555 |
| val_cross_entropy | 0.251411 ± 0.0304006 |
| crst_text_mean_abs_theta | 0.249505 ± 0.0339449 |
| crst_text_std_abs_theta | 0.185457 ± 0.0244076 |
| crst_text_p95_abs_theta | 0.611917 ± 0.0855201 |
| crst_text_mean_cos_base_rotated | 0.944497 ± 0.0154928 |
| crst_text_norm_preservation_max_abs_error | 1.90735e-06 ± 0 |
| crst_text_valid_context_edge_ratio | 0.983114 ± 0 |
| crst_text_angle_saturation_ratio | 0 ± 0 |
| crst_visual_mean_abs_theta | 0.304174 ± 0.0528973 |
| crst_visual_std_abs_theta | 0.215617 ± 0.0402941 |
| crst_visual_p95_abs_theta | 0.69944 ± 0.125288 |
| crst_visual_mean_cos_base_rotated | 0.915046 ± 0.0295853 |
| crst_visual_norm_preservation_max_abs_error | 1.90735e-06 ± 0 |
| crst_visual_valid_context_edge_ratio | 0.983114 ± 0 |
| crst_visual_angle_saturation_ratio | 0 ± 0 |

### v05_absorb_only

| Metric | Mean ± population std |
|---|---:|
| val_acc | 0.962672 ± 0.000784659 |
| val_macro_f1 | 0.930758 ± 0.00238254 |
| val_cross_entropy | 0.217579 ± 0.00569072 |
| crst_text_mean_abs_theta | 0 ± 0 |
| crst_text_std_abs_theta | 0 ± 0 |
| crst_text_p95_abs_theta | 0 ± 0 |
| crst_text_mean_cos_base_rotated | 1 ± 0 |
| crst_text_norm_preservation_max_abs_error | 0 ± 0 |
| crst_text_valid_context_edge_ratio | 0 ± 0 |
| crst_text_angle_saturation_ratio | 0 ± 0 |
| imci_text_mean_hop_attention_entropy | 1.09575 ± 0.00166022 |
| imci_text_mean_hop_attention_1 | 0.343884 ± 0.0113646 |
| imci_text_mean_hop_attention_2 | 0.328337 ± 0.000364443 |
| imci_text_mean_hop_attention_3 | 0.327779 ± 0.0117223 |
| imci_text_per_hop_attention_std_across_nodes_heads_1 | 0.0260226 ± 0.00826376 |
| imci_text_per_hop_attention_std_across_nodes_heads_2 | 0.0230613 ± 0.00425277 |
| imci_text_per_hop_attention_std_across_nodes_heads_3 | 0.0140174 ± 0.0022657 |
| crst_visual_mean_abs_theta | 0 ± 0 |
| crst_visual_std_abs_theta | 0 ± 0 |
| crst_visual_p95_abs_theta | 0 ± 0 |
| crst_visual_mean_cos_base_rotated | 1 ± 0 |
| crst_visual_norm_preservation_max_abs_error | 0 ± 0 |
| crst_visual_valid_context_edge_ratio | 0 ± 0 |
| crst_visual_angle_saturation_ratio | 0 ± 0 |
| imci_visual_mean_hop_attention_entropy | 1.09736 ± 0.000687353 |
| imci_visual_mean_hop_attention_1 | 0.331392 ± 0.00402272 |
| imci_visual_mean_hop_attention_2 | 0.333222 ± 0.00547693 |
| imci_visual_mean_hop_attention_3 | 0.335386 ± 0.00152608 |
| imci_visual_per_hop_attention_std_across_nodes_heads_1 | 0.0173236 ± 0.0057635 |
| imci_visual_per_hop_attention_std_across_nodes_heads_2 | 0.0139983 ± 0.0058535 |
| imci_visual_per_hop_attention_std_across_nodes_heads_3 | 0.0140501 ± 0.00296193 |

### v05_full

| Metric | Mean ± population std |
|---|---:|
| val_acc | 0.962986 ± 0.00171011 |
| val_macro_f1 | 0.926414 ± 0.00319394 |
| val_cross_entropy | 0.208661 ± 0.0157217 |
| crst_text_mean_abs_theta | 0.312902 ± 0.00329364 |
| crst_text_std_abs_theta | 0.204347 ± 0.0119219 |
| crst_text_p95_abs_theta | 0.685708 ± 0.0328869 |
| crst_text_mean_cos_base_rotated | 0.928698 ± 0.00237581 |
| crst_text_norm_preservation_max_abs_error | 1.90735e-06 ± 0 |
| crst_text_valid_context_edge_ratio | 0.983114 ± 0 |
| crst_text_angle_saturation_ratio | 0 ± 0 |
| imci_text_mean_hop_attention_entropy | 1.09179 ± 0.0045113 |
| imci_text_mean_hop_attention_1 | 0.368187 ± 0.021807 |
| imci_text_mean_hop_attention_2 | 0.312406 ± 0.0104173 |
| imci_text_mean_hop_attention_3 | 0.319407 ± 0.0125422 |
| imci_text_per_hop_attention_std_across_nodes_heads_1 | 0.0305462 ± 0.0141806 |
| imci_text_per_hop_attention_std_across_nodes_heads_2 | 0.0208231 ± 0.0100132 |
| imci_text_per_hop_attention_std_across_nodes_heads_3 | 0.0194886 ± 0.00580861 |
| crst_visual_mean_abs_theta | 0.246813 ± 0.0219154 |
| crst_visual_std_abs_theta | 0.191967 ± 0.0243512 |
| crst_visual_p95_abs_theta | 0.619503 ± 0.0722428 |
| crst_visual_mean_cos_base_rotated | 0.945365 ± 0.0126825 |
| crst_visual_norm_preservation_max_abs_error | 1.90735e-06 ± 0 |
| crst_visual_valid_context_edge_ratio | 0.983114 ± 0 |
| crst_visual_angle_saturation_ratio | 0 ± 0 |
| imci_visual_mean_hop_attention_entropy | 1.09699 ± 0.000864672 |
| imci_visual_mean_hop_attention_1 | 0.33553 ± 0.0104362 |
| imci_visual_mean_hop_attention_2 | 0.325843 ± 0.00621526 |
| imci_visual_mean_hop_attention_3 | 0.338627 ± 0.00770384 |
| imci_visual_per_hop_attention_std_across_nodes_heads_1 | 0.0151048 ± 0.0019755 |
| imci_visual_per_hop_attention_std_across_nodes_heads_2 | 0.0155856 ± 0.0101864 |
| imci_visual_per_hop_attention_std_across_nodes_heads_3 | 0.0139092 ± 0.00319879 |

## Paired contrasts

Descriptive matched dataset/seed differences; no significance claims.

| Contrast | Mean Acc delta (pp) | Mean Macro-F1 delta | Positive datasets (Acc / F1) | Positive seed pairs (Acc / F1) |
|---|---:|---:|---|---:|
| crst_only_minus_plain | 0.4362 | 0.004181 | Grocery, Movies, Toys, ele-fashion / Grocery, Toys, ele-fashion | 8/15 / 9/15 |
| absorb_only_minus_plain | 2.3315 | 0.037900 | Grocery, Movies, Reddit-S, Toys, ele-fashion / Grocery, Movies, Reddit-S, ele-fashion | 14/15 / 12/15 |
| full_minus_absorb_only | -0.0730 | -0.002005 | Grocery, Reddit-S / Grocery, ele-fashion | 7/15 / 6/15 |
| full_minus_crst_only | 1.8223 | 0.031713 | Grocery, Movies, Reddit-S, ele-fashion / Grocery, Movies, Reddit-S, ele-fashion | 13/15 / 12/15 |
| full_minus_plain | 2.2585 | 0.035894 | Grocery, Movies, Reddit-S, ele-fashion / Grocery, Movies, Reddit-S, ele-fashion | 13/15 / 12/15 |
