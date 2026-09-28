# RISA v0.5 formal validation analysis

- Protocol: unified_full_graph_nc_v1
- Test and LP evaluation: disabled.
- Val Acc / Macro-F1: read from validation-selected checkpoints.
- Val cross-entropy: post-hoc calculation on val_idx only; it does not select checkpoints.
- CRST p95/std/saturation: deterministic random edge sample; means and norm/context summaries stream over edges.
- IMCI attention summaries aggregate over nodes and heads.

## Movies

### v05_full

| Metric | Mean ± population std |
|---|---:|
| val_acc | 0.561988 ± 0.00275264 |
| val_macro_f1 | 0.461045 ± 0.0098049 |
| val_cross_entropy | 1.45425 ± 0.00880133 |
| crst_text_mean_abs_theta | 0.298367 ± 0.0478941 |
| crst_text_std_abs_theta | 0.217793 ± 0.0323081 |
| crst_text_p95_abs_theta | 0.71911 ± 0.113851 |
| crst_text_mean_cos_base_rotated | 0.929035 ± 0.0224111 |
| crst_text_norm_preservation_max_abs_error | 1.90735e-06 ± 0 |
| crst_text_valid_context_edge_ratio | 0.978825 ± 0 |
| crst_text_angle_saturation_ratio | 0 ± 0 |
| imci_text_mean_hop_attention_entropy | 1.07036 ± 0.0126146 |
| imci_text_mean_hop_attention_1 | 0.310645 ± 0.00346709 |
| imci_text_mean_hop_attention_2 | 0.319749 ± 0.00635803 |
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
| imci_visual_mean_hop_attention_entropy | 0.924412 ± 0.0274732 |
| imci_visual_mean_hop_attention_1 | 0.182768 ± 0.0340184 |
| imci_visual_mean_hop_attention_2 | 0.364133 ± 0.0294531 |
| imci_visual_mean_hop_attention_3 | 0.453099 ± 0.0598453 |
| imci_visual_per_hop_attention_std_across_nodes_heads_1 | 0.159355 ± 0.022799 |
| imci_visual_per_hop_attention_std_across_nodes_heads_2 | 0.112984 ± 0.00799799 |
| imci_visual_per_hop_attention_std_across_nodes_heads_3 | 0.145843 ± 0.00695418 |

## Grocery

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
| imci_visual_mean_hop_attention_1 | 0.350103 ± 0.12721 |
| imci_visual_mean_hop_attention_2 | 0.334002 ± 0.0641036 |
| imci_visual_mean_hop_attention_3 | 0.315895 ± 0.0632259 |
| imci_visual_per_hop_attention_std_across_nodes_heads_1 | 0.256787 ± 0.046698 |
| imci_visual_per_hop_attention_std_across_nodes_heads_2 | 0.147605 ± 0.0188802 |
| imci_visual_per_hop_attention_std_across_nodes_heads_3 | 0.190417 ± 0.0217894 |

## ele-fashion

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
| imci_text_per_hop_attention_std_across_nodes_heads_1 | 0.234939 ± 0.00554698 |
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
| imci_text_mean_hop_attention_entropy | 1.09179 ± 0.00451135 |
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
| imci_visual_per_hop_attention_std_across_nodes_heads_1 | 0.0151048 ± 0.00197551 |
| imci_visual_per_hop_attention_std_across_nodes_heads_2 | 0.0155856 ± 0.0101864 |
| imci_visual_per_hop_attention_std_across_nodes_heads_3 | 0.0139092 ± 0.00319878 |
