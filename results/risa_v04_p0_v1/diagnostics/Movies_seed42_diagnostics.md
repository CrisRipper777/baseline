# RISA v0.4 P0 diagnostics: Movies seed 42

- Checkpoint: `/hdd1/DataInHere/YHF/baseline/outputs/risa_v04_p0_v1/smoke/Movies/p0_operator_routed/best.pt`
- Smoke validation metrics: `{"val_acc": 0.3293341398239136, "val_macro_f1": 0.02477436823104693}`
- Physical P_rel unchanged: `True`
- delta_off versus S45 identity-uniform max absolute errors (tolerance `2e-06`): `{"C1_text": 9.5367431640625e-07, "C1_visual": 9.5367431640625e-07, "C2_text": 9.5367431640625e-07, "C2_visual": 1.430511474609375e-06, "C3_text": 9.5367431640625e-07, "C3_visual": 9.5367431640625e-07, "H0_text": 0.0, "H0_visual": 0.0, "fused_z": 5.960464477539062e-07}`
- Selected diagnostic edges: `401` of `401` edges incident to the selected validation nodes
- Correction norm ratios: `{"text": 0.008541916497051716, "visual": 0.006079719867557287}`
- cos(base, base + delta): `{"text": 0.9999637603759766, "visual": 0.9999824166297913}`
- Router and operator summary: `{"text": {"entropy": 1.3862839937210083, "max_probability_sum_error": 1.1920928955078125e-07, "mean_probability": [0.2511703372001648, 0.24886702001094818, 0.2510952651500702, 0.248867467045784], "operator_usage": [0.2511703073978424, 0.248867005109787, 0.2510952353477478, 0.248867467045784], "pairwise_output_cosine": [0.9545018672943115, 0.9505166411399841, 0.9497645497322083, 0.9542223215103149, 0.9544891715049744, 0.9564952254295349], "selected_edge_count": 401}, "visual": {"entropy": 1.3862886428833008, "max_probability_sum_error": 1.1920928955078125e-07, "mean_probability": [0.24915826320648193, 0.2507145404815674, 0.25095751881599426, 0.24916972219944], "operator_usage": [0.24915827810764313, 0.2507145404815674, 0.2509574890136719, 0.24916972219944], "pairwise_output_cosine": [0.9428917765617371, 0.9442174434661865, 0.9226291179656982, 0.9454317688941956, 0.923478901386261, 0.922135591506958], "selected_edge_count": 401}}`
- Parallel/orthogonal decomposition: `{"text": {"max_abs_delta_minus_parallel_plus_orthogonal": 9.313225746154785e-10, "max_abs_parallel_dot_orthogonal": 2.9103830456733704e-11, "selected_edges": 401}, "visual": {"max_abs_delta_minus_parallel_plus_orthogonal": 4.656612873077393e-10, "max_abs_parallel_dot_orthogonal": 7.548806024715304e-11, "selected_edges": 401}}`
- Sanity checks: `PASS`
- Test and LP evaluation: disabled.
