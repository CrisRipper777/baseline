# S4.3 Phase 2: Adaptive Structural Utility Audit

Training commit: `9b9f1eb04cb7d83f55506f47b654905efe021d9b`. Analysis commit: `89e4bfecdb67d48663e8452e7bbcf8efe6e99606`. Source commit: `601bad98ff28e85c9aeac23b84b5db51ad4c7c86`.
Protocol: unified full-graph node classification; validation-only selection/evaluation; test evaluation disabled. Formal scope is Movies, Grocery, ele-fashion, Reddit-S; Toys remains a holdout.

## Result status

- H1.5: **MECHANISM_SUPPORT** — lambda*=0 shares span 0.083–0.256; modality-only aggregate scan minima differ in 11/12 dataset-seed pairs; mean node-oracle headroom=0.29839.
- H1R: **MIXED** — Some paired metric effects are positive, but pooled effects do not consistently favor the right-hand variant.
- H2a scalar adaptation: **MECHANISM_SUPPORT** — At least one pooled metric favors the right-hand variant; effects vary across pairs.
- H2a group adaptation: **MECHANISM_SUPPORT** — At least one pooled metric favors the right-hand variant; effects vary across pairs.
- H2b global differential vs matched generic correction: **MIXED** — Some paired metric effects are positive, but pooled effects do not consistently favor the right-hand variant.
- H2b node differential vs matched generic correction: **MIXED** — Some paired metric effects are positive, but pooled effects do not consistently favor the right-hand variant.

## Answers

1. **Differential usage heterogeneity:** node-oracle mean validation CE headroom is 0.2984; lambda*=0 fractions vary by dataset/seed, and Text-only versus Visual-only scan minima differ in 11/12 checkpoint pairs. This is an optimistic descriptive bound and aggregate validation scan, not a deployable selector.
2. **ReLU/sign preservation:** signed functional minus signed aggregative averages -0.0172 accuracy, -0.0467 Macro-F1, and +0.0021 CE, so signed preservation does not rescue the functional basis here (MIXED). Signed functional versus historical ReLU is ACTIVATION_CHANGED and averages -0.0045 accuracy and -0.0135 Macro-F1; it is not a pure causal comparison.
3. **Structural dosage:** global scalar beta versus reused P0 residual averages +0.0011 accuracy and +0.0066 Macro-F1, with CE +0.0606. Node-vs-global effects are +0.0004 accuracy for scalar and -0.0000 accuracy / -0.0717 CE for group dosage. Effects are small and metric-dependent; group indices have no intrinsic semantic meaning, and global beta values are per-dataset priors.
4. **Functional node assignment:** forcing beta=1 on trained H2a node gates changes accuracy/F1 by -0.0115/-0.0187; validation-mean replacement changes them by -0.0013/-0.0027 (CE +0.0044), while degree-bin shuffling averages -0.0024/-0.0047 (CE +0.0045). H2b node mean replacement gives -0.0017 accuracy (CE +0.0054) and degree-bin shuffling gives -0.0033/-0.0053 (CE +0.0069). This is modest frozen sensitivity, not a retrained causal ablation.
5. **D correction value:** global differential-minus-generic averages -0.0052 accuracy, -0.0127 Macro-F1, -0.0256 CE; node differential-minus-generic averages -0.0146, -0.0248, +0.1340. Differential corrections do not show consistent utility over matched generic corrections.
6. **D correction adaptation:** node_diff minus global_diff averages -0.0091 accuracy. Node shuffling and mean replacement move validation metrics only modestly; coefficients are learned dosage, not causal contributions.
7. **Next-stage carrier:** if one carrier is carried forward for a later hypothesis test, the simplest candidate is modality-specific global structural dosage (`h2a_global_scalar`). Node/group adaptation has no consistent accuracy gain, and the tested D corrections regress on the primary classification contrasts. Treat that choice as provisional; this report does not define an H3/H4 architecture.

## Complexity

The complexity table contains 40 formal variant-dataset profiles plus the reused P0 residual reference for each dataset. See `complexity_table.csv` for parameters, peak memory, wall time, mean epoch proxy, and best epoch.

## Interpretation boundaries

- H1.5 node oracle uses validation labels and is DESCRIPTIVE_ORACLE_ONLY; modality-only scan minima are descriptive validation summaries.
- H1R tests the current handcrafted differential basis, not all learned experts or transforms.
- Learned beta/lambda do not equal causal contribution.
- Group indices have no intrinsic semantic meaning.
- Global beta/lambda are per-dataset trained priors, not cross-dataset adaptation.
- Frozen interventions are sensitivity tests, not retrained ablations.
- No H3/H4 experiment was started.
