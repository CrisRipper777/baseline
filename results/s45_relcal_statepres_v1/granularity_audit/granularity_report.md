# S4.5 G0: Frozen controller granularity audit

S4.4 training SHA: `8e53b1b8f1eed4aac6a61082222327f4dca17da3`. Checkpoints read: 36; all were validation-selected NC checkpoints with test evaluation disabled.

Mean weighted within-target variance fraction across all controller dimensions: 0.096800 (population SD 0.089376).

Controller-family means (population SD): scalar_rel: 0.2615 (0.1120); lowrank_rel: 0.0786 (0.0592); expert_rel: 0.0875 (0.0931).

Matched scalar-minus-family within-ratio differences: lowrank_rel: mean 0.1829, positive in 23/24 pairs; expert_rel: mean 0.1740, positive in 23/24 pairs.

The CSV retains modality, controller dimension, total/within/between variance, within ratio, decomposition error, and weighted within-target pairwise L2. Frozen target-mean, dosage-only, and redistribution-only results are checkpoint sensitivity probes, not retrained causal ablations.

A high within-target variance fraction indicates edge-specific controller assignment at the checkpoint; it does not establish that the assigned magnitudes are useful. See the intervention CSV for validation accuracy, macro-F1, true-label CE, and prediction flips.

Row-mass preservation in the later S4.5 model constrains only one-step off-diagonal target mass. It does not imply symmetry or spectral equivalence.
