# H1.5 Frozen Differential-Usage Heterogeneity Audit

This is a frozen validation-only scan of historical `h1_dual_functional_static` checkpoints. No model was retrained.
The node oracle uses validation labels and is `DESCRIPTIVE_ORACLE_ONLY`; oracle-selected lambdas were not used for training, routing, thresholds, or hyperparameter selection.

- Checkpoints: 12 historical checkpoints.
- Fixed lambda grid: `(-1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0)`.
- Max absolute logit deviation at lambda=1 from historical NORMAL: 0.
- Lambda=0 retains `0.5 * R_A` and removes D while preserving A scale.

## Validation-node oracle proportions by dataset and seed

| Dataset | Seed | lambda*=0 | lambda*>0 | lambda*<0 | Mean headroom |
|---|---:|---:|---:|---:|---:|
| Movies | 42 | 0.087 | 0.517 | 0.397 | 0.50480 |
| Movies | 43 | 0.086 | 0.524 | 0.390 | 0.49970 |
| Movies | 44 | 0.083 | 0.531 | 0.386 | 0.44113 |
| Grocery | 42 | 0.185 | 0.501 | 0.314 | 0.46940 |
| Grocery | 43 | 0.167 | 0.568 | 0.265 | 0.28299 |
| Grocery | 44 | 0.172 | 0.558 | 0.271 | 0.29439 |
| ele-fashion | 42 | 0.145 | 0.516 | 0.339 | 0.19924 |
| ele-fashion | 43 | 0.137 | 0.484 | 0.379 | 0.31155 |
| ele-fashion | 44 | 0.154 | 0.496 | 0.349 | 0.23658 |
| Reddit-S | 42 | 0.223 | 0.308 | 0.469 | 0.11330 |
| Reddit-S | 43 | 0.256 | 0.396 | 0.348 | 0.11865 |
| Reddit-S | 44 | 0.255 | 0.391 | 0.354 | 0.10900 |

## Modality-only validation scans

These aggregate scan minima also use validation labels and are descriptive summaries of the preregistered grid; they were not used to select a model or training hyperparameter.
| Dataset | Seed | Text-only best lambda | Visual-only best lambda | Best lambdas differ |
|---|---:|---:|---:|:---:|
| Movies | 42 | 1.5 | 0 | yes |
| Movies | 43 | 2 | 0 | yes |
| Movies | 44 | 1 | 0.25 | yes |
| Grocery | 42 | -0.5 | 0 | yes |
| Grocery | 43 | 0.5 | 1 | yes |
| Grocery | 44 | 0.25 | 1 | yes |
| ele-fashion | 42 | 1.5 | 0.25 | yes |
| ele-fashion | 43 | 1 | 0.5 | yes |
| ele-fashion | 44 | 1 | 0.5 | yes |
| Reddit-S | 42 | 0 | 0 | no |
| Reddit-S | 43 | -1 | -0.5 | yes |
| Reddit-S | 44 | -0.5 | 0 | yes |

## Interpretation boundary

The node oracle is optimistic because it minimizes each validation node's true-label CE over the same fixed grid. It does not estimate deployable routing value. Modality-only minima are likewise descriptive validation scans, not selected operating points.
