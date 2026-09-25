# Sparse spectral audit

Uses the exact model normalized operator and `scipy.sparse.linalg.eigsh`; no dense eigendecomposition or graph changes were used.

## Movies

- Nodes: 16672; undirected edge pairs: 80401.
- Largest 32 convergence: True (32 values). Failure: none.
- Smallest 32 convergence: True (32 values). Failure: none.
- Observed spectral span: [-0.461725, 1.000000].

## Toys

- Nodes: 20695; undirected edge pairs: 56701.
- Largest 32 convergence: True (32 values). Failure: none.
- Smallest 32 convergence: True (32 values). Failure: none.
- Observed spectral span: [-0.618389, 1.000000].

## Grocery

- Nodes: 17074; undirected edge pairs: 71131.
- Largest 32 convergence: True (32 values). Failure: none.
- Smallest 32 convergence: True (32 values). Failure: none.
- Observed spectral span: [-0.569966, 1.000000].

## ele-fashion

- Nodes: 97766; undirected edge pairs: 199586.
- Largest 32 convergence: True (32 values). Failure: none.
- Smallest 32 convergence: True (32 values). Failure: none.
- Observed spectral span: [-0.804969, 1.000011].

## Reddit-S

- Nodes: 15894; undirected edge pairs: 141540.
- Largest 32 convergence: True (32 values). Failure: none.
- Smallest 32 convergence: True (32 values). Failure: none.
- Observed spectral span: [-0.000000, 1.000002].

## Response summary by dataset and GPR seed

Mid-frequency retention is mean absolute response over observed eigenvalues with 0.25 <= |lambda| <= 0.75. Negative-frequency attenuation is reported as mean absolute response over observed negative eigenvalues. GPR distance is RMSE to Uniform over the audited eigenvalues.

| Dataset | Seed | Uniform mid | Terminal mid | GPR text mid | GPR visual mid | GPR text vs Uniform RMSE | GPR visual vs Uniform RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Movies | 42 | 0.17411 | 0.06436 | 0.26555 | 0.25315 | 0.06940 | 0.19133 |
| Movies | 43 | 0.17411 | 0.06436 | 0.25921 | 0.24838 | 0.06262 | 0.18057 |
| Movies | 44 | 0.17411 | 0.06436 | 0.25429 | 0.24496 | 0.06117 | 0.18260 |
| Toys | 42 | 0.14705 | 0.16489 | 0.18009 | 0.16219 | 0.10430 | 0.07556 |
| Toys | 43 | 0.14705 | 0.16489 | 0.17984 | 0.16350 | 0.09829 | 0.08068 |
| Toys | 44 | 0.14705 | 0.16489 | 0.20173 | 0.17086 | 0.13017 | 0.08991 |
| Grocery | 42 | 0.16489 | 0.09477 | 0.25077 | 0.21416 | 0.15546 | 0.12428 |
| Grocery | 43 | 0.16489 | 0.09477 | 0.22242 | 0.19655 | 0.12260 | 0.10229 |
| Grocery | 44 | 0.16489 | 0.09477 | 0.24333 | 0.20607 | 0.14532 | 0.11438 |
| ele-fashion | 42 | 0.11846 | 0.30750 | 0.30536 | 0.29561 | 0.13974 | 0.13457 |
| ele-fashion | 43 | 0.11846 | 0.30750 | 0.30730 | 0.31112 | 0.14346 | 0.14901 |
| ele-fashion | 44 | 0.11846 | 0.30750 | 0.31150 | 0.30972 | 0.14401 | 0.14516 |
| Reddit-S | 42 | nan | nan | nan | nan | 0.09960 | 0.09029 |
| Reddit-S | 43 | nan | nan | nan | nan | 0.10514 | 0.09543 |
| Reddit-S | 44 | nan | nan | nan | nan | 0.10751 | 0.09028 |