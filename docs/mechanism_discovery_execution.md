# Mechanism discovery execution plan

## Source and completion record

- Baseline input: branch `multi_order_bank`, commit `31468c5a1439b3561487936bf5d83749b502f61e`.
- Upstream model source: `/hdd1/DataInHere/YHF/MoPF_IAMOC`, ref `V3`, commit `9f4514ed9eec4d14795a8aadbc125616880c3c9d`.
- Discovery implementation/preflight checkpoint: `9ae5db82a93277b9e3423705bfc1e66eeba4497b`.
- Completed analysis and result tables: `4573ae60f90c4fbd7f7703b9bfedddfc07d654c8`.
- Formal discovery training completed: 25 order-source jobs (75 runs) and 20 unimodal jobs (60 runs), each job internally uses seeds 42/43/44. The separate full 5-dataset × 3-readout benchmark was not started.


All commands run from the repository root on branch `mechanism_discovery`. Formal jobs use the shared dynamic GPU scheduler (`--gpus 0,1`), one active job per GPU, and each individual dataset/variant command sets `seed=42,num_runs=3` so the task executes seeds 42, 43, and 44 internally.

## Step A: existing-checkpoint diagnostics

```bash
conda run --no-capture-output -n yhf_env python scripts/analyze_mechanism_discovery.py step-a
```

This performs no training. It runs representation geometry, fixed frozen probes, sparse spectral analysis, and validation-only stability perturbations from existing terminal/uniform/GPR plain checkpoints. The stability table was recomputed once after a cache-identity regression was found and fixed; the rerun also performs no training:

```bash
conda run --no-capture-output -n yhf_env python scripts/analyze_mechanism_discovery.py stability
```

## Required CUDA preflight

Run the one-epoch/one-run preflight on ele-fashion for all five new order-source readouts under both/text/visual, before any of the 135 formal runs:

```bash
conda run --no-capture-output -n yhf_env python scripts/run_mechanism_discovery_nc.py preflight --gpus 0,1
```

Any correctness issue, nonfinite output, or OOM stops the formal phases. Do not reduce the frozen formal configuration to work around a failure.

## Step B: order-source controls (75 runs)

```bash
conda run --no-capture-output -n yhf_env python scripts/run_mechanism_discovery_nc.py order --gpus 0,1
```

This creates 25 dataset/readout jobs, each with three internal seeds: 75 runs total. Existing `terminal_plain`, `uniform_plain`, and `gpr_plain` checkpoints are reused as frozen controls.

## Step C: unimodal controls (60 runs)

```bash
conda run --no-capture-output -n yhf_env python scripts/run_mechanism_discovery_nc.py unimodal --gpus 0,1
```

This creates 20 dataset/mode jobs, each with three internal seeds: 60 runs total.

## Step D: dependent analysis

```bash
conda run --no-capture-output -n yhf_env python scripts/analyze_mechanism_discovery.py final
```

The final pass writes the order-source paired contrasts, all Experiment 2 tables, validation-node tables with SHA256 manifest, and the Problem Discovery Matrix. It does not use test labels for discovery.

## Outputs

Training outputs are under `outputs/mechanism_discovery_v1/order_source/` and `outputs/mechanism_discovery_v1/unimodal/`. One-epoch preflight outputs are separately stored under `outputs/mechanism_discovery_v1/preflight/`. Analysis results are under `results/mechanism_discovery_v1/experiment1/`, `results/mechanism_discovery_v1/experiment2/`, and `results/mechanism_discovery_v1/problem_discovery_matrix.csv`.

Do not start the 135 formal runs until implementation, full tests, `compileall`, `git diff --check`, branch commit/push, provenance generation, and successful CUDA preflight are complete. After the first formal job starts, do not modify model or training code. A necessary analyzer-only correction after formal completion must be committed separately and documented without retraining.
