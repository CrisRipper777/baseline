# Multi-Order Bank NC factorial plan

## Frozen design

The formal factorial contains six variants:

- `mob_terminal_plain`, `mob_uniform_plain`, `mob_gpr_plain`
- `mob_terminal_residual`, `mob_uniform_residual`, `mob_gpr_residual`

Cross these with `Movies`, `Toys`, `Grocery`, `ele-fashion`, and `Reddit-S`: 30 dataset/variant jobs and 90 runs. Each job is one command with `seed=42 num_runs=3`; the task executes seeds 42, 43, and 44 internally. Two GPUs are assigned across independent jobs. No seed-specific shell jobs are created.

Every command uses `unified_full_graph_nc_v1`, full-graph training for graph encoders, Validation Accuracy checkpoint selection, and descriptive-only test metrics. Model capacity is frozen at hidden dimension 256, three propagation orders, three layers, and dropout 0.2.

Outputs are separated by dataset and variant under `outputs/mob_factorial_nc_v1/<dataset>/<variant>/`. Best checkpoints contain all three run states and validation-selected epoch metadata. They remain local and are not committed.

## Launcher

Dry-run the fixed matrix:

```bash
conda run --no-capture-output -n yhf_env python scripts/run_mob_factorial_nc.py --dry-run
```

Run the 30 jobs across the two GPUs:

```bash
GPU_IDS=0,1 scripts/run_multi_order_bank_nc.sh
```

The launcher assigns one physical GPU per job, passes `device=cuda:0` within that process, and leaves `seed=42,num_runs=3` intact. Completion metadata and per-job logs are stored under each variant directory.

## Paired contrasts

The analyzer computes all four NC metrics for each variant and same-seed paired differences:

- Retention: `uniform_plain - terminal_plain`; `uniform_residual - terminal_residual`.
- Adaptive response: `gpr_plain - uniform_plain`; `gpr_residual - uniform_residual`.
- Fusion: each residual variant minus its matching plain variant.

Practical support rules are defined in `multi_order_bank_spec.md`; they are based on validation accuracy, not test performance, and are not significance claims. A multi-order effect is called fusion-robust only when its direction has a positive mean and at least 3/5 positive dataset means in both fusion settings.

## Conditional fusion attribution

After the six primary variants, the analyzer checks the preregistered trigger. If any residual-minus-plain contrast reaches strong support, or has at least +0.30 pp mean validation accuracy and at least 3/5 positive dataset means, add the GPR-only `modality_refine_only` and `fusion_residual_only` variants for five datasets and three internal seeds each. Do not run this conditional stage if the trigger is false.

A parameter-count-matched plain GPR control is optional only if residual fusion clearly helps and capacity needs to be examined. It is not part of the primary matrix.

## Result analysis

After the factorial is complete, run:

```bash
conda run --no-capture-output -n yhf_env python scripts/summarize_multi_order_bank_nc.py \
  --factorial-root outputs/mob_factorial_nc_v1 \
  --baseline-root outputs/nc_benchmark_v1 \
  --output-root results/nc_benchmark_v1 \
  --device cuda:0
```

The analyzer writes run-level and aggregate tables, paired contrasts, gamma profiles, actual order contributions, hop redundancy, frozen sensitivity, `summary.json`, and `report.md`. It writes `fusion_attribution.csv` only when the trigger is met; the conditional jobs must then be completed before component effects can be reported.

## Baseline benchmark gate

The nine formal baselines are MLP, GCN, GraphSAGE, MMGCN, MGAT, DiP, DGF, DMGC, and LGMRec. The 45 dataset/model jobs use the same one-command `seed=42,num_runs=3` convention. Before launching them, finish the factorial and run 1-epoch/1-run full-graph CUDA preflight for DiP, MMGCN, MGAT, DGF, DMGC, and LGMRec on the largest graph, `ele-fashion`. Any OOM stops the baseline launch. The baseline launcher is:

```bash
conda run --no-capture-output -n yhf_env python scripts/run_nc_baselines.py --dry-run
GPU_IDS=0,1 conda run --no-capture-output -n yhf_env python scripts/run_nc_baselines.py
```

Provenance is stored at `outputs/nc_benchmark_v1/provenance.json`. No LP training is included in this round.


## Pre-launch verification record

- Full repository regression: 96 passed, one upstream PyG deprecation warning.
- Python `compileall`, `git diff --check`, and shell syntax check passed.
- All six Movies-NC endpoint workflow smokes (three readouts by two fusion modes) completed two epochs on CUDA and saved finite validation-selected checkpoints.
- The 30 formal factorial jobs, heavy-model preflight, and 45 baseline jobs have not started in this snapshot.
