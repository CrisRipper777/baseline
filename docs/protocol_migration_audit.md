# Protocol migration audit: baseline master vs MoPF V3

## Source revisions and safety

- Target baseline source commit: `8f7e825285ccff6e69f561b257cf160fcb3a1188` (`master`). The worktree was clean before creating `multi_order_bank`.
- MoPF protocol/model source ref: `V3`, commit `9f4514ed9eec4d14795a8aadbc125616880c3c9d`.
- All comparisons below use the target `master` snapshot and files from MoPF `V3`. The source checkout was clean on branch `V3`.
- The target node and edge split implementation is identical to V3; do not replace it as part of this migration.

## Files compared

The requested configs, tasks, model helpers, main entry point, utilities, all `src/data/**`, and all `tests/**` paths were compared. Key results:

| File(s) | Baseline vs V3 | Audit result |
|---|---|---|
| `configs/config.yaml` | Changed | Both already use `seed: 42`, `num_runs: 3`; V3 additionally has a research ablation selector, which is excluded. |
| `configs/task/nc.yaml` | Changed | Baseline is an older sampled-or-full-per-model protocol; V3 defines the frozen unified full-graph protocol. |
| `configs/task/lp.yaml` | Changed | Baseline has a scalar 5-neighbor setting and older schedule; V3 defines the unified sampled protocol. The project will pin three hops as `[5, 5, 5]`. |
| `src/tasks/nc.py`, `src/tasks/lp.py` | Changed | Protocol, optimizer, loader, deterministic sampling, checkpoint, and selection behavior differ; migrate only the generic frozen task behavior. |
| `src/tasks/common.py` | Changed | V3 adds generic optimizer, scheduler, and neighbor-depth helpers, plus many research-model diagnostic keys. Only the generic helpers are in scope. |
| `src/tasks/inference.py` | Identical | Full and layerwise inference interfaces already match. |
| `src/models/predictor.py`, `src/models/factory.py` | Identical | No framework migration is needed; model configs can use the existing dynamic factory. |
| `src/main.py` | Changed | Keep the baseline entry point; add only generic `torch_threads` handling. Task code writes configured best checkpoints directly, so V3 output-directory analysis plumbing is not needed. Do not port V3 ablation manifests or research bookkeeping. |
| `src/utils/seeds.py`, `src/utils/metrics.py` | Identical | Existing seed setter and metrics are sufficient. |
| `src/utils/summary.py` | Changed | Baseline uses sample standard deviation (`ddof=1`); V3 uses population standard deviation (`ddof=0`). The frozen framework convention is population std. |
| `src/data/splits.py`, `src/data/graph_utils.py`, `src/data/types.py`, `src/data/__init__.py` | Identical | Split semantics and core edge utilities already match. |
| `src/data/loaders.py` | Changed | The V3-only addition is `dataset.feature_mode` for modality masking/ablation. It is research-only and is not needed for the allowed migration. |
| `tests/test_splits.py`, `tests/test_metrics.py` | Identical | Existing split and metric tests can remain. |
| `tests/test_lp_sampling.py`, `tests/test_inference_equivalence.py` | Changed | V3 contains additional protocol and model-specific cases; add focused framework tests without importing V3 research tests wholesale. |
| Other V3-only tests | Source-only | Mostly MoPF/SSI/CoSI/ablation/analysis coverage; excluded. Only model tests for the three explicitly permitted imports may be adapted. |

## A. Baseline/master current NC protocol

- Defaults: 100 epochs, Adam, `lr=5e-3`, `weight_decay=1e-5`, per-epoch evaluation, patience 20, full inference with batch size 4096.
- Graph encoders use full-graph training only when `model.full_graph_training` or `model.requires_full_graph_training` requests it. Otherwise they enter `NeighborLoader`; this means graph models can silently receive different NC training protocols.
- MLP uses a feature-only node minibatch path.
- Gradient clipping is hard-coded to 1.0. There is no minimum early-stop epoch or minimum delta setting.
- Checkpoint selection is validation accuracy; validation Macro-F1 is computed and retained for the selected checkpoint. Test metrics are computed after selecting the checkpoint.
- There is no task-level protocol version or best-epoch checkpoint export contract.

## B. MoPF/V3 frozen NC protocol

- `protocol_version: unified_full_graph_nc_v1`; `training_mode: full_graph`; AdamW by default; 300 epochs, `lr=1e-3`, `weight_decay=1e-4`.
- Evaluate every epoch; patience 30 validation events; minimum epoch 30; minimum accuracy delta `1e-4`; gradient clipping 1.0; full inference, batch size 4096; scheduler null.
- Graph encoders receive one full-graph forward per training epoch, with cross-entropy supervised only on train nodes. MLP remains feature-only. The training mode is a task protocol and must not be decided solely from a model capability flag.
- Select by validation accuracy and record validation Macro-F1. Restore the selected checkpoint before final test evaluation. Report Val Accuracy, Val Macro-F1, Test Accuracy, and Test Macro-F1. Test metrics do not control selection or stopping.
- Support per-run `save_ckpt_path` and persist the selected epoch and selection metadata.
- Macro-F1 uses one stable set of valid labels observed over the supervised task splits, matching V3 evaluation semantics.

## C. Baseline/master current LP protocol

- Defaults: 50 epochs, Adam, `lr=1e-3`, `weight_decay=1e-5`, batch size 2048, scalar `num_neighbors: 5`, one training negative per positive.
- Link prediction is sampled using `LinkNeighborLoader` for graph encoders and edge-label minibatches for MLP. Train negatives are filtered against known positive edges, but the current implementation does not carry over V3's separate deterministic random streams or global edge-ID positive mask.
- Evaluation is full-embedding inference. Evaluation occurs every epoch with patience 20, evaluation edge batches of 2048. There is no minimum epoch/delta setting, explicit bidirectional subgraph setting, or protocol version.
- Decoder defaults are MLP, hidden dimension 256, 3 layers, dropout 0.02; baseline lacks the shared `proj_dim: 128` option.
- Selection is by validation MRR; final test ranking metrics are evaluated after selection.

## D. MoPF/V3 frozen LP protocol

- `protocol_version: unified_sampled_lp_v1`, `training_mode: sampled`, Adam; 150 epochs, `lr=1e-3`, `weight_decay=1e-5`; patience 10 validation events, minimum epoch 20, minimum delta `1e-4`; gradient clipping 1.0.
- Batch size 2048; formal project setting `[5, 5, 5]` (three layers); bidirectional `LinkNeighborLoader`; one train negative per positive; `global_eid` positive-message-edge masking. `local_keys` is retained only for equivalence/audit coverage.
- Filter train negatives against every train/valid/test positive and, for undirected data, each reverse edge. Regenerate deterministic filtered train negatives each epoch.
- Use independent per-epoch random streams: negative `seed + 10000 + epoch`, shuffle `seed + 20000 + epoch`, neighbor sampling `seed + 30000 + epoch`.
- Four loader workers, prefetch factor 2, 8 torch threads; full-graph inference with batch size 4096 and preloaded evaluation embeddings; evaluate every 2 epochs in edge batches of 512.
- `train_pos_per_epoch` and `max_train_batches` default to null. Decoder: MLP, hidden 256, 3 layers, dropout 0.02, projection dimension 128.
- Select by validation MRR with the configured minimum-delta/epoch rule; evaluate test only after restoring the best validation checkpoint. Report Val MRR, Test MRR, Hits@1, Hits@3, and Hits@10.

## E. Required migration surface

1. Replace `configs/task/nc.yaml` and `configs/task/lp.yaml` with the frozen task configs, with the project-required LP neighbor vector `[5, 5, 5]`.
2. Migrate generic `build_optimizer`, `scheduler_step`, and `resolve_num_neighbors` logic in `src/tasks/common.py`; preserve model-level optimizer preset overrides (`model.lr`, `model.weight_decay`) and explicit no-weight-decay parameter support. Do not copy research-only auxiliary-stat key lists.
3. Rework `src/tasks/nc.py` around task-level full-graph training, validation-only selection, full test reporting, best-epoch/checkpoint output, and run seeds `base_seed + run_id`.
4. Rework generic sampled LP helpers and loop in `src/tasks/lp.py`: deterministic filtered negatives, separate generators, bidirectional loader configuration, global-eID masking, full inference, Val-MRR checkpoint selection, and test-after-selection.
5. Keep `src/tasks/inference.py`, `src/models/predictor.py`, `src/models/factory.py`, `src/utils/seeds.py`, and `src/utils/metrics.py` unless a narrowly justified framework fix is required; their V3 copies are identical.
6. Change `src/utils/summary.py` to population std. Keep `configs/config.yaml` seeds/run count unchanged. Add only generic main/task plumbing needed for thread count and checkpoints.
7. Add protocol tests for NC graph-loader invariance, three-hop resolution, LP filtering/masking/RNG behavior, multi-run seeds, and population aggregation.

## F. MoPF research-only code excluded

Do not copy MoPF, SSI, CoSI, MAP, prototype, ablation, paper mechanism, or analysis implementations, naming, scripts, reports, or historical outputs. In particular exclude `src/ablation.py`, `src/analysis/**`, `src/tasks/analysis.py`, `src/tasks/modality.py`, `src/tasks/prototype_analysis.py`, research model files and configs, research launchers, and the V3 modality-feature-mode loader extension. Within V3 NC/LP, omit ablation manifests, modality-mask diagnostics, prototype/edge/node analysis exports, relation-specific research logging, and other model-mechanism hooks. The allowed model files are only DGF, DMGC, and LGMRec, plus their configs and minimum framework adaptation.

## G. Dataset split consistency

Yes. Target and V3 implementations of `src/data/splits.py`, `src/data/graph_utils.py`, and `src/data/types.py` are identical. The V3 `src/data/loaders.py` difference is a feature-mode masking extension only; it does not change split construction and is excluded. Keep the baseline split implementation.

## H. Inference semantics consistency

Yes for the common inference API: `src/tasks/inference.py` is byte-for-byte identical. Both use full graph features/edges for `inference_mode: full` and call `model.inference(...)` only for layerwise mode. The protocol configs default to full inference with batch size 4096. The NC mismatch is in training-loader selection (baseline model-flag-dependent sampling vs V3 task-level full graph), not in inference. LP evaluation is full-graph inference in both; the V3 protocol adds its explicit sampled-training invariants and evaluation preload behavior.


## Migration completion and final QA

### A. Modified files

- Protocol config and runner: configs/task/nc.yaml, configs/task/lp.yaml, src/tasks/common.py, src/tasks/nc.py, src/tasks/lp.py, src/main.py, src/utils/summary.py.
- Imported V3 baselines: src/models/dgf.py, src/models/dmgc.py, src/models/lgmrec.py and their three configs under configs/model/.
- New backbone and formal preparation: src/models/multi_order_bank.py, configs/model/multi_order_bank.yaml, scripts/run_multi_order_bank_nc.sh, scripts/summarize_multi_order_bank_nc.py.
- Documentation and focused tests: the four requested docs plus tests for protocol, imported models, the new backbone, and future result analysis.
- src/data/splits.py and src/data/loaders.py remain unchanged; no research-only V3 files or outputs were copied.

### B. Protocol before/after

NC changed from per-model sampled-or-full behavior, 100 epochs, Adam, and the older learning-rate schedule to unified_full_graph_nc_v1: graph models always use full graph, MLP uses feature-only batches, 300 epochs, AdamW, validation accuracy selection, and the configured min-epoch/min-delta stopping rule. The final run reports all four required validation/test accuracy and Macro-F1 metrics.

LP changed from scalar 5-neighbor fanout and the older training cadence to unified_sampled_lp_v1 with three fanouts [5, 5, 5], deterministic filtered negatives and independent streams, bidirectional LinkNeighborLoader, global_eid positive-edge masking, full inference, and validation-MRR selection.

Both tasks call set_seed(base_seed + run_id). Population std is the aggregation convention. Model-level learning-rate and weight-decay presets override task defaults through the migrated generic optimizer builder.

### C. Imported model mapping

See imported_baseline_notes.md. DGF, DMGC, and LGMRec were copied from the recorded V3 source commit. Their current supervised adaptations and dropped auxiliary objectives are documented there. DMGC received one CUDA inference-device interface fix after the CUDA test found a CPU/CUDA mismatch.

### D. Multi-Order Bank formula mapping

See multi_order_bank_spec.md. The implementation uses two independent modality projectors, one shared physical GCN operator, explicitly retained S0 through S3 states, terminal/uniform/signed-GPR readouts, and one shared fusion projection.

### E. Unit and regression results

- Targeted protocol/import/backbone/analyzer tests pass.
- Full repository regression: 69 passed, 1 upstream PyG deprecation warning.
- py_compile: 47 Python files compiled.
- bash -n scripts/run_multi_order_bank_nc.sh passed.
- git diff --check passed.

### F. Smoke results

- DGF, DMGC, and LGMRec each passed CPU/CUDA forward/inference tests and one-epoch Movies-NC smoke.
- Each imported model passed a two-batch sports-LP sampled smoke with [5, 5, 5] and global_eid masking; removed edge counts were 4, 2, and 4 respectively.
- Multi-Order Bank terminal, uniform, and gpr each passed a two-epoch Movies-NC smoke and wrote a finite validation-selected checkpoint. GPR gamma parameters are present in its checkpoint.
- Smoke outputs are under outputs/imported_smoke/ and outputs/multi_order_bank_nc/smoke/.

### G. Formal commands

The exact 15 seed=42, num_runs=3 commands and launcher are in multi_order_bank_nc_plan.md. Each command runs seeds 42, 43, and 44 internally and separates dataset/readout output paths.

### H. Launch status

No formal imported-model benchmark or five-dataset Multi-Order Bank benchmark was started.

### Frozen config SHA-256

- NC: b1e678381bc996c35b1db3d68675bb54ecfe34ca324348008ba4b016d1aab8dd
- LP: 0e0315fa34a72c653e354cff6aad6b6d73d99aeb87c545313674f70faccbe534
