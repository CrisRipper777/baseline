# Baseline implementation alignment audit (MoPF origin/V3)

Target: `baseline`, branch `multi_order_bank`. Source of truth: read-only Git object `origin/V3` at `9f4514ed9eec4d14795a8aadbc125616880c3c9d` (fetched before alignment). Source worktree was not edited.

The audit covers all nine formal NC baselines. SHA columns are SHA-256 of file bytes. “Initial target SHA” is the pre-alignment value where a code/config mismatch was detected; blank means the original file already matched the source.

| Model | File | V3 SHA-256 | Aligned target SHA-256 | Initial target SHA-256 | Code/config difference and final action |
|---|---|---|---|---|---|
| mlp | `src/models/mlp.py` | `499501fd6f2d642d06a3dd6e30d34a7cd12c0caa347df5bf248848bad9626d1b` | `499501fd6f2d642d06a3dd6e30d34a7cd12c0caa347df5bf248848bad9626d1b` | `499501fd6f2d642d06a3dd6e30d34a7cd12c0caa347df5bf248848bad9626d1b` | No code changes; byte-identical to V3. |
| mlp | `configs/model/mlp.yaml` | `f6ebad28404f2383ecee6db7621d70be5a5724a7063c0e919a0d9a808e35b7d0` | `f6ebad28404f2383ecee6db7621d70be5a5724a7063c0e919a0d9a808e35b7d0` | `—` | Config is byte-identical to V3. |
| gcn | `src/models/gcn.py` | `a99db763834bbd3b95337dec7a19b7e5ae723c7ce5b510944cb06b28bb3765a3` | `a99db763834bbd3b95337dec7a19b7e5ae723c7ce5b510944cb06b28bb3765a3` | `a99db763834bbd3b95337dec7a19b7e5ae723c7ce5b510944cb06b28bb3765a3` | No code changes; byte-identical to V3. |
| gcn | `configs/model/gcn.yaml` | `60ab35c7e7bbcd122e60a1a5619c564571bd57561aa0f03fa6efd78a2a02b1ae` | `60ab35c7e7bbcd122e60a1a5619c564571bd57561aa0f03fa6efd78a2a02b1ae` | `—` | Config is byte-identical to V3. |
| sage | `src/models/sage.py` | `f85af461171768aec5a2bc60c1d3741ad6b4c6c8cf41808ccd10b53501845dc5` | `f85af461171768aec5a2bc60c1d3741ad6b4c6c8cf41808ccd10b53501845dc5` | `f85af461171768aec5a2bc60c1d3741ad6b4c6c8cf41808ccd10b53501845dc5` | No code changes; byte-identical to V3. |
| sage | `configs/model/sage.yaml` | `d14c28849dd378f27164960bb725c3c32ae6539b0398f3019b386fbaf8af35ff` | `d14c28849dd378f27164960bb725c3c32ae6539b0398f3019b386fbaf8af35ff` | `65cd3c54a82869b8280f14561784da9d96064873d72d0108bd2cdc53249e57bd` | Changed dropout from 0.5 to V3 0.2. |
| mmgcn | `src/models/mmgcn.py` | `1b79ebe3ea524310b7909022de048a8cca5bc2594c3c395c89a0981fe2eb16ca` | `1b79ebe3ea524310b7909022de048a8cca5bc2594c3c395c89a0981fe2eb16ca` | `1b79ebe3ea524310b7909022de048a8cca5bc2594c3c395c89a0981fe2eb16ca` | No code changes; byte-identical to V3. |
| mmgcn | `configs/model/mmgcn.yaml` | `e595fd305c7b3c7976a7b184404dbe53bf1bda0e5d10358893a625721237219d` | `e595fd305c7b3c7976a7b184404dbe53bf1bda0e5d10358893a625721237219d` | `—` | Config is byte-identical to V3. |
| mgat | `src/models/mgat.py` | `42ee1eff0cb5ea1e9c5857770dc9a7554e4882cbcb55c8d3e493632e85cc1fd8` | `42ee1eff0cb5ea1e9c5857770dc9a7554e4882cbcb55c8d3e493632e85cc1fd8` | `baccc745a63a6e5376d9c59b3c00d775e1457fa2528e7a52089ae53b97ee0f7f` | Replaced target-specific normalization, self-loop behavior, ID-embedding registration, and configurable branch normalization with the exact V3 implementation. |
| mgat | `configs/model/mgat.yaml` | `1b1a8cb21861b085a483c21f55868ef61cef2d58a22ab57204d47ed03ccf1bd1` | `1b1a8cb21861b085a483c21f55868ef61cef2d58a22ab57204d47ed03ccf1bd1` | `0c5b9243fc7e1f046295ceb2f66afda762ac1d5157870d8ea3ae5e20cb98a1a5` | Removed target-only norm=batchnorm; now byte-identical to V3. |
| dip | `src/models/dip.py` | `a9d14f7420b79e9aa7c14946a61983d121f8f20eebd187164d97148eccfb26a9` | `a9d14f7420b79e9aa7c14946a61983d121f8f20eebd187164d97148eccfb26a9` | `a9d14f7420b79e9aa7c14946a61983d121f8f20eebd187164d97148eccfb26a9` | No code changes; byte-identical to V3. |
| dip | `configs/model/dip.yaml` | `1361cf6c7cbca9ec5259d7b07803712fa3b03839bf26a3572d30e3cd56f562ba` | `1361cf6c7cbca9ec5259d7b07803712fa3b03839bf26a3572d30e3cd56f562ba` | `—` | Config is byte-identical to V3. |
| dgf | `src/models/dgf.py` | `a76e43932acaeeaf2eae264929726cf02ad25f19b5cedfa8b74400e4354d46cf` | `a76e43932acaeeaf2eae264929726cf02ad25f19b5cedfa8b74400e4354d46cf` | `3e9c06b1d2fd0598d60a8bd0e20dcd6922783d903ab7abcb39fdc06c52f7f7df` | Only stale comments differed; now byte-identical to V3. Filtering-core supervised adaptation preserved. |
| dgf | `configs/model/dgf.yaml` | `637626f86525b19451db80a153a8d6a6cfa74de992464b990e2f0aa99c2e50ff` | `637626f86525b19451db80a153a8d6a6cfa74de992464b990e2f0aa99c2e50ff` | `—` | Config is byte-identical to V3. |
| dmgc | `src/models/dmgc.py` | `e55629ce1fa691c87838ca15497b302e84c14985c0fe2263a1799d43a3e1cbac` | `f14c3708edcffd5ecca97d8e042bbd3da684505c17e3ec8c739300209adba5a4` | `f14c3708edcffd5ecca97d8e042bbd3da684505c17e3ec8c739300209adba5a4` | V3 forward math, parameters, graph construction, and config retained. The existing inference device-placement correction is the only code difference: INTERFACE_CORRECTNESS_FIX_ONLY. |
| dmgc | `configs/model/dmgc.yaml` | `2a6cec4e32709b5e873a964a254d582dd3da3a4c1a91f8e3b15fbb5afe4d2218` | `2a6cec4e32709b5e873a964a254d582dd3da3a4c1a91f8e3b15fbb5afe4d2218` | `—` | Config is byte-identical to V3. |
| lgmrec | `src/models/lgmrec.py` | `eb8802f2f01dc4a4009b483fc80d7ac3007cdfeefa1c72a671e6729c90fd3e54` | `eb8802f2f01dc4a4009b483fc80d7ac3007cdfeefa1c72a671e6729c90fd3e54` | `eb8802f2f01dc4a4009b483fc80d7ac3007cdfeefa1c72a671e6729c90fd3e54` | No code changes; byte-identical to V3. |
| lgmrec | `configs/model/lgmrec.yaml` | `8f79d327242af140131ca4ba8b716f3cb6d0067205d28894e478090a202e5788` | `8f79d327242af140131ca4ba8b716f3cb6d0067205d28894e478090a202e5788` | `—` | Config is byte-identical to V3. |

## Resolved NC presets

All models train under `unified_full_graph_nc_v1`: AdamW and task defaults `lr=1e-3`, `weight_decay=1e-4`, 300 epochs, full-graph graph-encoder forward, validation-accuracy checkpoint selection. `build_optimizer` honors model YAML overrides for DGF, DMGC, and LGMRec.

| Model | Configured dimensions/depth | Actual optimizer LR | Actual weight decay |
|---|---|---:|---:|
| MLP | hidden_dim=256, num_layers=2 | 1e-3 | 1e-4 |
| GCN | hidden_dim=256, num_layers=3 | 1e-3 | 1e-4 |
| GraphSAGE | hidden_dim=256, num_layers=3, dropout=0.2 | 1e-3 | 1e-4 |
| MMGCN | hidden_dim=128, num_layers=2 | 1e-3 | 1e-4 |
| MGAT | hidden_dim=128, num_layers=1 | 1e-3 | 1e-4 |
| DiP | d_model=256, q_dim=256, mp_hops=3 | 1e-3 | 1e-4 |
| DGF | hidden_dim=64, num_layers=10 | 5e-3 | 1e-5 |
| DMGC | hidden_dim=128, num_layers=1 | 5e-3 | 1e-5 |
| LGMRec | hidden_dim=128, num_layers=3 | 5e-3 | 1e-5 |

Per-dataset trainable parameter counts (model and classifier separately), actual optimizer groups, best epoch, and run metrics are recorded in each NC run sidecar and will be summarized in `results/nc_benchmark_v1/nc_baseline_table.csv` after the formal benchmark.

## Alignment decisions

- MLP, GCN, GraphSAGE, MMGCN, DiP, and LGMRec implementation/config bytes already matched V3 before this round, except GraphSAGE’s target-only dropout value.
- MGAT was replaced with the V3 model because its normalization, edge/self-loop semantics, and registered trainable ID embeddings changed its mathematical behavior. Its config now has no target-only norm setting.
- DGF now matches V3 exactly, including comments; no filtering-core math changed.
- DMGC retains only the prior inference device-placement fix. CPU and CUDA full-forward/inference equivalence tests pass. This fix does not change parameters, training forward, or graph-construction math.
- UniGraph2 code/config were removed. The active NC dataset launcher scope is Movies, Toys, Grocery, ele-fashion, and Reddit-S; books-nc is excluded.

## Verification state

Targeted CPU/CUDA forward, backward, and inference tests passed for all nine baselines; explicit DMGC CPU/CUDA full-forward/inference equivalence also passed. The full repository regression passed: 96 tests, with one upstream PyG deprecation warning. `compileall`, `git diff --check`, and launcher shell syntax passed. All six Movies-NC factorial endpoint smokes completed two epochs on CUDA with finite checkpoints and validation-selected metadata. The formal 30-job factorial and heavy-model preflight have not started at this audit snapshot.
