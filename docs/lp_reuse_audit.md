# LP historical-result reuse audit

Scope is provenance-only. No LP training was started. `books-lp` was excluded. The only datasets considered are `sports-copurchase` and `cloth-copurchase`.

## Current frozen protocol and config hashes

- Target protocol: `unified_sampled_lp_v1`; formal fanout `[5,5,5]`.
- Target `configs/task/lp.yaml` SHA-256: `0e0315fa34a72c653e354cff6aad6b6d73d99aeb87c545313674f70faccbe534`.
- Refreshed V3 task-config SHA-256: `3b1f63171d4c99b6ba104fbe1f5a7cbb51b33ccf0dd3b2b12ed7fb1dd580943e`. The byte difference is comments plus optional `eval_modality_masks` defaults; historical resolved config records that switch disabled. Runtime training/evaluation fields align, but the manifest hash is not byte-identical to the current target YAML.
- `sports-copurchase` dataset YAML SHA-256: `1026c2b5d839d71c1717d74377f8d9a983fba17c26f1eca15d81d5649213a3e5` (matches historical manifest). Current LP split SHA-256: `6f5ffeb99f1aca5ea05671c6a86e5ee01a33a3673b6c3f0a24c6d482f4fa710a`.
- `cloth-copurchase` dataset YAML SHA-256: `6fdac1675e94f39302d0805ef95bb0d5678f2790c59260a27793de508b097b16`. Current LP split SHA-256: `b0a0866c3595b1c4e676db7805b91ccd283ebdda71a4038f37ac9b6593964697`.
- Historical sports manifest says source branch `exp/iamoc-v2`, commit `e61e8c9d6cc6939c26e85da48242b3af83abafc4`, clean worktree, seeds/split seed `42`, internal seeds `[42,43,44]`. LP implementation/config/model/data code relevant to this audit is unchanged between that commit and refreshed `origin/V3`.
- Historical sports manifest lists 8 models: GCN, GraphSAGE, MMGCN, MGAT, DGF, DMGC, LGMRec, and DiP. Each has completed markers, resolved configs, aggregate results and three run checkpoints. MLP has no historical LP result.

## Field comparison

For the eight sports artifacts, resolved configs and benchmark manifests show: `[5,5,5]`; Adam; 150 epochs; `lr=1e-3`, `weight_decay=1e-5`; deterministic train negatives with independent negative/shuffle/neighbor offsets; `positive_edge_mask_backend=global_eid`; bidirectional LinkNeighborLoader; full-graph evaluation; MLP decoder `(hidden=256,layers=3,dropout=0.02,proj_dim=128)`; seed 42 with runs 42/43/44; validation MRR selection; and Val MRR, Test MRR, Hits@1/3/10. Model-config hashes match the current aligned target files. Historical resolved config points to the same sports split path as the current dataset config.

The current split file hashes above describe the bytes available now. The historical benchmark manifest and per-model artifacts contain no split-content hash, so they do not prove those were the bytes used during training. The target task YAML hash also differs from the recorded source config hash. Under the strict reuse rule, these sports results stay `UNVERIFIED`; they may be retained as contextual references but are not promoted as verified reused baselines.

There are no cloth-copurchase result artifacts in the authorized LP benchmark output tree. Those model/dataset rows are `NOT_REUSABLE`.

See `results/lp_reused_reference.csv` for all 18 dataset/model decisions. No new LP runs were launched.
