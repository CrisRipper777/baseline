# D3 analysis and execution specification

D3 is an analysis-only continuation from baseline commit `d31b095faa4f98538e60cfac293a831138dde3db`. It evaluates NC on Movies, Toys, Grocery, ele-fashion, and Reddit-S. It does not design a research model or make a novelty/contribution claim. Existing benchmark and mechanism-discovery checkpoints/results are read-only. New outputs belong under `outputs/problem_deep_dive_v1/` and `results/problem_deep_dive_v1/`.

## Training controls

- Operator transfer: GCN/SAGE × `deep_only`/`anchored25` × five datasets × three internal seeds = 60 runs.
- Topology control: `multi_order_bank`/`uniform` on five degree-preserving rewires × three internal seeds = 15 runs.
- Every formal command uses `seed=42 num_runs=3`; the framework creates seeds 42, 43, and 44 inside that command.
- The launcher is `scripts/run_problem_deep_dive_nc.py`; it uses the dynamic GPU scheduler and holds at most one job on each listed GPU.
- Preflight is five ele-fashion one-epoch jobs: four operator readouts plus one rewired uniform control. A failed preflight stops formal training.

Commands after code and provenance are committed:

```bash
python scripts/run_problem_deep_dive_nc.py preflight --gpus 0,1
python scripts/run_problem_deep_dive_nc.py operator --gpus 0,1
python scripts/run_problem_deep_dive_nc.py rewired --gpus 0,1
```

## Frozen definitions

Each modality has its own feature projector and three graph layers. Text and Visual do not share propagation states. The only readouts are `deep_only = H3` and `anchored25 = 0.25 H0 + 0.75 H3`; both use the existing `plain_mlp` late fusion. GCN normalization/self-loops and GCN/SAGE layer norms, activation, and dropout follow the project model configs.

Edge compatibility reuses the D2 deterministic source-matched non-edge percentile definition and seed `run_seed * 101 + 7`. Novelty is one minus cosine similarity between an edge neighbor and the receiving endpoint's mean final-layer neighbor representation after excluding that edge; it is NaN where endpoint degree is at most one. Similarity, novelty, and embeddedness quartiles are fit only on physical edges whose two endpoints are in Train. The resulting fixed thresholds define edge groups across the graph. Same-label edges are computed only when both endpoints are in Train or Validation and remain descriptive-only.

`FIXED_NORM_MASK` zeros both normalized directions for selected physical pairs while preserving every other normalized weight and self-loop. `RENORMALIZED_DELETE` removes both pair directions and rebuilds the symmetric normalized operator. Both are frozen-checkpoint diagnostics. They do not retrain the model. Each targeted edge group gets 20 deterministic controls matching edge count and the endpoint degree-quantile-pair and original normalized edge-weight-quantile marginals exactly. An infeasible allocation is recorded as `MATCHING_FAILED`; no ordinary random fallback is substituted.

The node utility targets are `loss(self_only)-loss(uniform_plain)`, `loss(text_self)-loss(text_uniform)`, and `loss(visual_self)-loss(visual_uniform)`. Ridge and logistic probes fit on Train only and report on Validation only. Selector feature lists contain confidence, entropy, Jensen-Shannon disagreement, log degree, compatibility, novelty, or modality disagreement. They never contain labels, same-label edges, or homophily.

## Gates

- Gate A is `STRONG_SUPPORT` only when (1) at least three datasets have positive low-similarity/high-novelty fixed-mask utility beyond the exact matched-control mean, (2) novelty-only validation prediction for `G_struct` exceeds similarity-only in at least three datasets, and (3) text/visual fixed-mask utility means differ by more than `1e-4` in at least three datasets. Otherwise it is `MIXED` or `UNSUPPORTED`.
- Gate B/C is `STRONG_SUPPORT` only when the train-only logistic selector gains at least 0.003 validation accuracy over the strongest non-learned simple reference selected by Train accuracy among static expert choice, confidence, entropy, and uniform_plain, has a positive mean gain in at least three datasets, and oracle headroom exceeds that gain by at least 0.003. The report includes all static, confidence, entropy, uniform, and oracle references.
- Anchoring transfer is supported only when both GCN and SAGE have positive five-dataset mean `anchored25 - deep_only` Validation Accuracy and at least three positive dataset means each.
- Test metrics in newly retrained formal control checkpoints are descriptive only. No gate, threshold, ranking, grouping, or selector uses Test.

## Analysis outputs

`analyze_problem_deep_dive.py` has `cache`, `selectors`, `interventions`, `operator`, `topology`, and `finalize` phases. The outputs include paired operator transfer, train-thresholded edge diagnostics, semantic/novelty/structure interventions, exact-matching diagnostics, raw gzip node effects with hashes, train-only utility-predictability probes, selector/oracle comparisons, topology reality check, and the final gate summary/report. Raw node tables are stored under `outputs/problem_deep_dive_v1/node_intervention_effects/` and are not committed.
