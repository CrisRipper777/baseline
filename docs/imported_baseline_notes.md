# Imported baseline model notes

## Provenance and boundary

The three models below were copied from MoPF V3 (9f4514ed9eec4d14795a8aadbc125616880c3c9d) into the clean baseline branch. Their current V3 supervised adaptations are the implementation source. No MoPF, SSI, CoSI, or other self-developed research model, analysis implementation, ablation, mechanism, launcher, or historical result was imported.

The imported models return the baseline five-part forward interface (z, aux_1, aux_2, aux_loss, aux_info). They expose out_dim and an inference method. All use a zero scalar auxiliary loss under the unified supervised task code. The task framework applies the same validation-selected NC and sampled LP protocols used by all encoders.

## Mapping

| Model | Source files | Feature handling | Current architecture carried over | Excluded objectives |
|---|---|---|---|---|
| DGF | src/models/dgf.py, configs/model/dgf.yaml | Splits concatenated x at text_dim, then uses separate text and visual projections. | L2-normalized modality projections and mean fusion; symmetric-softmax feature shift; normalized physical adjacency with self-loops; truncated alpha and beta Neumann filtering core; out_dim=hidden_dim. | OpenMAG cross-modal contrastive, graph contrastive, community, and clustering losses. |
| DMGC | src/models/dmgc.py, configs/model/dmgc.yaml | Splits concatenated x at text_dim, then applies separate modality projections. | Shared GCN encoder over normalized adjacency and Laplacian views; per-modality low/high frequency fusion; learned cross-modal attention; out_dim=hidden_dim. | OpenMAG dual-frequency and cross-modal InfoNCE auxiliary losses. |
| LGMRec | src/models/lgmrec.py, configs/model/lgmrec.yaml | Consumes the concatenated feature matrix through one feature encoder; no manual text/visual slice in the model. | Linear feature encoder; configured LightGCN stack and mean over propagation states; hypergraph assignment/aggregation branch; local plus scaled normalized global representation; out_dim=hidden_dim. Evaluation uses deterministic softmax assignments. | Modality reconstruction heads/decoders and reconstruction/InfoNCE auxiliary objectives. |

The imported YAML configs preserve the current V3 model settings. Each declares the frozen model preset lr=5e-3, weight_decay=1e-5. Generic build_optimizer applies model-level lr and weight_decay overrides before task defaults, matching V3. Consequently these explicit model presets override the common NC/LP task learning rate and weight decay for these imports.

## Framework compatibility

- CPU and CUDA forward/inference tests passed for each model; outputs are finite and match full forward output within tolerance.
- Each model accepts an induced local graph and feature matrix in the same forward interface used by sampled link batches.
- DGF and DMGC require both modalities and slice by text_dim/visual_dim; LGMRec requires the already-concatenated input dimension.
- All three pass one-epoch full-graph Movies-NC smoke runs and tiny sampled sports-LP smoke runs using [5, 5, 5], LinkNeighborLoader, full-graph evaluation inference, and global_eid positive-message masking.
- The LP smoke confirmed nonzero positive message edges removed: DGF removed 4 across the exercised batches, DMGC removed 2, and LGMRec removed 4.
- Minimal interface fix: V3 DMGC inference projected features in chunks, stored them on CPU, then passed them to graph-encoder modules on CUDA. The inference method now places the graph operators and post-projection features on the requested device before encoding and still returns CPU embeddings. CPU and CUDA equivalence tests pass.

## Sampled LP interpretation limits

The LinkNeighborLoader interface runs for all three imports. Their architecture-specific computations operate on each sampled induced subgraph:

- DGF's feature-domain symmetric-softmax affinity is computed over the nodes in the current sampled batch. Its LP behavior therefore follows the current V3 sampled adaptation and is batch-context dependent.
- DMGC recomputes normalized-adjacency and Laplacian operators on each sampled subgraph.
- LGMRec computes its hypergraph assignments over the nodes in each sampled subgraph.

These are model properties of the V3 implementations, not framework failures. No model equations were silently changed to make them look like full-graph LP encoders. No formal imported-model benchmark was run.
