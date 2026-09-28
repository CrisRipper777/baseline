from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.relcal_statepres_pilot import Model as S45Model


class ContextualRelationEncoder(nn.Module):
    """Modality-specific endpoint evidence plus leave-one-edge-out context."""

    def __init__(self, hidden_dim: int = 256, relation_dim: int = 32):
        super().__init__()
        self.query = nn.Linear(hidden_dim, relation_dim, bias=False)
        self.key = nn.Linear(hidden_dim, relation_dim, bias=False)
        self.endpoint_pair = nn.Sequential(
            nn.Linear(2 * relation_dim, relation_dim),
            nn.GELU(),
            nn.Linear(relation_dim, relation_dim),
        )
        self.context = nn.Linear(hidden_dim, relation_dim, bias=False)
        self.norm = nn.LayerNorm(relation_dim)

    def project(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.query(h), self.key(h)

    def forward(self, q_i: torch.Tensor, k_j: torch.Tensor,
                context_deviation: torch.Tensor) -> torch.Tensor:
        endpoint = self.endpoint_pair(torch.cat((q_i * k_j, (q_i - k_j).abs()), dim=-1))
        return self.norm(endpoint + self.context(context_deviation))


class GroupedRotationTransform(nn.Module):
    """Apply an independent norm-preserving 2-D rotation to each feature group."""

    def __init__(self, hidden_dim: int = 256, group_size: int = 2,
                 relation_dim: int = 32, max_angle: float = 1.57079632679):
        super().__init__()
        if group_size != 2 or hidden_dim % group_size:
            raise ValueError("GroupedRotationTransform requires 2-D groups dividing hidden_dim")
        self.hidden_dim = int(hidden_dim)
        self.group_size = int(group_size)
        self.num_groups = self.hidden_dim // self.group_size
        self.max_angle = float(max_angle)
        self.angle_head = nn.Linear(relation_dim, self.num_groups)
        nn.init.normal_(self.angle_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.angle_head.bias)

    def angles(self, relation: torch.Tensor) -> torch.Tensor:
        return self.max_angle * torch.tanh(self.angle_head(relation))

    def rotate(self, message: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        grouped = message.reshape(*message.shape[:-1], self.num_groups, 2)
        a, b = grouped.unbind(dim=-1)
        cosine, sine = torch.cos(angle), torch.sin(angle)
        rotated = torch.stack((a * cosine - b * sine,
                               a * sine + b * cosine), dim=-1)
        return rotated.reshape_as(message)

    def forward(self, message: torch.Tensor,
                relation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        angle = self.angles(relation)
        return self.rotate(message, angle), angle


class IntrinsicGuidedReconciler(nn.Module):
    """Query multi-order context with intrinsic node features as the anchor."""

    def __init__(self, hidden_dim: int = 256, num_heads: int = 4,
                 dropout: float = 0.2, ff_mult: int = 2,
                 node_chunk_size: int = 32768):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by iamr_num_heads")
        self.hidden_dim = int(hidden_dim)
        self.node_chunk_size = int(node_chunk_size)
        self.hop_embeddings = nn.Parameter(torch.empty(3, hidden_dim))
        nn.init.normal_(self.hop_embeddings, mean=0.0, std=0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_mult * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * hidden_dim, hidden_dim),
        )
        self.residual_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h0: torch.Tensor, contexts: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                need_weights: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        keys = torch.stack(
            [contexts[index] + self.hop_embeddings[index] for index in range(3)], dim=1
        )
        outputs: list[torch.Tensor] = []
        weight_chunks: list[torch.Tensor] = []
        for start in range(0, h0.size(0), self.node_chunk_size):
            end = min(start + self.node_chunk_size, h0.size(0))
            query = h0[start:end].unsqueeze(1)
            key_value = keys[start:end]
            attended, weights = self.attention(
                query, key_value, key_value,
                need_weights=need_weights,
                average_attn_weights=False if need_weights else None,
            )
            attended = attended.squeeze(1)
            outputs.append(self.norm(h0[start:end] + self.residual_dropout(self.ffn(attended))))
            if need_weights:
                # PyTorch returns [batch, heads, query=1, key=3].
                weight_chunks.append(weights)
        result = torch.cat(outputs, dim=0) if outputs else h0.new_empty(h0.shape)
        weights_out = torch.cat(weight_chunks, dim=0) if weight_chunks else None
        return result, weights_out


class Model(nn.Module):
    """Model-v0.5: contextual relation transform and intrinsic hop integration."""

    VARIANTS = (
        "v05_full",
        "v05_no_crst",
        "v05_no_relation_context",
        "v05_no_imci",
    )
    requires_full_lp_sampler_depth = False

    def __init__(self, cfg, data_info):
        super().__init__()
        model_values = dict(cfg.model)
        self.hidden_dim = int(model_values.get("hidden_dim", 256))
        self.dropout_p = float(model_values.get("dropout", 0.2))
        self.max_order = int(model_values.get("max_order", 3))
        self.relation_dim = int(model_values.get("relation_dim", 32))
        self.edge_chunk_size = int(model_values.get("edge_chunk_size", 16384))
        self.rotation_group_size = int(model_values.get("rotation_group_size", 2))
        self.max_rotation_angle = float(model_values.get("max_rotation_angle", 1.57079632679))
        self.iamr_num_heads = int(model_values.get("iamr_num_heads", 4))
        self.iamr_ff_mult = int(model_values.get("iamr_ff_mult", 2))
        self.node_chunk_size = int(model_values.get("node_chunk_size", 32768))
        self.variant = str(model_values.get("variant", "v05_full")).strip().lower()
        if self.variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}, got {self.variant!r}")
        if (self.hidden_dim, self.dropout_p, self.max_order, self.relation_dim) != (256, 0.2, 3, 32):
            raise ValueError("RISA v0.5 fixes hidden_dim=256, dropout=.2, max_order=3, relation_dim=32")
        if self.rotation_group_size != 2 or self.hidden_dim % self.rotation_group_size:
            raise ValueError("RISA v0.5 fixes rotation_group_size=2")
        if min(self.edge_chunk_size, self.node_chunk_size, self.iamr_num_heads,
               self.iamr_ff_mult) < 1:
            raise ValueError("chunk sizes, iamr_num_heads, and iamr_ff_mult must be positive")

        # Reuse S4.5's unchanged projectors, physical P/P_self/P_rel builder,
        # and plain multimodal fusion. This substrate is identity-uniform.
        substrate_cfg = SimpleNamespace(
            model={**model_values, "variant": "s45_identity_uniform"}
        )
        self.backbone = S45Model(substrate_cfg, data_info)
        self.text_dim = self.backbone.text_dim
        self.visual_dim = self.backbone.visual_dim
        self.input_dim = self.backbone.input_dim
        self.out_dim = self.backbone.out_dim

        if self.variant != "v05_no_crst":
            self.relation_encoder_text = ContextualRelationEncoder(self.hidden_dim, self.relation_dim)
            self.relation_encoder_visual = ContextualRelationEncoder(self.hidden_dim, self.relation_dim)
            self.shared_relation_encoder = nn.Sequential(
                nn.Linear(3 * self.relation_dim, self.relation_dim),
                nn.GELU(),
                nn.Linear(self.relation_dim, self.relation_dim),
            )
            self.shared_to_text = nn.Linear(self.relation_dim, self.relation_dim, bias=False)
            self.shared_to_visual = nn.Linear(self.relation_dim, self.relation_dim, bias=False)
            self.shared_norm_text = nn.LayerNorm(self.relation_dim)
            self.shared_norm_visual = nn.LayerNorm(self.relation_dim)
            self.rotation_text = GroupedRotationTransform(
                self.hidden_dim, self.rotation_group_size, self.relation_dim,
                self.max_rotation_angle,
            )
            self.rotation_visual = GroupedRotationTransform(
                self.hidden_dim, self.rotation_group_size, self.relation_dim,
                self.max_rotation_angle,
            )

        if self.variant != "v05_no_imci":
            self.imci_text = IntrinsicGuidedReconciler(
                self.hidden_dim, self.iamr_num_heads, self.dropout_p,
                self.iamr_ff_mult, self.node_chunk_size,
            )
            self.imci_visual = IntrinsicGuidedReconciler(
                self.hidden_dim, self.iamr_num_heads, self.dropout_p,
                self.iamr_ff_mult, self.node_chunk_size,
            )

    @staticmethod
    def _context_sums(h: torch.Tensor, row: torch.Tensor, col: torch.Tensor,
                      weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weighted_sum = h.new_zeros(h.shape)
        weighted_sum.index_add_(0, row, weight.unsqueeze(-1) * h[col])
        row_mass = h.new_zeros(h.size(0))
        row_mass.index_add_(0, row, weight)
        return weighted_sum, row_mass

    @staticmethod
    def leave_one_edge_out_context(h: torch.Tensor, row: torch.Tensor,
                                   col: torch.Tensor, weight: torch.Tensor,
                                   weighted_sum: torch.Tensor | None = None,
                                   row_mass: torch.Tensor | None = None,
                                   eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
        """Return H0_j - mean(neighbors(i) excluding j), zero for degree one."""
        if weighted_sum is None or row_mass is None:
            weighted_sum, row_mass = Model._context_sums(h, row, col, weight)
        other_sum = weighted_sum[row] - weight.unsqueeze(-1) * h[col]
        other_mass = row_mass[row] - weight
        valid = other_mass > eps
        safe_mass = torch.where(valid, other_mass, torch.ones_like(other_mass))
        mean_without_edge = other_sum / safe_mass.unsqueeze(-1)
        deviation = h[col] - mean_without_edge
        deviation = torch.where(valid.unsqueeze(-1), deviation, torch.zeros_like(deviation))
        return deviation, valid

    @staticmethod
    def _select_diagnostic_edges(row: torch.Tensor, num_nodes: int,
                                 target_nodes: torch.Tensor | None,
                                 edge_subset: torch.Tensor | None,
                                 max_edges: int, sample_seed: int) -> torch.Tensor:
        allowed = torch.ones(row.numel(), dtype=torch.bool, device=row.device)
        if target_nodes is not None:
            target_mask = torch.zeros(num_nodes, dtype=torch.bool, device=row.device)
            target_mask[target_nodes.to(device=row.device, dtype=torch.long)] = True
            allowed &= target_mask[row]
        if edge_subset is not None:
            supplied = edge_subset.to(device=row.device)
            if supplied.dtype == torch.bool:
                if supplied.numel() != row.numel():
                    raise ValueError("boolean edge_subset must have one value per P_rel edge")
                allowed &= supplied
            else:
                selected = torch.zeros(row.numel(), dtype=torch.bool, device=row.device)
                indices = supplied.long().reshape(-1)
                if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= row.numel()):
                    raise ValueError("edge_subset contains an out-of-range edge index")
                selected[indices] = True
                allowed &= selected
        candidates = torch.nonzero(allowed, as_tuple=False).flatten()
        if edge_subset is not None or candidates.numel() <= max_edges:
            return candidates
        generator = torch.Generator(device="cpu").manual_seed(int(sample_seed))
        sample_order = torch.randperm(candidates.numel(), generator=generator)[:max_edges]
        return candidates.index_select(0, sample_order.to(candidates.device))

    @staticmethod
    def _empty_edge_state(h: torch.Tensor, relation_dim: int, num_groups: int) -> dict[str, torch.Tensor]:
        return {
            "relation_state": h.new_empty((0, relation_dim)),
            "shared_relation_state": h.new_empty((0, relation_dim)),
            "rotation_angles": h.new_empty((0, num_groups)),
            "base_message": h.new_empty((0, h.size(-1))),
            "rotated_message": h.new_empty((0, h.size(-1))),
            "context_deviation": h.new_empty((0, h.size(-1))),
        }

    def _crst_propagate(self, h_text: torch.Tensor, h_visual: torch.Tensor,
                        p: torch.Tensor, p_self: torch.Tensor, p_rel: torch.Tensor,
                        selected: torch.Tensor, collect_edge_state: bool,
                        collect_diagnostics: bool) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        row, col = p_rel.indices()
        weight = p_rel.values()
        states_h = {"text": h_text, "visual": h_visual}
        aggregates = {m: torch.sparse.mm(p_self, h) for m, h in states_h.items()}
        diagnostic: dict[str, Any] = {m: {} for m in ("text", "visual")}
        edge_lists: dict[str, dict[str, list[torch.Tensor]]] = {
            m: {k: [] for k in ("relation_state", "shared_relation_state", "rotation_angles",
                               "base_message", "rotated_message", "context_deviation")}
            for m in ("text", "visual")
        }
        if self.variant == "v05_no_crst":
            c1 = {m: torch.sparse.mm(p, h) for m, h in states_h.items()}
            empty_stats = {
                "mean_abs_angle": 0.0, "p95_abs_angle": 0.0,
                "cos_base_rotated": 1.0, "norm_preservation_max_abs_error": 0.0,
                "valid_context_edge_ratio": 0.0,
            }
            return c1, {m: {**empty_stats, "edge_state": self._empty_edge_state(
                states_h[m], self.relation_dim, self.hidden_dim // 2)} for m in states_h}

        context_data = {}
        projections = {}
        for modality, h in states_h.items():
            sums, masses = self._context_sums(h, row, col, weight)
            encoder = getattr(self, f"relation_encoder_{modality}")
            q, k = encoder.project(h)
            projections[modality] = (q, k)
            context_data[modality] = (sums, masses)

        angle_abs_sum = {m: torch.zeros((), dtype=torch.float64, device=h_text.device)
                         for m in states_h}
        angle_value_count = 0
        norm_error_max = {m: 0.0 for m in states_h}
        cosine_sum = {m: torch.zeros((), dtype=torch.float64, device=h_text.device)
                      for m in states_h}
        edge_count = 0
        valid_context_count = 0
        sampled_angle_values: dict[str, list[torch.Tensor]] = {m: [] for m in states_h}

        for start in range(0, row.numel(), self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, row.numel())
            erow, ecol, eweight = row[start:end], col[start:end], weight[start:end]
            contexts: dict[str, torch.Tensor] = {}
            valid_mask = None
            for modality, h in states_h.items():
                sums, masses = context_data[modality]
                context, valid = self.leave_one_edge_out_context(
                    h, erow, ecol, eweight, sums, masses,
                )
                if self.variant == "v05_no_relation_context":
                    context = torch.zeros_like(context)
                contexts[modality] = context
                valid_mask = valid

            relation_states: dict[str, torch.Tensor] = {}
            for modality in states_h:
                q, k = projections[modality]
                encoder = getattr(self, f"relation_encoder_{modality}")
                relation_states[modality] = encoder(q[erow], k[ecol], contexts[modality])
            shared = self.shared_relation_encoder(torch.cat((
                relation_states["text"], relation_states["visual"],
                (relation_states["text"] - relation_states["visual"]).abs(),
            ), dim=-1))
            modality_relations = {
                "text": self.shared_norm_text(
                    relation_states["text"] + self.shared_to_text(shared)),
                "visual": self.shared_norm_visual(
                    relation_states["visual"] + self.shared_to_visual(shared)),
            }
            for modality, h in states_h.items():
                base = h[ecol]
                rotated, angle = getattr(self, f"rotation_{modality}")(base, modality_relations[modality])
                aggregates[modality].index_add_(
                    0, erow, eweight.unsqueeze(-1) * rotated,
                )
                if collect_diagnostics:
                    angle_abs_sum[modality] += angle.detach().double().abs().sum()
                    angle_value_count += 0 if modality != "text" else angle.numel()
                    norm_error = (base.float().norm(dim=-1) - rotated.float().norm(dim=-1)).abs()
                    norm_error_max[modality] = max(norm_error_max[modality],
                                                   float(norm_error.max().detach().cpu()) if norm_error.numel() else 0.0)
                    cosine_sum[modality] += F.cosine_similarity(
                        base.float(), rotated.float(), dim=-1, eps=1e-12
                    ).double().sum()
                chosen = selected[start:end]
                if collect_edge_state and bool(chosen.any()):
                    edge_lists[modality]["relation_state"].append(relation_states[modality][chosen])
                    edge_lists[modality]["shared_relation_state"].append(shared[chosen])
                    edge_lists[modality]["rotation_angles"].append(angle[chosen])
                    edge_lists[modality]["base_message"].append(base[chosen])
                    edge_lists[modality]["rotated_message"].append(rotated[chosen])
                    edge_lists[modality]["context_deviation"].append(contexts[modality][chosen])
                if collect_diagnostics and bool(chosen.any()):
                    sampled_angle_values[modality].append(angle[chosen].detach().abs().reshape(-1))
            if collect_diagnostics:
                edge_count += end - start
                valid_context_count += int(valid_mask.sum().detach().cpu()) if valid_mask is not None else 0

        c1 = aggregates
        for modality in states_h:
            sampled = (torch.cat(sampled_angle_values[modality]) if sampled_angle_values[modality]
                       else states_h[modality].new_empty((0,)))
            edge_state = {}
            for key, parts in edge_lists[modality].items():
                if collect_edge_state and parts:
                    edge_state[key] = torch.cat(parts, dim=0)
                else:
                    edge_state[key] = self._empty_edge_state(
                        states_h[modality], self.relation_dim,
                        self.hidden_dim // self.rotation_group_size,
                    )[key]
            diagnostic[modality] = {
                "mean_abs_angle": float((angle_abs_sum[modality] / max(edge_count * (self.hidden_dim // 2), 1)).cpu())
                if collect_diagnostics else 0.0,
                "p95_abs_angle": float(torch.quantile(sampled.float(), 0.95).cpu()) if sampled.numel() else 0.0,
                "cos_base_rotated": float((cosine_sum[modality] / max(edge_count, 1)).cpu())
                if collect_diagnostics else 1.0,
                "norm_preservation_max_abs_error": norm_error_max[modality],
                "valid_context_edge_ratio": valid_context_count / max(edge_count, 1)
                if collect_diagnostics else 0.0,
                "edge_state": edge_state,
            }
        return c1, diagnostic

    def _integrate(self, modality: str, h0: torch.Tensor,
                   contexts: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                   need_weights: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.variant == "v05_no_imci":
            return (h0 + contexts[0] + contexts[1] + contexts[2]) / 4.0, None
        return getattr(self, f"imci_{modality}")(h0, contexts, need_weights=need_weights)

    def analyze(self, x: torch.Tensor, edge_index: torch.Tensor,
                target_nodes: torch.Tensor | None = None,
                edge_subset: torch.Tensor | None = None,
                sample_seed: int = 42,
                max_diagnostic_edges: int = 512,
                collect_edge_state: bool = True,
                collect_attention: bool = True) -> dict[str, Any]:
        return self._compute(
            x, edge_index, target_nodes=target_nodes, edge_subset=edge_subset,
            sample_seed=sample_seed, max_diagnostic_edges=max_diagnostic_edges,
            collect_edge_state=collect_edge_state, collect_diagnostics=True,
            collect_attention=collect_attention,
        )

    def _compute(self, x: torch.Tensor, edge_index: torch.Tensor,
                 target_nodes: torch.Tensor | None = None,
                 edge_subset: torch.Tensor | None = None,
                 sample_seed: int = 42, max_diagnostic_edges: int = 512,
                 collect_edge_state: bool = False, collect_diagnostics: bool = False,
                 collect_attention: bool = False) -> dict[str, Any]:
        if edge_index is None or x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"expected physical edge_index and x [num_nodes, {self.input_dim}]")
        edge_index = edge_index.to(device=x.device, dtype=torch.long)
        p, p_self, p_rel = self.backbone._get_operators(edge_index, int(x.size(0)), x.dtype)
        # P_rel is deliberately only read below. Its indices and values are never
        # passed through a learned relation-dependent reweighting operation.
        row, col = p_rel.indices()
        edge_weight = p_rel.values()
        if collect_edge_state:
            selected_ids = self._select_diagnostic_edges(
                row, x.size(0), target_nodes, edge_subset,
                max_diagnostic_edges, sample_seed,
            )
            selected_mask = torch.zeros(row.numel(), dtype=torch.bool, device=row.device)
            selected_mask[selected_ids] = True
        else:
            selected_ids = torch.empty((0,), dtype=torch.long, device=row.device)
            selected_mask = torch.zeros(row.numel(), dtype=torch.bool, device=row.device)

        h_text = self.backbone.text_projector(x[:, :self.text_dim])
        h_visual = self.backbone.visual_projector(
            x[:, self.text_dim:self.text_dim + self.visual_dim]
        )
        c1, crst_stats = self._crst_propagate(
            h_text, h_visual, p, p_self, p_rel, selected_mask,
            collect_edge_state, collect_diagnostics,
        )
        c2 = {m: torch.sparse.mm(p, c1[m]) for m in ("text", "visual")}
        c3 = {m: torch.sparse.mm(p, c2[m]) for m in ("text", "visual")}
        z_text, attn_text = self._integrate(
            "text", h_text, (c1["text"], c2["text"], c3["text"]), collect_attention,
        )
        z_visual, attn_visual = self._integrate(
            "visual", h_visual, (c1["visual"], c2["visual"], c3["visual"]), collect_attention,
        )
        fused = self.backbone.plain_fusion(torch.cat((z_text, z_visual), dim=-1))

        attention_weights = {"text": attn_text, "visual": attn_visual}
        mean_attention: dict[str, torch.Tensor | None] = {}
        attention_entropy: dict[str, torch.Tensor | None] = {}
        for modality, weights in attention_weights.items():
            if weights is None:
                mean_attention[modality] = None
                attention_entropy[modality] = None
            else:
                # [N, heads, query=1, hops=3] -> mean over N and heads.
                hop_prob = weights.squeeze(-2)
                mean_attention[modality] = hop_prob.mean(dim=(0, 1))
                entropy = -(hop_prob.clamp_min(1e-12) * hop_prob.clamp_min(1e-12).log()).sum(-1)
                attention_entropy[modality] = entropy.mean()

        edge_states = {m: crst_stats[m]["edge_state"] for m in ("text", "visual")}
        result: dict[str, Any] = {
            "P": p, "P_self": p_self, "P_rel": p_rel,
            "H0_text": h_text, "H0_visual": h_visual,
            "C1_text": c1["text"], "C2_text": c2["text"], "C3_text": c3["text"],
            "C1_visual": c1["visual"], "C2_visual": c2["visual"], "C3_visual": c3["visual"],
            "C_text": [h_text, c1["text"], c2["text"], c3["text"]],
            "C_visual": [h_visual, c1["visual"], c2["visual"], c3["visual"]],
            "Z_text": z_text, "Z_visual": z_visual, "fused_z": fused,
            "edge_indices": selected_ids,
            "edge_row": row[selected_ids], "edge_col": col[selected_ids],
            "edge_weight": edge_weight[selected_ids],
            "relation_states": {m: edge_states[m]["relation_state"] for m in edge_states},
            "shared_relation_state": {m: edge_states[m]["shared_relation_state"] for m in edge_states},
            "rotation_angles": {m: edge_states[m]["rotation_angles"] for m in edge_states},
            "base_edge_message": {m: edge_states[m]["base_message"] for m in edge_states},
            "rotated_edge_message": {m: edge_states[m]["rotated_message"] for m in edge_states},
            "context_deviation": {m: edge_states[m]["context_deviation"] for m in edge_states},
            "mean_abs_angle": {m: crst_stats[m]["mean_abs_angle"] for m in crst_stats},
            "p95_abs_angle": {m: crst_stats[m]["p95_abs_angle"] for m in crst_stats},
            "cos_base_rotated": {m: crst_stats[m]["cos_base_rotated"] for m in crst_stats},
            "norm_preservation_max_abs_error": {
                m: crst_stats[m]["norm_preservation_max_abs_error"] for m in crst_stats
            },
            "valid_context_edge_ratio": {
                m: crst_stats[m]["valid_context_edge_ratio"] for m in crst_stats
            },
            "attention_weights": attention_weights,
            "mean_hop_attention": mean_attention,
            "hop_attention_entropy": attention_entropy,
        }
        return result

    def forward(self, x: torch.Tensor, edge_index=None):
        # NC consumes only fused_z; full edge diagnostics and attention matrices
        # are reserved for analyze() and are disabled during gradient training.
        result = self._compute(x, edge_index, collect_edge_state=False,
                               collect_diagnostics=False, collect_attention=False)
        z = result["fused_z"]
        return z, None, None, z.new_zeros(()), {}

    @torch.no_grad()
    def inference(self, x, edge_index=None, device=None, batch_size: int = 4096):
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        z, _, _, _, _ = self.forward(x.to(device), edge_index.to(device))
        return z.detach().cpu()
