from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.relcal_statepres_pilot import Model as S45Model


class ContextualRelationEncoder(nn.Module):
    """Encode a directed endpoint pair relative to its leave-one-out context."""

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
        pair = self.endpoint_pair(torch.cat((q_i * k_j, (q_i - k_j).abs()), dim=-1))
        return self.norm(pair + self.context(context_deviation))


class RelationalEvidenceExtractor(nn.Module):
    """Produce same- and cross-modal semantic evidence for directed edges."""

    def __init__(self, hidden_dim: int = 256, relation_dim: int = 32,
                 evidence_dim: int = 32):
        super().__init__()
        self.V_same = nn.Linear(hidden_dim, evidence_dim)
        self.V_cross = nn.Linear(hidden_dim, evidence_dim)
        self.G_same = nn.Linear(relation_dim, evidence_dim)
        self.G_cross = nn.Linear(relation_dim, evidence_dim)
        self.U_same = nn.Linear(evidence_dim, hidden_dim)
        self.U_cross = nn.Linear(evidence_dim, hidden_dim)
        for projection in (self.U_same, self.U_cross):
            nn.init.normal_(projection.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(projection.bias)

    def forward(self, relation: torch.Tensor, same_source: torch.Tensor,
                cross_source: torch.Tensor) -> dict[str, torch.Tensor]:
        u_same = F.gelu(self.V_same(same_source))
        u_cross = F.gelu(self.V_cross(cross_source))
        g_same = torch.tanh(self.G_same(relation))
        g_cross = torch.tanh(self.G_cross(relation))
        delta_same = self.U_same(g_same * u_same)
        delta_cross = self.U_cross(g_cross * u_cross)
        return {
            "u_same": u_same, "u_cross": u_cross,
            "g_same": g_same, "g_cross": g_cross,
            "delta_same": delta_same, "delta_cross": delta_cross,
        }


class IntrinsicGuidedReconciler(nn.Module):
    """Attend over three physical hop states, with an optional conditioned query."""

    def __init__(self, hidden_dim: int = 256, num_heads: int = 4,
                 dropout: float = 0.2, ff_mult: int = 2,
                 node_chunk_size: int = 32768):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
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

    def forward(self, h0: torch.Tensor,
                contexts: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                query: torch.Tensor | None = None,
                need_weights: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        # This is deliberately the v0.5 absorb-only computation when query is absent.
        anchor_query = h0 if query is None else query
        keys = torch.stack(
            [contexts[index] + self.hop_embeddings[index] for index in range(3)], dim=1
        )
        outputs: list[torch.Tensor] = []
        weight_chunks: list[torch.Tensor] = []
        for start in range(0, h0.size(0), self.node_chunk_size):
            end = min(start + self.node_chunk_size, h0.size(0))
            q = anchor_query[start:end].unsqueeze(1)
            key_value = keys[start:end]
            attended, weights = self.attention(
                q, key_value, key_value,
                need_weights=need_weights,
                average_attn_weights=False if need_weights else None,
            )
            attended = attended.squeeze(1)
            # The residual anchor remains intrinsic H0, including in full v0.6.
            outputs.append(self.norm(h0[start:end] + self.residual_dropout(self.ffn(attended))))
            if need_weights:
                weight_chunks.append(weights)
        result = torch.cat(outputs, dim=0) if outputs else h0.new_empty(h0.shape)
        weights_out = torch.cat(weight_chunks, dim=0) if weight_chunks else None
        return result, weights_out


class Model(nn.Module):
    """RISA v0.6: relation-specific evidence conditions intrinsic absorption."""

    VARIANTS = ("v06_full", "v06_no_relation_condition")
    requires_full_lp_sampler_depth = False

    def __init__(self, cfg, data_info):
        super().__init__()
        model_values = dict(cfg.model)
        self.hidden_dim = int(model_values.get("hidden_dim", 256))
        self.dropout_p = float(model_values.get("dropout", 0.2))
        self.max_order = int(model_values.get("max_order", 3))
        self.relation_dim = int(model_values.get("relation_dim", 32))
        self.evidence_dim = int(model_values.get("evidence_dim", 32))
        self.edge_chunk_size = int(model_values.get("edge_chunk_size", 16384))
        self.node_chunk_size = int(model_values.get("node_chunk_size", 32768))
        self.num_heads = int(model_values.get("num_heads", 4))
        self.ff_mult = int(model_values.get("ff_mult", 2))
        self.variant = str(model_values.get("variant", "v06_full")).strip().lower()
        if self.variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}, got {self.variant!r}")
        self.use_relation_condition = self.variant == "v06_full"
        if (self.hidden_dim, self.dropout_p, self.max_order, self.relation_dim,
                self.evidence_dim, self.num_heads, self.ff_mult) != (256, 0.2, 3, 32, 32, 4, 2):
            raise ValueError(
                "RISA v0.6 fixes hidden_dim=256, dropout=.2, max_order=3, "
                "relation_dim=evidence_dim=32, num_heads=4, ff_mult=2"
            )
        if min(self.edge_chunk_size, self.node_chunk_size) < 1:
            raise ValueError("edge_chunk_size and node_chunk_size must be positive")

        # Keep the v0.5-compatible projectors, physical P/P_self/P_rel builder,
        # and plain fusion as the shared substrate.
        substrate_cfg = SimpleNamespace(
            model={**model_values, "variant": "s45_identity_uniform"}
        )
        self.backbone = S45Model(substrate_cfg, data_info)
        self.text_dim = self.backbone.text_dim
        self.visual_dim = self.backbone.visual_dim
        self.input_dim = self.backbone.input_dim
        self.out_dim = self.backbone.out_dim

        if self.use_relation_condition:
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
            self.evidence_extractor_text = RelationalEvidenceExtractor(
                self.hidden_dim, self.relation_dim, self.evidence_dim,
            )
            self.evidence_extractor_visual = RelationalEvidenceExtractor(
                self.hidden_dim, self.relation_dim, self.evidence_dim,
            )
            self.W_rho_text = nn.Linear(self.relation_dim, self.hidden_dim)
            self.W_rho_visual = nn.Linear(self.relation_dim, self.hidden_dim)
            for projection in (self.W_rho_text, self.W_rho_visual):
                nn.init.normal_(projection.weight, mean=0.0, std=1e-3)
                nn.init.zeros_(projection.bias)

        # Same names and parameter shapes as the v0.5 intrinsic reconciler make
        # the no-relation variant directly mappable to v05_absorb_only.
        self.imci_text = IntrinsicGuidedReconciler(
            self.hidden_dim, self.num_heads, self.dropout_p,
            self.ff_mult, self.node_chunk_size,
        )
        self.imci_visual = IntrinsicGuidedReconciler(
            self.hidden_dim, self.num_heads, self.dropout_p,
            self.ff_mult, self.node_chunk_size,
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
        """Return H0_j minus its weighted context excluding j; zero at degree one."""
        if weighted_sum is None or row_mass is None:
            weighted_sum, row_mass = Model._context_sums(h, row, col, weight)
        other_sum = weighted_sum[row] - weight.unsqueeze(-1) * h[col]
        other_mass = row_mass[row] - weight
        valid = other_mass > eps
        safe_mass = torch.where(valid, other_mass, torch.ones_like(other_mass))
        deviation = h[col] - other_sum / safe_mass.unsqueeze(-1)
        deviation = torch.where(valid.unsqueeze(-1), deviation, torch.zeros_like(deviation))
        return deviation, valid

    @staticmethod
    def aggregate_edge_evidence(num_nodes: int, row: torch.Tensor,
                                weight: torch.Tensor,
                                delta: torch.Tensor) -> torch.Tensor:
        """Pool P_rel-weighted edge evidence with chunk-friendly index_add semantics."""
        aggregate = delta.new_zeros((num_nodes, delta.size(-1)))
        aggregate.index_add_(0, row, weight.unsqueeze(-1) * delta)
        return aggregate

    @staticmethod
    def pool_relation_context(num_nodes: int, row: torch.Tensor,
                              weight: torch.Tensor, relation: torch.Tensor,
                              row_mass: torch.Tensor,
                              eps: float = 1e-12) -> torch.Tensor:
        relation_sum = relation.new_zeros((num_nodes, relation.size(-1)))
        relation_sum.index_add_(0, row, weight.unsqueeze(-1) * relation)
        denominator = row_mass + eps
        pooled = relation_sum / denominator.unsqueeze(-1)
        return torch.where((row_mass > 0).unsqueeze(-1), pooled, torch.zeros_like(pooled))

    def _relation_condition(self, h_text: torch.Tensor, h_visual: torch.Tensor,
                            p_rel: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        row, col = p_rel.indices()
        weight = p_rel.values()
        num_nodes = h_text.size(0)
        row_mass = h_text.new_zeros(num_nodes)
        row_mass.index_add_(0, row, weight)
        states = {"text": h_text, "visual": h_visual}
        context_data = {name: self._context_sums(h, row, col, weight) for name, h in states.items()}
        projections = {
            name: getattr(self, f"relation_encoder_{name}").project(h)
            for name, h in states.items()
        }
        evidence_same = {name: h.new_zeros(h.shape) for name, h in states.items()}
        evidence_cross = {name: h.new_zeros(h.shape) for name, h in states.items()}
        rho_sum = {name: h.new_zeros((num_nodes, self.relation_dim)) for name, h in states.items()}
        other = {"text": "visual", "visual": "text"}

        for start in range(0, row.numel(), self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, row.numel())
            erow, ecol, eweight = row[start:end], col[start:end], weight[start:end]
            relation_states: dict[str, torch.Tensor] = {}
            for name, h in states.items():
                sums, masses = context_data[name]
                deviation, _ = self.leave_one_edge_out_context(
                    h, erow, ecol, eweight, sums, masses,
                )
                q, k = projections[name]
                relation_states[name] = getattr(self, f"relation_encoder_{name}")(
                    q[erow], k[ecol], deviation,
                )
            shared = self.shared_relation_encoder(torch.cat((
                relation_states["text"], relation_states["visual"],
                (relation_states["text"] - relation_states["visual"]).abs(),
            ), dim=-1))
            contextual = {
                "text": self.shared_norm_text(
                    relation_states["text"] + self.shared_to_text(shared)),
                "visual": self.shared_norm_visual(
                    relation_states["visual"] + self.shared_to_visual(shared)),
            }
            for target in ("text", "visual"):
                rho_sum[target].index_add_(
                    0, erow, eweight.unsqueeze(-1) * contextual[target],
                )
                extracted = getattr(self, f"evidence_extractor_{target}")(
                    contextual[target], states[target][ecol], states[other[target]][ecol],
                )
                evidence_same[target].index_add_(
                    0, erow, eweight.unsqueeze(-1) * extracted["delta_same"],
                )
                evidence_cross[target].index_add_(
                    0, erow, eweight.unsqueeze(-1) * extracted["delta_cross"],
                )

        rho = {}
        for name, h in states.items():
            pooled = rho_sum[name] / (row_mass + 1e-12).unsqueeze(-1)
            rho[name] = torch.where((row_mass > 0).unsqueeze(-1), pooled, torch.zeros_like(pooled))
        return {
            name: {
                "rho": rho[name], "E_same": evidence_same[name],
                "E_cross": evidence_cross[name],
                "E": evidence_same[name] + evidence_cross[name],
                "row_mass": row_mass,
            }
            for name in states
        }

    def analyze(self, x: torch.Tensor, edge_index: torch.Tensor,
                collect_attention: bool = True) -> dict[str, Any]:
        return self._compute(x, edge_index, collect_attention=collect_attention)

    def _compute(self, x: torch.Tensor, edge_index: torch.Tensor,
                 collect_attention: bool = False) -> dict[str, Any]:
        if edge_index is None or x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"expected physical edge_index and x [num_nodes, {self.input_dim}]")
        edge_index = edge_index.to(device=x.device, dtype=torch.long)
        p, p_self, p_rel = self.backbone._get_operators(edge_index, int(x.size(0)), x.dtype)
        h_text = self.backbone.text_projector(x[:, :self.text_dim])
        h_visual = self.backbone.visual_projector(
            x[:, self.text_dim:self.text_dim + self.visual_dim]
        )

        # Relation interpretation cannot modify this independent physical path.
        structural: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for name, h in (("text", h_text), ("visual", h_visual)):
            s1 = torch.sparse.mm(p, h)
            s2 = torch.sparse.mm(p, s1)
            s3 = torch.sparse.mm(p, s2)
            structural[name] = (s1, s2, s3)

        if self.use_relation_condition:
            relation = self._relation_condition(h_text, h_visual, p_rel)
        else:
            relation = {
                name: {
                    "rho": h.new_zeros((h.size(0), self.relation_dim)),
                    "E_same": torch.zeros_like(h), "E_cross": torch.zeros_like(h),
                    "E": torch.zeros_like(h),
                    "row_mass": h.new_zeros(h.size(0)),
                }
                for name, h in (("text", h_text), ("visual", h_visual))
            }

        outputs: dict[str, torch.Tensor] = {}
        queries: dict[str, torch.Tensor] = {}
        rho_shifts: dict[str, torch.Tensor] = {}
        attention_weights: dict[str, torch.Tensor | None] = {}
        contexts_by_modality = {"text": h_text, "visual": h_visual}
        for name, h0 in contexts_by_modality.items():
            if self.use_relation_condition:
                rho_shift = getattr(self, f"W_rho_{name}")(relation[name]["rho"])
                query = h0 + relation[name]["E"] + rho_shift
            else:
                rho_shift = torch.zeros_like(h0)
                query = h0
            z, weights = getattr(self, f"imci_{name}")(
                h0, structural[name], query=query, need_weights=collect_attention,
            )
            outputs[name] = z
            queries[name] = query
            rho_shifts[name] = rho_shift
            attention_weights[name] = weights

        fused = self.backbone.plain_fusion(torch.cat((outputs["text"], outputs["visual"]), dim=-1))
        mean_attention: dict[str, torch.Tensor | None] = {}
        attention_entropy: dict[str, torch.Tensor | None] = {}
        hop_attention_std: dict[str, torch.Tensor | None] = {}
        for name, weights in attention_weights.items():
            if weights is None:
                mean_attention[name] = attention_entropy[name] = hop_attention_std[name] = None
                continue
            hop_prob = weights.squeeze(-2)
            mean_attention[name] = hop_prob.mean(dim=(0, 1))
            probability = hop_prob.clamp_min(1e-12)
            attention_entropy[name] = -(probability * probability.log()).sum(-1).mean()
            hop_attention_std[name] = hop_prob.std(dim=(0, 1), unbiased=False)

        return {
            "P": p, "P_self": p_self, "P_rel": p_rel,
            "H0_text": h_text, "H0_visual": h_visual,
            "S1_text": structural["text"][0], "S2_text": structural["text"][1],
            "S3_text": structural["text"][2],
            "S1_visual": structural["visual"][0], "S2_visual": structural["visual"][1],
            "S3_visual": structural["visual"][2],
            "Z_text": outputs["text"], "Z_visual": outputs["visual"], "fused_z": fused,
            "rho_text": relation["text"]["rho"], "rho_visual": relation["visual"]["rho"],
            "E_same_text": relation["text"]["E_same"],
            "E_cross_text": relation["text"]["E_cross"], "E_text": relation["text"]["E"],
            "E_same_visual": relation["visual"]["E_same"],
            "E_cross_visual": relation["visual"]["E_cross"], "E_visual": relation["visual"]["E"],
            "rho_shift_text": rho_shifts["text"], "rho_shift_visual": rho_shifts["visual"],
            "Q_text": queries["text"], "Q_visual": queries["visual"],
            "query_shift_text": queries["text"] - h_text,
            "query_shift_visual": queries["visual"] - h_visual,
            "attention_weights": attention_weights,
            "mean_hop_attention": mean_attention,
            "hop_attention_entropy": attention_entropy,
            "hop_attention_std": hop_attention_std,
        }

    def forward(self, x: torch.Tensor, edge_index=None):
        result = self._compute(x, edge_index, collect_attention=False)
        z = result["fused_z"]
        return z, None, None, z.new_zeros(()), {}

    @torch.no_grad()
    def inference(self, x, edge_index=None, device=None, batch_size: int = 4096):
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        z, _, _, _, _ = self.forward(x.to(device), edge_index.to(device))
        return z.detach().cpu()
