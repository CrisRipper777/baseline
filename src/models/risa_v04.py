from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.relcal_statepres_pilot import Model as S45Model


class FactorizedRelationEncoder(nn.Module):
    """Compact, independent-modality factorized-bilinear edge encoder."""

    def __init__(self, hidden_dim: int = 256, relation_dim: int = 32):
        super().__init__()
        self.query = nn.Linear(hidden_dim, relation_dim, bias=False)
        self.key = nn.Linear(hidden_dim, relation_dim, bias=False)
        self.bilinear_query = nn.Linear(relation_dim, relation_dim, bias=False)
        self.bilinear_key = nn.Linear(relation_dim, relation_dim, bias=False)
        self.pair = nn.Sequential(
            nn.Linear(4 * relation_dim, relation_dim),
            nn.GELU(),
            nn.Linear(relation_dim, relation_dim),
        )

    def project(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.query(h), self.key(h)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        factored_bilinear = self.bilinear_query(q) * self.bilinear_key(k)
        values = torch.cat((q, k, (q - k).abs(), factored_bilinear), dim=-1)
        return self.pair(values)


class BottleneckResidualOperator(nn.Module):
    """A residual message correction with a narrow semantic bottleneck."""

    def __init__(self, hidden_dim: int = 256, bottleneck: int = 32):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck, hidden_dim)
        # Keep the initial propagation close to the identity carrier while
        # retaining gradients through every adapter from the first update.
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.up.bias)

    def bottleneck_state(self, message: torch.Tensor) -> torch.Tensor:
        return self.activation(self.down(message))

    def forward(self, message: torch.Tensor) -> torch.Tensor:
        return self.up(self.bottleneck_state(message))


class Model(nn.Module):
    """Model-v0.4 P0: relation-conditioned operators on the S45 identity substrate."""

    VARIANTS = (
        "p0_identity",
        "p0_masspres_scalar",
        "p0_single_dynamic_transform",
        "p0_operator_uniform",
        "p0_operator_routed",
    )
    OPERATOR_VARIANTS = {"p0_operator_uniform", "p0_operator_routed"}
    RELATION_VARIANTS = {
        "p0_single_dynamic_transform", "p0_operator_routed",
    }
    NUM_OPERATORS = 4
    requires_full_lp_sampler_depth = False

    def __init__(self, cfg, data_info):
        super().__init__()
        model_values = dict(cfg.model)
        self.hidden_dim = int(model_values.get("hidden_dim", 256))
        self.dropout_p = float(model_values.get("dropout", 0.2))
        self.max_order = int(model_values.get("max_order", 3))
        self.relation_dim = int(model_values.get("relation_dim", 32))
        self.bottleneck_dim = int(model_values.get("bottleneck_dim", 32))
        self.num_operators = int(model_values.get("num_operators", self.NUM_OPERATORS))
        self.edge_chunk_size = int(model_values.get("edge_chunk_size", 16384))
        self.variant = str(model_values.get("variant", "p0_operator_routed")).strip().lower()
        if self.variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}, got {self.variant!r}")
        if (self.hidden_dim, self.dropout_p, self.max_order) != (256, 0.2, 3):
            raise ValueError("RISA v0.4 P0 fixes hidden_dim=256, dropout=0.2, max_order=3")
        if self.relation_dim < 1 or self.bottleneck_dim < 1 or self.edge_chunk_size < 1:
            raise ValueError("relation_dim, bottleneck_dim, and edge_chunk_size must be positive")
        if self.variant in self.OPERATOR_VARIANTS and self.num_operators != self.NUM_OPERATORS:
            raise ValueError("operator variants fix num_operators=4")

        # The substrate owns the unchanged projectors, plain fusion, and the
        # exact physical P/P_self/P_rel construction used by S4.5.
        substrate_variant = (
            "s45_masspres_entry_uniform"
            if self.variant == "p0_masspres_scalar"
            else "s45_identity_uniform"
        )
        substrate_cfg = SimpleNamespace(model={**model_values, "variant": substrate_variant})
        self.backbone = S45Model(substrate_cfg, data_info)
        self.text_dim = self.backbone.text_dim
        self.visual_dim = self.backbone.visual_dim
        self.input_dim = self.backbone.input_dim
        self.out_dim = self.backbone.out_dim

        # Instantiate adapters before relation encoders so uniform and routed
        # variants receive identical adapter initializations under the same seed.
        if self.variant in self.OPERATOR_VARIANTS:
            self.operators_text = nn.ModuleList(
                BottleneckResidualOperator(self.hidden_dim, self.bottleneck_dim)
                for _ in range(self.NUM_OPERATORS)
            )
            self.operators_visual = nn.ModuleList(
                BottleneckResidualOperator(self.hidden_dim, self.bottleneck_dim)
                for _ in range(self.NUM_OPERATORS)
            )
        if self.variant in self.RELATION_VARIANTS:
            self.relation_encoder_text = FactorizedRelationEncoder(self.hidden_dim, self.relation_dim)
            self.relation_encoder_visual = FactorizedRelationEncoder(self.hidden_dim, self.relation_dim)
        if self.variant == "p0_single_dynamic_transform":
            self.dynamic_text = BottleneckResidualOperator(self.hidden_dim, self.bottleneck_dim)
            self.dynamic_visual = BottleneckResidualOperator(self.hidden_dim, self.bottleneck_dim)
            self.condition_text = nn.Linear(self.relation_dim, self.bottleneck_dim)
            self.condition_visual = nn.Linear(self.relation_dim, self.bottleneck_dim)
            for layer in (self.condition_text, self.condition_visual):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        elif self.variant == "p0_operator_routed":
            self.router_text = nn.Linear(self.relation_dim, self.NUM_OPERATORS)
            self.router_visual = nn.Linear(self.relation_dim, self.NUM_OPERATORS)
            # Equal operator contribution at initialization.
            for router in (self.router_text, self.router_visual):
                nn.init.zeros_(router.weight)
                nn.init.zeros_(router.bias)

    @staticmethod
    def _cosine_summary(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(left.float(), right.float(), dim=-1, eps=1e-12)

    def _select_edge_mask(self, row: torch.Tensor, num_nodes: int,
                          target_nodes: torch.Tensor | None,
                          edge_subset: torch.Tensor | None) -> torch.Tensor:
        selected = torch.ones(row.numel(), dtype=torch.bool, device=row.device)
        if target_nodes is not None:
            target_mask = torch.zeros(num_nodes, dtype=torch.bool, device=row.device)
            target_mask[target_nodes.to(device=row.device, dtype=torch.long)] = True
            selected &= target_mask[row]
        if edge_subset is not None:
            subset = edge_subset.to(device=row.device)
            if subset.dtype == torch.bool:
                if subset.numel() != row.numel():
                    raise ValueError("boolean edge_subset must have one entry per P_rel edge")
                selected &= subset
            else:
                subset_mask = torch.zeros(row.numel(), dtype=torch.bool, device=row.device)
                indices = subset.long().reshape(-1)
                if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= row.numel()):
                    raise ValueError("edge_subset contains an out-of-range P_rel edge index")
                subset_mask[indices] = True
                selected &= subset_mask
        return selected

    def analyze(self, x: torch.Tensor, edge_index: torch.Tensor,
                intervention: str | None = None,
                target_nodes: torch.Tensor | None = None,
                edge_subset: torch.Tensor | None = None,
                router_shuffle_permutation=None,
                shuffle_seed: int | None = None,
                return_edge_state: bool = True) -> dict[str, Any]:
        if intervention not in {None, "router_uniform", "router_shuffle", "delta_off",
                                "delta_parallel_only", "delta_orthogonal_only"}:
            raise ValueError(f"unknown RISA P0 intervention {intervention!r}")
        if edge_index is None or x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"expected physical edge_index and x [num_nodes, {self.input_dim}]")
        edge_index = edge_index.to(device=x.device, dtype=torch.long)

        if self.variant == "p0_identity":
            base = self.backbone.analyze(x, edge_index)
            return self._format_substrate_result(base, "identity", target_nodes,
                                                 edge_subset, return_edge_state)
        if self.variant == "p0_masspres_scalar":
            gate_override = "off" if intervention == "delta_off" else None
            base = self.backbone.analyze(x, edge_index, gate_override=gate_override)
            return self._format_substrate_result(base, "masspres_scalar", target_nodes,
                                                 edge_subset, return_edge_state)
        if intervention == "delta_off":
            # Delegate to the exact S45 identity-uniform path so disabling the
            # correction recovers the historical substrate operation-for-operation.
            base = self.backbone.analyze(x, edge_index)
            return self._format_substrate_result(base, "identity", target_nodes,
                                                 edge_subset, return_edge_state)

        p, p_self, p_rel = self.backbone._get_operators(edge_index, int(x.size(0)), x.dtype)
        row, col = p_rel.indices()
        weight = p_rel.values()
        selected = self._select_edge_mask(row, x.size(0), target_nodes, edge_subset)
        h_text = self.backbone.text_projector(x[:, :self.text_dim])
        h_visual = self.backbone.visual_projector(x[:, self.text_dim:self.text_dim + self.visual_dim])
        states: dict[str, list[torch.Tensor]] = {}
        edge_data: dict[str, dict[str, Any]] = {}
        readouts: dict[str, torch.Tensor] = {}
        for modality, h in (("text", h_text), ("visual", h_visual)):
            if intervention == "delta_off":
                edge_result = self._empty_edge_result(h, row, col, selected,
                                                      return_edge_state)
                correction = h.new_zeros(h.shape)
            elif self.variant == "p0_single_dynamic_transform":
                edge_result = self._dynamic_edges(
                    modality, h, row, col, weight, intervention, selected, return_edge_state,
                )
                correction = edge_result["correction"]
            else:
                edge_result = self._routed_edges(
                    modality, h, row, col, weight, intervention,
                    router_shuffle_permutation, shuffle_seed, selected, return_edge_state,
                )
                correction = edge_result["correction"]
            c0 = h
            # C1 is the unchanged P propagation plus a correction to only the
            # off-diagonal P_rel message representation. P_rel values are read
            # but never rewritten by the relation encoder or router.
            c1 = torch.sparse.mm(p, c0) + correction
            c2 = torch.sparse.mm(p, c1)
            c3 = torch.sparse.mm(p, c2)
            states[modality] = [c0, c1, c2, c3]
            readouts[modality] = (c0 + c1 + c2 + c3) / 4.0
            edge_data[modality] = edge_result
        fused = self.backbone.plain_fusion(torch.cat((readouts["text"], readouts["visual"]), dim=-1))
        return {
            "P": p, "P_self": p_self, "P_rel": p_rel,
            "H0_text": h_text, "H0_visual": h_visual,
            "C1_text": states["text"][1], "C2_text": states["text"][2], "C3_text": states["text"][3],
            "C1_visual": states["visual"][1], "C2_visual": states["visual"][2], "C3_visual": states["visual"][3],
            "C_text": states["text"], "C_visual": states["visual"],
            "S_text": states["text"], "S_visual": states["visual"],
            "Z_text": readouts["text"], "Z_visual": readouts["visual"], "fused_z": fused,
            "router_probabilities": {m: edge_data[m]["router_probabilities"] for m in ("text", "visual")},
            "relation_representation": {m: edge_data[m]["relation_representation"] for m in ("text", "visual")},
            "base_edge_message": {m: edge_data[m]["base_edge_message"] for m in ("text", "visual")},
            "operator_correction_delta": {m: edge_data[m]["operator_correction_delta"] for m in ("text", "visual")},
            "corrected_edge_message": {m: edge_data[m]["corrected_edge_message"] for m in ("text", "visual")},
            "edge_row": row[selected], "edge_col": col[selected], "edge_weight": weight[selected],
            "correction_norm_ratio": {m: edge_data[m]["stats"]["correction_norm_ratio"] for m in ("text", "visual")},
            "cos_base_corrected": {m: edge_data[m]["stats"]["cos_base_corrected"] for m in ("text", "visual")},
            "parallel_orthogonal": {m: {k: v for k, v in edge_data[m]["stats"].items()
                                         if k in {"parallel_norm_ratio", "orthogonal_norm_ratio", "mean_parallel_coefficient"}}
                                    for m in ("text", "visual")},
            "operator_usage": {m: edge_data[m]["usage"] for m in ("text", "visual")},
            "router_entropy": {m: edge_data[m]["entropy"] for m in ("text", "visual")},
            "operator_output_pairwise_cosine": {m: edge_data[m]["stats"]["operator_pairwise_cosine"]
                                                  for m in ("text", "visual")},
        }

    def _dynamic_edges(self, modality, h, row, col, weight, intervention, selected, collect):
        n_edges = row.numel()
        correction = h.new_zeros(h.shape)
        edge_base, edge_delta, edge_corrected, edge_relation = [], [], [], []
        base_sq = h.new_zeros(())
        delta_sq = h.new_zeros(())
        parallel_sq = h.new_zeros(())
        orthogonal_sq = h.new_zeros(())
        parallel_coeff = h.new_zeros(())
        cosine_sum = h.new_zeros(())
        count = h.new_zeros(())
        encoder = getattr(self, f"relation_encoder_{modality}")
        q, k = encoder.project(h)
        operator = getattr(self, f"dynamic_{modality}")
        conditioner = getattr(self, f"condition_{modality}")
        for start in range(0, n_edges, self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, n_edges)
            base = h[col[start:end]]
            relation = encoder(q[row[start:end]], k[col[start:end]])
            delta = operator.up(operator.bottleneck_state(base) * (2.0 * torch.sigmoid(conditioner(relation))))
            if intervention in {"delta_parallel_only", "delta_orthogonal_only"}:
                denom = base.square().sum(dim=-1, keepdim=True).clamp_min(1e-12)
                parallel = (delta * base).sum(dim=-1, keepdim=True) / denom * base
                delta = parallel if intervention == "delta_parallel_only" else delta - parallel
            correction.index_add_(0, row[start:end], weight[start:end, None] * delta)
            chosen = selected[start:end]
            if bool(chosen.any()):
                b, d = base[chosen], delta[chosen]
                norm_denom = b.square().sum(dim=-1, keepdim=True).clamp_min(1e-12)
                parallel = (d * b).sum(dim=-1, keepdim=True) / norm_denom * b
                orthogonal = d - parallel
                base_sq = base_sq + b.float().square().sum()
                delta_sq = delta_sq + d.float().square().sum()
                parallel_sq = parallel_sq + parallel.float().square().sum()
                orthogonal_sq = orthogonal_sq + orthogonal.float().square().sum()
                parallel_coeff = parallel_coeff + ((d * b).sum(dim=-1) / norm_denom.squeeze(-1)).float().sum()
                cosine_sum = cosine_sum + self._cosine_summary(b, b + d).sum()
                count = count + chosen.sum()
                if collect:
                    edge_base.append(b); edge_delta.append(d); edge_corrected.append(b + d); edge_relation.append(relation[chosen])
        return self._dynamic_result(h, correction, selected, collect, edge_base, edge_delta,
                                    edge_corrected, edge_relation, base_sq, delta_sq,
                                    parallel_sq, orthogonal_sq, parallel_coeff, cosine_sum, count)

    def _dynamic_result(self, h, correction, selected, collect, edge_base, edge_delta,
                        edge_corrected, edge_relation, base_sq, delta_sq,
                        parallel_sq, orthogonal_sq, parallel_coeff, cosine_sum, count):
        denom = count.clamp_min(1.0)
        empty = h.new_empty((0, self.hidden_dim))
        relation = torch.cat(edge_relation) if edge_relation else h.new_empty((0, self.relation_dim))
        base = torch.cat(edge_base) if collect and edge_base else empty
        delta = torch.cat(edge_delta) if collect and edge_delta else empty
        corrected = torch.cat(edge_corrected) if collect and edge_corrected else empty
        b_empty = h.new_empty((int(selected.sum()), 0))
        base_norm = base_sq.sqrt()
        return {
            "correction": correction, "router_probabilities": b_empty,
            "relation_representation": relation,
            "base_edge_message": base, "operator_correction_delta": delta,
            "corrected_edge_message": corrected,
            "usage": h.new_empty((0,)), "entropy": h.new_zeros(()),
            "stats": {
                "correction_norm_ratio": float((delta_sq.sqrt() / base_norm.clamp_min(1e-12)).detach().cpu()),
                "cos_base_corrected": float((cosine_sum / denom).detach().cpu()) if count.item() else 1.0,
                "parallel_norm_ratio": float((parallel_sq.sqrt() / base_norm.clamp_min(1e-12)).detach().cpu()),
                "orthogonal_norm_ratio": float((orthogonal_sq.sqrt() / base_norm.clamp_min(1e-12)).detach().cpu()),
                "mean_parallel_coefficient": float((parallel_coeff / denom).detach().cpu()) if count.item() else 0.0,
                "operator_pairwise_cosine": h.new_empty((0,)),
            },
        }

    def _routed_edges(self, modality, h, row, col, weight, intervention,
                      router_shuffle_permutation, shuffle_seed, selected, collect):
        # A streaming implementation retaining the exact edge order. For a
        # router shuffle, compute all probabilities once and apply the supplied
        # permutation globally; the analysis script can construct matched bins.
        if intervention == "router_shuffle" and self.variant == "p0_operator_routed":
            return self._routed_edges_shuffled(modality, h, row, col, weight,
                                               router_shuffle_permutation, shuffle_seed,
                                               selected, collect)
        n_edges = row.numel()
        correction = h.new_zeros(h.shape)
        ops = getattr(self, f"operators_{modality}")
        enc = (getattr(self, f"relation_encoder_{modality}")
               if self.variant == "p0_operator_routed" else None)
        if enc is not None:
            q, k = enc.project(h)
        # Operator transforms depend only on the source message, so compute
        # one output per node and gather it per edge to keep training memory
        # proportional to node count rather than edge count.
        node_outputs = torch.stack([op(h) for op in ops], dim=1)
        edge_base, edge_delta, edge_corrected, edge_probs, edge_relation = [], [], [], [], []
        base_sq = h.new_zeros(()); delta_sq = h.new_zeros(()); parallel_sq = h.new_zeros(()); orthogonal_sq = h.new_zeros(())
        parallel_coeff = h.new_zeros(()); cosine_sum = h.new_zeros(()); usage_sum = h.new_zeros(self.NUM_OPERATORS)
        entropy_sum = h.new_zeros(()); pair_sum = h.new_zeros(self.NUM_OPERATORS * (self.NUM_OPERATORS - 1) // 2); count = h.new_zeros(())
        for start in range(0, n_edges, self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, n_edges)
            erow, ecol = row[start:end], col[start:end]
            base = h[ecol]
            if self.variant == "p0_operator_uniform":
                relation = h.new_empty((end - start, 0))
                probs = h.new_full((end - start, self.NUM_OPERATORS), 1.0 / self.NUM_OPERATORS)
                outputs = node_outputs[ecol]
            else:
                relation = enc(q[erow], k[ecol])
                probs = torch.softmax(getattr(self, f"router_{modality}")(relation), dim=-1)
                outputs = node_outputs[ecol]
            delta = (outputs * probs.unsqueeze(-1)).sum(dim=1)
            if intervention == "router_uniform":
                probs = torch.full_like(probs, 1.0 / self.NUM_OPERATORS)
                delta = (outputs * probs.unsqueeze(-1)).sum(dim=1)
            if intervention in {"delta_parallel_only", "delta_orthogonal_only"}:
                denom = base.square().sum(dim=-1, keepdim=True).clamp_min(1e-12)
                parallel = (delta * base).sum(dim=-1, keepdim=True) / denom * base
                delta = parallel if intervention == "delta_parallel_only" else delta - parallel
            correction.index_add_(0, erow, weight[start:end, None] * delta)
            chosen = selected[start:end]
            if bool(chosen.any()):
                b, d = base[chosen], delta[chosen]
                denom = b.square().sum(dim=-1, keepdim=True).clamp_min(1e-12)
                parallel = (d * b).sum(dim=-1, keepdim=True) / denom * b
                orthogonal = d - parallel
                base_sq = base_sq + b.float().square().sum(); delta_sq = delta_sq + d.float().square().sum()
                parallel_sq = parallel_sq + parallel.float().square().sum(); orthogonal_sq = orthogonal_sq + orthogonal.float().square().sum()
                parallel_coeff = parallel_coeff + ((d * b).sum(dim=-1) / denom.squeeze(-1)).float().sum()
                cosine_sum = cosine_sum + self._cosine_summary(b, b + d).sum()
                usage_sum = usage_sum + probs[chosen].float().sum(dim=0)
                entropy_sum = entropy_sum + (-(probs[chosen].float().clamp_min(1e-12).log() * probs[chosen].float()).sum(dim=-1)).sum()
                count = count + chosen.sum()
                pair_vals = [self._cosine_summary(outputs[chosen, i], outputs[chosen, j]).sum()
                             for i in range(self.NUM_OPERATORS) for j in range(i + 1, self.NUM_OPERATORS)]
                pair_sum = pair_sum + torch.stack(pair_vals)
                if collect:
                    edge_base.append(b); edge_delta.append(d); edge_corrected.append(b + d)
                    edge_probs.append(probs[chosen]); edge_relation.append(relation[chosen])
        return self._routed_result(h, correction, collect, edge_base, edge_delta, edge_corrected,
                                   edge_probs, edge_relation, base_sq, delta_sq, parallel_sq,
                                   orthogonal_sq, parallel_coeff, cosine_sum, usage_sum,
                                   entropy_sum, pair_sum, count)

    def _routed_edges_shuffled(self, modality, h, row, col, weight, permutation,
                               shuffle_seed, selected, collect):
        enc = getattr(self, f"relation_encoder_{modality}")
        q, k = enc.project(h)
        probs = []
        for start in range(0, row.numel(), self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, row.numel())
            relation = enc(q[row[start:end]], k[col[start:end]])
            probs.append(torch.softmax(getattr(self, f"router_{modality}")(relation), dim=-1))
        all_probs = torch.cat(probs, dim=0) if probs else h.new_empty((0, self.NUM_OPERATORS))
        if isinstance(permutation, dict):
            permutation = permutation.get(modality)
        if permutation is None:
            generator = torch.Generator(device="cpu").manual_seed(int(shuffle_seed or 0))
            permutation = torch.randperm(row.numel(), generator=generator).to(row.device)
        else:
            permutation = permutation.to(device=row.device, dtype=torch.long).reshape(-1)
        if permutation.numel() != row.numel():
            raise ValueError("router shuffle permutation must have one index per P_rel edge")
        shuffled = all_probs.index_select(0, permutation)
        ops = getattr(self, f"operators_{modality}")
        node_outputs = torch.stack([op(h) for op in ops], dim=1)
        correction = h.new_zeros(h.shape)
        edge_base, edge_delta, edge_corrected = [], [], []
        base_sq = h.new_zeros(()); delta_sq = h.new_zeros(()); parallel_sq = h.new_zeros(()); orthogonal_sq = h.new_zeros(())
        parallel_coeff = h.new_zeros(()); cosine_sum = h.new_zeros(()); pair_sum = h.new_zeros(6)
        count = selected.sum().to(dtype=h.dtype); usage_sum = shuffled[selected].float().sum(dim=0) if bool(selected.any()) else h.new_zeros(4)
        entropy_sum = (-(shuffled[selected].float().clamp_min(1e-12).log() * shuffled[selected].float()).sum(dim=-1).sum()
                       if bool(selected.any()) else h.new_zeros(()))
        selected_prob, selected_relation = [], []
        for start in range(0, row.numel(), self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, row.numel())
            base = h[col[start:end]]
            outputs = node_outputs[col[start:end]]
            delta = (outputs * shuffled[start:end, :, None]).sum(dim=1)
            correction.index_add_(0, row[start:end], weight[start:end, None] * delta)
            chosen = selected[start:end]
            if bool(chosen.any()):
                b, d = base[chosen], delta[chosen]
                denom = b.square().sum(dim=-1, keepdim=True).clamp_min(1e-12)
                parallel = (d * b).sum(dim=-1, keepdim=True) / denom * b
                orthogonal = d - parallel
                base_sq += b.float().square().sum(); delta_sq += d.float().square().sum()
                parallel_sq += parallel.float().square().sum(); orthogonal_sq += orthogonal.float().square().sum()
                parallel_coeff += ((d * b).sum(dim=-1) / denom.squeeze(-1)).float().sum()
                cosine_sum += self._cosine_summary(b, b + d).sum()
                pair_sum += torch.stack([self._cosine_summary(outputs[chosen, i], outputs[chosen, j]).sum()
                                         for i in range(4) for j in range(i + 1, 4)])
                if collect:
                    edge_base.append(b); edge_delta.append(d); edge_corrected.append(b + d)
                    selected_prob.append(shuffled[start:end][chosen])
                    relation = enc(q[row[start:end]], k[col[start:end]])
                    selected_relation.append(relation[chosen])
        # Convert aggregate sums to means in the common result formatter.
        return self._routed_result(h, correction, collect, edge_base, edge_delta, edge_corrected,
                                   selected_prob, selected_relation, base_sq, delta_sq, parallel_sq,
                                   orthogonal_sq, parallel_coeff, cosine_sum, usage_sum, entropy_sum,
                                   pair_sum, count)

    def _routed_result(self, h, correction, collect, edge_base, edge_delta,
                       edge_corrected, edge_probs, edge_relation, base_sq, delta_sq,
                       parallel_sq, orthogonal_sq, parallel_coeff, cosine_sum,
                       usage_sum, entropy_sum, pair_sum, count):
        denom = count.clamp_min(1.0)
        base_norm = base_sq.sqrt()
        empty = h.new_empty((0, self.hidden_dim))
        edge_probs_out = torch.cat(edge_probs, dim=0) if collect and edge_probs else h.new_empty((0, self.NUM_OPERATORS))
        return {
            "correction": correction,
            "router_probabilities": edge_probs_out,
            "relation_representation": torch.cat(edge_relation, dim=0) if collect and edge_relation else h.new_empty((0, self.relation_dim)),
            "base_edge_message": torch.cat(edge_base, dim=0) if collect and edge_base else empty,
            "operator_correction_delta": torch.cat(edge_delta, dim=0) if collect and edge_delta else empty,
            "corrected_edge_message": torch.cat(edge_corrected, dim=0) if collect and edge_corrected else empty,
            "usage": usage_sum / denom,
            "entropy": entropy_sum / denom,
            "stats": {
                "correction_norm_ratio": float((delta_sq.sqrt() / base_norm.clamp_min(1e-12)).detach().cpu()),
                "cos_base_corrected": float((cosine_sum / denom).detach().cpu()) if count.item() else 1.0,
                "parallel_norm_ratio": float((parallel_sq.sqrt() / base_norm.clamp_min(1e-12)).detach().cpu()),
                "orthogonal_norm_ratio": float((orthogonal_sq.sqrt() / base_norm.clamp_min(1e-12)).detach().cpu()),
                "mean_parallel_coefficient": float((parallel_coeff / denom).detach().cpu()) if count.item() else 0.0,
                "operator_pairwise_cosine": pair_sum / denom,
            },
        }

    def _format_substrate_result(self, base, kind, target_nodes=None, edge_subset=None,
                                 return_edge_state=False):
        states = {"text": base["S_text"], "visual": base["S_visual"]}
        edge_row, edge_col = base["P_rel"].indices()
        weight = base["P_rel"].values()
        # S45 scalar gates affect only this explicit control; the original
        # physical P_rel tensor remains unchanged and is returned for audit.
        selected = self._select_edge_mask(edge_row, base["P_rel"].size(0), target_nodes, edge_subset)
        outputs = {}
        for modality in ("text", "visual"):
            h = base[f"H0_{modality}"]
            message = h[edge_col]
            if kind == "masspres_scalar":
                gate = base["normalized_gates"][modality].reshape(-1, 1)
                delta = message * (gate - 1.0)
                gates = gate
            else:
                delta = torch.zeros_like(message)
                gates = None
            b, d = message[selected], delta[selected]
            denom = b.square().sum(dim=-1, keepdim=True).clamp_min(1e-12)
            parallel = (d * b).sum(dim=-1, keepdim=True) / denom * b
            orth = d - parallel
            base_norm = b.float().norm().clamp_min(1e-12)
            outputs[modality] = {
                "base_edge_message": b if return_edge_state else h.new_empty((0, self.hidden_dim)),
                "operator_correction_delta": d if return_edge_state else h.new_empty((0, self.hidden_dim)),
                "corrected_edge_message": b + d if return_edge_state else h.new_empty((0, self.hidden_dim)),
                "router_probabilities": h.new_empty((int(selected.sum()), 0)),
                "scalar_gates": gates[selected] if gates is not None else None,
                "correction_norm_ratio": float(d.float().norm() / base_norm),
                "cos_base_corrected": float(F.cosine_similarity(b.float(), (b + d).float(), dim=-1).mean()) if b.numel() else 1.0,
                "parallel_norm_ratio": float(parallel.float().norm() / base_norm),
                "orthogonal_norm_ratio": float(orth.float().norm() / base_norm),
            }
        result = {
            **base,
            "C1_text": states["text"][1], "C2_text": states["text"][2], "C3_text": states["text"][3],
            "C1_visual": states["visual"][1], "C2_visual": states["visual"][2], "C3_visual": states["visual"][3],
            "C_text": states["text"], "C_visual": states["visual"],
            "router_probabilities": {m: outputs[m]["router_probabilities"] for m in ("text", "visual")},
            "base_edge_message": {m: outputs[m]["base_edge_message"] for m in ("text", "visual")},
            "operator_correction_delta": {m: outputs[m]["operator_correction_delta"] for m in ("text", "visual")},
            "corrected_edge_message": {m: outputs[m]["corrected_edge_message"] for m in ("text", "visual")},
            "edge_row": edge_row[selected], "edge_col": edge_col[selected], "edge_weight": weight[selected],
            "correction_norm_ratio": {m: outputs[m]["correction_norm_ratio"] for m in ("text", "visual")},
            "cos_base_corrected": {m: outputs[m]["cos_base_corrected"] for m in ("text", "visual")},
            "parallel_orthogonal": {m: {"parallel_norm_ratio": outputs[m]["parallel_norm_ratio"],
                                         "orthogonal_norm_ratio": outputs[m]["orthogonal_norm_ratio"]}
                                    for m in ("text", "visual")},
            "operator_usage": {m: None for m in ("text", "visual")},
            "router_entropy": {m: None for m in ("text", "visual")},
            "operator_output_pairwise_cosine": {m: None for m in ("text", "visual")},
            "scalar_gates": {m: outputs[m]["scalar_gates"] for m in ("text", "visual")},
        }
        return result

    def forward(self, x: torch.Tensor, edge_index=None):
        result = self.analyze(x, edge_index, return_edge_state=False)
        z = result["fused_z"]
        return z, None, None, z.new_zeros(()), {}

    @torch.no_grad()
    def inference(self, x, edge_index=None, device=None, batch_size: int = 4096):
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        z, _, _, _, _ = self.forward(x.to(device), edge_index.to(device))
        return z.detach().cpu()

