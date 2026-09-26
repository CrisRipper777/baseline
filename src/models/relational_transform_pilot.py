from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.relation_basis_pilot import Model as HistoricalRelationModel
from src.models.relation_basis_pilot import RelationTransform


class PairEncoder(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4 * dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, q: torch.Tensor, k: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "raw":
            zeros = torch.zeros_like(q)
            values = torch.cat((q, k, zeros, zeros), dim=-1)
        elif mode == "relation":
            values = torch.cat((q, k, (q - k).abs(), q * k), dim=-1)
        else:
            raise ValueError(f"unknown pair encoding {mode!r}")
        return self.net(values)


class Model(nn.Module):
    """S4.4 edge-conditioned relational transformations over historical P_rel."""

    VARIANTS = (
        "s44_scalar_global", "s44_scalar_raw", "s44_scalar_rel", "s44_scalar_rel_multi",
        "s44_lowrank_global", "s44_lowrank_rel", "s44_lowrank_rel_multi",
        "s44_expert_uniform", "s44_expert_rel", "s44_expert_rel_multi",
    )
    SCALAR_EDGE = {"s44_scalar_raw", "s44_scalar_rel", "s44_scalar_rel_multi"}
    LOWRANK = {"s44_lowrank_global", "s44_lowrank_rel", "s44_lowrank_rel_multi"}
    EXPERT = {"s44_expert_uniform", "s44_expert_rel", "s44_expert_rel_multi"}
    MULTI = {"s44_scalar_rel_multi", "s44_lowrank_rel_multi", "s44_expert_rel_multi"}
    requires_full_lp_sampler_depth = False

    def __init__(self, cfg, data_info):
        super().__init__()
        self.text_dim = int(data_info.get("text_dim", 0))
        self.visual_dim = int(data_info.get("visual_dim", 0))
        self.input_dim = int(data_info.get("input_dim", self.text_dim + self.visual_dim))
        if self.text_dim <= 0 or self.visual_dim <= 0 or self.input_dim != self.text_dim + self.visual_dim:
            raise ValueError("relational_transform_pilot requires concatenated text and visual features")
        self.hidden_dim = int(cfg.model.get("hidden_dim", 256))
        self.dropout_p = float(cfg.model.get("dropout", 0.2))
        self.relation_dim = int(cfg.model.get("relation_dim", 32))
        self.rank = int(cfg.model.get("lowrank_rank", 8))
        self.edge_chunk_size = int(cfg.model.get("edge_chunk_size", 16384))
        if (self.hidden_dim, self.dropout_p, self.relation_dim, self.rank) != (256, 0.2, 32, 8):
            raise ValueError("S4.4 fixes hidden_dim=256, dropout=0.2, relation_dim=32, rank=8")
        if self.edge_chunk_size < 1:
            raise ValueError("edge_chunk_size must be positive")
        self.variant = str(cfg.model.get("variant", "s44_scalar_global")).strip().lower()
        if self.variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}, got {self.variant!r}")
        h, d, p = self.hidden_dim, self.relation_dim, self.dropout_p
        self.out_dim = h
        self.text_projector = nn.Sequential(nn.Linear(self.text_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(p))
        self.visual_projector = nn.Sequential(nn.Linear(self.visual_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(p))
        self.transform_a_text = RelationTransform(h, p)
        self.transform_a_visual = RelationTransform(h, p)
        self.output_norm_text = nn.LayerNorm(h)
        self.output_norm_visual = nn.LayerNorm(h)
        self.plain_fusion = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, h))
        self.needs_relation = self.variant in self.SCALAR_EDGE or (
            self.variant in self.LOWRANK and self.variant != "s44_lowrank_global") or (
            self.variant in self.EXPERT and self.variant != "s44_expert_uniform")
        if self.needs_relation:
            self.q_text, self.k_text = nn.Linear(h, d), nn.Linear(h, d)
            self.q_visual, self.k_visual = nn.Linear(h, d), nn.Linear(h, d)
            self.pair_text, self.pair_visual = PairEncoder(d), PairEncoder(d)
        if self.variant in self.MULTI:
            self.cross_t, self.cross_v = nn.Linear(d, d), nn.Linear(d, d)
            self.cross_out_text, self.cross_out_visual = nn.Linear(d, d), nn.Linear(d, d)
            self.cross_norm_text, self.cross_norm_visual = nn.LayerNorm(d), nn.LayerNorm(d)
        if self.variant in self.SCALAR_EDGE:
            self.scalar_head_text, self.scalar_head_visual = nn.Linear(d, 1), nn.Linear(d, 1)
        if self.variant == "s44_scalar_global":
            self.global_scalar_text = nn.Parameter(torch.zeros(()))
            self.global_scalar_visual = nn.Parameter(torch.zeros(()))
        if self.variant in self.LOWRANK:
            if self.variant != "s44_lowrank_global":
                self.lowrank_ctrl_text, self.lowrank_ctrl_visual = nn.Linear(d, self.rank), nn.Linear(d, self.rank)
            else:
                self.global_a_text = nn.Parameter(torch.zeros(self.rank))
                self.global_a_visual = nn.Parameter(torch.zeros(self.rank))
            self.v_text, self.u_text = nn.Linear(h, self.rank, bias=False), nn.Linear(self.rank, h, bias=False)
            self.v_visual, self.u_visual = nn.Linear(h, self.rank, bias=False), nn.Linear(self.rank, h, bias=False)
        if self.variant in self.EXPERT:
            for modality in ("text", "visual"):
                setattr(self, f"expert_v1_{modality}", nn.Linear(h, self.rank, bias=False))
                setattr(self, f"expert_v2_{modality}", nn.Linear(h, self.rank, bias=False))
                setattr(self, f"expert_u1_{modality}", nn.Linear(self.rank, h, bias=False))
                setattr(self, f"expert_u2_{modality}", nn.Linear(self.rank, h, bias=False))
            if self.variant != "s44_expert_uniform":
                self.router_text, self.router_visual = nn.Linear(d, 2), nn.Linear(d, 2)
        self._initialize_s44()
        self._operator_cache_key = None
        self._operator_cache_edge_index = None
        self._p_rel_cache = None

    def _initialize_s44(self) -> None:
        # Identity controllers at initialization: g=1, a=0, and pi=(.5,.5).
        for name in ("scalar_head_text", "scalar_head_visual", "lowrank_ctrl_text", "lowrank_ctrl_visual",
                     "router_text", "router_visual"):
            head = getattr(self, name, None)
            if head is not None:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        # Dedicated per-run generators match adapter initialization between
        # global/relation and uniform/routed controls despite extra controller modules.
        base_seed = int(torch.initial_seed())
        for modality_index, modality in enumerate(("text", "visual")):
            for layer_index, name in enumerate(("v", "u", "expert_v1", "expert_v2", "expert_u1", "expert_u2")):
                layer = getattr(self, f"{name}_{modality}", None)
                if layer is None:
                    continue
                stream = base_seed + 44000 + 100 * modality_index + 7 * layer_index
                generator = torch.Generator(device=layer.weight.device).manual_seed(stream)
                if name.startswith("u") or name.startswith("expert_u"):
                    nn.init.normal_(layer.weight, mean=0.0, std=1e-3, generator=generator)
                else:
                    nn.init.normal_(layer.weight, mean=0.0,
                                    std=1.0 / max(1, layer.weight.size(1)) ** 0.5,
                                    generator=generator)

    def _get_p_rel(self, edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype):
        key = (edge_index.data_ptr(), int(getattr(edge_index, "_version", 0)), tuple(edge_index.shape),
               int(num_nodes), edge_index.device, dtype)
        if self._operator_cache_key == key and self._p_rel_cache is not None:
            return self._p_rel_cache
        _, p_rel, _ = HistoricalRelationModel._build_operators(edge_index, num_nodes, dtype)
        self._operator_cache_key, self._operator_cache_edge_index, self._p_rel_cache = key, edge_index, p_rel
        return p_rel

    @staticmethod
    def _tail(transform: RelationTransform, pre: torch.Tensor) -> torch.Tensor:
        return transform.dropout(transform.activation(transform.norm(pre)))

    def _pair(self, modality: str, q: torch.Tensor, k: torch.Tensor, src: torch.Tensor,
              dst: torch.Tensor, mode: str) -> torch.Tensor:
        return getattr(self, f"pair_{modality}")(q[dst], k[src], mode)

    def _relation_states(self, q_t, k_t, q_v, k_v, src, dst, mode: str,
                         cross_modal_mode: str = "normal"):
        r_t = self._pair("text", q_t, k_t, src, dst, mode)
        r_v = self._pair("visual", q_v, k_v, src, dst, mode)
        x_t, x_v = self.cross_t(r_t), self.cross_v(r_v)
        if cross_modal_mode == "remove":
            interaction = torch.zeros_like(x_t)
        elif cross_modal_mode == "swap":
            interaction = self.cross_t(r_v) * self.cross_v(r_t)
        elif cross_modal_mode == "normal":
            interaction = x_t * x_v
        else:
            raise ValueError(f"unknown cross-modal intervention {cross_modal_mode!r}")
        r_tm = self.cross_norm_text(r_t + self.cross_out_text(interaction))
        r_vm = self.cross_norm_visual(r_v + self.cross_out_visual(interaction))
        return r_tm, r_vm

    def _controller(self, modality: str, relation: torch.Tensor) -> torch.Tensor:
        if self.variant in self.SCALAR_EDGE:
            return 2.0 * torch.sigmoid(getattr(self, f"scalar_head_{modality}")(relation))
        if self.variant in self.LOWRANK:
            if self.variant == "s44_lowrank_global":
                return getattr(self, f"global_a_{modality}").expand(relation.size(0), -1)
            return getattr(self, f"lowrank_ctrl_{modality}")(relation)
        if self.variant in self.EXPERT:
            if self.variant == "s44_expert_uniform":
                return relation.new_full((relation.size(0), 2), 0.5)
            return torch.softmax(getattr(self, f"router_{modality}")(relation), dim=-1)
        raise ValueError(f"variant {self.variant} has no edge controller")

    def _compute_controllers(self, modality: str, src, dst, h, q, k, mode,
                             q_t, k_t, q_v, k_v, controller_override,
                             cross_modal_mode):
        chunks = []
        if controller_override is not None:
            return controller_override
        needs_multi = self.variant in self.MULTI
        for start in range(0, src.numel(), self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, src.numel())
            s, d = src[start:end], dst[start:end]
            if needs_multi:
                rtm, rvm = self._relation_states(q_t, k_t, q_v, k_v, s, d, mode, cross_modal_mode)
                relation = rtm if modality == "text" else rvm
            else:
                relation = self._pair(modality, q, k, s, d, mode)
            chunks.append(self._controller(modality, relation))
        if chunks:
            return torch.cat(chunks, dim=0)
        width = 1 if self.variant in self.SCALAR_EDGE else (2 if self.variant in self.EXPERT else self.rank)
        return h.new_empty((0, width))

    def _aggregate_weighted(self, h: torch.Tensor, row, col, weight, controller: torch.Tensor):
        out = h.new_zeros(h.shape)
        for start in range(0, row.numel(), self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, row.numel())
            scale = controller[start:end]
            if scale.ndim == 2:
                scale = scale[:, 0]
            msg = h[col[start:end]] * (weight[start:end] * scale).unsqueeze(-1)
            out.index_add_(0, row[start:end], msg)
        return out

    def _dynamic_delta(self, modality: str, h: torch.Tensor, row, col, weight,
                       controller: torch.Tensor, kind: str):
        out = h.new_zeros(h.shape)
        if kind == "lowrank":
            v = getattr(self, f"v_{modality}")
            u = getattr(self, f"u_{modality}")
        for start in range(0, row.numel(), self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, row.numel())
            hs = h[col[start:end]]
            coeff = controller[start:end]
            if kind == "lowrank":
                message = u(v(hs) * coeff)
            else:
                e1 = getattr(self, f"expert_u1_{modality}")(getattr(self, f"expert_v1_{modality}")(hs))
                e2 = getattr(self, f"expert_u2_{modality}")(getattr(self, f"expert_v2_{modality}")(hs))
                message = coeff[:, :1] * e1 + coeff[:, 1:2] * e2
            out.index_add_(0, row[start:end], weight[start:end, None] * message)
        return out

    def analyze(self, x: torch.Tensor, edge_index: torch.Tensor,
                controller_override: dict[str, torch.Tensor] | None = None,
                cross_modal_mode: str = "normal", return_edge_state: bool = False,
                target_nodes: torch.Tensor | None = None) -> dict[str, Any]:
        if edge_index is None or x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"Expected physical edge_index and x [num_nodes, {self.input_dim}]")
        edge_index = edge_index.to(device=x.device, dtype=torch.long)
        p_rel = self._get_p_rel(edge_index, int(x.size(0)), x.dtype)
        row, col = p_rel.indices()
        weight = p_rel.values()
        if target_nodes is not None:
            target_mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
            target_mask[target_nodes.to(device=x.device, dtype=torch.long)] = True
            keep_edges = target_mask[row]
            row, col, weight = row[keep_edges], col[keep_edges], weight[keep_edges]
        ht = self.text_projector(x[:, :self.text_dim])
        hv = self.visual_projector(x[:, self.text_dim:self.text_dim + self.visual_dim])
        outputs, controllers = {}, {}
        if target_nodes is None:
            ua_text = torch.sparse.mm(p_rel, ht)
            ua_visual = torch.sparse.mm(p_rel, hv)
        else:
            ua_text, ua_visual = ht.new_zeros(ht.shape), hv.new_zeros(hv.shape)
            ua_text.index_add_(0, row, weight[:, None] * ht[col])
            ua_visual.index_add_(0, row, weight[:, None] * hv[col])
        if self.needs_relation:
            q_t, k_t = self.q_text(ht), self.k_text(ht)
            q_v, k_v = self.q_visual(hv), self.k_visual(hv)
        else:
            q_t = k_t = q_v = k_v = None
        for modality, h, q, k in (("text", ht, q_t, k_t), ("visual", hv, q_v, k_v)):
            transform = getattr(self, f"transform_a_{modality}")
            base_pre = None
            if self.variant == "s44_scalar_global":
                scale = 2.0 * torch.sigmoid(getattr(self, f"global_scalar_{modality}"))
                if return_edge_state:
                    controllers[modality] = scale.expand(row.numel(), 1)
                ua = ua_text if modality == "text" else ua_visual
                pre = transform.linear(ua * scale)
            elif self.variant in self.SCALAR_EDGE:
                enc_mode = "raw" if self.variant == "s44_scalar_raw" else "relation"
                multi_mode = "relation" if self.variant == "s44_scalar_rel_multi" else enc_mode
                override = None if controller_override is None else controller_override.get(modality)
                ctrl = self._compute_controllers(modality, col, row, h, q, k, multi_mode,
                                                 q_t, k_t, q_v, k_v, override, cross_modal_mode)
                controllers[modality] = ctrl
                agg = self._aggregate_weighted(h, row, col, weight, ctrl)
                pre = transform.linear(agg)
            elif self.variant in self.LOWRANK:
                ua = ua_text if modality == "text" else ua_visual
                base = transform.linear(ua)
                base_pre = base
                if self.variant == "s44_lowrank_global":
                    ctrl = getattr(self, f"global_a_{modality}").expand(row.numel(), -1)
                else:
                    mode = "relation"
                    override = None if controller_override is None else controller_override.get(modality)
                    ctrl = self._compute_controllers(modality, col, row, h, q, k, mode,
                                                     q_t, k_t, q_v, k_v, override, cross_modal_mode)
                controllers[modality] = ctrl
                delta = self._dynamic_delta(modality, h, row, col, weight, ctrl, "lowrank")
                pre = base + delta
            else:
                ua = ua_text if modality == "text" else ua_visual
                base = transform.linear(ua)
                base_pre = base
                if self.variant == "s44_expert_uniform":
                    ctrl = h.new_full((row.numel(), 2), 0.5)
                else:
                    override = None if controller_override is None else controller_override.get(modality)
                    ctrl = self._compute_controllers(modality, col, row, h, q, k, "relation",
                                                     q_t, k_t, q_v, k_v, override, cross_modal_mode)
                controllers[modality] = ctrl
                delta = self._dynamic_delta(modality, h, row, col, weight, ctrl, "expert")
                pre = base + delta
            relation = self._tail(transform, pre)
            z = getattr(self, f"output_norm_{modality}")(h + relation)
            outputs[modality] = {"H": h, "base_pre": base_pre, "R": relation, "Z": z}
        fused = self.plain_fusion(torch.cat((outputs["text"]["Z"], outputs["visual"]["Z"]), dim=-1))
        result: dict[str, Any] = {
            "H_text": ht, "H_visual": hv, "P_rel": p_rel, "fused_z": fused,
            "Z_text": outputs["text"]["Z"], "Z_visual": outputs["visual"]["Z"],
            "operator_metadata": {"num_nodes": int(x.size(0)), "nnz": int(p_rel._nnz()),
                                  "normalized_off_diagonal_weights": True, "renormalized": False},
        }
        if return_edge_state:
            if self.variant in self.LOWRANK:
                result["base_pre_text"] = outputs["text"]["base_pre"]
                result["base_pre_visual"] = outputs["visual"]["base_pre"]
            result["edge_row"] = row
            result["edge_col"] = col
            result["edge_weight"] = weight
            result["controllers"] = controllers
        return result

    def forward(self, x: torch.Tensor, edge_index=None):
        analysis = self.analyze(x, edge_index)
        z = analysis["fused_z"]
        return z, None, None, z.new_tensor(0.0), {}

    @torch.no_grad()
    def inference(self, x, edge_index=None, device=None, batch_size: int = 4096):
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        z, _, _, _, _ = self.forward(x.to(device), edge_index.to(device))
        return z.detach().cpu()
