from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch_geometric.utils import remove_self_loops, to_undirected

from src.models.multi_order_bank import Model as HistoricalMultiOrderBank


class _SparseValueMatmul(torch.autograd.Function):
    """Sparse-dense product with edge-wise gradients for learned COO values.

    PyTorch's generic COO value backward can form a dense N-by-N gradient.
    Here dL/dA[row,col] is computed directly from the two endpoint features.
    """

    @staticmethod
    def forward(ctx, indices: torch.Tensor, values: torch.Tensor,
                size: tuple[int, int], dense: torch.Tensor):
        sparse = torch.sparse_coo_tensor(indices, values, size, device=values.device).coalesce()
        ctx.save_for_backward(sparse.indices(), sparse.values(), dense)
        ctx.size = size
        return torch.sparse.mm(sparse, dense)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        indices, values, dense = ctx.saved_tensors
        row, col = indices
        grad_values = (grad_output.index_select(0, row) * dense.index_select(0, col)).sum(dim=-1)
        transposed = torch.sparse_coo_tensor(
            indices.flip(0), values, (ctx.size[1], ctx.size[0]), device=values.device
        ).coalesce()
        grad_dense = torch.sparse.mm(transposed, grad_output)
        return None, grad_values, None, grad_dense


def _sparse_mm(operator: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
    if operator.layout != torch.sparse_coo:
        return torch.sparse.mm(operator, dense)
    indices, values = operator.indices(), operator.values()
    if values.requires_grad:
        return _SparseValueMatmul.apply(indices, values, tuple(operator.shape), dense)
    return torch.sparse.mm(operator, dense)


class Model(nn.Module):
    """S4.5 relation-calibrated, multi-order state-preserving NC pilot."""

    VARIANTS = (
        "s45_identity_terminal",
        "s45_identity_uniform",
        "s45_identity_propagated_uniform",
        "s45_unconstrained_entry_uniform",
        "s45_masspres_entry_terminal",
        "s45_masspres_entry_uniform",
        "s45_masspres_entry_propagated_uniform",
        "s45_masspres_persistent_uniform",
    )
    IDENTITY = set(VARIANTS[:3])
    MASS_PRESERVING = {
        "s45_masspres_entry_terminal",
        "s45_masspres_entry_uniform",
        "s45_masspres_entry_propagated_uniform",
        "s45_masspres_persistent_uniform",
    }
    PERSISTENT = {"s45_masspres_persistent_uniform"}
    READOUTS = {
        "s45_identity_terminal": "terminal",
        "s45_identity_uniform": "uniform",
        "s45_identity_propagated_uniform": "propagated_uniform",
        "s45_unconstrained_entry_uniform": "uniform",
        "s45_masspres_entry_terminal": "terminal",
        "s45_masspres_entry_uniform": "uniform",
        "s45_masspres_entry_propagated_uniform": "propagated_uniform",
        "s45_masspres_persistent_uniform": "uniform",
    }
    requires_full_lp_sampler_depth = False

    def __init__(self, cfg, data_info):
        super().__init__()
        self.text_dim = int(data_info.get("text_dim", 0))
        self.visual_dim = int(data_info.get("visual_dim", 0))
        self.input_dim = int(data_info.get("input_dim", self.text_dim + self.visual_dim))
        if self.text_dim <= 0 or self.visual_dim <= 0:
            raise ValueError("relcal_statepres_pilot requires positive text and visual dimensions")
        if self.input_dim != self.text_dim + self.visual_dim:
            raise ValueError("expected concatenated [text, visual] input features")

        self.hidden_dim = int(cfg.model.get("hidden_dim", 256))
        self.dropout_p = float(cfg.model.get("dropout", 0.2))
        self.max_order = int(cfg.model.get("max_order", 3))
        self.relation_dim = int(cfg.model.get("relation_dim", 32))
        self.edge_chunk_size = int(cfg.model.get("edge_chunk_size", 16384))
        if (self.hidden_dim, self.dropout_p, self.max_order, self.relation_dim) != (256, 0.2, 3, 32):
            raise ValueError("S4.5 fixes hidden_dim=256, dropout=.2, max_order=3, relation_dim=32")
        if self.edge_chunk_size < 1:
            raise ValueError("edge_chunk_size must be positive")
        self.variant = str(cfg.model.get("variant", self.VARIANTS[0])).strip().lower()
        if self.variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}, got {self.variant!r}")
        self.readout = self.READOUTS[self.variant]
        self.mass_preserving = self.variant in self.MASS_PRESERVING
        self.persistent = self.variant in self.PERSISTENT
        self.calibrated = self.variant not in self.IDENTITY

        h, p = self.hidden_dim, self.dropout_p
        self.text_projector = nn.Sequential(nn.Linear(self.text_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(p))
        self.visual_projector = nn.Sequential(nn.Linear(self.visual_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(p))
        self.plain_fusion = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, h))
        self.out_dim = h

        if self.calibrated:
            d = self.relation_dim
            self.q_text, self.k_text = nn.Linear(h, d), nn.Linear(h, d)
            self.q_visual, self.k_visual = nn.Linear(h, d), nn.Linear(h, d)
            self.pair_text = nn.Sequential(nn.Linear(4 * d, d), nn.ReLU(), nn.Linear(d, d))
            self.pair_visual = nn.Sequential(nn.Linear(4 * d, d), nn.ReLU(), nn.Linear(d, d))
            self.gate_text, self.gate_visual = nn.Linear(d, 1), nn.Linear(d, 1)
            for head in (self.gate_text, self.gate_visual):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

        self._operator_cache_key = None
        self._operator_cache_edge_index: torch.Tensor | None = None
        self._operator_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def _get_operators(self, edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype):
        key = (edge_index.data_ptr(), int(getattr(edge_index, "_version", 0)), tuple(edge_index.shape),
               int(num_nodes), edge_index.device, dtype)
        if self._operator_cache_key == key and self._operator_cache is not None:
            return self._operator_cache
        # Use the historical MOB operator builder verbatim: remove raw loops,
        # symmetrize, add one loop, coalesce, and symmetrically normalize.
        p = HistoricalMultiOrderBank._build_propagation_operator(self, edge_index, num_nodes, dtype)
        indices, values = p.indices(), p.values()
        diagonal = indices[0] == indices[1]
        p_self = torch.sparse_coo_tensor(indices[:, diagonal], values[diagonal], p.shape,
                                         dtype=dtype, device=edge_index.device).coalesce()
        p_rel = torch.sparse_coo_tensor(indices[:, ~diagonal], values[~diagonal], p.shape,
                                        dtype=dtype, device=edge_index.device).coalesce()
        self._operator_cache_key = key
        self._operator_cache_edge_index = edge_index
        self._operator_cache = (p, p_self, p_rel)
        return self._operator_cache

    def _raw_gates(self, h: torch.Tensor, row: torch.Tensor, col: torch.Tensor):
        gates: dict[str, torch.Tensor] = {}
        for modality, q_layer, k_layer, pair, head in (
            ("text", self.q_text, self.k_text, self.pair_text, self.gate_text),
            ("visual", self.q_visual, self.k_visual, self.pair_visual, self.gate_visual),
        ):
            feature = h[modality]
            q, k = q_layer(feature), k_layer(feature)
            parts = []
            for start in range(0, row.numel(), self.edge_chunk_size):
                end = min(start + self.edge_chunk_size, row.numel())
                qi, kj = q[row[start:end]], k[col[start:end]]
                pair_input = torch.cat((qi, kj, (qi - kj).abs(), qi * kj), dim=-1)
                relation = pair(pair_input)
                parts.append(2.0 * torch.sigmoid(head(relation)))
            gates[modality] = torch.cat(parts, dim=0) if parts else feature.new_empty((0, 1))
        return gates

    @staticmethod
    def _calibrated_operator(p_self: torch.Tensor, p_rel: torch.Tensor,
                             gate: torch.Tensor, preserve_mass: bool):
        row, col = p_rel.indices()
        base = p_rel.values()
        raw = gate.reshape(-1).to(dtype=base.dtype, device=base.device)
        if raw.numel() != base.numel():
            raise ValueError("gate override must have one value for each off-diagonal edge")
        # Accumulate row masses in float64 to avoid degree-dependent float32
        # reduction error on large neighborhoods; stored operator values remain
        # float32 and preserve the historical propagation dtype.
        base64, raw64 = base.double(), raw.double()
        row_mass = base64.new_zeros(p_rel.size(0))
        row_mass.index_add_(0, row, base64)
        if preserve_mass:
            weighted = base64.new_zeros(p_rel.size(0))
            weighted.index_add_(0, row, base64 * raw64)
            valid = (row_mass > 0) & (weighted > 0)
            scale = torch.where(valid, row_mass / weighted.clamp_min(torch.finfo(weighted.dtype).tiny),
                                torch.ones_like(row_mass))
            used_gate64 = raw64 * scale[row]
        else:
            scale = torch.ones_like(row_mass)
            used_gate64 = raw64
        rel_values = (base64 * used_gate64).to(base.dtype)
        used_gate = used_gate64.to(base.dtype)
        rel = torch.sparse_coo_tensor(p_rel.indices(), rel_values, p_rel.shape,
                                      dtype=base.dtype, device=base.device).coalesce()
        indices = torch.cat((p_self.indices(), rel.indices()), dim=1)
        values = torch.cat((p_self.values(), rel.values()), dim=0)
        full = torch.sparse_coo_tensor(indices, values, p_rel.shape,
                                       dtype=base.dtype, device=base.device).coalesce()
        return full, rel, row_mass, scale, used_gate

    def _readout(self, states: list[torch.Tensor]) -> torch.Tensor:
        if self.readout == "terminal":
            return states[3]
        if self.readout == "uniform":
            return sum(states) / 4.0
        return (states[1] + states[2] + states[3]) / 3.0

    def analyze(self, x: torch.Tensor, edge_index: torch.Tensor,
                gate_override: dict[str, torch.Tensor] | str | None = None,
                preserve_mass_override: bool | None = None) -> dict[str, Any]:
        if edge_index is None or x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"expected physical edge_index and x [num_nodes, {self.input_dim}]")
        edge_index = edge_index.to(device=x.device, dtype=torch.long)
        p, p_self, p_rel = self._get_operators(edge_index, int(x.size(0)), x.dtype)
        h = {
            "text": self.text_projector(x[:, :self.text_dim]),
            "visual": self.visual_projector(x[:, self.text_dim:self.text_dim + self.visual_dim]),
        }
        row, col = p_rel.indices()
        raw_gates: dict[str, torch.Tensor] = {}
        normalized_gates: dict[str, torch.Tensor] = {}
        operators = {"text": p, "visual": p}
        row_masses = {"text": None, "visual": None}
        if self.calibrated:
            raw_gates = self._raw_gates(h, row, col)
            for modality in ("text", "visual"):
                gate = raw_gates[modality]
                if gate_override == "off":
                    gate = torch.ones_like(gate)
                elif isinstance(gate_override, dict) and modality in gate_override:
                    gate = gate_override[modality].to(device=x.device, dtype=x.dtype).reshape(-1, 1)
                raw_gates[modality] = gate
                preserve_mass = self.mass_preserving if preserve_mass_override is None else preserve_mass_override
                op, rel, mass, _, normalized = self._calibrated_operator(
                    p_self, p_rel, gate, preserve_mass
                )
                operators[modality] = op
                row_masses[modality] = mass
                normalized_gates[modality] = normalized.reshape(-1, 1)

        states: dict[str, list[torch.Tensor]] = {}
        readouts: dict[str, torch.Tensor] = {}
        for modality in ("text", "visual"):
            state = [h[modality]]
            if self.calibrated:
                first_operator = operators[modality]
                state.append(_sparse_mm(first_operator, state[-1]))
                subsequent = first_operator if self.persistent else p
                state.append(_sparse_mm(subsequent, state[-1]))
                state.append(_sparse_mm(subsequent, state[-1]))
            else:
                for _ in range(3):
                    state.append(torch.sparse.mm(p, state[-1]))
            states[modality] = state
            readouts[modality] = self._readout(state)
        fused = self.plain_fusion(torch.cat((readouts["text"], readouts["visual"]), dim=-1))
        return {
            "P": p, "P_self": p_self, "P_rel": p_rel,
            "H0_text": h["text"], "H0_visual": h["visual"],
            "S_text": states["text"], "S_visual": states["visual"],
            "Z_text": readouts["text"], "Z_visual": readouts["visual"],
            "fused_z": fused, "raw_gates": raw_gates,
            "normalized_gates": normalized_gates, "operators": operators,
            "row_masses": row_masses,
        }

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
