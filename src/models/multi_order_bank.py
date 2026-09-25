from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import (
    add_self_loops,
    coalesce,
    remove_self_loops,
    to_undirected,
)


class Model(nn.Module):
    """Simple independent-modal multi-order propagation readout backbone."""

    MAX_ORDER = 3
    READOUTS = {"terminal", "uniform", "gpr"}
    requires_full_lp_sampler_depth = True

    def __init__(self, cfg, data_info):
        super().__init__()
        text_dim = int(data_info.get("text_dim", 0))
        visual_dim = int(data_info.get("visual_dim", 0))
        if text_dim <= 0 or visual_dim <= 0:
            raise ValueError(
                "multi_order_bank requires text_dim and visual_dim to be positive"
            )
        max_order = int(cfg.model.get("max_order", self.MAX_ORDER))
        if max_order != self.MAX_ORDER:
            raise ValueError(
                f"multi_order_bank fixes max_order={self.MAX_ORDER}, got {max_order}"
            )
        self.max_order = self.MAX_ORDER
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.input_dim = int(data_info.get("input_dim", text_dim + visual_dim))
        if self.input_dim != text_dim + visual_dim:
            raise ValueError(
                "multi_order_bank expects concatenated [text, visual] features: "
                f"input_dim={self.input_dim}, text_dim+visual_dim={text_dim + visual_dim}"
            )

        self.hidden_dim = int(cfg.model.hidden_dim)
        self.num_layers = self.MAX_ORDER
        self.dropout = float(cfg.model.get("dropout", 0.2))
        self.readout = str(cfg.model.get("readout", "terminal")).strip().lower()
        if self.readout not in self.READOUTS:
            raise ValueError(
                f"model.readout must be one of {sorted(self.READOUTS)}, got {self.readout!r}"
            )

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.visual_projector = nn.Sequential(
            nn.Linear(self.visual_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.fusion = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        if self.readout == "gpr":
            self.gamma_text = nn.Parameter(torch.full((4,), 0.25))
            self.gamma_visual = nn.Parameter(torch.full((4,), 0.25))
        self.out_dim = self.hidden_dim

        self._operator_cache_key = None
        self._operator_cache: torch.Tensor | None = None

    def _build_propagation_operator(
        self, edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """Build P = D_tilde^-1/2 (A + I) D_tilde^-1/2 as sparse COO."""
        edge_index = edge_index.long()
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_index = coalesce(edge_index, num_nodes=num_nodes)
        row, col = edge_index
        values = torch.ones(row.numel(), dtype=dtype, device=edge_index.device)
        degree = torch.zeros(num_nodes, dtype=dtype, device=edge_index.device)
        degree.index_add_(0, row, values)
        inv_sqrt = degree.clamp_min(1.0).pow(-0.5)
        normalized = inv_sqrt[row] * values * inv_sqrt[col]
        return torch.sparse_coo_tensor(
            edge_index,
            normalized,
            (num_nodes, num_nodes),
            dtype=dtype,
            device=edge_index.device,
        ).coalesce()

    def _get_propagation_operator(
        self, edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype
    ) -> torch.Tensor:
        key = (
            edge_index.data_ptr(),
            int(getattr(edge_index, "_version", 0)),
            tuple(edge_index.shape),
            int(num_nodes),
            edge_index.device,
            dtype,
        )
        if self._operator_cache_key == key and self._operator_cache is not None:
            return self._operator_cache
        operator = self._build_propagation_operator(edge_index, num_nodes, dtype)
        self._operator_cache_key = key
        self._operator_cache = operator
        return operator

    @staticmethod
    def _propagate(
        h0: torch.Tensor, operator: torch.Tensor, max_order: int
    ) -> list[torch.Tensor]:
        states = [h0]
        for _ in range(max_order):
            states.append(torch.sparse.mm(operator, states[-1]))
        return states

    def _readout(
        self, states: list[torch.Tensor], gamma: nn.Parameter | None = None
    ) -> torch.Tensor:
        if self.readout == "terminal":
            return states[3]
        if self.readout == "uniform":
            return sum(states) / 4.0
        if gamma is None:
            raise RuntimeError("GPR readout requires modality-specific gamma parameters")
        return sum(gamma[k] * states[k] for k in range(4))

    def analyze(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        """Return every propagation state and modality readout for inspection."""
        if edge_index is None:
            raise ValueError("multi_order_bank requires the physical edge_index")
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(
                f"Expected x with shape [num_nodes, {self.input_dim}], got {tuple(x.shape)}"
            )
        edge_index = edge_index.to(device=x.device)
        operator = self._get_propagation_operator(
            edge_index, int(x.size(0)), x.dtype
        )
        h0_text = self.text_projector(x[:, : self.text_dim])
        h0_visual = self.visual_projector(
            x[:, self.text_dim : self.text_dim + self.visual_dim]
        )
        states_text = self._propagate(h0_text, operator, self.max_order)
        states_visual = self._propagate(h0_visual, operator, self.max_order)
        z_text = self._readout(
            states_text, getattr(self, "gamma_text", None)
        )
        z_visual = self._readout(
            states_visual, getattr(self, "gamma_visual", None)
        )
        fused = self.fusion(torch.cat([z_text, z_visual], dim=-1))
        return {
            "H0_text": h0_text,
            "H0_visual": h0_visual,
            "S_text": states_text,
            "S_visual": states_visual,
            "Z_text": z_text,
            "Z_visual": z_visual,
            "fused_z": fused,
            "gamma_text": getattr(self, "gamma_text", None),
            "gamma_visual": getattr(self, "gamma_visual", None),
        }

    def forward(self, x: torch.Tensor, edge_index=None):
        analysis = self.analyze(x, edge_index)
        z = analysis["fused_z"]
        aux_loss = z.new_tensor(0.0)
        return z, None, None, aux_loss, {}

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        device: torch.device | None = None,
        batch_size: int = 4096,
    ) -> torch.Tensor:
        self.eval()
        if edge_index is None:
            raise ValueError("multi_order_bank requires the physical edge_index")
        if device is None:
            device = next(self.parameters()).device
        z, _, _, _, _ = self.forward(x.to(device), edge_index.to(device))
        return z.detach().cpu()
