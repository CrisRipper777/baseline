from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from .common import get_activation, make_norm


class OperatorControl(nn.Module):
    """Analysis-only independent-modal GCN/SAGE control with fixed readouts."""

    OPERATOR = "base"
    READOUTS = {"deep_only", "anchored25"}

    def __init__(self, cfg, data_info):
        super().__init__()
        self.text_dim = int(data_info.get("text_dim", 0))
        self.visual_dim = int(data_info.get("visual_dim", 0))
        if self.text_dim <= 0 or self.visual_dim <= 0:
            raise ValueError("operator controls require positive text_dim and visual_dim")
        if int(data_info.get("input_dim", self.text_dim + self.visual_dim)) != self.text_dim + self.visual_dim:
            raise ValueError("operator controls require concatenated [text, visual] input features")

        self.hidden_dim = int(cfg.model.hidden_dim)
        self.num_layers = int(cfg.model.num_layers)
        if self.num_layers != 3:
            raise ValueError(f"operator controls fix num_layers=3, got {self.num_layers}")
        self.dropout = nn.Dropout(float(cfg.model.dropout))
        self.activation = get_activation(str(cfg.model.get("activation", "relu")))
        norm = cfg.model.get("norm", "batchnorm")
        self.readout = str(cfg.model.get("readout", "deep_only")).strip().lower()
        if self.readout not in self.READOUTS:
            raise ValueError(f"readout must be one of {sorted(self.READOUTS)}, got {self.readout!r}")

        self.text_projector = nn.Linear(self.text_dim, self.hidden_dim)
        self.visual_projector = nn.Linear(self.visual_dim, self.hidden_dim)
        self.text_layers = self._make_layers()
        self.visual_layers = self._make_layers()
        self.text_norms = nn.ModuleList([make_norm(norm, self.hidden_dim) for _ in range(2)])
        self.visual_norms = nn.ModuleList([make_norm(norm, self.hidden_dim) for _ in range(2)])
        # Identical to MultiOrderBank's plain_mlp fusion.
        self.plain_fusion = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(cfg.model.dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.out_dim = self.hidden_dim

    def _make_layers(self) -> nn.ModuleList:
        if self.OPERATOR == "gcn":
            return nn.ModuleList([
                GCNConv(self.hidden_dim, self.hidden_dim, normalize=False, add_self_loops=False)
                for _ in range(self.num_layers)
            ])
        if self.OPERATOR == "sage":
            return nn.ModuleList([SAGEConv(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)])
        raise RuntimeError(f"Unknown operator {self.OPERATOR}")

    def _branch(self, h0, layers, norms, edge_index, edge_weight=None):
        h = h0
        for idx, conv in enumerate(layers):
            if self.OPERATOR == "gcn":
                h = conv(h, edge_index, edge_weight=edge_weight)
            else:
                h = conv(h, edge_index)
            if idx != len(layers) - 1:
                h = self.dropout(self.activation(norms[idx](h)))
        return h

    def analyze(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict[str, torch.Tensor]:
        if edge_index is None:
            raise ValueError("operator controls require the physical edge_index")
        if x.dim() != 2 or x.size(1) != self.text_dim + self.visual_dim:
            raise ValueError("input feature dimensions do not match text_dim + visual_dim")
        x_t, x_v = x[:, :self.text_dim], x[:, self.text_dim:]
        h0_t, h0_v = self.text_projector(x_t), self.visual_projector(x_v)
        if self.OPERATOR == "gcn":
            norm_edge, norm_weight = gcn_norm(
                edge_index, edge_weight=None, num_nodes=x.size(0), improved=False,
                add_self_loops=True, flow="source_to_target", dtype=x.dtype,
            )
        else:
            norm_edge, norm_weight = edge_index, None
        h3_t = self._branch(h0_t, self.text_layers, self.text_norms, norm_edge, norm_weight)
        h3_v = self._branch(h0_v, self.visual_layers, self.visual_norms, norm_edge, norm_weight)
        if self.readout == "deep_only":
            z_t, z_v = h3_t, h3_v
        else:
            z_t, z_v = 0.25 * h0_t + 0.75 * h3_t, 0.25 * h0_v + 0.75 * h3_v
        fused = self.plain_fusion(torch.cat([z_t, z_v], dim=-1))
        return {"H0_text": h0_t, "H0_visual": h0_v, "H3_text": h3_t,
                "H3_visual": h3_v, "Z_text": z_t, "Z_visual": z_v,
                "fused_z": fused}

    def forward(self, x, edge_index):
        result = self.analyze(x, edge_index)
        return result["fused_z"], None, None, x.new_tensor(0.0), {}

    @torch.no_grad()
    def inference(self, x, edge_index, device=None, batch_size=65536):
        self.eval()
        if device is not None:
            x, edge_index = x.to(device), edge_index.to(device)
        return self.analyze(x, edge_index)["fused_z"].detach().cpu()
