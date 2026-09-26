from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import add_self_loops, coalesce, remove_self_loops, to_undirected


class RelationTransform(nn.Module):
    """The shared one-layer transform used by every relation basis."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.norm(self.linear(x))))


class Model(nn.Module):
    """Independent-modality P0/H1 pilot over the physical graph relation operator."""

    VARIANTS = {
        "p0_relation_only",
        "p0_residual",
        "p0_concat",
        "h1_dual_agg",
        "h1_dual_functional_static",
        "h1_dual_functional_global",
    }
    FUNCTIONAL_VARIANTS = {"h1_dual_functional_static", "h1_dual_functional_global"}
    H1_VARIANTS = {"h1_dual_agg", *FUNCTIONAL_VARIANTS}
    requires_full_lp_sampler_depth = False

    def __init__(self, cfg, data_info):
        super().__init__()
        self.text_dim = int(data_info.get("text_dim", 0))
        self.visual_dim = int(data_info.get("visual_dim", 0))
        self.input_dim = int(data_info.get("input_dim", self.text_dim + self.visual_dim))
        if self.text_dim <= 0 or self.visual_dim <= 0:
            raise ValueError("relation_basis_pilot requires positive text_dim and visual_dim")
        if self.input_dim != self.text_dim + self.visual_dim:
            raise ValueError("relation_basis_pilot expects concatenated [text, visual] features")
        self.hidden_dim = int(cfg.model.get("hidden_dim", 256))
        self.dropout_p = float(cfg.model.get("dropout", 0.2))
        if self.hidden_dim != 256 or self.dropout_p != 0.2:
            raise ValueError("S4.3 fixes hidden_dim=256 and dropout=0.2")
        self.variant = str(cfg.model.get("variant", "p0_relation_only")).strip().lower()
        if self.variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {sorted(self.VARIANTS)}, got {self.variant!r}")

        h, p = self.hidden_dim, self.dropout_p
        self.text_projector = nn.Sequential(
            nn.Linear(self.text_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(p)
        )
        self.visual_projector = nn.Sequential(
            nn.Linear(self.visual_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(p)
        )
        self.transform_a_text = RelationTransform(h, p)
        self.transform_a_visual = RelationTransform(h, p)
        if self.variant == "p0_concat":
            self.concat_text = nn.Linear(2 * h, h)
            self.concat_visual = nn.Linear(2 * h, h)
        if self.variant == "h1_dual_agg":
            self.transform_a2_text = RelationTransform(h, p)
            self.transform_a2_visual = RelationTransform(h, p)
        if self.variant in self.FUNCTIONAL_VARIANTS:
            self.transform_d_text = RelationTransform(h, p)
            self.transform_d_visual = RelationTransform(h, p)
        if self.variant == "h1_dual_functional_global":
            self.theta_text = nn.Parameter(torch.zeros(2))
            self.theta_visual = nn.Parameter(torch.zeros(2))
        self.output_norm_text = nn.LayerNorm(h)
        self.output_norm_visual = nn.LayerNorm(h)
        self.plain_fusion = nn.Sequential(
            nn.Linear(2 * h, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, h)
        )
        self.out_dim = h

        self._operator_cache_key = None
        self._operator_cache_edge_index: torch.Tensor | None = None
        self._p_rel_cache: torch.Tensor | None = None
        self._row_sum_cache: torch.Tensor | None = None

    @staticmethod
    def _build_operators(
        edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return normalized P, its unrenormalized off-diagonal part, and row sums."""
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
        p = torch.sparse_coo_tensor(
            edge_index, normalized, (num_nodes, num_nodes), dtype=dtype, device=edge_index.device
        ).coalesce()
        keep = p.indices()[0] != p.indices()[1]
        rel_indices = p.indices()[:, keep]
        rel_values = p.values()[keep]
        p_rel = torch.sparse_coo_tensor(
            rel_indices, rel_values, (num_nodes, num_nodes), dtype=dtype, device=edge_index.device
        ).coalesce()
        row_sum = torch.zeros(num_nodes, dtype=dtype, device=edge_index.device)
        if rel_values.numel():
            row_sum.index_add_(0, rel_indices[0], rel_values)
        return p, p_rel, row_sum.unsqueeze(-1)

    def _get_operators(self, edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype):
        key = (
            edge_index.data_ptr(), int(getattr(edge_index, "_version", 0)),
            tuple(edge_index.shape), int(num_nodes), edge_index.device, dtype,
        )
        if self._operator_cache_key == key and self._p_rel_cache is not None:
            return self._p_rel_cache, self._row_sum_cache
        _, p_rel, row_sum = self._build_operators(edge_index, num_nodes, dtype)
        self._operator_cache_key = key
        # Retain one graph operator only; replacing the input graph releases the old one.
        self._operator_cache_edge_index = edge_index
        self._p_rel_cache, self._row_sum_cache = p_rel, row_sum
        return p_rel, row_sum

    @staticmethod
    def _aggregate(p_rel: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(p_rel, h)

    def _one_modality(
        self, modality: str, h: torch.Tensor, p_rel: torch.Tensor, row_sum: torch.Tensor,
        basis_intervention: str, swap_modality_alpha: bool,
    ) -> dict[str, torch.Tensor | None]:
        ua = self._aggregate(p_rel, h)
        ta = getattr(self, f"transform_a_{modality}")
        ra = ta(ua)
        ra2 = rd = ud = alpha = None
        if self.variant == "h1_dual_agg":
            ra2 = getattr(self, f"transform_a2_{modality}")(ua)
            relation = 0.5 * ra + 0.5 * ra2
            z = getattr(self, f"output_norm_{modality}")(h + relation)
        elif self.variant in self.FUNCTIONAL_VARIANTS:
            ud = row_sum * h - ua
            rd = getattr(self, f"transform_d_{modality}")(ud)
            if self.variant == "h1_dual_functional_global":
                theta = getattr(self, f"theta_{modality}")
                if swap_modality_alpha:
                    theta = self.theta_visual if modality == "text" else self.theta_text
                alpha = torch.softmax(theta, dim=0)
                relation = alpha[0] * ra + alpha[1] * rd
            else:
                relation = 0.5 * ra + 0.5 * rd
            if basis_intervention == "force_a":
                relation = ra
            elif basis_intervention == "force_d":
                relation = rd
            elif basis_intervention != "normal":
                raise ValueError(f"unknown basis intervention {basis_intervention!r}")
            z = getattr(self, f"output_norm_{modality}")(h + relation)
        elif self.variant == "p0_relation_only":
            z = getattr(self, f"output_norm_{modality}")(ra)
        elif self.variant == "p0_residual":
            z = getattr(self, f"output_norm_{modality}")(h + ra)
        elif self.variant == "p0_concat":
            concat = getattr(self, f"concat_{modality}")
            z = getattr(self, f"output_norm_{modality}")(concat(torch.cat([h, ra], dim=-1)))
        else:  # pragma: no cover - constructor validates the closed variant set
            raise RuntimeError(self.variant)
        return {"H": h, "U_A": ua, "U_D": ud, "R_A": ra, "R_A2": ra2, "R_D": rd,
                "alpha": alpha, "Z": z}

    def analyze(
        self, x: torch.Tensor, edge_index: torch.Tensor,
        basis_intervention: str = "normal", swap_modality_alpha: bool = False,
    ) -> dict[str, object]:
        if edge_index is None:
            raise ValueError("relation_basis_pilot requires the physical edge_index")
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"Expected x with shape [num_nodes, {self.input_dim}], got {tuple(x.shape)}")
        if self.variant not in self.FUNCTIONAL_VARIANTS and (basis_intervention != "normal" or swap_modality_alpha):
            raise ValueError("basis interventions are defined only for functional H1 variants")
        edge_index = edge_index.to(device=x.device)
        p_rel, row_sum = self._get_operators(edge_index, int(x.size(0)), x.dtype)
        h_text = self.text_projector(x[:, :self.text_dim])
        h_visual = self.visual_projector(x[:, self.text_dim:self.text_dim + self.visual_dim])
        text = self._one_modality("text", h_text, p_rel, row_sum, basis_intervention, swap_modality_alpha)
        visual = self._one_modality("visual", h_visual, p_rel, row_sum, basis_intervention, swap_modality_alpha)
        fused_z = self.plain_fusion(torch.cat([text["Z"], visual["Z"]], dim=-1))
        return {
            "H_text": text["H"], "H_visual": visual["H"], "P_rel": p_rel,
            "operator_metadata": {"num_nodes": int(x.size(0)), "nnz": int(p_rel._nnz()),
                                  "normalized_off_diagonal_weights": True, "renormalized": False},
            "s": row_sum,
            "U_A_text": text["U_A"], "U_A_visual": visual["U_A"],
            "U_D_text": text["U_D"], "U_D_visual": visual["U_D"],
            "R_A_text": text["R_A"], "R_A_visual": visual["R_A"],
            "R_A2_text": text["R_A2"], "R_A2_visual": visual["R_A2"],
            "R_D_text": text["R_D"], "R_D_visual": visual["R_D"],
            "alpha_text": text["alpha"], "alpha_visual": visual["alpha"],
            "Z_text": text["Z"], "Z_visual": visual["Z"], "fused_z": fused_z,
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
