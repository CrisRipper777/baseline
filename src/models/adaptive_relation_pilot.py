from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.models.relation_basis_pilot import Model as HistoricalModel
from src.models.relation_basis_pilot import RelationTransform


class SignedRelationTransform(nn.Module):
    """H1R transform that retains negative responses after normalization."""

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.norm(self.linear(x))))


class ConditionEncoderV1(nn.Module):
    """Small node controller over modality-local hidden and relation states."""

    def __init__(self, out_dim: int, correction: bool = False):
        super().__init__()
        self.q_h = nn.Linear(256, 32)
        self.q_relation = nn.Linear(256, 32)
        self.q_correction = nn.Linear(256, 32) if correction else None
        input_dim = 96 if correction else 64
        self.mlp = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, out_dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self, h: torch.Tensor, relation: torch.Tensor,
        correction: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [self.q_h(h), self.q_relation(relation)]
        if self.q_correction is not None:
            if correction is None:
                raise ValueError("correction state is required by this controller")
            parts.append(self.q_correction(correction))
        elif correction is not None:
            raise ValueError("this controller accepts exactly H and relation")
        return self.mlp(torch.cat(parts, dim=-1))


def apply_group_gate(relation: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Apply eight channel-group coefficients with an exact [N, 8, 32] view."""
    n = relation.shape[0]
    if beta.dim() == 1:
        beta = beta.reshape(1, 8).expand(n, 8)
    if beta.shape != (n, 8):
        raise ValueError(f"group beta must have shape [N,8], got {tuple(beta.shape)}")
    return (relation.reshape(n, 8, 32) * beta.unsqueeze(-1)).reshape_as(relation)


def apply_correction(ra: torch.Tensor, rc: torch.Tensor, coefficient: torch.Tensor,
                     intervention: str = "normal") -> torch.Tensor:
    """Apply a frozen H2b coefficient intervention to the correction branch."""
    if intervention == "zero_correction":
        coefficient = torch.zeros_like(coefficient)
    elif intervention == "sign_flip":
        coefficient = -coefficient
    elif intervention != "normal":
        raise ValueError(f"unknown H2b intervention {intervention!r}")
    return ra + coefficient * rc


class Model(nn.Module):
    """Independent-modality S4.3 signed and adaptive relation pilot models."""

    H1R_VARIANTS = {"h1r_dual_agg_signed", "h1r_dual_functional_signed"}
    H2A_VARIANTS = {
        "h2a_global_scalar", "h2a_node_scalar", "h2a_global_group", "h2a_node_group",
    }
    H2B_VARIANTS = {
        "h2b_global_agg_correction", "h2b_global_diff_correction",
        "h2b_node_agg_correction", "h2b_node_diff_correction",
    }
    VARIANTS = H1R_VARIANTS | H2A_VARIANTS | H2B_VARIANTS
    GROUPS = 8
    GROUP_SIZE = 32
    LAMBDA_BIAS = math.atanh(0.05)
    requires_full_lp_sampler_depth = False

    def __init__(self, cfg, data_info):
        super().__init__()
        self.text_dim = int(data_info.get("text_dim", 0))
        self.visual_dim = int(data_info.get("visual_dim", 0))
        self.input_dim = int(data_info.get("input_dim", self.text_dim + self.visual_dim))
        if self.text_dim <= 0 or self.visual_dim <= 0 or self.input_dim != self.text_dim + self.visual_dim:
            raise ValueError("adaptive_relation_pilot expects concatenated text and visual features")
        self.hidden_dim = int(cfg.model.get("hidden_dim", 256))
        self.dropout_p = float(cfg.model.get("dropout", 0.2))
        if self.hidden_dim != 256 or self.dropout_p != 0.2:
            raise ValueError("S4.3 fixes hidden_dim=256 and dropout=0.2")
        self.variant = str(cfg.model.get("variant", "h1r_dual_agg_signed")).strip().lower()
        if self.variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {sorted(self.VARIANTS)}, got {self.variant!r}")

        h, p = self.hidden_dim, self.dropout_p
        # Keep projector and plain_mlp module definitions identical to the frozen H1 model.
        self.text_projector = nn.Sequential(nn.Linear(self.text_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(p))
        self.visual_projector = nn.Sequential(nn.Linear(self.visual_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(p))
        self.output_norm_text = nn.LayerNorm(h)
        self.output_norm_visual = nn.LayerNorm(h)
        self.plain_fusion = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, h))
        self.out_dim = h

        self.transform_a_text = self._transform()
        self.transform_a_visual = self._transform()
        is_generic_correction = self.variant in {
            "h2b_global_agg_correction", "h2b_node_agg_correction",
        }
        self.transform_c_text = self._transform() if is_generic_correction else None
        self.transform_c_visual = self._transform() if is_generic_correction else None
        if self.variant in self.H1R_VARIANTS:
            self.transform_a_text = SignedRelationTransform(h, p)
            self.transform_a_visual = SignedRelationTransform(h, p)
            if self.variant == "h1r_dual_agg_signed":
                self.transform_a2_text = SignedRelationTransform(h, p)
                self.transform_a2_visual = SignedRelationTransform(h, p)
            else:
                self.transform_d_text = SignedRelationTransform(h, p)
                self.transform_d_visual = SignedRelationTransform(h, p)
        elif self.variant in self.H2B_VARIANTS and "diff" in self.variant:
            self.transform_d_text = self._transform()
            self.transform_d_visual = self._transform()

        self.base_bias = nn.ParameterDict()
        self.controllers = nn.ModuleDict()
        if self.variant in self.H2A_VARIANTS:
            width = self.GROUPS if self.variant.endswith("group") else 1
            for modality in ("text", "visual"):
                self.base_bias[modality] = nn.Parameter(torch.zeros(width))
                if self.variant.startswith("h2a_node"):
                    self.controllers[modality] = ConditionEncoderV1(width)
        elif self.variant in self.H2B_VARIANTS:
            for modality in ("text", "visual"):
                self.base_bias[modality] = nn.Parameter(torch.full((1,), self.LAMBDA_BIAS))
                if self.variant.startswith("h2b_node"):
                    self.controllers[modality] = ConditionEncoderV1(1, correction=True)

        self._operator_cache_key = None
        self._operator_cache_edge_index = None
        self._p_rel_cache = None
        self._row_sum_cache = None

    def _transform(self):
        return RelationTransform(self.hidden_dim, self.dropout_p)

    def _get_operators(self, edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype):
        # Call the exact historical P_rel builder: normalized off-diagonal operator, unrenormalized.
        key = (edge_index.data_ptr(), int(getattr(edge_index, "_version", 0)), tuple(edge_index.shape),
               int(num_nodes), edge_index.device, dtype)
        if self._operator_cache_key == key and self._p_rel_cache is not None:
            return self._p_rel_cache, self._row_sum_cache
        _, p_rel, row_sum = HistoricalModel._build_operators(edge_index, num_nodes, dtype)
        self._operator_cache_key = key
        self._operator_cache_edge_index = edge_index
        self._p_rel_cache, self._row_sum_cache = p_rel, row_sum
        return p_rel, row_sum

    @staticmethod
    def _aggregate(p_rel: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(p_rel, h)

    def _h2a_beta(self, modality: str, h: torch.Tensor, ra: torch.Tensor) -> torch.Tensor:
        base = self.base_bias[modality]
        if self.variant.startswith("h2a_node"):
            base = base + self.controllers[modality](h, ra)
        return 2.0 * torch.sigmoid(base)

    def _h2b_lambda(self, modality: str, h: torch.Tensor, ra: torch.Tensor, rc: torch.Tensor) -> torch.Tensor:
        value = self.base_bias[modality]
        if self.variant.startswith("h2b_node"):
            value = value + self.controllers[modality](h, ra, rc)
        return 2.0 * torch.tanh(value)

    def _one_modality(
        self, modality: str, h: torch.Tensor, p_rel: torch.Tensor, row_sum: torch.Tensor,
        beta_override: torch.Tensor | None, lambda_override: torch.Tensor | None,
        basis_intervention: str,
    ) -> dict[str, torch.Tensor | None]:
        ua = self._aggregate(p_rel, h)
        ra = getattr(self, f"transform_a_{modality}")(ua)
        ud = row_sum * h - ua
        ra2 = rd = beta = coefficient = rc = None
        if self.variant in self.H1R_VARIANTS:
            if self.variant == "h1r_dual_agg_signed":
                ra2 = getattr(self, f"transform_a2_{modality}")(ua)
                relation = 0.5 * ra + 0.5 * ra2
            else:
                rd = getattr(self, f"transform_d_{modality}")(ud)
                if basis_intervention == "normal":
                    relation = 0.5 * ra + 0.5 * rd
                elif basis_intervention == "zero_d_keep_scale":
                    relation = 0.5 * ra
                elif basis_intervention == "zero_a_keep_scale":
                    relation = 0.5 * rd
                elif basis_intervention == "force_a":
                    relation = ra
                elif basis_intervention == "force_d":
                    relation = rd
                else:
                    raise ValueError(f"unknown H1R intervention {basis_intervention!r}")
            z = getattr(self, f"output_norm_{modality}")(h + relation)
        elif self.variant in self.H2A_VARIANTS:
            beta = self._h2a_beta(modality, h, ra) if beta_override is None else beta_override
            if beta.dim() == 1 and beta.numel() == self.GROUPS:
                gated = apply_group_gate(ra, beta)
            elif beta.dim() == 2 and beta.shape[-1] == self.GROUPS:
                gated = apply_group_gate(ra, beta)
            else:
                gated = ra * beta.reshape(-1, 1) if beta.numel() != 1 else ra * beta
            z = getattr(self, f"output_norm_{modality}")(h + gated)
        else:
            if "diff" in self.variant:
                rc = getattr(self, f"transform_d_{modality}")(ud)
            else:
                rc = getattr(self, f"transform_c_{modality}")(ua)
            coefficient = self._h2b_lambda(modality, h, ra, rc) if lambda_override is None else lambda_override
            relation = apply_correction(ra, rc, coefficient, basis_intervention)
            z = getattr(self, f"output_norm_{modality}")(h + relation)
        return {"H": h, "U_A": ua, "U_D": ud, "R_A": ra, "R_A2": ra2,
                "R_D": rd, "R_C": rc, "beta": beta, "lambda": coefficient, "Z": z}

    def analyze(
        self, x: torch.Tensor, edge_index: torch.Tensor,
        beta_overrides: dict[str, torch.Tensor] | None = None,
        lambda_overrides: dict[str, torch.Tensor] | None = None,
        basis_intervention: str = "normal",
    ) -> dict[str, object]:
        if edge_index is None:
            raise ValueError("adaptive_relation_pilot requires the physical edge_index")
        if x.dim() != 2 or x.size(-1) != self.input_dim:
            raise ValueError(f"Expected x with shape [num_nodes, {self.input_dim}]")
        p_rel, row_sum = self._get_operators(edge_index.to(x.device), int(x.size(0)), x.dtype)
        h_text = self.text_projector(x[:, :self.text_dim])
        h_visual = self.visual_projector(x[:, self.text_dim:self.text_dim + self.visual_dim])
        beta_overrides = beta_overrides or {}
        lambda_overrides = lambda_overrides or {}
        text = self._one_modality("text", h_text, p_rel, row_sum,
                                  beta_overrides.get("text"), lambda_overrides.get("text"), basis_intervention)
        visual = self._one_modality("visual", h_visual, p_rel, row_sum,
                                    beta_overrides.get("visual"), lambda_overrides.get("visual"), basis_intervention)
        fused_z = self.plain_fusion(torch.cat([text["Z"], visual["Z"]], dim=-1))
        result: dict[str, object] = {"P_rel": p_rel, "s": row_sum, "fused_z": fused_z}
        for modality, values in (("text", text), ("visual", visual)):
            for key, value in values.items():
                result[f"{key}_{modality}"] = value
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
        return self.forward(x.to(device), edge_index.to(device))[0].detach().cpu()
