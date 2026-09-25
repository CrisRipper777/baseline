from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from scipy.stats import spearmanr


@dataclass(frozen=True)
class FrozenProbeConfig:
    name: str = "FrozenProbeProtocolV1"
    optimizer: str = "AdamW"
    lr: float = 1e-2
    weight_decay: float = 1e-4
    epochs: int = 200
    early_stopping: bool = False
    scheduler: str | None = None
    seed: int = 0
    fit_split: str = "train"
    report_split: str = "validation"
    test_access: bool = False

    def sha256(self) -> str:
        value = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(value.encode()).hexdigest()


PROBE_CONFIG = FrozenProbeConfig()


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor, class_ids: list[int]) -> dict[str, float]:
    pred = logits.argmax(dim=-1).detach().cpu().numpy()
    target = labels.detach().cpu().numpy()
    return {
        "accuracy": float(accuracy_score(target, pred)),
        "macro_f1": float(f1_score(target, pred, labels=class_ids, average="macro", zero_division=0)),
    }


def node_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, labels.long(), reduction="none")


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.float().clamp_min(eps)
    q = q.float().clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    m = 0.5 * (p + q)
    return 0.5 * (p * (p.log() - m.log())).sum(dim=-1) + 0.5 * (q * (q.log() - m.log())).sum(dim=-1)


def quantile_bins(values: torch.Tensor, bins: int = 4) -> torch.Tensor:
    """Stable rank bins, with ties kept together at right-side boundaries."""
    values = values.float()
    valid = torch.isfinite(values)
    out = torch.full(values.shape, -1, dtype=torch.long, device=values.device)
    if int(valid.sum()) == 0:
        return out
    thresholds = torch.quantile(values[valid], torch.arange(1, bins, device=values.device) / bins)
    out[valid] = torch.bucketize(values[valid].contiguous(), thresholds.contiguous(), right=False)
    return out


class FrozenProbeProtocolV1:
    """A fixed full-batch linear probe trained only on the caller's train indices."""

    config = PROBE_CONFIG

    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        train_idx: torch.Tensor,
        val_idx: torch.Tensor,
        class_ids: list[int],
        device: torch.device | str,
    ) -> dict[str, Any]:
        if train_idx.numel() == 0 or val_idx.numel() == 0:
            raise ValueError("Frozen probes require nonempty train and validation splits")
        # Explicitly retain only train labels for optimization and validation labels for reporting.
        train_idx = train_idx.long().cpu()
        val_idx = val_idx.long().cpu()
        y_train = labels[train_idx].long().to(device)
        y_val = labels[val_idx].long().to(device)
        x_train = features[train_idx].float().to(device)
        x_val = features[val_idx].float().to(device)
        if not torch.isfinite(x_train).all() or not torch.isfinite(x_val).all():
            raise FloatingPointError("FrozenProbeProtocolV1 received nonfinite features")
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.manual_seed_all(self.config.seed)
        head = nn.Linear(int(features.size(-1)), max(class_ids) + 1).to(device)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        head.train()
        for _ in range(self.config.epochs):
            optimizer.zero_grad(set_to_none=True)
            logits = head(x_train)
            loss = nn.functional.cross_entropy(logits, y_train)
            if not torch.isfinite(loss):
                raise FloatingPointError("FrozenProbeProtocolV1 optimization diverged")
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            train_logits = head(x_train).detach().cpu()
            val_logits = head(x_val).detach().cpu()
        if not torch.isfinite(val_logits).all():
            raise FloatingPointError("FrozenProbeProtocolV1 produced nonfinite validation logits")
        return {
            "head": head.cpu(),
            "train_logits": train_logits,
            "val_logits": val_logits,
            "train_metrics": classification_metrics(train_logits, labels[train_idx], class_ids),
            "val_metrics": classification_metrics(val_logits, labels[val_idx], class_ids),
            "config": asdict(self.config),
            "config_sha256": self.config.sha256(),
            "trainable_parameters": sum(p.numel() for p in head.parameters()),
        }


def geometry_summary(states: list[torch.Tensor]) -> list[dict[str, float]]:
    """Compute scale, centered covariance rank, and normalized Dirichlet terms."""
    rows: list[dict[str, float]] = []
    for order, state in enumerate(states):
        x = state.detach().float().cpu()
        mean = x.mean(dim=0, keepdim=True)
        centered = x - mean
        covariance = centered.T @ centered / max(x.size(0) - 1, 1)
        eig = torch.linalg.eigvalsh(covariance).clamp_min(0)
        total = eig.sum()
        pr = float((total.square() / eig.square().sum().clamp_min(1e-20)).item())
        generator = torch.Generator().manual_seed(0)
        sample_idx = torch.randperm(x.size(0), generator=generator)[:min(512, x.size(0))]
        normalized = nn.functional.normalize(x[sample_idx], dim=-1)
        similarity = normalized @ normalized.T
        if similarity.size(0) > 1:
            pair_mask = ~torch.eye(similarity.size(0), dtype=torch.bool)
            pairwise_node_cosine = float(similarity[pair_mask].mean().item())
        else:
            pairwise_node_cosine = float("nan")
        row = {
            "order": order,
            "mean_node_l2_norm": float(x.norm(dim=-1).mean().item()),
            "node_variance": float(centered.square().sum(dim=-1).mean().item()),
            "effective_rank": pr,
            "mean_pairwise_node_cosine": pairwise_node_cosine,
            "node_similarity_sample_size": int(sample_idx.numel()),
            "node_similarity_sampling": "fixed seed 0 node subsample",
            "normalized_dirichlet_energy": float("nan"),
        }
        rows.append(row)
    return rows


def cosine_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(nn.functional.cosine_similarity(a.reshape(1, -1).float(), b.reshape(1, -1).float()).item())


def normalized_dirichlet_energy(state: torch.Tensor, edge_index: torch.Tensor) -> float:
    """Tr(X^T L X)/||X||² for the same symmetric-normalized A+I operator."""
    x = state.detach().float()
    edge = edge_index.long().cpu()
    n = x.size(0)
    edge = torch.cat([edge, edge.flip(0)], dim=1)
    edge = edge[:, edge[0] != edge[1]]
    edge = torch.unique(edge.T, dim=0).T
    loops = torch.arange(n, dtype=torch.long).repeat(2, 1)
    edge = torch.cat([edge, loops], dim=1)
    row, col = edge
    degree = torch.zeros(n).index_add_(0, row, torch.ones(row.numel()))
    inv = degree.clamp_min(1).pow(-0.5)
    w = inv[row] * inv[col]
    p_x = torch.zeros_like(x)
    p_x.index_add_(0, row, w[:, None] * x[col])
    energy = ((x * (x - p_x)).sum()).item()
    return float(energy / max(float(x.square().sum().item()), 1e-20))


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float() - x.float().mean(0, keepdim=True)
    y = y.float() - y.float().mean(0, keepdim=True)
    xy = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    return float((xy.square().sum() / (xx.square().sum().sqrt() * yy.square().sum().sqrt()).clamp_min(1e-20)).item())


def cca_spectrum_summary(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-4) -> dict[str, float]:
    """CCA canonical correlations after ridge-stabilized covariance whitening."""
    x = x.float() - x.float().mean(0, keepdim=True)
    y = y.float() - y.float().mean(0, keepdim=True)
    n = max(x.size(0) - 1, 1)
    cxx = x.T @ x / n + eps * torch.eye(x.size(1))
    cyy = y.T @ y / n + eps * torch.eye(y.size(1))
    cxy = x.T @ y / n
    ex, ux = torch.linalg.eigh(cxx)
    ey, uy = torch.linalg.eigh(cyy)
    wx = (ux * ex.clamp_min(eps).rsqrt()[None, :]) @ ux.T
    wy = (uy * ey.clamp_min(eps).rsqrt()[None, :]) @ uy.T
    corr = torch.linalg.svdvals(wx @ cxy @ wy).clamp(0, 1)
    return {
        "cca_mean": float(corr.mean().item()),
        "cca_top10_mean": float(corr[: min(10, corr.numel())].mean().item()),
        "cca_spectrum_sum": float(corr.sum().item()),
    }


def orthogonal_procrustes_fit(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source = source.float() - source.float().mean(0, keepdim=True)
    target_mean = target.float().mean(0, keepdim=True)
    target = target.float() - target_mean
    u, _, vh = torch.linalg.svd(source.T @ target, full_matrices=False)
    return u @ vh


def edge_percentile_compatibility(
    states: torch.Tensor,
    edge_index: torch.Tensor,
    seed: int,
    max_sources: int = 100000,
    nonedges_per_source: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Compute source-matched empirical edge percentiles with deterministic bounded sampling."""
    x = states.detach().float().cpu()
    n = x.size(0)
    pairs = canonical_undirected_pairs(edge_index)
    adj: list[set[int]] = [set() for _ in range(n)]
    for a, b in pairs.tolist():
        adj[a].add(b)
        adj[b].add(a)
    active_sources = torch.tensor([i for i, neighbors in enumerate(adj) if neighbors], dtype=torch.long)
    gen = torch.Generator().manual_seed(int(seed))
    if active_sources.numel() > max_sources:
        active_sources = active_sources[torch.randperm(active_sources.numel(), generator=gen)[:max_sources]].sort().values
    scored_keys: list[torch.Tensor] = []
    scored_values: list[torch.Tensor] = []
    refs_count = 0
    for src in active_sources.tolist():
        vals: list[int] = []
        tries = 0
        while len(vals) < nonedges_per_source and tries < nonedges_per_source * 100:
            cand = torch.randint(n, (max(32, (nonedges_per_source - len(vals)) * 4),), generator=gen).tolist()
            tries += len(cand)
            for dst in cand:
                if dst != src and dst not in adj[src] and dst not in vals:
                    vals.append(dst)
                    if len(vals) == nonedges_per_source:
                        break
        if not vals:
            continue
        ref_ids = torch.tensor(vals, dtype=torch.long)
        ref_scores = nn.functional.cosine_similarity(x[src][None], x[ref_ids], dim=-1).sort().values
        refs_count += ref_scores.numel()
        neighbors = torch.tensor(sorted(adj[src]), dtype=torch.long)
        edge_scores = nn.functional.cosine_similarity(x[src][None], x[neighbors], dim=-1)
        percentiles = torch.searchsorted(ref_scores, edge_scores.contiguous(), right=True).float() / ref_scores.numel()
        keys = torch.minimum(neighbors, torch.tensor(src)) * n + torch.maximum(neighbors, torch.tensor(src))
        scored_keys.append(keys)
        scored_values.append(percentiles)
    if not scored_keys:
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0), {
            "physical_undirected_edges": int(pairs.size(0)), "sampled_source_nodes": 0,
            "sampled_edge_pairs": 0, "nonedge_reference_scores": 0,
        }
    keys = torch.cat(scored_keys)
    values = torch.cat(scored_values)
    unique, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    sums = torch.zeros(unique.numel()).index_add_(0, inverse, values)
    counts = torch.zeros(unique.numel()).index_add_(0, inverse, torch.ones_like(values))
    means = sums / counts
    edge_pairs = torch.stack([unique // n, unique % n])
    return edge_pairs, means, {
        "physical_undirected_edges": int(pairs.size(0)),
        "sampled_source_nodes": int(active_sources.numel()),
        "sampled_edge_pairs": int(edge_pairs.size(1)),
        "nonedge_reference_scores": int(refs_count),
    }


def classwise_metrics(logits: torch.Tensor, labels: torch.Tensor, class_ids: list[int]) -> list[dict[str, float | int]]:
    pred = logits.argmax(-1).detach().cpu().numpy()
    target = labels.detach().cpu().numpy()
    precision, recall, f1, support = precision_recall_fscore_support(
        target, pred, labels=class_ids, zero_division=0
    )
    return [
        {"class_id": int(c), "support": int(s), "accuracy": float(recall[i]),
         "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
        for i, (c, s) in enumerate(zip(class_ids, support, strict=True))
    ]


def safe_spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    mask = torch.isfinite(x) & torch.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    return float(spearmanr(x[mask].detach().cpu().numpy(), y[mask].detach().cpu().numpy()).statistic)


def canonical_undirected_pairs(edge_index: torch.Tensor) -> torch.Tensor:
    edge = edge_index.detach().long().cpu()
    if edge.ndim != 2:
        raise ValueError("edge pairs must be a rank-2 tensor")
    if edge.size(0) == 2:
        rows = edge.T.contiguous()
    elif edge.size(1) == 2:
        rows = edge.contiguous()
    else:
        raise ValueError("edge pairs must have shape [2, E] or [E, 2]")
    lo = torch.minimum(rows[:, 0], rows[:, 1])
    hi = torch.maximum(rows[:, 0], rows[:, 1])
    rows = torch.stack([lo, hi], dim=1)
    rows = rows[rows[:, 0] != rows[:, 1]]
    return torch.unique(rows, dim=0)


def remove_undirected_pairs(edge_index: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """Remove both directions of each listed undirected pair; keep self-loops."""
    edge = edge_index.long()
    pair_tensor = pairs.detach().long().cpu()
    if pair_tensor.ndim == 2 and pair_tensor.size(1) == 2:
        rows = pair_tensor
        lo = torch.minimum(rows[:, 0], rows[:, 1])
        hi = torch.maximum(rows[:, 0], rows[:, 1])
        rows = torch.stack([lo, hi], dim=1)
        pair_set = {tuple(map(int, row)) for row in rows[rows[:, 0] != rows[:, 1]].tolist()}
    else:
        pair_set = {tuple(map(int, row)) for row in canonical_undirected_pairs(pair_tensor).tolist()}
    if not pair_set:
        return edge.clone()
    src, dst = edge.detach().cpu()
    keep = torch.tensor(
        [int(a) == int(b) or (min(int(a), int(b)), max(int(a), int(b))) not in pair_set
         for a, b in zip(src.tolist(), dst.tolist(), strict=True)],
        dtype=torch.bool, device=edge.device,
    )
    return edge[:, keep]


def remove_random_same_size(edge_index: torch.Tensor, target_pairs: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = canonical_undirected_pairs(edge_index)
    count = min(max(int(target_pairs), 0), int(pairs.size(0)))
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(pairs.size(0), generator=generator)[:count]
    selected = pairs[order]
    return remove_undirected_pairs(edge_index, selected), selected


def train_median_threshold(train_values: torch.Tensor) -> float:
    values = train_values[torch.isfinite(train_values)].float()
    if values.numel() == 0:
        return float("nan")
    return float(torch.quantile(values, 0.5).item())
