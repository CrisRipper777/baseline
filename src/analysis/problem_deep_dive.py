from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch
from torch_geometric.utils import add_self_loops, coalesce, remove_self_loops, to_undirected


def canonical_pairs(edge_index: torch.Tensor) -> torch.Tensor:
    edge = edge_index.detach().long().cpu()
    rows = edge.T.contiguous() if edge.size(0) == 2 else edge.contiguous()
    if rows.ndim != 2 or rows.size(1) != 2:
        raise ValueError("edge_index must have shape [2,E] or [E,2]")
    rows = torch.stack([torch.minimum(rows[:, 0], rows[:, 1]), torch.maximum(rows[:, 0], rows[:, 1])], 1)
    rows = rows[rows[:, 0] != rows[:, 1]]
    return torch.unique(rows, dim=0, sorted=True)


def _percentile_rank(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    order = torch.argsort(values, stable=True)
    sorted_v = values[order]
    n = len(values)
    ranks = torch.empty(n, dtype=torch.float32)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_v[j] == sorted_v[i]:
            j += 1
        # Average empirical rank in [0,1]; ties receive the midpoint rank.
        ranks[order[i:j]] = ((i + j - 1) / 2) / max(n - 1, 1)
        i = j
    return ranks


def structural_edge_descriptors(edge_index: torch.Tensor, num_nodes: int,
                                labels: torch.Tensor | None = None,
                                known_idx: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    pairs = canonical_pairs(edge_index)
    adj: list[set[int]] = [set() for _ in range(num_nodes)]
    for i, j in pairs.tolist():
        adj[i].add(j); adj[j].add(i)
    degree = torch.tensor([len(neighbors) for neighbors in adj], dtype=torch.long)
    degree_q = _percentile_rank(degree)
    common, jaccard, aa, embedded = [], [], [], []
    for i, j in pairs.tolist():
        shared = adj[i].intersection(adj[j])
        c = len(shared)
        union = len(adj[i]) + len(adj[j]) - c
        common.append(c)
        jaccard.append(c / union if union else float("nan"))
        aa.append(sum(1.0 / math.log(max(len(adj[k]), 2)) for k in shared))
        denom = min(max(len(adj[i]) - 1, 0), max(len(adj[j]) - 1, 0))
        embedded.append(c / denom if denom else float("nan"))
    same = torch.full((pairs.size(0),), float("nan"), dtype=torch.float32)
    if labels is not None and known_idx is not None:
        known = torch.zeros(num_nodes, dtype=torch.bool)
        known[known_idx.detach().long().cpu()] = True
        y = torch.full((num_nodes,), -1, dtype=torch.long)
        ids = known_idx.detach().long().cpu()
        y[ids] = labels.detach().long().cpu()[ids]
        valid = known[pairs[:, 0]] & known[pairs[:, 1]]
        same[valid] = (y[pairs[valid, 0]] == y[pairs[valid, 1]]).float()
    return {
        "pairs": pairs,
        "degree_i": degree[pairs[:, 0]].float(),
        "degree_j": degree[pairs[:, 1]].float(),
        "degree_quantile_i": degree_q[pairs[:, 0]],
        "degree_quantile_j": degree_q[pairs[:, 1]],
        "common_neighbors": torch.tensor(common, dtype=torch.float32),
        "neighbor_jaccard": torch.tensor(jaccard, dtype=torch.float32),
        "adamic_adar": torch.tensor(aa, dtype=torch.float32),
        # Embeddedness is the fraction of the smaller endpoint's other neighbors shared across the edge.
        "edge_embeddedness": torch.tensor(embedded, dtype=torch.float32),
        "same_label_edge": same,
        "degree": degree.float(),
        "degree_quantile": degree_q,
    }


def context_novelty(states: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """Per undirected edge: mean of two directional 1-cosine context novelty values.

    Each direction uses the final-layer neighbor embeddings and excludes the target
    neighbor from the receiving node's context mean. Degree <= 1 produces NaN.
    """
    h = states.detach().float().cpu()
    pairs = pairs.detach().long().cpu()
    n = h.size(0)
    degree = torch.zeros(n, dtype=torch.long)
    neighbor_sum = torch.zeros_like(h)
    for src_col, dst_col in ((pairs[:, 0], pairs[:, 1]), (pairs[:, 1], pairs[:, 0])):
        degree.index_add_(0, src_col, torch.ones_like(src_col))
        neighbor_sum.index_add_(0, src_col, h[dst_col])
    outputs = []
    for src, dst in ((pairs[:, 0], pairs[:, 1]), (pairs[:, 1], pairs[:, 0])):
        denom = degree[src] - 1
        valid = denom > 0
        values = torch.full((pairs.size(0),), float("nan"))
        if valid.any():
            context = (neighbor_sum[src[valid]] - h[dst[valid]]) / denom[valid, None]
            values[valid] = 1.0 - torch.nn.functional.cosine_similarity(h[dst[valid]], context, dim=-1)
        outputs.append(values)
    return torch.stack(outputs).nanmean(0)


def train_edge_quartile_thresholds(values: torch.Tensor, pairs: torch.Tensor,
                                   train_idx: torch.Tensor) -> dict[str, float]:
    in_train = torch.zeros(int(max(int(pairs.max()) + 1 if pairs.numel() else 0,
                                   int(train_idx.max()) + 1 if train_idx.numel() else 0)), dtype=torch.bool)
    in_train[train_idx.detach().long().cpu()] = True
    valid = in_train[pairs[:, 0]] & in_train[pairs[:, 1]] & torch.isfinite(values)
    train_values = values[valid]
    if train_values.numel() == 0:
        return {"q25": float("nan"), "q75": float("nan"), "train_edge_count": 0}
    return {"q25": float(torch.quantile(train_values, 0.25)),
            "q50": float(torch.quantile(train_values, 0.50)),
            "q75": float(torch.quantile(train_values, 0.75)),
            "train_edge_count": int(train_values.numel())}


def train_edge_quantile_bins(values: torch.Tensor, pairs: torch.Tensor,
                             train_idx: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """Assign four bins using train-edge q25/q50/q75 thresholds only."""
    thresholds = train_edge_quartile_thresholds(values, pairs, train_idx)
    result = torch.full(values.shape, -1, dtype=torch.long)
    if thresholds["train_edge_count"] == 0:
        return result, thresholds
    valid = torch.isfinite(values)
    cuts = torch.tensor([thresholds["q25"], thresholds["q50"], thresholds["q75"]], dtype=values.dtype)
    result[valid] = torch.bucketize(values[valid].contiguous(), cuts.contiguous(), right=False)
    return result, thresholds


def four_regimes(similarity: torch.Tensor, novelty: torch.Tensor,
                 sim_thresholds: dict[str, float], nov_thresholds: dict[str, float]) -> torch.Tensor:
    result = torch.full((similarity.numel(),), -1, dtype=torch.long)
    low_s = similarity <= sim_thresholds["q25"]
    high_s = similarity >= sim_thresholds["q75"]
    low_n = novelty <= nov_thresholds["q25"]
    high_n = novelty >= nov_thresholds["q75"]
    valid = torch.isfinite(similarity) & torch.isfinite(novelty)
    result[valid & high_s & low_n] = 0  # high similarity, low novelty
    result[valid & high_s & high_n] = 1
    result[valid & low_s & high_n] = 2  # low similarity, high novelty
    result[valid & low_s & low_n] = 3
    return result


def normalized_physical_operator(edge_index: torch.Tensor, num_nodes: int,
                                 dtype: torch.dtype = torch.float32,
                                 device: torch.device | str = "cpu") -> torch.Tensor:
    edge = edge_index.detach().to(device=device, dtype=torch.long)
    edge, _ = remove_self_loops(edge)
    edge = to_undirected(edge, num_nodes=num_nodes)
    edge, _ = add_self_loops(edge, num_nodes=num_nodes)
    edge = coalesce(edge, num_nodes=num_nodes)
    row, col = edge
    degree = torch.zeros(num_nodes, dtype=dtype, device=device)
    degree.index_add_(0, row, torch.ones(row.numel(), dtype=dtype, device=device))
    inv = degree.clamp_min(1).pow(-0.5)
    weight = inv[row] * inv[col]
    return torch.sparse_coo_tensor(edge, weight, (num_nodes, num_nodes), device=device, dtype=dtype).coalesce()


def _canonical_pair_rows(pair_values: torch.Tensor) -> torch.Tensor:
    pair_values = pair_values.detach().long().cpu()
    if pair_values.ndim != 2:
        raise ValueError("target pairs must be rank-2")
    if pair_values.size(1) == 2:
        rows = pair_values
    elif pair_values.size(0) == 2:
        rows = pair_values.T.contiguous()
    else:
        raise ValueError("target pairs must have shape [E,2] or [2,E]")
    rows = torch.stack([torch.minimum(rows[:, 0], rows[:, 1]),
                        torch.maximum(rows[:, 0], rows[:, 1])], dim=1)
    rows = rows[rows[:, 0] != rows[:, 1]]
    return torch.unique(rows, dim=0, sorted=True)


def fixed_norm_mask(operator: torch.Tensor, target_pairs: torch.Tensor) -> torch.Tensor:
    """Zero both normalized target directions without changing any other P entry."""
    op = operator.coalesce()
    idx, val = op.indices(), op.values().clone()
    pairs = _canonical_pair_rows(target_pairs)
    target_keys = torch.cat([
        pairs[:, 0] * op.size(1) + pairs[:, 1],
        pairs[:, 1] * op.size(1) + pairs[:, 0],
    ]).to(idx.device)
    edge_keys = idx[0] * op.size(1) + idx[1]
    val[torch.isin(edge_keys, target_keys)] = 0
    return torch.sparse_coo_tensor(idx, val, op.shape, device=val.device, dtype=val.dtype).coalesce()


def _stratum_counts(indices: torch.Tensor, degree_pair_bin: torch.Tensor,
                    weight_bin: torch.Tensor) -> Counter:
    out = Counter()
    for idx in indices.tolist():
        a, b = map(int, degree_pair_bin[idx])
        out[(min(a, b), max(a, b), int(weight_bin[idx]))] += 1
    return out


def matched_random_edge_sets(pairs: torch.Tensor, target_indices: torch.Tensor,
                             degree_pair_bin: torch.Tensor, weight_bin: torch.Tensor,
                             repeats: int = 20, seed: int = 0) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Match edge count and both required marginals exactly, or fail explicitly.

    A small integral max-flow allocates candidate edges across degree-pair ×
    normalized-weight cells. The sampled controls therefore match each marginal
    without requiring the stronger and often infeasible joint distribution.
    """
    pairs = pairs.detach().long().cpu()
    target_indices = target_indices.detach().long().cpu().unique(sorted=True)
    n = pairs.size(0)
    target_mask = torch.zeros(n, dtype=torch.bool); target_mask[target_indices] = True
    target_degree = Counter()
    target_weight = Counter()
    pools: dict[tuple[tuple[int, int], int], list[int]] = defaultdict(list)
    for idx in target_indices.tolist():
        a, b = map(int, degree_pair_bin[idx])
        target_degree[(min(a, b), max(a, b))] += 1
        target_weight[int(weight_bin[idx])] += 1
    for idx in torch.where(~target_mask)[0].tolist():
        a, b = map(int, degree_pair_bin[idx])
        pools[((min(a, b), max(a, b)), int(weight_bin[idx]))].append(idx)
    degree_keys = sorted(target_degree)
    weight_keys = sorted(target_weight)
    source = 0
    deg_start = 1
    weight_start = deg_start + len(degree_keys)
    sink = weight_start + len(weight_keys)
    graph: list[list[list[int]]] = [[] for _ in range(sink + 1)]
    def add_edge(u: int, v: int, capacity: int):
        forward = [v, len(graph[v]), capacity, capacity]
        reverse = [u, len(graph[u]), 0, 0]
        graph[u].append(forward); graph[v].append(reverse)
        return forward
    for i, key in enumerate(degree_keys):
        add_edge(source, deg_start + i, int(target_degree[key]))
    for j, key in enumerate(weight_keys):
        add_edge(weight_start + j, sink, int(target_weight[key]))
    cell_edges = {}
    for i, dkey in enumerate(degree_keys):
        for j, wkey in enumerate(weight_keys):
            available = len(pools.get((dkey, wkey), []))
            if available:
                cell_edges[(dkey, wkey)] = add_edge(deg_start + i, weight_start + j, available)
    total = int(target_indices.numel())
    flow = 0
    while True:
        level = [-1] * len(graph); level[source] = 0
        queue = [source]
        for u in queue:
            for v, _, cap, _ in graph[u]:
                if cap > 0 and level[v] < 0:
                    level[v] = level[u] + 1; queue.append(v)
        if level[sink] < 0:
            break
        cursor = [0] * len(graph)
        def send(u: int, amount: int) -> int:
            if u == sink:
                return amount
            while cursor[u] < len(graph[u]):
                edge = graph[u][cursor[u]]
                v, rev, cap, _ = edge
                if cap > 0 and level[v] == level[u] + 1:
                    pushed = send(v, min(amount, cap))
                    if pushed:
                        edge[2] -= pushed; graph[v][rev][2] += pushed
                        return pushed
                cursor[u] += 1
            return 0
        while True:
            pushed = send(source, total - flow)
            if not pushed:
                break
            flow += pushed
            if flow == total:
                break
        if flow == total:
            break
    allocation = {}
    if flow == total:
        for key, edge in cell_edges.items():
            used = edge[3] - edge[2]
            if used:
                allocation[key] = used
    shortages = {} if flow == total else {
        "max_matched": flow, "required": total,
        "degree_pair_shortfall_by_bin": str(target_degree),
        "weight_quantile_shortfall_by_bin": str(target_weight),
    }
    diag: dict[str, Any] = {
        "status": "MATCHED" if flow == total and total else "MATCHING_FAILED" if total else "EMPTY_TARGET_GROUP",
        "target_edge_count": total, "matched_edge_count_per_control": total if flow == total else flow,
        "target_degree_pair_marginal": str(dict(target_degree)),
        "target_weight_quantile_marginal": str(dict(target_weight)),
        "allocated_cells": len(allocation), "allocation": {str(k): v for k, v in allocation.items()},
        "shortages": shortages, "degree_pair_exact": flow == total and total > 0,
        "weight_quantile_exact": flow == total and total > 0,
        "matching_method": "integral max-flow with exact endpoint-degree-quantile-pair and base normalized edge-weight quantile marginals",
    }
    if flow != total or total == 0:
        return [], diag
    rng = random.Random(seed)
    controls = []
    for _ in range(repeats):
        chosen = []
        for key, count in sorted(allocation.items()):
            chosen.extend(rng.sample(pools[key], count))
        chosen_t = torch.tensor(sorted(chosen), dtype=torch.long)
        if Counter((min(map(int, degree_pair_bin[i])), max(map(int, degree_pair_bin[i]))) for i in chosen) != target_degree:
            raise RuntimeError("matched control degree-pair distribution failed verification")
        if Counter(int(weight_bin[i]) for i in chosen) != target_weight:
            raise RuntimeError("matched control normalized-weight quantiles failed verification")
        controls.append(pairs[chosen_t])
    diag["verified_exact_marginals"] = True
    return controls, diag


def remove_pairs_vectorized(edge_index: torch.Tensor, target_pairs: torch.Tensor,
                            num_nodes: int | None = None) -> torch.Tensor:
    """Remove both directions of target pairs while preserving self-loops."""
    edge = edge_index.detach().long()
    pairs = _canonical_pair_rows(target_pairs)
    if pairs.numel() == 0 or edge.numel() == 0:
        return edge.clone()
    n = int(num_nodes if num_nodes is not None else int(edge.max()) + 1)
    keys = pairs[:, 0] * n + pairs[:, 1]
    src, dst = edge
    edge_keys = torch.minimum(src, dst) * n + torch.maximum(src, dst)
    keys, _ = torch.sort(keys)
    pos = torch.searchsorted(keys, edge_keys)
    inside = pos < keys.numel()
    clipped = pos.clamp(max=keys.numel() - 1)
    matched = inside & (keys[clipped] == edge_keys) & (src != dst)
    return edge[:, ~matched]


def build_csr_adjacency(edge_index: torch.Tensor, num_nodes: int):
    """Build a simple undirected CSR adjacency for repeated local-effect queries."""
    import scipy.sparse as sp
    pairs = canonical_pairs(edge_index).numpy()
    if pairs.size == 0:
        return sp.csr_matrix((num_nodes, num_nodes), dtype=np.int8)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    return sp.csr_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(num_nodes, num_nodes))


def affected_within_hops(edge_index: torch.Tensor, seed_nodes: torch.Tensor,
                         num_nodes: int, hops: int = 3, adjacency=None) -> torch.Tensor:
    """Return nodes within ``hops`` of targets using sparse breadth expansion."""
    if adjacency is None:
        adjacency = build_csr_adjacency(edge_index, num_nodes)
    affected = np.zeros(num_nodes, dtype=bool)
    frontier = np.unique(seed_nodes.detach().long().cpu().numpy())
    frontier = frontier[(frontier >= 0) & (frontier < num_nodes)]
    affected[frontier] = True
    for _ in range(hops):
        if frontier.size == 0:
            break
        neighbors = adjacency[frontier].indices
        frontier = np.unique(neighbors[~affected[neighbors]])
        affected[frontier] = True
    return torch.from_numpy(affected)


def source_matched_compatibility(states: torch.Tensor, edge_index: torch.Tensor, seed: int,
                                 max_sources: int = 100000, nonedges_per_source: int = 32,
                                 source_batch_size: int = 2048) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Vectorized implementation of D2's exact deterministic source-matched percentile definition.

    It retains the same sorted active-source selection, torch Generator, rejection
    batches, and right-sided empirical percentile calculation as
    `edge_percentile_compatibility`, while batching cosine calculations.
    """
    x = states.detach().float().cpu().contiguous()
    n = x.size(0)
    pairs = canonical_pairs(edge_index)
    adj: list[set[int]] = [set() for _ in range(n)]
    for a, b in pairs.tolist():
        adj[a].add(b); adj[b].add(a)
    active = torch.tensor([i for i, neighbors in enumerate(adj) if neighbors], dtype=torch.long)
    generator = torch.Generator().manual_seed(int(seed))
    if active.numel() > max_sources:
        active = active[torch.randperm(active.numel(), generator=generator)[:max_sources]].sort().values
    refs = torch.empty((active.numel(), nonedges_per_source), dtype=torch.long)
    for row, src in enumerate(active.tolist()):
        selected: list[int] = []
        tries = 0
        while len(selected) < nonedges_per_source and tries < nonedges_per_source * 100:
            candidates = torch.randint(n, (max(32, (nonedges_per_source - len(selected)) * 4),),
                                       generator=generator).tolist()
            tries += len(candidates)
            for dst in candidates:
                if dst != src and dst not in adj[src] and dst not in selected:
                    selected.append(dst)
                    if len(selected) == nonedges_per_source:
                        break
        if len(selected) != nonedges_per_source:
            raise RuntimeError(f"source {src} could not draw {nonedges_per_source} nonedges")
        refs[row] = torch.tensor(selected, dtype=torch.long)
    normed = torch.nn.functional.normalize(x, p=2, dim=-1, eps=1e-8)
    ref_sorted = torch.empty((active.numel(), nonedges_per_source), dtype=torch.float32)
    for start in range(0, active.numel(), source_batch_size):
        end = min(start + source_batch_size, active.numel())
        src = active[start:end]
        sim = (normed[src, None, :] * normed[refs[start:end]]).sum(-1)
        ref_sorted[start:end] = sim.sort(dim=1).values
    source_row = torch.full((n,), -1, dtype=torch.long)
    source_row[active] = torch.arange(active.numel())
    directed = torch.cat([pairs, pairs.flip(1)], dim=0)
    keys_all, values_all = [], []
    for start in range(0, active.numel(), source_batch_size):
        end = min(start + source_batch_size, active.numel())
        row_for_edge = source_row[directed[:, 0]]
        selected = (row_for_edge >= start) & (row_for_edge < end)
        if not selected.any():
            continue
        edges = directed[selected]
        rows = row_for_edge[selected]
        scores = (normed[edges[:, 0]] * normed[edges[:, 1]]).sum(-1)
        percentiles = (ref_sorted[rows] <= scores[:, None]).sum(-1).float() / nonedges_per_source
        keys_all.append(torch.minimum(edges[:, 0], edges[:, 1]) * n + torch.maximum(edges[:, 0], edges[:, 1]))
        values_all.append(percentiles)
    if not keys_all:
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0), {
            "physical_undirected_edges": int(pairs.size(0)), "sampled_source_nodes": 0,
            "sampled_edge_pairs": 0, "nonedge_reference_scores": 0,
        }
    keys, values = torch.cat(keys_all), torch.cat(values_all)
    unique, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    sums = torch.zeros(unique.numel()).index_add_(0, inverse, values)
    counts = torch.zeros(unique.numel()).index_add_(0, inverse, torch.ones_like(values))
    out_pairs = torch.stack([unique // n, unique % n])
    return out_pairs, sums / counts, {
        "physical_undirected_edges": int(pairs.size(0)), "sampled_source_nodes": int(active.numel()),
        "sampled_edge_pairs": int(out_pairs.size(1)),
        "nonedge_reference_scores": int(active.numel() * nonedges_per_source),
    }


def fit_train_only_logistic(features: torch.Tensor, target: torch.Tensor,
                            train_idx: torch.Tensor, val_idx: torch.Tensor,
                            random_state: int = 0) -> dict[str, Any]:
    """Fit median imputation, scaling, and binary logistic regression on train rows only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x = features.detach().float().cpu().numpy()
    y = target.detach().long().cpu().numpy()
    tr = train_idx.detach().long().cpu().numpy()
    va = val_idx.detach().long().cpu().numpy()
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("features and target must align as [nodes, features] and [nodes]")
    y_train = y[tr]
    keep = y_train >= 0
    tr = tr[keep]
    y_train = y_train[keep]
    if len(np.unique(y_train)) != 2:
        raise ValueError("train-only logistic fit requires both target classes")
    x_train = x[tr]
    x_val = x[va]
    medians = np.zeros(x.shape[1], dtype=np.float64)
    for col in range(x.shape[1]):
        finite = np.isfinite(x_train[:, col])
        medians[col] = np.median(x_train[finite, col]) if finite.any() else 0.0
    x_train = np.where(np.isfinite(x_train), x_train, medians)
    x_val = np.where(np.isfinite(x_val), x_val, medians)
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(max_iter=1000, random_state=int(random_state))
    model.fit(scaler.transform(x_train), y_train)
    pred = model.predict(scaler.transform(x_val))
    prob = model.predict_proba(scaler.transform(x_val))[:, 1]
    return {"model": model, "scaler": scaler, "medians": medians,
            "val_pred": pred, "val_probability": prob, "train_indices_used": tr.copy()}
