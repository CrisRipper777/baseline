from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")


def _load_graph(dataset: str) -> tuple[torch.Tensor, int]:
    from src.data import load_mag_data
    with initialize_config_dir(version_base=None, config_dir=str((ROOT / "configs").resolve())):
        cfg = compose(config_name="config", overrides=[f"dataset={dataset}", "task=nc", "model=mlp", "seed=42"])
    data = load_mag_data(cfg, "nc", 42)
    edge = data.edge_index.detach().long().cpu()
    edge = edge[:, edge[0] != edge[1]]
    pairs = torch.sort(edge.T.contiguous(), dim=1).values
    return torch.unique(pairs, dim=0, sorted=True), int(data.num_nodes)


def _degree(pairs: torch.Tensor, num_nodes: int) -> torch.Tensor:
    return torch.bincount(pairs.reshape(-1), minlength=num_nodes)


def _rewire(pairs: torch.Tensor, num_nodes: int, seed: int, target_factor: int = 10,
            attempt_factor: int = 100) -> tuple[torch.Tensor, int, int, float]:
    edges = [tuple(map(int, pair)) for pair in pairs.tolist()]
    edge_set = set(edges)
    edge_count = len(edges)
    target = target_factor * edge_count
    attempt_limit = max(attempt_factor * edge_count, 100_000)
    rng = random.Random(seed)
    successful = attempts = 0
    started = time.monotonic()
    while successful < target and attempts < attempt_limit:
        attempts += 1
        i = rng.randrange(edge_count)
        j = rng.randrange(edge_count - 1)
        if j >= i:
            j += 1
        a, b = edges[i]
        c, d = edges[j]
        if len({a, b, c, d}) != 4:
            continue
        if rng.getrandbits(1):
            first, second = tuple(sorted((a, d))), tuple(sorted((c, b)))
        else:
            first, second = tuple(sorted((a, c))), tuple(sorted((b, d)))
        if first == second or first in edge_set or second in edge_set:
            continue
        edge_set.remove((a, b))
        edge_set.remove((c, d))
        edge_set.add(first)
        edge_set.add(second)
        edges[i], edges[j] = first, second
        successful += 1
        if successful and successful % max(edge_count, 1) == 0:
            print(f"swap_progress {successful}/{target} ({time.monotonic()-started:.1f}s)", flush=True)
    result = torch.tensor(sorted(edge_set), dtype=torch.long)
    ratio = successful / max(edge_count, 1)
    if result.size(0) != pairs.size(0) or torch.unique(result, dim=0).size(0) != result.size(0):
        raise RuntimeError("rewiring changed edge count or introduced duplicate pairs")
    if (result[:, 0] == result[:, 1]).any():
        raise RuntimeError("rewiring introduced a self-loop")
    if not torch.equal(_degree(pairs, num_nodes), _degree(result, num_nodes)):
        raise RuntimeError("rewiring changed the per-node degree sequence")
    return result, successful, attempts, ratio


def _sha256(pairs: torch.Tensor) -> str:
    return hashlib.sha256(pairs.contiguous().numpy().tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed-seed degree-preserving physical graph rewires.")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/problem_deep_dive_v1/rewired_graphs"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": args.seed, "created_at_utc": datetime.now(timezone.utc).isoformat(), "graphs": {}}
    for dataset in args.datasets:
        print(f"REWIRE {dataset}", flush=True)
        pairs, num_nodes = _load_graph(dataset)
        rewired, swaps, attempts, ratio = _rewire(pairs, num_nodes, args.seed)
        if ratio < 5.0:
            record = {"dataset": dataset, "num_nodes": num_nodes, "num_undirected_edges": int(pairs.size(0)),
                      "successful_swaps": swaps, "attempts": attempts, "swap_ratio": ratio,
                      "status": "BELOW_5X_STOP_REWIRE_TRAINING", "graph_sha256": _sha256(rewired)}
            manifest["graphs"][dataset] = record
            (args.output_root / "rewired_graph_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            raise SystemExit(f"{dataset}: only {ratio:.2f}x successful swaps; stop rewired training (<5xE)")
        directed = torch.cat([rewired.T, rewired.flip(1).T], dim=1).contiguous()
        payload = {"edge_index": directed, "num_nodes": num_nodes, "dataset": dataset,
                   "rewire_seed": args.seed, "num_undirected_edges": int(rewired.size(0)),
                   "successful_swaps": swaps, "attempts": attempts, "swap_ratio": ratio,
                   "graph_sha256": _sha256(rewired), "degree_sequence_equal": True,
                   "self_loops": 0, "duplicate_undirected_pairs": 0}
        path = args.output_root / f"{dataset}.pt"
        torch.save(payload, path)
        manifest["graphs"][dataset] = {key: value for key, value in payload.items() if key != "edge_index"}
        manifest["graphs"][dataset]["path"] = str(path)
        manifest["graphs"][dataset]["status"] = "TARGET_10X_REACHED" if ratio >= 10.0 else "MINIMUM_5X_REACHED"
        print(json.dumps(manifest["graphs"][dataset]), flush=True)
    (args.output_root / "rewired_graph_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
