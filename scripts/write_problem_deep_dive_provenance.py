from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "d31b095faa4f98538e60cfac293a831138dde3db"
FILES = [
    "configs/task/nc.yaml",
    "configs/model/operator_control_gcn.yaml",
    "configs/model/operator_control_sage.yaml",
    "src/main.py",
    "src/models/operator_control.py",
    "src/models/operator_control_gcn.py",
    "src/models/operator_control_sage.py",
    "src/analysis/problem_deep_dive.py",
    "src/tasks/nc.py",
    "scripts/build_degree_preserving_rewires.py",
    "scripts/run_problem_deep_dive_nc.py",
    "scripts/analyze_problem_deep_dive.py",
    "scripts/problem_deep_dive_analysis.py",
    "tests/test_problem_deep_dive.py",
    "docs/problem_deep_dive_analysis_spec.md",
    "scripts/write_problem_deep_dive_provenance.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> None:
    branch = git("branch", "--show-current")
    commit = git("rev-parse", "HEAD")
    if branch != "problem_deep_dive":
        raise SystemExit(f"expected branch problem_deep_dive, got {branch}")
    if git("status", "--porcelain"):
        raise SystemExit("provenance requires a clean, committed worktree")
    subprocess.run(["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, "HEAD"], cwd=ROOT, check=True)
    if commit == SOURCE_COMMIT:
        raise SystemExit("D3 commit must be a descendant of its immutable source commit")
    manifest_path = ROOT / "outputs/problem_deep_dive_v1/rewired_graphs/rewired_graph_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing generated rewiring manifest: {manifest_path}")
    rewire_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(rewire_manifest.get("graphs", {})) != {"Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"}:
        raise SystemExit("rewire manifest must contain all five registered NC datasets")
    rewire_records = {}
    for dataset, row in rewire_manifest["graphs"].items():
        path = ROOT / row["path"]
        if row.get("status") != "TARGET_10X_REACHED" or float(row.get("swap_ratio", 0)) < 10:
            raise SystemExit(f"{dataset}: rewire did not reach the 10E target")
        if not path.is_file():
            raise SystemExit(f"missing rewired graph: {path}")
        if not row.get("degree_sequence_equal") or row.get("self_loops") != 0 or row.get("duplicate_undirected_pairs") != 0:
            raise SystemExit(f"{dataset}: invalid rewired graph assertions")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("graph_sha256") != row.get("graph_sha256"):
            raise SystemExit(f"{dataset}: stored graph hash differs from manifest")
        rewire_records[dataset] = {
            "path": str(path.relative_to(ROOT)),
            "file_sha256": sha256(path),
            "canonical_graph_sha256": row["graph_sha256"],
            "num_nodes": row["num_nodes"],
            "num_undirected_edges": row["num_undirected_edges"],
            "successful_swaps": row["successful_swaps"],
            "attempts": row["attempts"],
            "swap_ratio": row["swap_ratio"],
        }
    file_hashes = {name: sha256(ROOT / name) for name in FILES}
    payload = {
        "study": "D3 Principle Generalization & Problem Causal Deep-Dive",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "branch": branch,
        "code_commit": commit,
        "source_baseline_commit": SOURCE_COMMIT,
        "worktree_clean": True,
        "protocol_version": "unified_full_graph_nc_v1",
        "training_plan": {
            "operator_transfer_jobs": 20,
            "operator_transfer_runs": 60,
            "rewired_uniform_jobs": 5,
            "rewired_uniform_runs": 15,
            "seed_command": 42,
            "num_runs": 3,
            "internal_run_seeds": [42, 43, 44],
            "preflight_jobs": 5,
            "preflight_runs": 5,
            "formal_training_started": False,
        },
        "analysis_split_policy": {
            "selector_fit": "train only",
            "threshold_fit": "edges with both endpoints in train",
            "gate_and_ranking": "train/validation only",
            "test": "descriptive metrics only for newly retrained formal controls",
        },
        "nc_config_sha256": sha256(ROOT / "configs/task/nc.yaml"),
        "source_rewire_manifest_sha256": sha256(manifest_path),
        "rewired_graphs": rewire_records,
        "files_sha256": file_hashes,
        "python_torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    out = ROOT / "outputs/problem_deep_dive_v1/provenance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out)
    print(json.dumps({"branch": branch, "commit": commit, "rewired_graphs": len(rewire_records),
                      "sha256": sha256(out), "formal_training_started": False}, indent=2))


if __name__ == "__main__":
    main()
