from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/hdd1/DataInHere/YHF/MoPF_IAMOC")
SOURCE_REF = "origin/V3"
DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
MODELS = ("mlp", "gcn", "sage", "mmgcn", "mgat", "dip", "dgf", "dmgc", "lgmrec")
PROTOCOL_FILES = (
    "configs/task/nc.yaml",
    "src/tasks/nc.py",
    "src/tasks/common.py",
    "src/tasks/inference.py",
    "src/utils/seeds.py",
    "src/utils/metrics.py",
    "src/utils/summary.py",
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _composite_sha256(files: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relpath, value in sorted(files.items()):
        digest.update(relpath.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    branch = _git(ROOT, "branch", "--show-current")
    if branch != "multi_order_bank":
        raise SystemExit(f"Expected branch multi_order_bank, got {branch}")
    status = _git(ROOT, "status", "--porcelain")
    if status:
        raise SystemExit("Refuse to write benchmark provenance with a dirty worktree:\n" + status)
    commit = _git(ROOT, "rev-parse", "HEAD")
    upstream = _git(ROOT, "rev-parse", "origin/multi_order_bank")
    if commit != upstream:
        raise SystemExit(f"HEAD {commit} differs from origin/multi_order_bank {upstream}")
    source_status = _git(SOURCE, "status", "--porcelain")
    if source_status:
        raise SystemExit("MoPF source worktree is not clean; source was not modified by this script.")
    source_commit = _git(SOURCE, "rev-parse", SOURCE_REF)

    protocol_files = {rel: _sha256(ROOT / rel) for rel in PROTOCOL_FILES}
    model_hashes = {model: _sha256(ROOT / "configs/model" / f"{model}.yaml") for model in MODELS}
    dataset_hashes = {dataset: _sha256(ROOT / "configs/dataset" / f"{dataset}.yaml") for dataset in DATASETS}
    model_configs = {
        model: OmegaConf.to_container(OmegaConf.load(ROOT / "configs/model" / f"{model}.yaml"), resolve=True)
        for model in MODELS
    }
    nc_cfg = OmegaConf.load(ROOT / "configs/task/nc.yaml")
    optimizer_resolution = {}
    for model, cfg in model_configs.items():
        optimizer_resolution[model] = {
            "optimizer": str(nc_cfg.optimizer).upper(),
            "lr": float(cfg.get("lr", nc_cfg.lr)),
            "weight_decay": float(cfg.get("weight_decay", nc_cfg.weight_decay)),
            "hidden_dim": cfg.get("hidden_dim"),
            "num_layers": cfg.get("num_layers"),
            "other_depth_fields": {key: cfg[key] for key in ("d_model", "q_dim", "mp_hops") if key in cfg},
        }

    split_hashes = {}
    config_dir = str((ROOT / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        for dataset in DATASETS:
            cfg = compose(config_name="config", overrides=[f"dataset={dataset}", "task=nc", "seed=42"])
            ds = OmegaConf.to_container(cfg.dataset, resolve=True)
            split_key = "nc_split_path" if ds["source"].lower() == "magb" else "node_split_path"
            split = Path(ds[split_key]).expanduser()
            if not split.is_absolute():
                split = (ROOT / split).resolve()
            if not split.is_file():
                raise SystemExit(f"Missing NC split for provenance: {split}")
            split_hashes[dataset] = {"path": str(split), "sha256": _sha256(split)}

    payload = {
        "schema": "unified_full_graph_nc_v1_provenance",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_repo": str(ROOT),
        "baseline_branch": branch,
        "commit": commit,
        "origin_branch_head": upstream,
        "source_baseline_commit": "8f7e825285ccff6e69f561b257cf160fcb3a1188",
        "source_mopf_ref": SOURCE_REF,
        "source_mopf_commit": source_commit,
        "protocol_version": "unified_full_graph_nc_v1",
        "nc_task_config_sha256": _sha256(ROOT / "configs/task/nc.yaml"),
        "nc_protocol_files_sha256": protocol_files,
        "nc_protocol_sha256": _composite_sha256(protocol_files),
        "dataset_config_sha256": dataset_hashes,
        "nc_split_sha256_seed42": split_hashes,
        "baseline_model_config_sha256": model_hashes,
        "resolved_optimizer_and_model_settings": optimizer_resolution,
        "seed": 42,
        "num_runs": 3,
        "run_seeds": [42, 43, 44],
        "aggregation": "mean ± population std (ddof=0)",
        "formal_nc_datasets": list(DATASETS),
        "formal_models": list(MODELS),
        "optimizer_override_semantics": "model YAML lr/weight_decay override task defaults in build_optimizer",
        "checkpoint_selection": "Validation Accuracy",
        "test_usage": "descriptive only; never used for selection or stopping",
        "formal_baseline_benchmark_started": False,
    }
    output = ROOT / "outputs/nc_benchmark_v1/provenance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output)
    print(json.dumps({"commit": commit, "source_mopf_commit": source_commit, "nc_protocol_sha256": payload["nc_protocol_sha256"], "datasets": list(DATASETS), "models": len(MODELS)}, indent=2))


if __name__ == "__main__":
    main()
