from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

SOURCE_BRANCH = "multi_order_bank"
SOURCE_COMMIT = "31468c5a1439b3561487936bf5d83749b502f61e"
DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
CODE_FILES = (
    "src/models/multi_order_bank.py", "src/tasks/nc.py", "src/tasks/common.py",
    "configs/task/nc.yaml", "configs/model/multi_order_bank.yaml",
    "scripts/run_mechanism_discovery_nc.py", "scripts/analyze_mechanism_discovery.py",
    "scripts/write_mechanism_discovery_provenance.py", "src/analysis/mechanism_discovery.py",
    "tests/test_mechanism_discovery_model.py", "tests/test_mechanism_discovery_utils.py",
)


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def combined_hash(values: dict[str, str]) -> str:
    h = hashlib.sha256()
    for name, value in sorted(values.items()):
        h.update(name.encode()); h.update(b"\0"); h.update(value.encode()); h.update(b"\n")
    return h.hexdigest()


def main() -> None:
    branch = git("branch", "--show-current")
    if branch != "mechanism_discovery":
        raise SystemExit(f"Expected mechanism_discovery branch, got {branch}")
    status = git("status", "--porcelain")
    if status:
        raise SystemExit("Refuse provenance on dirty worktree:\n" + status)
    commit = git("rev-parse", "HEAD")
    source = git("rev-parse", f"{SOURCE_BRANCH}^{{commit}}")
    if source != SOURCE_COMMIT:
        raise SystemExit(f"Source branch moved: expected {SOURCE_COMMIT}, got {source}")
    upstream = git("rev-parse", "origin/mechanism_discovery^{commit}")
    if upstream != commit:
        raise SystemExit(f"Discovery HEAD {commit} is not pushed (origin has {upstream})")
    prior = json.loads((ROOT / "outputs/nc_benchmark_v1/provenance.json").read_text())
    code_hashes = {name: sha256(ROOT / name) for name in CODE_FILES}
    dataset_hashes = {name: sha256(ROOT / "configs/dataset" / f"{name}.yaml") for name in DATASETS}
    split_hashes = {}
    with initialize_config_dir(version_base=None, config_dir=str((ROOT / "configs").resolve())):
        for dataset in DATASETS:
            split_hashes[dataset] = {}
            for seed in SEEDS:
                cfg = compose(config_name="config", overrides=[f"dataset={dataset}", "task=nc", f"seed={seed}"])
                ds = OmegaConf.to_container(cfg.dataset, resolve=True)
                split_key = "nc_split_path" if ds["source"].lower() == "magb" else "node_split_path"
                path = Path(ds[split_key]).expanduser()
                if not path.is_absolute(): path = (ROOT / path).resolve()
                if not path.is_file(): raise SystemExit(f"Missing split file: {path}")
                split_hashes[dataset][str(seed)] = {"path": str(path), "sha256": sha256(path)}
    preflight_files = list((ROOT / "outputs/mechanism_discovery_v1/preflight").glob("*/*/complete.json"))
    if len(preflight_files) != 15:
        raise SystemExit(f"Expected 15 successful ele-fashion one-epoch preflight jobs, found {len(preflight_files)}")
    preflight_summary = []
    for complete in sorted(preflight_files):
        row = json.loads(complete.read_text(encoding="utf-8"))
        if row.get("phase") != "preflight" or row.get("dataset") != "ele-fashion" or row.get("num_runs") != 1 or row.get("epochs") != 1:
            raise SystemExit(f"Unexpected preflight metadata in {complete}")
        checkpoint = Path(row["checkpoints"][0])
        if not checkpoint.is_file(): raise SystemExit(f"Missing preflight checkpoint {checkpoint}")
        preflight_summary.append({"variant": row["variant"], "readout": row["readout"],
                                  "modality_mode": row["modality_mode"],
                                  "checkpoint_sha256": sha256(checkpoint),
                                  "result_path": row["results"]})
    step_a_root = ROOT / "results/mechanism_discovery_v1/experiment1"
    step_a_hashes = {path.name: sha256(path) for path in sorted(step_a_root.glob("*")) if path.is_file()}
    model_config = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/model/multi_order_bank.yaml"), resolve=True)
    task_config = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/task/nc.yaml"), resolve=True)
    from src.analysis.mechanism_discovery import PROBE_CONFIG
    analysis_config = {
        "order_source_readouts": ["self_only", "self25_terminal75", "self50_terminal50", "self75_terminal25", "propagated_uniform"],
        "unimodal_variants": ["text_self", "text_uniform", "visual_self", "visual_uniform"],
        "fixed_model": {"hidden_dim": 256, "max_order": 3, "dropout": 0.2, "fusion_mode": "plain_mlp"},
        "frozen_nc_protocol": task_config,
        "edge_role_thresholds": {"high": 0.75, "low": 0.25, "sensitivity": 0.5},
        "edge_nonedge_reference": {"max_source_nodes": 100000, "nonedges_per_source": 32, "seed_rule": "seed*101+7"},
        "frozen_edge_intervention_random_repeats": 10,
        "stability_edge_dropout_rates": [0.05, 0.10, 0.20],
        "stability_feature_noise_scales": [0.05, 0.10, 0.20],
        "stability_replicates": 10,
        "alignment_retrieval_max_validation_nodes": 5000,
        "interaction_cube_thresholds": "train-node medians",
        "training_output_root": "outputs/mechanism_discovery_v1/",
        "results_root": "results/mechanism_discovery_v1/",
        "model_config": model_config,
    }
    payload = {
        "schema": "mechanism_discovery_v1_provenance",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(ROOT), "branch": branch, "discovery_commit": commit,
        "source_branch": SOURCE_BRANCH, "source_commit": source,
        "existing_factorial_training_commit": prior["factorial_training_commit"],
        "existing_baseline_training_commit": prior["baseline_training_commit"],
        "existing_result_analysis_commit": prior["results_commit"],
        "existing_result_inputs": {
            "factorial_root": "outputs/mob_factorial_nc_v1/",
            "baseline_root": "outputs/nc_benchmark_v1/",
            "summary_json_sha256": sha256(ROOT / "results/nc_benchmark_v1/summary.json"),
            "factorial_table_sha256": sha256(ROOT / "results/nc_benchmark_v1/mob_factorial_table.csv"),
        },
        "formal_datasets": list(DATASETS), "task": "node classification only; no LP",
        "base_seed": 42, "num_runs": 3, "run_seeds": list(SEEDS),
        "aggregation": "mean ± population std (ddof=0)",
        "protocol_version": "unified_full_graph_nc_v1",
        "dataset_config_sha256": dataset_hashes,
        "split_file_sha256_by_dataset_and_seed": split_hashes,
        "code_sha256": code_hashes, "code_sha256_composite": combined_hash(code_hashes),
        "probe_config": PROBE_CONFIG.__dict__, "probe_config_sha256": PROBE_CONFIG.sha256(),
        "analysis_config": analysis_config,
        "analysis_config_sha256": hashlib.sha256(json.dumps(analysis_config, sort_keys=True, default=str).encode()).hexdigest(),
        "data_use_boundary": "Discovery uses only train and validation labels. Test labels are not used for problem selection, feature construction, thresholds, probes, correlations, or interventions; frozen NC test outputs are descriptive only.",
        "formal_training_started": False,
        "preflight_required_before_formal_training": True,
        "preflight_validation": {"status": "passed", "jobs": len(preflight_summary), "dataset": "ele-fashion", "epochs": 1, "num_runs": 1, "jobs_detail": preflight_summary},
        "step_a_artifact_sha256": step_a_hashes,
        "formal_run_plan": {"order_source_jobs": 25, "order_source_runs": 75,
                            "unimodal_jobs": 20, "unimodal_runs": 60,
                            "total_new_runs": 135, "gpu_scheduler": "one active worker per physical GPU; shared dynamic queue"},
        "existing_outputs_immutable": ["results/nc_benchmark_v1/", "outputs/mob_factorial_nc_v1/", "outputs/nc_benchmark_v1/"],
    }
    output = ROOT / "outputs/mechanism_discovery_v1/provenance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output)
    print(json.dumps({"branch": branch, "commit": commit, "source_commit": source,
                      "probe_config_sha256": payload["probe_config_sha256"],
                      "datasets": list(DATASETS), "seeds": list(SEEDS)}, indent=2))


if __name__ == "__main__":
    main()
