from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE_BRANCH = "problem_deep_dive"
SOURCE_COMMIT = "e535ad91911557790658ddc87bfaa516da531aad"
OLD_EVIDENCE_COMMITS = {
    "multi_order_bank": "31468c5a1439b3561487936bf5d83749b502f61e",
    "mechanism_discovery": "d31b095faa4f98538e60cfac293a831138dde3db",
    "problem_deep_dive": SOURCE_COMMIT,
}
DATASETS = ("Movies", "Grocery", "ele-fashion", "Reddit-S")
VARIANTS = (
    "p0_relation_only", "p0_residual", "p0_concat", "h1_dual_agg",
    "h1_dual_functional_static", "h1_dual_functional_global",
)
TRAINING_FILES = (
    "configs/model/relation_basis_pilot.yaml", "configs/task/nc.yaml",
    "configs/dataset/Movies.yaml", "configs/dataset/Grocery.yaml",
    "configs/dataset/ele-fashion.yaml", "configs/dataset/Reddit-S.yaml",
    "src/models/relation_basis_pilot.py", "scripts/run_s43_p0_h1_nc.py",
)
ANALYSIS_ONLY_FILES = {"scripts/analyze_s43_p0_h1.py", "scripts/write_s43_p0_h1_provenance.py"}
FILES = (
    "configs/model/relation_basis_pilot.yaml", "configs/task/nc.yaml",
    "configs/dataset/Movies.yaml", "configs/dataset/Grocery.yaml",
    "configs/dataset/ele-fashion.yaml", "configs/dataset/Reddit-S.yaml",
    "src/models/relation_basis_pilot.py", "scripts/run_s43_p0_h1_nc.py",
    "scripts/analyze_s43_p0_h1.py", "scripts/write_s43_p0_h1_provenance.py",
    "tests/test_s43_p0_h1.py",
)


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
    if branch != "s43_p0_h1":
        raise SystemExit(f"expected pilot branch s43_p0_h1, got {branch}")
    source_branch = git("branch", "--contains", SOURCE_COMMIT, "--format=%(refname:short)")
    if not source_branch:
        raise SystemExit("pilot history does not contain the required source commit")
    subprocess.run(["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, "HEAD"], cwd=ROOT, check=True)
    training_commits = set()
    completed_jobs = 0
    for dataset in DATASETS:
        for variant in VARIANTS:
            path = ROOT / "outputs/s43_p0_h1_v1/formal" / dataset / variant / "complete.json"
            if not path.is_file():
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("test_evaluation") is not False:
                raise SystemExit(f"test evaluation was not disabled in {path}")
            training_commits.add(row.get("training_commit"))
            completed_jobs += 1
    training_commits.discard(None)
    if completed_jobs != 24 or len(training_commits) != 1:
        raise SystemExit(f"expected 24 jobs with one fixed training commit, got jobs={completed_jobs}, commits={training_commits}")
    training_commit = next(iter(training_commits))
    analysis_commit = git("rev-parse", "HEAD")
    subprocess.run(["git", "merge-base", "--is-ancestor", training_commit, "HEAD"], cwd=ROOT, check=True)
    changed_paths = [name for name in git("diff", "--name-only", f"{training_commit}..HEAD").splitlines() if name]
    unexpected = sorted(set(changed_paths) - ANALYSIS_ONLY_FILES)
    if unexpected:
        raise SystemExit(f"post-training changes exceed analysis-only scope: {unexpected}")
    for name in TRAINING_FILES:
        frozen_blob = subprocess.check_output(["git", "show", f"{training_commit}:{name}"], cwd=ROOT)
        if hashlib.sha256(frozen_blob).hexdigest() != sha256(ROOT / name):
            raise SystemExit(f"frozen training input changed after formal training: {name}")
    code_hashes = {name: sha256(ROOT / name) for name in FILES}
    configs = {
        name: sha256(ROOT / name)
        for name in FILES if name.startswith("configs/")
    }
    payload = {
        "study": "S4.3 Hypothesis Validation Pilots, Phase 1: P0 + H1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_branch": SOURCE_BRANCH, "source_commit": SOURCE_COMMIT,
        "pilot_branch": branch, "training_commit": training_commit,
        "analysis_commit": analysis_commit,
        "analysis_commit_type": "same_as_training" if analysis_commit == training_commit else "analysis_only",
        "analysis_only_files_since_training": changed_paths,
        "old_evidence_commits": OLD_EVIDENCE_COMMITS,
        "protocol_version": "unified_full_graph_nc_v1",
        "formal_datasets": list(DATASETS), "formal_variants": list(VARIANTS),
        "variant_definitions": {
            "p0_relation_only": "Z=LN(T(P_rel H))",
            "p0_residual": "Z=LN(H+T(P_rel H)); reused as H1 single aggregative reference",
            "p0_concat": "Z=LN(Linear([H || T(P_rel H)])); CAPACITY_DIFFERENT_SECONDARY_CONTROL",
            "h1_dual_agg": "Z=LN(H+0.5*T_A1(P_rel H)+0.5*T_A2(P_rel H))",
            "h1_dual_functional_static": "Z=LN(H+0.5*T_A(P_rel H)+0.5*T_D(sH-P_rel H))",
            "h1_dual_functional_global": "Z=LN(H+alpha_A*T_A(P_rel H)+alpha_D*T_D(sH-P_rel H)); modality-global coefficients only",
        },
        "run_seeds": [42, 43, 44], "seed_command": 42, "num_runs": 3,
        "training_protocol": {
            "epochs": 300, "lr": 1e-3, "weight_decay": 1e-4,
            "patience": 30, "early_stop_min_epoch": 30,
            "early_stop_min_delta": 1e-4, "grad_clip": 1.0,
            "selection": "best_val_accuracy", "test_disabled": True,
        },
        "formal_jobs_completed": completed_jobs, "formal_runs_completed": completed_jobs * 3,
        "dataset_config_sha256": configs,
        "task_config_sha256": sha256(ROOT / "configs/task/nc.yaml"),
        "model_config_sha256": sha256(ROOT / "configs/model/relation_basis_pilot.yaml"),
        "code_file_sha256": code_hashes,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "worktree_code_changes": git("status", "--porcelain"),
    }
    output = ROOT / "outputs/s43_p0_h1_v1/provenance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"path": str(output), "training_commit": training_commit,
                      "jobs": completed_jobs, "sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
