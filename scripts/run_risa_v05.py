from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCREEN_DATASETS = ("Movies", "ele-fashion", "Reddit-S")
CONFIRM_DATASETS = ("Grocery",)
SEEDS = (42, 43, 44)
VARIANTS = ("v05_full", "v05_no_crst", "v05_no_relation_context", "v05_no_imci")
PROTOCOL = "unified_full_graph_nc_v1"
BRANCH = "risa_v05"
OUTPUT_ROOT = ROOT / "outputs/risa_v05_v1"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _check_branch() -> None:
    branch = _git("branch", "--show-current")
    if branch != BRANCH:
        raise RuntimeError(f"RISA v0.5 runner requires branch {BRANCH}, current branch is {branch}")


def expected_checkpoint_paths(output_dir: Path, num_runs: int) -> list[Path]:
    if num_runs < 1:
        raise ValueError("num_runs must be positive")
    if num_runs == 1:
        return [output_dir / "best.pt"]
    return [output_dir / f"best_run{run}.pt" for run in range(1, num_runs + 1)]


def build_command(dataset: str, variant: str, phase: str, gpu: str,
                  output_dir: Path, run_dir: Path) -> list[str]:
    if dataset not in SCREEN_DATASETS:
        raise ValueError(f"dataset must be one of SCREEN_DATASETS={SCREEN_DATASETS}")
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    if phase not in {"smoke", "formal"}:
        raise ValueError(f"unsupported phase {phase!r}")
    smoke = phase == "smoke"
    num_runs = 1 if smoke else len(SEEDS)
    epochs = 1 if smoke else 300
    checkpoint = output_dir / "best.pt"
    return [
        sys.executable, "-m", "src.main", f"dataset={dataset}", "task=nc", "model=risa_v05",
        f"model.variant={variant}", "model.hidden_dim=256", "model.dropout=0.2",
        "model.max_order=3", "model.relation_dim=32", "model.edge_chunk_size=16384",
        "model.rotation_group_size=2", "model.max_rotation_angle=1.57079632679",
        "model.iamr_num_heads=4", "model.iamr_ff_mult=2", "model.node_chunk_size=32768",
        "seed=42", f"num_runs={num_runs}", f"device=cuda:0",
        f"task.epochs={epochs}", "task.lr=1e-3", "task.weight_decay=1e-4",
        "task.patience=30", "task.early_stop_min_epoch=1" if smoke else "task.early_stop_min_epoch=30",
        "task.early_stop_min_delta=1e-4", "task.grad_clip=1.0",
        "task.training_mode=full_graph", f"task.protocol_version={PROTOCOL}",
        "task.evaluate_test=false", f"task.save_ckpt_path={checkpoint}",
        f"hydra.run.dir={run_dir}",
    ]


def _validate_checkpoint(path: Path, expected_seed: int) -> dict[str, Any]:
    import torch

    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("task") != "nc" or payload.get("protocol_version") != PROTOCOL:
        raise RuntimeError(f"wrong task/protocol in {path}")
    if int(payload.get("seed", -1)) != expected_seed:
        raise RuntimeError(f"wrong seed in {path}")
    if payload.get("selection") != "best_val_accuracy":
        raise RuntimeError(f"checkpoint was not selected by validation accuracy: {path}")
    if any(key.startswith("test_") for key in payload.get("metrics", {})):
        raise RuntimeError(f"test metrics found in smoke/formal checkpoint {path}")
    return payload


def _run_job(dataset: str, variant: str, phase: str, gpu: str) -> dict[str, Any]:
    _check_branch()
    output_dir = OUTPUT_ROOT / phase / dataset / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = OUTPUT_ROOT / "hydra_runs" / phase / dataset / variant / timestamp
    command = build_command(dataset, variant, phase, gpu, output_dir, run_dir)
    log_path = output_dir / f"launch_{timestamp}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"PHASE={phase}\nDATASET={dataset}\nVARIANT={variant}\nGPU={gpu}\n")
        log.write("COMMAND=" + " ".join(command) + "\n")
        log.flush()
        subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)

    seeds = (42,) if phase == "smoke" else SEEDS
    checkpoints = expected_checkpoint_paths(output_dir, len(seeds))
    payloads = [_validate_checkpoint(path, seed)
                for path, seed in zip(checkpoints, seeds, strict=True)]
    record = {
        "phase": phase, "dataset": dataset, "variant": variant,
        "training_branch": _git("branch", "--show-current"),
        "training_commit": _git("rev-parse", "HEAD"),
        "protocol_version": PROTOCOL, "task": "nc", "training_mode": "full_graph",
        "test_evaluation": False, "lp_evaluation": False,
        "base_seed": 42, "run_seeds": list(seeds), "num_runs": len(seeds),
        "epochs_max": 1 if phase == "smoke" else 300,
        "best_epochs": [int(item["epoch"]) for item in payloads],
        "model_trainable_params": payloads[0]["run_metadata"]["model_parameters"],
        "classifier_trainable_params": payloads[0]["run_metadata"]["classifier_parameters"],
        "checkpoint_paths": [str(path) for path in checkpoints],
        "checkpoint_metrics": [item.get("metrics", {}) for item in payloads],
        "checkpoint_validation": "passed", "launch_log": str(log_path),
        "command": command, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "complete.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    if phase == "smoke":
        analyzer = [sys.executable, "scripts/analyze_risa_v05.py", "--dataset", dataset,
                    "--checkpoint", str(checkpoints[0]), "--device", "cuda:0"]
        subprocess.run(analyzer, cwd=ROOT, env=env, check=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Model-v0.5 NC smoke or explicitly requested formal jobs")
    parser.add_argument("--phase", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--datasets", nargs="+", choices=SCREEN_DATASETS, default=None)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=None)
    parser.add_argument("--gpu", default=os.environ.get("GPU_ID", "0"),
                        help="physical GPU index exposed as cuda:0 to the training process")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    _check_branch()
    if args.phase == "smoke":
        if args.datasets not in (None, ["Movies"]):
            parser.error("smoke phase is fixed to Movies")
        if args.variants not in (None, ["v05_full"]):
            parser.error("smoke phase is fixed to v05_full")
        jobs = [("Movies", "v05_full")]
    else:
        jobs = [(dataset, variant)
                for dataset in (args.datasets or SCREEN_DATASETS)
                for variant in (args.variants or ("v05_full",))]
    if args.dry_run:
        print(json.dumps({
            "phase": args.phase, "jobs": jobs,
            "seeds": [42] if args.phase == "smoke" else list(SEEDS),
            "screen_datasets": list(SCREEN_DATASETS),
            "confirmation_datasets": list(CONFIRM_DATASETS),
            "test_evaluation": False, "lp_evaluation": False,
        }, indent=2))
        return
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for dataset, variant in jobs:
        results.append(_run_job(dataset, variant, args.phase, args.gpu))
        print(json.dumps({
            "dataset": dataset, "variant": variant, "status": "complete",
            "checkpoint_paths": results[-1]["checkpoint_paths"],
        }), flush=True)
    print(json.dumps({"phase": args.phase, "completed": len(results),
                      "test_evaluation": False, "lp_evaluation": False}, indent=2))
    if args.phase == "smoke":
        print("Movies seed 42 smoke finished; no formal 300-epoch sweep was launched.")


if __name__ == "__main__":
    main()
