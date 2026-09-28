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

DATASETS = ("Movies", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
VARIANTS = (
    "p0_identity", "p0_masspres_scalar", "p0_single_dynamic_transform",
    "p0_operator_uniform", "p0_operator_routed",
)
PROTOCOL = "unified_full_graph_nc_v1"
BRANCH = "risa_v04_p0"
OUTPUT_ROOT = ROOT / "outputs/risa_v04_p0_v1"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _check_branch() -> None:
    branch = _git("branch", "--show-current")
    if branch != BRANCH:
        raise RuntimeError(f"RISA P0 run requires branch {BRANCH}, current branch is {branch}")
    subprocess.run(["git", "merge-base", "--is-ancestor", "s45_relcal_statepres", "HEAD"],
                   cwd=ROOT, check=True)


def _validate_checkpoint(path: Path, seed: int) -> dict[str, Any]:
    import torch

    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("task") != "nc" or payload.get("protocol_version") != PROTOCOL:
        raise RuntimeError(f"wrong task/protocol in {path}")
    if int(payload.get("seed", -1)) != seed or payload.get("selection") != "best_val_accuracy":
        raise RuntimeError(f"wrong seed or selection in {path}")
    metrics = payload.get("metrics", {})
    if not {"val_acc", "val_macro_f1"}.issubset(metrics):
        raise RuntimeError(f"validation metrics missing in {path}")
    if any(key.startswith("test_") for key in metrics):
        raise RuntimeError(f"test metrics present in {path}")
    return payload


def _run_job(dataset: str, variant: str, phase: str, gpu: str) -> dict[str, Any]:
    _check_branch()
    smoke = phase == "smoke"
    run_seeds = (42,) if smoke else SEEDS
    num_runs = len(run_seeds)
    epochs = 1 if smoke else 300
    output_dir = OUTPUT_ROOT / phase / dataset / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = OUTPUT_ROOT / "hydra_runs" / phase / dataset / variant / timestamp
    checkpoint = output_dir / ("best.pt" if smoke else "best_run1.pt")
    metrics_path = output_dir / "run_metrics.json"
    complete_path = output_dir / "complete.json"
    if smoke and complete_path.is_file() and checkpoint.is_file():
        previous = json.loads(complete_path.read_text(encoding="utf-8"))
        if (previous.get("phase") == "smoke" and previous.get("test_evaluation") is False
                and previous.get("lp_evaluation") is False):
            _validate_checkpoint(checkpoint, 42)
            return previous
    command = [
        sys.executable, "-m", "src.main", f"dataset={dataset}", "task=nc", "model=risa_v04",
        f"model.variant={variant}", "model.hidden_dim=256", "model.dropout=0.2",
        "model.max_order=3", "model.relation_dim=32", "model.bottleneck_dim=32",
        "model.num_operators=4", "model.edge_chunk_size=16384", "seed=42",
        f"num_runs={num_runs}", "device=cuda:0", f"task.epochs={epochs}",
        "task.lr=1e-3", "task.weight_decay=1e-4", "task.patience=30",
        "task.early_stop_min_epoch=1" if smoke else "task.early_stop_min_epoch=30",
        "task.early_stop_min_delta=1e-4", "task.grad_clip=1.0",
        "task.training_mode=full_graph", f"task.protocol_version={PROTOCOL}",
        "task.evaluate_test=false", f"task.save_ckpt_path={checkpoint}",
        f"task.run_metrics_path={metrics_path}", f"hydra.run.dir={run_dir}",
    ]
    log_path = output_dir / f"launch_{timestamp}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"PHASE={phase}\nDATASET={dataset}\nVARIANT={variant}\nGPU={gpu}\n")
        log.write("COMMAND=" + " ".join(command) + "\n")
        log.flush()
        subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    checkpoint_paths = ([output_dir / "best.pt"] if smoke else
                        [output_dir / f"best_run{i}.pt" for i in range(1, 4)])
    payloads = [_validate_checkpoint(path, seed)
                for path, seed in zip(checkpoint_paths, run_seeds, strict=True)]
    run_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if run_metrics.get("run_seeds") != list(run_seeds):
        raise RuntimeError("run seed aggregation does not match the requested seeds")
    if any(any(key.startswith("test_") for key in row.get("metrics", {}))
           for row in run_metrics.get("runs", [])):
        raise RuntimeError("test evaluation detected in run metrics")
    record = {
        "phase": phase, "dataset": dataset, "variant": variant,
        "training_branch": _git("branch", "--show-current"),
        "training_commit": _git("rev-parse", "HEAD"), "protocol_version": PROTOCOL,
        "task": "nc", "training_mode": "full_graph", "test_evaluation": False,
        "lp_evaluation": False, "base_seed": 42, "run_seeds": list(run_seeds),
        "num_runs": num_runs, "epochs_max": epochs,
        "best_epochs": [int(payload["epoch"]) for payload in payloads],
        "trainable_params": payloads[0]["run_metadata"]["model_parameters"] +
                            payloads[0]["run_metadata"]["classifier_parameters"],
        "model_trainable_params": payloads[0]["run_metadata"]["model_parameters"],
        "classifier_trainable_params": payloads[0]["run_metadata"]["classifier_parameters"],
        "checkpoint_paths": [str(path) for path in checkpoint_paths],
        "checkpoint_validation": "passed", "run_metrics_path": str(metrics_path),
        "launch_log": str(log_path), "command": command,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "complete.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RISA v0.4 P0 full-graph NC experiments")
    parser.add_argument("--phase", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=None)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=None)
    parser.add_argument("--gpu", default=os.environ.get("GPU_ID", "0"),
                        help="physical GPU index exposed as cuda:0 to the training process")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    _check_branch()
    if args.phase == "smoke":
        if args.datasets not in (None, ["Movies"]):
            parser.error("smoke phase is fixed to Movies seed=42")
        if args.variants not in (None, ["p0_operator_routed"]):
            parser.error("smoke phase is fixed to p0_operator_routed")
        jobs = [("Movies", "p0_operator_routed")]
    else:
        jobs = [(dataset, variant)
                for dataset in (args.datasets or DATASETS)
                for variant in (args.variants or VARIANTS)]
    if args.dry_run:
        print(json.dumps({"phase": args.phase, "jobs": jobs,
                          "seeds": [42] if args.phase == "smoke" else list(SEEDS),
                          "test_evaluation": False, "lp_evaluation": False}, indent=2))
        return
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for dataset, variant in jobs:
        results.append(_run_job(dataset, variant, args.phase, args.gpu))
        print(json.dumps({"dataset": dataset, "variant": variant,
                          "status": "complete", "checkpoint_paths": results[-1]["checkpoint_paths"]}),
              flush=True)
    summary = {"phase": args.phase, "requested_jobs": len(jobs), "completed": len(results),
               "failed": [], "test_evaluation": False, "lp_evaluation": False,
               "results": results}
    summary_path = OUTPUT_ROOT / f"{args.phase}_jobs.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.phase == "smoke":
        analyzer = [sys.executable, "scripts/analyze_risa_p0.py", "--dataset", "Movies",
                    "--checkpoint", results[0]["checkpoint_paths"][0], "--device", "cuda:0"]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        subprocess.run(analyzer, cwd=ROOT, env=env, check=True)
        print("Smoke complete; no formal sweep was launched.")
    else:
        print(f"Completed explicitly requested formal jobs={len(jobs)}; runs={len(jobs) * len(SEEDS)}")


if __name__ == "__main__":
    main()
