from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from scripts.gpu_scheduler import GPUJobOutcome, run_gpu_jobs

DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
ORDER_READOUTS = (
    "self_only", "self25_terminal75", "self50_terminal50",
    "self75_terminal25", "propagated_uniform",
)
UNIMODAL_VARIANTS = (
    ("text_self", "text", "self_only"),
    ("text_uniform", "text", "uniform"),
    ("visual_self", "visual", "self_only"),
    ("visual_uniform", "visual", "uniform"),
)
SEEDS = (42, 43, 44)
PROTOCOL = "unified_full_graph_nc_v1"
OUTPUT_ROOT = Path("outputs/mechanism_discovery_v1")


def _checkpoint_errors(paths: list[Path], expected_seeds: tuple[int, ...], require_test_metrics: bool = True) -> list[str]:
    errors: list[str] = []
    for path, expected_seed in zip(paths, expected_seeds, strict=True):
        if not path.is_file():
            errors.append(f"missing checkpoint {path}")
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            errors.append(f"cannot load {path}: {exc}")
            continue
        if payload.get("task") != "nc" or payload.get("protocol_version") != PROTOCOL:
            errors.append(f"{path}: incorrect frozen NC protocol")
        if int(payload.get("seed", -1)) != expected_seed:
            errors.append(f"{path}: expected seed={expected_seed}, got {payload.get('seed')}")
        if payload.get("selection") != "best_val_accuracy" or payload.get("epoch") is None:
            errors.append(f"{path}: missing validation-selected checkpoint metadata")
        metrics = payload.get("metrics", {})
        required = {"val_acc", "val_macro_f1"}
        if require_test_metrics:
            required |= {"test_acc", "test_macro_f1"}
        if not required.issubset(metrics):
            errors.append(f"{path}: missing metrics {sorted(required - metrics.keys())}")
    return errors


def _jobs(phase: str, datasets: tuple[str, ...]):
    if phase == "order":
        for dataset in datasets:
            for readout in ORDER_READOUTS:
                yield (phase, dataset, readout, "both", f"{readout}_plain")
    elif phase == "unimodal":
        for dataset in datasets:
            for variant, mode, readout in UNIMODAL_VARIANTS:
                yield (phase, dataset, readout, mode, variant)
    elif phase == "preflight":
        for readout in ORDER_READOUTS:
            for mode in ("both", "text", "visual"):
                yield (phase, "ele-fashion", readout, mode, f"{readout}_{mode}")
    else:
        raise ValueError(phase)


def _run_job(job: tuple[str, str, str, str, str], gpu: str, root: Path) -> dict[str, Any]:
    phase, dataset, readout, modality_mode, variant = job
    group = {"preflight": "preflight", "order": "order_source", "unimodal": "unimodal"}[phase]
    output_dir = root / group / dataset / variant
    complete_path = output_dir / "complete.json"
    if phase == "preflight":
        checkpoint_paths = [output_dir / "best.pt"]
        expected_seeds = (42,)
        epochs, num_runs = 1, 1
    else:
        checkpoint_paths = [output_dir / f"best_run{i}.pt" for i in (1, 2, 3)]
        expected_seeds = SEEDS
        epochs, num_runs = 300, 3
        if complete_path.is_file():
            try:
                record = json.loads(complete_path.read_text(encoding="utf-8"))
                if record.get("phase") == phase and record.get("dataset") == dataset and record.get("variant") == variant and not _checkpoint_errors(checkpoint_paths, expected_seeds):
                    return {"phase": phase, "dataset": dataset, "variant": variant,
                            "status": "already_complete", "gpu": gpu}
            except (OSError, json.JSONDecodeError):
                pass

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / f"run_{stamp}"
    checkpoint_base = output_dir / ("best.pt" if num_runs == 1 else "best.pt")
    command = [
        sys.executable, "-m", "src.main", f"dataset={dataset}", "task=nc",
        "model=multi_order_bank", f"model.readout={readout}",
        "model.fusion_mode=plain_mlp", f"model.modality_mode={modality_mode}",
        "model.hidden_dim=256", "model.max_order=3", "model.dropout=0.2",
        "seed=42", f"num_runs={num_runs}", "device=cuda:0",
        f"task.epochs={epochs}", f"task.save_ckpt_path={checkpoint_base}",
        f"task.run_metrics_path={output_dir / 'run_metrics.json'}",
        f"task.evaluate_test={'false' if phase == 'preflight' else 'true'}",
        f"hydra.run.dir={run_dir}",
    ]
    if phase == "preflight":
        command.extend(["task.patience=1", "task.early_stop_min_epoch=1"])
    log_path = output_dir / f"launch_{stamp}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(command) + "\n")
        log.write(f"PHYSICAL_GPU: {gpu}\n")
        log.flush()
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - start
    if process.returncode:
        raise RuntimeError(f"{phase} {dataset}/{variant} failed on GPU {gpu}; log={log_path}")
    if phase == "preflight":
        expected_result = run_dir / "results.json"
        if not expected_result.is_file():
            raise RuntimeError(f"preflight returned without results: {log_path}")
    else:
        expected_result = output_dir / "run_metrics.json"
    errors = _checkpoint_errors(checkpoint_paths, expected_seeds, require_test_metrics=phase != "preflight")
    if errors:
        raise RuntimeError("checkpoint validation failed: " + "; ".join(errors))
    complete_path.write_text(json.dumps({
        "phase": phase, "dataset": dataset, "variant": variant,
        "model": "multi_order_bank", "readout": readout,
        "modality_mode": modality_mode, "fusion_mode": "plain_mlp",
        "protocol_version": PROTOCOL, "seed": 42, "num_runs": num_runs,
        "run_seeds": list(expected_seeds), "epochs": epochs,
        "gpu": gpu, "seconds": elapsed, "command": command,
        "results": str(expected_result),
        "checkpoints": [str(path) for path in checkpoint_paths],
        "checkpoint_validation": "passed",
    }, indent=2), encoding="utf-8")
    return {"phase": phase, "dataset": dataset, "variant": variant,
            "status": "complete", "gpu": gpu, "seconds": elapsed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen NC controls for mechanism discovery.")
    parser.add_argument("phase", choices=("preflight", "order", "unimodal"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--gpus", default=os.environ.get("GPU_IDS", "0,1"))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpus = tuple(part.strip() for part in args.gpus.split(",") if part.strip())
    if not gpus:
        parser.error("at least one GPU ID is required")
    jobs = list(_jobs(args.phase, tuple(args.datasets)))
    if args.dry_run:
        for job in jobs:
            print(" ".join(job))
        print(f"jobs={len(jobs)} seeds_per_job={1 if args.phase == 'preflight' else 3}")
        return
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)

    def report(outcome: GPUJobOutcome[Any, Any]) -> None:
        if outcome.error is None:
            print(json.dumps(outcome.result), flush=True)
        else:
            print(f"FAILED {outcome.job} on GPU {outcome.gpu}: {outcome.error}", flush=True)

    summary = run_gpu_jobs(jobs, gpus, lambda job, gpu: _run_job(job, gpu, root), on_finish=report)
    if summary.failures:
        details = "\n".join(f"{item.job} on GPU {item.gpu}: {item.error}" for item in summary.failures)
        if summary.not_started:
            details += "\nNot started after failure: " + ", ".join(str(job) for _, job in summary.not_started)
        raise SystemExit("Mechanism discovery jobs failed:\n" + details)
    print(f"Completed {len(jobs)} {args.phase} jobs; formal runs={0 if args.phase == 'preflight' else len(jobs) * 3}.", flush=True)


if __name__ == "__main__":
    main()
