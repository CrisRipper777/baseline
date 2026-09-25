from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

try:
    from scripts.gpu_scheduler import GPUJobOutcome, run_gpu_jobs
except ModuleNotFoundError:
    from gpu_scheduler import GPUJobOutcome, run_gpu_jobs


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
READOUTS = ("terminal", "uniform", "gpr")
FUSIONS = ("plain_mlp", "residual")
BASE_SEED = 42
NUM_RUNS = 3
EXPECTED_SEEDS = (42, 43, 44)
EXPECTED_METRICS = ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
OUTPUT_ROOT = Path("outputs/mob_factorial_nc_v1")


def _jobs(datasets: tuple[str, ...]):
    for dataset in datasets:
        for fusion in FUSIONS:
            for readout in READOUTS:
                variant = f"mob_{readout}_{'plain' if fusion == 'plain_mlp' else 'residual'}"
                yield dataset, readout, fusion, variant


def _checkpoint_errors(paths: list[Path], expected_seeds: tuple[int, ...]) -> list[str]:
    errors: list[str] = []
    for path, expected_seed in zip(paths, expected_seeds):
        if not path.is_file():
            errors.append(f"missing checkpoint {path}")
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            errors.append(f"cannot load {path}: {exc}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"{path}: checkpoint payload is not a mapping")
            continue
        if payload.get("task") != "nc":
            errors.append(f"{path}: task must be nc")
        if payload.get("protocol_version") != "unified_full_graph_nc_v1":
            errors.append(f"{path}: wrong protocol_version")
        if payload.get("seed") != expected_seed:
            errors.append(f"{path}: expected seed {expected_seed}, got {payload.get('seed')}")
        if payload.get("selection") != "best_val_accuracy":
            errors.append(f"{path}: wrong checkpoint selection")
        if payload.get("epoch") is None:
            errors.append(f"{path}: epoch is missing")
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            errors.append(f"{path}: metrics is missing or invalid")
        else:
            missing = sorted(set(EXPECTED_METRICS) - metrics.keys())
            if missing:
                errors.append(f"{path}: missing metrics {missing}")
    return errors


def _valid_resume(complete_path: Path, checkpoints: list[Path], dataset: str, variant: str) -> bool:
    try:
        previous = json.loads(complete_path.read_text(encoding="utf-8"))
        if previous.get("dataset") != dataset or previous.get("variant") != variant:
            return False
        if previous.get("model") != "multi_order_bank":
            return False
        if previous.get("protocol_version") != "unified_full_graph_nc_v1":
            return False
        if previous.get("run_seeds") != list(EXPECTED_SEEDS) or previous.get("num_runs") != NUM_RUNS:
            return False
        return Path(previous.get("results", "")).is_file() and not _checkpoint_errors(checkpoints, EXPECTED_SEEDS)
    except (OSError, json.JSONDecodeError, TypeError):
        return False


def _run_job(dataset: str, readout: str, fusion: str, variant: str, gpu: str, root: Path) -> dict:
    output_dir = root / dataset / variant
    complete_path = output_dir / "complete.json"
    checkpoint_base = output_dir / "best.pt"
    expected_checkpoints = [output_dir / f"best_run{run_id}.pt" for run_id in (1, 2, 3)]
    if complete_path.is_file() and _valid_resume(complete_path, expected_checkpoints, dataset, variant):
        return {"dataset": dataset, "variant": variant, "status": "already_complete", "gpu": gpu}

    output_dir.mkdir(parents=True, exist_ok=True)
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = output_dir / f"launch_{attempt}.log"
    run_dir = output_dir / f"run_{attempt}"
    command = [
        sys.executable, "-m", "src.main",
        f"dataset={dataset}", "task=nc", "model=multi_order_bank",
        f"model.readout={readout}", f"model.fusion_mode={fusion}",
        f"seed={BASE_SEED}", f"num_runs={NUM_RUNS}", "device=cuda:0",
        f"task.save_ckpt_path={checkpoint_base}",
        f"hydra.run.dir={run_dir}",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(command) + "\n")
        log.write(f"PHYSICAL_GPU: {gpu}\n")
        log.flush()
        completed = subprocess.run(
            command, cwd=Path.cwd(), env=env, stdout=log, stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.monotonic() - start
    if completed.returncode != 0:
        raise RuntimeError(
            f"{dataset}/{variant} failed on GPU {gpu} (exit {completed.returncode}); log={log_path}"
        )
    actual_results = run_dir / "results.json"
    if not actual_results.is_file():
        raise RuntimeError(f"{dataset}/{variant} returned success without results; log={log_path}")
    checkpoint_validation_errors = _checkpoint_errors(expected_checkpoints, EXPECTED_SEEDS)
    if checkpoint_validation_errors:
        raise RuntimeError(
            f"{dataset}/{variant} checkpoint validation failed: "
            + "; ".join(checkpoint_validation_errors)
            + f"; log={log_path}"
        )
    complete_path.write_text(json.dumps({
        "dataset": dataset,
        "model": "multi_order_bank",
        "protocol_version": "unified_full_graph_nc_v1",
        "readout": readout,
        "fusion_mode": fusion,
        "variant": variant,
        "seed": BASE_SEED,
        "num_runs": NUM_RUNS,
        "run_seeds": list(EXPECTED_SEEDS),
        "gpu": gpu,
        "seconds": elapsed,
        "command": command,
        "results": str(actual_results),
        "checkpoints": [str(path) for path in expected_checkpoints],
        "checkpoint_validation": "passed",
    }, indent=2), encoding="utf-8")
    return {"dataset": dataset, "variant": variant, "status": "complete", "gpu": gpu, "seconds": elapsed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen 3x2 Multi-Order Bank NC factorial.")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--gpus", default=os.environ.get("GPU_IDS", "0,1"), help="comma-separated physical GPU IDs")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpus = tuple(part.strip() for part in args.gpus.split(",") if part.strip())
    if not gpus:
        parser.error("at least one GPU ID is required")
    jobs = list(_jobs(tuple(args.datasets)))
    if args.dry_run:
        for dataset, readout, fusion, variant in jobs:
            print(dataset, readout, fusion, variant)
        print(
            f"jobs={len(jobs)} seeds_per_job={NUM_RUNS} seeds=42,43,44 "
            f"datasets={len(set(job[0] for job in jobs))} variants={len(set(job[3] for job in jobs))}"
        )
        return

    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)

    def report(outcome: GPUJobOutcome[tuple[str, str, str, str], dict]) -> None:
        if outcome.error is None:
            print(json.dumps(outcome.result), flush=True)
        else:
            print(f"FAILED {outcome.job} on GPU {outcome.gpu}: {outcome.error}", flush=True)

    summary = run_gpu_jobs(jobs, gpus, lambda job, gpu: _run_job(*job, gpu, root), on_finish=report)
    if summary.failures:
        details = "\n".join(f"{row.job} on GPU {row.gpu}: {row.error}" for row in summary.failures)
        if summary.not_started:
            details += "\nNot started after failure: " + ", ".join(str(job) for _, job in summary.not_started)
        raise SystemExit("Factorial jobs failed:\n" + details)
    print(f"Completed {len(jobs)} dataset/variant jobs ({len(jobs) * NUM_RUNS} runs).", flush=True)


if __name__ == "__main__":
    main()
