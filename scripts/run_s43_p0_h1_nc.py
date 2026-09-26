from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gpu_scheduler import GPUJobOutcome, run_gpu_jobs

DATASETS = ("Movies", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
PROTOCOL = "unified_full_graph_nc_v1"
OUTPUT_ROOT = Path("outputs/s43_p0_h1_v1")
P0_VARIANTS = ("p0_relation_only", "p0_residual", "p0_concat")
H1_VARIANTS = ("h1_dual_agg", "h1_dual_functional_static", "h1_dual_functional_global")
ALL_VARIANTS = P0_VARIANTS + H1_VARIANTS


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _jobs(phase: str, datasets: tuple[str, ...]):
    if phase == "preflight":
        return [("ele-fashion", variant) for variant in ALL_VARIANTS]
    if phase == "formal":
        return [(dataset, variant) for dataset in datasets for variant in ALL_VARIANTS]
    raise ValueError(phase)


def _cuda_memory_used(gpu: str) -> int | None:
    result = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False, timeout=5,
    )
    if result.returncode:
        return None
    try:
        return int(result.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def _checkpoint_errors(paths: list[Path], seeds: tuple[int, ...], require_three: bool) -> list[str]:
    import torch

    errors = []
    for path, seed in zip(paths, seeds, strict=True):
        if not path.is_file():
            errors.append(f"missing checkpoint {path}")
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            errors.append(f"cannot read {path}: {exc}")
            continue
        if payload.get("task") != "nc" or payload.get("protocol_version") != PROTOCOL:
            errors.append(f"wrong protocol in {path}")
        if int(payload.get("seed", -1)) != seed:
            errors.append(f"wrong seed in {path}")
        if payload.get("selection") != "best_val_accuracy" or payload.get("epoch") is None:
            errors.append(f"missing validation-selected checkpoint metadata in {path}")
        metrics = payload.get("metrics", {})
        if not {"val_acc", "val_macro_f1"}.issubset(metrics):
            errors.append(f"missing validation metrics in {path}")
        if any(key.startswith("test_") for key in metrics):
            errors.append(f"test metrics are forbidden in {path}")
    if require_three and len(paths) != 3:
        errors.append(f"expected three formal checkpoints, got {len(paths)}")
    return errors


def _monitor_process(proc: subprocess.Popen, gpu: str, interval: float = 1.0) -> dict[str, Any]:
    baseline = _cuda_memory_used(gpu)
    peak = baseline
    samples = 0
    while proc.poll() is None:
        used = _cuda_memory_used(gpu)
        if used is not None:
            peak = used if peak is None else max(peak, used)
            samples += 1
        time.sleep(interval)
    used = _cuda_memory_used(gpu)
    if used is not None:
        peak = used if peak is None else max(peak, used)
        samples += 1
    return {
        "baseline_device_memory_mb": baseline,
        "peak_device_memory_mb": peak,
        "peak_cuda_memory_mb": max(0, peak - (baseline or 0)) if peak is not None else None,
        "sampling_method": "nvidia-smi per physical GPU, one active job per GPU",
        "samples": samples,
    }


def _epoch_counts(log_path: Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    active_run = 1
    pattern = re.compile(r"\[Run (\d+)/\d+\] seed=")
    epoch = re.compile(r"Epoch\s+\d{5}\s+\|")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            active_run = int(match.group(1))
        elif epoch.search(line):
            counts[active_run] = counts.get(active_run, 0) + 1
    return counts


def _run_job(job: tuple[str, str], gpu: str, phase: str):
    import torch

    dataset, variant = job
    training_branch = _git("branch", "--show-current")
    training_commit = _git("rev-parse", "HEAD")
    if training_branch != "s43_p0_h1":
        raise RuntimeError(f"S4.3 jobs may run only on s43_p0_h1, got {training_branch}")
    group = "preflight" if phase == "preflight" else "formal"
    output_dir = OUTPUT_ROOT / group / dataset / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    is_preflight = phase == "preflight"
    seeds = (42,) if is_preflight else SEEDS
    num_runs, epochs = (1, 1) if is_preflight else (3, 300)
    checkpoints = [output_dir / ("best.pt" if is_preflight else f"best_run{i}.pt")
                   for i in range(1, num_runs + 1)]
    complete = output_dir / "complete.json"
    if complete.is_file():
        try:
            record = json.loads(complete.read_text(encoding="utf-8"))
            if record.get("phase") == phase and not _checkpoint_errors(checkpoints, seeds, not is_preflight):
                return {"job": job, "status": "already_complete", "gpu": gpu,
                        "trainable_params": record.get("trainable_params")}
        except Exception:
            pass

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / f"run_{stamp}"
    command = [
        sys.executable, "-m", "src.main", f"dataset={dataset}", "task=nc",
        "model=relation_basis_pilot", f"model.variant={variant}",
        "model.hidden_dim=256", "model.dropout=0.2",
        "seed=42", f"num_runs={num_runs}", "device=cuda:0",
        f"task.epochs={epochs}", "task.lr=1e-3", "task.weight_decay=1e-4",
        "task.patience=30", "task.early_stop_min_epoch=30",
        "task.early_stop_min_delta=1e-4", "task.grad_clip=1.0",
        "task.training_mode=full_graph", "task.protocol_version=unified_full_graph_nc_v1",
        "task.evaluate_test=false", f"task.save_ckpt_path={output_dir / 'best.pt'}",
        f"task.run_metrics_path={output_dir / 'run_metrics.json'}",
        f"hydra.run.dir={run_dir}",
    ]
    if is_preflight:
        command.extend(["task.patience=1", "task.early_stop_min_epoch=1"])
    log_path = output_dir / f"launch_{stamp}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"PHYSICAL_GPU={gpu}\nPHASE={phase}\nCOMMAND={' '.join(command)}\n")
        log.flush()
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        memory = _monitor_process(proc, gpu)
        return_code = proc.wait()
    elapsed = time.monotonic() - started
    if return_code:
        raise RuntimeError(f"{job} failed on GPU {gpu}; see {log_path}")
    errors = _checkpoint_errors(checkpoints, seeds, not is_preflight)
    if errors:
        raise RuntimeError("checkpoint validation: " + "; ".join(errors))
    if memory["samples"] == 0 or memory["peak_cuda_memory_mb"] is None:
        raise RuntimeError(f"CUDA memory could not be profiled for {job}")

    metrics_path = output_dir / "run_metrics.json"
    run_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if run_metrics.get("run_seeds") != list(seeds) or len(run_metrics.get("runs", [])) != num_runs:
        raise RuntimeError(f"invalid seed aggregation for {job}")
    if any(any(k.startswith("test_") for k in row.get("metrics", {})) for row in run_metrics["runs"]):
        raise RuntimeError(f"test evaluation detected for {job}")
    epoch_counts = _epoch_counts(log_path)
    actual_epochs = sum(epoch_counts.values())
    if actual_epochs < num_runs:
        raise RuntimeError(f"could not establish completed epoch count for {job}")
    data_info = run_metrics["runs"][0]["metadata"]
    trainable_params = int(data_info["model_parameters"]) + int(data_info["classifier_parameters"])
    record = {
        "phase": phase, "dataset": dataset, "variant": variant,
        "training_branch": training_branch, "training_commit": training_commit,
        "protocol_version": PROTOCOL, "task": "nc", "test_evaluation": False,
        "base_seed": 42, "run_seeds": list(seeds), "num_runs": num_runs, "epochs_max": epochs,
        "selection": "best_val_accuracy", "best_epochs": [int(row["metadata"]["best_epoch"]) for row in run_metrics["runs"]],
        "actual_epochs": epoch_counts, "trainable_params": trainable_params,
        "model_trainable_params": int(data_info["model_parameters"]),
        "classifier_trainable_params": int(data_info["classifier_parameters"]),
        "wall_clock_seconds": elapsed,
        "mean_epoch_seconds_proxy": elapsed / actual_epochs,
        "mean_epoch_seconds_note": "job wall time divided by observed completed epoch log lines; includes setup and validation overhead",
        **memory,
        "gpu_physical_id": str(gpu), "command": command,
        "checkpoint_paths": [str(path) for path in checkpoints],
        "checkpoint_validation": "passed", "launch_log": str(log_path),
        "run_metrics_path": str(metrics_path),
        "run_metrics": run_metrics,
    }
    complete.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return {"job": job, "status": "complete", "gpu": gpu, "seconds": elapsed,
            "peak_cuda_memory_mb": memory["peak_cuda_memory_mb"], "trainable_params": trainable_params}


def _assert_preflight_complete() -> None:
    missing = []
    for dataset, variant in _jobs("preflight", DATASETS):
        path = OUTPUT_ROOT / "preflight" / dataset / variant / "complete.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("phase") != "preflight" or record.get("checkpoint_validation") != "passed":
            missing.append(str(path))
    if missing:
        raise SystemExit("formal launch blocked: preflight is incomplete: " + ", ".join(missing))


def main():
    parser = argparse.ArgumentParser(description="Run the S4.3 P0/H1 node-classification pilots.")
    parser.add_argument("phase", choices=("preflight", "formal"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--gpus", default=os.environ.get("GPU_IDS", "0,1"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpus = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    if not gpus:
        parser.error("at least one GPU ID is required")
    if len(set(gpus)) != len(gpus):
        parser.error("each physical GPU may be scheduled only once")
    jobs = _jobs(args.phase, tuple(args.datasets))
    if args.dry_run:
        for job in jobs:
            print(job)
        print(f"jobs={len(jobs)} runs={len(jobs) * (1 if args.phase == 'preflight' else 3)}")
        return
    if args.phase == "formal":
        _assert_preflight_complete()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    def report(outcome: GPUJobOutcome[Any, Any]):
        print(json.dumps(outcome.result) if outcome.error is None else
              f"FAILED {outcome.job} gpu={outcome.gpu}: {outcome.error}", flush=True)

    summary = run_gpu_jobs(jobs, gpus, lambda job, gpu: _run_job(job, gpu, args.phase), report)
    payload = {
        "phase": args.phase, "requested_jobs": len(jobs),
        "completed": sum(outcome.error is None for outcome in summary.outcomes),
        "failed": [{"job": outcome.job, "gpu": outcome.gpu, "error": str(outcome.error)}
                   for outcome in summary.failures],
        "not_started": [job for _, job in summary.not_started],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT_ROOT / f"{args.phase}_jobs.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if summary.failures or summary.not_started:
        raise SystemExit(f"{args.phase} incomplete; details in {OUTPUT_ROOT / f'{args.phase}_jobs.json'}")
    print(f"Completed phase={args.phase} jobs={len(jobs)} runs={len(jobs) * (1 if args.phase == 'preflight' else 3)}",
          flush=True)


if __name__ == "__main__":
    main()
