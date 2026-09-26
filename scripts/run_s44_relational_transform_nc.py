from __future__ import annotations

import argparse
import hashlib
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
VARIANTS = (
    "s44_scalar_global", "s44_scalar_raw", "s44_scalar_rel", "s44_scalar_rel_multi",
    "s44_lowrank_global", "s44_lowrank_rel", "s44_lowrank_rel_multi",
    "s44_expert_uniform", "s44_expert_rel", "s44_expert_rel_multi",
)
PROTOCOL = "unified_full_graph_nc_v1"
SOURCE_COMMIT = "7b98b31a78fb978fadd96448bbcfd3ffa295d82f"
BRANCH = "s44_relational_transform"
OUTPUT_ROOT = ROOT / "outputs/s44_relational_transform_v1"
RESULT_ROOT = ROOT / "results/s44_relational_transform_v1"
MAX_PREFLIGHT_MEMORY_MB = 20 * 1024


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _check_branch() -> None:
    branch = _git("branch", "--show-current")
    if branch != BRANCH:
        raise RuntimeError(f"S4.4 must run on {BRANCH}, current branch is {branch}")
    subprocess.run(["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, "HEAD"], cwd=ROOT, check=True)


def _jobs(phase: str, datasets: tuple[str, ...]):
    if phase == "preflight":
        return [("ele-fashion", variant) for variant in VARIANTS]
    if phase == "formal":
        return [(dataset, variant) for dataset in datasets for variant in VARIANTS]
    raise ValueError(phase)


def _cuda_memory_used(gpu: str) -> int | None:
    result = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False, timeout=5,
    )
    try:
        return int(result.stdout.strip().splitlines()[0]) if result.returncode == 0 else None
    except (ValueError, IndexError):
        return None


def _monitor_process(proc: subprocess.Popen, gpu: str, interval: float = 1.0) -> dict[str, Any]:
    baseline = _cuda_memory_used(gpu)
    peak, samples = baseline, 0
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
        "baseline_device_memory_mb": baseline, "peak_device_memory_mb": peak,
        "peak_cuda_memory_mb": max(0, peak - (baseline or 0)) if peak is not None else None,
        "sampling_method": "nvidia-smi per physical GPU, one worker per GPU", "samples": samples,
    }


def _checkpoint_errors(paths: list[Path], seeds: tuple[int, ...]) -> list[str]:
    import torch
    errors = []
    for path, seed in zip(paths, seeds, strict=True):
        if not path.is_file():
            errors.append(f"missing checkpoint {path}")
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            errors.append(f"cannot load {path}: {exc}")
            continue
        if payload.get("task") != "nc" or payload.get("protocol_version") != PROTOCOL:
            errors.append(f"wrong task/protocol in {path}")
        if int(payload.get("seed", -1)) != seed or payload.get("selection") != "best_val_accuracy":
            errors.append(f"wrong seed/selection in {path}")
        metrics = payload.get("metrics", {})
        if not {"val_acc", "val_macro_f1"}.issubset(metrics):
            errors.append(f"validation metrics missing in {path}")
        if any(key.startswith("test_") for key in metrics):
            errors.append(f"test metric found in {path}")
        for group in (payload.get("model_state", {}), payload.get("head_state", {})):
            if any(not bool(torch.isfinite(t).all()) for t in group.values() if torch.is_tensor(t)):
                errors.append(f"nonfinite checkpoint tensor in {path}")
                break
    return errors


def _epoch_counts(path: Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    active = 1
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"\[Run (\d+)/\d+\] seed=", line)
        if match:
            active = int(match.group(1))
        elif re.search(r"Epoch\s+\d{5}\s+\|", line):
            counts[active] = counts.get(active, 0) + 1
    return counts


def _run_job(job: tuple[str, str], gpu: str, phase: str):
    import torch
    dataset, variant = job
    _check_branch()
    output_dir = OUTPUT_ROOT / ("preflight" if phase == "preflight" else "formal") / dataset / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = phase == "preflight"
    seeds = (42,) if preflight else SEEDS
    num_runs, epochs = (1, 1) if preflight else (3, 300)
    checkpoints = [output_dir / ("best.pt" if preflight else f"best_run{i}.pt")
                   for i in range(1, num_runs + 1)]
    complete_path = output_dir / "complete.json"
    if complete_path.is_file():
        try:
            old = json.loads(complete_path.read_text(encoding="utf-8"))
            if old.get("phase") == phase and not _checkpoint_errors(checkpoints, seeds):
                return {"job": job, "status": "already_complete", "gpu": gpu,
                        "trainable_params": old.get("trainable_params"),
                        "peak_cuda_memory_mb": old.get("peak_cuda_memory_mb")}
        except Exception:
            pass

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = OUTPUT_ROOT / "hydra_runs" / phase / dataset / variant / stamp
    command = [
        sys.executable, "-m", "src.main", f"dataset={dataset}", "task=nc",
        "model=relational_transform_pilot", f"model.variant={variant}",
        "model.hidden_dim=256", "model.dropout=0.2", "model.relation_dim=32", "model.lowrank_rank=8",
        "seed=42", f"num_runs={num_runs}", "device=cuda:0", f"task.epochs={epochs}",
        "task.lr=1e-3", "task.weight_decay=1e-4", "task.patience=30",
        "task.early_stop_min_epoch=30", "task.early_stop_min_delta=1e-4", "task.grad_clip=1.0",
        "task.training_mode=full_graph", f"task.protocol_version={PROTOCOL}", "task.evaluate_test=false",
        f"task.save_ckpt_path={output_dir / 'best.pt'}",
        f"task.run_metrics_path={output_dir / 'run_metrics.json'}", f"hydra.run.dir={run_dir}",
    ]
    if preflight:
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
        raise RuntimeError(f"{job} failed on GPU {gpu}, exit={return_code}; see {log_path}")
    errors = _checkpoint_errors(checkpoints, seeds)
    if errors:
        raise RuntimeError(f"checkpoint validation failed for {job}: {errors}")
    if memory["samples"] < 1 or memory["peak_cuda_memory_mb"] is None:
        raise RuntimeError(f"CUDA memory could not be profiled for {job}")
    run_metrics = json.loads((output_dir / "run_metrics.json").read_text(encoding="utf-8"))
    if run_metrics.get("run_seeds") != list(seeds) or len(run_metrics.get("runs", [])) != num_runs:
        raise RuntimeError(f"seed aggregation mismatch for {job}")
    if any(any(key.startswith("test_") for key in row.get("metrics", {})) for row in run_metrics["runs"]):
        raise RuntimeError(f"test evaluation detected for {job}")
    epoch_counts = _epoch_counts(log_path)
    actual_epochs = sum(epoch_counts.values())
    if actual_epochs < num_runs:
        raise RuntimeError(f"completed epoch count unavailable for {job}")
    first_meta = run_metrics["runs"][0]["metadata"]
    params = int(first_meta["model_parameters"]) + int(first_meta["classifier_parameters"])
    record = {
        "phase": phase, "dataset": dataset, "variant": variant, "training_branch": BRANCH,
        "training_commit": _git("rev-parse", "HEAD"), "source_commit": SOURCE_COMMIT,
        "protocol_version": PROTOCOL, "task": "nc", "test_evaluation": False,
        "base_seed": 42, "run_seeds": list(seeds), "num_runs": num_runs, "epochs_max": epochs,
        "selection": "best_val_accuracy", "best_epochs": [int(r["metadata"]["best_epoch"]) for r in run_metrics["runs"]],
        "actual_epochs": epoch_counts, "trainable_params": params,
        "model_trainable_params": int(first_meta["model_parameters"]),
        "classifier_trainable_params": int(first_meta["classifier_parameters"]),
        "wall_clock_seconds": elapsed, "mean_epoch_seconds_proxy": elapsed / actual_epochs,
        "mean_epoch_seconds_note": "job wall time / logged epochs; includes setup and validation",
        **memory, "gpu_physical_id": str(gpu), "command": command,
        "checkpoint_paths": [str(p) for p in checkpoints], "checkpoint_validation": "passed",
        "launch_log": str(log_path), "run_metrics_path": str(output_dir / "run_metrics.json"),
        "hydra_run_dir": str(run_dir), "run_metrics": run_metrics,
    }
    complete_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return {"job": job, "status": "complete", "gpu": gpu, "seconds": elapsed,
            "peak_cuda_memory_mb": memory["peak_cuda_memory_mb"], "trainable_params": params}


def _safe_job(job, gpu, phase):
    try:
        return _run_job(job, gpu, phase)
    except Exception as exc:
        return {"job": job, "status": "failed", "gpu": gpu, "error": repr(exc)}


def _assert_preflight_complete() -> None:
    missing, memory_overs = [], []
    for dataset, variant in _jobs("preflight", DATASETS):
        path = OUTPUT_ROOT / "preflight" / dataset / variant / "complete.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("phase") != "preflight" or record.get("checkpoint_validation") != "passed" or record.get("test_evaluation") is not False:
            missing.append(str(path))
        peak = record.get("peak_cuda_memory_mb")
        if peak is None or int(peak) >= MAX_PREFLIGHT_MEMORY_MB:
            memory_overs.append((str(path), peak))
    if missing:
        raise SystemExit("formal launch blocked: preflight incomplete/invalid: " + ", ".join(missing))
    if memory_overs:
        raise SystemExit(f"formal launch blocked: ele-fashion preflight peak memory >=20 GiB or unavailable: {memory_overs}")


def main():
    parser = argparse.ArgumentParser(description="S4.4 relational transformation pilot, full-graph NC only")
    parser.add_argument("phase", choices=("patch", "preflight", "formal", "analyze"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--gpus", default=os.environ.get("GPU_IDS", "0,1"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    _check_branch()
    if args.phase in {"patch", "analyze"}:
        if args.dry_run:
            print(f"phase={args.phase} datasets={','.join(args.datasets)} jobs=0 training_runs=0")
            return
        if args.phase == "patch":
            from scripts.analyze_s44_relational_transform import run_patch
            print(json.dumps(run_patch(), indent=2))
        else:
            from scripts.analyze_s44_relational_transform import run_analysis
            print(json.dumps(run_analysis(tuple(args.datasets)), indent=2))
        return
    gpus = tuple(item.strip() for item in args.gpus.split(",") if item.strip())
    if not gpus or len(set(gpus)) != len(gpus):
        parser.error("--gpus must list unique physical GPU IDs")
    jobs = _jobs(args.phase, tuple(args.datasets))
    if args.dry_run:
        for job in jobs:
            print(job)
        print(f"jobs={len(jobs)} runs={len(jobs) * (1 if args.phase == 'preflight' else 3)}")
        return
    if args.phase == "formal":
        _assert_preflight_complete()
        if _git("status", "--porcelain"):
            raise SystemExit("formal launch requires a clean committed training tree")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    result_lock = threading.Lock()
    safe_results: list[dict[str, Any]] = []

    def on_finish(outcome: GPUJobOutcome[Any, Any]):
        result = outcome.result if outcome.error is None else {
            "job": outcome.job, "status": "failed", "gpu": outcome.gpu, "error": repr(outcome.error)}
        with result_lock:
            safe_results.append(result)
        print(json.dumps(result), flush=True)

    summary = run_gpu_jobs(jobs, gpus, lambda job, gpu: _safe_job(job, gpu, args.phase), on_finish)
    payload = {
        "phase": args.phase, "requested_jobs": len(jobs),
        "completed": sum(row.get("status") in {"complete", "already_complete"} for row in safe_results),
        "failed": [row for row in safe_results if row.get("status") == "failed"],
        "not_started": [job for _, job in summary.not_started],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT_ROOT / f"{args.phase}_jobs.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if payload["failed"] or payload["not_started"]:
        raise SystemExit(f"{args.phase} completed with failures; see {OUTPUT_ROOT / f'{args.phase}_jobs.json'}")
    if args.phase == "preflight":
        _assert_preflight_complete()
    print(f"Completed phase={args.phase} jobs={len(jobs)} runs={len(jobs) * (1 if args.phase == 'preflight' else 3)}")


if __name__ == "__main__":
    main()
