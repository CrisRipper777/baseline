from __future__ import annotations

import argparse
import json
import math
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
SOURCE_COMMIT = "601bad98ff28e85c9aeac23b84b5db51ad4c7c86"
BRANCH = "s43_h15_h1r_h2ab"
OUTPUT_ROOT = Path("outputs/s43_h15_h1r_h2ab_v1")
VARIANTS = (
    "h1r_dual_agg_signed", "h1r_dual_functional_signed",
    "h2a_global_scalar", "h2a_node_scalar", "h2a_global_group", "h2a_node_group",
    "h2b_global_agg_correction", "h2b_global_diff_correction",
    "h2b_node_agg_correction", "h2b_node_diff_correction",
)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _check_branch() -> None:
    branch = _git("branch", "--show-current")
    if branch != BRANCH:
        raise RuntimeError(f"S4.3 Phase 2 must run on {BRANCH}, current branch is {branch}")
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
    if result.returncode:
        return None
    try:
        return int(result.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


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
        "baseline_device_memory_mb": baseline, "peak_device_memory_mb": peak,
        "peak_cuda_memory_mb": max(0, peak - (baseline or 0)) if peak is not None else None,
        "sampling_method": "nvidia-smi per physical GPU; one active job per GPU", "samples": samples,
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
        if any(k.startswith("test_") for k in metrics):
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
    checkpoints = [output_dir / ("best.pt" if preflight else f"best_run{i}.pt") for i in range(1, num_runs + 1)]
    complete_path = output_dir / "complete.json"
    if complete_path.is_file():
        try:
            old = json.loads(complete_path.read_text(encoding="utf-8"))
            if old.get("phase") == phase and not _checkpoint_errors(checkpoints, seeds):
                return {"job": job, "status": "already_complete", "gpu": gpu,
                        "trainable_params": old.get("trainable_params")}
        except Exception:
            pass

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / f"run_{stamp}"
    command = [
        sys.executable, "-m", "src.main", f"dataset={dataset}", "task=nc",
        "model=adaptive_relation_pilot", f"model.variant={variant}",
        "model.hidden_dim=256", "model.dropout=0.2", "seed=42", f"num_runs={num_runs}",
        "device=cuda:0", f"task.epochs={epochs}", "task.lr=1e-3", "task.weight_decay=1e-4",
        "task.patience=30", "task.early_stop_min_epoch=30", "task.early_stop_min_delta=1e-4",
        "task.grad_clip=1.0", "task.training_mode=full_graph", f"task.protocol_version={PROTOCOL}",
        "task.evaluate_test=false", f"task.save_ckpt_path={output_dir / 'best.pt'}",
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
    metrics_path = output_dir / "run_metrics.json"
    run_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if run_metrics.get("run_seeds") != list(seeds) or len(run_metrics.get("runs", [])) != num_runs:
        raise RuntimeError(f"seed aggregation mismatch for {job}")
    if any(any(k.startswith("test_") for k in row.get("metrics", {})) for row in run_metrics["runs"]):
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
        "launch_log": str(log_path), "run_metrics_path": str(metrics_path), "run_metrics": run_metrics,
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
    missing = []
    for _, variant in _jobs("preflight", DATASETS):
        path = OUTPUT_ROOT / "preflight" / "ele-fashion" / variant / "complete.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("phase") != "preflight" or record.get("checkpoint_validation") != "passed" or record.get("test_evaluation") is not False:
            missing.append(str(path))
    if missing:
        raise SystemExit("formal launch blocked: preflight incomplete/invalid: " + ", ".join(missing))


def main():
    parser = argparse.ArgumentParser(description="S4.3 Phase 2 independent H1.5/H1R/H2a/H2b audits")
    parser.add_argument("phase", choices=("h15", "preflight", "formal", "analyze"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--gpus", default=os.environ.get("GPU_IDS", "0,1"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    _check_branch()
    if args.phase in ("h15", "analyze"):
        if args.dry_run:
            print(f"phase={args.phase} datasets={','.join(args.datasets)} jobs=0 training_runs=0")
            return
        from scripts.analyze_s43_h15_h1r_h2ab import run_analysis
        run_analysis(args.phase, tuple(args.datasets))
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
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    result_lock = threading.Lock()
    safe_results: list[dict[str, Any]] = []

    def on_finish(outcome: GPUJobOutcome[Any, Any]):
        result = outcome.result if outcome.error is None else {"job": outcome.job, "status": "failed", "gpu": outcome.gpu, "error": repr(outcome.error)}
        with result_lock:
            safe_results.append(result)
        print(json.dumps(result), flush=True)

    # _safe_job converts per-job failures to records so unrelated variants/datasets continue.
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
    print(f"Completed phase={args.phase} jobs={len(jobs)} runs={len(jobs) * (1 if args.phase == 'preflight' else 3)}")


if __name__ == "__main__":
    main()
