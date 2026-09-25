from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
MODELS = ("mlp", "gcn", "sage", "mmgcn", "mgat", "dip", "dgf", "dmgc", "lgmrec")
BASE_SEED = 42
NUM_RUNS = 3
OUTPUT_ROOT = Path("outputs/nc_benchmark_v1")


def _run_job(dataset: str, model: str, gpu: str, root: Path) -> dict:
    output_dir = root / dataset / model
    complete_path = output_dir / "complete.json"
    metrics_path = output_dir / "per_run_metrics.json"
    if complete_path.is_file() and metrics_path.is_file():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        previous = json.loads(complete_path.read_text(encoding="utf-8"))
        if payload.get("run_seeds") == [42, 43, 44] and Path(previous.get("results", "")).is_file():
            return {"dataset": dataset, "model": model, "status": "already_complete", "gpu": gpu}

    output_dir.mkdir(parents=True, exist_ok=True)
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = output_dir / f"launch_{attempt}.log"
    run_dir = output_dir / f"run_{attempt}"
    metrics_path = output_dir / "per_run_metrics.json"
    command = [
        sys.executable, "-m", "src.main",
        f"dataset={dataset}", "task=nc", f"model={model}",
        f"seed={BASE_SEED}", f"num_runs={NUM_RUNS}", "device=cuda:0",
        f"task.run_metrics_path={metrics_path}",
        f"hydra.run.dir={run_dir}",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(command) + "\n")
        log.write(f"PHYSICAL_GPU: {gpu}\n")
        log.flush()
        completed = subprocess.run(command, cwd=Path.cwd(), env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - start
    if completed.returncode != 0:
        raise RuntimeError(f"{dataset}/{model} failed on GPU {gpu} (exit {completed.returncode}); log={log_path}")
    actual_results = run_dir / "results.json"
    if not metrics_path.is_file() or not actual_results.is_file():
        raise RuntimeError(f"{dataset}/{model} returned success without run metrics/results; log={log_path}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if payload.get("run_seeds") != [42, 43, 44] or len(payload.get("runs", [])) != 3:
        raise RuntimeError(f"{dataset}/{model} sidecar does not contain three expected runs")
    complete_path.write_text(json.dumps({
        "dataset": dataset,
        "model": model,
        "protocol_version": "unified_full_graph_nc_v1",
        "seed": BASE_SEED,
        "num_runs": NUM_RUNS,
        "run_seeds": [42, 43, 44],
        "gpu": gpu,
        "seconds": elapsed,
        "command": command,
        "run_metrics": str(metrics_path),
        "results": str(actual_results),
    }, indent=2), encoding="utf-8")
    return {"dataset": dataset, "model": model, "status": "complete", "gpu": gpu, "seconds": elapsed}


def _assert_formal_launch_ready() -> None:
    factorial_root = Path("outputs/mob_factorial_nc_v1")
    missing = [
        str(factorial_root / dataset / f"mob_{readout}_{fusion}" / "complete.json")
        for dataset in DATASETS
        for fusion in ("plain", "residual")
        for readout in ("terminal", "uniform", "gpr")
        if not (factorial_root / dataset / f"mob_{readout}_{fusion}" / "complete.json").is_file()
    ]
    if missing:
        raise RuntimeError("The six-way factorial is not complete; missing:\n" + "\n".join(missing))
    preflight_path = Path("outputs/nc_preflight_v1/preflight.json")
    if not preflight_path.is_file() or not json.loads(preflight_path.read_text(encoding="utf-8")).get("all_passed"):
        raise RuntimeError("The six heavy-model ele-fashion CUDA preflights have not all passed.")
    provenance_path = Path("outputs/nc_benchmark_v1/provenance.json")
    if not provenance_path.is_file():
        raise RuntimeError("Write and verify outputs/nc_benchmark_v1/provenance.json before the formal baseline launch.")
    summary_path = Path("results/nc_benchmark_v1/summary.json")
    if not summary_path.is_file():
        raise RuntimeError("Run the factorial analyzer before the baseline launch.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("factorial_runs") != 90:
        raise RuntimeError("Factorial summary does not contain the full 90-run matrix.")
    if summary.get("fusion_attribution_required"):
        marker = Path("results/nc_benchmark_v1/fusion_attribution.csv")
        if not marker.is_file() or "CONDITIONAL_FUSION_COMPLETE" not in marker.read_text(encoding="utf-8"):
            raise RuntimeError("Fusion attribution was triggered; complete the conditional GPR stage before formal baseline launch.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen 9-model by 5-dataset NC benchmark.")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--gpus", default=os.environ.get("GPU_IDS", "0,1"), help="comma-separated physical GPU IDs")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpus = tuple(part.strip() for part in args.gpus.split(",") if part.strip())
    if not gpus:
        parser.error("at least one GPU ID is required")
    jobs = [(dataset, model) for dataset in args.datasets for model in args.models]
    if args.dry_run:
        for index, (dataset, model) in enumerate(jobs):
            print(dataset, model, "GPU", gpus[index % len(gpus)])
        print(f"jobs={len(jobs)} seeds_per_job={NUM_RUNS} seeds=42,43,44")
        return

    _assert_formal_launch_ready()
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    failures = []
    iterator = iter(enumerate(jobs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        active = {}
        for _ in range(len(gpus)):
            try:
                index, job = next(iterator)
            except StopIteration:
                break
            dataset, model = job
            future = pool.submit(_run_job, dataset, model, gpus[index % len(gpus)], root)
            active[future] = job
        stop = False
        while active:
            done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                job = active.pop(future)
                try:
                    print(json.dumps(future.result()), flush=True)
                except Exception as exc:
                    failures.append((job, str(exc)))
                    print(f"FAILED {job}: {exc}", flush=True)
                    stop = True
            if stop:
                for future in active:
                    future.cancel()
                break
            while len(active) < len(gpus):
                try:
                    index, job = next(iterator)
                except StopIteration:
                    break
                dataset, model = job
                future = pool.submit(_run_job, dataset, model, gpus[index % len(gpus)], root)
                active[future] = job
    if failures:
        details = "\n".join(f"{job}: {message}" for job, message in failures)
        raise SystemExit("Baseline jobs failed:\n" + details)
    print(f"Completed {len(jobs)} dataset/model jobs ({len(jobs) * NUM_RUNS} runs).", flush=True)


if __name__ == "__main__":
    main()
