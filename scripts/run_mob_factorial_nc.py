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
READOUTS = ("terminal", "uniform", "gpr")
FUSIONS = ("plain_mlp", "residual")
BASE_SEED = 42
NUM_RUNS = 3
OUTPUT_ROOT = Path("outputs/mob_factorial_nc_v1")


def _jobs(datasets: tuple[str, ...]):
    for dataset in datasets:
        for fusion in FUSIONS:
            for readout in READOUTS:
                variant = f"mob_{readout}_{'plain' if fusion == 'plain_mlp' else 'residual'}"
                yield dataset, readout, fusion, variant


def _run_job(dataset: str, readout: str, fusion: str, variant: str, gpu: str, root: Path) -> dict:
    output_dir = root / dataset / variant
    complete_path = output_dir / "complete.json"
    checkpoint_base = output_dir / "best.pt"
    expected_checkpoints = [output_dir / f"best_run{run_id}.pt" for run_id in (1, 2, 3)]
    if complete_path.is_file() and all(p.is_file() for p in expected_checkpoints):
        previous = json.loads(complete_path.read_text(encoding="utf-8"))
        if Path(previous.get("results", "")).is_file():
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
    if not actual_results.is_file() or not all(p.is_file() for p in expected_checkpoints):
        raise RuntimeError(f"{dataset}/{variant} returned success without all expected outputs; log={log_path}")
    complete_path.write_text(json.dumps({
        "dataset": dataset,
        "model": "multi_order_bank",
        "readout": readout,
        "fusion_mode": fusion,
        "variant": variant,
        "seed": BASE_SEED,
        "num_runs": NUM_RUNS,
        "run_seeds": [42, 43, 44],
        "gpu": gpu,
        "seconds": elapsed,
        "command": command,
        "results": str(actual_results),
        "checkpoints": [str(path) for path in expected_checkpoints],
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
        for index, (dataset, readout, fusion, variant) in enumerate(jobs):
            gpu = gpus[index % len(gpus)]
            print(dataset, readout, fusion, variant, "GPU", gpu)
        print(f"jobs={len(jobs)} seeds_per_job={NUM_RUNS} seeds=42,43,44")
        return

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
            future = pool.submit(_run_job, *job, gpus[index % len(gpus)], root)
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
                future = pool.submit(_run_job, *job, gpus[index % len(gpus)], root)
                active[future] = job
    if failures:
        details = "\n".join(f"{job}: {message}" for job, message in failures)
        raise SystemExit("Factorial jobs failed:\n" + details)
    print(f"Completed {len(jobs)} dataset/variant jobs ({len(jobs) * NUM_RUNS} runs).", flush=True)


if __name__ == "__main__":
    main()
