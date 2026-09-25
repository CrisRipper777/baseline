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

try:
    from scripts.gpu_scheduler import GPUJobOutcome, run_gpu_jobs
except ModuleNotFoundError:
    from gpu_scheduler import GPUJobOutcome, run_gpu_jobs


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
MODELS = ("mlp", "gcn", "sage", "mmgcn", "mgat", "dip", "dgf", "dmgc", "lgmrec")
BASE_SEED = 42
NUM_RUNS = 3
EXPECTED_SEEDS = [42, 43, 44]
EXPECTED_METRICS = {"val_acc", "val_macro_f1", "test_acc", "test_macro_f1"}
EXPECTED_PROTOCOL = "unified_full_graph_nc_v1"
OUTPUT_ROOT = Path("outputs/nc_benchmark_v1")
FACTORIAL_VARIANTS = (
    "mob_terminal_plain", "mob_uniform_plain", "mob_gpr_plain",
    "mob_terminal_residual", "mob_uniform_residual", "mob_gpr_residual",
)


def _valid_run_metrics(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("protocol_version") != EXPECTED_PROTOCOL:
        return False
    if payload.get("run_seeds") != EXPECTED_SEEDS:
        return False
    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) != NUM_RUNS:
        return False
    required_metadata = {"best_epoch", "model_parameters", "optimizer", "optimizer_groups"}
    for expected_seed, run in zip(EXPECTED_SEEDS, runs):
        if not isinstance(run, dict) or run.get("seed") != expected_seed:
            return False
        metrics = run.get("metrics")
        if not isinstance(metrics, dict) or EXPECTED_METRICS - metrics.keys():
            return False
        metadata = run.get("metadata")
        if not isinstance(metadata, dict) or required_metadata - metadata.keys():
            return False
        if not isinstance(metadata.get("best_epoch"), int) or metadata["best_epoch"] < 1:
            return False
        if not isinstance(metadata.get("model_parameters"), (int, float)):
            return False
        if not isinstance(metadata.get("optimizer"), str) or not metadata["optimizer"]:
            return False
        if not isinstance(metadata.get("optimizer_groups"), list):
            return False
    return True


def _valid_complete_marker(marker: object, dataset: str, model: str, results_path: Path) -> bool:
    if not isinstance(marker, dict):
        return False
    return (
        marker.get("dataset") == dataset
        and marker.get("model") == model
        and marker.get("protocol_version") == EXPECTED_PROTOCOL
        and marker.get("run_seeds") == EXPECTED_SEEDS
        and marker.get("num_runs") == NUM_RUNS
        and Path(marker.get("results", "")).is_file()
        and results_path.is_file()
    )


def _can_resume(complete_path: Path, metrics_path: Path, dataset: str, model: str) -> bool:
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        marker = json.loads(complete_path.read_text(encoding="utf-8"))
        return _valid_run_metrics(payload) and _valid_complete_marker(marker, dataset, model, metrics_path)
    except (OSError, json.JSONDecodeError, TypeError):
        return False


def _run_job(dataset: str, model: str, gpu: str, root: Path) -> dict:
    output_dir = root / dataset / model
    complete_path = output_dir / "complete.json"
    metrics_path = output_dir / "per_run_metrics.json"
    if complete_path.is_file() and metrics_path.is_file() and _can_resume(complete_path, metrics_path, dataset, model):
        return {"dataset": dataset, "model": model, "status": "already_complete", "gpu": gpu}

    output_dir.mkdir(parents=True, exist_ok=True)
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = output_dir / f"launch_{attempt}.log"
    run_dir = output_dir / f"run_{attempt}"
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
    if not _valid_run_metrics(payload):
        raise RuntimeError(f"{dataset}/{model} sidecar failed protocol/run/metadata validation; log={log_path}")
    complete_path.write_text(json.dumps({
        "dataset": dataset,
        "model": model,
        "protocol_version": EXPECTED_PROTOCOL,
        "seed": BASE_SEED,
        "num_runs": NUM_RUNS,
        "run_seeds": EXPECTED_SEEDS,
        "gpu": gpu,
        "seconds": elapsed,
        "command": command,
        "run_metrics": str(metrics_path),
        "results": str(actual_results),
    }, indent=2), encoding="utf-8")
    return {"dataset": dataset, "model": model, "status": "complete", "gpu": gpu, "seconds": elapsed}


def _assert_formal_launch_ready() -> None:
    from scripts.run_mob_factorial_nc import _valid_resume

    factorial_root = Path("outputs/mob_factorial_nc_v1")
    missing = []
    for dataset in DATASETS:
        for variant in FACTORIAL_VARIANTS:
            output_dir = factorial_root / dataset / variant
            marker = output_dir / "complete.json"
            checkpoints = [output_dir / f"best_run{run_id}.pt" for run_id in (1, 2, 3)]
            if not marker.is_file() or not _valid_resume(marker, checkpoints, dataset, variant):
                missing.append(str(marker))
    if missing:
        raise RuntimeError("The six-way factorial is incomplete or has invalid checkpoints:\n" + "\n".join(missing))
    preflight_path = Path("outputs/nc_preflight_v1/preflight.json")
    if not preflight_path.is_file():
        raise RuntimeError("The node-heavy/edge-heavy CUDA preflight record is missing.")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    stats = preflight.get("dataset_statistics", {})
    if set(stats) != set(DATASETS):
        raise RuntimeError("Preflight must record statistics for all five formal NC datasets.")
    node_heavy = max(DATASETS, key=lambda dataset: stats[dataset]["num_nodes"])
    edge_heavy = max(DATASETS, key=lambda dataset: stats[dataset]["num_edges"])
    required_datasets = {node_heavy, edge_heavy}
    if set(preflight.get("selected_preflight_datasets", [])) != required_datasets:
        raise RuntimeError("Preflight selected datasets do not match the recorded node/edge-heavy union.")
    expected_pairs = {(dataset, model) for dataset in required_datasets for model in ("dip", "mmgcn", "mgat", "dgf", "dmgc", "lgmrec")}
    outcomes = preflight.get("outcomes", [])
    actual_pairs = {(row.get("dataset"), row.get("model")) for row in outcomes}
    if not preflight.get("all_passed") or actual_pairs != expected_pairs or any(
        row.get("status") != "complete" or row.get("exit_code") != 0 for row in outcomes
    ):
        raise RuntimeError("The required node-heavy/edge-heavy full-graph CUDA preflights have not all passed.")
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
            raise RuntimeError("Fusion attribution was triggered; stop before the formal baseline benchmark.")


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
        for dataset, model in jobs:
            print(dataset, model)
        print(f"jobs={len(jobs)} seeds_per_job={NUM_RUNS} seeds=42,43,44")
        return

    _assert_formal_launch_ready()
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)

    def report(outcome: GPUJobOutcome[tuple[str, str], dict]) -> None:
        if outcome.error is None:
            print(json.dumps(outcome.result), flush=True)
        else:
            print(f"FAILED {outcome.job} on GPU {outcome.gpu}: {outcome.error}", flush=True)

    summary = run_gpu_jobs(jobs, gpus, lambda job, gpu: _run_job(*job, gpu, root), on_finish=report)
    if summary.failures:
        details = "\n".join(f"{row.job} on GPU {row.gpu}: {row.error}" for row in summary.failures)
        if summary.not_started:
            details += "\nNot started after failure: " + ", ".join(str(job) for _, job in summary.not_started)
        raise SystemExit("Baseline jobs failed:\n" + details)
    print(f"Completed {len(jobs)} dataset/model jobs ({len(jobs) * NUM_RUNS} runs).", flush=True)


if __name__ == "__main__":
    main()
