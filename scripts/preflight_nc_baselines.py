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

from hydra import compose, initialize_config_dir

try:
    from scripts.gpu_scheduler import GPUJobOutcome, run_gpu_jobs
except ModuleNotFoundError:
    from gpu_scheduler import GPUJobOutcome, run_gpu_jobs


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
MODELS = ("dip", "mmgcn", "mgat", "dgf", "dmgc", "lgmrec")
OUTPUT_ROOT = Path("outputs/nc_preflight_v1")
FACTORIAL_ROOT = Path("outputs/mob_factorial_nc_v1")
FACTORIAL_VARIANTS = tuple(
    f"mob_{readout}_{fusion}"
    for fusion in ("plain", "residual")
    for readout in ("terminal", "uniform", "gpr")
)


class PreflightJobFailure(RuntimeError):
    def __init__(self, outcome: dict):
        self.outcome = outcome
        super().__init__(f"{outcome['dataset']}/{outcome['model']} status={outcome['status']} exit={outcome['exit_code']}")


def _assert_factorial_complete() -> None:
    try:
        from scripts.run_mob_factorial_nc import _valid_resume
    except ModuleNotFoundError:
        from run_mob_factorial_nc import _valid_resume

    missing: list[str] = []
    for dataset in DATASETS:
        for variant in FACTORIAL_VARIANTS:
            output_dir = FACTORIAL_ROOT / dataset / variant
            complete_path = output_dir / "complete.json"
            checkpoints = [output_dir / f"best_run{run_id}.pt" for run_id in (1, 2, 3)]
            if not complete_path.is_file() or not _valid_resume(complete_path, checkpoints, dataset, variant):
                missing.append(str(complete_path))
    if missing:
        raise RuntimeError("Complete and validate the 30-job factorial before heavy-model preflight; invalid/missing:\n" + "\n".join(missing))


def _dataset_statistics() -> dict[str, dict[str, int]]:
    from src.data import load_mag_data

    config_dir = str((Path.cwd() / "configs").resolve())
    statistics: dict[str, dict[str, int]] = {}
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        for dataset in DATASETS:
            cfg = compose(config_name="config", overrides=[f"dataset={dataset}", "task=nc", "seed=42"])
            data = load_mag_data(cfg, "nc", seed=42)
            statistics[dataset] = {
                "num_nodes": int(data.num_nodes),
                "num_edges": int(data.edge_index.size(1)),
                "input_dim": int(data.x.size(1)),
            }
            del data, cfg
    return statistics


def _select_heavy_datasets(statistics: dict[str, dict[str, int]]) -> tuple[str, ...]:
    node_heavy = max(DATASETS, key=lambda dataset: statistics[dataset]["num_nodes"])
    edge_heavy = max(DATASETS, key=lambda dataset: statistics[dataset]["num_edges"])
    selected = tuple(dataset for dataset in DATASETS if dataset in {node_heavy, edge_heavy})
    return selected


def _job(dataset: str, model: str, gpu: str, root: Path) -> dict:
    output = root / dataset / model
    output.mkdir(parents=True, exist_ok=True)
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output / f"run_{attempt}"
    log_path = output / f"preflight_{attempt}.log"
    command = [
        sys.executable, "-m", "src.main",
        f"dataset={dataset}", "task=nc", f"model={model}",
        "seed=42", "num_runs=1", "task.epochs=1", "device=cuda:0",
        f"hydra.run.dir={run_dir}",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    start = time.monotonic()
    exit_code: int | None = None
    launch_error: str | None = None
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(command) + "\nPHYSICAL_GPU: " + gpu + "\n")
        log.flush()
        try:
            result = subprocess.run(command, cwd=Path.cwd(), env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            exit_code = result.returncode
        except OSError as exc:
            launch_error = str(exc)
    elapsed = time.monotonic() - start
    has_results = (run_dir / "results.json").is_file()
    status = "complete" if exit_code == 0 and has_results else "failed"
    outcome = {
        "dataset": dataset,
        "model": model,
        "status": status,
        "exit_code": exit_code,
        "seconds": elapsed,
        "gpu": gpu,
        "log": str(log_path),
        "command": command,
        "results_present": has_results,
        "launch_error": launch_error,
    }
    if status != "complete":
        raise PreflightJobFailure(outcome)
    return outcome


def _outcome_record(outcome: GPUJobOutcome[tuple[str, str], dict]) -> dict:
    if outcome.error is None:
        return outcome.result
    if isinstance(outcome.error, PreflightJobFailure):
        row = dict(outcome.error.outcome)
        row["gpu"] = outcome.gpu
        return row
    dataset, model = outcome.job
    return {
        "dataset": dataset,
        "model": model,
        "status": "failed",
        "exit_code": None,
        "seconds": None,
        "gpu": outcome.gpu,
        "error": str(outcome.error),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="One-epoch full-graph CUDA preflight on node-heavy and edge-heavy NC datasets.")
    parser.add_argument("--gpus", default=os.environ.get("GPU_IDS", "0,1"))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpus = tuple(part.strip() for part in args.gpus.split(",") if part.strip())
    if not gpus:
        parser.error("at least one GPU ID is required")

    statistics = _dataset_statistics()
    node_heavy = max(DATASETS, key=lambda dataset: statistics[dataset]["num_nodes"])
    edge_heavy = max(DATASETS, key=lambda dataset: statistics[dataset]["num_edges"])
    selected = _select_heavy_datasets(statistics)
    print(json.dumps({
        "dataset_statistics": statistics,
        "node_heavy_dataset": node_heavy,
        "edge_heavy_dataset": edge_heavy,
        "selected_preflight_datasets": selected,
    }, indent=2), flush=True)

    jobs = [(dataset, model) for dataset in selected for model in MODELS]
    if args.dry_run:
        for dataset, model in jobs:
            print(dataset, model, "epochs=1", "num_runs=1", "full_graph_cuda")
        print(f"jobs={len(jobs)} datasets={len(selected)} models={len(MODELS)}")
        return

    _assert_factorial_complete()
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    records: dict[int, dict] = {}

    def report(outcome: GPUJobOutcome[tuple[str, str], dict]) -> None:
        row = _outcome_record(outcome)
        records[outcome.index] = row
        print(json.dumps(row), flush=True)

    summary = run_gpu_jobs(jobs, gpus, lambda job, gpu: _job(*job, gpu, root), on_finish=report)
    for index, job in summary.not_started:
        records[index] = {
            "dataset": job[0], "model": job[1], "status": "not_run_after_failure",
            "exit_code": None, "seconds": None,
        }
    outcomes = [records[index] for index in range(len(jobs))]
    all_passed = not summary.failures and not summary.not_started and all(row["status"] == "complete" for row in outcomes)
    payload = {
        "dataset_statistics": statistics,
        "node_heavy_dataset": node_heavy,
        "edge_heavy_dataset": edge_heavy,
        "selected_preflight_datasets": list(selected),
        "epochs": 1,
        "num_runs": 1,
        "device_type": "CUDA full graph",
        "models": list(MODELS),
        "required_jobs": len(jobs),
        "outcomes": outcomes,
        "all_passed": all_passed,
    }
    preflight_path = root / "preflight.json"
    preflight_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not all_passed:
        text = "\n".join(f"{row['dataset']}/{row['model']}: {row['status']} exit={row.get('exit_code')}" for row in outcomes if row["status"] != "complete")
        if any("out of memory" in Path(row["log"]).read_text(errors="ignore").lower() for row in outcomes if row.get("log") and row["status"] == "failed"):
            raise SystemExit("CUDA OOM during heavy-model preflight. Stop: do not launch the formal baseline benchmark.\n" + text)
        raise SystemExit("Heavy-model preflight failed. Stop: do not launch the formal baseline benchmark.\n" + text)
    print(f"All {len(jobs)} selected heavy-model preflights passed.", flush=True)


if __name__ == "__main__":
    main()
