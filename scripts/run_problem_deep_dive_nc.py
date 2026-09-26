from __future__ import annotations

import argparse
import hashlib
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
SEEDS = (42, 43, 44)
PROTOCOL = "unified_full_graph_nc_v1"
OUTPUT_ROOT = Path("outputs/problem_deep_dive_v1")
REWIRED_ROOT = OUTPUT_ROOT / "rewired_graphs"


def _jobs(phase: str, datasets: tuple[str, ...]):
    if phase == "preflight":
        for model in ("operator_control_gcn", "operator_control_sage"):
            for readout in ("deep_only", "anchored25"):
                yield (phase, "ele-fashion", model, readout)
        yield (phase, "ele-fashion", "multi_order_bank", "uniform")
    elif phase == "operator":
        for dataset in datasets:
            for model in ("operator_control_gcn", "operator_control_sage"):
                for readout in ("deep_only", "anchored25"):
                    yield (phase, dataset, model, readout)
    elif phase == "rewired":
        for dataset in datasets:
            yield (phase, dataset, "multi_order_bank", "uniform")
    else:
        raise ValueError(phase)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_errors(paths: list[Path], seeds: tuple[int, ...], require_test: bool) -> list[str]:
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
            errors.append(f"missing validation checkpoint metadata in {path}")
        required = {"val_acc", "val_macro_f1"}
        if require_test:
            required |= {"test_acc", "test_macro_f1"}
        if not required.issubset(payload.get("metrics", {})):
            errors.append(f"missing metrics in {path}: {sorted(required-set(payload.get('metrics',{})))}")
    return errors


def _run_job(job, gpu: str, root: Path):
    phase, dataset, model, readout = job
    group = {"preflight": "preflight", "operator": "operator_transfer", "rewired": "rewired_topology"}[phase]
    variant = f"{model}_{readout}"
    output_dir = root / group / dataset / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    complete = output_dir / "complete.json"
    is_preflight = phase == "preflight"
    num_runs, epochs = (1, 1) if is_preflight else (3, 300)
    seeds = (42,) if is_preflight else SEEDS
    checkpoint_paths = [output_dir / ("best.pt" if is_preflight else f"best_run{i}.pt") for i in range(1, num_runs + 1)]
    override = None
    if phase == "rewired" or (phase == "preflight" and model == "multi_order_bank"):
        graph_path = REWIRED_ROOT / f"{dataset}.pt"
        if not graph_path.is_file():
            raise FileNotFoundError(f"rewired physical graph missing: {graph_path}; run scripts/build_degree_preserving_rewires.py first")
        override = str(graph_path.resolve())
    if complete.is_file() and not is_preflight:
        try:
            saved = json.loads(complete.read_text())
            if saved.get("phase") == phase and saved.get("run_seeds") == list(seeds) and not _checkpoint_errors(checkpoint_paths, seeds, True):
                return {"job": job, "status": "already_complete", "gpu": gpu}
        except Exception:
            pass

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / f"run_{stamp}"
    base_ckpt = output_dir / ("best.pt" if is_preflight else "best.pt")
    command = [sys.executable, "-m", "src.main", f"dataset={dataset}", "task=nc",
               f"model={model}", f"model.readout={readout}", "seed=42",
               f"num_runs={num_runs}", "device=cuda:0", f"task.epochs={epochs}",
               f"task.save_ckpt_path={base_ckpt}",
               f"task.run_metrics_path={output_dir / 'run_metrics.json'}",
               f"task.evaluate_test={'false' if is_preflight else 'true'}",
               f"hydra.run.dir={run_dir}"]
    if override:
        command.append(f"task.edge_index_override_path={override}")
    if is_preflight:
        command.extend(["task.patience=1", "task.early_stop_min_epoch=1"])
    log_path = output_dir / f"launch_{stamp}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"PHYSICAL_GPU={gpu}\nCOMMAND={' '.join(command)}\n")
        log.flush()
        proc = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - started
    if proc.returncode:
        raise RuntimeError(f"{job} failed on GPU {gpu}; see {log_path}")
    errors = _checkpoint_errors(checkpoint_paths, seeds, not is_preflight)
    if errors:
        raise RuntimeError("checkpoint validation: " + "; ".join(errors))
    if not is_preflight:
        metrics = json.loads((output_dir / "run_metrics.json").read_text())
        if metrics.get("run_seeds") != list(SEEDS) or len(metrics.get("runs", [])) != 3:
            raise RuntimeError(f"invalid internal seed aggregation for {job}")
    record = {"phase": phase, "dataset": dataset, "model": model, "readout": readout,
              "protocol_version": PROTOCOL, "seed": 42, "num_runs": num_runs,
              "run_seeds": list(seeds), "epochs": epochs, "gpu": gpu, "seconds": elapsed,
              "command": command, "checkpoints": [str(p) for p in checkpoint_paths],
              "checkpoint_sha256": {str(p): _sha256(p) for p in checkpoint_paths},
              "run_metrics_sha256": _sha256(output_dir / "run_metrics.json") if not is_preflight else None,
              "checkpoint_validation": "passed", "edge_index_override_path": override}
    complete.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return {"job": job, "status": "complete", "gpu": gpu, "seconds": elapsed}


def main():
    parser = argparse.ArgumentParser(description="Preflight and train D3 NC operator and rewire controls.")
    parser.add_argument("phase", choices=("preflight", "operator", "rewired"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--gpus", default=os.environ.get("GPU_IDS", "0,1"))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpus = tuple(x.strip() for x in args.gpus.split(",") if x.strip())
    if not gpus:
        parser.error("at least one GPU ID is required")
    jobs = list(_jobs(args.phase, tuple(args.datasets)))
    if args.dry_run:
        for job in jobs:
            print(job)
        print(f"jobs={len(jobs)} runs={len(jobs) * (1 if args.phase == 'preflight' else 3)}")
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    def report(outcome: GPUJobOutcome[Any, Any]):
        print(json.dumps(outcome.result) if outcome.error is None else f"FAILED {outcome.job}: {outcome.error}", flush=True)
    summary = run_gpu_jobs(jobs, gpus, lambda job, gpu: _run_job(job, gpu, args.output_root), report)
    if summary.failures:
        detail = "\n".join(f"{x.job} gpu={x.gpu}: {x.error}" for x in summary.failures)
        if summary.not_started:
            detail += "\nnot started: " + ", ".join(map(str, summary.not_started))
        raise SystemExit("D3 training failed:\n" + detail)
    print(f"Completed phase={args.phase} jobs={len(jobs)} runs={len(jobs)*(1 if args.phase == 'preflight' else 3)}", flush=True)


if __name__ == "__main__":
    main()
