from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCREEN_DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
SMOKE_VARIANTS = ("v06_full", "v06_no_relation_condition")
FORMAL_VARIANT = "v06_full"
PROTOCOL = "unified_full_graph_nc_v1"
BRANCH = "risa_v06"
OUTPUT_ROOT = ROOT / "outputs/risa_v06_v1"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _check_branch() -> None:
    branch = _git("branch", "--show-current")
    if branch != BRANCH:
        raise RuntimeError(f"runner requires branch {BRANCH}; current branch is {branch}")


def expected_checkpoint_paths(output_dir: Path, num_runs: int) -> list[Path]:
    if num_runs < 1:
        raise ValueError("num_runs must be positive")
    if num_runs == 1:
        return [output_dir / "best.pt"]
    return [output_dir / f"best_run{i}.pt" for i in range(1, num_runs + 1)]


def build_command(dataset: str, variant: str, gpu: str,
                  output_dir: Path, run_dir: Path,
                  phase: str = "smoke") -> list[str]:
    if dataset not in SCREEN_DATASETS:
        raise ValueError(f"unsupported NC dataset {dataset!r}")
    if phase not in {"smoke", "formal"}:
        raise ValueError(f"unsupported training phase {phase!r}")
    allowed_variants = SMOKE_VARIANTS if phase == "smoke" else (FORMAL_VARIANT,)
    if variant not in allowed_variants:
        raise ValueError(f"{phase} supports variants {allowed_variants}, got {variant!r}")
    formal = phase == "formal"
    num_runs = len(SEEDS) if formal else 1
    return [
        sys.executable, "-m", "src.main", f"dataset={dataset}", "task=nc", "model=risa_v06",
        f"model.variant={variant}", "model.hidden_dim=256", "model.dropout=0.2",
        "model.max_order=3", "model.relation_dim=32", "model.evidence_dim=32",
        "model.edge_chunk_size=16384", "model.node_chunk_size=32768",
        "model.num_heads=4", "model.ff_mult=2",
        "seed=42", f"num_runs={num_runs}", "device=cuda:0",
        f"task.epochs={300 if formal else 1}", "task.lr=1e-3", "task.weight_decay=1e-4",
        "task.patience=30", f"task.early_stop_min_epoch={30 if formal else 1}",
        "task.early_stop_min_delta=1e-4", "task.grad_clip=1.0",
        "task.training_mode=full_graph", f"task.protocol_version={PROTOCOL}",
        "task.evaluate_test=false", f"task.save_ckpt_path={output_dir / 'best.pt'}",
        f"task.run_metrics_path={output_dir / 'run_metrics.json'}",
        f"hydra.run.dir={run_dir}",
    ]


def formal_dry_run() -> dict:
    jobs = []
    for dataset in SCREEN_DATASETS:
        output_dir = OUTPUT_ROOT / "formal" / dataset / FORMAL_VARIANT
        run_dir = OUTPUT_ROOT / "hydra_runs" / "formal" / dataset / FORMAL_VARIANT / "dry-run"
        command = build_command(dataset, FORMAL_VARIANT, "0", output_dir, run_dir, phase="formal")
        jobs.append({
            "dataset": dataset, "variant": FORMAL_VARIANT, "seeds": list(SEEDS),
            "epochs": 300, "num_runs": len(SEEDS),
            "checkpoint_paths": [str(path) for path in
                                 expected_checkpoint_paths(output_dir, len(SEEDS))],
            "command": command,
        })
    return {
        "phase": "formal", "jobs": jobs,
        "job_count": len(jobs), "runs_total": len(jobs) * len(SEEDS),
        "datasets": list(SCREEN_DATASETS), "seeds": list(SEEDS),
        "variant": FORMAL_VARIANT, "test_evaluation": False, "lp_evaluation": False,
        "launched": False,
    }


def _validate_checkpoint(path: Path, expected_seed: int) -> dict:
    import math
    import torch

    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("task") != "nc" or payload.get("protocol_version") != PROTOCOL:
        raise RuntimeError(f"wrong task or protocol in {path}")
    if int(payload.get("seed", -1)) != expected_seed:
        raise RuntimeError(f"seed mismatch in {path}: expected {expected_seed}")
    if payload.get("selection") != "best_val_accuracy":
        raise RuntimeError(f"checkpoint was not selected by Val Accuracy: {path}")
    metrics = payload.get("metrics", {})
    if not {"val_acc", "val_macro_f1"}.issubset(metrics):
        raise RuntimeError(f"validation metrics missing in {path}")
    if any(key.startswith("test_") for key in metrics):
        raise RuntimeError(f"test metric found in {path}")
    if any(not math.isfinite(float(metrics[key])) for key in ("val_acc", "val_macro_f1")):
        raise RuntimeError(f"non-finite validation metric in {path}")
    for section in ("model_state", "head_state"):
        for name, tensor in payload.get(section, {}).items():
            if torch.is_tensor(tensor) and not torch.isfinite(tensor).all():
                raise RuntimeError(f"NaN/Inf in {path}:{section}.{name}")
    return payload


def _validate_run_metrics(path: Path, seeds: tuple[int, ...]) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if metrics.get("protocol_version") != PROTOCOL or metrics.get("run_seeds") != list(seeds):
        raise RuntimeError(f"run_metrics protocol/seeds mismatch in {path}")
    rows = metrics.get("runs", [])
    if [int(row.get("seed", -1)) for row in rows] != list(seeds):
        raise RuntimeError(f"run_metrics rows do not match requested seeds {seeds}")
    if any(key.startswith("test_") for row in rows for key in row.get("metrics", {})):
        raise RuntimeError(f"test metrics found in {path}")
    return metrics


def _gpu_memory_for_pid(pid: int) -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode:
        return None
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        try:
            if len(parts) >= 2 and int(parts[0]) == pid:
                return int(parts[1])
        except ValueError:
            continue
    return None


def _run_monitored(command: list[str], env: dict[str, str], log) -> int | None:
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    peak = None
    while proc.poll() is None:
        current = _gpu_memory_for_pid(proc.pid)
        if current is not None:
            peak = current if peak is None else max(peak, current)
        time.sleep(0.25)
    status = proc.wait()
    if status:
        raise subprocess.CalledProcessError(status, command)
    return peak


def _training_environment(gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def _run_smoke_job(dataset: str, variant: str, gpu: str) -> dict:
    _check_branch()
    output_dir = OUTPUT_ROOT / "smoke" / dataset / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = OUTPUT_ROOT / "hydra_runs" / "smoke" / dataset / variant / timestamp
    command = build_command(dataset, variant, gpu, output_dir, run_dir, phase="smoke")
    log_path = output_dir / f"launch_{timestamp}.log"
    env = _training_environment(gpu)
    checkpoint = output_dir / "best.pt"
    metrics_path = output_dir / "run_metrics.json"
    existing_record_path = output_dir / "complete.json"
    reuse_checkpoint = checkpoint.is_file() and metrics_path.is_file() and existing_record_path.is_file()
    if reuse_checkpoint:
        previous = json.loads(existing_record_path.read_text(encoding="utf-8"))
        if previous.get("dataset") != dataset or previous.get("variant") != variant:
            raise RuntimeError(f"existing smoke record does not match {dataset}/{variant}")
        payload = _validate_checkpoint(checkpoint, expected_seed=42)
        peak_gpu_mib = previous.get("peak_process_gpu_memory_mib")
        log_path = Path(previous.get("launch_log", log_path))
    else:
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"PHASE=smoke\nDATASET={dataset}\nVARIANT={variant}\nGPU={gpu}\n")
            log.write("COMMAND=" + " ".join(command) + "\n")
            log.flush()
            peak_gpu_mib = _run_monitored(command, env, log)
        payload = _validate_checkpoint(checkpoint, expected_seed=42)
    _validate_run_metrics(metrics_path, (42,))
    record = {
        "phase": "smoke", "dataset": dataset, "variant": variant, "status": "complete",
        "training_branch": _git("branch", "--show-current"),
        "training_commit": _git("rev-parse", "HEAD"),
        "protocol_version": PROTOCOL, "task": "nc", "training_mode": "full_graph",
        "test_evaluation": False, "lp_evaluation": False, "seed": 42, "epochs_max": 1,
        "best_epoch": int(payload["epoch"]),
        "model_trainable_params": payload["run_metadata"]["model_parameters"],
        "classifier_trainable_params": payload["run_metadata"]["classifier_parameters"],
        "checkpoint_path": str(checkpoint), "checkpoint_selection": payload["selection"],
        "checkpoint_metrics": payload.get("metrics", {}), "run_metrics_path": str(metrics_path),
        "peak_process_gpu_memory_mib": peak_gpu_mib,
        "peak_process_gpu_memory_status": "sampled_by_nvidia_smi" if peak_gpu_mib is not None
                                          else "unavailable",
        "launch_log": str(log_path), "command": command,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "complete.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    subprocess.run([
        sys.executable, "scripts/analyze_risa_v06.py", "--dataset", dataset,
        "--variant", variant, "--checkpoint", str(checkpoint), "--device", "cuda:0",
    ], cwd=ROOT, env=env, check=True)
    return record


def _run_formal_dataset(dataset: str, gpu: str) -> dict:
    _check_branch()
    variant = FORMAL_VARIANT
    seeds = SEEDS
    output_dir = OUTPUT_ROOT / "formal" / dataset / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = OUTPUT_ROOT / "hydra_runs" / "formal" / dataset / variant / timestamp
    command = build_command(dataset, variant, gpu, output_dir, run_dir, phase="formal")
    log_path = output_dir / f"launch_{timestamp}.log"
    env = _training_environment(gpu)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"PHASE=formal\nDATASET={dataset}\nVARIANT={variant}\n")
        log.write(f"SEEDS={','.join(map(str, seeds))}\nGPU={gpu}\n")
        log.write("COMMAND=" + " ".join(command) + "\n")
        log.flush()
        peak_gpu_mib = _run_monitored(command, env, log)

    checkpoints = expected_checkpoint_paths(output_dir, len(seeds))
    payloads = [_validate_checkpoint(path, seed)
                for path, seed in zip(checkpoints, seeds, strict=True)]
    metrics_path = output_dir / "run_metrics.json"
    _validate_run_metrics(metrics_path, seeds)
    record = {
        "phase": "formal", "dataset": dataset, "variant": variant, "status": "complete",
        "training_branch": _git("branch", "--show-current"),
        "training_commit": _git("rev-parse", "HEAD"),
        "protocol_version": PROTOCOL, "task": "nc", "training_mode": "full_graph",
        "test_evaluation": False, "lp_evaluation": False, "base_seed": 42,
        "run_seeds": list(seeds), "num_runs": len(seeds), "epochs_max": 300,
        "best_epochs": [int(payload["epoch"]) for payload in payloads],
        "model_trainable_params": payloads[0]["run_metadata"]["model_parameters"],
        "classifier_trainable_params": payloads[0]["run_metadata"]["classifier_parameters"],
        "checkpoint_paths": [str(path) for path in checkpoints],
        "checkpoint_audit": [{
            "path": str(path), "expected_seed": seed, "actual_seed": int(payload["seed"]),
            "protocol_version": payload["protocol_version"], "selection": payload["selection"],
            "test_metrics_absent": not any(
                key.startswith("test_") for key in payload.get("metrics", {})
            ),
            "finite_model_and_metrics": True,
            "metrics": payload.get("metrics", {}),
        } for path, seed, payload in zip(checkpoints, seeds, payloads, strict=True)],
        "checkpoint_metrics": [payload.get("metrics", {}) for payload in payloads],
        "run_metrics_path": str(metrics_path), "run_metrics_audit": "passed",
        "peak_process_gpu_memory_mib": peak_gpu_mib,
        "peak_process_gpu_memory_status": "sampled_by_nvidia_smi" if peak_gpu_mib is not None
                                          else "unavailable",
        "launch_log": str(log_path), "command": command,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "complete.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def _print_smoke_summary(records: list[dict]) -> None:
    for record in records:
        print(json.dumps({
            "dataset": record["dataset"], "variant": record["variant"],
            "status": record["status"], "checkpoint_metrics": record["checkpoint_metrics"],
            "peak_process_gpu_memory_mib": record["peak_process_gpu_memory_mib"],
        }), flush=True)
    print(json.dumps({
        "phase": "smoke", "completed": len(records), "runs_total": len(records),
        "test_evaluation": False, "lp_evaluation": False,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RISA v0.6 smoke or formal NC jobs")
    parser.add_argument("--phase", choices=("smoke", "formal", "formal-dry-run"), default="smoke")
    parser.add_argument("--gpu", default=os.environ.get("GPU_ID", "0"))
    parser.add_argument("--variants", nargs="+", choices=SMOKE_VARIANTS, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    _check_branch()

    if args.phase in {"formal", "formal-dry-run"}:
        if args.variants is not None:
            parser.error("formal phase is fixed to v06_full")
        if args.phase == "formal-dry-run" and not args.dry_run:
            parser.error("formal-dry-run is planning-only; pass --dry-run")
        if args.dry_run or args.phase == "formal-dry-run":
            print(json.dumps(formal_dry_run(), indent=2))
            return
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        records = []
        for dataset in SCREEN_DATASETS:
            record = _run_formal_dataset(dataset, args.gpu)
            records.append(record)
            print(json.dumps({
                "dataset": dataset, "variant": FORMAL_VARIANT, "status": record["status"],
                "run_seeds": record["run_seeds"],
                "checkpoint_metrics": record["checkpoint_metrics"],
                "peak_process_gpu_memory_mib": record["peak_process_gpu_memory_mib"],
            }), flush=True)
        print(json.dumps({
            "phase": "formal", "datasets_completed": len(records),
            "runs_completed": sum(record["num_runs"] for record in records),
            "runs_total": len(SCREEN_DATASETS) * len(SEEDS),
            "test_evaluation": False, "lp_evaluation": False,
        }, indent=2))
        return

    variants = tuple(args.variants) if args.variants is not None else SMOKE_VARIANTS
    jobs = [("Movies", variant) for variant in variants]
    if args.dry_run:
        print(json.dumps({
            "phase": "smoke", "jobs": [{"dataset": dataset, "variant": variant,
                                          "seed": 42, "epochs": 1}
                                         for dataset, variant in jobs],
            "runs_total": len(jobs), "test_evaluation": False, "lp_evaluation": False,
            "launched": False,
        }, indent=2))
        return
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset, variant in jobs:
        record = _run_smoke_job(dataset, variant, args.gpu)
        records.append(record)
        print(json.dumps({
            "dataset": dataset, "variant": variant, "status": record["status"],
            "checkpoint_metrics": record["checkpoint_metrics"],
            "peak_process_gpu_memory_mib": record["peak_process_gpu_memory_mib"],
        }), flush=True)
    print(json.dumps({
        "phase": "smoke", "completed": len(records), "runs_total": len(jobs),
        "test_evaluation": False, "lp_evaluation": False,
    }, indent=2))


if __name__ == "__main__":
    main()
