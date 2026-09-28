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

NC_DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
COMPATIBILITY_DATASETS = ("Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
VARIANTS = ("v05_full", "v05_no_crst", "v05_no_relation_context", "v05_no_imci")
PROTOCOL = "unified_full_graph_nc_v1"
BRANCH = "risa_v05"
OUTPUT_ROOT = ROOT / "outputs/risa_v05_v1"


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


def phase_seeds(phase: str) -> tuple[int, ...]:
    if phase in {"smoke", "compatibility"}:
        return (42,)
    if phase == "formal":
        return SEEDS
    raise ValueError(f"unknown phase {phase!r}")


def jobs_for_phase(phase: str, datasets: tuple[str, ...] | None = None,
                   variants: tuple[str, ...] | None = None) -> list[tuple[str, str]]:
    if phase == "smoke":
        if datasets not in (None, ("Movies",)) or variants not in (None, ("v05_full",)):
            raise ValueError("smoke is fixed to Movies x v05_full")
        return [("Movies", "v05_full")]
    if phase == "compatibility":
        chosen = datasets or COMPATIBILITY_DATASETS
        if any(name not in COMPATIBILITY_DATASETS for name in chosen):
            raise ValueError(f"compatibility datasets must be in {COMPATIBILITY_DATASETS}")
        if variants not in (None, ("v05_full",)):
            raise ValueError("compatibility runs only v05_full")
        return [(name, "v05_full") for name in chosen]
    if phase == "formal":
        chosen_data = datasets or NC_DATASETS
        chosen_variants = variants or ("v05_full",)
        if any(name not in NC_DATASETS for name in chosen_data):
            raise ValueError(f"datasets must be in {NC_DATASETS}")
        if any(name not in VARIANTS for name in chosen_variants):
            raise ValueError(f"variants must be in {VARIANTS}")
        return [(name, variant) for name in chosen_data for variant in chosen_variants]
    raise ValueError(f"unknown phase {phase!r}")


def build_command(dataset: str, variant: str, phase: str, gpu: str,
                  output_dir: Path, run_dir: Path) -> list[str]:
    if dataset not in NC_DATASETS or variant not in VARIANTS:
        raise ValueError("unsupported NC dataset or v0.5 variant")
    if phase not in {"smoke", "compatibility", "formal"}:
        raise ValueError(f"unsupported phase {phase!r}")
    formal = phase == "formal"
    checkpoint = output_dir / "best.pt"
    return [
        sys.executable, "-m", "src.main", f"dataset={dataset}", "task=nc", "model=risa_v05",
        f"model.variant={variant}", "model.hidden_dim=256", "model.dropout=0.2",
        "model.max_order=3", "model.relation_dim=32", "model.edge_chunk_size=16384",
        "model.rotation_group_size=2", "model.max_rotation_angle=1.57079632679",
        "model.iamr_num_heads=4", "model.iamr_ff_mult=2", "model.node_chunk_size=32768",
        "seed=42", f"num_runs={len(phase_seeds(phase))}", "device=cuda:0",
        f"task.epochs={300 if formal else 1}", "task.lr=1e-3", "task.weight_decay=1e-4",
        "task.patience=30",
        f"task.early_stop_min_epoch={30 if formal else 1}",
        "task.early_stop_min_delta=1e-4", "task.grad_clip=1.0",
        "task.training_mode=full_graph", f"task.protocol_version={PROTOCOL}",
        "task.evaluate_test=false", f"task.save_ckpt_path={checkpoint}",
        f"task.run_metrics_path={output_dir / 'run_metrics.json'}",
        f"hydra.run.dir={run_dir}",
    ]


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


def _validate_run_metrics(path: Path, seeds: tuple[int, ...]) -> None:
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


def _run_job(dataset: str, variant: str, phase: str, gpu: str) -> dict:
    _check_branch()
    output_dir = OUTPUT_ROOT / phase / dataset / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = OUTPUT_ROOT / "hydra_runs" / phase / dataset / variant / timestamp
    command = build_command(dataset, variant, phase, gpu, output_dir, run_dir)
    log_path = output_dir / f"launch_{timestamp}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"PHASE={phase}\nDATASET={dataset}\nVARIANT={variant}\nGPU={gpu}\n")
        log.write("COMMAND=" + " ".join(command) + "\n")
        log.flush()
        peak_gpu_mib = _run_monitored(command, env, log)

    seeds = phase_seeds(phase)
    checkpoints = expected_checkpoint_paths(output_dir, len(seeds))
    payloads = [_validate_checkpoint(path, seed)
                for path, seed in zip(checkpoints, seeds, strict=True)]
    metrics_path = output_dir / "run_metrics.json"
    _validate_run_metrics(metrics_path, seeds)
    record = {
        "phase": phase, "dataset": dataset, "variant": variant, "status": "complete",
        "training_branch": _git("branch", "--show-current"),
        "training_commit": _git("rev-parse", "HEAD"),
        "protocol_version": PROTOCOL, "task": "nc", "training_mode": "full_graph",
        "test_evaluation": False, "lp_evaluation": False, "base_seed": 42,
        "run_seeds": list(seeds), "num_runs": len(seeds),
        "epochs_max": 300 if phase == "formal" else 1,
        "best_epochs": [int(item["epoch"]) for item in payloads],
        "model_trainable_params": payloads[0]["run_metadata"]["model_parameters"],
        "classifier_trainable_params": payloads[0]["run_metadata"]["classifier_parameters"],
        "checkpoint_paths": [str(path) for path in checkpoints],
        "checkpoint_audit": [{
            "path": str(path), "expected_seed": seed, "actual_seed": int(payload["seed"]),
            "protocol_version": payload["protocol_version"],
            "selection": payload["selection"],
            "test_metrics_absent": not any(
                key.startswith("test_") for key in payload.get("metrics", {})
            ),
            "finite_model_and_metrics": True,
        } for path, seed, payload in zip(checkpoints, seeds, payloads, strict=True)],
        "checkpoint_metrics": [item.get("metrics", {}) for item in payloads],
        "run_metrics_path": str(metrics_path), "run_metrics_audit": "passed",
        "peak_process_gpu_memory_mib": peak_gpu_mib,
        "peak_process_gpu_memory_status": "sampled_by_nvidia_smi" if peak_gpu_mib is not None
                                          else "unavailable",
        "checkpoint_validation": "passed", "launch_log": str(log_path),
        "command": command, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "complete.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    if phase == "smoke":
        subprocess.run([
            sys.executable, "scripts/analyze_risa_v05.py", "--dataset", dataset,
            "--checkpoint", str(checkpoints[0]), "--device", "cuda:0",
        ], cwd=ROOT, env=env, check=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Model-v0.5 NC jobs")
    parser.add_argument("--phase", choices=("smoke", "compatibility", "formal"), default="smoke")
    parser.add_argument("--datasets", nargs="+", choices=NC_DATASETS, default=None)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=None)
    parser.add_argument("--gpu", default=os.environ.get("GPU_ID", "0"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    _check_branch()
    try:
        jobs = jobs_for_phase(
            args.phase,
            tuple(args.datasets) if args.datasets is not None else None,
            tuple(args.variants) if args.variants is not None else None,
        )
    except ValueError as exc:
        parser.error(str(exc))
    seeds = phase_seeds(args.phase)
    if args.dry_run:
        formal = args.phase == "formal"
        print(json.dumps({
            "phase": args.phase, "jobs": [{
                "dataset": dataset, "variant": variant, "seeds": list(seeds),
                "epochs": 300 if formal else 1,
                "checkpoint_paths": [str(path) for path in expected_checkpoint_paths(
                    OUTPUT_ROOT / args.phase / dataset / variant, len(seeds)
                )],
            } for dataset, variant in jobs],
            "job_count": len(jobs), "runs_total": len(jobs) * len(seeds),
            "datasets": list(NC_DATASETS), "test_evaluation": False, "lp_evaluation": False,
        }, indent=2))
        return
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for dataset, variant in jobs:
        result = _run_job(dataset, variant, args.phase, args.gpu)
        results.append(result)
        print(json.dumps({
            "dataset": dataset, "variant": variant, "status": result["status"],
            "run_seeds": result["run_seeds"], "checkpoint_paths": result["checkpoint_paths"],
            "checkpoint_metrics": result["checkpoint_metrics"],
            "peak_process_gpu_memory_mib": result["peak_process_gpu_memory_mib"],
        }), flush=True)
    print(json.dumps({
        "phase": args.phase, "job_count": len(jobs), "runs_total": len(jobs) * len(seeds),
        "completed": len(results), "test_evaluation": False, "lp_evaluation": False,
    }, indent=2))
    if args.phase == "compatibility":
        print("Compatibility phase complete; no formal 300-epoch jobs were launched.")


if __name__ == "__main__":
    main()

