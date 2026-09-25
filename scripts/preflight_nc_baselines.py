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


DATASET = "ele-fashion"
MODELS = ("dip", "mmgcn", "mgat", "dgf", "dmgc", "lgmrec")
OUTPUT_ROOT = Path("outputs/nc_preflight_v1")
FACTORIAL_ROOT = Path("outputs/mob_factorial_nc_v1")
FACTORIAL_DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
FACTORIAL_VARIANTS = (
    "mob_terminal_plain", "mob_uniform_plain", "mob_gpr_plain",
    "mob_terminal_residual", "mob_uniform_residual", "mob_gpr_residual",
)


def _assert_factorial_complete() -> None:
    missing = [
        str(FACTORIAL_ROOT / dataset / variant / "complete.json")
        for dataset in FACTORIAL_DATASETS
        for variant in FACTORIAL_VARIANTS
        if not (FACTORIAL_ROOT / dataset / variant / "complete.json").is_file()
    ]
    if missing:
        raise RuntimeError("Complete the 30-job factorial before heavy-model preflight; missing:\n" + "\n".join(missing))


def _job(model: str, gpu: str, root: Path) -> dict:
    output = root / DATASET / model
    output.mkdir(parents=True, exist_ok=True)
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output / f"run_{attempt}"
    log_path = output / f"preflight_{attempt}.log"
    command = [
        sys.executable, "-m", "src.main",
        f"dataset={DATASET}", "task=nc", f"model={model}",
        "seed=42", "num_runs=1", "task.epochs=1", "device=cuda:0",
        f"hydra.run.dir={run_dir}",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(command) + "\nPHYSICAL_GPU: " + gpu + "\n")
        log.flush()
        result = subprocess.run(command, cwd=Path.cwd(), env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - start
    status = "complete" if result.returncode == 0 and (run_dir / "results.json").is_file() else "failed"
    return {
        "dataset": DATASET, "model": model, "status": status, "exit_code": result.returncode,
        "gpu": gpu, "seconds": elapsed, "log": str(log_path), "command": command,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="One-epoch full-graph CUDA preflight on the largest NC graph.")
    parser.add_argument("--gpus", default=os.environ.get("GPU_IDS", "0,1"))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpus = tuple(part.strip() for part in args.gpus.split(",") if part.strip())
    if not gpus:
        parser.error("at least one GPU ID is required")
    if args.dry_run:
        for index, model in enumerate(MODELS):
            print(DATASET, model, "epochs=1", "num_runs=1", "GPU", gpus[index % len(gpus)])
        return

    _assert_factorial_complete()
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    outcomes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(_job, model, gpus[index % len(gpus)], root) for index, model in enumerate(MODELS)]
        for future in concurrent.futures.as_completed(futures):
            outcome = future.result()
            outcomes.append(outcome)
            print(json.dumps(outcome), flush=True)
    outcomes.sort(key=lambda row: MODELS.index(row["model"]))
    payload = {
        "dataset": DATASET,
        "dataset_nodes": 97766,
        "epochs": 1,
        "num_runs": 1,
        "device_type": "CUDA full graph",
        "models": outcomes,
        "all_passed": all(row["status"] == "complete" for row in outcomes),
    }
    (root / "preflight.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not payload["all_passed"]:
        failed = [row for row in outcomes if row["status"] != "complete"]
        if any("out of memory" in Path(row["log"]).read_text(errors="ignore").lower() for row in failed):
            raise SystemExit("CUDA OOM during heavy-model preflight. Stop: do not launch the formal baseline benchmark.")
        raise SystemExit("Heavy-model preflight failed. Stop: do not launch the formal baseline benchmark.")
    print("All six heavy-model preflights passed.", flush=True)


if __name__ == "__main__":
    main()
