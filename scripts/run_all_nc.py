from __future__ import annotations

import subprocess
import sys


DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
MODELS = ["mlp", "gcn", "sage", "mmgcn", "mgat", "dip", "dgf", "dmgc", "lgmrec"]


def main() -> None:
    for dataset in DATASETS:
        for model in MODELS:
            cmd = [sys.executable, "-m", "src.main", f"dataset={dataset}", "task=nc", f"model={model}"]
            print(" ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
