#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
conda run --no-capture-output -n yhf_env python scripts/run_mob_factorial_nc.py "$@"
