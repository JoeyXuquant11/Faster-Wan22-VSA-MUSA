#!/usr/bin/env bash
# Rebuild frozen reference timing, Gate 2 reference records/thresholds, and Gate 3
# 6-seed trajectory delta on an 8x MTT S5000 host, then run bench.py check --final.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ASSETS="${ASSETS:-$REPO_ROOT/assets}"
export MKL_NUM_THREADS=1
export TORCH_COMPILE_DISABLE=1
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True

cd "$REPO_ROOT"
mkdir -p runs

echo "=== [1/6] Freezing input fingerprints and loader digests ==="
python3 bench/bench.py freeze-inputs

echo "=== [2/6] Freezing reference timing (2 cold-cache runs on 8x MTT S5000) ==="
python3 bench/bench.py time --tree targets/reference --freeze --out runs/repro_ref_timing

echo "=== [3/6] Freezing Gate 2 deterministic bitwise records & thresholds ==="
python3 bench/bench.py freeze-bitwise --out runs/repro_ref_bitwise

echo "=== [4/6] Freezing Gate 3 6-seed reference trajectories & delta ==="
python3 bench/bench.py freeze-traj --out runs/repro_ref_traj

echo "=== [5/6] Refreshing formulation.json input hashes ==="
python3 bench/bench.py freeze-inputs

echo "=== [6/6] Running paired A-B-A certification (bench.py check --final) ==="
touch task/final-evaluation
python3 bench/bench.py check --final --tree wan22_vsa_s5000 --out runs/repro_final_check
cat runs/repro_final_check/check.json
