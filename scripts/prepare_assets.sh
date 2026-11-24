#!/usr/bin/env bash
# Rebuild public Wan2.2-TI2V-5B weights, deterministic 704x1280x121f pre-encoded
# dataset, and targets/reference baseline tree so bench/bench.py check --final
# can be reproduced from scratch on any 8x MTT S5000 host.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ASSETS="${ASSETS:-$REPO_ROOT/assets}"
MODEL_DIR="$ASSETS/models/Wan2.2-TI2V-5B"
DATA_DIR="$ASSETS/data/wan22_vsa_data_704x1280_121f"
USE_MODELSCOPE="${USE_MODELSCOPE:-0}"

mkdir -p "$MODEL_DIR" "$DATA_DIR"

echo "=== [1/3] Downloading public Wan-AI/Wan2.2-TI2V-5B weights into $MODEL_DIR ==="
if [[ ! -f "$MODEL_DIR/Wan2.2_VAE.pth" || ! -f "$MODEL_DIR/models_t5_umt5-xxl-enc-bf16.pth" ]]; then
  if [[ "$USE_MODELSCOPE" == "1" ]]; then
    python3 -c "from modelscope import snapshot_download; snapshot_download('Wan-AI/Wan2.2-TI2V-5B', local_dir='$MODEL_DIR')"
  else
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Wan-AI/Wan2.2-TI2V-5B', local_dir='$MODEL_DIR')"
  fi
else
  echo "  Weights already present in $MODEL_DIR, skipping download."
fi

echo "=== [2/3] Generating 32-sample 704x1280x121f dataset via real Wan2.2 VAE + UMT5-XXL ==="
if [[ ! -f "$DATA_DIR/manifest.json" ]]; then
  # Requires DiffSynth-Studio on PYTHONPATH or DIFFSYNTH_PATH (https://github.com/modelscope/DiffSynth-Studio)
  MKL_NUM_THREADS=1 TORCH_COMPILE_DISABLE=1 PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True \
    python3 "$REPO_ROOT/wan22_vsa_s5000/scripts/prepare_dataset.py" \
      --model_dir "$MODEL_DIR" \
      --out_dir "$DATA_DIR" \
      --num_samples 32 \
      --num_frames 121 \
      --height 704 \
      --width 1280
else
  echo "  Dataset manifest already present in $DATA_DIR, skipping generation."
fi

echo "=== [3/3] Materializing targets/reference from the baseline branch ==="
if [[ ! -d "$REPO_ROOT/targets/reference" ]]; then
  mkdir -p "$REPO_ROOT/targets"
  if git -C "$REPO_ROOT" rev-parse --verify origin/baseline >/dev/null 2>&1; then
    git -C "$REPO_ROOT" archive origin/baseline wan22_vsa_s5000 | tar -x -C "$REPO_ROOT/targets"
    mv "$REPO_ROOT/targets/wan22_vsa_s5000" "$REPO_ROOT/targets/reference"
  elif git -C "$REPO_ROOT" rev-parse --verify baseline >/dev/null 2>&1; then
    git -C "$REPO_ROOT" archive baseline wan22_vsa_s5000 | tar -x -C "$REPO_ROOT/targets"
    mv "$REPO_ROOT/targets/wan22_vsa_s5000" "$REPO_ROOT/targets/reference"
  else
    cp -a "$REPO_ROOT/wan22_vsa_s5000" "$REPO_ROOT/targets/reference"
    git -C "$REPO_ROOT/targets/reference" apply -R "$REPO_ROOT/patches/wan22_vsa_r1_to_r12.patch"
  fi
else
  echo "  targets/reference already exists."
fi

echo "Done. Next run: ASSETS=$ASSETS bash scripts/freeze_judge_and_verify.sh"
