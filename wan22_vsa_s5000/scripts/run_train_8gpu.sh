#!/usr/bin/env bash
# Standard 8x MTT S5000 launch script for Wan2.2-TI2V-5B + VSA full finetuning.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-mthreads/musa-torch2.9-s5000:latest}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/../assets/models/Wan2.2-TI2V-5B}"
DATA_DIR="${DATA_DIR:-$REPO_DIR/../assets/data/wan22_vsa_data_704x1280_121f}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/wan22_vsa_train_out}"
HOSTNAME_SHORT="$(hostname -s)"

mkdir -p "/tmp/tl_cache_${HOSTNAME_SHORT}_wan22vsa"

docker run --rm \
  --device mthreads.com/gpu=gpu0 --device mthreads.com/gpu=gpu1 \
  --device mthreads.com/gpu=gpu2 --device mthreads.com/gpu=gpu3 \
  --device mthreads.com/gpu=gpu4 --device mthreads.com/gpu=gpu5 \
  --device mthreads.com/gpu=gpu6 --device mthreads.com/gpu=gpu7 \
  --ipc=host --shm-size=64g --ulimit memlock=-1 \
  -v "$REPO_DIR/..:$REPO_DIR/.." \
  -v "/tmp/tl_cache_${HOSTNAME_SHORT}_wan22vsa:/root/.tilelang/cache" \
  -w "$REPO_DIR" \
  -e MKL_NUM_THREADS=1 \
  -e OMP_NUM_THREADS=4 \
  -e TORCH_COMPILE_DISABLE=1 \
  -e PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  "$IMAGE" \
  torchrun --standalone --nproc_per_node=8 train.py \
    --model_dir "$MODEL_DIR" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_latent_t 31 \
    --latent_height 44 \
    --latent_width 80 \
    --batch_size 1 \
    --gradient_accumulation_steps 1 \
    --vsa_sparsity 0.8 \
    --learning_rate 1e-5 \
    --flow_shift 5.0 \
    --max_train_steps 100 \
    "$@"
