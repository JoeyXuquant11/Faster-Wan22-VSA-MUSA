# Wan2.2-TI2V-5B + VSA Full Finetuning on MTT S5000

Full-parameter finetuning port of FastVideo Video Sparse Attention (VSA) on
official `Wan2.2-TI2V-5B` (`Wan-AI/Wan2.2-TI2V-5B`) for
8x Moore Threads MTT S5000 GPUs.

## Architecture & Recipe
- **Model**: Official `Wan2.2-TI2V-5B` DiT (30 blocks, `dim=3072`, `ffn_dim=14336`,
  `num_heads=24`, `head_dim=128`, `in_dim=48`, `out_dim=48`, `patch_size=(1,2,2)`),
  extended with FastVideo VSA compression gates (`to_gate_compress`, zero-initialized
  from the dense checkpoint; ~5.28B trainable parameters).
- **Attention**: Differentiable two-stage Video Sparse Attention with `(4, 4, 4)`
  64-token 3D tiles, FP32 block-mean coarse branch, top-k tile routing,
  block-sparse fine attention, and gated coarse-fine combination.
- **Distributed & Precision**: PyTorch FSDP2 (`fully_shard`) over MCCL across 8x S5000
  with BF16 compute, FP32 master weights, FP32 gradient reduction, FP32 AdamW states,
  and full per-block activation checkpointing.
- **Objective**: Rectified Flow Matching velocity MSE loss with logit-normal
  timestep sampling (`flow_shift=5.0`).

## Usage
1. Prepare pre-encoded dataset (using real Wan2.2 VAE + UMT5-XXL):
   ```bash
   python3 scripts/prepare_dataset.py \
     --model_dir ./assets/models/Wan2.2-TI2V-5B \
     --out_dir ./assets/data/wan22_vsa_data_704x1280_121f \
     --num_samples 32 --num_frames 121 --height 704 --width 1280
   ```
2. Run 8-GPU full finetuning:
   ```bash
   bash scripts/run_train_8gpu.sh
   ```
