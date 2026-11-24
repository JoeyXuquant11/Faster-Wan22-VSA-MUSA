# Baseline Report: Wan2.2-TI2V-5B + VSA Full Finetuning (8x MTT S5000)

*Archived note: internal microarchitecture parameters, measured hardware peak specs, and proprietary vendor kernel symbol names have been redacted; step timings, category shares, memory, and numerics figures are unchanged.*

## 1. Environment & Workload
- **Node / Hardware**: 8x Moore Threads MTT S5000 (80 GiB HBM per GPU).
- **Runtime Image**: MUSA 5.2.0 runtime container (`torch 2.9.1.post1+musa5.2.0`, `torch_musa 2.9.1.post1+musa5.2.0`, `tilelang 0.1.8+musa.3`, `mate 0.2.5`, `mcc` compiler).
- **Reference Commit**: `6e832aaf0dd2ef8edc5400d6448529f166575f98` (`targets/reference`).
- **Model**: Official `Wan2.2-TI2V-5B` (`$ASSETS/models/Wan2.2-TI2V-5B`, 30 blocks, `dim=3072`, `ffn_dim=14336`, `num_heads=24`, `head_dim=128`, `in_dim=48`, `out_dim=48`, `patch_size=(1,2,2)`) + 30 zero-initialized VSA compression gates (`to_gate_compress`), total **`5,282,995,392`** trainable parameters.
- **Workload Recipe**:
  - Latent shape: `[1, 48, 31, 44, 80]` (`704x1280`, 121 frames -> `dit_seq_shape = (31, 22, 40)`, `S = 27,280` tokens; 480 `(4,4,4)` VSA tiles, `S_pad = 30,720`; `vsa_sparsity = 0.80` -> `topk = 96` KV tiles per Q tile).
  - Global batch = 8 (`batch_size=1`, `gradient_accumulation_steps=1`, 8 GPUs), Rectified Flow Matching velocity MSE loss (`flow_shift=5.0`), AdamW (`lr=1e-5`, `betas=(0.9, 0.999)`, `weight_decay=0.01`, `max_grad_norm=1.0`).
  - Parallelism & Precision: PyTorch FSDP2 (`fully_shard` per block + root) with BF16 compute (`param_dtype=bf16`, `cast_forward_inputs=False`), FP32 master weights, FP32 gradient reduction, FP32 AdamW optimizer states, and full per-block activation checkpointing.

## 2. Frozen Reference Timing (Uninstrumented Steady State)
- **Frozen Reference (`task/timing_ref.json`)**: 2 independent cold-cache runs (warmup=2, window=4, repeats=2 = 8 timed steps per run):
  - Run 0: `25.2884 s/iter` (windows: `101.145 s`, `101.162 s`)
  - Run 1: `25.2890 s/iter` (windows: `101.147 s`, `101.165 s`)
  - **Mean**: **`25.2887 s/iter`** (`reference_spread = 0.002%`)
  - **Peak Allocated HBM**: **`56.19 GiB / 80 GiB`**
- **Effective Model FLOPs (per GPU, 3x fwd without GC)**: `~966 TFLOPs/step` (`~757 TFLOPs` Linear/FFN + `~209 TFLOPs` VSA 80% sparse attention).

## 3. Baseline Profile & Kernel Utilization (`baseline_profile/kernels.md`)
- Profiled window: `25,277.8 ms` (1 steady-state step), GPU busy `25,218.6 ms` (`99.8%`), `34,054` kernel launches, `144` kernel classes.
- **Category Breakdown**:

| Category | Time (ms) | Share |
|---|---:|---:|
| `memcpy/cast` (incl. index gather/scatter-add, concat, casts, fills) | 14,920.6 | 59.2% |
| `attention` (masked FA fwd/bwd + cross-attn) | 4,573.6 | 18.1% |
| `gemm` (vendor BF16 GEMMs, incl. GC recompute) | 2,491.0 | 9.9% |
| `elementwise/reduce` (Add/Mul/GELU/TopK/Reduce) | 2,081.2 | 8.3% |
| `mccl` (FSDP2 AllGather / ReduceScatter) | 462.6 | 1.8% |
| `norm` (LayerNorm fwd/bwd) | 208.0 | 0.8% |
| `optimizer` (AdamW) | 89.7 | 0.4% |
| `other` | 389.9 | 1.5% |

- **Top Bottleneck Classes**:
  1. VSA gather backward scatter-add + tile scatter (`930` calls): `8,320.5 ms` (`33.0%`)
  2. VSA forward block gather (`1,800` calls): `2,784.4 ms` (`11.0%`)
  3. Chunked masked SDPA backward (`450` calls): `2,597.9 ms` (`10.3%`)
  4. Chunked masked SDPA forward (`900` calls, 2x with full GC): `1,895.1 ms` (`7.5%`)
  5. Linear / FFN BF16 GEMMs: `~2,491.0 ms` (`9.9%`)
  6. RoPE `torch.stack` concat (`120` calls): `844.5 ms` (`3.3%`)
  7. Unfused FP32 cast/norm/modulation/residual elementwise ops: `~3,100 ms` (`12.3%`)
  8. FSDP2 FP32 `ReduceScatter` (`31` calls): `324.4 ms` (`1.3%`)

## 4. Frozen Reference Numerics & Trajectories
- **Bitwise Reference (`task/records/ref`)**: 2-step deterministic recording verified **71,008 / 71,008 tensors bitwise identical** across two independent 8-GPU runs (`numerics selftest`: 12/12 mutations PASS).
- **Trajectory Reference (`task/traj/frozen.json`)**: 6 independent deterministic seeds (`10..15`, 6 steps each):
  - `loss_tail` (`steps [2, 6)`): mean `0.037748`, `sigma_seed = 0.006635`
  - `gate_top10_step1` (`steps [1, 2)`): mean `1.96933e-5`, `sigma_seed = 3.3949e-8` (`0.172%` relative sigma; `3*sigma = 0.517%`)
