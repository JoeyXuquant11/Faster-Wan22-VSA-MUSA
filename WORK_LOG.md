# Optimization Work Log: `wan22_vsa_s5000` (Wan2.2-TI2V-5B + VSA Full Finetuning on 8x MTT S5000)

*Archived note: internal GPU microarchitecture identifiers, hardware peak/bandwidth parameters, compiler/runtime internal flags, and proprietary vendor kernel symbol names have been redacted; all round timings, speedups, memory numbers, numerics verdicts, and algorithmic/systems optimization details are preserved.*

- **Reference Commit**: `6e832aaf0dd2ef8edc5400d6448529f166575f98`
- **Frozen Reference Timing (`task/timing_ref.json`)**: `25.2887 s/iter` (spread `0.002%`, peak HBM `56.19 GiB / 80 GiB`)
- **Target**: `>= 4.0x` (`<= 6.32 s/iter`), minimum `>= 2.5x` (`<= 10.12 s/iter`), `status == PASS`, measured tier `<= N2`, time budget `600 min` (10 hours).
- **Effective Model FLOPs (per GPU, 3x fwd without GC)**: `~966 TFLOPs/step` (`~757 TFLOPs` Linear/FFN + `~209 TFLOPs` VSA 80% sparse attention).

---

## Round 0: Baseline Profile & Bytes Model (`runs/r0_baseline_profile`)
- **Command**: `python3 bench/bench.py profile --tree wan22_vsa_s5000 --out runs/r0_baseline_profile --iters 1`
- **Measured (profiled 1 iter)**: `25.956 s` wall, GPU busy `25,214.0 ms` (`99.8%` of `25,273.6 ms` trace window), `34,054` launches, `144` kernel classes, Peak HBM `56.19 GiB / 80 GiB`.
- **Category Breakdown**:
  - `memcpy/cast`: `14,921.5 ms` (`59.2%`) — dominated by VSA gather/scatter-add (`8,324 ms` + `2,783 ms`), RoPE `torch.stack` concat (`843 ms`), FP32/BF16 casts (`~1,160 ms`), VSA tile/transpose/fill/copy (`~900 ms`).
  - `attention`: `4,576.6 ms` (`18.1%`) — chunked masked SDPA bwd (`2,602.5 ms`) + fwd/recompute (`1,893.5 ms`), cross-attn (`~80 ms`).
  - `gemm`: `2,490.4 ms` (`9.9%`) — vendor BF16 GEMMs (incl. GC recompute).
  - `elementwise/reduce`: `2,080.8 ms` (`8.3%`) — unfused FP32 AdaLN/RMSNorm/RoPE/residual/VSA ops.
  - `mccl`: `454.7 ms` (`1.8%`) — FSDP2 FP32 `ReduceScatter` (`313.7 ms`, flagged `under-filled`) + `AllGather` (`141.1 ms`, `under-filled`).
  - `norm`: `207.9 ms` (`0.8%`), `optimizer`: `90.3 ms` (`0.4%`), `other`: `389.8 ms` (`1.5%`).
- **Top Bottleneck Classes (`kernels.md`) & Bytes Model (`runs/r0_bytes_model.json`)**:
  1. VSA KV gather backward scatter-add (`930` calls): `8,324.2 ms` (`33.0%`), `bw-poor`.
  2. VSA KV gather forward (`1,800` calls): `2,782.9 ms` (`11.0%`) — `2.17 TB/step` wasted copy.
  3. Chunked masked SDPA backward (`450` calls): `2,602.5 ms` (`10.3%`).
  4. Chunked masked SDPA forward (`900` calls): `1,893.5 ms` (`7.5%`) — oversized query tiling on 64-token Q blocks wastes 75% of tile work.
  5. Linear / FFN BF16 GEMMs (`1,143` calls): `2,490.4 ms` (`9.9%`), inflated `1.33x` by full GC.
  6. RoPE `torch.stack` concat (`120` calls): `842.7 ms` (`3.3%`), `bw-poor`.
  7. Unfused FP32 cast/add/mul/pow/LayerNorm intermediates: `~3,100 ms` (`12.3%`).
  8. FSDP2 FP32 `ReduceScatter` (`31` calls): `313.7 ms` (`1.2%`), `under-filled`.
- **Repeated Short Chains**: 13 short chains totaling `< 22 ms` (`0.1%` of step); main waste is large unfused kernels (`100 us – 9 ms`), not tiny launch-bound chains yet.
- **Root-Cause Memory Insight**: `36.2 GiB` of the `56.19 GiB` peak HBM is materialized `k_sel`/`v_sel` inside naive Python VSA. Eliminating KV materialization unlocks `~38 GiB` HBM, enabling selective/zero-GC and ZeRO-2 in later rounds.
- **Optimization Plan**:
  - **Round 1**: Custom from-scratch TileLang-MUSA Block-Sparse VSA Attention (Fwd + zero-atomic deterministic Bwd via KV-to-Q CSR). Target: cut VSA fine attention from `~16.0 s` to `~1.0 s` (`~2.3x` total speedup) and drop peak HBM to `~18 GiB`.
  - **Round 2**: Fused MUSA C++ kernels for RoPE+RMSNorm, AdaLN/LayerNorm/Gated-Residual, and VSA Tile+MeanPool / Combine+Untile + Fused AdamW. Target: cut `~4.5 s` elementwise/layout overhead to `~0.4 s`.
  - **Round 3**: Leverage freed HBM to eliminate full GC (recompute-only-GELU / selective GC / async offload) + FSDP2 ZeRO-2 (`reshard_after_forward=False`) + block bucketing.

---

## Round 1: Custom TileLang-MUSA Block-Sparse VSA Attention (`9f5a40b`)
- **Hotspot Targeted**: VSA fine block-sparse attention (gather/scatter-add + masked FA fwd/bwd = `~16.0 s`, `63.5%` of step; `36.2 GiB` materialized KV buffers).
- **Changes**:
  - Added from-scratch `wan22_vsa/vsa_kernels.py`:
    - `vsa_fwd_bn128` (and `bn64` for odd `topk`): TileLang-MUSA pipelined block-sparse attention (`2`-stage async load pipeline) + hoisted `vbs < 64` mask with vectorized `exp2`, zero KV materialization.
    - `vsa_bwd_dq_bn128` (and `bn64`): fused `delta = rowsum(O*dO)` + Q-block-parallel `dQ` block-sparse kernel, zero atomics.
    - `build_kv_csr_kernel` (MUSA C++ via `mcc`): deterministic ascending-Q transpose CSR builder (`~0.45 ms/layer`).
    - `vsa_bwd_dkdv_bm128`: KV-block-parallel 2-Q-blocks/iter `dK`/`dV` kernel, zero atomics, 100% bitwise deterministic.
  - Wired `block_sparse_attention_tilelang` into `wan22_vsa/vsa.py` (BF16, MUSA, `blk=64, D=128`).
- **Verification (`runs/r1_vsa_tl_check/check.json`)**:
  - **Status**: **`PASS`** (`N1`, Gate 3 trajectory `loss_tail = -0.098 sigma`, `gate_top10_step1 = -0.893 sigma`, well within `±3 sigma`).
  - **Steady-state timing**: `25.2887 s/iter` -> **`12.4750 s/iter`** (**`2.0271x` speedup**, window spread `0.08%`).
  - **Peak HBM**: `56.19 GiB` -> **`22.72 GiB / 80 GiB`** (**33.5 GiB freed**).
- **Post-Round-1 Profile (`runs/r1_vsa_tl_profile/kernels.md`, `12,444.3 ms` GPU busy, `99.7%`)**:
  - `attention`: `3,871.4 ms` (`31.1%`) — `vsa_bwd_dkdv_bm128` (`1,615.5 ms`, median `52.4 ms`, p90 `66.1 ms` -> real Wan2.2 KV degree skew vs `36.6 ms` uniform), `vsa_fwd_bn128` (`1,287.6 ms`, 60 calls w/ GC), `vsa_bwd_dq_bn128` (`886.4 ms`).
  - `memcpy/cast` + `elementwise/reduce` + `norm`: **`5,184.3 ms` (`41.7%`)** — RoPE concat (`836.9 ms`, `bw-poor`), BF16<->FP32 casts (`1,148.2 ms`), unfused FP32 AdaLN/Mul/Add/Pow/LN (`~2,200 ms`), VSA tile/transpose/pool/combine/untile (`~800 ms`).
  - `gemm`: `2,495.8 ms` (`20.1%`, inflated `1.33x` by full GC).
  - `mccl`: `462.3 ms` (`3.7%`, `under-filled`).
  - `optimizer`: `89.9 ms` (`0.7%`).
- **Next (Round 2)**: Target the #1 remaining category (`5,184 ms` = `41.7%` unfused elementwise/cast/RoPE/norm/VSA-pre-post) with fused zero-atomic deterministic MUSA C++ kernels (fwd + bwd).

---

## Round 2: Fused MUSA C++ Elementwise, Norm, RoPE & VSA Pre/Post Kernels (`f4118f6`)
- **Hotspot Targeted**: `5,184 ms` (`41.7%` of Round 1 step) in unfused `memcpy/cast` (`3,404 ms`), `elementwise/reduce` (`1,572 ms`), and `norm` (`208 ms`) — dominated by RoPE concat (`837 ms`), BF16<->FP32 casts (`1,148 ms`), unfused FP32 AdaLN/LayerNorm/Mul/Add/Pow (`~2,200 ms`), and VSA tile/transpose/pool/combine/untile (`~800 ms`).
- **Changes**:
  - Added from-scratch `wan22_vsa/fused_ops.py` (MUSA C++ compiled via `mcc -O3`, zero local arrays / zero unions, pure scalar register unpacking for `D=3072` to eliminate compiler local-memory spilling, deterministic zero-atomic split-M backward reductions):
    - `fused_adaln`, `fused_gated_res_ln`, `fused_res_adaln`, `fused_gated_res`
    - `fused_rmsnorm` & `fused_rmsnorm_rope` (eliminates `837 ms` RoPE stack + casts)
    - `fused_vsa_tile_pool` & `fused_vsa_combine_untile` (reads raster `gate_compress` directly without tiling it)
  - Integrated into `wan22_vsa/model.py` (`WanVSATransformerBlock`, `WanVSASelfAttention`, `RMSNorm`, `WanHead`) and `wan22_vsa/vsa.py` (`video_sparse_attention`).
- **Verification (`runs/r2_fused_ops_check/check.json`)**:
  - **Status**: **`PASS`** (`N1`, Gate 3 trajectory `loss_tail = +0.045 sigma`, `gate_top10_step1 = +0.420 sigma`).
  - **Steady-state timing**: `12.4750 s/iter` -> **`7.9739 s/iter`** (**`3.1714x` speedup vs frozen reference**, window spread `0.09%`).
  - **Peak HBM**: `22.72 GiB` -> **`18.92 GiB / 80 GiB`** (`61 GiB` free).
- **Post-Round-2 Profile (`runs/r2_fused_ops_profile/kernels.md`, `7,960.1 ms` GPU busy, `99.7%`)**:
  - `attention`: `4,078.8 ms` (`51.2%`) — `vsa_bwd_dkdv_bm128` (`1,616.4 ms`, median `52.0 ms`, p90 `63.2 ms` — KV degree skew), `vsa_fwd_bn128` (`1,362.6 ms`, 60 calls w/ GC), `vsa_bwd_dq_bn128` (`900.2 ms`).
  - `gemm`: `2,518.9 ms` (`31.6%`, `1,143` calls, inflated `1.33x` by full GC).
  - `mccl`: `438.7 ms` (`5.5%`, `under-filled`).
  - `fused/elementwise/norm/cast`: `~750 ms` (`9.4%`, down from `5,184 ms`).
  - `optimizer`: `89.8 ms` (`1.1%`).
- **Next (Round 3)**: Exploit the `61 GiB` free HBM to eliminate full activation checkpointing (selective/minimal recompute + ZeRO-2 `reshard_after_forward=False` + coarse FSDP2 bucketing) and eliminate KV-block degree skew in `vsa_bwd_dkdv`.

---

## Round 3: Async Pinned-CPU Activation Offload (Static Staging Pool) + GELU-Only Recompute + Bucketed FSDP2 ZeRO-2 (`1741071`)
- **Hotspot Targeted**: Full gradient checkpointing inflated Round 2 by `~1.6 s` (`+687 ms` duplicate `vsa_fwd_bn128`, `+550 ms` duplicate forward GEMMs, `+200 ms` duplicate fused ops), and per-block ZeRO-3 (`bpb=1, reshard=True`) spent `438.7 ms` in `under-filled` MCCL (`317.7 ms` ReduceScatter + `120.9 ms` AllGather across `92` calls), while `61 GiB` of HBM and PCIe DMA engines sat idle.
- **Changes**:
  - Added `wan22_vsa/offload.py` (`AsyncOffloadPool`): overlaps D2H and 1-block-ahead H2D on a dedicated `torch.musa.Stream` using reusable pinned CPU buffers and **static per-slot GPU staging buffers** (`fc1`, `vsa`, `qk_raw`, `gres` — zero `torch.empty` allocations in backward, eliminating caching-allocator thrashing and `152 ms` reclaim stalls).
  - Updated `wan22_vsa/model.py`, `wan22_vsa/vsa.py`, `wan22_vsa/vsa_kernels.py`, `wan22_vsa/fused_ops.py`:
    - Replaced full per-block gradient checkpointing with zero-GEMM/zero-VSA recompute: `_RecomputeGeluLinear` (recomputes only `1.2 ms` GELU, offloads `fc1_out`), offloads `vsa(q4,k4,v4,out4)`, `qk_raw`, and `gres(x_mid, sa_out)` on blocks `0..28`, keeps `cross_attn` checkpointed (`30` blocks, saving `~10 GiB` for `~20 ms`), preserving exact FP32/BF16 math.
    - Expanded `reduce_partial_f32_kernel` grid from `12×256` (`under-filled`) to `96×32` for full GPU occupancy.
  - Updated `wan22_vsa/distributed.py`: grouped FSDP2 wrapping (`WAN22_FSDP_BPB=5`, 6 block buckets + root = 7 buckets) and ZeRO-2 (`WAN22_FSDP_RESHARD=0`, `reshard_after_forward=False`), eliminating backward AllGather.
- **Sweep (`bench.py time`)**:
  - `z3_bpb1`: `6.657 s/iter` (`3.7985x`, peak alloc `62.52 GiB`)
  - `z2_extra_bpb1`: `6.629 s/iter` (`3.8150x`, peak alloc `62.81 GiB`)
  - `z2_extra_bpb2`: `6.510 s/iter` (`3.8846x`, peak alloc `62.81 GiB`)
  - `z2_extra_bpb5`: `6.441 s/iter` (`3.9262x`, peak alloc `62.85 GiB`, peak res `69.25 GiB`, spread `0.03%`) -> **chosen default**
  - `z2_extra_bpb10`: `6.418 s/iter` (`3.9405x`, peak alloc `64.47 GiB`, peak res `72.06 GiB`, spread `0.58%`)
- **Verification (`runs/r3_offload_fsdp_check/check.json`)**:
  - **Status**: **`PASS`** (`N1`, Gate 3 trajectory `loss_tail = +0.049 sigma`, `gate_top10_step1 = +0.399 sigma`).
  - **Steady-state timing**: `7.9739 s/iter` -> **`6.4442 s/iter`** (**`3.9243x` speedup vs frozen reference**, window spread `0.13%`).
  - **Peak HBM**: **`62.85 GiB` allocated / `69.23 GiB` reserved / `80 GiB`** (`10.8 GiB` headroom).
- **Post-Round-3 Profile (`runs/r3_offload_fsdp_profile/kernels.md`, `6,434.4 ms` GPU busy = `99.6%` of `6,458.2 ms` window; `2,265.4 ms` PCIe DMA 100% overlapped, stream-0 gaps `>0.5 ms` only `11.4 ms`)**:
  - **Custom VSA TileLang kernels (`3,165.7 ms`, `49.2%` of step)**:
    - `vsa_bwd_dkdv_bm128_kernel`: `1,588.8 ms` (`24.7%`, 30 calls, median `52.15 ms`, p90 `63.75 ms` — KV degree skew + `4×` half-width `64×128×64` GEMMs & redundant shared-memory transpose copies)
    - `vsa_bwd_dq_bn128_kernel`: `901.3 ms` (`14.0%`, 30 calls, median `29.92 ms` — `2×` half-width `64×128×64` GEMMs & shared-memory transpose copies)
    - `vsa_fwd_bn128_kernel`: `675.6 ms` (`10.5%`, 30 calls, median `22.47 ms`)
  - **Vendor BF16 GEMMs**: `1,966.9 ms` (`30.6%`, `903` calls, down from `2,518.9 ms`).
  - **MCCL**: `349.8 ms` (`5.4%` — `ReduceScatter` `278.7 ms` + `AllGather` `71.1 ms`, down from `438.7 ms` in R2; still `under-filled`).
  - **Other**: `GeluTanh` fwd+bwd (`122.5 ms`), Cross-Attn FA (`84.9 ms`), custom fused ops (`~340 ms`), optimizer/FSDP pack (`~140 ms`), `bitonic_sort_kernel` (`34.3 ms`), `build_kv_csr_kernel` (`14.7 ms`, `under-filled`).
- **Next (Round 4)**: Optimize VSA kernels (`49.2%` of step) — eliminate real-data KV degree skew (`max=480` vs `med=90`, `+520 ms` tail imbalance) and remove shared-memory transpose copies / half-width `K=64` GEMMs in `vsa_bwd_dkdv` and `vsa_bwd_dq`.

---

## Round 4: LPT Descending-Degree VSA `dK/dV` + Direct Full-`K=128` `dQ/dK/dV` + Parallel 288-Block KV CSR (`f45b9f0`)
- **Hotspot Targeted**: Custom TileLang VSA kernels accounted for `3,165.7 ms` (`49.2%`) of Round 3's step (`vsa_bwd_dkdv`: `1,588.8 ms` with real-data KV degree skew `max=480` vs `med=90`, `4×` half-width `64×128×64` GEMMs, `4×` shared-memory transpose copies & `8` barriers/pair; `vsa_bwd_dq`: `901.3 ms` with `2×` transpose copies & `2×` `K=64` GEMMs; `build_kv_csr_kernel`: `14.7 ms`, `under-filled`).
- **Root-Cause Insights & Changes (`wan22_vsa/vsa_kernels.py`)**:
  - **Direct Layout-Matched Full-`K=128` Tiling (`dq_bn128` & `dkdv_bm128`)**: Replaced explicit shared-memory transpose copies and half-width `K=64` GEMMs with direct layout-matched TileLang loads from L2 cache + shared `P_shared[64, 128]`, running full `64×128×128` tiled matrix multiplies with `num_stages=1` staggered pipelining and consumption-ordered loads.
  - **Pure LPT Descending-Degree CTA Ordering (`dkdv_bm128`)**: Ordered all `11,520` `(bh, kv)` blocks by descending degree (`torch.argsort(kv_deg, descending=True, stable=True)`) so heavy KV blocks (`deg=300..480`) start at wave 0 and finish early with zero wave-tail stragglers, zero partial buffers, and `0.0` bitwise difference vs original accumulation.
  - **Parallel 288-Block Deterministic KV CSR (`launch_fast_kv_csr`)**: Split Q-rows into `G=12` chunks (`288` blocks, full GPU coverage, `12×` fewer barriers), dropping CSR time from `15.6 ms` -> `2.3 ms` bitwise-exact.
- **Negative Ablations Tested & Rejected (R1–R4)**:
  - Disabling TileLang default pipelining or `num_stages=0` with `T.Pipelined` hung on async barriers; `threads=128` on `fwd_bn64`/`bn128` doubled latency (`41–44 ms` vs `22 ms`); `gemm_rs` on `dq_bn128` hit TileLang `InternalError: bad optional access`; mid-loop `T.copy` after `T.gemm` inside `T.Pipelined` deadlocked; Split-Q chunking (`chunk=32..128`) was slower (`1,082–1,269 ms`) than pure LPT descending-degree ordering (`932 ms`); custom shared-memory `topk_radix` (`1,308 us`) lost to `bitonic_sort_kernel` (`1,124 us`) due to shared-memory histogram atomic contention; MUSA C++ `mcc` kernels with local arrays/unions spilled heavily to HBM vs pure-scalar register unpacking; dynamic `torch.empty` per block on DMA stream in offload backward caused `600 ms` caching-allocator thrashing at `76 GiB` (eliminated by static per-slot GPU staging buffers).
- **Verification (`runs/r4_vsa_bwd_opt_check/check.json`)**:
  - **Status**: **`PASS`** (`N1`, Gate 3 trajectory `loss_tail = +0.049 sigma`, `gate_top10_step1 = +0.399 sigma` — bitwise identical to Round 3).
  - **Steady-state timing**: `6.4442 s/iter` -> **`5.4875 s/iter`** (**`4.6084x` speedup vs frozen reference** — surpasses the `>= 4.0x` target; window spread `0.04%`).
  - **Peak HBM**: **`62.85 GiB` allocated / `69.23 GiB` reserved / `80 GiB`**.
- **Post-Round-4 Profile (`runs/r4_vsa_bwd_opt_profile/kernels.md`, `5,460.1 ms` GPU busy = `99.3%` of `5,498.9 ms` window)**:
  - `vsa_bwd_dkdv_bm128`: `1,588.8 ms` -> **`938.2 ms`** (`-650.6 ms`, median `31.65 ms`, p90 `33.20 ms` — skew eliminated).
  - `vsa_bwd_dq_bn128`: `901.3 ms` -> **`765.7 ms`** (`-135.6 ms`).
  - `vsa_fwd_bn128`: `678.5 ms` (`12.4%`).
  - Vendor BF16 GEMMs: `1,971.5 ms` (`36.1%`, `903` calls).
  - `MCCL`: `349.8 ms` -> **`163.4 ms`** (`-186.4 ms` bonus from eliminating cross-rank skew in `vsa_bwd_dkdv`; `ReduceScatter` `93.6 ms` + `AllGather` `69.9 ms`).
  - `under-filled` dropped to `174.2 ms` (`3.2%`, `94%` of which is MCCL), `short` `6.6 ms` (`0.1%`), `skewed` `27.0 ms` (`0.5%`).
- **Next (Round 5)**: Probe remaining non-VSA levers (fused QKV+Gate projection, cross-attention recompute elimination, custom `topk-96` replacing `bitonic_sort_kernel`, fast `db2` reduction, 1-pass fused FP32 AdamW) to push toward `~5.0x`.

---

## Round 5: Cross-Attention Async Offload + Fused QKVG `addmm_`/Bias-Grad + 8-Warp `reduce_partial_f32` + 1-Pass Fused FP32 AdamW (`7ff6865`)
- **Hotspot Targeted**: Remaining non-VSA/non-GEMM overhead in Round 4 (`5,460.1 ms` GPU busy): `30/30` Cross-Attention forward recomputes (`~118 ms` across `ca.q`, `ca.o`, `FA fwd`, `norm_q`), 90 Q/K/V/Gate `dx` additions (`44.3 ms`) + 333 slow bias-grad reductions (`69.0 ms`, incl. `skewed` `[27280, 14336]` `db1`), 1-warp `reduce_partial_f32_kernel` (`32.5 ms`), and 5-pass foreach `AdamW` (`~66 ms`).
- **Changes**:
  - `wan22_vsa/model.py`:
    - Added `_run_cross_attn_offload` using `torch.autograd.graph.saved_tensors_hooks` deduplicating underlying storages (`q_raw`, `q`, `out` = `480 MiB/blk` unique, skipping `norm3_x` already on GPU) and switched default `WAN22_CKPT_CA_BLOCKS="0"` (zero Cross-Attention recompute across all 30 blocks; prefetched in `_RecomputeGeluLinear.backward`, released in `fused_gated_res_ln.backward`).
    - Added FSDP2-safe `_FusedQKVGProj` (fuses 4-way `dx` accumulation via `addmm_` `beta=1.0` epilogue + fast `fused_sum_dim0_bf16` bias grads) and `_FastBiasLinear` (for `ffn[0]` `db1` and `self_attn.o`). (Fused QKVG GEMM with dynamic `torch.cat` was tested and rejected: `+9.7 ms` slower due to `640 MiB` `dy` concat.)
  - `wan22_vsa/fused_ops.py`:
    - Upgraded `reduce_partial_f32_kernel` from 1-warp (`32` threads) to 8-warp (`256` threads) with deterministic shared-memory reduction (`92 us -> 13.2 us`, `32.5 ms -> 4.78 ms`).
    - Added `sum_dim0_bf16_fast` (`block=(32, 8)`) for linear bias grads.
  - `wan22_vsa/optim.py` + `train.py` (`Trainer._build_optimizer()` only; `parse_args`, `main`, and `Trainer.train` remain AST-identical):
    - Added `FusedMusaAdamW(torch.optim.AdamW)`: single-pass MUSA C++ chunked multi-tensor FP32 AdamW compiled with strict IEEE-754 FP32 (no FTZ), unwrapping FSDP2 `DTensor._local_tensor` (`66.2 ms -> 24.9 ms`, 1 single kernel launch).
- **Verification (`runs/r5_polish_check/check.json`)**:
  - **Status**: **`PASS`** (`N1`, Gate 3 trajectory `loss_tail = +0.074 sigma`, `gate_top10_step1 = +0.329 sigma`).
  - **Steady-state timing**: `5.4875 s/iter` -> **`5.3212 s/iter`** (**`4.7525x` speedup vs frozen reference**, window spread `0.03%`).
  - **Peak HBM**: **`64.13 GiB` allocated / `72.85 GiB` reserved / `80 GiB`** (`7.15 GiB` headroom).

---

## Round 6: L2-Coherent VSA Scheduling + `clear_accum` + Async Prefetch (`37eac32`)
- **Hotspot Targeted**: Post-R5 profile (`runs/r5_polish_prof/kernels.md`) showed custom TileLang VSA kernels still dominated at `2,380.3 ms` (`45.0%` of GPU time: `dkdv 934.8 ms`, `dq 768.2 ms`, `fwd 677.3 ms`). Real-data routing inspection revealed concurrent thread blocks were reading disjoint KV tiles across heads/blocks, thrashing L2 cache, and `dkdv` pure LPT scattered blocks across all 24 heads in every wave.
- **Changes (`wan22_vsa/vsa_kernels.py`)**:
  - **Ascending-KV `topk_idx` + 4-Quantile Per-Head `QOrder`**: Sorted each Q-block's selected KV indices ascending and clustered Q-blocks per head by 4 quantile keys `(q1/8, q3/8, q5/8, q7/8) >> 6` (`stable=True`) so concurrent blocks walk overlapping K/V tiles in lockstep through L2 cache.
  - **`vsa_fwd_bn128`**: Applied `QOrder` + hoisted `V_shared` loads ahead of `Q@K^T` + `clear_accum=True`.
  - **`vsa_bwd_dq_bn128`**: Applied `QOrder` + `clear_accum=True` + TileLang async prefetch enabled.
  - **`vsa_bwd_dkdv_bm128`**: Switched pure LPT to global bucket-8 LPT (`kv_deg >> 3`, `descending=True, stable=True`) preserving descending-degree tail balance while keeping same-head contiguous KV blocks resident across concurrent blocks, plus `clear_accum=True` + async prefetch (bitwise exact).
- **Negative Ablations Tested & Rejected**: `dkdv` per-head LPT (`903 ms`), bucket `B >= 16` (`804–824 ms`), Q-quantile tie-breaking (`792 ms`), and hybrid big-block LPT (`838–857 ms`) were all slower than global bucket-8 LPT (`786.6 ms` 1-GPU microbench).
- **Verification (`runs/r6_l2_sched_check/check.json`)**:
  - **Status**: **`PASS`** (`N1`, Gate 3 trajectory `loss_tail = -0.070 sigma`, `gate_top10_step1 = -0.609 sigma`).
  - **Steady-state timing**: `5.3212 s/iter` -> **`5.0993 s/iter`** (**`-222.0 ms`**, **`4.9593x`** speedup vs frozen reference, window spread `0.10%`).
  - **Peak HBM**: **`64.13 GiB` allocated / `72.91 GiB` reserved / `80 GiB`**.
- **Post-Round-6 Profile (`runs/r6_l2_sched_profile/kernels.md`)**:
  - `vsa_bwd_dkdv_bm128`: `934.8 ms` -> **`813.0 ms`** (`-121.8 ms`).
  - `vsa_bwd_dq_bn128`: `768.2 ms` -> **`703.0 ms`** (`-65.2 ms`).
  - `vsa_fwd_bn128`: `677.3 ms` -> **`626.0 ms`** (`-51.3 ms`).
  - Next non-VSA hotspots surfaced: `bitonic_sort_kernel<float>` (`34.34 ms`), `bitonic_sort_kernel<int>` (`6.16 ms`), and small-kernel chains (`10.05 ms`).

---

## Round 7: 16-Bit BF16 Warp-Ballot Top-K + Shared-Memory Bitonic QOrder + 64-Bucket Parallel LPT + 3-Block Offload Window (`62325e5`)
- **Hotspot Targeted**: `runs/r6_l2_sched_profile/kernels.md` showed VSA routing & sorting overhead (`bitonic_sort_kernel<float>` `34.34 ms`, `bitonic_sort_kernel<int>` `6.16 ms`, `argsort` chains `#1–#3` `10.05 ms` = `~50.5 ms`, `5,940` kernels/step).
- **Changes (`wan22_vsa/vsa_kernels.py`, `wan22_vsa/vsa.py`, `wan22_vsa/offload.py`)**:
  - **`fused_topk_sorted_bf16_n512_kernel`**: Zero-shared-memory 1-warp/row 16-bit BF16 warp-ballot binary-search Top-K emitting ascending int32 indices + 4-quantile Q-key in a single pass (`34.34 ms -> 3.46 ms` for 30L), guarded to `64 <= N_blk <= 512` (deterministic ascending index tie-breaking).
  - **`sort_qorder_bitonic512_kernel`**: 256-thread shared-memory bitonic sort on `(q_key << 9) | qi` (`0.60 ms`, bitwise exact; replaced rejected 64-thread serial radix `~8 ms`).
  - **`launch_bucket_lpt_order`**: 64-bucket deterministic parallel LPT order for `dkdv` (`64 <= N_blk <= 504`), eliminating `torch.argsort` `bitonic_sort_kernel<int>` (`6.16 ms`) and repeated small-kernel chains.
  - **`offload.py`**: Expanded D2H in-flight window from `2` to `3` blocks (`blk_idx >= 3`) to avoid forward CPU sync stalls as forward time shrank.
- **Negative Ablation Tested & Rejected**: Screened FSDP2 compute–communication overlap toggle via `bench.py time` (`runs/r7_ovl2_time`): `5.1001 s/iter` vs `5.0993 s/iter` (`+0.8 ms`, zero net gain due to collective–compute contention).
- **Verification (`runs/r7_warp_topk_check/check.json`)**:
  - **Status**: **`PASS`** (`N1`, Gate 3 trajectory `loss_tail = -0.007 sigma`, `gate_top10_step1 = -0.296 sigma`).
  - **Steady-state timing**: `5.0993 s/iter` -> **`5.0616 s/iter`** (**`-37.6 ms`**, **`4.9962x`** speedup vs frozen reference, window spread `0.03%`).
  - **Peak HBM**: **`64.13 GiB` allocated / `73.94 GiB` reserved / `80 GiB`**.
- **Post-Round-7 Profile (`runs/r7_warp_topk_profile/kernels.md`)**: Kernel launches dropped `5,940 -> 4,710`; remaining non-VSA/non-GEMM hotspots: `sum_dim0_bf16_fast_s1` (`41.90 ms`), `ReduceABCKernel` (`10.76 ms`), residual skip `AddOp` (`11.34 ms`), and `GeluTanh` fwd/bwd (`122 ms` + `754 MiB` per-block alloc).

---

## Round 8: Fused GELU-Tanh Bwd+Bias + Static 754 MiB FFN Scratch + 7-Linear Fused Bias Grads + AdaLN Skip-Add Fusion (`088fbf0`)
- **Hotspot Targeted**: `runs/r7_warp_topk_profile/kernels.md` exposed `41.90 ms` in `sum_dim0_bf16_fast_s1`, `10.76 ms` in `ReduceABCKernel`, `11.34 ms` in residual `AddOp`, plus dynamic `754 MiB` FFN activation allocations in `_RecomputeGeluLinear`.
- **Changes (`wan22_vsa/fused_ops.py`, `wan22_vsa/offload.py`, `wan22_vsa/model.py`)**:
  - **`fused_gelu_tanh_fwd_u4_kernel` + static `ffn_act` scratch**: Pure-scalar vectorized BF16 GELU-tanh writing directly into a static `754 MiB` `pool.get_scratch("ffn_act")` buffer (bitwise exact, zero dynamic alloc).
  - **`fused_gelu_tanh_bwd_bias_s1`** (`split_m=4`, `50.0 ms`): Fuses `ffn[0].bias` reduction directly into GELU-tanh backward (eliminating `17.2 ms` `fc1.bias` `sum_dim0`; rejected `split_m in {5, 10, 15, 20}` which ran `51.6–58.6 ms`).
  - **Fused bias-grad accumulation across 7 linear layers/block**: Added fused `db` accumulation to `fused_gated_res_bwd` (`fc2.bias`), `fused_res_adaln_bwd` (`ca.o.bias`), `fused_gated_res_ln_bwd` (`sa.o.bias`), and `fused_rmsnorm_rope_bwd` (`sa.q/k.bias`, `ca.q.bias`), slashing `sum_dim0_bf16_fast_s1` from `41.90 ms -> 8.25 ms` and `ReduceABCKernel` from `10.76 ms -> 0 ms`.
  - **AdaLN skip-add fusion**: Added `dx_add_box` to `fused_adaln_bwd` to fuse the block residual skip addition directly (eliminating `11.34 ms` `AddOp`).
- **Verification (`runs/r8_bias_gelu_fusion_check/check.json`)**:
  - **Status**: **`PASS`** (`N1`, Gate 3 trajectory `loss_tail = -0.008 sigma`, `gate_top10_step1 = -0.151 sigma`).
  - **Steady-state timing**: `5.0616 s/iter` -> **`5.0315 s/iter`** (**`-30.2 ms`**, **`5.0261x`** speedup vs frozen reference — breaks `5.0x`; window spread `0.19%`).
  - **Peak HBM**: **`64.31 GiB` allocated / `74.96 GiB` reserved / `80 GiB`**.
- **Post-Round-8 Profile (`runs/r8_bias_gelu_fusion_profile/kernels.md`, `5,004.9 ms` GPU busy = `99.4%`)**:
  - Revealed two high-payoff norm bottlenecks: (1) adding 8 live `dbx0..7` register accumulators in `fused_rmsnorm_rope_bwd_d3072` triggered a register-occupancy cliff (`39.8 -> 53.5 ms`, `+13.7 ms`); (2) all four 384-thread forward norm kernels (`117.5 ms` combined) were `__syncthreads()` barrier-bound, and all three LayerNorm/AdaLN backward kernels performed two sequential `block384_reduce_sum` calls (`4` barriers/row).

---

## Round 9: 1-Warp/Row Zero-Barrier Forward Norms + Fused `block384_reduce_sum2` & Shared-Memory-Accumulated Backward Norms (`d601204`)
- **Hotspot Targeted**: `runs/r8_bias_gelu_fusion_profile/kernels.md` showed `253.5 ms` across 8 custom norm/residual kernels (`117.5 ms` barrier-bound forward norms + `136.0 ms` register-heavy/double-barrier backward norms).
- **Changes (`wan22_vsa/fused_ops.py`)**:
  - **1-Warp/Row Zero-Barrier Forward Norms (ZERO `__syncthreads()`)**: Rewrote all four `D=3072` forward norms (`fused_adaln_fwd_d3072` 4 warps/CTA with shared-memory row cache; `fused_rmsnorm_rope_fwd_d3072<DO_ROPE>` 4 warps/CTA + `12×` hoisted `cos/sin`; `fused_res_adaln_fwd_d3072` 8 warps/CTA; `fused_gated_res_ln_fwd_d3072` 8 warps/CTA). Each warp processes one row via pure `__shfl_xor_sync`.
  - **Fused `block384_reduce_sum2` + Shared-Memory Accumulators in Backward Norms**: Merged the two sequential `block384_reduce_sum` calls into one dual-sum barrier pair (halving `__syncthreads()` per row) and moved per-column FP32 partial accumulators into `__shared__` arrays (zero register spills, zero extra barriers) in `fused_rmsnorm_rope_bwd_d3072<DO_ROPE, DO_DB>`, `fused_adaln_bwd_d3072<DO_ADD>`, `fused_res_adaln_bwd_d3072<DO_DB>`, and `fused_gated_res_ln_bwd_d3072<DO_DB>`.
- **Negative Ablations Tested & Rejected**: `adaln_bwd_warp` (large shared-memory footprint) starved occupancy (`151 ms`); shared-memory row cache on 2-input `res_adaln_fwd` slightly slowed it (`20.0–20.7 ms` vs `19.1 ms` L1 reload).
- **Verification (`runs/r9_warp_norms_check/check.json`)**:
  - **Status**: **`PASS`** (`N1`, Gate 3 trajectory `loss_tail = -0.090 sigma`, `gate_top10_step1 = -0.860 sigma`; kernel unit tests verified `< 3.5e-3` vs FP32 across `M in {64, 385, 480, 2720}` and bitwise determinism).
  - **Steady-state timing**: `5.0315 s/iter` -> **`4.9634 s/iter`** (**`-68.1 ms`**, **`5.0950x`** speedup vs frozen reference — breaks the 5-second barrier; window spread `0.04%`).
  - **Peak HBM**: **`64.31 GiB` allocated / `74.96 GiB` reserved / `80 GiB`**.
- **Post-Round-9 Profile (`runs/r9_warp_norms_profile/kernels.md`, `4,923.9 ms` GPU busy = `99.4%` of `4,954.5 ms`)**:
  - Norm kernels dropped sharply: `adaln_fwd` `23.9 -> 10.4 ms`, `rmsnorm_rope_fwd` `43.5 -> 25.5 ms`, `rmsnorm_rope_bwd` `53.5 -> 39.2 ms`, `adaln_bwd` `28.3 -> 15.0 ms`, `res_adaln_bwd` `25.3 -> 15.7 ms`.
  - Trace inspection exposed remaining levers: (1) `13.31 ms` of Stream 0 GPU idle gaps at blocks 27–29 caused by giant `754 MiB` / `540 MiB` `aten::empty()` caching-allocator compaction + `record_stream` waits (`127 ms + 74 ms + 51 ms` CPU stalls), also slowing `chunk_cat` to `45.37 ms`; (2) last `60` calls of `sum_dim0_bf16_fast_s1` (`8.26 ms` for `sa.v.bias` & `sa.to_gate_compress.bias`).

---

## Round 10: Static 3-Slot GPU Ring Buffers (`fc1`/`vsa`) + Fused VSA Bwd `db_v`/`db_gate` + 256-Thread Swapped `tile_pool_fwd` (`0355829`)
- **Hotspot Targeted**: `runs/r9_warp_norms_profile` trace showed `13.31 ms` Stream 0 idle gaps from `aten::empty()` (`754 MiB` `fc1` + `720 MiB` `vsa`) + `record_stream` fragmentation (`10.65 GiB` reserved overhead, `chunk_cat` `45.37 ms`), plus the last `8.26 ms` of `sum_dim0_bf16_fast_s1` (`sa.v.bias` & `sa.to_gate_compress.bias`).
- **Changes (`wan22_vsa/offload.py`, `wan22_vsa/fused_ops.py`, `wan22_vsa/vsa_kernels.py`, `wan22_vsa/vsa.py`, `wan22_vsa/model.py`)**:
  - **Static 3-Slot GPU Ring Buffers (`AsyncOffloadPool.get_ring_slot`)**: Replaced dynamic per-block `aten::empty()` + `record_stream()` for offloaded `fc1` (`754 MiB` via `torch.addmm(..., out=fc1_out)` in `_FastBiasLinear`) and `vsa` (`q_t, k_t, v_t, out_s = 720 MiB` via `out_qkv` / `out_buf`) across blocks `0..28` with static 3-slot GPU ring buffers synchronized asynchronously on GPU via `compute_stream.wait_event(slot_d2h_evt)` (zero CPU blocking, zero `42.7 GiB/step` dynamic allocation churn).
  - **Fused VSA Bwd `db_v` & `db_gate` (`sum_dim0_bf16_fast_s1` `8.26 ms -> 0 ms`)**: Converted `fused_vsa_tile_pool_bwd_d128` (`grid=(BH, P_blk=240, 3)`) and `fused_vsa_combine_untile_bwd_d128` (`grid=(BH, P_blk=240)`) to accumulate FP32 partial bias grads (`p_dbv`, `p_dbg`) reduced via `reduce_partial_f32_to_bf16_kernel`, wired to `_FusedQKVGProj` (`dbv_box`, `dbg_box`).
  - **VSA `tile_pool_fwd` & Norm Polish**: Upgraded `fused_vsa_tile_pool_fwd_d128` to 256 threads with L2-local grid `(BH, N_blk, 3)`, tuned `fused_gated_res_bwd` to `P=120`, and deferred `dxmv/uv` loads in `fused_gated_res_ln_bwd` (`P=180`).
- **Negative Ablations Tested & Rejected**:
  - **TileLang-MUSA Cross-Attention (`S_q=27280, S_k=512, D=128`)**: Microbenchmarked custom TileLang kernels vs native vendor FlashAttention (`17.4 ms` fwd + `74.8 ms` bwd = `92.2 ms`); on short `S_k=512` (`4` KV blocks), custom TileLang ran `61.1 ms` fwd + `300.3 ms` bwd (`3.9x` slower).
  - **FSDP2 Bucket Size (`WAN22_FSDP_BPB`)**: Screened via `bench.py time --config`: `bpb=6` (`4.9769 s`, `+13.5 ms`), `bpb=10` (`5.1351 s`, `+171.7 ms`) vs default `bpb=5` (`4.9634 s`).
  - **Offload Block Count (`WAN22_OFFLOAD_BLOCKS`)**: `28` (`5.0184 s`, `+55.0 ms`) and `27` (`4.9911 s`, `+27.7 ms`) broke the `blk_idx - 1` `fc1` prefetch handoff.
  - **`fused_adamw_f32_chunk_kernel` fast math (`__fdividef`/`__fsqrt_rn`)**: `25.57 ms` vs `25.60 ms` (`0.03 ms`, already HBM write-bound).
- **Verification (`runs/r10_ring_vsa_bias_check/check.json`)**:
  - **Status**: **`PASS`** (`N1`, Gate 3 trajectory `loss_tail = -0.096 sigma`, `gate_top10_step1 = -0.960 sigma`; 5-block ring-wraparound parity & 2-run bitwise determinism test + `test_vsa_correctness.py` passed with `MUSA_LAUNCH_BLOCKING=1`).
  - **Steady-state timing**: `4.9634 s/iter` -> **`4.9468 s/iter`** (**`-16.6 ms`**, **`5.1122x`** speedup vs frozen reference, window spread `0.07%`).
  - **Peak HBM**: **`68.60 GiB` allocated / `74.92 GiB` reserved / `80 GiB`** (fragmentation reduced from `10.65 GiB` -> `6.32 GiB`, `5.08 GiB` headroom).

---

## Round 11: VSA Fwd `BN=256` + Fused Scale, `dQ` Register-Hoisted `lse/delta`, `grln_bwd` `P=240`, and `blk 29` `fc1_last` Scratch (`a5b1bbd`)
- **Profile First (`runs/r10_ring_vsa_bias_profile/kernels.md`)**:
  - Top custom hotspots: `vsa_bwd_dkdv_bm128` (`819.4 ms`), `vsa_bwd_dq_bn128` (`701.4 ms`), `vsa_fwd_bn128` (`628.8 ms`), `fused_gated_res_ln_bwd` (`31.1 ms`, regressed `+5.8 ms` in R10 from `P=180` wave-imbalance scheduling), and `blk 29` forward `4.32 ms` `aten::empty(754 MiB)` stall.
  - Generated-source inspection showed `vsa_bwd_dq_bn128` issued repeated shared-memory broadcast loads of `lse_shared`/`delta_shared` inside its 48-iter KV loop, and `vsa_fwd_bn128` (`BN=128`, 48 iters) executed a separate scalar `sm_scale` mul pass + twice as many online-softmax rescalings/barriers as `BN=256`.
- **Changes**:
  - `wan22_vsa/vsa_kernels.py`:
    - Added `_build_fwd_bn256` (`BN=256`, `4` KV tiles/iter, `num_stages=1`, dispatched when `topk % 4 == 0`) with fused `sm_scale` inside `T.exp2(acc_s * sm_scale - scores_max * sm_scale)` (`-23.7 ms` in full training: `628.8 -> 605.1 ms`).
    - Updated `_build_bwd_dq_bn128`: hoisted per-row `lse_shared`/`delta_shared` into 2-float registers `lse_reg`/`delta_reg` once before the KV loop and reordered the second `K` load after `Q@K^T` and `dO@V^T` (`-42.6 ms` in full training: `701.4 -> 658.8 ms`, 100% bitwise exact).
  - `wan22_vsa/fused_ops.py`: Reverted `_FusedGatedResLN.backward` `P` from `min(180, ...)` back to `_pick_p(S)` (`240` for full wave balance, `31.1 -> 25.5 ms`, `-5.6 ms`).
  - `wan22_vsa/model.py`: Added `fc1_last` static scratch buffer in `WanVSATransformerBlock._run_ffn` for the non-offloaded `blk 29` (`off is None`, eliminating the last `4.32 ms` forward `aten::empty` stall at zero extra peak HBM).
- **Negative Ablations Tested & Rejected in R11**:
  - **Explicit shared-memory transpose copies**: `+134.2 ms` slower on `dq` (`781.5 vs 647.3 ms`), `+439.2 ms` slower on `dkdv` (`1,256.6 vs 817.5 ms`), and `num_stages=2` exceeded shared-memory capacity.
  - **Disabled TileLang pipelining toggle**: deadlocked async barriers across all tested kernels.
  - **`T.GemmWarpPolicy.Square` / `FullCol`**: broke `T.reduce_max(..., dim=1)` layout in TileLang-MUSA (`err = 1.00`).
  - **`BN=192`**: failed TileLang tile-layout divisibility (`mat_continuous % chunk_cols == 0`); `threads=128` and `BN=64` were `1.65–4x` slower.
  - **VSA fast-math flag / staged output store / `dkdv` `lse/delta` register hoisting & `head_grp=1..4` LPT**: `-0.4 ms` noise to `+345 ms` slower.
  - **Manual bit-shift IEEE RNE `f32x2_to_bf16x2` in `fused_ops.py`**: `+9.62 ms` slower across fused kernels than `__float2bfloat16` (`281.1 vs 271.5 ms`).
  - **FSDP2 overlap toggle**: `+0.8 ms` regression.
- **Verification (`runs/r11_vsa_bn256_regdq_check/check.json`, profile `runs/r11_vsa_bn256_regdq_profile/kernels.md`)**:
  - **Status**: **`PASS`** (`N1`, Gate 3 trajectory `loss_tail = -0.034 sigma`, `gate_top10_step1 = -0.350 sigma`; `test_vsa_correctness.py` + `bn256` vs `bn128` + 5-block 2-step parity & bitwise determinism passed with `MUSA_LAUNCH_BLOCKING=1`).
  - **Steady-state timing**: `4.9468 s/iter` -> **`4.8866 s/iter`** (**`-60.1 ms`**, **`5.1751x`** speedup vs frozen reference, window spread `0.32%`).
  - **Peak HBM**: **`68.60 GiB` allocated / `75.37 GiB` reserved / `80 GiB`**.

---

## Round 12: Branchless 4-Tile Arithmetic `vbs` Mask in `vsa_fwd_bn256` + Clamp-Safe Zero-`vbs`-Mask in `vsa_bwd_dq_bn128` (`3258fd7`)
- **Profile First (`runs/r11_vsa_bn256_regdq_profile/kernels.md`) & Root-Cause Discovery**:
  - Comparing R11 training profile (`vsa_fwd_bn256 = 605.1 ms`, `vsa_bwd_dq_bn128 = 658.8 ms`) against synthetic `all_64` microbenchmarks (`500.6 ms`, `557.3 ms`) revealed a **`+116 ms` real-workload tax** caused solely by boundary partial blocks on `dit_seq_shape = (31, 22, 40)` (`130 / 480 = 27.1%` partial blocks with `vbs in {24, 32, 48}` -> `71.7%` of `BN=256` forward iterations and `46.8%` of `BN=128` `dQ` iterations entered `T.if_then_else` masking).
  - Generated-source inspection showed TileLang lowers nested `T.if_then_else` into nested C `if/else` control-flow branches inside the 64-element register loop (`acc_s`), causing severe branch-predication overhead (`+67 ms` in `fwd`, `+52 ms` in `dq`).
- **Iterative Numerics Safety Diagnosis (R12a -> R12b)**:
  - **R12a (`runs/r12_vsa_branchless_vbs_check`, REJECTED)**: Initial `dq_nomask` (dropping `vbs` masking in `dq` because `K[j >= vbs] == 0, V[j >= vbs] == 0 => dS * K == 0`) + multiplicative `dkdv` mask ran **`4.7431 s/iter` (`5.3317x`)** and matched R11 bitwise across all `71,008` tensors in `train_2step`, **but hit `NaN` on `traj_s10` step 1 backward** (`grad_norm = NaN`).
  - **Root cause proven**: On pretrained Wan2.2-5B QK-RMSNorm heads where a valid token has `lse[i] < -127` (large negative valid logits), padded keys (`K=0 => S=0`) evaluate `exp2(0 - lse[i]) = +Inf`, producing `+Inf * 0.0 = NaN` in matrix multiplication (`dq_nomask`) and in `acc_s_t * T.Cast("float", j < vbs_k)` (`dkdv`).
  - **R12b (Clamp-safe mathematical theorem, ACCEPTED)**: For EVERY valid key `j < vbs`, `S_ij * scale_log2 - lse_i <= 0.0` strictly. Splitting `acc_s = T.min(acc_s * scale_log2 - lse_reg[i], 1.0)` then `acc_s = T.exp2(acc_s)` preserves vectorized `exp2` codegen, is **100% bitwise identical** on all valid keys, bounds padded `exp2(...) <= 2.0` (so `dS * K_zero == finite * 0.0 == 0.0` even at `lse = -700`), and eliminates all `Vbs` loads/branches in `vsa_bwd_dq_bn128`. Reverted `vsa_bwd_dkdv_bm128` to `T.if_then_else` (safe against `+Inf * 0.0`).
- **Changes (`wan22_vsa/vsa_kernels.py`)**:
  - `_build_fwd_bn256`: Replaced nested 4-level `T.if_then_else` mask with `(vbs0 + vbs1 + vbs2 + vbs3) < BN` + 4 unnested branchless per-tile arithmetic masks `acc_s[i, j + off] += T.Cast("float", j >= vbsK) * (-1e30)` (zero extra barriers, `605.1 -> 558.1 ms`, **`-46.9 ms`** in full training, bitwise identical).
  - `_build_bwd_dq_bn128`: Removed `Vbs` argument/loads and `if (vbs0 < blk) or (vbs1 < blk)` mask loop, replacing with clamp-safe `T.min(acc_s * scale_log2 - lse_reg[i], 1.0)` + vectorized `T.exp2` (`658.8 -> 605.7 ms`, **`-53.1 ms`** in full training, bitwise identical).
- **Negative Ablations Tested & Rejected in R12–R13**:
  - `m_shared[256]` shared-memory mask buffer in `vsa_fwd_bn256`: inserted 2 extra barrier syncs (`+28 ms` even on `all_64`).
  - `VbsMask` GMEM `[N_blk, 64]` lookup / `two_phase` (`0..n_full4` + `n_full4..24`) / `post_clamp` (`T.min` after `exp2` -> `+121.6 ms`) / `dkdv` epilogue mask (`+5.0 ms`) / `dkdv` pre-exp2 mask (`+37.6 ms`) / `late_vbs` (`+140.2 ms`).
  - `fullfirst` `topk_idx` sort: gave zero incremental gain on top of 4-tile branchless `four_arith` (`531.5 vs 531.4 ms`); `dkdv` bucket LPT `shift` sweep (`0,3,4,5,6,7,9`) confirmed `shift=3` (`bucket=8`, `743.3 ms`) is the global optimum (`shift=4/5/6/9` were `+6` to `+45 ms` slower).
  - **R13 (`runs/r13_vsa_dq_fastsafe_check`, REJECTED)**: CTA-uniform `need_clamp` (`lse < -20` / `|delta| > 1e15`) `if/else` fast path in `vsa_bwd_dq_bn128` passed (`PASS`, `N1`) at `4.8049 s/iter` (`5.2631x`), `+20.3 ms` slower than R12's branchless `split_clamp` (`4.7846 s/iter`, `5.2854x`); reverted cleanly to `3258fd7`.
  - **Multi-stage async copy reordering probe (pure `/tmp` microbench, zero repo edits)**: Deferring `V_shared` load after `Q@K^T` in `vsa_fwd_bn256` (`-0.5 to -0.7 ms`) and staging backward loads into 2/3 stages in `vsa_bwd_dkdv_bm128` (`-0.2 to -0.3 ms`) confirmed TileLang-MUSA's default pipeline already achieves full load/compute overlap.
  - **8-GPU MCCL collective parameter sweep (pure `/tmp` microbench on exact `bpb=5` `AG`/`RS` sizes)**: Default MCCL configuration (`120.7 ms` for 6×AG+6×RS) beat all alternative channel counts (`129.1–173.1 ms`); larger buffer size gained only `-5.6 ms` (`0.11%`) at `~0.9 GiB` extra HBM cost and was rejected.
- **Verification (`runs/r12_vsa_branchless_vbs_safe_check/check.json`, profile `runs/r12_vsa_branchless_vbs_profile/kernels.md`)**:
  - **Status**: **`PASS`** (`N1`, Gate 3 trajectory `loss_tail = -0.034 sigma`, `gate_top10_step1 = -0.350 sigma` — **bitwise identical to R11 to all 17 decimal digits**; passed `test_vsa_correctness.py` + adversarial `lse ~ -700 << -127` NaN stress test with `MUSA_LAUNCH_BLOCKING=1`).
  - **Steady-state timing**: `4.8866 s/iter` -> **`4.7846 s/iter`** (**`-102.0 ms`**, **`5.2854x`** speedup vs frozen reference, window spread `0.21%`).
  - **Peak HBM**: **`68.60 GiB` allocated / `75.33 GiB` reserved / `80 GiB`**.

---

## Final Summary & Plateau Analysis (`runs/r12_vsa_branchless_vbs_profile/kernels.md`)

| Round | Commit | s/iter | Speedup vs Ref | Peak HBM (Alloc / Res) | Numerics | Check |
|---|---|---:|---:|---:|---|---|
| Ref (R0) | `6e832aa` | `25.2887 s` | `1.0000x` | `56.19 / 56.70 GiB` | Ref | `task/timing_ref.json` |
| R1 (TileLang VSA) | `9f5a40b` | `12.4750 s` | `2.0271x` | `22.72 / — GiB` | `N1` (`-0.10σ`, `-0.89σ`) | `runs/r1_vsa_tl_check/check.json` |
| R2 (Fused MUSA C++) | `f4118f6` | `7.9739 s` | `3.1714x` | `18.92 / — GiB` | `N1` (`+0.05σ`, `+0.42σ`) | `runs/r2_fused_ops_check/check.json` |
| R3 (Async Offload + FSDP2) | `1741071` | `6.4442 s` | `3.9243x` | `62.85 / 69.23 GiB` | `N1` (`+0.05σ`, `+0.40σ`) | `runs/r3_offload_fsdp_check/check.json` |
| R4 (LPT + Full-K128 VSA) | `f45b9f0` | `5.4875 s` | `4.6084x` | `62.85 / 69.23 GiB` | `N1` (`+0.05σ`, `+0.40σ`) | `runs/r4_vsa_bwd_opt_check/check.json` |
| R5 (CA Offload + Fused AdamW/QKVG) | `7ff6865` | `5.3212 s` | `4.7525x` | `64.13 / 72.85 GiB` | `N1` (`+0.07σ`, `+0.33σ`) | `runs/r5_polish_check/check.json` |
| R6 (L2-Coherent VSA Sched) | `37eac32` | `5.0993 s` | `4.9593x` | `64.13 / 72.91 GiB` | `N1` (`-0.07σ`, `-0.61σ`) | `runs/r6_l2_sched_check/check.json` |
| R7 (Warp-Ballot TopK + LPT) | `62325e5` | `5.0616 s` | `4.9962x` | `64.13 / 73.94 GiB` | `N1` (`-0.01σ`, `-0.30σ`) | `runs/r7_warp_topk_check/check.json` |
| R8 (GELU+7-Bias+Skip Fusion) | `088fbf0` | `5.0315 s` | `5.0261x` | `64.31 / 74.96 GiB` | `N1` (`-0.01σ`, `-0.15σ`) | `runs/r8_bias_gelu_fusion_check/check.json` |
| R9 (1-Warp Fwd + SMEM Bwd Norms) | `d601204` | `4.9634 s` | `5.0950x` | `64.31 / 74.96 GiB` | `N1` (`-0.09σ`, `-0.86σ`) | `runs/r9_warp_norms_check/check.json` |
| R10 (Ring Buf + VSA Bias Fusion) | `0355829` | `4.9468 s` | `5.1122x` | `68.60 / 74.92 GiB` | `N1` (`-0.10σ`, `-0.96σ`) | `runs/r10_ring_vsa_bias_check/check.json` |
| R11 (VSA BN256 + Reg-dQ + fc1_last) | `a5b1bbd` | `4.8866 s` | `5.1751x` | `68.60 / 75.37 GiB` | `N1` (`-0.03σ`, `-0.35σ`) | `runs/r11_vsa_bn256_regdq_check/check.json` |
| **R12 (Branchless VBS + Clamp-Safe dQ)** | **`3258fd7`** | **`4.7846 s`** | **`5.2854x`** | **`68.60 / 75.33 GiB`** | **`N1` (`-0.03σ`, `-0.35σ`)** | **`runs/r12_vsa_branchless_vbs_safe_check/check.json`** |

- **Final Plateau Evidence (`runs/r12_vsa_branchless_vbs_profile/kernels.md`, `4,753.281 ms` GPU busy = `99.4%` of `4,781.450 ms` window, `4,470` kernels/step, `2,773.4 ms` PCIe DMA 100% overlapped, Stream 0 gaps = `28.22 ms` total / `1.6 us` median)**:
  - **Tier 1 — BF16 Matrix Compute & Sparse Attention (`3,931.9 ms`, `82.7%` of GPU time)**:
    - Custom TileLang-MUSA block-sparse VSA (`1,990.6 ms`, `41.9%`, down **`-87.4%`** from `15,826 ms` baseline): `vsa_bwd_dkdv_bm128` (`826.8 ms`), `vsa_bwd_dq_bn128` (`605.7 ms`, register-hoisted `lse/delta` + clamp-safe zero-Vbs-mask), `vsa_fwd_bn256` (`558.1 ms`, fused `sm_scale`, branchless 4-tile `vbs` mask).
    - Vendor BF16 GEMMs (`1,941.3 ms`, `40.8%`, `873` calls, zero forward GEMM recompute).
  - **Tier 2 — 8-GPU FSDP2 ZeRO-2 + MCCL (`236.1 ms`, `5.0%`)**:
    - `ReduceScatter` (`108.3 ms`) + `AllGather` (`64.0 ms`) + `chunk_cat` (`32.2 ms`) + `split_with_sizes` (`31.6 ms`).
  - **Tier 3 — Cross-Attention (`96.3 ms`, `2.0%`)**:
    - Native vendor FlashAttention fwd (`17.5 ms`) + bwd (`78.8 ms`, zero recompute; verified `3.75x` faster than custom TileLang on short `S_k=512`).
  - **Tier 4 — Bandwidth-Bound Custom MUSA C++ Elementwise / Norms / VSA Pre-Post / Optimizer (`~410 ms`, `8.6%`)**:
    - `fused_gelu_tanh_fwd` (`72.2 ms`) + `fused_gelu_tanh_bwd_bias` (`50.1 ms`).
    - `fused_vsa_tile_pool_fwd` (`26.7 ms`) + `bwd` (`26.7 ms`) + `fused_vsa_combine_untile_bwd` (`18.1 ms`) + `fwd` (`11.6 ms`) + `fused_topk_sorted_bf16` (`3.4 ms`).
    - 1-warp/row zero-barrier norm fwd + SMEM/sum2 bwd norm/residual kernels (`~160 ms`), 1-pass `fused_adamw_f32_chunk_kernel` (`25.1 ms`, 1 launch).
  - **Flagged Residuals (`~79 ms`, `1.7%`)**: `short` = `4.39 ms` (`0.1%`), top repeated chain = `2.77 ms` (`0.06%`), `under-filled` = `182.9 ms` (`3.8%`, `94.2%` MCCL), `skewed` = `524.3 ms` (histogram artifact of one GEMM class mixing `6.15 ms` FFN GEMMs with `1.35 ms` attention GEMMs). All major compute, memory-bandwidth, and distributed communication bottlenecks are near saturation within the BF16/FP32 precision rules.
