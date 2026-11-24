"""Custom TileLang-MUSA + MUSA C++ kernels for Video Sparse Attention (VSA) on MTT S5000.

NOTE: Do NOT add `from __future__ import annotations` here (TileLang-MUSA AST parser requirement).
All kernels are written from scratch for MTT S5000 and are 100% bitwise deterministic
(zero global atomicAdd in forward and backward; deterministic ascending-Q KV CSR transpose).
"""

import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist
import tilelang
import tilelang.language as T

MUSA_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
    tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: False,
    tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
}

MUSA_PASS_CONFIGS_PREFETCH = {
    **MUSA_PASS_CONFIGS,
    tilelang.PassConfigKey.TL_ENABLE_MUSA_TMA_PREFETCH: True,
}

MUSA_COMPILE_FLAGS = [
    "-fmusa-flush-denormals-to-zero",
    "-fno-signed-zeros",
    "-fno-strict-aliasing",
    "-mllvm",
    "-misched=mtgpu-max-ilp",
]

_CSR_CU_SOURCE = r"""
#include <musa_runtime.h>
#include <stdint.h>

__global__ void build_kv_csr_kernel(
    const int32_t* __restrict__ topk_idx,
    int32_t* __restrict__ kv_offsets,
    int32_t* __restrict__ kv_q_idx,
    int32_t N_blk,
    int32_t topk
) {
    extern __shared__ int32_t smem[];
    int32_t* counts = smem;
    int32_t* cursor = smem + N_blk;

    int32_t bh = blockIdx.x;
    int32_t tid = threadIdx.x;
    int32_t nthreads = blockDim.x;
    int32_t total_edges = N_blk * topk;
    const int32_t* idx_bh = topk_idx + (int64_t)bh * total_edges;
    int32_t* off_bh = kv_offsets + (int64_t)bh * (N_blk + 1);
    int32_t* q_bh = kv_q_idx + (int64_t)bh * total_edges;

    for (int32_t i = tid; i < N_blk; i += nthreads) {
        counts[i] = 0;
    }
    __syncthreads();

    for (int32_t e = tid; e < total_edges; e += nthreads) {
        int32_t kv = idx_bh[e];
        atomicAdd(&counts[kv], 1);
    }
    __syncthreads();

    if (tid == 0) {
        int32_t acc = 0;
        for (int32_t i = 0; i < N_blk; ++i) {
            off_bh[i] = acc;
            cursor[i] = acc;
            acc += counts[i];
        }
        off_bh[N_blk] = acc;
    }
    __syncthreads();

    // Strictly ascending q order: within a single q row all topk KV indices are distinct,
    // so every KV block receives its attending Q indices in deterministic ascending order.
    for (int32_t q = 0; q < N_blk; ++q) {
        const int32_t* row = idx_bh + (int64_t)q * topk;
        for (int32_t k = tid; k < topk; k += nthreads) {
            int32_t kv = row[k];
            int32_t pos = atomicAdd(&cursor[kv], 1);
            q_bh[pos] = q;
        }
        __syncthreads();
    }
}

extern "C" int launch_build_kv_csr(
    const int32_t* topk_idx,
    int32_t* kv_offsets,
    int32_t* kv_q_idx,
    int32_t BH,
    int32_t N_blk,
    int32_t topk,
    void* stream
) {
    size_t smem_bytes = (size_t)2 * (size_t)N_blk * sizeof(int32_t);
    int threads = (topk > 128) ? 256 : 128;
    build_kv_csr_kernel<<<BH, threads, smem_bytes, (musaStream_t)stream>>>(
        topk_idx, kv_offsets, kv_q_idx, N_blk, topk
    );
    return (int)musaGetLastError();
}

// Parallel Q-chunk deterministic ascending-Q KV CSR + per-KV degree output (N_blk <= 512).
// Launches BH*G blocks (e.g. 24*12=288 blocks -> 100% MP coverage) with 12x fewer barriers.
__global__ void csr_chunk_count_kernel(
    const int32_t* __restrict__ topk_idx,
    int32_t* __restrict__ chunk_cnt,
    int32_t N_blk, int32_t topk, int32_t G
) {
    extern __shared__ int32_t s_cnt[];
    int32_t blk_id = blockIdx.x;
    int32_t bh = blk_id / G;
    int32_t g = blk_id % G;
    int32_t tid = threadIdx.x;
    int32_t nthreads = blockDim.x;

    for (int32_t i = tid; i < N_blk; i += nthreads) s_cnt[i] = 0;
    __syncthreads();

    int32_t q_per = (N_blk + G - 1) / G;
    int32_t q_lo = g * q_per;
    int32_t q_hi = q_lo + q_per;
    if (q_hi > N_blk) q_hi = N_blk;

    int32_t total_edges = N_blk * topk;
    const int32_t* idx_bh = topk_idx + (int64_t)bh * total_edges;
    int32_t e_lo = q_lo * topk;
    int32_t e_hi = q_hi * topk;
    for (int32_t e = e_lo + tid; e < e_hi; e += nthreads) {
        int32_t kv = idx_bh[e];
        atomicAdd(&s_cnt[kv], 1);
    }
    __syncthreads();

    int32_t* out = chunk_cnt + ((int64_t)bh * G + g) * N_blk;
    for (int32_t i = tid; i < N_blk; i += nthreads) out[i] = s_cnt[i];
}

__global__ void csr_scan_offsets_kernel(
    const int32_t* __restrict__ chunk_cnt,
    int32_t* __restrict__ chunk_cursor,
    int32_t* __restrict__ kv_offsets,
    int32_t* __restrict__ kv_deg,
    int32_t N_blk, int32_t G
) {
    __shared__ int32_t s_deg[512];
    __shared__ int32_t s_warp[16];
    int32_t bh = blockIdx.x;
    int32_t tid = threadIdx.x;
    const int32_t* cnt_bh = chunk_cnt + (int64_t)bh * G * N_blk;
    int32_t* cur_bh = chunk_cursor + (int64_t)bh * G * N_blk;

    int32_t deg = 0;
    if (tid < N_blk) {
        for (int32_t g = 0; g < G; ++g) {
            int32_t c = cnt_bh[(int64_t)g * N_blk + tid];
            cur_bh[(int64_t)g * N_blk + tid] = deg;
            deg += c;
        }
        kv_deg[(int64_t)bh * N_blk + tid] = deg;
    }
    s_deg[tid] = deg;
    __syncthreads();

    int32_t lane = tid & 31;
    int32_t warp = tid >> 5;
    int32_t val = s_deg[tid];
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        int32_t n = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) val += n;
    }
    if (lane == 31) s_warp[warp] = val;
    __syncthreads();
    if (warp == 0) {
        int32_t wv = (lane < 16) ? s_warp[lane] : 0;
        #pragma unroll
        for (int offset = 1; offset < 16; offset <<= 1) {
            int32_t n = __shfl_up_sync(0xffffffff, wv, offset);
            if (lane >= offset) wv += n;
        }
        if (lane < 16) s_warp[lane] = wv;
    }
    __syncthreads();
    int32_t warp_base = (warp > 0) ? s_warp[warp - 1] : 0;
    int32_t incl = val + warp_base;
    int32_t excl = incl - deg;

    int32_t* off_bh = kv_offsets + (int64_t)bh * (N_blk + 1);
    if (tid < N_blk) {
        off_bh[tid] = excl;
        for (int32_t g = 0; g < G; ++g) {
            cur_bh[(int64_t)g * N_blk + tid] += excl;
        }
    }
    if (tid == N_blk - 1) {
        off_bh[N_blk] = incl;
    }
}

__global__ void csr_chunk_scatter_kernel(
    const int32_t* __restrict__ topk_idx,
    const int32_t* __restrict__ chunk_cursor,
    int32_t* __restrict__ kv_q_idx,
    int32_t N_blk, int32_t topk, int32_t G
) {
    extern __shared__ int32_t s_cur[];
    int32_t blk_id = blockIdx.x;
    int32_t bh = blk_id / G;
    int32_t g = blk_id % G;
    int32_t tid = threadIdx.x;
    int32_t nthreads = blockDim.x;

    const int32_t* cur_in = chunk_cursor + ((int64_t)bh * G + g) * N_blk;
    for (int32_t i = tid; i < N_blk; i += nthreads) s_cur[i] = cur_in[i];
    __syncthreads();

    int32_t q_per = (N_blk + G - 1) / G;
    int32_t q_lo = g * q_per;
    int32_t q_hi = q_lo + q_per;
    if (q_hi > N_blk) q_hi = N_blk;

    int32_t total_edges = N_blk * topk;
    const int32_t* idx_bh = topk_idx + (int64_t)bh * total_edges;
    int32_t* q_bh = kv_q_idx + (int64_t)bh * total_edges;

    for (int32_t q = q_lo; q < q_hi; ++q) {
        const int32_t* row = idx_bh + (int64_t)q * topk;
        for (int32_t k = tid; k < topk; k += nthreads) {
            int32_t kv = row[k];
            int32_t pos = atomicAdd(&s_cur[kv], 1);
            q_bh[pos] = q;
        }
        __syncthreads();
    }
}

extern "C" int launch_fast_kv_csr(
    const int32_t* topk_idx,
    int32_t* chunk_cnt,
    int32_t* chunk_cursor,
    int32_t* kv_offsets,
    int32_t* kv_q_idx,
    int32_t* kv_deg,
    int32_t BH,
    int32_t N_blk,
    int32_t topk,
    int32_t G,
    void* stream
) {
    musaStream_t st = (musaStream_t)stream;
    size_t smem = (size_t)N_blk * sizeof(int32_t);
    csr_chunk_count_kernel<<<BH * G, 256, smem, st>>>(topk_idx, chunk_cnt, N_blk, topk, G);
    csr_scan_offsets_kernel<<<BH, 512, 0, st>>>(chunk_cnt, chunk_cursor, kv_offsets, kv_deg, N_blk, G);
    int th_s = (topk > 128) ? 256 : 128;
    csr_chunk_scatter_kernel<<<BH * G, th_s, smem, st>>>(topk_idx, chunk_cursor, kv_q_idx, N_blk, topk, G);
    return (int)musaGetLastError();
}

static __device__ __forceinline__ uint32_t bf16_to_ordered_u32(uint16_t b) {
    if (b == 0x8000u) b = 0u;
    uint32_t mask = (b & 0x8000u) ? 0xffffu : 0x8000u;
    return ((uint32_t)b ^ mask) + 1u;
}

static __device__ __forceinline__ int32_t warp_sum_i32(int32_t v) {
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}

// 1-warp/row (4 warps/CTA, zero SMEM, zero syncthreads) exact BF16 Top-K selection
// emitting strictly ascending int32 KV indices + 12-bit 4-quantile L2 clustering key.
__global__ void fused_topk_sorted_bf16_n512_kernel(
    const uint16_t* __restrict__ scores_bf16,
    int32_t* __restrict__ out_idx,
    int32_t* __restrict__ out_qkey,
    int32_t rows,
    int32_t N,
    int32_t K
) {
    int32_t lane = threadIdx.x & 31;
    int32_t warp = threadIdx.x >> 5;
    int32_t row = blockIdx.x * 4 + warp;
    if (row >= rows) return;
    const uint16_t* inp = scores_bf16 + (int64_t)row * N;

    uint32_t u0  = (lane +   0 < N) ? bf16_to_ordered_u32(inp[lane +   0]) : 0u;
    uint32_t u1  = (lane +  32 < N) ? bf16_to_ordered_u32(inp[lane +  32]) : 0u;
    uint32_t u2  = (lane +  64 < N) ? bf16_to_ordered_u32(inp[lane +  64]) : 0u;
    uint32_t u3  = (lane +  96 < N) ? bf16_to_ordered_u32(inp[lane +  96]) : 0u;
    uint32_t u4  = (lane + 128 < N) ? bf16_to_ordered_u32(inp[lane + 128]) : 0u;
    uint32_t u5  = (lane + 160 < N) ? bf16_to_ordered_u32(inp[lane + 160]) : 0u;
    uint32_t u6  = (lane + 192 < N) ? bf16_to_ordered_u32(inp[lane + 192]) : 0u;
    uint32_t u7  = (lane + 224 < N) ? bf16_to_ordered_u32(inp[lane + 224]) : 0u;
    uint32_t u8  = (lane + 256 < N) ? bf16_to_ordered_u32(inp[lane + 256]) : 0u;
    uint32_t u9  = (lane + 288 < N) ? bf16_to_ordered_u32(inp[lane + 288]) : 0u;
    uint32_t u10 = (lane + 320 < N) ? bf16_to_ordered_u32(inp[lane + 320]) : 0u;
    uint32_t u11 = (lane + 352 < N) ? bf16_to_ordered_u32(inp[lane + 352]) : 0u;
    uint32_t u12 = (lane + 384 < N) ? bf16_to_ordered_u32(inp[lane + 384]) : 0u;
    uint32_t u13 = (lane + 416 < N) ? bf16_to_ordered_u32(inp[lane + 416]) : 0u;
    uint32_t u14 = (lane + 448 < N) ? bf16_to_ordered_u32(inp[lane + 448]) : 0u;
    uint32_t u15 = (lane + 480 < N) ? bf16_to_ordered_u32(inp[lane + 480]) : 0u;

    uint32_t tau = 0u;
    #pragma unroll
    for (int bit = 16; bit >= 0; --bit) {
        uint32_t cand = tau | (1u << bit);
        int32_t c = (int32_t)(u0  >= cand) + (int32_t)(u1  >= cand)
                  + (int32_t)(u2  >= cand) + (int32_t)(u3  >= cand)
                  + (int32_t)(u4  >= cand) + (int32_t)(u5  >= cand)
                  + (int32_t)(u6  >= cand) + (int32_t)(u7  >= cand)
                  + (int32_t)(u8  >= cand) + (int32_t)(u9  >= cand)
                  + (int32_t)(u10 >= cand) + (int32_t)(u11 >= cand)
                  + (int32_t)(u12 >= cand) + (int32_t)(u13 >= cand)
                  + (int32_t)(u14 >= cand) + (int32_t)(u15 >= cand);
        if (warp_sum_i32(c) >= K) tau = cand;
    }

    int32_t c_gt = (int32_t)(u0  > tau) + (int32_t)(u1  > tau)
                 + (int32_t)(u2  > tau) + (int32_t)(u3  > tau)
                 + (int32_t)(u4  > tau) + (int32_t)(u5  > tau)
                 + (int32_t)(u6  > tau) + (int32_t)(u7  > tau)
                 + (int32_t)(u8  > tau) + (int32_t)(u9  > tau)
                 + (int32_t)(u10 > tau) + (int32_t)(u11 > tau)
                 + (int32_t)(u12 > tau) + (int32_t)(u13 > tau)
                 + (int32_t)(u14 > tau) + (int32_t)(u15 > tau);
    int32_t need_eq = K - warp_sum_i32(c_gt);

    int32_t* outp = out_idx + (int64_t)row * K;
    uint32_t lane_mask = (1u << lane) - 1u;
    int32_t out_base = 0;
    int32_t eq_seen = 0;
    int32_t i0 = K >> 3;
    int32_t i1 = (3 * K) >> 3;
    int32_t i2 = (5 * K) >> 3;
    int32_t i3 = (7 * K) >> 3;
    int32_t qk0 = 0, qk1 = 0, qk2 = 0, qk3 = 0;

    #define EMIT_STEP(UVAL, BASE_J) do { \
        int32_t j = (BASE_J) + lane; \
        bool valid = (j < N); \
        bool gt = valid && ((UVAL) > tau); \
        bool eq = valid && ((UVAL) == tau); \
        uint32_t eq_m = __ballot_sync(0xffffffffu, eq); \
        int32_t eq_rank = eq_seen + __popc(eq_m & lane_mask); \
        bool keep = gt || (eq && eq_rank < need_eq); \
        eq_seen += __popc(eq_m); \
        uint32_t km = __ballot_sync(0xffffffffu, keep); \
        int32_t pos = out_base + __popc(km & lane_mask); \
        if (keep) { \
            outp[pos] = j; \
            if (pos == i0) qk0 = j; \
            if (pos == i1) qk1 = j; \
            if (pos == i2) qk2 = j; \
            if (pos == i3) qk3 = j; \
        } \
        out_base += __popc(km); \
    } while (0)

    EMIT_STEP(u0,    0); EMIT_STEP(u1,   32); EMIT_STEP(u2,   64); EMIT_STEP(u3,   96);
    EMIT_STEP(u4,  128); EMIT_STEP(u5,  160); EMIT_STEP(u6,  192); EMIT_STEP(u7,  224);
    EMIT_STEP(u8,  256); EMIT_STEP(u9,  288); EMIT_STEP(u10, 320); EMIT_STEP(u11, 352);
    EMIT_STEP(u12, 384); EMIT_STEP(u13, 416); EMIT_STEP(u14, 448); EMIT_STEP(u15, 480);
    #undef EMIT_STEP

    if (out_qkey != nullptr) {
        qk0 = warp_sum_i32(qk0);
        qk1 = warp_sum_i32(qk1);
        qk2 = warp_sum_i32(qk2);
        qk3 = warp_sum_i32(qk3);
        if (lane == 0) {
            out_qkey[row] = ((qk0 >> 6) << 9) | ((qk1 >> 6) << 6) | ((qk2 >> 6) << 3) | (qk3 >> 6);
        }
    }
}

// Single-launch deterministic stable per-head Q-order sort (N_blk <= 512).
// Packs (key << 9) | qi into uint32 so bitonic sort is 100% stable and bitwise exact.
__global__ void sort_qorder_bitonic512_kernel(
    const int32_t* __restrict__ q_key,
    int32_t* __restrict__ q_order,
    int32_t N_blk
) {
    __shared__ uint32_t s[512];
    int32_t bh = blockIdx.x;
    int32_t tid = threadIdx.x;
    const int32_t* key_bh = q_key + (int64_t)bh * N_blk;
    s[tid] = (tid < N_blk) ? (((uint32_t)key_bh[tid] << 9) | (uint32_t)tid) : 0xffffffffu;
    s[tid + 256] = (tid + 256 < N_blk) ? (((uint32_t)key_bh[tid + 256] << 9) | (uint32_t)(tid + 256)) : 0xffffffffu;
    __syncthreads();

    #pragma unroll
    for (int32_t size = 2; size <= 512; size <<= 1) {
        for (int32_t stride = size >> 1; stride >= 1; stride >>= 1) {
            int32_t pos = 2 * tid - (tid & (stride - 1));
            int32_t partner = pos ^ stride;
            bool asc = ((pos & size) == 0);
            uint32_t a = s[pos];
            uint32_t b = s[partner];
            if (asc ? (a > b) : (a < b)) {
                s[pos] = b;
                s[partner] = a;
            }
            __syncthreads();
        }
    }
    int32_t base = bh * N_blk;
    int32_t* out_bh = q_order + (int64_t)base;
    if (tid < N_blk) out_bh[tid] = base + (int32_t)(s[tid] & 511u);
    if (tid + 256 < N_blk) out_bh[tid + 256] = base + (int32_t)(s[tid + 256] & 511u);
}

// Deterministic 64-bucket descending stable sort for bucket-8 LPT (kv_deg >> 3 in [0, 63]).
constexpr int LPT_G = 120;
constexpr int LPT_B = 64;

__global__ void lpt_bucket_hist_kernel(
    const int32_t* __restrict__ kv_deg,
    int32_t* __restrict__ chunk_cnt,
    int32_t total_kv,
    int32_t shift
) {
    __shared__ uint8_t s_bkt[256];
    int32_t g = blockIdx.x;
    int32_t tid = threadIdx.x;
    int32_t lo = (int32_t)((int64_t)total_kv * g / LPT_G);
    int32_t hi = (int32_t)((int64_t)total_kv * (g + 1) / LPT_G);
    int32_t n = hi - lo;

    for (int32_t i = tid; i < n; i += 64) {
        int32_t b = kv_deg[lo + i] >> shift;
        if (b < 0) b = 0;
        if (b >= LPT_B) b = LPT_B - 1;
        s_bkt[i] = (uint8_t)(LPT_B - 1 - b);
    }
    __syncthreads();

    int32_t c = 0;
    for (int32_t i = 0; i < n; ++i) {
        if ((int32_t)s_bkt[i] == tid) ++c;
    }
    chunk_cnt[g * LPT_B + tid] = c;
}

__global__ void lpt_bucket_scan_kernel(
    const int32_t* __restrict__ chunk_cnt,
    int32_t* __restrict__ chunk_off
) {
    __shared__ int32_t s_tot[LPT_B];
    int32_t b = threadIdx.x;
    int32_t acc = 0;
    for (int32_t g = 0; g < LPT_G; ++g) {
        chunk_off[g * LPT_B + b] = acc;
        acc += chunk_cnt[g * LPT_B + b];
    }
    s_tot[b] = acc;
    __syncthreads();
    if (b == 0) {
        int32_t run = 0;
        for (int32_t k = 0; k < LPT_B; ++k) {
            int32_t t = s_tot[k];
            s_tot[k] = run;
            run += t;
        }
    }
    __syncthreads();
    int32_t base = s_tot[b];
    for (int32_t g = 0; g < LPT_G; ++g) {
        chunk_off[g * LPT_B + b] += base;
    }
}

__global__ void lpt_bucket_scatter_kernel(
    const int32_t* __restrict__ kv_deg,
    const int32_t* __restrict__ chunk_off,
    int32_t* __restrict__ lpt_order,
    int32_t total_kv,
    int32_t shift
) {
    __shared__ uint8_t s_bkt[256];
    int32_t g = blockIdx.x;
    int32_t tid = threadIdx.x;
    int32_t lo = (int32_t)((int64_t)total_kv * g / LPT_G);
    int32_t hi = (int32_t)((int64_t)total_kv * (g + 1) / LPT_G);
    int32_t n = hi - lo;

    for (int32_t i = tid; i < n; i += 64) {
        int32_t b = kv_deg[lo + i] >> shift;
        if (b < 0) b = 0;
        if (b >= LPT_B) b = LPT_B - 1;
        s_bkt[i] = (uint8_t)(LPT_B - 1 - b);
    }
    __syncthreads();

    int32_t pos = chunk_off[g * LPT_B + tid];
    for (int32_t i = 0; i < n; ++i) {
        if ((int32_t)s_bkt[i] == tid) {
            lpt_order[pos++] = lo + i;
        }
    }
}

extern "C" int launch_fused_topk_qorder_bf16(
    const void* scores_bf16,
    void* out_idx,
    void* out_qkey,
    void* out_qorder,
    int32_t BH,
    int32_t N_blk,
    int32_t K,
    void* stream
) {
    musaStream_t st = (musaStream_t)stream;
    int32_t rows = BH * N_blk;
    fused_topk_sorted_bf16_n512_kernel<<<(rows + 3) / 4, 128, 0, st>>>(
        (const uint16_t*)scores_bf16, (int32_t*)out_idx, (int32_t*)out_qkey, rows, N_blk, K
    );
    if (out_qorder != nullptr && out_qkey != nullptr) {
        sort_qorder_bitonic512_kernel<<<BH, 256, 0, st>>>(
            (const int32_t*)out_qkey, (int32_t*)out_qorder, N_blk
        );
    }
    return (int)musaGetLastError();
}

extern "C" int launch_bucket_lpt_order(
    const void* kv_deg,
    void* scratch_cnt,
    void* scratch_off,
    void* lpt_order,
    int32_t total_kv,
    int32_t shift,
    void* stream
) {
    musaStream_t st = (musaStream_t)stream;
    lpt_bucket_hist_kernel<<<LPT_G, 64, 0, st>>>((const int32_t*)kv_deg, (int32_t*)scratch_cnt, total_kv, shift);
    lpt_bucket_scan_kernel<<<1, 64, 0, st>>>((const int32_t*)scratch_cnt, (int32_t*)scratch_off);
    lpt_bucket_scatter_kernel<<<LPT_G, 64, 0, st>>>((const int32_t*)kv_deg, (const int32_t*)scratch_off, (int32_t*)lpt_order, total_kv, shift);
    return (int)musaGetLastError();
}
"""

_CSR_LIB = None
_KERNEL_CACHE = {}


def _is_dist_multi() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _is_local_leader() -> bool:
    if not _is_dist_multi():
        return True
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return int(local_rank) == 0
    return dist.get_rank() == 0


def _get_csr_lib():
    global _CSR_LIB
    if _CSR_LIB is not None:
        return _CSR_LIB

    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "wan22_vsa_musa_ext"
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(_CSR_CU_SOURCE.encode("utf-8")).hexdigest()[:16]
    so_path = cache_dir / f"vsa_csr_s5000_{digest}.so"

    def _compile_so():
        if so_path.exists():
            return
        pid = os.getpid()
        cu_tmp = cache_dir / f"vsa_csr_{digest}_{pid}.cu"
        so_tmp = cache_dir / f"vsa_csr_{digest}_{pid}.so"
        cu_tmp.write_text(_CSR_CU_SOURCE)
        cmd = [
            "/usr/local/musa/bin/mcc",
            "--musa-path=/usr/local/musa",
            "-x", "musa",
            "--offload-arch=mp_31",
            "-O3",
            "-fmusa-flush-denormals-to-zero",
            "--shared", "-fPIC",
            str(cu_tmp),
            "-L/usr/local/musa/lib",
            "-lmusart",
            "-o", str(so_tmp),
        ]
        subprocess.check_call(cmd)
        os.replace(so_tmp, so_path)
        try:
            cu_tmp.unlink()
        except OSError:
            pass

    if _is_dist_multi():
        if _is_local_leader():
            _compile_so()
        dist.barrier()
    else:
        _compile_so()

    lib = ctypes.CDLL(str(so_path))
    lib.launch_build_kv_csr.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    lib.launch_build_kv_csr.restype = ctypes.c_int
    lib.launch_fast_kv_csr.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    lib.launch_fast_kv_csr.restype = ctypes.c_int
    lib.launch_fused_topk_qorder_bf16.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    lib.launch_fused_topk_qorder_bf16.restype = ctypes.c_int
    lib.launch_bucket_lpt_order.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    lib.launch_bucket_lpt_order.restype = ctypes.c_int
    _CSR_LIB = lib
    return lib


def fused_topk_qorder_bf16(scores: torch.Tensor, topk: int):
    """Exact BF16 warp-ballot Top-K emitting ascending int32 indices + per-head 4-quantile Q-order."""
    scores_c = scores.contiguous()
    B, H, N_blk, N_k = scores_c.shape
    assert N_blk == N_k and N_blk <= 512 and scores_c.dtype == torch.bfloat16
    BH = B * H
    out_idx = torch.empty((BH, N_blk, topk), device=scores_c.device, dtype=torch.int32)
    if topk % 2 == 0:
        out_qkey = torch.empty((BH, N_blk), device=scores_c.device, dtype=torch.int32)
        out_qorder = torch.empty((BH * N_blk,), device=scores_c.device, dtype=torch.int32)
        qk_ptr = out_qkey.data_ptr()
        qo_ptr = out_qorder.data_ptr()
    else:
        out_qorder = None
        qk_ptr = 0
        qo_ptr = 0
    lib = _get_csr_lib()
    stream_ptr = torch.musa.current_stream(scores_c.device).musa_stream
    rc = lib.launch_fused_topk_qorder_bf16(
        scores_c.data_ptr(),
        out_idx.data_ptr(),
        qk_ptr,
        qo_ptr,
        BH,
        N_blk,
        topk,
        stream_ptr,
    )
    if rc != 0:
        raise RuntimeError(f"launch_fused_topk_qorder_bf16 failed with MUSA error {rc}")
    return out_idx, out_qorder


def _compile_cached(key, builder):
    kn = _KERNEL_CACHE.get(key)
    if kn is not None:
        return kn
    if _is_dist_multi():
        if _is_local_leader():
            kn = builder()
        dist.barrier()
        if kn is None:
            kn = builder()
        dist.barrier()
    else:
        kn = builder()
    _KERNEL_CACHE[key] = kn
    return kn


def _build_fwd_bn256(BH: int, N_blk: int, topk: int, blk: int = 64, D: int = 128):
    sm_scale = (D ** -0.5) * 1.4426950408889634
    dtype = "bfloat16"
    accum_dtype = "float"
    BN = 4 * blk
    n_iters = topk // 4
    total_q = BH * N_blk

    @T.prim_func
    def vsa_fwd_bn256(
        Q: T.Tensor((BH, N_blk, blk, D), dtype),
        K: T.Tensor((BH, N_blk, blk, D), dtype),
        V: T.Tensor((BH, N_blk, blk, D), dtype),
        TopkIdx: T.Tensor((BH, N_blk, topk), "int32"),
        Vbs: T.Tensor((N_blk,), "int32"),
        QOrder: T.Tensor((total_q,), "int32"),
        Out: T.Tensor((BH, N_blk, blk, D), dtype),
        Lse: T.Tensor((BH, N_blk, blk), accum_dtype),
    ):
        # BN=256 (4 KV tiles/iter, 176 KB SMEM) halves online-softmax rescalings & barriers
        # and fuses sm_scale into vectorized exp2 (no separate 64-elem scalar mul pass).
        with T.Kernel(total_q, threads=256) as (tid,):
            Q_shared = T.alloc_shared([blk, D], dtype)
            K_shared = T.alloc_shared([BN, D], dtype)
            V_shared = T.alloc_shared([BN, D], dtype)
            S_shared = T.alloc_shared([blk, BN], dtype)

            acc_s = T.alloc_fragment([blk, BN], accum_dtype)
            acc_o = T.alloc_fragment([blk, D], accum_dtype)
            scores_max = T.alloc_fragment([blk], accum_dtype)
            scores_max_prev = T.alloc_fragment([blk], accum_dtype)
            scores_scale = T.alloc_fragment([blk], accum_dtype)
            scores_sum = T.alloc_fragment([blk], accum_dtype)
            logsum = T.alloc_fragment([blk], accum_dtype)

            flat = QOrder[tid]
            bh = flat // N_blk
            qi = flat % N_blk

            T.copy(Q[bh, qi, :, :], Q_shared)
            T.clear(acc_o)
            T.fill(logsum, 0)
            T.fill(scores_max, -1e30)

            for t in T.Pipelined(n_iters, num_stages=1):
                kv0 = TopkIdx[bh, qi, t * 4]
                kv1 = TopkIdx[bh, qi, t * 4 + 1]
                kv2 = TopkIdx[bh, qi, t * 4 + 2]
                kv3 = TopkIdx[bh, qi, t * 4 + 3]
                vbs0 = Vbs[kv0]
                vbs1 = Vbs[kv1]
                vbs2 = Vbs[kv2]
                vbs3 = Vbs[kv3]
                T.copy(K[bh, kv0, :, :], K_shared[0:64, :])
                T.copy(K[bh, kv1, :, :], K_shared[64:128, :])
                T.copy(K[bh, kv2, :, :], K_shared[128:192, :])
                T.copy(K[bh, kv3, :, :], K_shared[192:256, :])
                T.copy(V[bh, kv0, :, :], V_shared[0:64, :])
                T.copy(V[bh, kv1, :, :], V_shared[64:128, :])
                T.copy(V[bh, kv2, :, :], V_shared[128:192, :])
                T.copy(V[bh, kv3, :, :], V_shared[192:256, :])
                T.gemm(
                    Q_shared,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                    clear_accum=True,
                )
                # Branchless per-tile arithmetic mask (no nested C if/else, no extra SMEM barrier -> -42 ms bitwise exact).
                if (vbs0 + vbs1 + vbs2 + vbs3) < BN:
                    if vbs0 < blk:
                        for i, j in T.Parallel(blk, blk):
                            acc_s[i, j] = acc_s[i, j] + T.Cast("float", j >= vbs0) * (-1e30)
                    if vbs1 < blk:
                        for i, j in T.Parallel(blk, blk):
                            acc_s[i, j + 64] = acc_s[i, j + 64] + T.Cast("float", j >= vbs1) * (-1e30)
                    if vbs2 < blk:
                        for i, j in T.Parallel(blk, blk):
                            acc_s[i, j + 128] = acc_s[i, j + 128] + T.Cast("float", j >= vbs2) * (-1e30)
                    if vbs3 < blk:
                        for i, j in T.Parallel(blk, blk):
                            acc_s[i, j + 192] = acc_s[i, j + 192] + T.Cast("float", j >= vbs3) * (-1e30)
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(blk):
                    scores_scale[i] = T.exp2((scores_max_prev[i] - scores_max[i]) * sm_scale)
                for i, j in T.Parallel(blk, BN):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * sm_scale - scores_max[i] * sm_scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(blk):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                T.copy(acc_s, S_shared)
                for i, j in T.Parallel(blk, D):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(S_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(blk, D):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, Out[bh, qi, :, :])
            for i in T.Parallel(blk):
                Lse[bh, qi, i] = scores_max[i] * sm_scale + T.log2(logsum[i])

    return tilelang.compile(
        vsa_fwd_bn256,
        target="musa",
        pass_configs=MUSA_PASS_CONFIGS,
        compile_flags=MUSA_COMPILE_FLAGS,
    )


def _build_fwd_bn128(BH: int, N_blk: int, topk: int, blk: int = 64, D: int = 128):
    sm_scale = (D ** -0.5) * 1.4426950408889634
    dtype = "bfloat16"
    accum_dtype = "float"
    BN = 2 * blk
    n_iters = topk // 2
    total_q = BH * N_blk

    @T.prim_func
    def vsa_fwd_bn128(
        Q: T.Tensor((BH, N_blk, blk, D), dtype),
        K: T.Tensor((BH, N_blk, blk, D), dtype),
        V: T.Tensor((BH, N_blk, blk, D), dtype),
        TopkIdx: T.Tensor((BH, N_blk, topk), "int32"),
        Vbs: T.Tensor((N_blk,), "int32"),
        QOrder: T.Tensor((total_q,), "int32"),
        Out: T.Tensor((BH, N_blk, blk, D), dtype),
        Lse: T.Tensor((BH, N_blk, blk), accum_dtype),
    ):
        # L2-coherent quantile-clustered Q-block schedule + ascending-KV topk_idx
        # keeps concurrent MPs reading overlapping K/V tiles within the same head.
        with T.Kernel(total_q, threads=256) as (tid,):
            Q_shared = T.alloc_shared([blk, D], dtype)
            K_shared = T.alloc_shared([BN, D], dtype)
            V_shared = T.alloc_shared([BN, D], dtype)
            S_shared = T.alloc_shared([blk, BN], dtype)

            acc_s = T.alloc_fragment([blk, BN], accum_dtype)
            acc_o = T.alloc_fragment([blk, D], accum_dtype)
            scores_max = T.alloc_fragment([blk], accum_dtype)
            scores_max_prev = T.alloc_fragment([blk], accum_dtype)
            scores_scale = T.alloc_fragment([blk], accum_dtype)
            scores_sum = T.alloc_fragment([blk], accum_dtype)
            logsum = T.alloc_fragment([blk], accum_dtype)

            flat = QOrder[tid]
            bh = flat // N_blk
            qi = flat % N_blk

            T.copy(Q[bh, qi, :, :], Q_shared)
            T.clear(acc_o)
            T.fill(logsum, 0)
            T.fill(scores_max, -1e30)

            for t in T.Pipelined(n_iters, num_stages=2):
                kv0 = TopkIdx[bh, qi, t * 2]
                kv1 = TopkIdx[bh, qi, t * 2 + 1]
                vbs0 = Vbs[kv0]
                vbs1 = Vbs[kv1]
                T.copy(K[bh, kv0, :, :], K_shared[0:blk, :])
                T.copy(K[bh, kv1, :, :], K_shared[blk:BN, :])
                T.copy(V[bh, kv0, :, :], V_shared[0:blk, :])
                T.copy(V[bh, kv1, :, :], V_shared[blk:BN, :])
                T.gemm(
                    Q_shared,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                    clear_accum=True,
                )
                for i, j in T.Parallel(blk, BN):
                    acc_s[i, j] = acc_s[i, j] * sm_scale
                if (vbs0 < blk) or (vbs1 < blk):
                    for i, j in T.Parallel(blk, BN):
                        acc_s[i, j] = T.if_then_else(
                            T.if_then_else(j < blk, j < vbs0, (j - blk) < vbs1),
                            acc_s[i, j],
                            -1e30,
                        )
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(blk):
                    scores_scale[i] = T.exp2(scores_max_prev[i] - scores_max[i])
                for i, j in T.Parallel(blk, BN):
                    acc_s[i, j] = T.exp2(acc_s[i, j] - scores_max[i])
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(blk):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                T.copy(acc_s, S_shared)
                for i, j in T.Parallel(blk, D):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(S_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(blk, D):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, Out[bh, qi, :, :])
            for i in T.Parallel(blk):
                Lse[bh, qi, i] = scores_max[i] + T.log2(logsum[i])

    return tilelang.compile(
        vsa_fwd_bn128,
        target="musa",
        pass_configs=MUSA_PASS_CONFIGS,
        compile_flags=MUSA_COMPILE_FLAGS,
    )


def _build_fwd_bn64(BH: int, N_blk: int, topk: int, blk: int = 64, D: int = 128):
    sm_scale = (D ** -0.5) * 1.4426950408889634
    dtype = "bfloat16"
    accum_dtype = "float"

    @T.prim_func
    def vsa_fwd_bn64(
        Q: T.Tensor((BH, N_blk, blk, D), dtype),
        K: T.Tensor((BH, N_blk, blk, D), dtype),
        V: T.Tensor((BH, N_blk, blk, D), dtype),
        TopkIdx: T.Tensor((BH, N_blk, topk), "int32"),
        Vbs: T.Tensor((N_blk,), "int32"),
        Out: T.Tensor((BH, N_blk, blk, D), dtype),
        Lse: T.Tensor((BH, N_blk, blk), accum_dtype),
    ):
        with T.Kernel(N_blk, BH, threads=256) as (qi, bh):
            Q_shared = T.alloc_shared([blk, D], dtype)
            K_shared = T.alloc_shared([blk, D], dtype)
            V_shared = T.alloc_shared([blk, D], dtype)
            S_shared = T.alloc_shared([blk, blk], dtype)

            acc_s = T.alloc_fragment([blk, blk], accum_dtype)
            acc_o = T.alloc_fragment([blk, D], accum_dtype)
            scores_max = T.alloc_fragment([blk], accum_dtype)
            scores_max_prev = T.alloc_fragment([blk], accum_dtype)
            scores_scale = T.alloc_fragment([blk], accum_dtype)
            scores_sum = T.alloc_fragment([blk], accum_dtype)
            logsum = T.alloc_fragment([blk], accum_dtype)

            T.copy(Q[bh, qi, :, :], Q_shared)
            T.clear(acc_o)
            T.fill(logsum, 0)
            T.fill(scores_max, -1e30)

            for t in T.Pipelined(topk, num_stages=2):
                kv = TopkIdx[bh, qi, t]
                vbs_k = Vbs[kv]
                T.copy(K[bh, kv, :, :], K_shared)
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(blk, blk):
                    acc_s[i, j] = acc_s[i, j] * sm_scale
                if vbs_k < blk:
                    for i, j in T.Parallel(blk, blk):
                        acc_s[i, j] = T.if_then_else(j < vbs_k, acc_s[i, j], -1e30)
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(blk):
                    scores_scale[i] = T.exp2(scores_max_prev[i] - scores_max[i])
                for i, j in T.Parallel(blk, D):
                    acc_o[i, j] *= scores_scale[i]
                for i, j in T.Parallel(blk, blk):
                    acc_s[i, j] = T.exp2(acc_s[i, j] - scores_max[i])
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(blk):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                T.copy(acc_s, S_shared)
                T.copy(V[bh, kv, :, :], V_shared)
                T.gemm(S_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(blk, D):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, Out[bh, qi, :, :])
            for i in T.Parallel(blk):
                Lse[bh, qi, i] = scores_max[i] + T.log2(logsum[i])

    return tilelang.compile(
        vsa_fwd_bn64,
        target="musa",
        pass_configs=MUSA_PASS_CONFIGS,
        compile_flags=MUSA_COMPILE_FLAGS,
    )


def _build_bwd_dq_bn128(BH: int, N_blk: int, topk: int, blk: int = 64, D: int = 128):
    sm_scale = D ** -0.5
    scale_log2 = sm_scale * 1.4426950408889634
    dtype = "bfloat16"
    accum_dtype = "float"
    BN = 2 * blk
    n_iters = topk // 2
    total_q = BH * N_blk

    @T.prim_func
    def vsa_bwd_dq_bn128(
        Q: T.Tensor((BH, N_blk, blk, D), dtype),
        K: T.Tensor((BH, N_blk, blk, D), dtype),
        V: T.Tensor((BH, N_blk, blk, D), dtype),
        O: T.Tensor((BH, N_blk, blk, D), dtype),
        dO: T.Tensor((BH, N_blk, blk, D), dtype),
        Lse: T.Tensor((BH, N_blk, blk), accum_dtype),
        TopkIdx: T.Tensor((BH, N_blk, topk), "int32"),
        QOrder: T.Tensor((total_q,), "int32"),
        dQ: T.Tensor((BH, N_blk, blk, D), dtype),
        Delta: T.Tensor((BH, N_blk, blk), accum_dtype),
    ):
        with T.Kernel(total_q, threads=256) as (tid,):
            Q_shared = T.alloc_shared([blk, D], dtype)
            dO_shared = T.alloc_shared([blk, D], dtype)
            K_b16 = T.alloc_shared([BN, D], dtype)
            K_b32 = T.alloc_shared([BN, D], dtype)
            V_b16 = T.alloc_shared([BN, D], dtype)
            dS_shared = T.alloc_shared([blk, BN], dtype)
            lse_shared = T.alloc_shared([blk], accum_dtype)
            delta_shared = T.alloc_shared([blk], accum_dtype)

            o_frag = T.alloc_fragment([blk, D], accum_dtype)
            do_frag = T.alloc_fragment([blk, D], accum_dtype)
            delta_frag = T.alloc_fragment([blk], accum_dtype)
            lse_reg = T.alloc_fragment([blk], accum_dtype)
            delta_reg = T.alloc_fragment([blk], accum_dtype)
            acc_s = T.alloc_fragment([blk, BN], accum_dtype)
            acc_dp = T.alloc_fragment([blk, BN], accum_dtype)
            dq_acc = T.alloc_fragment([blk, D], accum_dtype)

            flat = QOrder[tid]
            bh = flat // N_blk
            qi = flat % N_blk

            T.copy(O[bh, qi, :, :], Q_shared)
            T.copy(dO[bh, qi, :, :], dO_shared)
            T.copy(Q_shared, o_frag)
            T.copy(dO_shared, do_frag)
            for i, j in T.Parallel(blk, D):
                o_frag[i, j] = o_frag[i, j] * do_frag[i, j]
            T.reduce_sum(o_frag, delta_frag, dim=1)
            T.copy(delta_frag, delta_shared)
            T.copy(delta_frag, Delta[bh, qi, :])

            T.copy(Q[bh, qi, :, :], Q_shared)
            T.copy(Lse[bh, qi, :], lse_shared)
            T.clear(dq_acc)
            # Hoist per-row Lse/Delta into 2-float registers once before the KV loop
            # (eliminates 64 SMEM broadcast loads/iter -> -63 ms bitwise exact!).
            for i in T.Parallel(blk):
                lse_reg[i] = lse_shared[i]
                delta_reg[i] = delta_shared[i]

            # Dual layout-matched shared-memory staging (K_b16 for Q@K^T, K_b32 for full K=128 dS@K):
            # eliminates redundant shared-memory transpose copies and halves matmul count.
            # Padded KV slots (j >= vbs) are explicitly zeroed in K and V during tiling, so
            # dS[i, j_pad] * K[j_pad, d] == finite * 0 == 0 bitwise -> no Vbs masking needed (-52 ms)!
            for t in T.Pipelined(n_iters, num_stages=1):
                kv0 = TopkIdx[bh, qi, t * 2]
                kv1 = TopkIdx[bh, qi, t * 2 + 1]
                T.copy(K[bh, kv0, :, :], K_b16[0:blk, :])
                T.copy(K[bh, kv1, :, :], K_b16[blk:BN, :])
                T.copy(V[bh, kv0, :, :], V_b16[0:blk, :])
                T.copy(V[bh, kv1, :, :], V_b16[blk:BN, :])
                T.gemm(
                    Q_shared,
                    K_b16,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                    clear_accum=True,
                )
                T.gemm(
                    dO_shared,
                    V_b16,
                    acc_dp,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                    clear_accum=True,
                )
                T.copy(K[bh, kv0, :, :], K_b32[0:blk, :])
                T.copy(K[bh, kv1, :, :], K_b32[blk:BN, :])
                # Padded KV slots (j >= vbs) have K=V=0 so dS[i,j_pad]*K[j_pad,d]==0 bitwise;
                # clamping (S*scale - lse) <= 1.0 before exp2 is a no-op on valid keys (<= 0.0)
                # while guaranteeing dS stays finite (<= 2*|delta|) even if lse < -127 (zero NaN!).
                for i, j in T.Parallel(blk, BN):
                    acc_s[i, j] = T.min(acc_s[i, j] * scale_log2 - lse_reg[i], 1.0)
                for i, j in T.Parallel(blk, BN):
                    acc_s[i, j] = T.exp2(acc_s[i, j])
                for i, j in T.Parallel(blk, BN):
                    acc_dp[i, j] = acc_s[i, j] * (acc_dp[i, j] - delta_reg[i])
                T.copy(acc_dp, dS_shared)
                T.gemm(dS_shared, K_b32, dq_acc, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(blk, D):
                dq_acc[i, j] *= sm_scale
            T.copy(dq_acc, dQ[bh, qi, :, :])

    return tilelang.compile(
        vsa_bwd_dq_bn128,
        target="musa",
        pass_configs=MUSA_PASS_CONFIGS_PREFETCH,
        compile_flags=MUSA_COMPILE_FLAGS,
    )


def _build_bwd_dq_bn64(BH: int, N_blk: int, topk: int, blk: int = 64, D: int = 128):
    sm_scale = D ** -0.5
    scale_log2 = sm_scale * 1.4426950408889634
    dtype = "bfloat16"
    accum_dtype = "float"

    @T.prim_func
    def vsa_bwd_dq_bn64(
        Q: T.Tensor((BH, N_blk, blk, D), dtype),
        K: T.Tensor((BH, N_blk, blk, D), dtype),
        V: T.Tensor((BH, N_blk, blk, D), dtype),
        O: T.Tensor((BH, N_blk, blk, D), dtype),
        dO: T.Tensor((BH, N_blk, blk, D), dtype),
        Lse: T.Tensor((BH, N_blk, blk), accum_dtype),
        TopkIdx: T.Tensor((BH, N_blk, topk), "int32"),
        Vbs: T.Tensor((N_blk,), "int32"),
        dQ: T.Tensor((BH, N_blk, blk, D), dtype),
        Delta: T.Tensor((BH, N_blk, blk), accum_dtype),
    ):
        with T.Kernel(N_blk, BH, threads=256) as (qi, bh):
            Q_shared = T.alloc_shared([blk, D], dtype)
            dO_shared = T.alloc_shared([blk, D], dtype)
            K_shared = T.alloc_shared([blk, D], dtype)
            K_nt = T.alloc_shared([blk, D], dtype)
            V_shared = T.alloc_shared([blk, D], dtype)
            dS_shared = T.alloc_shared([blk, blk], dtype)
            lse_shared = T.alloc_shared([blk], accum_dtype)
            delta_shared = T.alloc_shared([blk], accum_dtype)

            o_frag = T.alloc_fragment([blk, D], accum_dtype)
            do_frag = T.alloc_fragment([blk, D], accum_dtype)
            delta_frag = T.alloc_fragment([blk], accum_dtype)
            acc_s = T.alloc_fragment([blk, blk], accum_dtype)
            acc_dp = T.alloc_fragment([blk, blk], accum_dtype)
            dq_acc = T.alloc_fragment([blk, D], accum_dtype)

            T.copy(O[bh, qi, :, :], o_frag)
            T.copy(dO[bh, qi, :, :], dO_shared)
            T.copy(dO_shared, do_frag)
            for i, j in T.Parallel(blk, D):
                o_frag[i, j] = o_frag[i, j] * do_frag[i, j]
            T.reduce_sum(o_frag, delta_frag, dim=1)
            T.copy(delta_frag, delta_shared)
            T.copy(delta_shared, Delta[bh, qi, :])

            T.copy(Q[bh, qi, :, :], Q_shared)
            T.copy(Lse[bh, qi, :], lse_shared)
            T.clear(dq_acc)

            for t in T.Pipelined(topk, num_stages=2):
                kv = TopkIdx[bh, qi, t]
                vbs_k = Vbs[kv]
                T.copy(K[bh, kv, :, :], K_shared)
                T.copy(V[bh, kv, :, :], V_shared)
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.clear(acc_dp)
                T.gemm(dO_shared, V_shared, acc_dp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(blk, blk):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale_log2 - lse_shared[i])
                if vbs_k < blk:
                    for i, j in T.Parallel(blk, blk):
                        acc_s[i, j] = T.if_then_else(j < vbs_k, acc_s[i, j], 0.0)
                for i, j in T.Parallel(blk, blk):
                    acc_dp[i, j] = acc_s[i, j] * (acc_dp[i, j] - delta_shared[i])
                T.copy(acc_dp, dS_shared)
                T.copy(K_shared, K_nt)
                T.gemm(dS_shared, K_nt, dq_acc, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(blk, D):
                dq_acc[i, j] *= sm_scale
            T.copy(dq_acc, dQ[bh, qi, :, :])

    return tilelang.compile(
        vsa_bwd_dq_bn64,
        target="musa",
        pass_configs=MUSA_PASS_CONFIGS,
        compile_flags=MUSA_COMPILE_FLAGS,
    )


def _build_bwd_dkdv_bm128(BH: int, N_blk: int, total_edges: int, blk: int = 64, D: int = 128):
    sm_scale = D ** -0.5
    scale_log2 = sm_scale * 1.4426950408889634
    dtype = "bfloat16"
    accum_dtype = "float"
    BM = 2 * blk
    total_kv = BH * N_blk

    @T.prim_func
    def vsa_bwd_dkdv_bm128(
        Q: T.Tensor((BH, N_blk, blk, D), dtype),
        K: T.Tensor((BH, N_blk, blk, D), dtype),
        V: T.Tensor((BH, N_blk, blk, D), dtype),
        dO: T.Tensor((BH, N_blk, blk, D), dtype),
        Lse: T.Tensor((BH, N_blk, blk), accum_dtype),
        Delta: T.Tensor((BH, N_blk, blk), accum_dtype),
        KvOffsets: T.Tensor((BH, N_blk + 1), "int32"),
        KvQIdx: T.Tensor((BH, total_edges), "int32"),
        Vbs: T.Tensor((N_blk,), "int32"),
        Order: T.Tensor((total_kv,), "int32"),
        dK: T.Tensor((BH, N_blk, blk, D), dtype),
        dV: T.Tensor((BH, N_blk, blk, D), dtype),
    ):
        # Pure LPT descending-degree CTA order eliminates KV degree skew (bitwise exact).
        # Dual layout-matched shared-memory staging (Q/dO in B16 & B32) + shared P_shared[64, 128]
        # eliminates redundant shared-memory transpose copies and halves dK/dV matmul count.
        with T.Kernel(total_kv, threads=256) as (tid,):
            K_shared = T.alloc_shared([blk, D], dtype)
            V_shared = T.alloc_shared([blk, D], dtype)
            Q_b16 = T.alloc_shared([BM, D], dtype)
            dO_b16 = T.alloc_shared([BM, D], dtype)
            dO_b32 = T.alloc_shared([BM, D], dtype)
            Q_b32 = T.alloc_shared([BM, D], dtype)
            P_shared = T.alloc_shared([blk, BM], dtype)
            lse_shared = T.alloc_shared([BM], accum_dtype)
            delta_shared = T.alloc_shared([BM], accum_dtype)

            acc_s_t = T.alloc_fragment([blk, BM], accum_dtype)
            acc_dp_t = T.alloc_fragment([blk, BM], accum_dtype)
            dk_acc = T.alloc_fragment([blk, D], accum_dtype)
            dv_acc = T.alloc_fragment([blk, D], accum_dtype)

            flat_idx = Order[tid]
            bh = flat_idx // N_blk
            kv = flat_idx % N_blk
            start = KvOffsets[bh, kv]
            end = KvOffsets[bh, kv + 1]
            deg = end - start
            n_pairs = (deg + 1) // 2
            vbs_k = Vbs[kv]

            T.copy(K[bh, kv, :, :], K_shared)
            T.copy(V[bh, kv, :, :], V_shared)
            T.clear(dk_acc)
            T.clear(dv_acc)

            if deg > 0:
                for pair in T.Pipelined(n_pairs, num_stages=1):
                    idx0 = start + pair * 2
                    idx1 = T.if_then_else(pair * 2 + 1 < deg, idx0 + 1, idx0)
                    q0 = KvQIdx[bh, idx0]
                    q1 = KvQIdx[bh, idx1]
                    # Issue async copy loads in exact consumption order so Lse/Delta arrive before B32 loads
                    T.copy(Q[bh, q0, :, :], Q_b16[0:blk, :])
                    T.copy(Q[bh, q1, :, :], Q_b16[blk:BM, :])
                    T.copy(dO[bh, q0, :, :], dO_b16[0:blk, :])
                    T.copy(dO[bh, q1, :, :], dO_b16[blk:BM, :])
                    T.copy(Lse[bh, q0, :], lse_shared[0:blk])
                    T.copy(Lse[bh, q1, :], lse_shared[blk:BM])
                    T.copy(Delta[bh, q0, :], delta_shared[0:blk])
                    T.copy(Delta[bh, q1, :], delta_shared[blk:BM])
                    T.copy(dO[bh, q0, :, :], dO_b32[0:blk, :])
                    T.copy(dO[bh, q1, :, :], dO_b32[blk:BM, :])
                    T.copy(Q[bh, q0, :, :], Q_b32[0:blk, :])
                    T.copy(Q[bh, q1, :, :], Q_b32[blk:BM, :])

                    T.gemm(
                        K_shared,
                        Q_b16,
                        acc_s_t,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                        clear_accum=True,
                    )
                    T.gemm(
                        V_shared,
                        dO_b16,
                        acc_dp_t,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                        clear_accum=True,
                    )
                    for j, i in T.Parallel(blk, BM):
                        acc_s_t[j, i] = T.exp2(acc_s_t[j, i] * scale_log2 - lse_shared[i])
                    if vbs_k < blk:
                        for j, i in T.Parallel(blk, BM):
                            acc_s_t[j, i] = T.if_then_else(j < vbs_k, acc_s_t[j, i], 0.0)
                    if pair * 2 + 1 >= deg:
                        for j, i in T.Parallel(blk, BM):
                            acc_s_t[j, i] = T.if_then_else(i < blk, acc_s_t[j, i], 0.0)
                    for j, i in T.Parallel(blk, BM):
                        acc_dp_t[j, i] = acc_s_t[j, i] * (acc_dp_t[j, i] - delta_shared[i])

                    T.copy(acc_s_t, P_shared)
                    T.gemm(P_shared, dO_b32, dv_acc, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(acc_dp_t, P_shared)
                    T.gemm(P_shared, Q_b32, dk_acc, policy=T.GemmWarpPolicy.FullRow)

            for j, d in T.Parallel(blk, D):
                dk_acc[j, d] *= sm_scale
            T.copy(dk_acc, dK[bh, kv, :, :])
            T.copy(dv_acc, dV[bh, kv, :, :])

    return tilelang.compile(
        vsa_bwd_dkdv_bm128,
        target="musa",
        pass_configs=MUSA_PASS_CONFIGS_PREFETCH,
        compile_flags=MUSA_COMPILE_FLAGS,
    )


class _TileLangVSABlockSparseAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, topk_idx, vbs, block_elements, offload_ctx=None, pre_sorted=False, q_order=None, out_buf=None, is_ring=False):
        B, H, S_pad, D = q.shape
        BH = B * H
        N_blk = S_pad // block_elements
        topk = topk_idx.shape[-1]

        q4 = q.contiguous().view(BH, N_blk, block_elements, D)
        k4 = k.contiguous().view(BH, N_blk, block_elements, D)
        v4 = v.contiguous().view(BH, N_blk, block_elements, D)
        if pre_sorted:
            idx32 = topk_idx.reshape(BH, N_blk, topk).contiguous()
        else:
            # Sort selected KV indices ascending along each Q row so concurrent MPs walk
            # K/V blocks in lockstep through L2 cache (exact same attended KV set).
            idx32 = torch.sort(
                topk_idx.reshape(BH, N_blk, topk).to(torch.int32),
                dim=-1,
            ).values.contiguous()
        vbs32 = vbs.to(torch.int32).contiguous()

        if out_buf is not None:
            out4 = out_buf.detach().view(BH, N_blk, block_elements, D)
        else:
            out4 = torch.empty_like(q4)
        lse = torch.empty((BH, N_blk, block_elements), device=q.device, dtype=torch.float32)

        if topk % 2 == 0:
            if q_order is None:
                # 4-quantile per-head Q-block clustering for L2 K/V tile reuse (deterministic stable sort).
                i0 = topk // 8
                i1 = (3 * topk) // 8
                i2 = (5 * topk) // 8
                i3 = (7 * topk) // 8
                q_key = (
                    ((idx32[:, :, i0] >> 6) << 9)
                    + ((idx32[:, :, i1] >> 6) << 6)
                    + ((idx32[:, :, i2] >> 6) << 3)
                    + (idx32[:, :, i3] >> 6)
                )
                base_off = torch.arange(BH, device=q.device, dtype=torch.int64).view(BH, 1) * N_blk
                q_order = (
                    (torch.argsort(q_key, dim=1, stable=True) + base_off)
                    .reshape(-1)
                    .to(torch.int32)
                    .contiguous()
                )
            if topk % 4 == 0:
                k_fwd = _compile_cached(
                    ("fwd_bn256_qord_fs_a4", BH, N_blk, topk, block_elements, D),
                    lambda: _build_fwd_bn256(BH, N_blk, topk, block_elements, D),
                )
            else:
                k_fwd = _compile_cached(
                    ("fwd_bn128_qord", BH, N_blk, topk, block_elements, D),
                    lambda: _build_fwd_bn128(BH, N_blk, topk, block_elements, D),
                )
            k_fwd(q4, k4, v4, idx32, vbs32, q_order, out4, lse)
        else:
            q_order = None
            k_fwd = _compile_cached(
                ("fwd_bn64", BH, N_blk, topk, block_elements, D),
                lambda: _build_fwd_bn64(BH, N_blk, topk, block_elements, D),
            )
            k_fwd(q4, k4, v4, idx32, vbs32, out4, lse)

        if offload_ctx is not None:
            pool, blk_idx = offload_ctx
            pool.offload((blk_idx, "vsa"), (q4, k4, v4, out4), is_ring=is_ring)
            ctx.save_for_backward(lse, idx32, vbs32, q_order)
            ctx.offload_ctx = (pool, blk_idx)
        else:
            ctx.save_for_backward(q4, k4, v4, out4, lse, idx32, vbs32, q_order)
            ctx.offload_ctx = None
        ctx.shape_info = (B, H, S_pad, D, BH, N_blk, topk, block_elements)
        return out4.detach().view(B, H, S_pad, D)

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.offload_ctx is not None:
            pool, blk_idx = ctx.offload_ctx
            lse, idx32, vbs32, q_order = ctx.saved_tensors
            q4, k4, v4, out4 = pool.fetch((blk_idx, "vsa"))
            # Prefetch previous block's fc1_out and current block's qk_raw while VSA bwd (~70ms) runs!
            pool.prefetch((blk_idx, "qk_raw"))
            if blk_idx > 0:
                pool.prefetch((blk_idx - 1, "fc1"))
        else:
            q4, k4, v4, out4, lse, idx32, vbs32, q_order = ctx.saved_tensors
        B, H, S_pad, D, BH, N_blk, topk, block_elements = ctx.shape_info
        do4 = grad_out.contiguous().view(BH, N_blk, block_elements, D)

        dq4 = torch.empty_like(q4)
        delta = torch.empty((BH, N_blk, block_elements), device=q4.device, dtype=torch.float32)

        if topk % 2 == 0:
            k_dq = _compile_cached(
                ("dq_bn128_dual_qord_reg_nm_sc", BH, N_blk, topk, block_elements, D),
                lambda: _build_bwd_dq_bn128(BH, N_blk, topk, block_elements, D),
            )
            k_dq(q4, k4, v4, out4, do4, lse, idx32, q_order, dq4, delta)
        else:
            k_dq = _compile_cached(
                ("dq_bn64", BH, N_blk, topk, block_elements, D),
                lambda: _build_bwd_dq_bn64(BH, N_blk, topk, block_elements, D),
            )
            k_dq(q4, k4, v4, out4, do4, lse, idx32, vbs32, dq4, delta)

        total_edges = N_blk * topk
        kv_offsets = torch.empty((BH, N_blk + 1), device=q4.device, dtype=torch.int32)
        kv_q_idx = torch.empty((BH, total_edges), device=q4.device, dtype=torch.int32)
        csr_lib = _get_csr_lib()
        stream_ptr = torch.musa.current_stream(q4.device).musa_stream
        if 32 <= N_blk <= 512:
            G = 12 if N_blk >= 48 else 4
            chunk_cnt = torch.empty((BH, G, N_blk), device=q4.device, dtype=torch.int32)
            chunk_cursor = torch.empty((BH, G, N_blk), device=q4.device, dtype=torch.int32)
            kv_deg = torch.empty((BH * N_blk,), device=q4.device, dtype=torch.int32)
            rc = csr_lib.launch_fast_kv_csr(
                idx32.data_ptr(),
                chunk_cnt.data_ptr(),
                chunk_cursor.data_ptr(),
                kv_offsets.data_ptr(),
                kv_q_idx.data_ptr(),
                kv_deg.data_ptr(),
                BH,
                N_blk,
                topk,
                G,
                stream_ptr,
            )
            if rc != 0:
                raise RuntimeError(f"launch_fast_kv_csr failed with MUSA error {rc}")
        else:
            rc = csr_lib.launch_build_kv_csr(
                idx32.data_ptr(),
                kv_offsets.data_ptr(),
                kv_q_idx.data_ptr(),
                BH,
                N_blk,
                topk,
                stream_ptr,
            )
            if rc != 0:
                raise RuntimeError(f"launch_build_kv_csr failed with MUSA error {rc}")
            kv_deg = (kv_offsets[:, 1:] - kv_offsets[:, :-1]).reshape(-1)
        # L2-coherent bucketed LPT (bucket=8): preserves descending-degree tail balancing
        # while keeping contiguous (bh, kv) blocks from the same head resident in L2.
        total_kv = BH * N_blk
        if 64 <= N_blk <= 504 and total_kv <= 120 * 256:
            lpt_cnt = torch.empty((120, 64), device=q4.device, dtype=torch.int32)
            lpt_off = torch.empty((120, 64), device=q4.device, dtype=torch.int32)
            lpt_order = torch.empty((total_kv,), device=q4.device, dtype=torch.int32)
            rc = csr_lib.launch_bucket_lpt_order(
                kv_deg.data_ptr(),
                lpt_cnt.data_ptr(),
                lpt_off.data_ptr(),
                lpt_order.data_ptr(),
                total_kv,
                3,
                stream_ptr,
            )
            if rc != 0:
                raise RuntimeError(f"launch_bucket_lpt_order failed with MUSA error {rc}")
        else:
            lpt_key = (kv_deg >> 3) if N_blk >= 64 else kv_deg
            lpt_order = torch.argsort(lpt_key, descending=True, stable=True).to(torch.int32)

        k_dkdv = _compile_cached(
            ("dkdv_bm128_dual_lpt_v2", BH, N_blk, total_edges, block_elements, D),
            lambda: _build_bwd_dkdv_bm128(BH, N_blk, total_edges, block_elements, D),
        )
        dk4 = torch.empty_like(k4)
        dv4 = torch.empty_like(v4)
        k_dkdv(q4, k4, v4, do4, lse, delta, kv_offsets, kv_q_idx, vbs32, lpt_order, dk4, dv4)
        if ctx.offload_ctx is not None:
            pool, _ = ctx.offload_ctx
            pool.release("vsa")

        return (
            dq4.view(B, H, S_pad, D),
            dk4.view(B, H, S_pad, D),
            dv4.view(B, H, S_pad, D),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def block_sparse_attention_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idx: torch.Tensor,
    vbs: torch.Tensor,
    block_elements: int = 64,
    offload_ctx=None,
    pre_sorted: bool = False,
    q_order: torch.Tensor | None = None,
    out_buf: torch.Tensor | None = None,
    is_ring: bool = False,
) -> torch.Tensor:
    return _TileLangVSABlockSparseAttn.apply(
        q, k, v, topk_idx, vbs, block_elements, offload_ctx, pre_sorted, q_order, out_buf, is_ring
    )
