"""Custom MUSA C++ fused kernels for Wan2.2-TI2V-5B + VSA training on MTT S5000.

All kernels are written from scratch for Moore Threads MTT S5000
and are 100% bitwise deterministic (zero global atomicAdd; deterministic split-M reduction
for parameter/modulation gradients).

Implementation note (mcc/LLVM SROA):
- Zero local arrays and zero unions: all per-thread data is held in pure scalar registers
  (384 threads x 8 BF16 elements = 3072 hidden dim, 1x uint4 per thread) to guarantee
  zero local-memory spill and ~1.3-1.5 TB/s HBM bandwidth.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.distributed as dist

_FUSED_CU_SOURCE = r"""
#include <musa_runtime.h>
#include <musa_bf16.h>
#include <math.h>
#include <stdint.h>

// Pure scalar BF16 <-> FP32 pack/unpack helpers (zero unions, zero arrays -> zero LLVM spill!)
static __device__ __forceinline__ float bf16_lo_to_f32(uint32_t w) {
    return __uint_as_float((w & 0xffffu) << 16);
}
static __device__ __forceinline__ float bf16_hi_to_f32(uint32_t w) {
    return __uint_as_float(w & 0xffff0000u);
}
static __device__ __forceinline__ uint32_t f32x2_to_bf16x2(float a, float b) {
    __mt_bfloat16 ba = __float2bfloat16(a);
    __mt_bfloat16 bb = __float2bfloat16(b);
    uint16_t ua = *reinterpret_cast<const uint16_t*>(&ba);
    uint16_t ub = *reinterpret_cast<const uint16_t*>(&bb);
    return (uint32_t)ua | ((uint32_t)ub << 16);
}
static __device__ __forceinline__ float round_bf16_f32(float a) {
    __mt_bfloat16 ba = __float2bfloat16(a);
    return __bfloat162float(ba);
}

static __device__ __forceinline__ float warp_reduce_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}

// 12-warp (384-thread) block sum reduction, race-free on back-to-back calls
static __device__ __forceinline__ float block384_reduce_sum(float v, float* smem16) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) smem16[warp] = v;
    __syncthreads();
    float res = (smem16[0] + smem16[1] + smem16[2] + smem16[3])
              + (smem16[4] + smem16[5] + smem16[6] + smem16[7])
              + (smem16[8] + smem16[9] + smem16[10] + smem16[11]);
    __syncthreads();
    return res;
}

// Fused 2-value 12-warp block sum reduction (halves __syncthreads() barriers in LayerNorm/AdaLN bwd!)
static __device__ __forceinline__ void block384_reduce_sum2(
    float v1, float v2, float* smem32, float& o1, float& o2
) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    v1 = warp_reduce_sum(v1);
    v2 = warp_reduce_sum(v2);
    if (lane == 0) {
        smem32[warp] = v1;
        smem32[16 + warp] = v2;
    }
    __syncthreads();
    o1 = (smem32[0] + smem32[1] + smem32[2] + smem32[3])
       + (smem32[4] + smem32[5] + smem32[6] + smem32[7])
       + (smem32[8] + smem32[9] + smem32[10] + smem32[11]);
    o2 = (smem32[16] + smem32[17] + smem32[18] + smem32[19])
       + (smem32[20] + smem32[21] + smem32[22] + smem32[23])
       + (smem32[24] + smem32[25] + smem32[26] + smem32[27]);
    __syncthreads();
}

// Deterministic reduction of partial[B, P, D] -> out[B, D] (or [D] summed over B when sum_batch=1)
__global__ void reduce_partial_f32_kernel(
    const float* __restrict__ partial,
    float* __restrict__ out,
    int32_t B,
    int32_t P,
    int32_t D,
    int32_t sum_batch
) {
    __shared__ float smem[8][32];
    int32_t tx = threadIdx.x;
    int32_t ty = threadIdx.y;
    int32_t col = blockIdx.x * 32 + tx;
    float acc = 0.0f;
    if (col < D) {
        if (sum_batch) {
            for (int32_t b = 0; b < B; ++b) {
                const float* base = partial + (int64_t)b * P * D + col;
                for (int32_t p = ty; p < P; p += 8) {
                    acc += base[(int64_t)p * D];
                }
            }
        } else {
            int32_t b = blockIdx.y;
            const float* base = partial + (int64_t)b * P * D + col;
            for (int32_t p = ty; p < P; p += 8) {
                acc += base[(int64_t)p * D];
            }
        }
    }
    smem[ty][tx] = acc;
    __syncthreads();
    if (ty == 0 && col < D) {
        float s = smem[0][tx] + smem[1][tx] + smem[2][tx] + smem[3][tx]
                + smem[4][tx] + smem[5][tx] + smem[6][tx] + smem[7][tx];
        if (sum_batch) out[col] = s;
        else out[(int64_t)blockIdx.y * D + col] = s;
    }
}

// Deterministic reduction of partial[B, P, D] (FP32) summed over B & P -> BF16 out[D]
__global__ void reduce_partial_f32_to_bf16_kernel(
    const float* __restrict__ partial,
    __mt_bfloat16* __restrict__ out,
    int32_t B,
    int32_t P,
    int32_t D
) {
    __shared__ float smem[8][32];
    int32_t tx = threadIdx.x;
    int32_t ty = threadIdx.y;
    int32_t col = blockIdx.x * 32 + tx;
    float acc = 0.0f;
    if (col < D) {
        for (int32_t b = 0; b < B; ++b) {
            const float* base = partial + (int64_t)b * P * D + col;
            for (int32_t p = ty; p < P; p += 8) {
                acc += base[(int64_t)p * D];
            }
        }
    }
    smem[ty][tx] = acc;
    __syncthreads();
    if (ty == 0 && col < D) {
        float s = smem[0][tx] + smem[1][tx] + smem[2][tx] + smem[3][tx]
                + smem[4][tx] + smem[5][tx] + smem[6][tx] + smem[7][tx];
        out[col] = __float2bfloat16(s);
    }
}

// Pure-scalar vectorized GELU-tanh forward (zero dynamic alloc when writing into static scratch)
static __device__ __forceinline__ float gelu_tanh_fwd_elem(float x) {
    constexpr float kBeta = 0.7978845608028654f;
    constexpr float kKappa = 0.044715f;
    float inner = kBeta * (x + kKappa * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void fused_gelu_tanh_fwd_u4_kernel(
    const uint4* __restrict__ x,
    uint4* __restrict__ y,
    int64_t n_u4
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n_u4; idx += stride) {
        uint4 v = x[idx];
        uint4 o;
        o.x = f32x2_to_bf16x2(gelu_tanh_fwd_elem(bf16_lo_to_f32(v.x)), gelu_tanh_fwd_elem(bf16_hi_to_f32(v.x)));
        o.y = f32x2_to_bf16x2(gelu_tanh_fwd_elem(bf16_lo_to_f32(v.y)), gelu_tanh_fwd_elem(bf16_hi_to_f32(v.y)));
        o.z = f32x2_to_bf16x2(gelu_tanh_fwd_elem(bf16_lo_to_f32(v.z)), gelu_tanh_fwd_elem(bf16_hi_to_f32(v.z)));
        o.w = f32x2_to_bf16x2(gelu_tanh_fwd_elem(bf16_lo_to_f32(v.w)), gelu_tanh_fwd_elem(bf16_hi_to_f32(v.w)));
        y[idx] = o;
    }
}

// Fused GELU-tanh backward + deterministic split-M FP32 bias-grad partial accumulation (D % 8 == 0)
static __device__ __forceinline__ float gelu_tanh_bwd_elem(float dy, float x) {
    constexpr float kBeta = 0.7978845608028654f;
    constexpr float kKappa = 0.044715f;
    float x2 = x * x;
    float inner = kBeta * (x + kKappa * x2 * x);
    float t = tanhf(inner);
    float sech2 = 1.0f - t * t;
    float d_inner = kBeta * (1.0f + 3.0f * kKappa * x2);
    return dy * (0.5f * ((1.0f + t) + x * sech2 * d_inner));
}

__global__ void fused_gelu_tanh_bwd_bias_s1(
    const __mt_bfloat16* __restrict__ grad_out,
    const __mt_bfloat16* __restrict__ x,
    __mt_bfloat16* __restrict__ dx,
    float* __restrict__ partial_db,
    int32_t M,
    int32_t D,
    int32_t split_m
) {
    __shared__ float smem[8][256];
    int32_t tx = threadIdx.x;
    int32_t ty = threadIdx.y;
    int32_t col0 = blockIdx.x * 256 + tx * 8;
    int32_t mb = blockIdx.y;
    int32_t r0 = (int32_t)((int64_t)M * mb / split_m);
    int32_t r1 = (int32_t)((int64_t)M * (mb + 1) / split_m);

    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f;
    if (col0 < D) {
        const uint4* dy_u4 = (const uint4*)grad_out;
        const uint4* x_u4 = (const uint4*)x;
        uint4* dx_u4 = (uint4*)dx;
        int64_t stride_u4 = D >> 3;
        int64_t col_u4 = col0 >> 3;
        for (int32_t r = r0 + ty; r < r1; r += 8) {
            int64_t idx = (int64_t)r * stride_u4 + col_u4;
            uint4 gv = dy_u4[idx];
            uint4 xv = x_u4[idx];
            float d0 = gelu_tanh_bwd_elem(bf16_lo_to_f32(gv.x), bf16_lo_to_f32(xv.x));
            float d1 = gelu_tanh_bwd_elem(bf16_hi_to_f32(gv.x), bf16_hi_to_f32(xv.x));
            float d2 = gelu_tanh_bwd_elem(bf16_lo_to_f32(gv.y), bf16_lo_to_f32(xv.y));
            float d3 = gelu_tanh_bwd_elem(bf16_hi_to_f32(gv.y), bf16_hi_to_f32(xv.y));
            float d4 = gelu_tanh_bwd_elem(bf16_lo_to_f32(gv.z), bf16_lo_to_f32(xv.z));
            float d5 = gelu_tanh_bwd_elem(bf16_hi_to_f32(gv.z), bf16_hi_to_f32(xv.z));
            float d6 = gelu_tanh_bwd_elem(bf16_lo_to_f32(gv.w), bf16_lo_to_f32(xv.w));
            float d7 = gelu_tanh_bwd_elem(bf16_hi_to_f32(gv.w), bf16_hi_to_f32(xv.w));
            uint4 ov;
            ov.x = f32x2_to_bf16x2(d0, d1);
            ov.y = f32x2_to_bf16x2(d2, d3);
            ov.z = f32x2_to_bf16x2(d4, d5);
            ov.w = f32x2_to_bf16x2(d6, d7);
            dx_u4[idx] = ov;
            if (partial_db != nullptr) {
                a0 += bf16_lo_to_f32(ov.x); a1 += bf16_hi_to_f32(ov.x);
                a2 += bf16_lo_to_f32(ov.y); a3 += bf16_hi_to_f32(ov.y);
                a4 += bf16_lo_to_f32(ov.z); a5 += bf16_hi_to_f32(ov.z);
                a6 += bf16_lo_to_f32(ov.w); a7 += bf16_hi_to_f32(ov.w);
            }
        }
    }
    if (partial_db != nullptr) {
        int32_t c_loc = tx * 8;
        smem[ty][c_loc + 0] = a0; smem[ty][c_loc + 1] = a1;
        smem[ty][c_loc + 2] = a2; smem[ty][c_loc + 3] = a3;
        smem[ty][c_loc + 4] = a4; smem[ty][c_loc + 5] = a5;
        smem[ty][c_loc + 6] = a6; smem[ty][c_loc + 7] = a7;
        __syncthreads();
        int32_t lin = ty * 32 + tx;
        int32_t gcol = blockIdx.x * 256 + lin;
        if (gcol < D) {
            float s = smem[0][lin] + smem[1][lin] + smem[2][lin] + smem[3][lin]
                    + smem[4][lin] + smem[5][lin] + smem[6][lin] + smem[7][lin];
            partial_db[(int64_t)mb * D + gcol] = s;
        }
    }
}

// Fast deterministic split-M BF16 bias-grad sum over dim 0 (D % 8 == 0).
__global__ void sum_dim0_bf16_fast_s1(
    const __mt_bfloat16* __restrict__ x,
    float* __restrict__ partial,
    int32_t M,
    int32_t D,
    int32_t split_m
) {
    __shared__ float smem[8][256];
    int32_t tx = threadIdx.x;
    int32_t ty = threadIdx.y;
    int32_t col0 = blockIdx.x * 256 + tx * 8;
    int32_t mb = blockIdx.y;
    int32_t r0 = (int32_t)((int64_t)M * mb / split_m);
    int32_t r1 = (int32_t)((int64_t)M * (mb + 1) / split_m);

    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f;
    if (col0 < D) {
        const uint4* x_u4 = (const uint4*)x;
        int64_t stride_u4 = D >> 3;
        int64_t col_u4 = col0 >> 3;
        for (int32_t r = r0 + ty; r < r1; r += 8) {
            uint4 v = x_u4[(int64_t)r * stride_u4 + col_u4];
            a0 += bf16_lo_to_f32(v.x); a1 += bf16_hi_to_f32(v.x);
            a2 += bf16_lo_to_f32(v.y); a3 += bf16_hi_to_f32(v.y);
            a4 += bf16_lo_to_f32(v.z); a5 += bf16_hi_to_f32(v.z);
            a6 += bf16_lo_to_f32(v.w); a7 += bf16_hi_to_f32(v.w);
        }
    }
    int32_t c_loc = tx * 8;
    smem[ty][c_loc + 0] = a0; smem[ty][c_loc + 1] = a1;
    smem[ty][c_loc + 2] = a2; smem[ty][c_loc + 3] = a3;
    smem[ty][c_loc + 4] = a4; smem[ty][c_loc + 5] = a5;
    smem[ty][c_loc + 6] = a6; smem[ty][c_loc + 7] = a7;
    __syncthreads();
    int32_t lin = ty * 32 + tx;
    int32_t gcol = blockIdx.x * 256 + lin;
    if (gcol < D) {
        float s = smem[0][lin] + smem[1][lin] + smem[2][lin] + smem[3][lin]
                + smem[4][lin] + smem[5][lin] + smem[6][lin] + smem[7][lin];
        partial[(int64_t)mb * D + gcol] = s;
    }
}

__global__ void sum_dim0_bf16_fast_s2(
    const float* __restrict__ partial,
    __mt_bfloat16* __restrict__ out,
    int32_t split_m,
    int32_t D
) {
    int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= D) return;
    float s = 0.0f;
    for (int32_t m = 0; m < split_m; ++m) {
        s += partial[(int64_t)m * D + col];
    }
    out[col] = __float2bfloat16(s);
}

constexpr int D_WAN = 3072; // 384 threads * 8 BF16 elements (1x uint4 per thread!)
constexpr float INV_D_WAN = 1.0f / 3072.0f;

// ============================================================================
// 1. Fused AdaLN (1-warp/row zero-syncthreads SMEM fwd + fused-sum2 SMEM bwd!)
// ============================================================================
__global__ void fused_adaln_fwd_d3072(
    const __mt_bfloat16* __restrict__ x,
    const float* __restrict__ scale,
    const float* __restrict__ shift,
    __mt_bfloat16* __restrict__ y,
    float* __restrict__ mean_out,
    float* __restrict__ rstd_out,
    int32_t S,
    int32_t total_rows,
    float eps
) {
    __shared__ uint4 s_row[4][384]; // 24 KB SMEM row cache -> 1.16 TB/s!
    int32_t lane = threadIdx.x & 31;
    int32_t warp = threadIdx.x >> 5;
    int32_t row = blockIdx.x * 4 + warp;
    if (row >= total_rows) return;
    int32_t b = row / S;

    const uint4* xr = reinterpret_cast<const uint4*>(x + (int64_t)row * D_WAN);
    uint4* yr = reinterpret_cast<uint4*>(y + (int64_t)row * D_WAN);
    const float4* scr = reinterpret_cast<const float4*>(scale + (int64_t)b * D_WAN);
    const float4* shr = reinterpret_cast<const float4*>(shift + (int64_t)b * D_WAN);

    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        int idx = k * 32 + lane;
        uint4 v = xr[idx];
        s_row[warp][idx] = v;
        sum += (bf16_lo_to_f32(v.x) + bf16_hi_to_f32(v.x))
             + (bf16_lo_to_f32(v.y) + bf16_hi_to_f32(v.y))
             + (bf16_lo_to_f32(v.z) + bf16_hi_to_f32(v.z))
             + (bf16_lo_to_f32(v.w) + bf16_hi_to_f32(v.w));
    }
    float mean = warp_reduce_sum(sum) * INV_D_WAN;

    float vsum = 0.0f;
    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        uint4 v = s_row[warp][k * 32 + lane];
        float d0 = bf16_lo_to_f32(v.x) - mean, d1 = bf16_hi_to_f32(v.x) - mean;
        float d2 = bf16_lo_to_f32(v.y) - mean, d3 = bf16_hi_to_f32(v.y) - mean;
        float d4 = bf16_lo_to_f32(v.z) - mean, d5 = bf16_hi_to_f32(v.z) - mean;
        float d6 = bf16_lo_to_f32(v.w) - mean, d7 = bf16_hi_to_f32(v.w) - mean;
        vsum += (d0*d0 + d1*d1) + (d2*d2 + d3*d3) + (d4*d4 + d5*d5) + (d6*d6 + d7*d7);
    }
    float rstd = rsqrtf(warp_reduce_sum(vsum) * INV_D_WAN + eps);
    if (lane == 0) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
    }

    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        int idx = k * 32 + lane;
        uint4 v = s_row[warp][idx];
        float4 sc0 = scr[idx * 2], sc1 = scr[idx * 2 + 1];
        float4 sh0 = shr[idx * 2], sh1 = shr[idx * 2 + 1];
        uint4 yv;
        yv.x = f32x2_to_bf16x2(((bf16_lo_to_f32(v.x) - mean) * rstd) * (1.0f + sc0.x) + sh0.x,
                               ((bf16_hi_to_f32(v.x) - mean) * rstd) * (1.0f + sc0.y) + sh0.y);
        yv.y = f32x2_to_bf16x2(((bf16_lo_to_f32(v.y) - mean) * rstd) * (1.0f + sc0.z) + sh0.z,
                               ((bf16_hi_to_f32(v.y) - mean) * rstd) * (1.0f + sc0.w) + sh0.w);
        yv.z = f32x2_to_bf16x2(((bf16_lo_to_f32(v.z) - mean) * rstd) * (1.0f + sc1.x) + sh1.x,
                               ((bf16_hi_to_f32(v.z) - mean) * rstd) * (1.0f + sc1.y) + sh1.y);
        yv.w = f32x2_to_bf16x2(((bf16_lo_to_f32(v.w) - mean) * rstd) * (1.0f + sc1.z) + sh1.z,
                               ((bf16_hi_to_f32(v.w) - mean) * rstd) * (1.0f + sc1.w) + sh1.w);
        yr[idx] = yv;
    }
}

template <bool DO_ADD>
__global__ void fused_adaln_bwd_d3072(
    const __mt_bfloat16* __restrict__ dy,
    const __mt_bfloat16* __restrict__ x,
    const float* __restrict__ scale,
    const float* __restrict__ mean_in,
    const float* __restrict__ rstd_in,
    const __mt_bfloat16* __restrict__ dx_add,
    __mt_bfloat16* __restrict__ dx,
    float* __restrict__ p_dscale,
    float* __restrict__ p_dshift,
    int32_t S,
    int32_t P
) {
    __shared__ float smem[32];
    __shared__ float s_dsc[D_WAN];
    __shared__ float s_dsh[D_WAN];
    int32_t p = blockIdx.x;
    int32_t b = blockIdx.y;
    int32_t tid = threadIdx.x;
    int32_t c0 = tid * 8;
    float4 z = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    *reinterpret_cast<float4*>(s_dsc + c0) = z;
    *reinterpret_cast<float4*>(s_dsc + c0 + 4) = z;
    *reinterpret_cast<float4*>(s_dsh + c0) = z;
    *reinterpret_cast<float4*>(s_dsh + c0 + 4) = z;

    int32_t rows_per_p = (S + P - 1) / P;
    int32_t r_start = p * rows_per_p;
    int32_t r_end = r_start + rows_per_p;
    if (r_end > S) r_end = S;

    const float* sc_b = scale + (int64_t)b * D_WAN + c0;
    float4 sc0 = *reinterpret_cast<const float4*>(sc_b);
    float4 sc1 = *reinterpret_cast<const float4*>(sc_b + 4);
    float s0 = 1.0f + sc0.x, s1 = 1.0f + sc0.y, s2 = 1.0f + sc0.z, s3 = 1.0f + sc0.w;
    float s4 = 1.0f + sc1.x, s5 = 1.0f + sc1.y, s6 = 1.0f + sc1.z, s7 = 1.0f + sc1.w;

    for (int32_t s = r_start; s < r_end; ++s) {
        int64_t row = (int64_t)b * S + s;
        float mean = mean_in[row];
        float rstd = rstd_in[row];
        uint4 xv = reinterpret_cast<const uint4*>(x + row * D_WAN)[tid];
        uint4 gv = reinterpret_cast<const uint4*>(dy + row * D_WAN)[tid];

        float xh0 = (bf16_lo_to_f32(xv.x) - mean) * rstd, xh1 = (bf16_hi_to_f32(xv.x) - mean) * rstd;
        float xh2 = (bf16_lo_to_f32(xv.y) - mean) * rstd, xh3 = (bf16_hi_to_f32(xv.y) - mean) * rstd;
        float xh4 = (bf16_lo_to_f32(xv.z) - mean) * rstd, xh5 = (bf16_hi_to_f32(xv.z) - mean) * rstd;
        float xh6 = (bf16_lo_to_f32(xv.w) - mean) * rstd, xh7 = (bf16_hi_to_f32(xv.w) - mean) * rstd;

        float g0 = bf16_lo_to_f32(gv.x), g1 = bf16_hi_to_f32(gv.x);
        float g2 = bf16_lo_to_f32(gv.y), g3 = bf16_hi_to_f32(gv.y);
        float g4 = bf16_lo_to_f32(gv.z), g5 = bf16_hi_to_f32(gv.z);
        float g6 = bf16_lo_to_f32(gv.w), g7 = bf16_hi_to_f32(gv.w);

        float4 a0 = *reinterpret_cast<float4*>(s_dsc + c0), a1 = *reinterpret_cast<float4*>(s_dsc + c0 + 4);
        a0.x += g0*xh0; a0.y += g1*xh1; a0.z += g2*xh2; a0.w += g3*xh3;
        a1.x += g4*xh4; a1.y += g5*xh5; a1.z += g6*xh6; a1.w += g7*xh7;
        *reinterpret_cast<float4*>(s_dsc + c0) = a0;
        *reinterpret_cast<float4*>(s_dsc + c0 + 4) = a1;

        float4 b0 = *reinterpret_cast<float4*>(s_dsh + c0), b1 = *reinterpret_cast<float4*>(s_dsh + c0 + 4);
        b0.x += g0; b0.y += g1; b0.z += g2; b0.w += g3;
        b1.x += g4; b1.y += g5; b1.z += g6; b1.w += g7;
        *reinterpret_cast<float4*>(s_dsh + c0) = b0;
        *reinterpret_cast<float4*>(s_dsh + c0 + 4) = b1;

        float dxh0 = g0*s0, dxh1 = g1*s1, dxh2 = g2*s2, dxh3 = g3*s3;
        float dxh4 = g4*s4, dxh5 = g5*s5, dxh6 = g6*s6, dxh7 = g7*s7;

        float sum1 = (dxh0 + dxh1) + (dxh2 + dxh3) + (dxh4 + dxh5) + (dxh6 + dxh7);
        float sum2 = (dxh0*xh0 + dxh1*xh1) + (dxh2*xh2 + dxh3*xh3) + (dxh4*xh4 + dxh5*xh5) + (dxh6*xh6 + dxh7*xh7);
        float m1, m2;
        block384_reduce_sum2(sum1, sum2, smem, m1, m2);
        m1 *= INV_D_WAN;
        m2 *= INV_D_WAN;

        uint4 dxv;
        if (DO_ADD) {
            uint4 av = reinterpret_cast<const uint4*>(dx_add + row * D_WAN)[tid];
            dxv.x = f32x2_to_bf16x2(rstd * (dxh0 - m1 - xh0 * m2) + bf16_lo_to_f32(av.x),
                                    rstd * (dxh1 - m1 - xh1 * m2) + bf16_hi_to_f32(av.x));
            dxv.y = f32x2_to_bf16x2(rstd * (dxh2 - m1 - xh2 * m2) + bf16_lo_to_f32(av.y),
                                    rstd * (dxh3 - m1 - xh3 * m2) + bf16_hi_to_f32(av.y));
            dxv.z = f32x2_to_bf16x2(rstd * (dxh4 - m1 - xh4 * m2) + bf16_lo_to_f32(av.z),
                                    rstd * (dxh5 - m1 - xh5 * m2) + bf16_hi_to_f32(av.z));
            dxv.w = f32x2_to_bf16x2(rstd * (dxh6 - m1 - xh6 * m2) + bf16_lo_to_f32(av.w),
                                    rstd * (dxh7 - m1 - xh7 * m2) + bf16_hi_to_f32(av.w));
        } else {
            dxv.x = f32x2_to_bf16x2(rstd * (dxh0 - m1 - xh0 * m2), rstd * (dxh1 - m1 - xh1 * m2));
            dxv.y = f32x2_to_bf16x2(rstd * (dxh2 - m1 - xh2 * m2), rstd * (dxh3 - m1 - xh3 * m2));
            dxv.z = f32x2_to_bf16x2(rstd * (dxh4 - m1 - xh4 * m2), rstd * (dxh5 - m1 - xh5 * m2));
            dxv.w = f32x2_to_bf16x2(rstd * (dxh6 - m1 - xh6 * m2), rstd * (dxh7 - m1 - xh7 * m2));
        }
        reinterpret_cast<uint4*>(dx + row * D_WAN)[tid] = dxv;
    }

    int64_t p_off = ((int64_t)b * P + p) * D_WAN + c0;
    *reinterpret_cast<float4*>(p_dscale + p_off)     = *reinterpret_cast<float4*>(s_dsc + c0);
    *reinterpret_cast<float4*>(p_dscale + p_off + 4) = *reinterpret_cast<float4*>(s_dsc + c0 + 4);
    *reinterpret_cast<float4*>(p_dshift + p_off)     = *reinterpret_cast<float4*>(s_dsh + c0);
    *reinterpret_cast<float4*>(p_dshift + p_off + 4) = *reinterpret_cast<float4*>(s_dsh + c0 + 4);
}

// ============================================================================
// 2. Fused Gated Residual + Affine LayerNorm (1-warp/row fwd + SMEM sum2 bwd!)
// ============================================================================
__global__ void fused_gated_res_ln_fwd_d3072(
    const __mt_bfloat16* __restrict__ x,
    const __mt_bfloat16* __restrict__ upd,
    const float* __restrict__ gate,
    const float* __restrict__ w,
    const float* __restrict__ b_bias,
    __mt_bfloat16* __restrict__ x_mid_out,
    __mt_bfloat16* __restrict__ ln_out,
    float* __restrict__ mean_out,
    float* __restrict__ rstd_out,
    int32_t S,
    int32_t total_rows,
    float eps
) {
    int32_t lane = threadIdx.x & 31;
    int32_t warp = threadIdx.x >> 5;
    int32_t row = blockIdx.x * 8 + warp;
    if (row >= total_rows) return;
    int32_t b = row / S;

    const uint4* xr = reinterpret_cast<const uint4*>(x + (int64_t)row * D_WAN);
    const uint4* ur = reinterpret_cast<const uint4*>(upd + (int64_t)row * D_WAN);
    uint4* xmr = reinterpret_cast<uint4*>(x_mid_out + (int64_t)row * D_WAN);
    uint4* yr = reinterpret_cast<uint4*>(ln_out + (int64_t)row * D_WAN);
    const float4* gr = reinterpret_cast<const float4*>(gate + (int64_t)b * D_WAN);
    const float4* wr = reinterpret_cast<const float4*>(w);
    const float4* br = reinterpret_cast<const float4*>(b_bias);

    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        int idx = k * 32 + lane;
        uint4 xv = xr[idx], uv = ur[idx];
        float4 g0 = gr[idx * 2], g1 = gr[idx * 2 + 1];
        uint4 xmv;
        xmv.x = f32x2_to_bf16x2(bf16_lo_to_f32(xv.x) + g0.x * bf16_lo_to_f32(uv.x),
                                bf16_hi_to_f32(xv.x) + g0.y * bf16_hi_to_f32(uv.x));
        xmv.y = f32x2_to_bf16x2(bf16_lo_to_f32(xv.y) + g0.z * bf16_lo_to_f32(uv.y),
                                bf16_hi_to_f32(xv.y) + g0.w * bf16_hi_to_f32(uv.y));
        xmv.z = f32x2_to_bf16x2(bf16_lo_to_f32(xv.z) + g1.x * bf16_lo_to_f32(uv.z),
                                bf16_hi_to_f32(xv.z) + g1.y * bf16_hi_to_f32(uv.z));
        xmv.w = f32x2_to_bf16x2(bf16_lo_to_f32(xv.w) + g1.z * bf16_lo_to_f32(uv.w),
                                bf16_hi_to_f32(xv.w) + g1.w * bf16_hi_to_f32(uv.w));
        xmr[idx] = xmv;
        sum += (bf16_lo_to_f32(xmv.x) + bf16_hi_to_f32(xmv.x))
             + (bf16_lo_to_f32(xmv.y) + bf16_hi_to_f32(xmv.y))
             + (bf16_lo_to_f32(xmv.z) + bf16_hi_to_f32(xmv.z))
             + (bf16_lo_to_f32(xmv.w) + bf16_hi_to_f32(xmv.w));
    }
    float mean = warp_reduce_sum(sum) * INV_D_WAN;

    float vsum = 0.0f;
    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        uint4 v = xmr[k * 32 + lane];
        float d0 = bf16_lo_to_f32(v.x) - mean, d1 = bf16_hi_to_f32(v.x) - mean;
        float d2 = bf16_lo_to_f32(v.y) - mean, d3 = bf16_hi_to_f32(v.y) - mean;
        float d4 = bf16_lo_to_f32(v.z) - mean, d5 = bf16_hi_to_f32(v.z) - mean;
        float d6 = bf16_lo_to_f32(v.w) - mean, d7 = bf16_hi_to_f32(v.w) - mean;
        vsum += (d0*d0 + d1*d1) + (d2*d2 + d3*d3) + (d4*d4 + d5*d5) + (d6*d6 + d7*d7);
    }
    float rstd = rsqrtf(warp_reduce_sum(vsum) * INV_D_WAN + eps);
    if (lane == 0) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
    }

    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        int idx = k * 32 + lane;
        uint4 v = xmr[idx];
        float4 w0 = wr[idx * 2], w1 = wr[idx * 2 + 1];
        float4 b0 = br[idx * 2], b1 = br[idx * 2 + 1];
        uint4 yv;
        yv.x = f32x2_to_bf16x2(((bf16_lo_to_f32(v.x) - mean) * rstd) * w0.x + b0.x,
                               ((bf16_hi_to_f32(v.x) - mean) * rstd) * w0.y + b0.y);
        yv.y = f32x2_to_bf16x2(((bf16_lo_to_f32(v.y) - mean) * rstd) * w0.z + b0.z,
                               ((bf16_hi_to_f32(v.y) - mean) * rstd) * w0.w + b0.w);
        yv.z = f32x2_to_bf16x2(((bf16_lo_to_f32(v.z) - mean) * rstd) * w1.x + b1.x,
                               ((bf16_hi_to_f32(v.z) - mean) * rstd) * w1.y + b1.y);
        yv.w = f32x2_to_bf16x2(((bf16_lo_to_f32(v.w) - mean) * rstd) * w1.z + b1.z,
                               ((bf16_hi_to_f32(v.w) - mean) * rstd) * w1.w + b1.w);
        yr[idx] = yv;
    }
}

template <bool DO_DB>
__global__ void fused_gated_res_ln_bwd_d3072(
    const __mt_bfloat16* __restrict__ d_xmid,
    const __mt_bfloat16* __restrict__ d_ln,
    const __mt_bfloat16* __restrict__ x_mid,
    const __mt_bfloat16* __restrict__ upd,
    const float* __restrict__ gate,
    const float* __restrict__ w,
    const float* __restrict__ mean_in,
    const float* __restrict__ rstd_in,
    __mt_bfloat16* __restrict__ dx_out,
    __mt_bfloat16* __restrict__ dupd_out,
    float* __restrict__ p_dgate,
    float* __restrict__ p_dw,
    float* __restrict__ p_db,
    float* __restrict__ p_db_upd,
    int32_t S,
    int32_t P
) {
    __shared__ float smem[32];
    __shared__ float s_dw[D_WAN];
    __shared__ float s_db[D_WAN];
    __shared__ float s_dbu[D_WAN];
    int32_t p = blockIdx.x;
    int32_t b = blockIdx.y;
    int32_t tid = threadIdx.x;
    int32_t c0 = tid * 8;
    float4 z = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    *reinterpret_cast<float4*>(s_dw + c0) = z;  *reinterpret_cast<float4*>(s_dw + c0 + 4) = z;
    *reinterpret_cast<float4*>(s_db + c0) = z;  *reinterpret_cast<float4*>(s_db + c0 + 4) = z;
    if (DO_DB) {
        *reinterpret_cast<float4*>(s_dbu + c0) = z;
        *reinterpret_cast<float4*>(s_dbu + c0 + 4) = z;
    }

    int32_t rows_per_p = (S + P - 1) / P;
    int32_t r_start = p * rows_per_p;
    int32_t r_end = r_start + rows_per_p;
    if (r_end > S) r_end = S;

    float4 w0 = *reinterpret_cast<const float4*>(w + c0);
    float4 w1 = *reinterpret_cast<const float4*>(w + c0 + 4);
    const float* g_b = gate + (int64_t)b * D_WAN + c0;
    float4 g0 = *reinterpret_cast<const float4*>(g_b);
    float4 g1 = *reinterpret_cast<const float4*>(g_b + 4);
    float dg0=0.0f, dg1=0.0f, dg2=0.0f, dg3=0.0f, dg4=0.0f, dg5=0.0f, dg6=0.0f, dg7=0.0f;

    for (int32_t s = r_start; s < r_end; ++s) {
        int64_t row = (int64_t)b * S + s;
        float mean = mean_in[row];
        float rstd = rstd_in[row];

        uint4 xmv = reinterpret_cast<const uint4*>(x_mid + row * D_WAN)[tid];
        uint4 dlnv = reinterpret_cast<const uint4*>(d_ln + row * D_WAN)[tid];

        float xh0 = (bf16_lo_to_f32(xmv.x) - mean) * rstd, xh1 = (bf16_hi_to_f32(xmv.x) - mean) * rstd;
        float xh2 = (bf16_lo_to_f32(xmv.y) - mean) * rstd, xh3 = (bf16_hi_to_f32(xmv.y) - mean) * rstd;
        float xh4 = (bf16_lo_to_f32(xmv.z) - mean) * rstd, xh5 = (bf16_hi_to_f32(xmv.z) - mean) * rstd;
        float xh6 = (bf16_lo_to_f32(xmv.w) - mean) * rstd, xh7 = (bf16_hi_to_f32(xmv.w) - mean) * rstd;

        float l0 = bf16_lo_to_f32(dlnv.x), l1 = bf16_hi_to_f32(dlnv.x);
        float l2 = bf16_lo_to_f32(dlnv.y), l3 = bf16_hi_to_f32(dlnv.y);
        float l4 = bf16_lo_to_f32(dlnv.z), l5 = bf16_hi_to_f32(dlnv.z);
        float l6 = bf16_lo_to_f32(dlnv.w), l7 = bf16_hi_to_f32(dlnv.w);

        float4 a0 = *reinterpret_cast<float4*>(s_dw + c0), a1 = *reinterpret_cast<float4*>(s_dw + c0 + 4);
        a0.x += l0*xh0; a0.y += l1*xh1; a0.z += l2*xh2; a0.w += l3*xh3;
        a1.x += l4*xh4; a1.y += l5*xh5; a1.z += l6*xh6; a1.w += l7*xh7;
        *reinterpret_cast<float4*>(s_dw + c0) = a0; *reinterpret_cast<float4*>(s_dw + c0 + 4) = a1;

        a0 = *reinterpret_cast<float4*>(s_db + c0); a1 = *reinterpret_cast<float4*>(s_db + c0 + 4);
        a0.x += l0; a0.y += l1; a0.z += l2; a0.w += l3;
        a1.x += l4; a1.y += l5; a1.z += l6; a1.w += l7;
        *reinterpret_cast<float4*>(s_db + c0) = a0; *reinterpret_cast<float4*>(s_db + c0 + 4) = a1;

        float dxh0 = l0*w0.x, dxh1 = l1*w0.y, dxh2 = l2*w0.z, dxh3 = l3*w0.w;
        float dxh4 = l4*w1.x, dxh5 = l5*w1.y, dxh6 = l6*w1.z, dxh7 = l7*w1.w;

        float sum1 = (dxh0 + dxh1) + (dxh2 + dxh3) + (dxh4 + dxh5) + (dxh6 + dxh7);
        float sum2 = (dxh0*xh0 + dxh1*xh1) + (dxh2*xh2 + dxh3*xh3) + (dxh4*xh4 + dxh5*xh5) + (dxh6*xh6 + dxh7*xh7);
        float m1, m2;
        block384_reduce_sum2(sum1, sum2, smem, m1, m2);
        m1 *= INV_D_WAN;
        m2 *= INV_D_WAN;

        uint4 dxmv = reinterpret_cast<const uint4*>(d_xmid + row * D_WAN)[tid];
        uint4 uv = reinterpret_cast<const uint4*>(upd + row * D_WAN)[tid];

        float t0 = bf16_lo_to_f32(dxmv.x) + rstd * (dxh0 - m1 - xh0 * m2);
        float t1 = bf16_hi_to_f32(dxmv.x) + rstd * (dxh1 - m1 - xh1 * m2);
        float t2 = bf16_lo_to_f32(dxmv.y) + rstd * (dxh2 - m1 - xh2 * m2);
        float t3 = bf16_hi_to_f32(dxmv.y) + rstd * (dxh3 - m1 - xh3 * m2);
        float t4 = bf16_lo_to_f32(dxmv.z) + rstd * (dxh4 - m1 - xh4 * m2);
        float t5 = bf16_hi_to_f32(dxmv.z) + rstd * (dxh5 - m1 - xh5 * m2);
        float t6 = bf16_lo_to_f32(dxmv.w) + rstd * (dxh6 - m1 - xh6 * m2);
        float t7 = bf16_hi_to_f32(dxmv.w) + rstd * (dxh7 - m1 - xh7 * m2);

        uint4 dxv, duv;
        dxv.x = f32x2_to_bf16x2(t0, t1); dxv.y = f32x2_to_bf16x2(t2, t3);
        dxv.z = f32x2_to_bf16x2(t4, t5); dxv.w = f32x2_to_bf16x2(t6, t7);
        duv.x = f32x2_to_bf16x2(t0 * g0.x, t1 * g0.y);
        duv.y = f32x2_to_bf16x2(t2 * g0.z, t3 * g0.w);
        duv.z = f32x2_to_bf16x2(t4 * g1.x, t5 * g1.y);
        duv.w = f32x2_to_bf16x2(t6 * g1.z, t7 * g1.w);
        reinterpret_cast<uint4*>(dx_out + row * D_WAN)[tid] = dxv;
        reinterpret_cast<uint4*>(dupd_out + row * D_WAN)[tid] = duv;

        if (DO_DB) {
            a0 = *reinterpret_cast<float4*>(s_dbu + c0); a1 = *reinterpret_cast<float4*>(s_dbu + c0 + 4);
            a0.x += bf16_lo_to_f32(duv.x); a0.y += bf16_hi_to_f32(duv.x);
            a0.z += bf16_lo_to_f32(duv.y); a0.w += bf16_hi_to_f32(duv.y);
            a1.x += bf16_lo_to_f32(duv.z); a1.y += bf16_hi_to_f32(duv.z);
            a1.z += bf16_lo_to_f32(duv.w); a1.w += bf16_hi_to_f32(duv.w);
            *reinterpret_cast<float4*>(s_dbu + c0) = a0; *reinterpret_cast<float4*>(s_dbu + c0 + 4) = a1;
        }

        dg0 += t0 * bf16_lo_to_f32(uv.x); dg1 += t1 * bf16_hi_to_f32(uv.x);
        dg2 += t2 * bf16_lo_to_f32(uv.y); dg3 += t3 * bf16_hi_to_f32(uv.y);
        dg4 += t4 * bf16_lo_to_f32(uv.z); dg5 += t5 * bf16_hi_to_f32(uv.z);
        dg6 += t6 * bf16_lo_to_f32(uv.w); dg7 += t7 * bf16_hi_to_f32(uv.w);
    }

    int64_t p_off = ((int64_t)b * P + p) * D_WAN + c0;
    *reinterpret_cast<float4*>(p_dgate + p_off)     = make_float4(dg0, dg1, dg2, dg3);
    *reinterpret_cast<float4*>(p_dgate + p_off + 4) = make_float4(dg4, dg5, dg6, dg7);
    *reinterpret_cast<float4*>(p_dw + p_off)        = *reinterpret_cast<float4*>(s_dw + c0);
    *reinterpret_cast<float4*>(p_dw + p_off + 4)    = *reinterpret_cast<float4*>(s_dw + c0 + 4);
    *reinterpret_cast<float4*>(p_db + p_off)        = *reinterpret_cast<float4*>(s_db + c0);
    *reinterpret_cast<float4*>(p_db + p_off + 4)    = *reinterpret_cast<float4*>(s_db + c0 + 4);
    if (DO_DB) {
        *reinterpret_cast<float4*>(p_db_upd + p_off)     = *reinterpret_cast<float4*>(s_dbu + c0);
        *reinterpret_cast<float4*>(p_db_upd + p_off + 4) = *reinterpret_cast<float4*>(s_dbu + c0 + 4);
    }
}

// ============================================================================
// 3. Fused Residual + AdaLN (1-warp/row fwd + SMEM sum2 bwd!)
// ============================================================================
__global__ void fused_res_adaln_fwd_d3072(
    const __mt_bfloat16* __restrict__ x,
    const __mt_bfloat16* __restrict__ upd,
    const float* __restrict__ scale,
    const float* __restrict__ shift,
    __mt_bfloat16* __restrict__ x_mid_out,
    __mt_bfloat16* __restrict__ y_out,
    float* __restrict__ mean_out,
    float* __restrict__ rstd_out,
    int32_t S,
    int32_t total_rows,
    float eps
) {
    int32_t lane = threadIdx.x & 31;
    int32_t warp = threadIdx.x >> 5;
    int32_t row = blockIdx.x * 8 + warp;
    if (row >= total_rows) return;
    int32_t b = row / S;

    const uint4* xr = reinterpret_cast<const uint4*>(x + (int64_t)row * D_WAN);
    const uint4* ur = reinterpret_cast<const uint4*>(upd + (int64_t)row * D_WAN);
    uint4* xmr = reinterpret_cast<uint4*>(x_mid_out + (int64_t)row * D_WAN);
    uint4* yr = reinterpret_cast<uint4*>(y_out + (int64_t)row * D_WAN);
    const float4* scr = reinterpret_cast<const float4*>(scale + (int64_t)b * D_WAN);
    const float4* shr = reinterpret_cast<const float4*>(shift + (int64_t)b * D_WAN);

    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        int idx = k * 32 + lane;
        uint4 xv = xr[idx], uv = ur[idx];
        uint4 xmv;
        xmv.x = f32x2_to_bf16x2(bf16_lo_to_f32(xv.x) + bf16_lo_to_f32(uv.x),
                                bf16_hi_to_f32(xv.x) + bf16_hi_to_f32(uv.x));
        xmv.y = f32x2_to_bf16x2(bf16_lo_to_f32(xv.y) + bf16_lo_to_f32(uv.y),
                                bf16_hi_to_f32(xv.y) + bf16_hi_to_f32(uv.y));
        xmv.z = f32x2_to_bf16x2(bf16_lo_to_f32(xv.z) + bf16_lo_to_f32(uv.z),
                                bf16_hi_to_f32(xv.z) + bf16_hi_to_f32(uv.z));
        xmv.w = f32x2_to_bf16x2(bf16_lo_to_f32(xv.w) + bf16_lo_to_f32(uv.w),
                                bf16_hi_to_f32(xv.w) + bf16_hi_to_f32(uv.w));
        xmr[idx] = xmv;
        sum += (bf16_lo_to_f32(xmv.x) + bf16_hi_to_f32(xmv.x))
             + (bf16_lo_to_f32(xmv.y) + bf16_hi_to_f32(xmv.y))
             + (bf16_lo_to_f32(xmv.z) + bf16_hi_to_f32(xmv.z))
             + (bf16_lo_to_f32(xmv.w) + bf16_hi_to_f32(xmv.w));
    }
    float mean = warp_reduce_sum(sum) * INV_D_WAN;

    float vsum = 0.0f;
    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        uint4 v = xmr[k * 32 + lane];
        float d0 = bf16_lo_to_f32(v.x) - mean, d1 = bf16_hi_to_f32(v.x) - mean;
        float d2 = bf16_lo_to_f32(v.y) - mean, d3 = bf16_hi_to_f32(v.y) - mean;
        float d4 = bf16_lo_to_f32(v.z) - mean, d5 = bf16_hi_to_f32(v.z) - mean;
        float d6 = bf16_lo_to_f32(v.w) - mean, d7 = bf16_hi_to_f32(v.w) - mean;
        vsum += (d0*d0 + d1*d1) + (d2*d2 + d3*d3) + (d4*d4 + d5*d5) + (d6*d6 + d7*d7);
    }
    float rstd = rsqrtf(warp_reduce_sum(vsum) * INV_D_WAN + eps);
    if (lane == 0) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
    }

    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        int idx = k * 32 + lane;
        uint4 v = xmr[idx];
        float4 sc0 = scr[idx * 2], sc1 = scr[idx * 2 + 1];
        float4 sh0 = shr[idx * 2], sh1 = shr[idx * 2 + 1];
        uint4 yv;
        yv.x = f32x2_to_bf16x2(((bf16_lo_to_f32(v.x) - mean) * rstd) * (1.0f + sc0.x) + sh0.x,
                               ((bf16_hi_to_f32(v.x) - mean) * rstd) * (1.0f + sc0.y) + sh0.y);
        yv.y = f32x2_to_bf16x2(((bf16_lo_to_f32(v.y) - mean) * rstd) * (1.0f + sc0.z) + sh0.z,
                               ((bf16_hi_to_f32(v.y) - mean) * rstd) * (1.0f + sc0.w) + sh0.w);
        yv.z = f32x2_to_bf16x2(((bf16_lo_to_f32(v.z) - mean) * rstd) * (1.0f + sc1.x) + sh1.x,
                               ((bf16_hi_to_f32(v.z) - mean) * rstd) * (1.0f + sc1.y) + sh1.y);
        yv.w = f32x2_to_bf16x2(((bf16_lo_to_f32(v.w) - mean) * rstd) * (1.0f + sc1.z) + sh1.z,
                               ((bf16_hi_to_f32(v.w) - mean) * rstd) * (1.0f + sc1.w) + sh1.w);
        yr[idx] = yv;
    }
}

template <bool DO_DB>
__global__ void fused_res_adaln_bwd_d3072(
    const __mt_bfloat16* __restrict__ d_xmid,
    const __mt_bfloat16* __restrict__ dy,
    const __mt_bfloat16* __restrict__ x_mid,
    const float* __restrict__ scale,
    const float* __restrict__ mean_in,
    const float* __restrict__ rstd_in,
    __mt_bfloat16* __restrict__ dx_out,
    float* __restrict__ p_dscale,
    float* __restrict__ p_dshift,
    float* __restrict__ p_db_upd,
    int32_t S,
    int32_t P
) {
    __shared__ float smem[32];
    __shared__ float s_dsc[D_WAN];
    __shared__ float s_dsh[D_WAN];
    __shared__ float s_dbu[D_WAN];
    int32_t p = blockIdx.x;
    int32_t b = blockIdx.y;
    int32_t tid = threadIdx.x;
    int32_t c0 = tid * 8;
    float4 z = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    *reinterpret_cast<float4*>(s_dsc + c0) = z; *reinterpret_cast<float4*>(s_dsc + c0 + 4) = z;
    *reinterpret_cast<float4*>(s_dsh + c0) = z; *reinterpret_cast<float4*>(s_dsh + c0 + 4) = z;
    if (DO_DB) {
        *reinterpret_cast<float4*>(s_dbu + c0) = z;
        *reinterpret_cast<float4*>(s_dbu + c0 + 4) = z;
    }

    int32_t rows_per_p = (S + P - 1) / P;
    int32_t r_start = p * rows_per_p;
    int32_t r_end = r_start + rows_per_p;
    if (r_end > S) r_end = S;

    const float* sc_b = scale + (int64_t)b * D_WAN + c0;
    float4 sc0 = *reinterpret_cast<const float4*>(sc_b);
    float4 sc1 = *reinterpret_cast<const float4*>(sc_b + 4);
    float s0 = 1.0f + sc0.x, s1 = 1.0f + sc0.y, s2 = 1.0f + sc0.z, s3 = 1.0f + sc0.w;
    float s4 = 1.0f + sc1.x, s5 = 1.0f + sc1.y, s6 = 1.0f + sc1.z, s7 = 1.0f + sc1.w;

    for (int32_t s = r_start; s < r_end; ++s) {
        int64_t row = (int64_t)b * S + s;
        float mean = mean_in[row];
        float rstd = rstd_in[row];

        uint4 xmv = reinterpret_cast<const uint4*>(x_mid + row * D_WAN)[tid];
        uint4 gv = reinterpret_cast<const uint4*>(dy + row * D_WAN)[tid];
        uint4 dxmv = reinterpret_cast<const uint4*>(d_xmid + row * D_WAN)[tid];

        float xh0 = (bf16_lo_to_f32(xmv.x) - mean) * rstd, xh1 = (bf16_hi_to_f32(xmv.x) - mean) * rstd;
        float xh2 = (bf16_lo_to_f32(xmv.y) - mean) * rstd, xh3 = (bf16_hi_to_f32(xmv.y) - mean) * rstd;
        float xh4 = (bf16_lo_to_f32(xmv.z) - mean) * rstd, xh5 = (bf16_hi_to_f32(xmv.z) - mean) * rstd;
        float xh6 = (bf16_lo_to_f32(xmv.w) - mean) * rstd, xh7 = (bf16_hi_to_f32(xmv.w) - mean) * rstd;

        float g0 = bf16_lo_to_f32(gv.x), g1 = bf16_hi_to_f32(gv.x);
        float g2 = bf16_lo_to_f32(gv.y), g3 = bf16_hi_to_f32(gv.y);
        float g4 = bf16_lo_to_f32(gv.z), g5 = bf16_hi_to_f32(gv.z);
        float g6 = bf16_lo_to_f32(gv.w), g7 = bf16_hi_to_f32(gv.w);

        float4 a0 = *reinterpret_cast<float4*>(s_dsc + c0), a1 = *reinterpret_cast<float4*>(s_dsc + c0 + 4);
        a0.x += g0*xh0; a0.y += g1*xh1; a0.z += g2*xh2; a0.w += g3*xh3;
        a1.x += g4*xh4; a1.y += g5*xh5; a1.z += g6*xh6; a1.w += g7*xh7;
        *reinterpret_cast<float4*>(s_dsc + c0) = a0; *reinterpret_cast<float4*>(s_dsc + c0 + 4) = a1;

        a0 = *reinterpret_cast<float4*>(s_dsh + c0); a1 = *reinterpret_cast<float4*>(s_dsh + c0 + 4);
        a0.x += g0; a0.y += g1; a0.z += g2; a0.w += g3;
        a1.x += g4; a1.y += g5; a1.z += g6; a1.w += g7;
        *reinterpret_cast<float4*>(s_dsh + c0) = a0; *reinterpret_cast<float4*>(s_dsh + c0 + 4) = a1;

        float dxh0 = g0*s0, dxh1 = g1*s1, dxh2 = g2*s2, dxh3 = g3*s3;
        float dxh4 = g4*s4, dxh5 = g5*s5, dxh6 = g6*s6, dxh7 = g7*s7;

        float sum1 = (dxh0 + dxh1) + (dxh2 + dxh3) + (dxh4 + dxh5) + (dxh6 + dxh7);
        float sum2 = (dxh0*xh0 + dxh1*xh1) + (dxh2*xh2 + dxh3*xh3) + (dxh4*xh4 + dxh5*xh5) + (dxh6*xh6 + dxh7*xh7);
        float m1, m2;
        block384_reduce_sum2(sum1, sum2, smem, m1, m2);
        m1 *= INV_D_WAN;
        m2 *= INV_D_WAN;

        uint4 dxv;
        dxv.x = f32x2_to_bf16x2(bf16_lo_to_f32(dxmv.x) + rstd * (dxh0 - m1 - xh0 * m2),
                                bf16_hi_to_f32(dxmv.x) + rstd * (dxh1 - m1 - xh1 * m2));
        dxv.y = f32x2_to_bf16x2(bf16_lo_to_f32(dxmv.y) + rstd * (dxh2 - m1 - xh2 * m2),
                                bf16_hi_to_f32(dxmv.y) + rstd * (dxh3 - m1 - xh3 * m2));
        dxv.z = f32x2_to_bf16x2(bf16_lo_to_f32(dxmv.z) + rstd * (dxh4 - m1 - xh4 * m2),
                                bf16_hi_to_f32(dxmv.z) + rstd * (dxh5 - m1 - xh5 * m2));
        dxv.w = f32x2_to_bf16x2(bf16_lo_to_f32(dxmv.w) + rstd * (dxh6 - m1 - xh6 * m2),
                                bf16_hi_to_f32(dxmv.w) + rstd * (dxh7 - m1 - xh7 * m2));
        reinterpret_cast<uint4*>(dx_out + row * D_WAN)[tid] = dxv;
        if (DO_DB) {
            a0 = *reinterpret_cast<float4*>(s_dbu + c0); a1 = *reinterpret_cast<float4*>(s_dbu + c0 + 4);
            a0.x += bf16_lo_to_f32(dxv.x); a0.y += bf16_hi_to_f32(dxv.x);
            a0.z += bf16_lo_to_f32(dxv.y); a0.w += bf16_hi_to_f32(dxv.y);
            a1.x += bf16_lo_to_f32(dxv.z); a1.y += bf16_hi_to_f32(dxv.z);
            a1.z += bf16_lo_to_f32(dxv.w); a1.w += bf16_hi_to_f32(dxv.w);
            *reinterpret_cast<float4*>(s_dbu + c0) = a0; *reinterpret_cast<float4*>(s_dbu + c0 + 4) = a1;
        }
    }

    int64_t p_off = ((int64_t)b * P + p) * D_WAN + c0;
    *reinterpret_cast<float4*>(p_dscale + p_off)     = *reinterpret_cast<float4*>(s_dsc + c0);
    *reinterpret_cast<float4*>(p_dscale + p_off + 4) = *reinterpret_cast<float4*>(s_dsc + c0 + 4);
    *reinterpret_cast<float4*>(p_dshift + p_off)     = *reinterpret_cast<float4*>(s_dsh + c0);
    *reinterpret_cast<float4*>(p_dshift + p_off + 4) = *reinterpret_cast<float4*>(s_dsh + c0 + 4);
    if (DO_DB) {
        *reinterpret_cast<float4*>(p_db_upd + p_off)     = *reinterpret_cast<float4*>(s_dbu + c0);
        *reinterpret_cast<float4*>(p_db_upd + p_off + 4) = *reinterpret_cast<float4*>(s_dbu + c0 + 4);
    }
}

// ============================================================================
// 4. Fused Gated Residual (384 threads, pure scalars!)
// ============================================================================
__global__ void fused_gated_res_fwd_d3072(
    const __mt_bfloat16* __restrict__ x,
    const __mt_bfloat16* __restrict__ upd,
    const float* __restrict__ gate,
    __mt_bfloat16* __restrict__ out,
    int32_t S
) {
    int32_t row = blockIdx.x;
    int32_t b = row / S;
    int32_t tid = threadIdx.x;
    int32_t c0 = tid * 8;

    uint4 xv = reinterpret_cast<const uint4*>(x + (int64_t)row * D_WAN)[tid];
    uint4 uv = reinterpret_cast<const uint4*>(upd + (int64_t)row * D_WAN)[tid];
    const float* g_b = gate + (int64_t)b * D_WAN + c0;
    float4 g0 = *reinterpret_cast<const float4*>(g_b);
    float4 g1 = *reinterpret_cast<const float4*>(g_b + 4);

    uint4 ov;
    ov.x = f32x2_to_bf16x2(bf16_lo_to_f32(xv.x) + g0.x * bf16_lo_to_f32(uv.x),
                           bf16_hi_to_f32(xv.x) + g0.y * bf16_hi_to_f32(uv.x));
    ov.y = f32x2_to_bf16x2(bf16_lo_to_f32(xv.y) + g0.z * bf16_lo_to_f32(uv.y),
                           bf16_hi_to_f32(xv.y) + g0.w * bf16_hi_to_f32(uv.y));
    ov.z = f32x2_to_bf16x2(bf16_lo_to_f32(xv.z) + g1.x * bf16_lo_to_f32(uv.z),
                           bf16_hi_to_f32(xv.z) + g1.y * bf16_hi_to_f32(uv.z));
    ov.w = f32x2_to_bf16x2(bf16_lo_to_f32(xv.w) + g1.z * bf16_lo_to_f32(uv.w),
                           bf16_hi_to_f32(xv.w) + g1.w * bf16_hi_to_f32(uv.w));
    reinterpret_cast<uint4*>(out + (int64_t)row * D_WAN)[tid] = ov;
}

__global__ void fused_gated_res_bwd_d3072(
    const __mt_bfloat16* __restrict__ dout,
    const __mt_bfloat16* __restrict__ upd,
    const float* __restrict__ gate,
    __mt_bfloat16* __restrict__ dupd_out,
    float* __restrict__ p_dgate,
    float* __restrict__ p_db_upd,
    int32_t S,
    int32_t P
) {
    int32_t p = blockIdx.x;
    int32_t b = blockIdx.y;
    int32_t tid = threadIdx.x;
    int32_t c0 = tid * 8;

    int32_t rows_per_p = (S + P - 1) / P;
    int32_t r_start = p * rows_per_p;
    int32_t r_end = r_start + rows_per_p;
    if (r_end > S) r_end = S;

    const float* g_b = gate + (int64_t)b * D_WAN + c0;
    float4 g0 = *reinterpret_cast<const float4*>(g_b);
    float4 g1 = *reinterpret_cast<const float4*>(g_b + 4);
    float dg0=0, dg1=0, dg2=0, dg3=0, dg4=0, dg5=0, dg6=0, dg7=0;
    float dbu0=0, dbu1=0, dbu2=0, dbu3=0, dbu4=0, dbu5=0, dbu6=0, dbu7=0;

    for (int32_t s = r_start; s < r_end; ++s) {
        int64_t row = (int64_t)b * S + s;
        uint4 dov = reinterpret_cast<const uint4*>(dout + row * D_WAN)[tid];
        uint4 uv = reinterpret_cast<const uint4*>(upd + row * D_WAN)[tid];

        float d0 = bf16_lo_to_f32(dov.x), d1 = bf16_hi_to_f32(dov.x);
        float d2 = bf16_lo_to_f32(dov.y), d3 = bf16_hi_to_f32(dov.y);
        float d4 = bf16_lo_to_f32(dov.z), d5 = bf16_hi_to_f32(dov.z);
        float d6 = bf16_lo_to_f32(dov.w), d7 = bf16_hi_to_f32(dov.w);

        uint4 duv;
        duv.x = f32x2_to_bf16x2(d0 * g0.x, d1 * g0.y);
        duv.y = f32x2_to_bf16x2(d2 * g0.z, d3 * g0.w);
        duv.z = f32x2_to_bf16x2(d4 * g1.x, d5 * g1.y);
        duv.w = f32x2_to_bf16x2(d6 * g1.z, d7 * g1.w);
        reinterpret_cast<uint4*>(dupd_out + row * D_WAN)[tid] = duv;

        if (p_db_upd != nullptr) {
            dbu0 += bf16_lo_to_f32(duv.x); dbu1 += bf16_hi_to_f32(duv.x);
            dbu2 += bf16_lo_to_f32(duv.y); dbu3 += bf16_hi_to_f32(duv.y);
            dbu4 += bf16_lo_to_f32(duv.z); dbu5 += bf16_hi_to_f32(duv.z);
            dbu6 += bf16_lo_to_f32(duv.w); dbu7 += bf16_hi_to_f32(duv.w);
        }

        dg0 += d0 * bf16_lo_to_f32(uv.x); dg1 += d1 * bf16_hi_to_f32(uv.x);
        dg2 += d2 * bf16_lo_to_f32(uv.y); dg3 += d3 * bf16_hi_to_f32(uv.y);
        dg4 += d4 * bf16_lo_to_f32(uv.z); dg5 += d5 * bf16_hi_to_f32(uv.z);
        dg6 += d6 * bf16_lo_to_f32(uv.w); dg7 += d7 * bf16_hi_to_f32(uv.w);
    }

    int64_t p_off = ((int64_t)b * P + p) * D_WAN + c0;
    *reinterpret_cast<float4*>(p_dgate + p_off)     = make_float4(dg0, dg1, dg2, dg3);
    *reinterpret_cast<float4*>(p_dgate + p_off + 4) = make_float4(dg4, dg5, dg6, dg7);
    if (p_db_upd != nullptr) {
        *reinterpret_cast<float4*>(p_db_upd + p_off)     = make_float4(dbu0, dbu1, dbu2, dbu3);
        *reinterpret_cast<float4*>(p_db_upd + p_off + 4) = make_float4(dbu4, dbu5, dbu6, dbu7);
    }
}

// ============================================================================
// 5. Fused RMSNorm (+ Optional Interleaved 3D RoPE) (1-warp/row fwd + SMEM-db bwd!)
// ============================================================================
template <bool DO_ROPE>
__global__ void fused_rmsnorm_rope_fwd_d3072(
    const __mt_bfloat16* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ cos_tab,
    const float* __restrict__ sin_tab,
    __mt_bfloat16* __restrict__ out,
    float* __restrict__ rstd_out,
    int32_t S,
    int32_t total_rows,
    float eps
) {
    int32_t lane = threadIdx.x & 31;
    int32_t warp = threadIdx.x >> 5;
    int32_t row = blockIdx.x * 4 + warp;
    if (row >= total_rows) return;
    int32_t s = row % S;

    const uint4* xr = reinterpret_cast<const uint4*>(x + (int64_t)row * D_WAN);
    uint4* yr = reinterpret_cast<uint4*>(out + (int64_t)row * D_WAN);
    const float4* wr = reinterpret_cast<const float4*>(w);

    float sq = 0.0f;
    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        uint4 v = xr[k * 32 + lane];
        float x0 = bf16_lo_to_f32(v.x), x1 = bf16_hi_to_f32(v.x);
        float x2 = bf16_lo_to_f32(v.y), x3 = bf16_hi_to_f32(v.y);
        float x4 = bf16_lo_to_f32(v.z), x5 = bf16_hi_to_f32(v.z);
        float x6 = bf16_lo_to_f32(v.w), x7 = bf16_hi_to_f32(v.w);
        sq += (x0*x0 + x1*x1) + (x2*x2 + x3*x3) + (x4*x4 + x5*x5) + (x6*x6 + x7*x7);
    }
    float rstd = rsqrtf(warp_reduce_sum(sq) * INV_D_WAN + eps);
    if (lane == 0) rstd_out[row] = rstd;

    // ((k * 32 + lane) * 8 & 127) >> 1 == (lane & 15) * 4, independent of k -> 12x fewer cos/sin loads!
    float4 cv = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    float4 sv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (DO_ROPE) {
        int32_t p_off = (lane & 15) * 4;
        cv = *reinterpret_cast<const float4*>(cos_tab + (int64_t)s * 64 + p_off);
        sv = *reinterpret_cast<const float4*>(sin_tab + (int64_t)s * 64 + p_off);
    }

    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        int idx = k * 32 + lane;
        uint4 v = xr[idx];
        float4 w0 = wr[idx * 2], w1 = wr[idx * 2 + 1];
        float x0 = bf16_lo_to_f32(v.x), x1 = bf16_hi_to_f32(v.x);
        float x2 = bf16_lo_to_f32(v.y), x3 = bf16_hi_to_f32(v.y);
        float x4 = bf16_lo_to_f32(v.z), x5 = bf16_hi_to_f32(v.z);
        float x6 = bf16_lo_to_f32(v.w), x7 = bf16_hi_to_f32(v.w);
        uint4 ov;
        if (DO_ROPE) {
            float u0 = round_bf16_f32((x0 * rstd) * w0.x);
            float u1 = round_bf16_f32((x1 * rstd) * w0.y);
            float u2 = round_bf16_f32((x2 * rstd) * w0.z);
            float u3 = round_bf16_f32((x3 * rstd) * w0.w);
            float u4 = round_bf16_f32((x4 * rstd) * w1.x);
            float u5 = round_bf16_f32((x5 * rstd) * w1.y);
            float u6 = round_bf16_f32((x6 * rstd) * w1.z);
            float u7 = round_bf16_f32((x7 * rstd) * w1.w);
            ov.x = f32x2_to_bf16x2(u0 * cv.x - u1 * sv.x, u0 * sv.x + u1 * cv.x);
            ov.y = f32x2_to_bf16x2(u2 * cv.y - u3 * sv.y, u2 * sv.y + u3 * cv.y);
            ov.z = f32x2_to_bf16x2(u4 * cv.z - u5 * sv.z, u4 * sv.z + u5 * cv.z);
            ov.w = f32x2_to_bf16x2(u6 * cv.w - u7 * sv.w, u6 * sv.w + u7 * cv.w);
        } else {
            ov.x = f32x2_to_bf16x2((x0 * rstd) * w0.x, (x1 * rstd) * w0.y);
            ov.y = f32x2_to_bf16x2((x2 * rstd) * w0.z, (x3 * rstd) * w0.w);
            ov.z = f32x2_to_bf16x2((x4 * rstd) * w1.x, (x5 * rstd) * w1.y);
            ov.w = f32x2_to_bf16x2((x6 * rstd) * w1.z, (x7 * rstd) * w1.w);
        }
        yr[idx] = ov;
    }
}

template <bool DO_ROPE, bool DO_DB>
__global__ void fused_rmsnorm_rope_bwd_d3072(
    const __mt_bfloat16* __restrict__ dy,
    const __mt_bfloat16* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ cos_tab,
    const float* __restrict__ sin_tab,
    const float* __restrict__ rstd_in,
    __mt_bfloat16* __restrict__ dx,
    float* __restrict__ p_dw,
    float* __restrict__ p_db_x,
    int32_t S,
    int32_t P
) {
    __shared__ float smem[16];
    __shared__ float s_db[D_WAN]; // 12 KB SMEM db accumulator -> ZERO register spill!
    int32_t p = blockIdx.x;
    int32_t b = blockIdx.y;
    int32_t tid = threadIdx.x;
    int32_t c0 = tid * 8;
    int32_t p_off_rope = (c0 & 127) >> 1;

    if (DO_DB) {
        float4 z = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        *reinterpret_cast<float4*>(s_db + c0) = z;
        *reinterpret_cast<float4*>(s_db + c0 + 4) = z;
    }

    int32_t rows_per_p = (S + P - 1) / P;
    int32_t r_start = p * rows_per_p;
    int32_t r_end = r_start + rows_per_p;
    if (r_end > S) r_end = S;

    float4 w0 = *reinterpret_cast<const float4*>(w + c0);
    float4 w1 = *reinterpret_cast<const float4*>(w + c0 + 4);
    float dw0=0, dw1=0, dw2=0, dw3=0, dw4=0, dw5=0, dw6=0, dw7=0;

    for (int32_t s = r_start; s < r_end; ++s) {
        int64_t row = (int64_t)b * S + s;
        float rstd = rstd_in[row];
        uint4 xv = reinterpret_cast<const uint4*>(x + row * D_WAN)[tid];
        uint4 gv = reinterpret_cast<const uint4*>(dy + row * D_WAN)[tid];

        float x0 = bf16_lo_to_f32(xv.x), x1 = bf16_hi_to_f32(xv.x);
        float x2 = bf16_lo_to_f32(xv.y), x3 = bf16_hi_to_f32(xv.y);
        float x4 = bf16_lo_to_f32(xv.z), x5 = bf16_hi_to_f32(xv.z);
        float x6 = bf16_lo_to_f32(xv.w), x7 = bf16_hi_to_f32(xv.w);

        float g0 = bf16_lo_to_f32(gv.x), g1 = bf16_hi_to_f32(gv.x);
        float g2 = bf16_lo_to_f32(gv.y), g3 = bf16_hi_to_f32(gv.y);
        float g4 = bf16_lo_to_f32(gv.z), g5 = bf16_hi_to_f32(gv.z);
        float g6 = bf16_lo_to_f32(gv.w), g7 = bf16_hi_to_f32(gv.w);

        float du0, du1, du2, du3, du4, du5, du6, du7;
        if (DO_ROPE) {
            float4 cv = *reinterpret_cast<const float4*>(cos_tab + (int64_t)s * 64 + p_off_rope);
            float4 sv = *reinterpret_cast<const float4*>(sin_tab + (int64_t)s * 64 + p_off_rope);
            du0 = g0 * cv.x + g1 * sv.x; du1 = -g0 * sv.x + g1 * cv.x;
            du2 = g2 * cv.y + g3 * sv.y; du3 = -g2 * sv.y + g3 * cv.y;
            du4 = g4 * cv.z + g5 * sv.z; du5 = -g4 * sv.z + g5 * cv.z;
            du6 = g6 * cv.w + g7 * sv.w; du7 = -g6 * sv.w + g7 * cv.w;
        } else {
            du0 = g0; du1 = g1; du2 = g2; du3 = g3;
            du4 = g4; du5 = g5; du6 = g6; du7 = g7;
        }

        dw0 += du0 * (x0 * rstd); dw1 += du1 * (x1 * rstd);
        dw2 += du2 * (x2 * rstd); dw3 += du3 * (x3 * rstd);
        dw4 += du4 * (x4 * rstd); dw5 += du5 * (x5 * rstd);
        dw6 += du6 * (x6 * rstd); dw7 += du7 * (x7 * rstd);

        float dwx0 = du0 * w0.x, dwx1 = du1 * w0.y, dwx2 = du2 * w0.z, dwx3 = du3 * w0.w;
        float dwx4 = du4 * w1.x, dwx5 = du5 * w1.y, dwx6 = du6 * w1.z, dwx7 = du7 * w1.w;

        float dot = (dwx0*x0 + dwx1*x1) + (dwx2*x2 + dwx3*x3) + (dwx4*x4 + dwx5*x5) + (dwx6*x6 + dwx7*x7);
        float coeff = block384_reduce_sum(dot, smem) * INV_D_WAN * (rstd * rstd);

        uint4 dxv;
        dxv.x = f32x2_to_bf16x2(rstd * (dwx0 - x0 * coeff), rstd * (dwx1 - x1 * coeff));
        dxv.y = f32x2_to_bf16x2(rstd * (dwx2 - x2 * coeff), rstd * (dwx3 - x3 * coeff));
        dxv.z = f32x2_to_bf16x2(rstd * (dwx4 - x4 * coeff), rstd * (dwx5 - x5 * coeff));
        dxv.w = f32x2_to_bf16x2(rstd * (dwx6 - x6 * coeff), rstd * (dwx7 - x7 * coeff));
        reinterpret_cast<uint4*>(dx + row * D_WAN)[tid] = dxv;
        if (DO_DB) {
            float4 b0 = *reinterpret_cast<float4*>(s_db + c0);
            float4 b1 = *reinterpret_cast<float4*>(s_db + c0 + 4);
            b0.x += bf16_lo_to_f32(dxv.x); b0.y += bf16_hi_to_f32(dxv.x);
            b0.z += bf16_lo_to_f32(dxv.y); b0.w += bf16_hi_to_f32(dxv.y);
            b1.x += bf16_lo_to_f32(dxv.z); b1.y += bf16_hi_to_f32(dxv.z);
            b1.z += bf16_lo_to_f32(dxv.w); b1.w += bf16_hi_to_f32(dxv.w);
            *reinterpret_cast<float4*>(s_db + c0) = b0;
            *reinterpret_cast<float4*>(s_db + c0 + 4) = b1;
        }
    }

    int64_t p_off = ((int64_t)b * P + p) * D_WAN + c0;
    *reinterpret_cast<float4*>(p_dw + p_off)     = make_float4(dw0, dw1, dw2, dw3);
    *reinterpret_cast<float4*>(p_dw + p_off + 4) = make_float4(dw4, dw5, dw6, dw7);
    if (DO_DB) {
        *reinterpret_cast<float4*>(p_db_x + p_off)     = *reinterpret_cast<float4*>(s_db + c0);
        *reinterpret_cast<float4*>(p_db_x + p_off + 4) = *reinterpret_cast<float4*>(s_db + c0 + 4);
    }
}

// ============================================================================
// 6. Fused VSA Tile + Transpose + MeanPool (D = 128, blk = 64, pure scalars!)
// ============================================================================
__global__ void fused_vsa_tile_pool_fwd_d128(
    const __mt_bfloat16* __restrict__ q,
    const __mt_bfloat16* __restrict__ k,
    const __mt_bfloat16* __restrict__ v,
    const int32_t* __restrict__ pad_to_raster,
    const int32_t* __restrict__ vbs,
    __mt_bfloat16* __restrict__ qt,
    __mt_bfloat16* __restrict__ kt,
    __mt_bfloat16* __restrict__ vt,
    __mt_bfloat16* __restrict__ qc,
    __mt_bfloat16* __restrict__ kc,
    __mt_bfloat16* __restrict__ vc,
    int32_t S,
    int32_t H,
    int32_t N_blk
) {
    __shared__ float smem[16][130];
    int32_t bh = blockIdx.x;
    int32_t blk = blockIdx.y;
    int32_t which = blockIdx.z;
    int32_t b = bh / H;
    int32_t h = bh % H;
    int32_t tid = threadIdx.x;
    int32_t d_grp = tid & 15;
    int32_t t_grp = tid >> 4;
    int32_t vbs_k = vbs[blk];

    const __mt_bfloat16* src = (which == 0) ? q : ((which == 1) ? k : v);
    __mt_bfloat16* dst_t = (which == 0) ? qt : ((which == 1) ? kt : vt);
    __mt_bfloat16* dst_c = (which == 0) ? qc : ((which == 1) ? kc : vc);

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
    int64_t tiled_blk_base = ((int64_t)bh * N_blk + blk) * 64 * 128;
    uint4 zero4 = make_uint4(0, 0, 0, 0);

    for (int32_t t = t_grp; t < 64; t += 16) {
        uint4* out_ptr = reinterpret_cast<uint4*>(dst_t + tiled_blk_base + (int64_t)t * 128);
        if (t < vbs_k) {
            int32_t r = pad_to_raster[blk * 64 + t];
            int64_t src_off = (((int64_t)b * S + r) * H + h) * 128;
            uint4 pv = reinterpret_cast<const uint4*>(src + src_off)[d_grp];
            out_ptr[d_grp] = pv;
            s0 += bf16_lo_to_f32(pv.x); s1 += bf16_hi_to_f32(pv.x);
            s2 += bf16_lo_to_f32(pv.y); s3 += bf16_hi_to_f32(pv.y);
            s4 += bf16_lo_to_f32(pv.z); s5 += bf16_hi_to_f32(pv.z);
            s6 += bf16_lo_to_f32(pv.w); s7 += bf16_hi_to_f32(pv.w);
        } else {
            out_ptr[d_grp] = zero4;
        }
    }

    int32_t d0 = d_grp * 8;
    smem[t_grp][d0 + 0] = s0; smem[t_grp][d0 + 1] = s1;
    smem[t_grp][d0 + 2] = s2; smem[t_grp][d0 + 3] = s3;
    smem[t_grp][d0 + 4] = s4; smem[t_grp][d0 + 5] = s5;
    smem[t_grp][d0 + 6] = s6; smem[t_grp][d0 + 7] = s7;
    __syncthreads();

    if (tid < 128) {
        float total = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) total += smem[i][tid];
        dst_c[((int64_t)bh * N_blk + blk) * 128 + tid] = __float2bfloat16(total / (float)vbs_k);
    }
}

__global__ void fused_vsa_tile_pool_bwd_d128(
    const __mt_bfloat16* __restrict__ dqt,
    const __mt_bfloat16* __restrict__ dkt,
    const __mt_bfloat16* __restrict__ dvt,
    const __mt_bfloat16* __restrict__ dqc,
    const __mt_bfloat16* __restrict__ dkc,
    const __mt_bfloat16* __restrict__ dvc,
    const int32_t* __restrict__ pad_to_raster,
    const int32_t* __restrict__ vbs,
    __mt_bfloat16* __restrict__ dq,
    __mt_bfloat16* __restrict__ dk,
    __mt_bfloat16* __restrict__ dv,
    float* __restrict__ p_dbv,
    int32_t S,
    int32_t H,
    int32_t N_blk,
    int32_t P_blk
) {
    __shared__ float smem[8][130];
    int32_t bh = blockIdx.x;
    int32_t pb = blockIdx.y;
    int32_t which = blockIdx.z;
    int32_t b = bh / H;
    int32_t h = bh % H;
    int32_t tid = threadIdx.x;
    int32_t d_grp = tid & 15;
    int32_t t_grp = tid >> 4;

    const __mt_bfloat16* dt = (which == 0) ? dqt : ((which == 1) ? dkt : dvt);
    const __mt_bfloat16* dc = (which == 0) ? dqc : ((which == 1) ? dkc : dvc);
    __mt_bfloat16* dr = (which == 0) ? dq : ((which == 1) ? dk : dv);

    int32_t b_per = (N_blk + P_blk - 1) / P_blk;
    int32_t b0 = pb * b_per;
    int32_t b1 = b0 + b_per;
    if (b1 > N_blk) b1 = N_blk;

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
    bool do_dbv = (which == 2 && p_dbv != nullptr);

    for (int32_t blk = b0; blk < b1; ++blk) {
        int32_t vbs_k = vbs[blk];
        float inv_vbs = 1.0f / (float)vbs_k;
        uint4 dcv = reinterpret_cast<const uint4*>(dc + ((int64_t)bh * N_blk + blk) * 128)[d_grp];
        float c0 = bf16_lo_to_f32(dcv.x) * inv_vbs, c1 = bf16_hi_to_f32(dcv.x) * inv_vbs;
        float c2 = bf16_lo_to_f32(dcv.y) * inv_vbs, c3 = bf16_hi_to_f32(dcv.y) * inv_vbs;
        float c4 = bf16_lo_to_f32(dcv.z) * inv_vbs, c5 = bf16_hi_to_f32(dcv.z) * inv_vbs;
        float c6 = bf16_lo_to_f32(dcv.w) * inv_vbs, c7 = bf16_hi_to_f32(dcv.w) * inv_vbs;

        int64_t tiled_blk_base = ((int64_t)bh * N_blk + blk) * 64 * 128;
        for (int32_t t = t_grp; t < vbs_k; t += 8) {
            uint4 dtv = reinterpret_cast<const uint4*>(dt + tiled_blk_base + (int64_t)t * 128)[d_grp];
            uint4 ov;
            ov.x = f32x2_to_bf16x2(bf16_lo_to_f32(dtv.x) + c0, bf16_hi_to_f32(dtv.x) + c1);
            ov.y = f32x2_to_bf16x2(bf16_lo_to_f32(dtv.y) + c2, bf16_hi_to_f32(dtv.y) + c3);
            ov.z = f32x2_to_bf16x2(bf16_lo_to_f32(dtv.z) + c4, bf16_hi_to_f32(dtv.z) + c5);
            ov.w = f32x2_to_bf16x2(bf16_lo_to_f32(dtv.w) + c6, bf16_hi_to_f32(dtv.w) + c7);
            int32_t r = pad_to_raster[blk * 64 + t];
            int64_t dst_off = (((int64_t)b * S + r) * H + h) * 128;
            reinterpret_cast<uint4*>(dr + dst_off)[d_grp] = ov;
            if (do_dbv) {
                s0 += bf16_lo_to_f32(ov.x); s1 += bf16_hi_to_f32(ov.x);
                s2 += bf16_lo_to_f32(ov.y); s3 += bf16_hi_to_f32(ov.y);
                s4 += bf16_lo_to_f32(ov.z); s5 += bf16_hi_to_f32(ov.z);
                s6 += bf16_lo_to_f32(ov.w); s7 += bf16_hi_to_f32(ov.w);
            }
        }
    }
    if (do_dbv) {
        int32_t d0 = d_grp * 8;
        smem[t_grp][d0 + 0] = s0; smem[t_grp][d0 + 1] = s1;
        smem[t_grp][d0 + 2] = s2; smem[t_grp][d0 + 3] = s3;
        smem[t_grp][d0 + 4] = s4; smem[t_grp][d0 + 5] = s5;
        smem[t_grp][d0 + 6] = s6; smem[t_grp][d0 + 7] = s7;
        __syncthreads();
        float total = smem[0][tid] + smem[1][tid] + smem[2][tid] + smem[3][tid]
                    + smem[4][tid] + smem[5][tid] + smem[6][tid] + smem[7][tid];
        p_dbv[(((int64_t)b * P_blk + pb) * H + h) * 128 + tid] = total;
    }
}

// ============================================================================
// 7. Fused VSA Combine + Untile (D = 128, blk = 64, pure scalars!)
// ============================================================================
__global__ void fused_vsa_combine_untile_fwd_d128(
    const __mt_bfloat16* __restrict__ out_s,
    const __mt_bfloat16* __restrict__ out_c,
    const __mt_bfloat16* __restrict__ gate,
    const int32_t* __restrict__ pad_to_raster,
    const int32_t* __restrict__ vbs,
    __mt_bfloat16* __restrict__ out_raster,
    int32_t S,
    int32_t H,
    int32_t N_blk
) {
    int32_t blk = blockIdx.x;
    int32_t bh = blockIdx.y;
    int32_t b = bh / H;
    int32_t h = bh % H;
    int32_t tid = threadIdx.x;
    int32_t d_grp = tid & 15;
    int32_t t_grp = tid >> 4;
    int32_t vbs_k = vbs[blk];

    uint4 ocv = reinterpret_cast<const uint4*>(out_c + ((int64_t)bh * N_blk + blk) * 128)[d_grp];
    float oc0 = bf16_lo_to_f32(ocv.x), oc1 = bf16_hi_to_f32(ocv.x);
    float oc2 = bf16_lo_to_f32(ocv.y), oc3 = bf16_hi_to_f32(ocv.y);
    float oc4 = bf16_lo_to_f32(ocv.z), oc5 = bf16_hi_to_f32(ocv.z);
    float oc6 = bf16_lo_to_f32(ocv.w), oc7 = bf16_hi_to_f32(ocv.w);

    int64_t tiled_blk_base = ((int64_t)bh * N_blk + blk) * 64 * 128;
    for (int32_t t = t_grp; t < vbs_k; t += 8) {
        int32_t r = pad_to_raster[blk * 64 + t];
        int64_t r_off = (((int64_t)b * S + r) * H + h) * 128;
        uint4 osv = reinterpret_cast<const uint4*>(out_s + tiled_blk_base + (int64_t)t * 128)[d_grp];
        uint4 ov;
        if (gate != nullptr) {
            uint4 gv = reinterpret_cast<const uint4*>(gate + r_off)[d_grp];
            ov.x = f32x2_to_bf16x2(bf16_lo_to_f32(osv.x) + oc0 * bf16_lo_to_f32(gv.x),
                                   bf16_hi_to_f32(osv.x) + oc1 * bf16_hi_to_f32(gv.x));
            ov.y = f32x2_to_bf16x2(bf16_lo_to_f32(osv.y) + oc2 * bf16_lo_to_f32(gv.y),
                                   bf16_hi_to_f32(osv.y) + oc3 * bf16_hi_to_f32(gv.y));
            ov.z = f32x2_to_bf16x2(bf16_lo_to_f32(osv.z) + oc4 * bf16_lo_to_f32(gv.z),
                                   bf16_hi_to_f32(osv.z) + oc5 * bf16_hi_to_f32(gv.z));
            ov.w = f32x2_to_bf16x2(bf16_lo_to_f32(osv.w) + oc6 * bf16_lo_to_f32(gv.w),
                                   bf16_hi_to_f32(osv.w) + oc7 * bf16_hi_to_f32(gv.w));
        } else {
            ov.x = f32x2_to_bf16x2(bf16_lo_to_f32(osv.x) + oc0, bf16_hi_to_f32(osv.x) + oc1);
            ov.y = f32x2_to_bf16x2(bf16_lo_to_f32(osv.y) + oc2, bf16_hi_to_f32(osv.y) + oc3);
            ov.z = f32x2_to_bf16x2(bf16_lo_to_f32(osv.z) + oc4, bf16_hi_to_f32(osv.z) + oc5);
            ov.w = f32x2_to_bf16x2(bf16_lo_to_f32(osv.w) + oc6, bf16_hi_to_f32(osv.w) + oc7);
        }
        reinterpret_cast<uint4*>(out_raster + r_off)[d_grp] = ov;
    }
}

__global__ void fused_vsa_combine_untile_bwd_d128(
    const __mt_bfloat16* __restrict__ dout,
    const __mt_bfloat16* __restrict__ out_c,
    const __mt_bfloat16* __restrict__ gate,
    const int32_t* __restrict__ pad_to_raster,
    const int32_t* __restrict__ vbs,
    __mt_bfloat16* __restrict__ d_out_s,
    __mt_bfloat16* __restrict__ d_out_c,
    __mt_bfloat16* __restrict__ d_gate,
    float* __restrict__ p_dbg,
    int32_t S,
    int32_t H,
    int32_t N_blk,
    int32_t P_blk
) {
    __shared__ float smem[8][130];
    int32_t bh = blockIdx.x;
    int32_t pb = blockIdx.y;
    int32_t b = bh / H;
    int32_t h = bh % H;
    int32_t tid = threadIdx.x;
    int32_t d_grp = tid & 15;
    int32_t t_grp = tid >> 4;

    int32_t b_per = (N_blk + P_blk - 1) / P_blk;
    int32_t b0 = pb * b_per;
    int32_t b1 = b0 + b_per;
    if (b1 > N_blk) b1 = N_blk;

    uint4 zero4 = make_uint4(0, 0, 0, 0);
    float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
    float g4 = 0.0f, g5 = 0.0f, g6 = 0.0f, g7 = 0.0f;
    bool do_dbg = (gate != nullptr && p_dbg != nullptr);

    for (int32_t blk = b0; blk < b1; ++blk) {
        int32_t vbs_k = vbs[blk];
        uint4 ocv = reinterpret_cast<const uint4*>(out_c + ((int64_t)bh * N_blk + blk) * 128)[d_grp];
        float oc0 = bf16_lo_to_f32(ocv.x), oc1 = bf16_hi_to_f32(ocv.x);
        float oc2 = bf16_lo_to_f32(ocv.y), oc3 = bf16_hi_to_f32(ocv.y);
        float oc4 = bf16_lo_to_f32(ocv.z), oc5 = bf16_hi_to_f32(ocv.z);
        float oc6 = bf16_lo_to_f32(ocv.w), oc7 = bf16_hi_to_f32(ocv.w);

        float doc0 = 0.0f, doc1 = 0.0f, doc2 = 0.0f, doc3 = 0.0f;
        float doc4 = 0.0f, doc5 = 0.0f, doc6 = 0.0f, doc7 = 0.0f;
        int64_t tiled_blk_base = ((int64_t)bh * N_blk + blk) * 64 * 128;

        for (int32_t t = t_grp; t < 64; t += 8) {
            uint4* dos_ptr = reinterpret_cast<uint4*>(d_out_s + tiled_blk_base + (int64_t)t * 128);
            if (t < vbs_k) {
                int32_t r = pad_to_raster[blk * 64 + t];
                int64_t r_off = (((int64_t)b * S + r) * H + h) * 128;
                uint4 dov = reinterpret_cast<const uint4*>(dout + r_off)[d_grp];
                dos_ptr[d_grp] = dov;
                float d0 = bf16_lo_to_f32(dov.x), d1 = bf16_hi_to_f32(dov.x);
                float d2 = bf16_lo_to_f32(dov.y), d3 = bf16_hi_to_f32(dov.y);
                float d4 = bf16_lo_to_f32(dov.z), d5 = bf16_hi_to_f32(dov.z);
                float d6 = bf16_lo_to_f32(dov.w), d7 = bf16_hi_to_f32(dov.w);
                if (gate != nullptr) {
                    uint4 gv = reinterpret_cast<const uint4*>(gate + r_off)[d_grp];
                    uint4 dgv;
                    dgv.x = f32x2_to_bf16x2(d0 * oc0, d1 * oc1);
                    dgv.y = f32x2_to_bf16x2(d2 * oc2, d3 * oc3);
                    dgv.z = f32x2_to_bf16x2(d4 * oc4, d5 * oc5);
                    dgv.w = f32x2_to_bf16x2(d6 * oc6, d7 * oc7);
                    reinterpret_cast<uint4*>(d_gate + r_off)[d_grp] = dgv;
                    if (do_dbg) {
                        g0 += bf16_lo_to_f32(dgv.x); g1 += bf16_hi_to_f32(dgv.x);
                        g2 += bf16_lo_to_f32(dgv.y); g3 += bf16_hi_to_f32(dgv.y);
                        g4 += bf16_lo_to_f32(dgv.z); g5 += bf16_hi_to_f32(dgv.z);
                        g6 += bf16_lo_to_f32(dgv.w); g7 += bf16_hi_to_f32(dgv.w);
                    }
                    doc0 += d0 * bf16_lo_to_f32(gv.x); doc1 += d1 * bf16_hi_to_f32(gv.x);
                    doc2 += d2 * bf16_lo_to_f32(gv.y); doc3 += d3 * bf16_hi_to_f32(gv.y);
                    doc4 += d4 * bf16_lo_to_f32(gv.z); doc5 += d5 * bf16_hi_to_f32(gv.z);
                    doc6 += d6 * bf16_lo_to_f32(gv.w); doc7 += d7 * bf16_hi_to_f32(gv.w);
                } else {
                    doc0 += d0; doc1 += d1; doc2 += d2; doc3 += d3;
                    doc4 += d4; doc5 += d5; doc6 += d6; doc7 += d7;
                }
            } else {
                dos_ptr[d_grp] = zero4;
            }
        }

        int32_t d0 = d_grp * 8;
        smem[t_grp][d0 + 0] = doc0; smem[t_grp][d0 + 1] = doc1;
        smem[t_grp][d0 + 2] = doc2; smem[t_grp][d0 + 3] = doc3;
        smem[t_grp][d0 + 4] = doc4; smem[t_grp][d0 + 5] = doc5;
        smem[t_grp][d0 + 6] = doc6; smem[t_grp][d0 + 7] = doc7;
        __syncthreads();

        float total = smem[0][tid] + smem[1][tid] + smem[2][tid] + smem[3][tid]
                    + smem[4][tid] + smem[5][tid] + smem[6][tid] + smem[7][tid];
        d_out_c[((int64_t)bh * N_blk + blk) * 128 + tid] = __float2bfloat16(total);
        __syncthreads();
    }

    if (do_dbg) {
        int32_t d0 = d_grp * 8;
        smem[t_grp][d0 + 0] = g0; smem[t_grp][d0 + 1] = g1;
        smem[t_grp][d0 + 2] = g2; smem[t_grp][d0 + 3] = g3;
        smem[t_grp][d0 + 4] = g4; smem[t_grp][d0 + 5] = g5;
        smem[t_grp][d0 + 6] = g6; smem[t_grp][d0 + 7] = g7;
        __syncthreads();
        float tot_g = smem[0][tid] + smem[1][tid] + smem[2][tid] + smem[3][tid]
                    + smem[4][tid] + smem[5][tid] + smem[6][tid] + smem[7][tid];
        p_dbg[(((int64_t)b * P_blk + pb) * H + h) * 128 + tid] = tot_g;
    }
}

// ============================================================================
// C Launch Wrappers
// ============================================================================
extern "C" {

int launch_adaln_fwd(const void* x, const void* scale, const void* shift,
                     void* y, void* mean, void* rstd,
                     int32_t B, int32_t S, float eps, void* stream) {
    int32_t total_rows = B * S;
    fused_adaln_fwd_d3072<<<(total_rows + 3) / 4, 128, 0, (musaStream_t)stream>>>(
        (const __mt_bfloat16*)x, (const float*)scale, (const float*)shift,
        (__mt_bfloat16*)y, (float*)mean, (float*)rstd, S, total_rows, eps);
    return (int)musaGetLastError();
}

int launch_adaln_bwd(const void* dy, const void* x, const void* scale,
                     const void* mean, const void* rstd, const void* dx_add,
                     void* dx, void* p_dscale, void* p_dshift,
                     void* dscale, void* dshift,
                     int32_t B, int32_t S, int32_t P, void* stream) {
    musaStream_t st = (musaStream_t)stream;
    dim3 grid(P, B);
    if (dx_add != nullptr) {
        fused_adaln_bwd_d3072<true><<<grid, 384, 0, st>>>(
            (const __mt_bfloat16*)dy, (const __mt_bfloat16*)x, (const float*)scale,
            (const float*)mean, (const float*)rstd, (const __mt_bfloat16*)dx_add,
            (__mt_bfloat16*)dx, (float*)p_dscale, (float*)p_dshift, S, P);
    } else {
        fused_adaln_bwd_d3072<false><<<grid, 384, 0, st>>>(
            (const __mt_bfloat16*)dy, (const __mt_bfloat16*)x, (const float*)scale,
            (const float*)mean, (const float*)rstd, nullptr,
            (__mt_bfloat16*)dx, (float*)p_dscale, (float*)p_dshift, S, P);
    }
    dim3 rgrid((D_WAN + 31) / 32, B);
    dim3 rblk(32, 8);
    reduce_partial_f32_kernel<<<rgrid, rblk, 0, st>>>((const float*)p_dscale, (float*)dscale, B, P, D_WAN, 0);
    reduce_partial_f32_kernel<<<rgrid, rblk, 0, st>>>((const float*)p_dshift, (float*)dshift, B, P, D_WAN, 0);
    return (int)musaGetLastError();
}

int launch_gated_res_ln_fwd(const void* x, const void* upd, const void* gate,
                            const void* w, const void* b_bias,
                            void* x_mid, void* ln_out, void* mean, void* rstd,
                            int32_t B, int32_t S, float eps, void* stream) {
    int32_t total_rows = B * S;
    fused_gated_res_ln_fwd_d3072<<<(total_rows + 7) / 8, 256, 0, (musaStream_t)stream>>>(
        (const __mt_bfloat16*)x, (const __mt_bfloat16*)upd, (const float*)gate,
        (const float*)w, (const float*)b_bias,
        (__mt_bfloat16*)x_mid, (__mt_bfloat16*)ln_out, (float*)mean, (float*)rstd, S, total_rows, eps);
    return (int)musaGetLastError();
}

int launch_gated_res_ln_bwd(const void* d_xmid, const void* d_ln, const void* x_mid,
                            const void* upd, const void* gate, const void* w,
                            const void* mean, const void* rstd,
                            void* dx, void* dupd,
                            void* p_dg, void* p_dw, void* p_db, void* p_db_upd,
                            void* dg, void* dw, void* db, void* db_upd,
                            int32_t B, int32_t S, int32_t P, void* stream) {
    musaStream_t st = (musaStream_t)stream;
    dim3 grid(P, B);
    if (p_db_upd != nullptr && db_upd != nullptr) {
        fused_gated_res_ln_bwd_d3072<true><<<grid, 384, 0, st>>>(
            (const __mt_bfloat16*)d_xmid, (const __mt_bfloat16*)d_ln,
            (const __mt_bfloat16*)x_mid, (const __mt_bfloat16*)upd,
            (const float*)gate, (const float*)w, (const float*)mean, (const float*)rstd,
            (__mt_bfloat16*)dx, (__mt_bfloat16*)dupd,
            (float*)p_dg, (float*)p_dw, (float*)p_db, (float*)p_db_upd, S, P);
    } else {
        fused_gated_res_ln_bwd_d3072<false><<<grid, 384, 0, st>>>(
            (const __mt_bfloat16*)d_xmid, (const __mt_bfloat16*)d_ln,
            (const __mt_bfloat16*)x_mid, (const __mt_bfloat16*)upd,
            (const float*)gate, (const float*)w, (const float*)mean, (const float*)rstd,
            (__mt_bfloat16*)dx, (__mt_bfloat16*)dupd,
            (float*)p_dg, (float*)p_dw, (float*)p_db, nullptr, S, P);
    }
    dim3 rgrid_b((D_WAN + 31) / 32, B);
    dim3 rgrid_1((D_WAN + 31) / 32, 1);
    dim3 rblk(32, 8);
    reduce_partial_f32_kernel<<<rgrid_b, rblk, 0, st>>>((const float*)p_dg, (float*)dg, B, P, D_WAN, 0);
    reduce_partial_f32_kernel<<<rgrid_1, rblk, 0, st>>>((const float*)p_dw, (float*)dw, B, P, D_WAN, 1);
    reduce_partial_f32_kernel<<<rgrid_1, rblk, 0, st>>>((const float*)p_db, (float*)db, B, P, D_WAN, 1);
    if (p_db_upd != nullptr && db_upd != nullptr) {
        reduce_partial_f32_to_bf16_kernel<<<rgrid_1, rblk, 0, st>>>((const float*)p_db_upd, (__mt_bfloat16*)db_upd, B, P, D_WAN);
    }
    return (int)musaGetLastError();
}

int launch_res_adaln_fwd(const void* x, const void* upd, const void* scale, const void* shift,
                         void* x_mid, void* y_out, void* mean, void* rstd,
                         int32_t B, int32_t S, float eps, void* stream) {
    int32_t total_rows = B * S;
    fused_res_adaln_fwd_d3072<<<(total_rows + 7) / 8, 256, 0, (musaStream_t)stream>>>(
        (const __mt_bfloat16*)x, (const __mt_bfloat16*)upd,
        (const float*)scale, (const float*)shift,
        (__mt_bfloat16*)x_mid, (__mt_bfloat16*)y_out, (float*)mean, (float*)rstd, S, total_rows, eps);
    return (int)musaGetLastError();
}

int launch_res_adaln_bwd(const void* d_xmid, const void* dy, const void* x_mid,
                         const void* scale, const void* mean, const void* rstd,
                         void* dx, void* p_dscale, void* p_dshift, void* p_db_upd,
                         void* dscale, void* dshift, void* db_upd,
                         int32_t B, int32_t S, int32_t P, void* stream) {
    musaStream_t st = (musaStream_t)stream;
    dim3 grid(P, B);
    if (p_db_upd != nullptr && db_upd != nullptr) {
        fused_res_adaln_bwd_d3072<true><<<grid, 384, 0, st>>>(
            (const __mt_bfloat16*)d_xmid, (const __mt_bfloat16*)dy, (const __mt_bfloat16*)x_mid,
            (const float*)scale, (const float*)mean, (const float*)rstd,
            (__mt_bfloat16*)dx, (float*)p_dscale, (float*)p_dshift, (float*)p_db_upd, S, P);
    } else {
        fused_res_adaln_bwd_d3072<false><<<grid, 384, 0, st>>>(
            (const __mt_bfloat16*)d_xmid, (const __mt_bfloat16*)dy, (const __mt_bfloat16*)x_mid,
            (const float*)scale, (const float*)mean, (const float*)rstd,
            (__mt_bfloat16*)dx, (float*)p_dscale, (float*)p_dshift, nullptr, S, P);
    }
    dim3 rgrid((D_WAN + 31) / 32, B);
    dim3 rgrid_1((D_WAN + 31) / 32, 1);
    dim3 rblk(32, 8);
    reduce_partial_f32_kernel<<<rgrid, rblk, 0, st>>>((const float*)p_dscale, (float*)dscale, B, P, D_WAN, 0);
    reduce_partial_f32_kernel<<<rgrid, rblk, 0, st>>>((const float*)p_dshift, (float*)dshift, B, P, D_WAN, 0);
    if (p_db_upd != nullptr && db_upd != nullptr) {
        reduce_partial_f32_to_bf16_kernel<<<rgrid_1, rblk, 0, st>>>((const float*)p_db_upd, (__mt_bfloat16*)db_upd, B, P, D_WAN);
    }
    return (int)musaGetLastError();
}

int launch_gated_res_fwd(const void* x, const void* upd, const void* gate,
                         void* out, int32_t B, int32_t S, void* stream) {
    fused_gated_res_fwd_d3072<<<B * S, 384, 0, (musaStream_t)stream>>>(
        (const __mt_bfloat16*)x, (const __mt_bfloat16*)upd, (const float*)gate,
        (__mt_bfloat16*)out, S);
    return (int)musaGetLastError();
}

int launch_gated_res_bwd(const void* dout, const void* upd, const void* gate,
                         void* dupd, void* p_dgate, void* p_db_upd,
                         void* dgate, void* db_upd,
                         int32_t B, int32_t S, int32_t P, void* stream) {
    musaStream_t st = (musaStream_t)stream;
    dim3 grid(P, B);
    fused_gated_res_bwd_d3072<<<grid, 384, 0, st>>>(
        (const __mt_bfloat16*)dout, (const __mt_bfloat16*)upd, (const float*)gate,
        (__mt_bfloat16*)dupd, (float*)p_dgate, (float*)p_db_upd, S, P);
    dim3 rgrid((D_WAN + 31) / 32, B);
    dim3 rgrid_1((D_WAN + 31) / 32, 1);
    dim3 rblk(32, 8);
    reduce_partial_f32_kernel<<<rgrid, rblk, 0, st>>>((const float*)p_dgate, (float*)dgate, B, P, D_WAN, 0);
    if (p_db_upd != nullptr && db_upd != nullptr) {
        reduce_partial_f32_to_bf16_kernel<<<rgrid_1, rblk, 0, st>>>((const float*)p_db_upd, (__mt_bfloat16*)db_upd, B, P, D_WAN);
    }
    return (int)musaGetLastError();
}

int launch_rmsnorm_rope_fwd(const void* x, const void* w, const void* cos_tab, const void* sin_tab,
                            void* out, void* rstd, int32_t B, int32_t S, float eps, int32_t do_rope, void* stream) {
    int32_t total_rows = B * S;
    if (do_rope) {
        fused_rmsnorm_rope_fwd_d3072<true><<<(total_rows + 3) / 4, 128, 0, (musaStream_t)stream>>>(
            (const __mt_bfloat16*)x, (const float*)w, (const float*)cos_tab, (const float*)sin_tab,
            (__mt_bfloat16*)out, (float*)rstd, S, total_rows, eps);
    } else {
        fused_rmsnorm_rope_fwd_d3072<false><<<(total_rows + 3) / 4, 128, 0, (musaStream_t)stream>>>(
            (const __mt_bfloat16*)x, (const float*)w, nullptr, nullptr,
            (__mt_bfloat16*)out, (float*)rstd, S, total_rows, eps);
    }
    return (int)musaGetLastError();
}

int launch_rmsnorm_rope_bwd(const void* dy, const void* x, const void* w,
                            const void* cos_tab, const void* sin_tab, const void* rstd,
                            void* dx, void* p_dw, void* p_db_x, void* dw, void* db_x,
                            int32_t B, int32_t S, int32_t P, int32_t do_rope, void* stream) {
    musaStream_t st = (musaStream_t)stream;
    dim3 grid(P, B);
    bool do_db = (p_db_x != nullptr && db_x != nullptr);
    if (do_rope && do_db) {
        fused_rmsnorm_rope_bwd_d3072<true, true><<<grid, 384, 0, st>>>(
            (const __mt_bfloat16*)dy, (const __mt_bfloat16*)x, (const float*)w,
            (const float*)cos_tab, (const float*)sin_tab, (const float*)rstd,
            (__mt_bfloat16*)dx, (float*)p_dw, (float*)p_db_x, S, P);
    } else if (do_rope) {
        fused_rmsnorm_rope_bwd_d3072<true, false><<<grid, 384, 0, st>>>(
            (const __mt_bfloat16*)dy, (const __mt_bfloat16*)x, (const float*)w,
            (const float*)cos_tab, (const float*)sin_tab, (const float*)rstd,
            (__mt_bfloat16*)dx, (float*)p_dw, nullptr, S, P);
    } else if (do_db) {
        fused_rmsnorm_rope_bwd_d3072<false, true><<<grid, 384, 0, st>>>(
            (const __mt_bfloat16*)dy, (const __mt_bfloat16*)x, (const float*)w,
            nullptr, nullptr, (const float*)rstd,
            (__mt_bfloat16*)dx, (float*)p_dw, (float*)p_db_x, S, P);
    } else {
        fused_rmsnorm_rope_bwd_d3072<false, false><<<grid, 384, 0, st>>>(
            (const __mt_bfloat16*)dy, (const __mt_bfloat16*)x, (const float*)w,
            nullptr, nullptr, (const float*)rstd,
            (__mt_bfloat16*)dx, (float*)p_dw, nullptr, S, P);
    }
    dim3 rgrid_1((D_WAN + 31) / 32, 1);
    dim3 rblk(32, 8);
    reduce_partial_f32_kernel<<<rgrid_1, rblk, 0, st>>>((const float*)p_dw, (float*)dw, B, P, D_WAN, 1);
    if (do_db) {
        reduce_partial_f32_to_bf16_kernel<<<rgrid_1, rblk, 0, st>>>((const float*)p_db_x, (__mt_bfloat16*)db_x, B, P, D_WAN);
    }
    return (int)musaGetLastError();
}

int launch_vsa_tile_pool_fwd(const void* q, const void* k, const void* v,
                             const void* pad_to_raster, const void* vbs,
                             void* qt, void* kt, void* vt,
                             void* qc, void* kc, void* vc,
                             int32_t B, int32_t S, int32_t H, int32_t N_blk, void* stream) {
    dim3 grid(B * H, N_blk, 3);
    fused_vsa_tile_pool_fwd_d128<<<grid, 256, 0, (musaStream_t)stream>>>(
        (const __mt_bfloat16*)q, (const __mt_bfloat16*)k, (const __mt_bfloat16*)v,
        (const int32_t*)pad_to_raster, (const int32_t*)vbs,
        (__mt_bfloat16*)qt, (__mt_bfloat16*)kt, (__mt_bfloat16*)vt,
        (__mt_bfloat16*)qc, (__mt_bfloat16*)kc, (__mt_bfloat16*)vc,
        S, H, N_blk);
    return (int)musaGetLastError();
}

int launch_vsa_tile_pool_bwd(const void* dqt, const void* dkt, const void* dvt,
                             const void* dqc, const void* dkc, const void* dvc,
                             const void* pad_to_raster, const void* vbs,
                             void* dq, void* dk, void* dv,
                             void* p_dbv, void* dbv,
                             int32_t B, int32_t S, int32_t H, int32_t N_blk, int32_t P_blk, void* stream) {
    musaStream_t st = (musaStream_t)stream;
    dim3 grid(B * H, P_blk, 3);
    fused_vsa_tile_pool_bwd_d128<<<grid, 128, 0, st>>>(
        (const __mt_bfloat16*)dqt, (const __mt_bfloat16*)dkt, (const __mt_bfloat16*)dvt,
        (const __mt_bfloat16*)dqc, (const __mt_bfloat16*)dkc, (const __mt_bfloat16*)dvc,
        (const int32_t*)pad_to_raster, (const int32_t*)vbs,
        (__mt_bfloat16*)dq, (__mt_bfloat16*)dk, (__mt_bfloat16*)dv,
        (float*)p_dbv, S, H, N_blk, P_blk);
    if (p_dbv != nullptr && dbv != nullptr) {
        int32_t D_tot = H * 128;
        dim3 rgrid((D_tot + 31) / 32, 1);
        dim3 rblk(32, 8);
        reduce_partial_f32_to_bf16_kernel<<<rgrid, rblk, 0, st>>>((const float*)p_dbv, (__mt_bfloat16*)dbv, B, P_blk, D_tot);
    }
    return (int)musaGetLastError();
}

int launch_vsa_combine_untile_fwd(const void* out_s, const void* out_c, const void* gate,
                                  const void* pad_to_raster, const void* vbs,
                                  void* out_raster,
                                  int32_t B, int32_t S, int32_t H, int32_t N_blk, void* stream) {
    dim3 grid(N_blk, B * H);
    fused_vsa_combine_untile_fwd_d128<<<grid, 128, 0, (musaStream_t)stream>>>(
        (const __mt_bfloat16*)out_s, (const __mt_bfloat16*)out_c, (const __mt_bfloat16*)gate,
        (const int32_t*)pad_to_raster, (const int32_t*)vbs,
        (__mt_bfloat16*)out_raster, S, H, N_blk);
    return (int)musaGetLastError();
}

int launch_vsa_combine_untile_bwd(const void* dout, const void* out_c, const void* gate,
                                  const void* pad_to_raster, const void* vbs,
                                  void* d_out_s, void* d_out_c, void* d_gate,
                                  void* p_dbg, void* dbg,
                                  int32_t B, int32_t S, int32_t H, int32_t N_blk, int32_t P_blk, void* stream) {
    musaStream_t st = (musaStream_t)stream;
    dim3 grid(B * H, P_blk);
    fused_vsa_combine_untile_bwd_d128<<<grid, 128, 0, st>>>(
        (const __mt_bfloat16*)dout, (const __mt_bfloat16*)out_c, (const __mt_bfloat16*)gate,
        (const int32_t*)pad_to_raster, (const int32_t*)vbs,
        (__mt_bfloat16*)d_out_s, (__mt_bfloat16*)d_out_c, (__mt_bfloat16*)d_gate,
        (float*)p_dbg, S, H, N_blk, P_blk);
    if (gate != nullptr && p_dbg != nullptr && dbg != nullptr) {
        int32_t D_tot = H * 128;
        dim3 rgrid((D_tot + 31) / 32, 1);
        dim3 rblk(32, 8);
        reduce_partial_f32_to_bf16_kernel<<<rgrid, rblk, 0, st>>>((const float*)p_dbg, (__mt_bfloat16*)dbg, B, P_blk, D_tot);
    }
    return (int)musaGetLastError();
}

int launch_sum_dim0_bf16(const void* x, void* partial, void* out,
                         int32_t M, int32_t D, int32_t split_m, void* stream) {
    musaStream_t st = (musaStream_t)stream;
    dim3 g1((D + 255) / 256, split_m);
    dim3 b1(32, 8);
    sum_dim0_bf16_fast_s1<<<g1, b1, 0, st>>>((const __mt_bfloat16*)x, (float*)partial, M, D, split_m);
    sum_dim0_bf16_fast_s2<<<(D + 255) / 256, 256, 0, st>>>((const float*)partial, (__mt_bfloat16*)out, split_m, D);
    return (int)musaGetLastError();
}

int launch_gelu_tanh_fwd(const void* x, void* y, int64_t n_elems, void* stream) {
    int64_t n_u4 = n_elems >> 3;
    int32_t blocks = (int32_t)((n_u4 + 255) / 256);
    if (blocks > 480) blocks = 480;
    if (blocks < 1) blocks = 1;
    fused_gelu_tanh_fwd_u4_kernel<<<blocks, 256, 0, (musaStream_t)stream>>>((const uint4*)x, (uint4*)y, n_u4);
    return (int)musaGetLastError();
}

int launch_gelu_tanh_bwd_bias(const void* dy, const void* x, void* dx,
                              void* partial_db, void* db,
                              int32_t M, int32_t D, int32_t split_m, void* stream) {
    musaStream_t st = (musaStream_t)stream;
    dim3 g1((D + 255) / 256, split_m);
    dim3 b1(32, 8);
    fused_gelu_tanh_bwd_bias_s1<<<g1, b1, 0, st>>>(
        (const __mt_bfloat16*)dy, (const __mt_bfloat16*)x, (__mt_bfloat16*)dx,
        (float*)partial_db, M, D, split_m);
    if (partial_db != nullptr && db != nullptr) {
        sum_dim0_bf16_fast_s2<<<(D + 255) / 256, 256, 0, st>>>((const float*)partial_db, (__mt_bfloat16*)db, split_m, D);
    }
    return (int)musaGetLastError();
}

} // extern "C"
"""

_FUSED_LIB = None


def _is_dist_multi() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _is_local_leader() -> bool:
    if not _is_dist_multi():
        return True
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return int(local_rank) == 0
    return dist.get_rank() == 0


def _get_fused_lib():
    global _FUSED_LIB
    if _FUSED_LIB is not None:
        return _FUSED_LIB

    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "wan22_vsa_musa_ext"
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(_FUSED_CU_SOURCE.encode("utf-8")).hexdigest()[:16]
    so_path = cache_dir / f"wan22_fused_s5000_{digest}.so"

    def _compile_so():
        if so_path.exists():
            return
        pid = os.getpid()
        cu_tmp = cache_dir / f"wan22_fused_{digest}_{pid}.cu"
        so_tmp = cache_dir / f"wan22_fused_{digest}_{pid}.so"
        cu_tmp.write_text(_FUSED_CU_SOURCE)
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
    vp, i32, i64, f32 = ctypes.c_void_p, ctypes.c_int32, ctypes.c_int64, ctypes.c_float
    sigs = {
        "launch_adaln_fwd": [vp, vp, vp, vp, vp, vp, i32, i32, f32, vp],
        "launch_adaln_bwd": [vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, i32, i32, i32, vp],
        "launch_gated_res_ln_fwd": [vp, vp, vp, vp, vp, vp, vp, vp, vp, i32, i32, f32, vp],
        "launch_gated_res_ln_bwd": [vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, i32, i32, i32, vp],
        "launch_res_adaln_fwd": [vp, vp, vp, vp, vp, vp, vp, vp, i32, i32, f32, vp],
        "launch_res_adaln_bwd": [vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, i32, i32, i32, vp],
        "launch_gated_res_fwd": [vp, vp, vp, vp, i32, i32, vp],
        "launch_gated_res_bwd": [vp, vp, vp, vp, vp, vp, vp, vp, i32, i32, i32, vp],
        "launch_rmsnorm_rope_fwd": [vp, vp, vp, vp, vp, vp, i32, i32, f32, i32, vp],
        "launch_rmsnorm_rope_bwd": [vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, i32, i32, i32, i32, vp],
        "launch_vsa_tile_pool_fwd": [vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, i32, i32, i32, i32, vp],
        "launch_vsa_tile_pool_bwd": [vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, i32, i32, i32, i32, i32, vp],
        "launch_vsa_combine_untile_fwd": [vp, vp, vp, vp, vp, vp, i32, i32, i32, i32, vp],
        "launch_vsa_combine_untile_bwd": [vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, i32, i32, i32, i32, i32, vp],
        "launch_sum_dim0_bf16": [vp, vp, vp, i32, i32, i32, vp],
        "launch_gelu_tanh_fwd": [vp, vp, i64, vp],
        "launch_gelu_tanh_bwd_bias": [vp, vp, vp, vp, vp, i32, i32, i32, vp],
    }
    for name, argtypes in sigs.items():
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = ctypes.c_int
    _FUSED_LIB = lib
    return lib


def _stream(t: torch.Tensor) -> int:
    return torch.musa.current_stream(t.device).musa_stream


def _pick_p(S: int) -> int:
    return min(240, max(1, S))


class _FusedAdaLN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float, dx_add_box=None):
        x_c = x.contiguous()
        B, S, D = x_c.shape
        sc_f = scale.float().contiguous().view(B, D)
        sh_f = shift.float().contiguous().view(B, D)
        y = torch.empty_like(x_c)
        mean = torch.empty((B * S,), device=x.device, dtype=torch.float32)
        rstd = torch.empty((B * S,), device=x.device, dtype=torch.float32)
        rc = _get_fused_lib().launch_adaln_fwd(
            x_c.data_ptr(), sc_f.data_ptr(), sh_f.data_ptr(),
            y.data_ptr(), mean.data_ptr(), rstd.data_ptr(),
            B, S, float(eps), _stream(x_c),
        )
        if rc != 0:
            raise RuntimeError(f"launch_adaln_fwd MUSA error {rc}")
        ctx.save_for_backward(x_c, sc_f, mean, rstd)
        ctx.scale_shape = scale.shape
        ctx.scale_dtype = scale.dtype
        ctx.shift_dtype = shift.dtype
        ctx.dx_add_box = dx_add_box
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x_c, sc_f, mean, rstd = ctx.saved_tensors
        B, S, D = x_c.shape
        dy_c = dy.contiguous()
        dx_add = None
        if ctx.dx_add_box is not None:
            dx_add = ctx.dx_add_box[0]
            ctx.dx_add_box[0] = None
            if dx_add is not None:
                dx_add = dx_add.contiguous()
        P = _pick_p(S)
        dx = torch.empty_like(x_c)
        p_dsc = torch.empty((B, P, D), device=x_c.device, dtype=torch.float32)
        p_dsh = torch.empty((B, P, D), device=x_c.device, dtype=torch.float32)
        dsc = torch.empty((B, D), device=x_c.device, dtype=torch.float32)
        dsh = torch.empty((B, D), device=x_c.device, dtype=torch.float32)
        rc = _get_fused_lib().launch_adaln_bwd(
            dy_c.data_ptr(), x_c.data_ptr(), sc_f.data_ptr(),
            mean.data_ptr(), rstd.data_ptr(),
            dx_add.data_ptr() if dx_add is not None else 0,
            dx.data_ptr(), p_dsc.data_ptr(), p_dsh.data_ptr(),
            dsc.data_ptr(), dsh.data_ptr(),
            B, S, P, _stream(x_c),
        )
        if rc != 0:
            raise RuntimeError(f"launch_adaln_bwd MUSA error {rc}")
        return (
            dx,
            dsc.view(ctx.scale_shape).to(ctx.scale_dtype),
            dsh.view(ctx.scale_shape).to(ctx.shift_dtype),
            None,
            None,
        )


class _FusedGatedResLN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, upd: torch.Tensor, gate: torch.Tensor,
                weight: torch.Tensor, bias: torch.Tensor, eps: float, offload_ctx=None,
                db_upd_box=None, dx_out_box=None):
        x_c = x.contiguous()
        u_c = upd.contiguous()
        B, S, D = x_c.shape
        g_f = gate.float().contiguous().view(B, D)
        w_f = weight.float().contiguous().view(D)
        b_f = bias.float().contiguous().view(D)
        x_mid = torch.empty_like(x_c)
        ln_out = torch.empty_like(x_c)
        mean = torch.empty((B * S,), device=x.device, dtype=torch.float32)
        rstd = torch.empty((B * S,), device=x.device, dtype=torch.float32)
        rc = _get_fused_lib().launch_gated_res_ln_fwd(
            x_c.data_ptr(), u_c.data_ptr(), g_f.data_ptr(), w_f.data_ptr(), b_f.data_ptr(),
            x_mid.data_ptr(), ln_out.data_ptr(), mean.data_ptr(), rstd.data_ptr(),
            B, S, float(eps), _stream(x_c),
        )
        if rc != 0:
            raise RuntimeError(f"launch_gated_res_ln_fwd MUSA error {rc}")
        if offload_ctx is not None:
            pool, blk_idx = offload_ctx
            pool.offload((blk_idx, "gres"), (x_mid, u_c))
            ctx.save_for_backward(g_f, w_f, mean, rstd)
            ctx.offload_ctx = (pool, blk_idx)
        else:
            ctx.save_for_backward(x_mid, u_c, g_f, w_f, mean, rstd)
            ctx.offload_ctx = None
        ctx.gate_shape = gate.shape
        ctx.gate_dtype = gate.dtype
        ctx.w_dtype = weight.dtype
        ctx.b_dtype = bias.dtype
        ctx.db_upd_box = db_upd_box
        ctx.dx_out_box = dx_out_box
        return x_mid, ln_out

    @staticmethod
    def backward(ctx, d_xmid: torch.Tensor, d_ln: torch.Tensor):
        if ctx.offload_ctx is not None:
            pool, blk_idx = ctx.offload_ctx
            g_f, w_f, mean, rstd = ctx.saved_tensors
            x_mid, u_c = pool.fetch((blk_idx, "gres"))
        else:
            x_mid, u_c, g_f, w_f, mean, rstd = ctx.saved_tensors
        B, S, D = x_mid.shape
        dxm_c = d_xmid.contiguous() if d_xmid is not None else torch.zeros_like(x_mid)
        dln_c = d_ln.contiguous() if d_ln is not None else torch.zeros_like(x_mid)
        P = _pick_p(S)
        dx = torch.empty_like(x_mid)
        dupd = torch.empty_like(x_mid)
        p_dg = torch.empty((B, P, D), device=x_mid.device, dtype=torch.float32)
        p_dw = torch.empty((B, P, D), device=x_mid.device, dtype=torch.float32)
        p_db = torch.empty((B, P, D), device=x_mid.device, dtype=torch.float32)
        dg = torch.empty((B, D), device=x_mid.device, dtype=torch.float32)
        dw = torch.empty((D,), device=x_mid.device, dtype=torch.float32)
        db = torch.empty((D,), device=x_mid.device, dtype=torch.float32)
        if ctx.db_upd_box is not None:
            p_db_upd = torch.empty((B, P, D), device=x_mid.device, dtype=torch.float32)
            db_upd = torch.empty((D,), device=x_mid.device, dtype=torch.bfloat16)
            p_dbu_ptr, dbu_ptr = p_db_upd.data_ptr(), db_upd.data_ptr()
        else:
            db_upd = None
            p_dbu_ptr, dbu_ptr = 0, 0
        rc = _get_fused_lib().launch_gated_res_ln_bwd(
            dxm_c.data_ptr(), dln_c.data_ptr(), x_mid.data_ptr(), u_c.data_ptr(),
            g_f.data_ptr(), w_f.data_ptr(), mean.data_ptr(), rstd.data_ptr(),
            dx.data_ptr(), dupd.data_ptr(),
            p_dg.data_ptr(), p_dw.data_ptr(), p_db.data_ptr(), p_dbu_ptr,
            dg.data_ptr(), dw.data_ptr(), db.data_ptr(), dbu_ptr,
            B, S, P, _stream(x_mid),
        )
        if rc != 0:
            raise RuntimeError(f"launch_gated_res_ln_bwd MUSA error {rc}")
        if ctx.offload_ctx is not None:
            pool, _ = ctx.offload_ctx
            pool.release("gres")
            pool.release("ca")
        if ctx.db_upd_box is not None:
            ctx.db_upd_box[0] = db_upd
        if ctx.dx_out_box is not None:
            ctx.dx_out_box[0] = dx
            dx_ret = None
        else:
            dx_ret = dx
        return (
            dx_ret,
            dupd,
            dg.view(ctx.gate_shape).to(ctx.gate_dtype),
            dw.to(ctx.w_dtype),
            db.to(ctx.b_dtype),
            None,
            None,
            None,
            None,
        )


class _FusedResAdaLN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, upd: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor,
                eps: float, db_upd_box=None):
        x_c = x.contiguous()
        u_c = upd.contiguous()
        B, S, D = x_c.shape
        sc_f = scale.float().contiguous().view(B, D)
        sh_f = shift.float().contiguous().view(B, D)
        x_mid = torch.empty_like(x_c)
        y_out = torch.empty_like(x_c)
        mean = torch.empty((B * S,), device=x.device, dtype=torch.float32)
        rstd = torch.empty((B * S,), device=x.device, dtype=torch.float32)
        rc = _get_fused_lib().launch_res_adaln_fwd(
            x_c.data_ptr(), u_c.data_ptr(), sc_f.data_ptr(), sh_f.data_ptr(),
            x_mid.data_ptr(), y_out.data_ptr(), mean.data_ptr(), rstd.data_ptr(),
            B, S, float(eps), _stream(x_c),
        )
        if rc != 0:
            raise RuntimeError(f"launch_res_adaln_fwd MUSA error {rc}")
        ctx.save_for_backward(x_mid, sc_f, mean, rstd)
        ctx.scale_shape = scale.shape
        ctx.scale_dtype = scale.dtype
        ctx.shift_dtype = shift.dtype
        ctx.db_upd_box = db_upd_box
        return x_mid, y_out

    @staticmethod
    def backward(ctx, d_xmid: torch.Tensor, dy: torch.Tensor):
        x_mid, sc_f, mean, rstd = ctx.saved_tensors
        B, S, D = x_mid.shape
        dxm_c = d_xmid.contiguous() if d_xmid is not None else torch.zeros_like(x_mid)
        dy_c = dy.contiguous() if dy is not None else torch.zeros_like(x_mid)
        P = _pick_p(S)
        dx = torch.empty_like(x_mid)
        p_dsc = torch.empty((B, P, D), device=x_mid.device, dtype=torch.float32)
        p_dsh = torch.empty((B, P, D), device=x_mid.device, dtype=torch.float32)
        dsc = torch.empty((B, D), device=x_mid.device, dtype=torch.float32)
        dsh = torch.empty((B, D), device=x_mid.device, dtype=torch.float32)
        if ctx.db_upd_box is not None:
            p_db_upd = torch.empty((B, P, D), device=x_mid.device, dtype=torch.float32)
            db_upd = torch.empty((D,), device=x_mid.device, dtype=torch.bfloat16)
            p_dbu_ptr, dbu_ptr = p_db_upd.data_ptr(), db_upd.data_ptr()
        else:
            db_upd = None
            p_dbu_ptr, dbu_ptr = 0, 0
        rc = _get_fused_lib().launch_res_adaln_bwd(
            dxm_c.data_ptr(), dy_c.data_ptr(), x_mid.data_ptr(), sc_f.data_ptr(),
            mean.data_ptr(), rstd.data_ptr(),
            dx.data_ptr(), p_dsc.data_ptr(), p_dsh.data_ptr(), p_dbu_ptr,
            dsc.data_ptr(), dsh.data_ptr(), dbu_ptr,
            B, S, P, _stream(x_mid),
        )
        if rc != 0:
            raise RuntimeError(f"launch_res_adaln_bwd MUSA error {rc}")
        if ctx.db_upd_box is not None:
            ctx.db_upd_box[0] = db_upd
        return (
            dx,
            dx,
            dsc.view(ctx.scale_shape).to(ctx.scale_dtype),
            dsh.view(ctx.scale_shape).to(ctx.shift_dtype),
            None,
            None,
        )


class _FusedGatedRes(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, upd: torch.Tensor, gate: torch.Tensor, db_upd_box=None):
        x_c = x.contiguous()
        u_c = upd.contiguous()
        B, S, D = x_c.shape
        g_f = gate.float().contiguous().view(B, D)
        out = torch.empty_like(x_c)
        rc = _get_fused_lib().launch_gated_res_fwd(
            x_c.data_ptr(), u_c.data_ptr(), g_f.data_ptr(),
            out.data_ptr(), B, S, _stream(x_c),
        )
        if rc != 0:
            raise RuntimeError(f"launch_gated_res_fwd MUSA error {rc}")
        ctx.save_for_backward(u_c, g_f)
        ctx.gate_shape = gate.shape
        ctx.gate_dtype = gate.dtype
        ctx.db_upd_box = db_upd_box
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        u_c, g_f = ctx.saved_tensors
        B, S, D = u_c.shape
        do_c = dout.contiguous()
        P = min(120, max(1, S))
        dupd = torch.empty_like(u_c)
        p_dg = torch.empty((B, P, D), device=u_c.device, dtype=torch.float32)
        dg = torch.empty((B, D), device=u_c.device, dtype=torch.float32)
        if ctx.db_upd_box is not None:
            p_db_upd = torch.empty((B, P, D), device=u_c.device, dtype=torch.float32)
            db_upd = torch.empty((D,), device=u_c.device, dtype=torch.bfloat16)
            p_dbu_ptr, dbu_ptr = p_db_upd.data_ptr(), db_upd.data_ptr()
        else:
            db_upd = None
            p_dbu_ptr, dbu_ptr = 0, 0
        rc = _get_fused_lib().launch_gated_res_bwd(
            do_c.data_ptr(), u_c.data_ptr(), g_f.data_ptr(),
            dupd.data_ptr(), p_dg.data_ptr(), p_dbu_ptr, dg.data_ptr(), dbu_ptr,
            B, S, P, _stream(u_c),
        )
        if rc != 0:
            raise RuntimeError(f"launch_gated_res_bwd MUSA error {rc}")
        if ctx.db_upd_box is not None:
            ctx.db_upd_box[0] = db_upd
        return do_c, dupd, dg.view(ctx.gate_shape).to(ctx.gate_dtype), None


class _FusedRMSNormRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor,
                cos: Optional[torch.Tensor], sin: Optional[torch.Tensor], eps: float, db_x_box=None):
        x_c = x.contiguous()
        B, S, D = x_c.shape
        w_f = weight.float().contiguous().view(D)
        do_rope = 1 if (cos is not None and sin is not None) else 0
        cs_f = cos.float().contiguous() if do_rope else None
        sn_f = sin.float().contiguous() if do_rope else None
        out = torch.empty_like(x_c)
        rstd = torch.empty((B * S,), device=x.device, dtype=torch.float32)
        rc = _get_fused_lib().launch_rmsnorm_rope_fwd(
            x_c.data_ptr(), w_f.data_ptr(),
            cs_f.data_ptr() if do_rope else 0,
            sn_f.data_ptr() if do_rope else 0,
            out.data_ptr(), rstd.data_ptr(),
            B, S, float(eps), do_rope, _stream(x_c),
        )
        if rc != 0:
            raise RuntimeError(f"launch_rmsnorm_rope_fwd MUSA error {rc}")
        if do_rope:
            ctx.save_for_backward(x_c, w_f, rstd, cs_f, sn_f)
        else:
            ctx.save_for_backward(x_c, w_f, rstd)
        ctx.do_rope = do_rope
        ctx.w_dtype = weight.dtype
        ctx.db_x_box = db_x_box
        return out

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        do_rope = ctx.do_rope
        if do_rope:
            x_c, w_f, rstd, cs_f, sn_f = ctx.saved_tensors
        else:
            x_c, w_f, rstd = ctx.saved_tensors
            cs_f = sn_f = None
        B, S, D = x_c.shape
        dy_c = dy.contiguous()
        P = _pick_p(S)
        dx = torch.empty_like(x_c)
        p_dw = torch.empty((B, P, D), device=x_c.device, dtype=torch.float32)
        dw = torch.empty((D,), device=x_c.device, dtype=torch.float32)
        if ctx.db_x_box is not None:
            p_db_x = torch.empty((B, P, D), device=x_c.device, dtype=torch.float32)
            db_x = torch.empty((D,), device=x_c.device, dtype=torch.bfloat16)
            p_dbx_ptr, dbx_ptr = p_db_x.data_ptr(), db_x.data_ptr()
        else:
            db_x = None
            p_dbx_ptr, dbx_ptr = 0, 0
        rc = _get_fused_lib().launch_rmsnorm_rope_bwd(
            dy_c.data_ptr(), x_c.data_ptr(), w_f.data_ptr(),
            cs_f.data_ptr() if do_rope else 0,
            sn_f.data_ptr() if do_rope else 0,
            rstd.data_ptr(),
            dx.data_ptr(), p_dw.data_ptr(), p_dbx_ptr, dw.data_ptr(), dbx_ptr,
            B, S, P, do_rope, _stream(x_c),
        )
        if rc != 0:
            raise RuntimeError(f"launch_rmsnorm_rope_bwd MUSA error {rc}")
        if ctx.db_x_box is not None:
            ctx.db_x_box[0] = db_x
        return dx, dw.to(ctx.w_dtype), None, None, None, None


class _FusedVSATilePool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                pad_to_raster: torch.Tensor, vbs: torch.Tensor,
                out_qkv=None, dbv_box=None):
        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        B, S, H, D = q_c.shape
        N_blk = int(vbs.numel())
        S_pad = N_blk * 64
        p2r = pad_to_raster.to(torch.int32).contiguous()
        vbs32 = vbs.to(torch.int32).contiguous()

        if out_qkv is not None:
            qt = out_qkv[0].detach().view(B, H, S_pad, D)
            kt = out_qkv[1].detach().view(B, H, S_pad, D)
            vt = out_qkv[2].detach().view(B, H, S_pad, D)
        else:
            qt = torch.empty((B, H, S_pad, D), device=q.device, dtype=q.dtype)
            kt = torch.empty((B, H, S_pad, D), device=q.device, dtype=q.dtype)
            vt = torch.empty((B, H, S_pad, D), device=q.device, dtype=q.dtype)
        qc = torch.empty((B, H, N_blk, D), device=q.device, dtype=q.dtype)
        kc = torch.empty((B, H, N_blk, D), device=q.device, dtype=q.dtype)
        vc = torch.empty((B, H, N_blk, D), device=q.device, dtype=q.dtype)

        rc = _get_fused_lib().launch_vsa_tile_pool_fwd(
            q_c.data_ptr(), k_c.data_ptr(), v_c.data_ptr(),
            p2r.data_ptr(), vbs32.data_ptr(),
            qt.data_ptr(), kt.data_ptr(), vt.data_ptr(),
            qc.data_ptr(), kc.data_ptr(), vc.data_ptr(),
            B, S, H, N_blk, _stream(q_c),
        )
        if rc != 0:
            raise RuntimeError(f"launch_vsa_tile_pool_fwd MUSA error {rc}")
        ctx.save_for_backward(p2r, vbs32)
        ctx.meta = (B, S, H, D, N_blk, q.dtype, q.device)
        ctx.dbv_box = dbv_box
        return qt, kt, vt, qc, kc, vc

    @staticmethod
    def backward(ctx, dqt, dkt, dvt, dqc, dkc, dvc):
        p2r, vbs32 = ctx.saved_tensors
        B, S, H, D, N_blk, dtype, device = ctx.meta
        S_pad = N_blk * 64
        dqt_c = dqt.contiguous() if dqt is not None else torch.zeros((B, H, S_pad, D), device=device, dtype=dtype)
        dkt_c = dkt.contiguous() if dkt is not None else torch.zeros((B, H, S_pad, D), device=device, dtype=dtype)
        dvt_c = dvt.contiguous() if dvt is not None else torch.zeros((B, H, S_pad, D), device=device, dtype=dtype)
        dqc_c = dqc.contiguous() if dqc is not None else torch.zeros((B, H, N_blk, D), device=device, dtype=dtype)
        dkc_c = dkc.contiguous() if dkc is not None else torch.zeros((B, H, N_blk, D), device=device, dtype=dtype)
        dvc_c = dvc.contiguous() if dvc is not None else torch.zeros((B, H, N_blk, D), device=device, dtype=dtype)

        dq = torch.empty((B, S, H, D), device=device, dtype=dtype)
        dk = torch.empty((B, S, H, D), device=device, dtype=dtype)
        dv = torch.empty((B, S, H, D), device=device, dtype=dtype)
        P_blk = min(240, max(1, N_blk))
        if ctx.dbv_box is not None:
            p_dbv = torch.empty((B, P_blk, H * D), device=device, dtype=torch.float32)
            dbv = torch.empty((H * D,), device=device, dtype=torch.bfloat16)
            p_ptr, db_ptr = p_dbv.data_ptr(), dbv.data_ptr()
        else:
            dbv = None
            p_ptr, db_ptr = 0, 0
        rc = _get_fused_lib().launch_vsa_tile_pool_bwd(
            dqt_c.data_ptr(), dkt_c.data_ptr(), dvt_c.data_ptr(),
            dqc_c.data_ptr(), dkc_c.data_ptr(), dvc_c.data_ptr(),
            p2r.data_ptr(), vbs32.data_ptr(),
            dq.data_ptr(), dk.data_ptr(), dv.data_ptr(),
            p_ptr, db_ptr,
            B, S, H, N_blk, P_blk, _stream(dq),
        )
        if rc != 0:
            raise RuntimeError(f"launch_vsa_tile_pool_bwd MUSA error {rc}")
        if ctx.dbv_box is not None:
            ctx.dbv_box[0] = dbv
        return dq, dk, dv, None, None, None, None


class _FusedVSACombineUntile(torch.autograd.Function):
    @staticmethod
    def forward(ctx, out_s: torch.Tensor, out_c: torch.Tensor, gate: Optional[torch.Tensor],
                pad_to_raster: torch.Tensor, vbs: torch.Tensor, S: int, dbg_box=None):
        os_c = out_s.contiguous()
        oc_c = out_c.contiguous()
        B, H, S_pad, D = os_c.shape
        N_blk = S_pad // 64
        has_gate = gate is not None
        g_c = gate.contiguous() if has_gate else None
        p2r = pad_to_raster.to(torch.int32).contiguous()
        vbs32 = vbs.to(torch.int32).contiguous()

        out_r = torch.empty((B, S, H, D), device=os_c.device, dtype=os_c.dtype)
        rc = _get_fused_lib().launch_vsa_combine_untile_fwd(
            os_c.data_ptr(), oc_c.data_ptr(),
            g_c.data_ptr() if has_gate else 0,
            p2r.data_ptr(), vbs32.data_ptr(),
            out_r.data_ptr(),
            B, S, H, N_blk, _stream(os_c),
        )
        if rc != 0:
            raise RuntimeError(f"launch_vsa_combine_untile_fwd MUSA error {rc}")
        if has_gate:
            ctx.save_for_backward(oc_c, g_c, p2r, vbs32)
        else:
            ctx.save_for_backward(oc_c, p2r, vbs32)
        ctx.has_gate = has_gate
        ctx.meta = (B, S, H, D, N_blk, os_c.dtype, os_c.device)
        ctx.dbg_box = dbg_box
        return out_r

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        has_gate = ctx.has_gate
        if has_gate:
            oc_c, g_c, p2r, vbs32 = ctx.saved_tensors
        else:
            oc_c, p2r, vbs32 = ctx.saved_tensors
            g_c = None
        B, S, H, D, N_blk, dtype, device = ctx.meta
        do_c = dout.contiguous()
        d_out_s = torch.empty((B, H, N_blk * 64, D), device=device, dtype=dtype)
        d_out_c = torch.empty((B, H, N_blk, D), device=device, dtype=dtype)
        d_gate = torch.empty((B, S, H, D), device=device, dtype=dtype) if has_gate else None
        P_blk = min(240, max(1, N_blk))
        if has_gate and ctx.dbg_box is not None:
            p_dbg = torch.empty((B, P_blk, H * D), device=device, dtype=torch.float32)
            dbg = torch.empty((H * D,), device=device, dtype=torch.bfloat16)
            p_ptr, db_ptr = p_dbg.data_ptr(), dbg.data_ptr()
        else:
            dbg = None
            p_ptr, db_ptr = 0, 0

        rc = _get_fused_lib().launch_vsa_combine_untile_bwd(
            do_c.data_ptr(), oc_c.data_ptr(),
            g_c.data_ptr() if has_gate else 0,
            p2r.data_ptr(), vbs32.data_ptr(),
            d_out_s.data_ptr(), d_out_c.data_ptr(),
            d_gate.data_ptr() if has_gate else 0,
            p_ptr, db_ptr,
            B, S, H, N_blk, P_blk, _stream(do_c),
        )
        if rc != 0:
            raise RuntimeError(f"launch_vsa_combine_untile_bwd MUSA error {rc}")
        if has_gate and ctx.dbg_box is not None:
            ctx.dbg_box[0] = dbg
        return d_out_s, d_out_c, d_gate, None, None, None, None


def can_use_fused_d3072(x: torch.Tensor) -> bool:
    return x.device.type == "musa" and x.dtype == torch.bfloat16 and x.ndim == 3 and x.shape[-1] == 3072


def fused_adaln(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float, dx_add_box=None) -> torch.Tensor:
    return _FusedAdaLN.apply(x, scale, shift, eps, dx_add_box)


def fused_gated_res_ln(
    x: torch.Tensor, upd: torch.Tensor, gate: torch.Tensor,
    weight: torch.Tensor, bias: torch.Tensor, eps: float, offload_ctx=None,
    db_upd_box=None, dx_out_box=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _FusedGatedResLN.apply(x, upd, gate, weight, bias, eps, offload_ctx, db_upd_box, dx_out_box)


def fused_res_adaln(
    x: torch.Tensor, upd: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float,
    db_upd_box=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _FusedResAdaLN.apply(x, upd, scale, shift, eps, db_upd_box)


def fused_gated_res(x: torch.Tensor, upd: torch.Tensor, gate: torch.Tensor, db_upd_box=None) -> torch.Tensor:
    return _FusedGatedRes.apply(x, upd, gate, db_upd_box)


def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float, db_x_box=None) -> torch.Tensor:
    return _FusedRMSNormRoPE.apply(x, weight, None, None, eps, db_x_box)


class _FusedQKRMSNormRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_raw: torch.Tensor, k_raw: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor, eps_q: float, eps_k: float, offload_ctx=None,
                dbq_box=None, dbk_box=None):
        qc = q_raw.contiguous()
        kc = k_raw.contiguous()
        B, S, D = qc.shape
        wq_f = wq.float().contiguous().view(D)
        wk_f = wk.float().contiguous().view(D)
        cs_f = cos.float().contiguous()
        sn_f = sin.float().contiguous()
        q_out = torch.empty_like(qc)
        k_out = torch.empty_like(kc)
        rstd_q = torch.empty((B * S,), device=qc.device, dtype=torch.float32)
        rstd_k = torch.empty((B * S,), device=kc.device, dtype=torch.float32)
        lib = _get_fused_lib()
        st = _stream(qc)
        rc1 = lib.launch_rmsnorm_rope_fwd(
            qc.data_ptr(), wq_f.data_ptr(), cs_f.data_ptr(), sn_f.data_ptr(),
            q_out.data_ptr(), rstd_q.data_ptr(), B, S, float(eps_q), 1, st,
        )
        rc2 = lib.launch_rmsnorm_rope_fwd(
            kc.data_ptr(), wk_f.data_ptr(), cs_f.data_ptr(), sn_f.data_ptr(),
            k_out.data_ptr(), rstd_k.data_ptr(), B, S, float(eps_k), 1, st,
        )
        if rc1 != 0 or rc2 != 0:
            raise RuntimeError(f"launch_rmsnorm_rope_fwd MUSA error {rc1}/{rc2}")
        if offload_ctx is not None:
            pool, blk_idx = offload_ctx
            pool.offload((blk_idx, "qk_raw"), (qc, kc))
            ctx.save_for_backward(wq_f, wk_f, rstd_q, rstd_k, cs_f, sn_f)
            ctx.offload_ctx = (pool, blk_idx)
        else:
            ctx.save_for_backward(qc, kc, wq_f, wk_f, rstd_q, rstd_k, cs_f, sn_f)
            ctx.offload_ctx = None
        ctx.wq_dtype = wq.dtype
        ctx.wk_dtype = wk.dtype
        ctx.dbq_box = dbq_box
        ctx.dbk_box = dbk_box
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq_out: torch.Tensor, dk_out: torch.Tensor):
        if ctx.offload_ctx is not None:
            pool, blk_idx = ctx.offload_ctx
            wq_f, wk_f, rstd_q, rstd_k, cs_f, sn_f = ctx.saved_tensors
            qc, kc = pool.fetch((blk_idx, "qk_raw"))
        else:
            qc, kc, wq_f, wk_f, rstd_q, rstd_k, cs_f, sn_f = ctx.saved_tensors
        B, S, D = qc.shape
        dq_c = dq_out.contiguous()
        dk_c = dk_out.contiguous()
        P = _pick_p(S)
        dxq = torch.empty_like(qc)
        dxk = torch.empty_like(kc)
        p_dw = torch.empty((B, P, D), device=qc.device, dtype=torch.float32)
        dwq = torch.empty((D,), device=qc.device, dtype=torch.float32)
        dwk = torch.empty((D,), device=kc.device, dtype=torch.float32)
        p_db = torch.empty((B, P, D), device=qc.device, dtype=torch.float32) if (ctx.dbq_box is not None or ctx.dbk_box is not None) else None
        dbq = torch.empty((D,), device=qc.device, dtype=torch.bfloat16) if ctx.dbq_box is not None else None
        dbk = torch.empty((D,), device=kc.device, dtype=torch.bfloat16) if ctx.dbk_box is not None else None
        lib = _get_fused_lib()
        st = _stream(qc)
        rc1 = lib.launch_rmsnorm_rope_bwd(
            dq_c.data_ptr(), qc.data_ptr(), wq_f.data_ptr(), cs_f.data_ptr(), sn_f.data_ptr(),
            rstd_q.data_ptr(), dxq.data_ptr(), p_dw.data_ptr(),
            p_db.data_ptr() if dbq is not None else 0,
            dwq.data_ptr(), dbq.data_ptr() if dbq is not None else 0,
            B, S, P, 1, st,
        )
        rc2 = lib.launch_rmsnorm_rope_bwd(
            dk_c.data_ptr(), kc.data_ptr(), wk_f.data_ptr(), cs_f.data_ptr(), sn_f.data_ptr(),
            rstd_k.data_ptr(), dxk.data_ptr(), p_dw.data_ptr(),
            p_db.data_ptr() if dbk is not None else 0,
            dwk.data_ptr(), dbk.data_ptr() if dbk is not None else 0,
            B, S, P, 1, st,
        )
        if rc1 != 0 or rc2 != 0:
            raise RuntimeError(f"launch_rmsnorm_rope_bwd MUSA error {rc1}/{rc2}")
        if ctx.offload_ctx is not None:
            pool, _ = ctx.offload_ctx
            pool.release("qk_raw")
        if ctx.dbq_box is not None:
            ctx.dbq_box[0] = dbq
        if ctx.dbk_box is not None:
            ctx.dbk_box[0] = dbk
        return dxq, dxk, dwq.to(ctx.wq_dtype), dwk.to(ctx.wk_dtype), None, None, None, None, None, None, None


def fused_rmsnorm_rope(
    x: torch.Tensor, weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, eps: float
) -> torch.Tensor:
    return _FusedRMSNormRoPE.apply(x, weight, cos, sin, eps, None)


def fused_qk_rmsnorm_rope(
    q_raw: torch.Tensor, k_raw: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor, eps_q: float, eps_k: float, offload_ctx=None,
    dbq_box=None, dbk_box=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _FusedQKRMSNormRoPE.apply(
        q_raw, k_raw, wq, wk, cos, sin, eps_q, eps_k, offload_ctx, dbq_box, dbk_box
    )


def fused_gelu_tanh_fwd(x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Bitwise-exact MTT S5000 BF16 GELU-tanh forward with optional static output buffer (zero alloc)."""
    xc = x.contiguous()
    if xc.device.type != "musa" or xc.dtype != torch.bfloat16 or (xc.numel() & 7) != 0:
        y = torch.nn.functional.gelu(xc, approximate="tanh")
        if out is not None:
            out.copy_(y)
            return out
        return y
    y = out if (out is not None and out.shape == xc.shape and out.dtype == xc.dtype and out.is_contiguous()) else torch.empty_like(xc)
    rc = _get_fused_lib().launch_gelu_tanh_fwd(xc.data_ptr(), y.data_ptr(), xc.numel(), _stream(xc))
    if rc != 0:
        raise RuntimeError(f"launch_gelu_tanh_fwd MUSA error {rc}")
    return y


def fused_gelu_tanh_bwd_bias(
    dy: torch.Tensor,
    x: torch.Tensor,
    need_bias_grad: bool = True,
    dx_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Fused GELU-tanh backward + deterministic FP32 bias-grad sum over dim 0 (zero extra HBM read)."""
    dy_c = dy.contiguous()
    x_c = x.contiguous()
    D = x_c.shape[-1]
    M = x_c.numel() // D
    if (
        x_c.device.type != "musa"
        or x_c.dtype != torch.bfloat16
        or (D & 7) != 0
        or M < 64
    ):
        with torch.enable_grad():
            xr = x_c.detach().requires_grad_(True)
            a = torch.nn.functional.gelu(xr, approximate="tanh")
        (dx,) = torch.autograd.grad(a, xr, dy_c)
        db = dx.reshape(-1, D).sum(dim=0) if need_bias_grad else None
        return dx, db
    dx = dx_out if (dx_out is not None and dx_out.shape == x_c.shape and dx_out.dtype == x_c.dtype and dx_out.is_contiguous()) else torch.empty_like(x_c)
    split_m = 4 if D >= 8192 else 16
    if need_bias_grad:
        partial = torch.empty((split_m, D), device=x_c.device, dtype=torch.float32)
        db = torch.empty((D,), device=x_c.device, dtype=torch.bfloat16)
        p_ptr, db_ptr = partial.data_ptr(), db.data_ptr()
    else:
        db = None
        p_ptr, db_ptr = 0, 0
    rc = _get_fused_lib().launch_gelu_tanh_bwd_bias(
        dy_c.data_ptr(), x_c.data_ptr(), dx.data_ptr(),
        p_ptr, db_ptr, M, D, split_m, _stream(x_c),
    )
    if rc != 0:
        raise RuntimeError(f"launch_gelu_tanh_bwd_bias MUSA error {rc}")
    return dx, db


def fused_vsa_tile_pool(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    pad_to_raster: torch.Tensor, vbs: torch.Tensor,
    out_qkv=None, dbv_box=None,
):
    return _FusedVSATilePool.apply(q, k, v, pad_to_raster, vbs, out_qkv, dbv_box)


def fused_vsa_combine_untile(
    out_s: torch.Tensor, out_c: torch.Tensor, gate: Optional[torch.Tensor],
    pad_to_raster: torch.Tensor, vbs: torch.Tensor, S: int, dbg_box=None,
) -> torch.Tensor:
    return _FusedVSACombineUntile.apply(out_s, out_c, gate, pad_to_raster, vbs, S, dbg_box)


def fused_sum_dim0_bf16(x2d: torch.Tensor) -> torch.Tensor:
    """Fast deterministic MTT S5000 BF16 bias-grad sum over dim 0."""
    if (
        x2d.device.type != "musa"
        or x2d.dtype != torch.bfloat16
        or x2d.ndim != 2
        or x2d.shape[0] < 128
        or (x2d.shape[1] & 7) != 0
    ):
        return x2d.sum(dim=0)
    xc = x2d.contiguous()
    M, D = xc.shape
    split_m = 20 if D <= 4096 else 8
    partial = torch.empty((split_m, D), device=xc.device, dtype=torch.float32)
    out = torch.empty((D,), device=xc.device, dtype=torch.bfloat16)
    rc = _get_fused_lib().launch_sum_dim0_bf16(
        xc.data_ptr(), partial.data_ptr(), out.data_ptr(), M, D, split_m, _stream(xc)
    )
    if rc != 0:
        raise RuntimeError(f"launch_sum_dim0_bf16 MUSA error {rc}")
    return out
