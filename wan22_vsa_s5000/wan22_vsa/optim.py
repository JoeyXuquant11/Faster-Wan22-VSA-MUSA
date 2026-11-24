"""Single-pass fused FP32 AdamW optimizer kernel for Moore Threads MTT S5000.

Replaces PyTorch's 5-pass multi-tensor foreach AdamW with a single-pass MUSA C++ chunked
multi-tensor kernel compiled with strict IEEE-754 FP32 (no FTZ) for exact numerical parity
with torch.optim.AdamW on FSDP2 DTensor shards.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
import subprocess
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

_ADAMW_CU_SOURCE = r"""
#include <musa_runtime.h>
#include <stdint.h>
#include <math.h>

__global__ void fused_adamw_f32_chunk_kernel(
    float* const* __restrict__ p_ptrs,
    const float* const* __restrict__ g_ptrs,
    float* const* __restrict__ m_ptrs,
    float* const* __restrict__ v_ptrs,
    const int64_t* __restrict__ sizes,
    const int32_t* __restrict__ chunk_tid,
    const int64_t* __restrict__ chunk_off,
    int32_t num_chunks,
    float beta1,
    float beta2,
    float one_minus_b1,
    float one_minus_b2,
    float step_size,
    float inv_sqrt_bc2,
    float wd_scale,
    float eps
) {
    int32_t cid = blockIdx.x;
    if (cid >= num_chunks) return;
    int32_t t_idx = chunk_tid[cid];
    int64_t start = chunk_off[cid];
    int64_t sz = sizes[t_idx];
    int64_t end = start + 65536;
    if (end > sz) end = sz;

    float* __restrict__ p = p_ptrs[t_idx];
    const float* __restrict__ g = g_ptrs[t_idx];
    float* __restrict__ m = m_ptrs[t_idx];
    float* __restrict__ v = v_ptrs[t_idx];

    int64_t tid = threadIdx.x;
    int64_t n = end - start;
    int64_t n4 = n >> 2;
    if (((uintptr_t)(p + start) & 15) == 0 && ((uintptr_t)(g + start) & 15) == 0 &&
        ((uintptr_t)(m + start) & 15) == 0 && ((uintptr_t)(v + start) & 15) == 0) {
        float4* p4 = (float4*)(p + start);
        const float4* g4 = (const float4*)(g + start);
        float4* m4 = (float4*)(m + start);
        float4* v4 = (float4*)(v + start);
        for (int64_t i = tid; i < n4; i += 256) {
            float4 pv = p4[i];
            float4 gv = g4[i];
            float4 mv = m4[i];
            float4 vv = v4[i];
            mv.x = beta1 * mv.x + one_minus_b1 * gv.x;
            mv.y = beta1 * mv.y + one_minus_b1 * gv.y;
            mv.z = beta1 * mv.z + one_minus_b1 * gv.z;
            mv.w = beta1 * mv.w + one_minus_b1 * gv.w;
            vv.x = beta2 * vv.x + one_minus_b2 * gv.x * gv.x;
            vv.y = beta2 * vv.y + one_minus_b2 * gv.y * gv.y;
            vv.z = beta2 * vv.z + one_minus_b2 * gv.z * gv.z;
            vv.w = beta2 * vv.w + one_minus_b2 * gv.w * gv.w;
            pv.x = pv.x * wd_scale - step_size * (mv.x / (sqrtf(vv.x) * inv_sqrt_bc2 + eps));
            pv.y = pv.y * wd_scale - step_size * (mv.y / (sqrtf(vv.y) * inv_sqrt_bc2 + eps));
            pv.z = pv.z * wd_scale - step_size * (mv.z / (sqrtf(vv.z) * inv_sqrt_bc2 + eps));
            pv.w = pv.w * wd_scale - step_size * (mv.w / (sqrtf(vv.w) * inv_sqrt_bc2 + eps));
            p4[i] = pv;
            m4[i] = mv;
            v4[i] = vv;
        }
        for (int64_t i = (n4 << 2) + tid; i < n; i += 256) {
            int64_t idx = start + i;
            float gv = g[idx];
            float mv = beta1 * m[idx] + one_minus_b1 * gv;
            float vv = beta2 * v[idx] + one_minus_b2 * gv * gv;
            p[idx] = p[idx] * wd_scale - step_size * (mv / (sqrtf(vv) * inv_sqrt_bc2 + eps));
            m[idx] = mv;
            v[idx] = vv;
        }
    } else {
        for (int64_t i = start + tid; i < end; i += 256) {
            float gv = g[i];
            float mv = beta1 * m[i] + one_minus_b1 * gv;
            float vv = beta2 * v[i] + one_minus_b2 * gv * gv;
            p[i] = p[i] * wd_scale - step_size * (mv / (sqrtf(vv) * inv_sqrt_bc2 + eps));
            m[i] = mv;
            v[i] = vv;
        }
    }
}

extern "C" int launch_fused_adamw_f32(
    const void* p_ptrs,
    const void* g_ptrs,
    const void* m_ptrs,
    const void* v_ptrs,
    const void* sizes,
    const void* chunk_tid,
    const void* chunk_off,
    int32_t num_chunks,
    float beta1,
    float beta2,
    float one_minus_b1,
    float one_minus_b2,
    float step_size,
    float inv_sqrt_bc2,
    float wd_scale,
    float eps,
    void* stream
) {
    if (num_chunks <= 0) return 0;
    fused_adamw_f32_chunk_kernel<<<num_chunks, 256, 0, (musaStream_t)stream>>>(
        (float* const*)p_ptrs,
        (const float* const*)g_ptrs,
        (float* const*)m_ptrs,
        (float* const*)v_ptrs,
        (const int64_t*)sizes,
        (const int32_t*)chunk_tid,
        (const int64_t*)chunk_off,
        num_chunks,
        beta1,
        beta2,
        one_minus_b1,
        one_minus_b2,
        step_size,
        inv_sqrt_bc2,
        wd_scale,
        eps
    );
    return (int)musaGetLastError();
}
"""

_ADAMW_LIB = None


def _get_adamw_lib():
    global _ADAMW_LIB
    if _ADAMW_LIB is not None:
        return _ADAMW_LIB

    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "wan22_vsa_musa_ext"
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(_ADAMW_CU_SOURCE.encode("utf-8")).hexdigest()[:16]
    so_path = cache_dir / f"fused_adamw_s5000_{digest}.so"

    def _compile():
        if so_path.exists():
            return
        pid = os.getpid()
        cu_tmp = cache_dir / f"fused_adamw_{digest}_{pid}.cu"
        so_tmp = cache_dir / f"fused_adamw_{digest}_{pid}.so"
        cu_tmp.write_text(_ADAMW_CU_SOURCE)
        # Strict IEEE-754 FP32 (intentionally omit -fmusa-flush-denormals-to-zero for AdamW exp_avg_sq)
        cmd = [
            "/usr/local/musa/bin/mcc",
            "--musa-path=/usr/local/musa",
            "-x", "musa",
            "--offload-arch=mp_31",
            "-O3",
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

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        local_rank = os.environ.get("LOCAL_RANK")
        is_leader = (int(local_rank) == 0) if local_rank is not None else (dist.get_rank() == 0)
        if is_leader:
            _compile()
        dist.barrier()
    else:
        _compile()

    lib = ctypes.CDLL(str(so_path))
    lib.launch_fused_adamw_f32.argtypes = (
        [ctypes.c_void_p] * 7 + [ctypes.c_int32] + [ctypes.c_float] * 8 + [ctypes.c_void_p]
    )
    lib.launch_fused_adamw_f32.restype = ctypes.c_int
    _ADAMW_LIB = lib
    return lib


def _unwrap_local(t: torch.Tensor) -> torch.Tensor:
    if isinstance(t, DTensor):
        return t._local_tensor
    return t


class FusedMusaAdamW(torch.optim.AdamW):
    """Drop-in torch.optim.AdamW using a 1-pass MUSA C++ fused FP32 multi-tensor kernel on MTT S5000."""

    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self._meta_cache: dict[int, Tuple[Any, ...]] = {}

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            return super().step(closure)

        # Verify all groups are eligible for the fast FP32 MUSA path
        for group in self.param_groups:
            if group.get("amsgrad", False) or group.get("maximize", False) or group.get("capturable", False):
                return super().step(closure)

        lib = _get_adamw_lib()
        for g_idx, group in enumerate(self.param_groups):
            beta1, beta2 = group["betas"]
            lr = float(group["lr"])
            wd = float(group["weight_decay"])
            eps = float(group["eps"])

            p_locals: List[torch.Tensor] = []
            g_locals: List[torch.Tensor] = []
            m_locals: List[torch.Tensor] = []
            v_locals: List[torch.Tensor] = []
            step_val: Optional[int] = None
            dev: Optional[torch.device] = None

            fallback = False
            for p in group["params"]:
                if p.grad is None:
                    continue
                pl = _unwrap_local(p)
                gl = _unwrap_local(p.grad)
                if (
                    pl.device.type != "musa"
                    or pl.dtype != torch.float32
                    or gl.dtype != torch.float32
                    or not pl.is_contiguous()
                    or not gl.is_contiguous()
                ):
                    fallback = True
                    break
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = torch.tensor(0.0, dtype=torch.float32)
                    st["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    st["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                st["step"] += 1.0
                sv = int(st["step"].item())
                if step_val is None:
                    step_val = sv
                elif step_val != sv:
                    fallback = True
                    break
                ml = _unwrap_local(st["exp_avg"])
                vl = _unwrap_local(st["exp_avg_sq"])
                if not ml.is_contiguous() or not vl.is_contiguous() or ml.dtype != torch.float32 or vl.dtype != torch.float32:
                    fallback = True
                    break
                if pl.numel() == 0:
                    continue
                if dev is None:
                    dev = pl.device
                p_locals.append(pl)
                g_locals.append(gl)
                m_locals.append(ml)
                v_locals.append(vl)

            if fallback:
                # Roll back step increments before calling super().step()
                for p in group["params"]:
                    if p.grad is not None and p in self.state and "step" in self.state[p]:
                        self.state[p]["step"] -= 1.0
                return super().step(closure)

            if not p_locals or step_val is None or dev is None:
                continue

            # Build or reuse static device pointer & chunk tables (g_ptrs change if grads are reallocated)
            p_key = tuple(t.data_ptr() for t in p_locals)
            m_key = tuple(t.data_ptr() for t in m_locals)
            v_key = tuple(t.data_ptr() for t in v_locals)
            cached = self._meta_cache.get(g_idx)
            if cached is not None and cached[0] == p_key and cached[1] == m_key and cached[2] == v_key:
                _, _, _, p_ptrs, m_ptrs, v_ptrs, sz_t, ctid_t, coff_t, num_chunks = cached
            else:
                sizes = [t.numel() for t in p_locals]
                ctid = []
                coff = []
                for ti, sz in enumerate(sizes):
                    for off in range(0, sz, 65536):
                        ctid.append(ti)
                        coff.append(off)
                p_ptrs = torch.tensor(p_key, device=dev, dtype=torch.int64)
                m_ptrs = torch.tensor(m_key, device=dev, dtype=torch.int64)
                v_ptrs = torch.tensor(v_key, device=dev, dtype=torch.int64)
                sz_t = torch.tensor(sizes, device=dev, dtype=torch.int64)
                ctid_t = torch.tensor(ctid, device=dev, dtype=torch.int32)
                coff_t = torch.tensor(coff, device=dev, dtype=torch.int64)
                num_chunks = len(ctid)
                self._meta_cache[g_idx] = (
                    p_key, m_key, v_key, p_ptrs, m_ptrs, v_ptrs, sz_t, ctid_t, coff_t, num_chunks
                )

            g_ptrs = torch.tensor([t.data_ptr() for t in g_locals], device=dev, dtype=torch.int64)
            bc1 = 1.0 - (beta1 ** step_val)
            bc2 = 1.0 - (beta2 ** step_val)
            step_size = lr / bc1
            inv_sqrt_bc2 = 1.0 / math.sqrt(bc2)
            wd_scale = 1.0 - lr * wd

            rc = lib.launch_fused_adamw_f32(
                p_ptrs.data_ptr(),
                g_ptrs.data_ptr(),
                m_ptrs.data_ptr(),
                v_ptrs.data_ptr(),
                sz_t.data_ptr(),
                ctid_t.data_ptr(),
                coff_t.data_ptr(),
                num_chunks,
                float(beta1),
                float(beta2),
                float(1.0 - beta1),
                float(1.0 - beta2),
                float(step_size),
                float(inv_sqrt_bc2),
                float(wd_scale),
                float(eps),
                torch.musa.current_stream(dev).musa_stream,
            )
            if rc != 0:
                raise RuntimeError(f"launch_fused_adamw_f32 failed with MUSA error {rc}")
        return None
