"""Wan2.2-TI2V-5B DiT with Video Sparse Attention (VSA) for MTT S5000.

Weight layout matches the official Wan2.2-TI2V-5B checkpoint directly
(Wan-AI/Wan2.2-TI2V-5B, 825 keys across 3 safetensors shards),
extended with FastVideo VSA learnable compression gates
(blocks.{i}.self_attn.to_gate_compress.{weight,bias}, zero-initialized when
loading a dense Wan2.2 checkpoint).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .fused_ops import (
    can_use_fused_d3072,
    fused_adaln,
    fused_gated_res,
    fused_gated_res_ln,
    fused_gelu_tanh_bwd_bias,
    fused_gelu_tanh_fwd,
    fused_qk_rmsnorm_rope,
    fused_res_adaln,
    fused_rmsnorm,
    fused_rmsnorm_rope,
    fused_sum_dim0_bf16,
)
from .offload import AsyncOffloadPool
from .vsa import VSA_TILE_SIZE, VSAMetadata, build_vsa_metadata, video_sparse_attention


class _RecomputeGeluLinear(torch.autograd.Function):
    """y = F.linear(F.gelu(h, approximate='tanh'), w2, b2) saving only h (optionally offloaded).

    Uses static FFN scratch buffer (zero 754 MiB alloc stalls) and fused GELU-tanh bwd + fc1 bias grad.
    Reads linear_mod.weight/bias dynamically in backward so FSDP2 reshard_after_forward=True
    (ZeRO-3) unsharded storages are always valid.
    """

    @staticmethod
    def forward(ctx, h: torch.Tensor, w2: torch.Tensor, b2: Optional[torch.Tensor],
                linear_mod: Optional[nn.Linear] = None, offload_ctx=None,
                scratch_pool=None, db_in_box=None, db_out_box=None, fc1_is_ring: bool = False):
        h_c = h.contiguous()
        if offload_ctx is not None:
            pool, blk_idx = offload_ctx
            pool.offload((blk_idx, "fc1"), (h_c,), is_ring=fc1_is_ring)
            if linear_mod is None:
                ctx.save_for_backward(w2, b2)
            ctx.offload_ctx = (pool, blk_idx)
        else:
            if linear_mod is None:
                ctx.save_for_backward(h_c, w2, b2)
            else:
                ctx.save_for_backward(h_c)
            ctx.offload_ctx = None

        gelu_buf = scratch_pool.get_scratch("ffn_act", tuple(h_c.shape), h_c.dtype) if scratch_pool is not None else None
        a = fused_gelu_tanh_fwd(h_c, out=gelu_buf)
        out = F.linear(a, w2, b2)
        del a
        ctx.linear_mod = linear_mod
        ctx.has_bias = b2 is not None
        ctx.bias_dtype = b2.dtype if b2 is not None else h_c.dtype
        ctx.scratch_pool = scratch_pool
        ctx.db_in_box = db_in_box
        ctx.db_out_box = db_out_box
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if ctx.offload_ctx is not None:
            pool, blk_idx = ctx.offload_ctx
            if ctx.linear_mod is not None:
                w2 = ctx.linear_mod.weight
                b2 = ctx.linear_mod.bias
            else:
                w2, b2 = ctx.saved_tensors
            (h,) = pool.fetch((blk_idx, "fc1"))
            # Prefetch current block's ca, gres, and VSA tensors while FFN bwd (~22ms) runs!
            pool.prefetch((blk_idx, "ca"))
            pool.prefetch((blk_idx, "gres"))
            pool.prefetch((blk_idx, "vsa"))
        else:
            if ctx.linear_mod is not None:
                (h,) = ctx.saved_tensors
                w2 = ctx.linear_mod.weight
                b2 = ctx.linear_mod.bias
            else:
                h, w2, b2 = ctx.saved_tensors

        gelu_buf = ctx.scratch_pool.get_scratch("ffn_act", tuple(h.shape), h.dtype) if ctx.scratch_pool is not None else None
        a = fused_gelu_tanh_fwd(h, out=gelu_buf)
        go2d = grad_out.reshape(-1, grad_out.shape[-1])
        a2d = a.reshape(-1, a.shape[-1])
        dw2 = torch.matmul(go2d.t(), a2d)
        del a, a2d
        da = torch.matmul(go2d, w2).view_as(h)
        if ctx.has_bias:
            db2 = None
            if ctx.db_out_box is not None:
                db2 = ctx.db_out_box[0]
                ctx.db_out_box[0] = None
            if db2 is None:
                db2 = fused_sum_dim0_bf16(go2d)
            if db2.dtype != ctx.bias_dtype:
                db2 = db2.to(ctx.bias_dtype)
        else:
            db2 = None
        need_db1 = ctx.db_in_box is not None
        dh, db1 = fused_gelu_tanh_bwd_bias(da, h, need_bias_grad=need_db1, dx_out=gelu_buf)
        del da
        if need_db1:
            ctx.db_in_box[0] = db1
        if ctx.offload_ctx is not None:
            pool, _ = ctx.offload_ctx
            pool.release("fc1")
        return dh, dw2, db2, None, None, None, None, None, None


class _CheckpointFc1GeluLinear(torch.autograd.Function):
    """FFN fc1+gelu+fc2 recomputing only fc1+gelu in backward (never recomputing fc2)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, w1: torch.Tensor, b1: Optional[torch.Tensor],
                w2: torch.Tensor, b2: Optional[torch.Tensor],
                fc1_mod: Optional[nn.Linear] = None, fc2_mod: Optional[nn.Linear] = None):
        h = F.linear(x, w1, b1)
        a = F.gelu(h, approximate="tanh")
        del h
        out = F.linear(a, w2, b2)
        del a
        ctx.fc1_mod = fc1_mod
        ctx.fc2_mod = fc2_mod
        ctx.has_b1 = b1 is not None
        ctx.has_b2 = b2 is not None
        if fc1_mod is None or fc2_mod is None:
            ctx.save_for_backward(x, w1, b1, w2, b2)
        else:
            ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if ctx.fc1_mod is not None and ctx.fc2_mod is not None:
            (x,) = ctx.saved_tensors
            w1, b1 = ctx.fc1_mod.weight, ctx.fc1_mod.bias
            w2, b2 = ctx.fc2_mod.weight, ctx.fc2_mod.bias
        else:
            x, w1, b1, w2, b2 = ctx.saved_tensors
        with torch.enable_grad():
            h = F.linear(x.detach(), w1.detach(), b1.detach() if b1 is not None else None)
            h_req = h.detach().requires_grad_(True)
            a = F.gelu(h_req, approximate="tanh")
        go2d = grad_out.reshape(-1, grad_out.shape[-1])
        a2d = a.detach().reshape(-1, a.shape[-1])
        da = torch.matmul(go2d, w2).view_as(a)
        dw2 = torch.matmul(go2d.t(), a2d)
        db2 = go2d.sum(dim=0) if ctx.has_b2 else None
        del a2d
        (dh,) = torch.autograd.grad(a, h_req, da)
        del a, h, h_req, da
        dh2d = dh.reshape(-1, dh.shape[-1])
        x2d = x.reshape(-1, x.shape[-1])
        dx = torch.matmul(dh2d, w1).view_as(x)
        dw1 = torch.matmul(dh2d.t(), x2d)
        db1 = fused_sum_dim0_bf16(dh2d) if ctx.has_b1 else None
        return dx, dw1, db1, dw2, db2, None, None


class _FastBiasLinear(torch.autograd.Function):
    """FSDP2-safe linear reusing pre-accumulated fused bias grad (with fused fallback)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor],
                linear_mod: nn.Linear, db_box=None, out_buf=None):
        ctx.linear_mod = linear_mod
        ctx.has_bias = b is not None
        ctx.bias_dtype = b.dtype if b is not None else x.dtype
        ctx.db_box = db_box
        ctx.save_for_backward(x)
        if out_buf is not None:
            x2d = x.contiguous().reshape(-1, x.shape[-1])
            out2d = out_buf.detach().view(-1, w.shape[0])
            if b is not None:
                torch.addmm(b, x2d, w.t(), out=out2d)
            else:
                torch.mm(x2d, w.t(), out=out2d)
            return out_buf.detach().view_as(out_buf)
        return F.linear(x, w, b)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x,) = ctx.saved_tensors
        w = ctx.linear_mod.weight
        go2d = grad_out.reshape(-1, grad_out.shape[-1])
        x2d = x.reshape(-1, x.shape[-1])
        dx = torch.matmul(go2d, w).view_as(x) if ctx.needs_input_grad[0] else None
        dw = torch.matmul(go2d.t(), x2d) if ctx.needs_input_grad[1] else None
        if ctx.has_bias and ctx.needs_input_grad[2]:
            db = None
            if ctx.db_box is not None:
                db = ctx.db_box[0]
                ctx.db_box[0] = None
            if db is None:
                db = fused_sum_dim0_bf16(go2d)
            if db.dtype != ctx.bias_dtype:
                db = db.to(ctx.bias_dtype)
        else:
            db = None
        return dx, dw, db, None, None, None


class _FusedQKVGProj(torch.autograd.Function):
    """FSDP2-safe 4-way Q/K/V/Gate projection fusing dx accumulation via muBLAS addmm_ (beta=1.0)
    and reusing fused RMSNorm+RoPE / VSA Q/K/V/Gate bias grads, keeping exact original Linear parameter shards."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        wq: torch.Tensor, bq: Optional[torch.Tensor],
        wk: torch.Tensor, bk: Optional[torch.Tensor],
        wv: torch.Tensor, bv: Optional[torch.Tensor],
        wg: torch.Tensor, bg: Optional[torch.Tensor],
        q_mod: nn.Linear, k_mod: nn.Linear, v_mod: nn.Linear, g_mod: nn.Linear,
        dbq_box=None, dbk_box=None, dbv_box=None, dbg_box=None,
    ):
        ctx.mods = (q_mod, k_mod, v_mod, g_mod)
        ctx.has_bias = (bq is not None, bk is not None, bv is not None, bg is not None)
        ctx.bias_dtypes = (
            bq.dtype if bq is not None else x.dtype,
            bk.dtype if bk is not None else x.dtype,
            bv.dtype if bv is not None else x.dtype,
            bg.dtype if bg is not None else x.dtype,
        )
        ctx.dbq_box = dbq_box
        ctx.dbk_box = dbk_box
        ctx.dbv_box = dbv_box
        ctx.dbg_box = dbg_box
        ctx.save_for_backward(x)
        return (
            F.linear(x, wq, bq),
            F.linear(x, wk, bk),
            F.linear(x, wv, bv),
            F.linear(x, wg, bg),
        )

    @staticmethod
    def backward(ctx, dq: torch.Tensor, dk: torch.Tensor, dv: torch.Tensor, dg: torch.Tensor):
        (x,) = ctx.saved_tensors
        q_mod, k_mod, v_mod, g_mod = ctx.mods
        wq, wk, wv, wg = q_mod.weight, k_mod.weight, v_mod.weight, g_mod.weight
        x2d = x.reshape(-1, x.shape[-1])
        dq2d = dq.reshape(-1, dq.shape[-1])
        dk2d = dk.reshape(-1, dk.shape[-1])
        dv2d = dv.reshape(-1, dv.shape[-1])
        dg2d = dg.reshape(-1, dg.shape[-1])

        if ctx.needs_input_grad[0]:
            dx2d = torch.matmul(dq2d, wq)
            dx2d.addmm_(dk2d, wk)
            dx2d.addmm_(dv2d, wv)
            dx2d.addmm_(dg2d, wg)
            dx = dx2d.view_as(x)
        else:
            dx = None

        dwq = torch.matmul(dq2d.t(), x2d) if ctx.needs_input_grad[1] else None
        if ctx.has_bias[0] and ctx.needs_input_grad[2]:
            dbq = ctx.dbq_box[0] if ctx.dbq_box is not None else None
            if ctx.dbq_box is not None:
                ctx.dbq_box[0] = None
            if dbq is None:
                dbq = fused_sum_dim0_bf16(dq2d)
            if dbq.dtype != ctx.bias_dtypes[0]:
                dbq = dbq.to(ctx.bias_dtypes[0])
        else:
            dbq = None
        dwk = torch.matmul(dk2d.t(), x2d) if ctx.needs_input_grad[3] else None
        if ctx.has_bias[1] and ctx.needs_input_grad[4]:
            dbk = ctx.dbk_box[0] if ctx.dbk_box is not None else None
            if ctx.dbk_box is not None:
                ctx.dbk_box[0] = None
            if dbk is None:
                dbk = fused_sum_dim0_bf16(dk2d)
            if dbk.dtype != ctx.bias_dtypes[1]:
                dbk = dbk.to(ctx.bias_dtypes[1])
        else:
            dbk = None
        dwv = torch.matmul(dv2d.t(), x2d) if ctx.needs_input_grad[5] else None
        if ctx.has_bias[2] and ctx.needs_input_grad[6]:
            dbv = ctx.dbv_box[0] if ctx.dbv_box is not None else None
            if ctx.dbv_box is not None:
                ctx.dbv_box[0] = None
            if dbv is None:
                dbv = fused_sum_dim0_bf16(dv2d)
            if dbv.dtype != ctx.bias_dtypes[2]:
                dbv = dbv.to(ctx.bias_dtypes[2])
        else:
            dbv = None
        dwg = torch.matmul(dg2d.t(), x2d) if ctx.needs_input_grad[7] else None
        if ctx.has_bias[3] and ctx.needs_input_grad[8]:
            dbg = ctx.dbg_box[0] if ctx.dbg_box is not None else None
            if ctx.dbg_box is not None:
                ctx.dbg_box[0] = None
            if dbg is None:
                dbg = fused_sum_dim0_bf16(dg2d)
            if dbg.dtype != ctx.bias_dtypes[3]:
                dbg = dbg.to(ctx.bias_dtypes[3])
        else:
            dbg = None
        return dx, dwq, dbq, dwk, dbk, dwv, dbv, dwg, dbg, None, None, None, None, None, None, None, None


@dataclass
class Wan22VSAConfig:
    dim: int = 3072
    ffn_dim: int = 14336
    in_dim: int = 48
    out_dim: int = 48
    text_dim: int = 4096
    text_len: int = 512
    freq_dim: int = 256
    num_heads: int = 24
    num_layers: int = 30
    eps: float = 1e-6
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    rope_theta: float = 10000.0
    rope_max_len: int = 1024
    vsa_tile_size: Tuple[int, int, int] = VSA_TILE_SIZE
    vsa_q_chunk_blocks: int = 32

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "Wan22VSAConfig":
        cfg_path = Path(model_dir) / "config.json"
        if not cfg_path.exists():
            return cls()
        raw = json.loads(cfg_path.read_text())
        return cls(
            dim=int(raw.get("dim", 3072)),
            ffn_dim=int(raw.get("ffn_dim", 14336)),
            in_dim=int(raw.get("in_dim", 48)),
            out_dim=int(raw.get("out_dim", 48)),
            text_len=int(raw.get("text_len", 512)),
            freq_dim=int(raw.get("freq_dim", 256)),
            num_heads=int(raw.get("num_heads", 24)),
            num_layers=int(raw.get("num_layers", 30)),
            eps=float(raw.get("eps", 1e-6)),
        )


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    half = dim // 2
    freqs = torch.pow(
        10000.0,
        -torch.arange(half, dtype=torch.float64, device=position.device).div(half),
    )
    sinusoid = torch.outer(position.to(torch.float64), freqs)
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1).to(position.dtype)


def precompute_rope_3d_cos_sin(
    head_dim: int,
    max_len: int = 1024,
    theta: float = 10000.0,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Precompute 1D (cos, sin) tables for (F, H, W) matching Wan2.2 3D RoPE."""

    def _1d(d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64)[: d // 2] / d))
        angles = torch.outer(torch.arange(max_len, dtype=torch.float64), inv)
        return torch.cos(angles).to(torch.float32), torch.sin(angles).to(torch.float32)

    d_h = head_dim // 3
    d_w = head_dim // 3
    d_f = head_dim - 2 * d_h
    return _1d(d_f), _1d(d_h), _1d(d_w)


def apply_rope_interleaved(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply interleaved complex 3D RoPE to [B, S, H, D] tensor in FP32."""
    orig_dtype = x.dtype
    x_pair = x.float().reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    x0 = x_pair[..., 0]
    x1 = x_pair[..., 1]
    # cos, sin: [S, 1, D // 2] -> broadcast over B and H
    c = cos.unsqueeze(0)
    s = sin.unsqueeze(0)
    out0 = x0 * c - x1 * s
    out1 = x0 * s + x1 * c
    return torch.stack([out0, out1], dim=-1).flatten(-2).to(orig_dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if can_use_fused_d3072(x):
            return fused_rmsnorm(x, self.weight, self.eps)
        orig_dtype = x.dtype
        x_fp32 = x.float()
        normed = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(orig_dtype)


class WanVSASelfAttention(nn.Module):
    """Wan2.2 Self-Attention using Video Sparse Attention (VSA) + learnable gate_compress."""

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, q_chunk_blocks: int = 32):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_chunk_blocks = q_chunk_blocks

        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.to_gate_compress = nn.Linear(dim, dim, bias=True)
        self.o = nn.Linear(dim, dim, bias=True)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        vsa_meta: VSAMetadata,
        offload_ctx=None,
        o_db_box=None,
    ) -> torch.Tensor:
        B, S, _ = x.shape
        use_qkvg_fused = can_use_fused_d3072(x) and torch.is_grad_enabled()
        use_qk_rope_fused = use_qkvg_fused and self.head_dim == 128
        dbq_box = [None] if (use_qk_rope_fused and self.q.bias is not None) else None
        dbk_box = [None] if (use_qk_rope_fused and self.k.bias is not None) else None
        dbv_box = [None] if (use_qk_rope_fused and self.v.bias is not None) else None
        dbg_box = [None] if (use_qk_rope_fused and self.to_gate_compress.bias is not None) else None
        if use_qkvg_fused:
            q_raw, k_raw, v_raw, g_raw = _FusedQKVGProj.apply(
                x,
                self.q.weight, self.q.bias,
                self.k.weight, self.k.bias,
                self.v.weight, self.v.bias,
                self.to_gate_compress.weight, self.to_gate_compress.bias,
                self.q, self.k, self.v, self.to_gate_compress,
                dbq_box, dbk_box, dbv_box, dbg_box,
            )
            v = v_raw.view(B, S, self.num_heads, self.head_dim)
            gate_compress = g_raw.view(B, S, self.num_heads, self.head_dim)
        else:
            q_raw = self.q(x)
            k_raw = self.k(x)
            v = self.v(x).view(B, S, self.num_heads, self.head_dim)
            gate_compress = self.to_gate_compress(x).view(B, S, self.num_heads, self.head_dim)

        if can_use_fused_d3072(q_raw) and self.head_dim == 128:
            q_rot, k_rot = fused_qk_rmsnorm_rope(
                q_raw,
                k_raw,
                self.norm_q.weight,
                self.norm_k.weight,
                freqs_cos,
                freqs_sin,
                self.norm_q.eps,
                self.norm_k.eps,
                offload_ctx=offload_ctx,
                dbq_box=dbq_box,
                dbk_box=dbk_box,
            )
            q = q_rot.view(B, S, self.num_heads, self.head_dim)
            k = k_rot.view(B, S, self.num_heads, self.head_dim)
        else:
            q = apply_rope_interleaved(
                self.norm_q(q_raw).view(B, S, self.num_heads, self.head_dim), freqs_cos, freqs_sin
            )
            k = apply_rope_interleaved(
                self.norm_k(k_raw).view(B, S, self.num_heads, self.head_dim), freqs_cos, freqs_sin
            )

        out = video_sparse_attention(
            q, k, v, gate_compress, vsa_meta, q_chunk_blocks=self.q_chunk_blocks,
            offload_ctx=offload_ctx, dbv_box=dbv_box, dbg_box=dbg_box,
        )
        out = out.reshape(B, S, self.dim)
        if can_use_fused_d3072(out) and torch.is_grad_enabled():
            return _FastBiasLinear.apply(out, self.o.weight, self.o.bias, self.o, o_db_box)
        return self.o(out)


class WanCrossAttention(nn.Module):
    """Wan2.2 T2V/TI2V Cross-Attention over UMT5 text embeddings."""

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.o = nn.Linear(dim, dim, bias=True)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, context: torch.Tensor, o_db_box=None) -> torch.Tensor:
        B, S, _ = x.shape
        L = context.shape[1]
        if can_use_fused_d3072(x) and torch.is_grad_enabled():
            q_db_box = [None] if self.q.bias is not None else None
            q_raw = _FastBiasLinear.apply(x, self.q.weight, self.q.bias, self.q, q_db_box)
            q = fused_rmsnorm(q_raw, self.norm_q.weight, self.norm_q.eps, db_x_box=q_db_box).view(
                B, S, self.num_heads, self.head_dim
            ).transpose(1, 2)
        else:
            q = self.norm_q(self.q(x)).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.norm_k(self.k(context)).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(context).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, S, self.dim)
        if can_use_fused_d3072(out) and torch.is_grad_enabled():
            return _FastBiasLinear.apply(out, self.o.weight, self.o.bias, self.o, o_db_box)
        return self.o(out)


class WanVSATransformerBlock(nn.Module):
    """Wan2.2 DiT block with VSA self-attention, cross-attention, and GELU-tanh FFN."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        eps: float = 1e-6,
        q_chunk_blocks: int = 32,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.eps = eps

        self.self_attn = WanVSASelfAttention(dim, num_heads, eps=eps, q_chunk_blocks=q_chunk_blocks)
        self.cross_attn = WanCrossAttention(dim, num_heads, eps=eps)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim, bias=True),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / math.sqrt(dim))

    def _run_ffn(
        self,
        ffn_in: torch.Tensor,
        ffn_mode: str,
        offload_ctx=None,
        scratch_pool=None,
        fc2_db_box=None,
    ) -> torch.Tensor:
        if ffn_mode in ("recompute_gelu_offload", "recompute_gelu"):
            fc1_db_box = [None] if (torch.is_grad_enabled() and self.ffn[0].bias is not None) else None
            off = offload_ctx if ffn_mode == "recompute_gelu_offload" else None
            fc1_out = None
            fc1_is_ring = False
            if off is not None:
                pool, blk_idx = off
                (fc1_out,) = pool.get_ring_slot(
                    "fc1", blk_idx,
                    ((tuple(ffn_in.shape[:-1]) + (self.ffn[0].out_features,), ffn_in.dtype),),
                    num_slots=3,
                )
                fc1_is_ring = True
            elif scratch_pool is not None:
                fc1_out = scratch_pool.get_scratch(
                    "fc1_last",
                    tuple(ffn_in.shape[:-1]) + (self.ffn[0].out_features,),
                    ffn_in.dtype,
                )
            h = _FastBiasLinear.apply(ffn_in, self.ffn[0].weight, self.ffn[0].bias, self.ffn[0], fc1_db_box, fc1_out)
            return _RecomputeGeluLinear.apply(
                h, self.ffn[2].weight, self.ffn[2].bias, self.ffn[2],
                off, scratch_pool, fc1_db_box, fc2_db_box, fc1_is_ring,
            )
        if ffn_mode == "ckpt_fc1_gelu":
            return _CheckpointFc1GeluLinear.apply(
                ffn_in,
                self.ffn[0].weight,
                self.ffn[0].bias,
                self.ffn[2].weight,
                self.ffn[2].bias,
                self.ffn[0],
                self.ffn[2],
            )
        return self.ffn(ffn_in)

    def _run_cross_attn_offload(
        self,
        norm3_x: torch.Tensor,
        context: torch.Tensor,
        offload_ctx,
        o_db_box=None,
    ) -> torch.Tensor:
        """Run cross_attn without recompute, asynchronously offloading unique >=100 MiB storages
        (q_raw, q, out = 480 MiB/blk) to pinned CPU memory via saved_tensors_hooks."""
        pool, blk_idx = offload_ctx
        skip_ptr = norm3_x.untyped_storage().data_ptr()
        ptr_to_idx: dict[int, int] = {}
        raw_views: list[torch.Tensor] = []
        fetch_box: list[Optional[Tuple[torch.Tensor, ...]]] = [None]

        def _pack(t: torch.Tensor):
            if (
                isinstance(t, torch.Tensor)
                and t.device.type == "musa"
                and t.untyped_storage().nbytes() >= 100 * 1024 * 1024
            ):
                s = t.untyped_storage()
                ptr = s.data_ptr()
                if ptr == skip_ptr:
                    return t
                slot_i = ptr_to_idx.get(ptr)
                if slot_i is None:
                    slot_i = len(raw_views)
                    ptr_to_idx[ptr] = slot_i
                    n_elems = s.nbytes() // t.element_size()
                    flat = torch.empty(0, dtype=t.dtype, device=t.device).set_(s, 0, (n_elems,))
                    raw_views.append(flat)
                return ("__ca_off__", slot_i, tuple(t.shape), tuple(t.stride()), int(t.storage_offset()), t.dtype)
            return t

        def _unpack(packed):
            if isinstance(packed, tuple) and len(packed) == 6 and packed[0] == "__ca_off__":
                _, slot_i, shape, stride, offset, dtype = packed
                if fetch_box[0] is None:
                    fetch_box[0] = pool.fetch((blk_idx, "ca"))
                flat_gpu = fetch_box[0][slot_i]
                if flat_gpu.dtype != dtype:
                    flat_gpu = flat_gpu.view(dtype)
                return flat_gpu.as_strided(shape, stride, offset)
            return packed

        with torch.autograd.graph.saved_tensors_hooks(_pack, _unpack):
            ca_out = self.cross_attn(norm3_x, context, o_db_box=o_db_box)
        if raw_views:
            pool.offload((blk_idx, "ca"), tuple(raw_views))
            raw_views.clear()
        return ca_out

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        vsa_meta: VSAMetadata,
        offload_ctx=None,
        offload_extra: bool = False,
        ffn_mode: str = "default",
        ckpt_ca: bool = False,
        scratch_pool=None,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        has_seq = t_mod.ndim == 4
        if has_seq:
            e = self.modulation.float().unsqueeze(1) + t_mod.float()
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = e.chunk(6, dim=2)
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            shift_mlp = shift_mlp.squeeze(2)
            scale_mlp = scale_mlp.squeeze(2)
            gate_mlp = gate_mlp.squeeze(2)
        else:
            e = self.modulation.float() + t_mod.float()
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = e.chunk(6, dim=1)

        if not has_seq and can_use_fused_d3072(x):
            grad_on = torch.is_grad_enabled()
            skip_box = [None] if (grad_on and x.requires_grad) else None
            sa_o_db_box = [None] if (grad_on and self.self_attn.o.bias is not None) else None
            ca_o_db_box = [None] if (grad_on and self.cross_attn.o.bias is not None and not ckpt_ca) else None
            fc2_db_box = (
                [None]
                if (grad_on and self.ffn[2].bias is not None and ffn_mode in ("recompute_gelu_offload", "recompute_gelu"))
                else None
            )

            # 1. Self-attention (VSA)
            sa_in = fused_adaln(x, scale_msa, shift_msa, self.eps, dx_add_box=skip_box)
            sa_out = self.self_attn(
                sa_in, freqs_cos, freqs_sin, vsa_meta, offload_ctx=offload_ctx, o_db_box=sa_o_db_box
            )

            # 2. Gated residual + affine LN3 -> Cross-attention
            x, norm3_x = fused_gated_res_ln(
                x, sa_out, gate_msa, self.norm3.weight, self.norm3.bias, self.eps,
                offload_ctx=offload_ctx if offload_extra else None,
                db_upd_box=sa_o_db_box,
                dx_out_box=skip_box,
            )
            if ckpt_ca and grad_on:
                ca_out = grad_checkpoint(self.cross_attn, norm3_x, context, use_reentrant=False)
            elif offload_ctx is not None and grad_on:
                ca_out = self._run_cross_attn_offload(norm3_x, context, offload_ctx, o_db_box=ca_o_db_box)
            else:
                ca_out = self.cross_attn(norm3_x, context, o_db_box=ca_o_db_box)

            # 3. Residual + AdaLN2 -> FFN -> Gated residual
            x, ffn_in = fused_res_adaln(x, ca_out, scale_mlp, shift_mlp, self.eps, db_upd_box=ca_o_db_box)
            ffn_out = self._run_ffn(
                ffn_in, ffn_mode, offload_ctx=offload_ctx, scratch_pool=scratch_pool, fc2_db_box=fc2_db_box
            )
            x = fused_gated_res(x, ffn_out, gate_mlp, db_upd_box=fc2_db_box)
            return x

        # 1. Self-attention (VSA)
        norm1_x = F.layer_norm(x.float(), (self.dim,), None, None, self.eps)
        sa_in = (norm1_x * (1.0 + scale_msa) + shift_msa).to(orig_dtype)
        sa_out = self.self_attn(sa_in, freqs_cos, freqs_sin, vsa_meta)
        x = (x.float() + gate_msa * sa_out.float()).to(orig_dtype)

        # 2. Cross-attention
        norm3_x = F.layer_norm(
            x.float(), (self.dim,), self.norm3.weight.float(), self.norm3.bias.float(), self.eps
        ).to(orig_dtype)
        ca_out = self.cross_attn(norm3_x, context)
        x = (x.float() + ca_out.float()).to(orig_dtype)

        # 3. FFN
        norm2_x = F.layer_norm(x.float(), (self.dim,), None, None, self.eps)
        ffn_in = (norm2_x * (1.0 + scale_mlp) + shift_mlp).to(orig_dtype)
        ffn_out = self.ffn(ffn_in)
        x = (x.float() + gate_mlp * ffn_out.float()).to(orig_dtype)
        return x


class WanHead(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size), bias=True)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / math.sqrt(dim))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        if t_emb.ndim == 2 and can_use_fused_d3072(x):
            e = self.modulation.float() + t_emb.float().unsqueeze(1)
            shift, scale = e.chunk(2, dim=1)
            h = fused_adaln(x, scale, shift, self.eps)
            return self.head(h)
        orig_dtype = x.dtype
        norm_x = F.layer_norm(x.float(), (self.dim,), None, None, self.eps)
        if t_emb.ndim == 3:
            e = self.modulation.float().unsqueeze(1) + t_emb.float().unsqueeze(2)
            shift, scale = e.chunk(2, dim=2)
            h = (norm_x * (1.0 + scale.squeeze(2)) + shift.squeeze(2)).to(orig_dtype)
        else:
            e = self.modulation.float() + t_emb.float().unsqueeze(1)
            shift, scale = e.chunk(2, dim=1)
            h = (norm_x * (1.0 + scale) + shift).to(orig_dtype)
        return self.head(h)


class Wan22VSAModel(nn.Module):
    """Full Wan2.2-TI2V-5B Diffusion Transformer with VSA self-attention."""

    def __init__(self, config: Wan22VSAConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.in_dim = config.in_dim
        self.out_dim = config.out_dim
        self.freq_dim = config.freq_dim
        self.patch_size = config.patch_size
        self.num_heads = config.num_heads
        self.head_dim = config.dim // config.num_heads
        self.gradient_checkpointing = False

        self.patch_embedding = nn.Conv3d(
            config.in_dim,
            config.dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True,
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(config.text_dim, config.dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.dim, config.dim, bias=True),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(config.freq_dim, config.dim, bias=True),
            nn.SiLU(),
            nn.Linear(config.dim, config.dim, bias=True),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.dim, config.dim * 6, bias=True),
        )
        self.blocks = nn.ModuleList(
            [
                WanVSATransformerBlock(
                    dim=config.dim,
                    ffn_dim=config.ffn_dim,
                    num_heads=config.num_heads,
                    eps=config.eps,
                    q_chunk_blocks=config.vsa_q_chunk_blocks,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.head = WanHead(config.dim, config.out_dim, config.patch_size, eps=config.eps)

        (f_cos, f_sin), (h_cos, h_sin), (w_cos, w_sin) = precompute_rope_3d_cos_sin(
            self.head_dim, max_len=config.rope_max_len, theta=config.rope_theta
        )
        self.register_buffer("rope_f_cos", f_cos, persistent=False)
        self.register_buffer("rope_f_sin", f_sin, persistent=False)
        self.register_buffer("rope_h_cos", h_cos, persistent=False)
        self.register_buffer("rope_h_sin", h_sin, persistent=False)
        self.register_buffer("rope_w_cos", w_cos, persistent=False)
        self.register_buffer("rope_w_sin", w_sin, persistent=False)

    def zero_init_vsa_gates(self) -> None:
        """Zero-initialize VSA compression gates to match dense Wan2.2 at step 0."""
        with torch.no_grad():
            for blk in self.blocks:
                blk.self_attn.to_gate_compress.weight.zero_()
                blk.self_attn.to_gate_compress.bias.zero_()

    def _build_3d_rope(
        self, f: int, h: int, w: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = torch.cat(
            [
                self.rope_f_cos[:f].to(device).view(f, 1, 1, -1).expand(f, h, w, -1),
                self.rope_h_cos[:h].to(device).view(1, h, 1, -1).expand(f, h, w, -1),
                self.rope_w_cos[:w].to(device).view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1)
        sin = torch.cat(
            [
                self.rope_f_sin[:f].to(device).view(f, 1, 1, -1).expand(f, h, w, -1),
                self.rope_h_sin[:h].to(device).view(1, h, 1, -1).expand(f, h, w, -1),
                self.rope_w_sin[:w].to(device).view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1)
        return cos, sin

    def unpatchify(self, x: torch.Tensor, grid_size: Tuple[int, int, int]) -> torch.Tensor:
        B = x.shape[0]
        f, h, w = grid_size
        pt, ph, pw = self.patch_size
        x = x.reshape(B, f, h, w, pt, ph, pw, self.out_dim)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        return x.reshape(B, self.out_dim, f * pt, h * ph, w * pw)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        vsa_sparsity: float = 0.8,
        vsa_meta: Optional[VSAMetadata] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, 48, T, H, W] noisy video latents.
            timestep: [B] (T2V) or [B, L] (TI2V per-token) timesteps in [0, 1000].
            encoder_hidden_states: [B, 512, 4096] UMT5 text embeddings.
            vsa_sparsity: target sparsity in [0, 1).
        Returns:
            predicted velocity [B, 48, T, H, W].
        """
        orig_dtype = hidden_states.dtype
        # 1. Patchify
        x = self.patch_embedding(hidden_states)  # [B, dim, f, h, w]
        _, _, f, h, w = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()  # [B, L, dim]

        # 2. Time & text conditioning
        if timestep.ndim == 2:
            bsz, seq_len = timestep.shape
            t_in = sinusoidal_embedding_1d(self.freq_dim, timestep.reshape(-1)).to(orig_dtype)
            t_emb = self.time_embedding(t_in).reshape(bsz, seq_len, self.dim)
            t_mod = self.time_projection(t_emb).unflatten(-1, (6, self.dim))
        else:
            t_in = sinusoidal_embedding_1d(self.freq_dim, timestep).to(orig_dtype)
            t_emb = self.time_embedding(t_in)  # [B, dim]
            t_mod = self.time_projection(t_emb).unflatten(1, (6, self.dim))  # [B, 6, dim]

        context = self.text_embedding(encoder_hidden_states.to(orig_dtype))

        # 3. 3D RoPE and VSA metadata
        freqs_cos, freqs_sin = self._build_3d_rope(f, h, w, hidden_states.device)
        if vsa_meta is None:
            vsa_meta = build_vsa_metadata(
                (f, h, w),
                sparsity=vsa_sparsity,
                device=hidden_states.device,
                tile_size=self.config.vsa_tile_size,
            )

        # 4. Transformer blocks
        if self.gradient_checkpointing and torch.is_grad_enabled():
            gc_mode = os.environ.get("WAN22_GC_MODE", "offload")
            if gc_mode == "full":
                for block in self.blocks:
                    x = grad_checkpoint(
                        block,
                        x,
                        context,
                        t_mod,
                        freqs_cos,
                        freqs_sin,
                        vsa_meta,
                        use_reentrant=False,
                    )
            elif gc_mode == "selective_gpu":
                full_gc_n = int(os.environ.get("WAN22_FULL_GC_BLOCKS", "8"))
                ckpt_ca_n = int(os.environ.get("WAN22_CKPT_CA_BLOCKS", "30"))
                for i, block in enumerate(self.blocks):
                    if i < full_gc_n:
                        x = grad_checkpoint(
                            block,
                            x,
                            context,
                            t_mod,
                            freqs_cos,
                            freqs_sin,
                            vsa_meta,
                            use_reentrant=False,
                        )
                    else:
                        x = block(
                            x,
                            context,
                            t_mod,
                            freqs_cos,
                            freqs_sin,
                            vsa_meta,
                            ffn_mode="ckpt_fc1_gelu",
                            ckpt_ca=(i < ckpt_ca_n),
                        )
            else:
                # Default "offload": zero GEMM & VSA recompute via overlapped DMA offload + GELU-only recompute
                if not hasattr(self, "_offload_pool") or self._offload_pool is None or self._offload_pool.device != hidden_states.device:
                    self._offload_pool = AsyncOffloadPool(hidden_states.device)
                pool = self._offload_pool
                pool.reset_step()
                offload_n = int(os.environ.get("WAN22_OFFLOAD_BLOCKS", "29"))
                offload_extra = os.environ.get("WAN22_OFFLOAD_EXTRA", "1") == "1"
                ckpt_ca_n = int(os.environ.get("WAN22_CKPT_CA_BLOCKS", "0"))
                full_gc_n = int(os.environ.get("WAN22_FULL_GC_BLOCKS", "0"))
                for i, block in enumerate(self.blocks):
                    if i < full_gc_n:
                        x = grad_checkpoint(
                            block,
                            x,
                            context,
                            t_mod,
                            freqs_cos,
                            freqs_sin,
                            vsa_meta,
                            use_reentrant=False,
                        )
                    else:
                        do_off = i < offload_n
                        x = block(
                            x,
                            context,
                            t_mod,
                            freqs_cos,
                            freqs_sin,
                            vsa_meta,
                            offload_ctx=(pool, i) if do_off else None,
                            offload_extra=offload_extra,
                            ffn_mode="recompute_gelu_offload" if do_off else "recompute_gelu",
                            ckpt_ca=(i < ckpt_ca_n),
                            scratch_pool=pool,
                        )
        else:
            for block in self.blocks:
                x = block(x, context, t_mod, freqs_cos, freqs_sin, vsa_meta)

        # 5. Output head & unpatchify
        x = self.head(x, t_emb)
        return self.unpatchify(x, (f, h, w))


def load_wan22_ti2v_5b_vsa(
    model_dir: str | Path,
    dtype: torch.dtype = torch.float32,
    vsa_q_chunk_blocks: int = 32,
    num_layers_override: Optional[int] = None,
) -> Wan22VSAModel:
    """Instantiate Wan22VSAModel and load pretrained Wan2.2-TI2V-5B safetensors."""
    model_dir = Path(model_dir)
    cfg = Wan22VSAConfig.from_model_dir(model_dir)
    cfg.vsa_q_chunk_blocks = vsa_q_chunk_blocks
    if num_layers_override is not None:
        cfg.num_layers = num_layers_override
    model = Wan22VSAModel(cfg)

    idx_path = model_dir / "diffusion_pytorch_model.safetensors.index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"Missing safetensors index: {idx_path}")
    weight_map = json.loads(idx_path.read_text())["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for k, fname in weight_map.items():
        if num_layers_override is not None and k.startswith("blocks."):
            blk_idx = int(k.split(".")[1])
            if blk_idx >= num_layers_override:
                continue
        by_shard.setdefault(fname, []).append(k)

    state_dict: dict[str, torch.Tensor] = {}
    for fname, keys in sorted(by_shard.items()):
        with safe_open(str(model_dir / fname), framework="pt", device="cpu") as sf:
            for k in keys:
                state_dict[k] = sf.get_tensor(k).to(dtype=dtype)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {
        f"blocks.{i}.self_attn.to_gate_compress.{s}"
        for i in range(cfg.num_layers)
        for s in ("weight", "bias")
    }
    bad_missing = set(missing) - allowed_missing
    if bad_missing or unexpected:
        raise RuntimeError(
            f"Weight mismatch loading {model_dir}: bad_missing={sorted(bad_missing)[:10]}, "
            f"unexpected={sorted(unexpected)[:10]}"
        )
    model.zero_init_vsa_gates()
    if dtype != torch.float32:
        model.to(dtype=dtype)
    return model
