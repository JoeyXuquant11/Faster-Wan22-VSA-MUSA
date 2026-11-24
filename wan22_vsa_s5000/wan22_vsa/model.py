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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .vsa import VSA_TILE_SIZE, VSAMetadata, build_vsa_metadata, video_sparse_attention


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
    ) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.norm_q(self.q(x)).view(B, S, self.num_heads, self.head_dim)
        k = self.norm_k(self.k(x)).view(B, S, self.num_heads, self.head_dim)
        v = self.v(x).view(B, S, self.num_heads, self.head_dim)
        gate_compress = self.to_gate_compress(x).view(B, S, self.num_heads, self.head_dim)

        q = apply_rope_interleaved(q, freqs_cos, freqs_sin)
        k = apply_rope_interleaved(k, freqs_cos, freqs_sin)

        out = video_sparse_attention(
            q, k, v, gate_compress, vsa_meta, q_chunk_blocks=self.q_chunk_blocks
        )
        out = out.reshape(B, S, self.dim)
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

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        L = context.shape[1]
        q = self.norm_q(self.q(x)).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.norm_k(self.k(context)).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(context).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, S, self.dim)
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

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        vsa_meta: VSAMetadata,
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
