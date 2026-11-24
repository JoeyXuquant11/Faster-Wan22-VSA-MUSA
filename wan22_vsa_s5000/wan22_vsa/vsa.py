"""Video Sparse Attention (VSA) faithfully ported from FastVideo to PyTorch-MUSA.

Implements the exact two-stage differentiable VSA formulation from
"VSA: Faster Video Diffusion with Trainable Sparse Attention" (arXiv:2505.13389)
and FastVideo (hao-ai-lab/FastVideo):
  1. 3D spatial-temporal tile partition with tile_size = (4, 4, 4) (64 tokens/tile)
     and zero-padding of partial boundary tiles.
  2. Coarse compression branch: FP32 mean pooling over valid tokens per tile,
     dense coarse attention over tiles, and block-level coarse output out_c.
  3. Top-k tile routing per (batch, head, query_tile) from coarse scores.
  4. Fine block-sparse attention over the selected top-k KV tiles per query tile,
     masking padded slots beyond variable_block_sizes[kv_idx].
  5. Learnable gated combination out_s + out_c * gate_compress and untile back
     to raster token order.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

VSA_TILE_SIZE: Tuple[int, int, int] = (4, 4, 4)


@dataclass
class VSAMetadata:
    dit_seq_shape: Tuple[int, int, int]
    tile_size: Tuple[int, int, int]
    num_tiles: Tuple[int, int, int]
    total_seq_length: int
    padded_seq_length: int
    sparsity: float
    topk: int
    tile_partition_indices: torch.Tensor
    reverse_tile_partition_indices: torch.Tensor
    variable_block_sizes: torch.Tensor
    non_pad_index: torch.Tensor
    untile_combined_index: torch.Tensor
    pad_to_raster: torch.Tensor


def compute_vsa_topk(sparsity: float, num_blocks: int) -> int:
    """Number of KV tiles kept per query tile, matching FastVideo compute_topk."""
    return max(1, min(math.ceil((1.0 - sparsity) * num_blocks), num_blocks))


def compute_vsa_sparsity(
    step: int,
    target_sparsity: float,
    decay_rate: float = 0.0,
    decay_interval_steps: int = 1,
) -> float:
    """Progressive VSA sparsity schedule matching FastVideo TrainingPipeline.

    When decay_rate <= 0, uses target_sparsity directly on every step.
    Otherwise ramps from 0.0 up to target_sparsity in increments of decay_rate
    every decay_interval_steps steps.
    """
    if decay_rate <= 0.0 or decay_interval_steps <= 0:
        return float(target_sparsity)
    max_steps = int(target_sparsity // decay_rate)
    cur_times = min(step // decay_interval_steps, max_steps)
    return min(float(target_sparsity), float(cur_times * decay_rate))


def _canon_device(device: torch.device | str) -> torch.device:
    dev = torch.device(device)
    if dev.index is None:
        if dev.type == "musa" and hasattr(torch, "musa") and torch.musa.is_available():
            return torch.device("musa", torch.musa.current_device())
        if dev.type == "cuda" and torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
    return dev


@functools.lru_cache(maxsize=32)
def _cached_vsa_indices(
    dit_seq_shape: Tuple[int, int, int],
    tile_size: Tuple[int, int, int],
    device: torch.device,
):
    T, H, W = dit_seq_shape
    ts, hs, ws = tile_size
    n_t = math.ceil(T / ts)
    n_h = math.ceil(H / hs)
    n_w = math.ceil(W / ws)
    num_tiles = (n_t, n_h, n_w)
    block_elements = ts * hs * ws

    grid = torch.arange(T * H * W, device=device, dtype=torch.long).reshape(T, H, W)
    chunks = []
    for t in range(n_t):
        for h in range(n_h):
            for w in range(n_w):
                chunks.append(
                    grid[
                        t * ts : min((t + 1) * ts, T),
                        h * hs : min((h + 1) * hs, H),
                        w * ws : min((w + 1) * ws, W),
                    ].reshape(-1)
                )
    tile_partition_indices = torch.cat(chunks, dim=0)
    reverse_tile_partition_indices = torch.argsort(tile_partition_indices)

    def _axis_sizes(dim_len: int, tile: int, n: int) -> torch.Tensor:
        sizes = torch.full((n,), tile, dtype=torch.int32, device=device)
        rem = dim_len - (n - 1) * tile
        sizes[-1] = rem if rem > 0 else tile
        return sizes

    vbs = (
        _axis_sizes(T, ts, n_t)[:, None, None]
        * _axis_sizes(H, hs, n_h)[None, :, None]
        * _axis_sizes(W, ws, n_w)[None, None, :]
    ).reshape(-1)

    n_win = vbs.shape[0]
    starts_pad = torch.arange(n_win, device=device, dtype=torch.long) * block_elements
    index_pad = starts_pad[:, None] + torch.arange(block_elements, device=device, dtype=torch.long)[None, :]
    index_mask = torch.arange(block_elements, device=device, dtype=torch.int32)[None, :] < vbs[:, None]
    non_pad_index = index_pad[index_mask]
    untile_combined_index = non_pad_index[reverse_tile_partition_indices]
    padded_seq_length = n_win * block_elements
    pad_to_raster = torch.full((padded_seq_length,), -1, dtype=torch.int32, device=device)
    pad_to_raster[non_pad_index] = tile_partition_indices.to(torch.int32)
    return (
        num_tiles,
        padded_seq_length,
        tile_partition_indices,
        reverse_tile_partition_indices,
        vbs,
        non_pad_index,
        untile_combined_index,
        pad_to_raster,
    )


def build_vsa_metadata(
    dit_seq_shape: Tuple[int, int, int],
    sparsity: float,
    device: torch.device | str,
    tile_size: Tuple[int, int, int] = VSA_TILE_SIZE,
) -> VSAMetadata:
    dev = _canon_device(device)
    (
        num_tiles,
        padded_seq_length,
        tile_partition_indices,
        reverse_tile_partition_indices,
        vbs,
        non_pad_index,
        untile_combined_index,
        pad_to_raster,
    ) = _cached_vsa_indices(dit_seq_shape, tile_size, dev)
    total_seq_length = math.prod(dit_seq_shape)
    topk = compute_vsa_topk(sparsity, int(vbs.numel()))
    return VSAMetadata(
        dit_seq_shape=dit_seq_shape,
        tile_size=tile_size,
        num_tiles=num_tiles,
        total_seq_length=total_seq_length,
        padded_seq_length=padded_seq_length,
        sparsity=float(sparsity),
        topk=topk,
        tile_partition_indices=tile_partition_indices,
        reverse_tile_partition_indices=reverse_tile_partition_indices,
        variable_block_sizes=vbs,
        non_pad_index=non_pad_index,
        untile_combined_index=untile_combined_index,
        pad_to_raster=pad_to_raster,
    )


def tile_qkvg(qkvg: torch.Tensor, meta: VSAMetadata) -> torch.Tensor:
    """Scatter [N, S, H, D] into zero-padded tiled layout [N, S_pad, H, D]."""
    target_shape = (qkvg.shape[0], meta.padded_seq_length, qkvg.shape[2], qkvg.shape[3])
    buf = torch.zeros(target_shape, device=qkvg.device, dtype=qkvg.dtype)
    buf[:, meta.non_pad_index] = qkvg[:, meta.tile_partition_indices]
    return buf


def untile_output(out: torch.Tensor, meta: VSAMetadata) -> torch.Tensor:
    """Gather [B, S_pad, H, D] back to raster token order [B, S, H, D]."""
    return out[:, meta.untile_combined_index]


def _block_mean_pool(
    x: torch.Tensor,
    vbs: torch.Tensor,
    block_elements: int,
) -> torch.Tensor:
    """Exact FastVideo coarse pooling: FP32 sum over block / variable_block_sizes -> input dtype."""
    B, H, S_pad, D = x.shape
    num_blocks = S_pad // block_elements
    x_blk = x.view(B, H, num_blocks, block_elements, D)
    denom = vbs.to(dtype=torch.float32).view(1, 1, num_blocks, 1)
    return (x_blk.float().sum(dim=3) / denom).to(x.dtype)


def _block_sparse_attention_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idx: torch.Tensor,
    vbs: torch.Tensor,
    block_elements: int,
    q_chunk_blocks: int = 32,
) -> torch.Tensor:
    """Chunked PyTorch block-sparse fine attention using SDPA with valid-token masks.

    Args:
        q, k, v: [B, H, S_pad, D] contiguous tensors in tiled order.
        topk_idx: [B, H, Q_blks, topk] selected KV block indices (int64).
        vbs: [KV_blks] valid token count per KV block (int32).
        block_elements: tokens per block (64 for (4,4,4)).
    Returns:
        out_s: [B, H, S_pad, D]
    """
    B, H, S_pad, D = q.shape
    num_blocks = S_pad // block_elements
    topk = topk_idx.shape[-1]

    q_blocks = q.view(B, H, num_blocks, block_elements, D)
    k_blocks = k.view(B, H, num_blocks, block_elements, D)
    v_blocks = v.view(B, H, num_blocks, block_elements, D)

    bh_arange = torch.arange(B * H, device=q.device, dtype=torch.long)[:, None, None]
    col_arange = torch.arange(block_elements, device=q.device, dtype=torch.int32)
    k_flat = k_blocks.reshape(B * H, num_blocks, block_elements, D)
    v_flat = v_blocks.reshape(B * H, num_blocks, block_elements, D)
    q_flat = q_blocks.reshape(B * H, num_blocks, block_elements, D)
    idx_flat = topk_idx.reshape(B * H, num_blocks, topk)

    out_chunks = []
    for q0 in range(0, num_blocks, q_chunk_blocks):
        q1 = min(q0 + q_chunk_blocks, num_blocks)
        c = q1 - q0
        idx_c = idx_flat[:, q0:q1, :]  # [B*H, c, topk]

        # Gather selected KV blocks for this query chunk: [B*H, c, topk, blk, D]
        k_sel = k_flat[bh_arange, idx_c].reshape(B * H * c, 1, topk * block_elements, D)
        v_sel = v_flat[bh_arange, idx_c].reshape(B * H * c, 1, topk * block_elements, D)
        q_c = q_flat[:, q0:q1, :, :].reshape(B * H * c, 1, block_elements, D)

        # Mask out padded columns beyond vbs[kv_idx]
        vbs_sel = vbs[idx_c]  # [B*H, c, topk]
        col_valid = col_arange.view(1, 1, 1, block_elements) < vbs_sel.unsqueeze(-1)
        attn_mask = col_valid.reshape(B * H * c, 1, 1, topk * block_elements)

        o_c = F.scaled_dot_product_attention(q_c, k_sel, v_sel, attn_mask=attn_mask)
        out_chunks.append(o_c.reshape(B, H, c, block_elements, D))

    out_s = torch.cat(out_chunks, dim=2).reshape(B, H, S_pad, D)
    return out_s


def video_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_compress: torch.Tensor | None,
    meta: VSAMetadata,
    q_chunk_blocks: int = 32,
    offload_ctx=None,
    dbv_box=None,
    dbg_box=None,
) -> torch.Tensor:
    """Run full VSA on raster-order [B, S, H, D] inputs, returning [B, S, H, D]."""
    B, S, H, D = q.shape
    block_elements = math.prod(meta.tile_size)
    vbs = meta.variable_block_sizes

    use_musa_fast = (
        q.device.type == "musa"
        and q.dtype == torch.bfloat16
        and block_elements == 64
        and D == 128
    )

    if use_musa_fast:
        from .fused_ops import fused_vsa_combine_untile, fused_vsa_tile_pool
        from .vsa_kernels import block_sparse_attention_tilelang, fused_topk_qorder_bf16

        # Static 3-slot GPU ring buffers for offloaded (q_t, k_t, v_t, out_s) -> zero 720MiB alloc stalls
        out_qkv = None
        out_s_buf = None
        use_ring = False
        if offload_ctx is not None:
            pool, blk_idx = offload_ctx
            sp = (B, H, meta.padded_seq_length, D)
            qt_b, kt_b, vt_b, os_b = pool.get_ring_slot(
                "vsa", blk_idx, ((sp, q.dtype), (sp, q.dtype), (sp, q.dtype), (sp, q.dtype)), num_slots=3
            )
            out_qkv = (qt_b, kt_b, vt_b)
            out_s_buf = os_b
            use_ring = True

        # 1-3. Fused Tile + Transpose + FP32 MeanPool (no gate tiling; fused db_v in bwd!)
        q_bhsd, k_bhsd, v_bhsd, q_c, k_c, v_c = fused_vsa_tile_pool(
            q, k, v, meta.pad_to_raster, vbs, out_qkv=out_qkv, dbv_box=dbv_box
        )
        scores = torch.matmul(q_c, k_c.transpose(-2, -1)) * (D ** -0.5)
        attn_c = torch.softmax(scores.float(), dim=-1).to(q_bhsd.dtype)
        out_c = torch.matmul(attn_c, v_c)  # [B, H, num_blocks, D]

        # 4. Top-k tile routing (deterministic ascending warp-ballot BF16 top-k + 4-quantile Q-order)
        if scores.dtype == torch.bfloat16 and 64 <= int(vbs.numel()) <= 512:
            topk_idx, q_order = fused_topk_qorder_bf16(scores.detach(), meta.topk)
            pre_sorted = True
        else:
            topk_idx = torch.topk(scores.detach().float(), k=meta.topk, dim=-1, largest=True, sorted=True).indices
            q_order = None
            pre_sorted = False

        # 5. Fine block-sparse attention (TileLang-MUSA block-sparse kernel)
        out_s = block_sparse_attention_tilelang(
            q_bhsd,
            k_bhsd,
            v_bhsd,
            topk_idx,
            vbs,
            block_elements,
            offload_ctx=offload_ctx,
            pre_sorted=pre_sorted,
            q_order=q_order,
            out_buf=out_s_buf,
            is_ring=use_ring,
        )

        # 6-7. Fused Combine + Untile directly with raster gate_compress (fused db_gate in bwd!)
        return fused_vsa_combine_untile(
            out_s, out_c, gate_compress, meta.pad_to_raster, vbs, S, dbg_box=dbg_box
        )

    # Fallback path (non-MUSA / non-BF16 / non-standard tile or head_dim)
    stack = [q, k, v] if gate_compress is None else [q, k, v, gate_compress]
    qkvg = torch.cat(stack, dim=0)
    qkvg_tiled = tile_qkvg(qkvg, meta)
    if gate_compress is None:
        q_t, k_t, v_t = qkvg_tiled.chunk(3, dim=0)
        g_t = None
    else:
        q_t, k_t, v_t, g_t = qkvg_tiled.chunk(4, dim=0)

    q_bhsd = q_t.transpose(1, 2).contiguous()
    k_bhsd = k_t.transpose(1, 2).contiguous()
    v_bhsd = v_t.transpose(1, 2).contiguous()
    g_bhsd = None if g_t is None else g_t.transpose(1, 2)

    num_blocks = int(vbs.numel())
    q_c = _block_mean_pool(q_bhsd, vbs, block_elements)
    k_c = _block_mean_pool(k_bhsd, vbs, block_elements)
    v_c = _block_mean_pool(v_bhsd, vbs, block_elements)

    scores = torch.matmul(q_c, k_c.transpose(-2, -1)) * (D ** -0.5)
    attn_c = torch.softmax(scores.float(), dim=-1).to(q_bhsd.dtype)
    out_c = torch.matmul(attn_c, v_c)

    topk_idx = torch.topk(scores.detach().float(), k=meta.topk, dim=-1, largest=True, sorted=True).indices
    out_s = _block_sparse_attention_pytorch(
        q_bhsd, k_bhsd, v_bhsd, topk_idx, vbs, block_elements, q_chunk_blocks=q_chunk_blocks
    )

    out_s_b = out_s.view(B, H, num_blocks, block_elements, D)
    out_c_exp = out_c.unsqueeze(3)
    if g_bhsd is not None:
        g_b = g_bhsd.view(B, H, num_blocks, block_elements, D)
        combined = torch.addcmul(out_s_b, out_c_exp, g_b)
    else:
        combined = out_s_b + out_c_exp

    out_bshd = combined.view(B, H, meta.padded_seq_length, D).transpose(1, 2)
    return untile_output(out_bshd, meta)
