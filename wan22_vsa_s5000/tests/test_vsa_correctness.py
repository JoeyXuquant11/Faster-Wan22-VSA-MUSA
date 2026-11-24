"""Unit test for VSA forward/backward numerical correctness and determinism on MTT S5000."""

import math
import sys
from pathlib import Path

import torch
import torch_musa  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wan22_vsa.vsa import build_vsa_metadata, tile_qkvg, untile_output, video_sparse_attention


def explicit_vsa_reference(q, k, v, gate, meta):
    """Independent unchunked explicit FP32-score reference for VSA."""
    B, S, H, D = q.shape
    blk = math.prod(meta.tile_size)
    qkvg = torch.cat([q, k, v, gate], dim=0)
    q_t, k_t, v_t, g_t = tile_qkvg(qkvg, meta).chunk(4, dim=0)
    q_h = q_t.transpose(1, 2).float()
    k_h = k_t.transpose(1, 2).float()
    v_h = v_t.transpose(1, 2).float()
    g_h = g_t.transpose(1, 2).float()
    vbs = meta.variable_block_sizes
    n_blk = int(vbs.numel())

    # Coarse
    q_c = q_h.view(B, H, n_blk, blk, D).sum(dim=3) / vbs.float().view(1, 1, n_blk, 1)
    k_c = k_h.view(B, H, n_blk, blk, D).sum(dim=3) / vbs.float().view(1, 1, n_blk, 1)
    v_c = v_h.view(B, H, n_blk, blk, D).sum(dim=3) / vbs.float().view(1, 1, n_blk, 1)
    scores = torch.matmul(q_c, k_c.transpose(-2, -1)) / math.sqrt(D)
    out_c = torch.matmul(torch.softmax(scores, dim=-1), v_c)  # [B, H, n_blk, D]

    # Top-k mask expanded to full [B, H, S_pad, S_pad]
    topk_idx = torch.topk(scores.detach(), k=meta.topk, dim=-1, largest=True, sorted=True).indices
    blk_mask = torch.zeros(B, H, n_blk, n_blk, device=q.device, dtype=torch.bool)
    blk_mask.scatter_(-1, topk_idx, True)
    valid_kv = (torch.arange(blk, device=q.device)[None, :] < vbs[:, None]).reshape(1, 1, 1, n_blk * blk)
    full_mask = blk_mask.repeat_interleave(blk, dim=2).repeat_interleave(blk, dim=3) & valid_kv

    attn_logits = torch.matmul(q_h, k_h.transpose(-2, -1)) / math.sqrt(D)
    attn_logits = attn_logits.masked_fill(~full_mask, float("-inf"))
    out_s = torch.matmul(torch.softmax(attn_logits, dim=-1), v_h)

    combined = out_s.view(B, H, n_blk, blk, D) + out_c.unsqueeze(3) * g_h.view(B, H, n_blk, blk, D)
    out_bshd = combined.view(B, H, meta.padded_seq_length, D).transpose(1, 2)
    return untile_output(out_bshd, meta)


def main():
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = torch.device("musa:0")
    # Non-divisible 3D grid to exercise boundary variable_block_sizes
    shape = (5, 7, 9)
    S = math.prod(shape)
    B, H, D = 1, 6, 128

    # 1. Sparse forward + backward vs explicit FP32 reference
    meta = build_vsa_metadata(shape, sparsity=0.75, device=dev)
    torch.manual_seed(123)
    q0 = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16)
    k0 = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16)
    v0 = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16)
    g0 = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16) * 0.25
    grad_out = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16)

    q = q0.clone().requires_grad_(True)
    k = k0.clone().requires_grad_(True)
    v = v0.clone().requires_grad_(True)
    g = g0.clone().requires_grad_(True)
    out = video_sparse_attention(q, k, v, g, meta, q_chunk_blocks=4)
    (out * grad_out).sum().backward()

    q_ref = q0.float().clone().requires_grad_(True)
    k_ref = k0.float().clone().requires_grad_(True)
    v_ref = v0.float().clone().requires_grad_(True)
    g_ref = g0.float().clone().requires_grad_(True)
    out_ref = explicit_vsa_reference(q_ref, k_ref, v_ref, g_ref, meta)
    (out_ref * grad_out.float()).sum().backward()

    for name, a, b in [
        ("out", out.float(), out_ref),
        ("dq", q.grad.float(), q_ref.grad),
        ("dk", k.grad.float(), k_ref.grad),
        ("dv", v.grad.float(), v_ref.grad),
        ("dgate", g.grad.float(), g_ref.grad),
    ]:
        rel_l2 = (a - b).norm().item() / (b.norm().item() + 1e-12)
        max_abs = (a - b).abs().max().item()
        print(f"[sparse {name:5s}] rel_l2={rel_l2:.6f} max_abs={max_abs:.6f}")
        assert rel_l2 < 2e-2, f"{name} rel_l2 too large: {rel_l2}"

    # 2. Dense limit (sparsity=0.0, gate=0) matches unpadded dense SDPA + nonzero dgate!
    meta_dense = build_vsa_metadata(shape, sparsity=0.0, device=dev)
    q = q0.clone().requires_grad_(True)
    k = k0.clone().requires_grad_(True)
    v = v0.clone().requires_grad_(True)
    g = torch.zeros_like(g0, requires_grad=True)
    out_vsa_dense = video_sparse_attention(q, k, v, g, meta_dense, q_chunk_blocks=4)
    (out_vsa_dense * grad_out).sum().backward()

    q_d = q0.transpose(1, 2).contiguous()
    k_d = k0.transpose(1, 2).contiguous()
    v_d = v0.transpose(1, 2).contiguous()
    out_sdpa = torch.nn.functional.scaled_dot_product_attention(q_d, k_d, v_d).transpose(1, 2)
    dense_rel_l2 = (out_vsa_dense.float() - out_sdpa.float()).norm().item() / out_sdpa.float().norm().item()
    dgate_norm = g.grad.float().norm().item()
    print(f"[dense-limit] rel_l2_vs_sdpa={dense_rel_l2:.6f} zero_gate_dgate_norm={dgate_norm:.4f}")
    assert dense_rel_l2 < 1e-2, f"Dense limit mismatch: {dense_rel_l2}"
    assert dgate_norm > 0.0, "Zero-initialized gate_compress must still receive nonzero gradient!"

    # 3. Bitwise determinism across two runs
    outs, dqs = [], []
    for _ in range(2):
        q1 = q0.clone().requires_grad_(True)
        k1 = k0.clone().requires_grad_(True)
        v1 = v0.clone().requires_grad_(True)
        g1 = g0.clone().requires_grad_(True)
        o1 = video_sparse_attention(q1, k1, v1, g1, meta, q_chunk_blocks=4)
        (o1 * grad_out).sum().backward()
        outs.append(o1.detach().clone())
        dqs.append(q1.grad.detach().clone())
    assert torch.equal(outs[0], outs[1]), "VSA forward not bitwise deterministic!"
    assert torch.equal(dqs[0], dqs[1]), "VSA backward not bitwise deterministic!"
    print("ALL VSA CORRECTNESS & DETERMINISM TESTS PASSED!")


if __name__ == "__main__":
    main()
