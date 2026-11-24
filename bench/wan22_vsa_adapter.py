"""ResearchBoost-Agent numerics adapter for Wan2.2-TI2V-5B + VSA full finetuning on 8x MTT S5000.

In deterministic mode only (torch.are_deterministic_algorithms_enabled()), routes MUSA's
nondeterministic atomicAdd backward kernels (F.scaled_dot_product_attention dQ and PyTorch
VSA gather scatter-add) to O(N)-memory, same-precision deterministic implementations
(_DetChunkedSDPA and _DetBlockSparseAttn using deterministic bmm reduction), applied
identically to the reference and every candidate using those PyTorch paths
(FORMULATOR_BRIEF.md section 2.7). Candidates implementing custom VSA kernels are NOT
bypassed and are judged directly.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_musa  # noqa: F401
from torch.distributed.tensor import DTensor

from numerics.adapter import Adapter, Objects
from bench import recipe

WRAPPER_PREFIXES = ("module.", "_orig_mod.", "_checkpoint_wrapped_module.")
_ORIG_SDPA = F.scaled_dot_product_attention


class _DetChunkedSDPA(torch.autograd.Function):
    """O(N)-memory deterministic BF16/FP32-softmax SDPA via chunked matmul."""

    @staticmethod
    def forward(ctx, q, k, v, attn_mask, scale):
        sc = (q.shape[-1] ** -0.5) if scale is None else float(scale)
        S_q = q.shape[-2]
        q_chunk = 2048 if S_q > 2048 else S_q
        outs = []
        for i in range(0, S_q, q_chunk):
            qi = q[..., i : i + q_chunk, :]
            logits = torch.matmul(qi, k.transpose(-2, -1)).float() * sc
            if attn_mask is not None:
                mi = attn_mask if attn_mask.shape[-2] == 1 else attn_mask[..., i : i + q_chunk, :]
                logits = logits.masked_fill(~mi, float("-inf")) if mi.dtype == torch.bool else (logits + mi.float())
            probs = torch.softmax(logits, dim=-1).to(q.dtype)
            outs.append(torch.matmul(probs, v))
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-2)
        ctx.save_for_backward(q, k, v, attn_mask)
        ctx.scale = sc
        ctx.q_chunk = q_chunk
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, attn_mask = ctx.saved_tensors
        sc, q_chunk = ctx.scale, ctx.q_chunk
        S_q = q.shape[-2]
        dq = torch.empty_like(q)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        kt = k.transpose(-2, -1)
        vt = v.transpose(-2, -1)
        for i in range(0, S_q, q_chunk):
            j = min(i + q_chunk, S_q)
            qi = q[..., i:j, :]
            doi = grad_out[..., i:j, :]
            logits = torch.matmul(qi, kt).float() * sc
            if attn_mask is not None:
                mi = attn_mask if attn_mask.shape[-2] == 1 else attn_mask[..., i:j, :]
                logits = logits.masked_fill(~mi, float("-inf")) if mi.dtype == torch.bool else (logits + mi.float())
            p_fp32 = torch.softmax(logits, dim=-1)
            p = p_fp32.to(q.dtype)
            dv.add_(torch.matmul(p.transpose(-2, -1), doi).float())
            dp = torch.matmul(doi, vt).float()
            ds = (p_fp32 * (dp - (dp * p_fp32).sum(dim=-1, keepdim=True)) * sc).to(q.dtype)
            dq[..., i:j, :] = torch.matmul(ds, k)
            dk.add_(torch.matmul(ds.transpose(-2, -1), qi).float())
        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None


class _DetBlockSparseAttn(torch.autograd.Function):
    """O(N)-memory, zero-atomicAdd deterministic VSA block-sparse attention."""

    @staticmethod
    def forward(ctx, q, k, v, topk_idx, vbs, block_elements, q_chunk_blocks):
        B, H, S_pad, D = q.shape
        num_blocks = S_pad // block_elements
        topk = topk_idx.shape[-1]
        sc = D ** -0.5
        q_flat = q.view(B * H, num_blocks, block_elements, D)
        k_flat = k.view(B * H, num_blocks, block_elements, D)
        v_flat = v.view(B * H, num_blocks, block_elements, D)
        idx_flat = topk_idx.reshape(B * H, num_blocks, topk)
        bh = torch.arange(B * H, device=q.device, dtype=torch.long)[:, None, None]
        col_arange = torch.arange(block_elements, device=q.device, dtype=torch.int32)
        out = torch.empty_like(q_flat)
        qc = min(int(q_chunk_blocks), 16)
        for q0 in range(0, num_blocks, qc):
            q1 = min(q0 + qc, num_blocks)
            c = q1 - q0
            idx_c = idx_flat[:, q0:q1, :]
            k_sel = k_flat[bh, idx_c].reshape(B * H * c, topk * block_elements, D)
            v_sel = v_flat[bh, idx_c].reshape(B * H * c, topk * block_elements, D)
            q_c = q_flat[:, q0:q1].reshape(B * H * c, block_elements, D)
            col_valid = (col_arange.view(1, 1, 1, block_elements) < vbs[idx_c].unsqueeze(-1)).reshape(
                B * H * c, 1, topk * block_elements
            )
            logits = torch.bmm(q_c, k_sel.transpose(1, 2)).float() * sc
            logits = logits.masked_fill(~col_valid, float("-inf"))
            p = torch.softmax(logits, dim=-1).to(q.dtype)
            out[:, q0:q1] = torch.bmm(p, v_sel).reshape(B * H, c, block_elements, D)
        ctx.save_for_backward(q, k, v, topk_idx, vbs)
        ctx.block_elements = block_elements
        ctx.q_chunk = qc
        return out.view(B, H, S_pad, D)

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, topk_idx, vbs = ctx.saved_tensors
        block_elements = ctx.block_elements
        qc = ctx.q_chunk
        B, H, S_pad, D = q.shape
        num_blocks = S_pad // block_elements
        topk = topk_idx.shape[-1]
        sc = D ** -0.5
        q_flat = q.view(B * H, num_blocks, block_elements, D)
        k_flat = k.view(B * H, num_blocks, block_elements, D)
        v_flat = v.view(B * H, num_blocks, block_elements, D)
        do_flat = grad_out.contiguous().view(B * H, num_blocks, block_elements, D)
        idx_flat = topk_idx.reshape(B * H, num_blocks, topk)
        bh = torch.arange(B * H, device=q.device, dtype=torch.long)[:, None, None]
        col_arange = torch.arange(block_elements, device=q.device, dtype=torch.int32)

        dq = torch.empty_like(q_flat)
        dk = torch.zeros(B * H, num_blocks, block_elements * D, device=q.device, dtype=torch.float32)
        dv = torch.zeros(B * H, num_blocks, block_elements * D, device=q.device, dtype=torch.float32)
        for q0 in range(0, num_blocks, qc):
            q1 = min(q0 + qc, num_blocks)
            c = q1 - q0
            M = c * topk
            idx_c = idx_flat[:, q0:q1, :]
            k_sel = k_flat[bh, idx_c].reshape(B * H * c, topk * block_elements, D)
            v_sel = v_flat[bh, idx_c].reshape(B * H * c, topk * block_elements, D)
            q_c = q_flat[:, q0:q1].reshape(B * H * c, block_elements, D)
            do_c = do_flat[:, q0:q1].reshape(B * H * c, block_elements, D)
            col_valid = (col_arange.view(1, 1, 1, block_elements) < vbs[idx_c].unsqueeze(-1)).reshape(
                B * H * c, 1, topk * block_elements
            )
            logits = torch.bmm(q_c, k_sel.transpose(1, 2)).float() * sc
            logits = logits.masked_fill(~col_valid, float("-inf"))
            p_fp32 = torch.softmax(logits, dim=-1)
            p = p_fp32.to(q.dtype)

            dv_sel = torch.bmm(p.transpose(1, 2), do_c).reshape(B * H, M, block_elements * D)
            dp = torch.bmm(do_c, v_sel.transpose(1, 2)).float()
            ds = (p_fp32 * (dp - (dp * p_fp32).sum(dim=-1, keepdim=True)) * sc).to(q.dtype)
            dq[:, q0:q1] = torch.bmm(ds, k_sel).reshape(B * H, c, block_elements, D)
            dk_sel = torch.bmm(ds.transpose(1, 2), q_c).reshape(B * H, M, block_elements * D)

            oh = torch.zeros(B * H, num_blocks, M, device=q.device, dtype=q.dtype)
            oh.scatter_(1, idx_c.reshape(B * H, 1, M), 1.0)
            dk.add_(torch.bmm(oh, dk_sel).float())
            dv.add_(torch.bmm(oh, dv_sel).float())

        return (
            dq.view(B, H, S_pad, D),
            dk.to(k.dtype).view(B, H, S_pad, D),
            dv.to(v.dtype).view(B, H, S_pad, D),
            None,
            None,
            None,
            None,
        )


def _det_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    if dropout_p != 0.0 or is_causal or enable_gqa:
        return _ORIG_SDPA(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa
        )
    return _DetChunkedSDPA.apply(q, k, v, attn_mask, scale)


def _det_block_sparse_attention_pytorch(q, k, v, topk_idx, vbs, block_elements, q_chunk_blocks=32):
    return _DetBlockSparseAttn.apply(q, k, v, topk_idx, vbs, block_elements, q_chunk_blocks)


def _strip_wrappers(name: str) -> str:
    for p in WRAPPER_PREFIXES:
        name = name.replace(p, "")
    return name


def _digest_batch(batch) -> str:
    h = hashlib.sha256()
    items = batch.items() if isinstance(batch, dict) else [("tensor", batch)]
    for k, v in sorted(items, key=lambda kv: str(kv[0])):
        h.update(str(k).encode())
        if isinstance(v, torch.Tensor):
            h.update(str(v.dtype).encode() + str(tuple(v.shape)).encode())
            h.update(v.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
        else:
            h.update(repr(v).encode())
    return h.hexdigest()[:16]


class Feed:
    def __init__(self, it):
        self.it = it
        self.digests = []

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self.it)
        self.digests.append(_digest_batch(batch))
        return batch


class Wan22VSAAdapter(Adapter):
    name = "wan22_vsa_s5000"

    def __init__(self, tree: str, scratch: str = "/tmp/wan22_vsa_rba_tmp", lr_scale: float = 1.0):
        self.tree = str(Path(tree).resolve())
        self.scratch = scratch
        self.lr_scale = float(lr_scale)
        self.trainer = None
        self.feed = None
        self._case = {}

    def configure(self, case):
        self._case = dict(case or {})

    def build(self, seed, golden=False):
        if golden:
            raise NotImplementedError("fp64 golden not used for 5.28B model")
        if self.tree not in sys.path:
            sys.path.insert(0, self.tree)
        for mod_name in list(sys.modules):
            if mod_name == "train" or mod_name.startswith("wan22_vsa"):
                del sys.modules[mod_name]

        train_mod = importlib.import_module("train")
        if torch.are_deterministic_algorithms_enabled():
            F.scaled_dot_product_attention = _det_sdpa
            vsa_mod = sys.modules.get("wan22_vsa.vsa")
            if vsa_mod is not None and hasattr(vsa_mod, "_block_sparse_attention_pytorch"):
                vsa_mod._block_sparse_attention_pytorch = _det_block_sparse_attention_pytorch

        steps = int(self._case.get("steps", 2)) + 2
        lr = recipe.RECIPE["learning_rate"] * float(self._case.get("lr_scale", self.lr_scale))
        argv = recipe.build_argv(
            max_train_steps=steps,
            output_dir=Path(self.scratch) / "adapter_out",
            seed=int(seed),
            learning_rate=lr,
        )
        args = train_mod.parse_args(argv)
        self.trainer = train_mod.Trainer(args)
        self.feed = Feed(self.trainer.data_iter)
        self.trainer.data_iter = self.feed

    def objects(self):
        return Objects(
            models={"dit": self.trainer.model},
            optimizers={"adamw": self.trainer.optimizer},
            sync={"dit": "local"},
            capture_outputs=("dit.blocks.0", "dit.blocks.14", "dit.blocks.29", "dit.head"),
            count_forwards=True,
        )

    def phases(self):
        return ["train"]

    def batch(self, step, rank):
        return None

    def _gate_top10_abs_mean(self) -> torch.Tensor:
        vals = []
        blocks = self.trainer.model.blocks
        for idx in (0, len(blocks) // 2, len(blocks) - 1):
            p = blocks[idx].self_attn.to_gate_compress.weight
            t = p.full_tensor() if isinstance(p, DTensor) else p
            flat = t.detach().float().abs().reshape(-1)
            k = max(1, flat.numel() // 10)
            topk_vals = torch.topk(flat, k=k, largest=True, sorted=False).values
            vals.append(topk_vals.mean())
        return torch.stack(vals).mean().detach()

    def step(self, step, batch):
        metrics = self.trainer.train_one_step(step)
        gate_metric = self._gate_top10_abs_mean()
        return {
            "loss": metrics["loss"],
            "grad_norm": metrics["grad_norm"],
            "gate_top10_abs_mean": gate_metric,
        }

    def invariants(self):
        return {
            "step": int(self.trainer.step),
            "world_size": int(self.trainer.world_size),
            "vsa_sparsity": float(self.trainer.args.vsa_sparsity),
            "flow_shift": float(self.trainer.args.flow_shift),
        }

    def canonical(self, kind, name, tensor):
        clean_name = _strip_wrappers(name)
        t = tensor.to_local() if isinstance(tensor, DTensor) else tensor
        canon_fn = getattr(self.trainer.model, "canonicalize_param", None)
        if callable(canon_fn):
            mapped = canon_fn(kind, clean_name, t)
            if mapped is not None:
                return {k: v.detach().cpu() for k, v in mapped.items()}
        return {clean_name: t.detach().cpu()}
