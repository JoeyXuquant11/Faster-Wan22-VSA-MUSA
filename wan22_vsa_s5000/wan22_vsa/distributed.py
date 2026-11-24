"""MUSA distributed initialization (MCCL) and FSDP2 wrapping for Wan2.2-TI2V-5B."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import torch
import torch.distributed as dist
import torch_musa  # noqa: F401
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor


def init_distributed() -> Tuple[int, int, int, torch.device]:
    """Initialize MCCL process group on MTT S5000 (or single-GPU fallback)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.musa.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="mccl")
        device = torch.device(f"musa:{local_rank}")
        return rank, world_size, local_rank, device
    torch.musa.set_device(0)
    return 0, 1, 0, torch.device("musa:0")


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_fsdp2(
    model: torch.nn.Module,
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
) -> torch.nn.Module:
    """Wrap WanVSATransformerBlocks (bucketed) and root model with PyTorch FSDP2."""
    bpb = int(os.environ.get("WAN22_FSDP_BPB", "5"))
    reshard = os.environ.get("WAN22_FSDP_RESHARD", "0") == "1"
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=False)
    blocks = list(model.blocks)
    if bpb <= 1:
        for block in blocks:
            fully_shard(block, mp_policy=mp_policy, reshard_after_forward=reshard)
    else:
        for i in range(0, len(blocks), bpb):
            grp = blocks[i : i + bpb]
            fully_shard(grp if len(grp) > 1 else grp[0], mp_policy=mp_policy, reshard_after_forward=reshard)
    fully_shard(model, mp_policy=mp_policy, reshard_after_forward=reshard)
    return model


def save_full_checkpoint(
    model: torch.nn.Module,
    output_dir: str | Path,
    step: int,
    rank: int,
) -> None:
    """Gather FSDP2/DTensor state dict and write checkpoint on rank 0."""
    out_dir = Path(output_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    full_sd = {}
    for k, v in model.state_dict().items():
        if isinstance(v, DTensor):
            t = v.full_tensor().detach().cpu()
        else:
            t = v.detach().cpu()
        if rank == 0:
            full_sd[k] = t
    if rank == 0:
        ckpt_path = out_dir / f"checkpoint_step_{step:06d}.pt"
        torch.save({"step": step, "model": full_sd}, ckpt_path)
    if dist.is_initialized():
        dist.barrier()
