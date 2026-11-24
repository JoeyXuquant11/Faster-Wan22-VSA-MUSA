"""End-to-end Wan2.2-TI2V-5B + VSA full finetuning entry point for MTT S5000."""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch_musa  # noqa: F401

from wan22_vsa.data import build_dataloader
from wan22_vsa.distributed import (
    cleanup_distributed,
    init_distributed,
    save_full_checkpoint,
    wrap_fsdp2,
)
from wan22_vsa.model import load_wan22_ti2v_5b_vsa
from wan22_vsa.vsa import compute_vsa_sparsity


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wan2.2-TI2V-5B + VSA Full Finetuning (MTT S5000)")
    parser.add_argument("--model_dir", type=str, default=os.environ.get("WAN22_MODEL_DIR", "./assets/models/Wan2.2-TI2V-5B"))
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/tmp/wan22_vsa_out")
    parser.add_argument("--num_latent_t", type=int, default=31)
    parser.add_argument("--latent_height", type=int, default=44)
    parser.add_argument("--latent_width", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--flow_shift", type=float, default=5.0)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--vsa_sparsity", type=float, default=0.8)
    parser.add_argument("--vsa_decay_rate", type=float, default=0.0)
    parser.add_argument("--vsa_decay_interval_steps", type=int, default=50)
    parser.add_argument("--vsa_q_chunk_blocks", type=int, default=32)
    parser.add_argument("--ti2v_first_frame_cond", action="store_true", default=False)
    parser.add_argument("--no_gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--preload_dataset", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--num_layers_override", type=int, default=None)
    return parser.parse_args(argv)


def seed_rank(seed: int, rank: int) -> None:
    s = int(seed) + int(rank) * 1009
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.musa.is_available():
        torch.musa.manual_seed_all(s)


def sample_logit_normal_sigmas(
    batch_size: int,
    device: torch.device,
    flow_shift: float = 5.0,
) -> torch.Tensor:
    """Rectified Flow logit-normal sigma sampling with Wan2.2 flow shift."""
    u = torch.sigmoid(torch.randn((batch_size,), device=device, dtype=torch.float32))
    # Apply flow shift: sigma = (shift * u) / (1 + (shift - 1) * u)
    sigma = (flow_shift * u) / (1.0 + (flow_shift - 1.0) * u)
    return sigma.clamp(1e-4, 1.0 - 1e-4)


class Trainer:
    """Full finetuning trainer for Wan2.2-TI2V-5B + VSA on MTT S5000."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.rank, self.world_size, self.local_rank, self.device = init_distributed()
        seed_rank(args.seed, self.rank)
        self.start_step = 0
        self.step = 0

        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.dataloader = self._build_dataloader()
        self.data_iter = iter(self.dataloader)

    def _build_model(self) -> torch.nn.Module:
        model = load_wan22_ti2v_5b_vsa(
            self.args.model_dir,
            dtype=torch.float32,
            vsa_q_chunk_blocks=self.args.vsa_q_chunk_blocks,
            num_layers_override=self.args.num_layers_override,
        )
        model.gradient_checkpointing = not self.args.no_gradient_checkpointing
        model.train()
        if self.world_size > 1:
            model = wrap_fsdp2(model, param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        else:
            model = model.to(device=self.device, dtype=torch.bfloat16)
        return model

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            betas=(self.args.beta1, self.args.beta2),
            eps=self.args.eps,
            weight_decay=self.args.weight_decay,
        )

    def _build_dataloader(self):
        return build_dataloader(
            data_dir=self.args.data_dir,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            rank=self.rank,
            world_size=self.world_size,
            seed=self.args.seed,
            num_latent_t=self.args.num_latent_t,
            latent_height=self.args.latent_height,
            latent_width=self.args.latent_width,
            preload=self.args.preload_dataset,
        )

    def _prepare_flow_inputs(self, batch: Dict[str, torch.Tensor]):
        latents = batch["latents"].to(device=self.device, dtype=torch.bfloat16)
        text_embeds = batch["text_embeds"].to(device=self.device, dtype=torch.bfloat16)
        B, C, T, H, W = latents.shape

        noise = torch.randn(latents.shape, device=self.device, dtype=torch.float32)
        sigma = sample_logit_normal_sigmas(B, self.device, flow_shift=self.args.flow_shift)
        sigma_5d = sigma.view(B, 1, 1, 1, 1)

        latents_fp32 = latents.float()
        noisy = (1.0 - sigma_5d) * latents_fp32 + sigma_5d * noise
        target = noise - latents_fp32

        if self.args.ti2v_first_frame_cond and T > 1:
            # TI2V conditioning: frame 0 stays clean with timestep 0
            noisy[:, :, 0:1] = latents_fp32[:, :, 0:1]
            f_p = T // self.model.patch_size[0]
            h_p = H // self.model.patch_size[1]
            w_p = W // self.model.patch_size[2]
            t_per_frame = sigma.view(B, 1).expand(B, f_p).clone() * float(self.args.num_train_timesteps)
            t_per_frame[:, 0] = 0.0
            timestep = t_per_frame.unsqueeze(-1).expand(B, f_p, h_p * w_p).reshape(B, f_p * h_p * w_p).to(torch.bfloat16)
            loss_mask = torch.ones((B, 1, T, 1, 1), device=self.device, dtype=torch.float32)
            loss_mask[:, :, 0:1] = 0.0
        else:
            timestep = (sigma * float(self.args.num_train_timesteps)).to(torch.bfloat16)
            loss_mask = None

        return noisy.to(torch.bfloat16), target, timestep, text_embeds, loss_mask

    def train_one_step(self, step: int) -> Dict[str, Any]:
        self.step = step
        self.optimizer.zero_grad(set_to_none=True)
        grad_accum = self.args.gradient_accumulation_steps
        cur_sparsity = compute_vsa_sparsity(
            step=step,
            target_sparsity=self.args.vsa_sparsity,
            decay_rate=self.args.vsa_decay_rate,
            decay_interval_steps=self.args.vsa_decay_interval_steps,
        )

        total_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        for _ in range(grad_accum):
            batch = next(self.data_iter)
            noisy, target, timestep, text_embeds, loss_mask = self._prepare_flow_inputs(batch)
            pred = self.model(
                hidden_states=noisy,
                timestep=timestep,
                encoder_hidden_states=text_embeds,
                vsa_sparsity=cur_sparsity,
            )
            sq_err = (pred.float() - target) ** 2
            if loss_mask is not None:
                loss = (sq_err * loss_mask).sum() / (loss_mask.sum() * sq_err.shape[1] * sq_err.shape[3] * sq_err.shape[4])
            else:
                loss = sq_err.mean()
            scaled_loss = loss / float(grad_accum)
            scaled_loss.backward()
            total_loss = total_loss + scaled_loss.detach()

        if dist.is_initialized():
            dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.args.max_grad_norm
        )
        self.optimizer.step()

        return {
            "loss": total_loss,
            "grad_norm": grad_norm.detach() if isinstance(grad_norm, torch.Tensor) else torch.tensor(float(grad_norm), device=self.device),
            "vsa_sparsity": cur_sparsity,
        }

    def train(self) -> None:
        for step in range(self.start_step, self.args.max_train_steps):
            t0 = time.perf_counter()
            metrics = self.train_one_step(step)
            torch.musa.synchronize()
            dt = time.perf_counter() - t0

            if self.rank == 0 and (step % self.args.log_steps == 0 or step == self.args.max_train_steps - 1):
                loss_val = float(metrics["loss"].item())
                gn_val = float(metrics["grad_norm"].item())
                mem_gib = torch.musa.max_memory_allocated(self.device) / (1024 ** 3)
                print(
                    f"[step {step + 1}/{self.args.max_train_steps}] "
                    f"loss={loss_val:.6f} grad_norm={gn_val:.4f} "
                    f"vsa_sparsity={metrics['vsa_sparsity']:.2f} "
                    f"step_time={dt:.3f}s peak_mem={mem_gib:.2f}GiB",
                    flush=True,
                )

            if self.args.save_steps > 0 and (step + 1) % self.args.save_steps == 0:
                save_full_checkpoint(self.model, self.args.output_dir, step + 1, self.rank)


def main() -> None:
    args = parse_args()
    trainer = Trainer(args)
    try:
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
