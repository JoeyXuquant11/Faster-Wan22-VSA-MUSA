"""Paired end-to-end training & checkpoint/eval validation for Wan2.2-TI2V-5B + VSA."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch_musa  # noqa: F401
from torch.distributed.tensor import DTensor


def gather_param_fp32(p: torch.Tensor) -> torch.Tensor:
    if isinstance(p, DTensor):
        return p.full_tensor().detach().float().cpu()
    return p.detach().float().cpu()


@torch.no_grad()
def evaluate_fixed(trainer, num_eval_batches: int = 4, eval_seed: int = 99999) -> dict:
    """Deterministic held-out velocity MSE eval over fixed (sample, noise, sigma) draws."""
    model = trainer.model
    was_training = model.training
    model.eval()
    ds = trainer.dataloader.dataset
    results = {}
    for sp in (0.8, 0.0):
        g_cpu = torch.Generator(device="cpu").manual_seed(eval_seed + trainer.rank * 7919)
        losses = []
        for b in range(num_eval_batches):
            idx = (b * trainer.world_size + trainer.rank) % len(ds)
            item = ds[idx]
            latents = item["latents"].unsqueeze(0).to(device=trainer.device, dtype=torch.bfloat16)
            text_embeds = item["text_embeds"].unsqueeze(0).to(device=trainer.device, dtype=torch.bfloat16)
            noise = torch.randn(latents.shape, generator=g_cpu, dtype=torch.float32).to(trainer.device)
            u = torch.sigmoid(torch.randn((1,), generator=g_cpu, dtype=torch.float32)).to(trainer.device)
            shift = trainer.args.flow_shift
            sigma = ((shift * u) / (1.0 + (shift - 1.0) * u)).clamp(1e-4, 1.0 - 1e-4)
            s5 = sigma.view(1, 1, 1, 1, 1)
            lat32 = latents.float()
            noisy = ((1.0 - s5) * lat32 + s5 * noise).to(torch.bfloat16)
            target = noise - lat32
            timestep = (sigma * float(trainer.args.num_train_timesteps)).to(torch.bfloat16)
            pred = model(
                hidden_states=noisy,
                timestep=timestep,
                encoder_hidden_states=text_embeds,
                vsa_sparsity=sp,
            )
            loss = ((pred.float() - target) ** 2).mean()
            losses.append(loss)
        mean_loss = torch.stack(losses).mean()
        if dist.is_initialized():
            dist.all_reduce(mean_loss, op=dist.ReduceOp.AVG)
        results[f"eval_loss_sp{int(sp * 100)}"] = float(mean_loss.item())
    if was_training:
        model.train()
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--ckpt_path", default="")
    p.add_argument("--save_init_path", default="")
    args, train_argv = p.parse_known_args()

    sys.path.insert(0, args.repo_dir)
    import train as wan_train  # noqa: E402

    targs = wan_train.parse_args(train_argv)
    trainer = wan_train.Trainer(targs)
    rank = trainer.rank

    try:
        if args.save_init_path:
            init_sd = {}
            for name, param in trainer.model.named_parameters():
                t = gather_param_fp32(param)
                if rank == 0:
                    init_sd[name] = t
            if rank == 0:
                Path(args.save_init_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(init_sd, args.save_init_path)
            del init_sd
            if dist.is_initialized():
                dist.barrier()

        eval_step0 = evaluate_fixed(trainer, num_eval_batches=4, eval_seed=99999)
        torch.musa.synchronize()
        torch.musa.reset_peak_memory_stats(trainer.device)
        wan_train.seed_rank(targs.seed, rank)

        steps_log = []
        for step in range(targs.max_train_steps):
            torch.musa.synchronize()
            t0 = time.perf_counter()
            metrics = trainer.train_one_step(step)
            torch.musa.synchronize()
            dt = time.perf_counter() - t0

            rec = {
                "step": step + 1,
                "loss": float(metrics["loss"].item()),
                "grad_norm": float(metrics["grad_norm"].item()),
                "step_time_s": round(dt, 4),
                "peak_alloc_gib": round(torch.musa.max_memory_allocated(trainer.device) / (1024 ** 3), 3),
                "peak_reserved_gib": round(torch.musa.max_memory_reserved(trainer.device) / (1024 ** 3), 3),
            }
            steps_log.append(rec)
            if rank == 0:
                print(
                    f"[{args.tag}] step {step+1:02d}/{targs.max_train_steps} "
                    f"loss={rec['loss']:.6f} gn={rec['grad_norm']:.4f} dt={dt:.3f}s",
                    flush=True,
                )

        eval_final = evaluate_fixed(trainer, num_eval_batches=4, eval_seed=99999)
        if args.ckpt_path:
            final_sd = {}
            for name, param in trainer.model.named_parameters():
                t = gather_param_fp32(param)
                if rank == 0:
                    final_sd[name] = t
            if rank == 0:
                Path(args.ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(final_sd, args.ckpt_path)
            del final_sd
            if dist.is_initialized():
                dist.barrier()

        if rank == 0:
            steady = [r["step_time_s"] for r in steps_log[2:]] if len(steps_log) > 2 else [r["step_time_s"] for r in steps_log]
            summary = {
                "tag": args.tag,
                "world_size": trainer.world_size,
                "batch_size_per_gpu": targs.batch_size,
                "gradient_accumulation_steps": targs.gradient_accumulation_steps,
                "global_batch_size": trainer.world_size * targs.batch_size * targs.gradient_accumulation_steps,
                "max_train_steps": targs.max_train_steps,
                "seed": targs.seed,
                "learning_rate": targs.learning_rate,
                "vsa_sparsity": targs.vsa_sparsity,
                "fsdp_reshard_env": os.environ.get("WAN22_FSDP_RESHARD", "default"),
                "eval_step0": eval_step0,
                "eval_final": eval_final,
                "steady_mean_step_s": round(sum(steady) / len(steady), 4),
                "peak_alloc_gib": max(r["peak_alloc_gib"] for r in steps_log),
                "peak_reserved_gib": max(r["peak_reserved_gib"] for r in steps_log),
                "steps": steps_log,
            }
            Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out_json).write_text(json.dumps(summary, indent=2) + "\n")
    finally:
        wan_train.cleanup_distributed()


if __name__ == "__main__":
    main()
