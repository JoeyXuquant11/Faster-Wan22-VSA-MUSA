"""Steady-state iteration timing & Torch-MUSA profiling of train.py without modifying the training code.

Run under torchrun with the candidate/reference tree as cwd:
  torchrun --standalone --nproc_per_node=8 <bench>/timing.py --out <prefix> [options] -- <train.py args>
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import runpy
import signal
import sys
import time
from pathlib import Path

import torch
import torch_musa  # noqa: F401
from torch.optim.optimizer import register_optimizer_step_post_hook

T_START = time.perf_counter()


def compile_count() -> int:
    utils = sys.modules.get("torch._dynamo.utils")
    if utils is None:
        return 0
    return int(utils.counters["stats"]["unique_graphs"] + utils.counters["aot_autograd"]["total"])


def rng_fingerprint() -> str:
    h = hashlib.sha256(torch.get_rng_state().numpy().tobytes())
    if hasattr(torch, "musa") and torch.musa.is_available() and torch.musa.is_initialized():
        h.update(torch.musa.get_rng_state().numpy().tobytes())
    return h.hexdigest()[:16]


def kill_descendants() -> int:
    children = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(d))
    stack, found = [os.getpid()], []
    while stack:
        for c in children.get(stack.pop(), []):
            found.append(c)
            stack.append(c)
    for pid in found:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    return len(found)


class MusaTorchProfiler:
    """Torch-MUSA profiler (using PrivateUse1/MUSA activity) with lightweight module ranges."""

    def __init__(self, out_prefix: Path):
        self.out_prefix = out_prefix
        self.prof = None
        self.handles = []

    def _attach_hooks(self):
        main_mod = sys.modules.get("__main__")
        # Find Trainer/model from gc if not a top-level global
        model = None
        if main_mod is not None and hasattr(main_mod, "trainer"):
            model = getattr(main_mod.trainer, "model", None)
        if model is None:
            import gc
            for obj in gc.get_objects():
                if isinstance(obj, torch.nn.Module) and hasattr(obj, "blocks") and hasattr(obj, "patch_embedding"):
                    model = obj
                    break
        if model is None:
            return
        targets = [
            (model.patch_embedding, "patch_embedding"),
            (model.text_embedding, "text_embedding"),
            (model.time_embedding, "time_embedding"),
            (model.head, "head"),
        ]
        for i, blk in enumerate(model.blocks):
            if i == 0:
                targets.append((blk, "block0"))
                if hasattr(blk, "self_attn"):
                    targets.append((blk.self_attn, "self_attn"))
                if hasattr(blk, "cross_attn"):
                    targets.append((blk.cross_attn, "cross_attn"))
                if hasattr(blk, "ffn"):
                    targets.append((blk.ffn, "ffn"))
            else:
                if hasattr(blk, "self_attn"):
                    targets.append((blk.self_attn, "self_attn"))
                if hasattr(blk, "cross_attn"):
                    targets.append((blk.cross_attn, "cross_attn"))
                if hasattr(blk, "ffn"):
                    targets.append((blk.ffn, "ffn"))

        for mod, label in targets:
            self._wrap(mod, label)

    def _wrap(self, mod: torch.nn.Module, label: str):
        stack = []

        def pre(_m, _a):
            r = torch.autograd.profiler.record_function(f"## {label}")
            r.__enter__()
            stack.append(r)

        def post(_m, _a, _o):
            if stack:
                stack.pop().__exit__(None, None, None)

        self.handles.append(mod.register_forward_pre_hook(pre))
        self.handles.append(mod.register_forward_hook(post))

    def start(self):
        rank = int(os.environ.get("RANK", "0"))
        if rank != 0:
            return
        self._attach_hooks()
        from torch.profiler import ProfilerActivity, profile

        dev_act = getattr(ProfilerActivity, "MUSA", ProfilerActivity.PrivateUse1)
        self.prof = profile(
            activities=[ProfilerActivity.CPU, dev_act],
            record_shapes=False,
            with_stack=False,
            profile_memory=False,
        )
        self.prof.__enter__()

    def stop(self, span_s: float, iterations: int):
        rank = int(os.environ.get("RANK", "0"))
        if rank != 0 or self.prof is None:
            return
        torch.musa.synchronize()
        self.prof.__exit__(None, None, None)
        for h in self.handles:
            h.remove()
        self.handles.clear()

        # Export compressed Chrome trace for kernel_util.py
        raw_trace = Path(f"{self.out_prefix}.trace.r0.json")
        gz_trace = Path(f"{self.out_prefix}.trace.r0.json.gz")
        self.prof.export_chrome_trace(str(raw_trace))
        with raw_trace.open("rb") as fi, gzip.open(gz_trace, "wb") as fo:
            while True:
                chunk = fi.read(4 << 20)
                if not chunk:
                    break
                fo.write(chunk)
        raw_trace.unlink(missing_ok=True)

        # Aggregate summary from key_averages()
        evts = self.prof.key_averages()
        kernels = []
        annotations = {}
        total_dev_us = 0.0
        for e in evts:
            dev_us = float(getattr(e, "self_device_time_total", 0.0) or 0.0)
            tot_dev_us = float(getattr(e, "device_time_total", 0.0) or 0.0)
            cpu_us = float(getattr(e, "cpu_time_total", 0.0) or 0.0)
            if e.key.startswith("## "):
                annotations[e.key[3:]] = {
                    "device_ms": round(tot_dev_us / 1e3, 3),
                    "cpu_ms": round(cpu_us / 1e3, 3),
                    "calls": int(e.count),
                }
            elif dev_us > 0:
                total_dev_us += dev_us
                kernels.append((dev_us, e.key, int(e.count), float(getattr(e, "self_cpu_time_total", 0.0) or 0.0)))

        kernels.sort(key=lambda x: -x[0])
        summary = {
            "span_s": round(span_s, 4),
            "iterations": iterations,
            "kernel_time_s": round(total_dev_us / 1e6, 4),
            "gpu_busy_ratio_approx": round((total_dev_us / 1e6) / span_s, 4) if span_s > 0 else None,
            "annotations": annotations,
            "kernels": [
                {
                    "name": k[1][:300],
                    "total_ms": round(k[0] / 1e3, 3),
                    "calls": k[2],
                    "share_of_kernel_time": round(k[0] / total_dev_us, 4) if total_dev_us > 0 else None,
                    "self_cpu_ms": round(k[3] / 1e3, 3),
                }
                for k in kernels[:100]
            ],
            "trace_gz": str(gz_trace),
        }
        Path(f"{self.out_prefix}.profile.r0.json").write_text(json.dumps(summary, indent=1) + "\n")


def main():
    if "--" not in sys.argv:
        raise SystemExit("usage: timing.py --out PREFIX [options] -- <train.py args>")
    sep = sys.argv.index("--")
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--profile", choices=("none", "torch"), default="none")
    p.add_argument("--profile-iters", type=int, default=1)
    p.add_argument("--entry", default="train.py")
    a = p.parse_args(sys.argv[1:sep])
    train_argv = sys.argv[sep + 1 :]

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    out_prefix = Path(a.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "iter": 0,
        "warmup_end_t": None,
        "win_start_t": None,
        "window_s": [],
        "rng": [],
        "compiles_before": None,
        "compiles_after": None,
        "prof": MusaTorchProfiler(out_prefix) if a.profile == "torch" else None,
        "prof_start_t": None,
    }
    total_iters = (
        a.warmup + a.profile_iters if a.profile == "torch" else a.warmup + a.window * a.repeats
    )

    def _finish_and_exit():
        torch.musa.synchronize()
        peak_alloc = torch.musa.max_memory_allocated() / (1024 ** 3)
        peak_resv = torch.musa.max_memory_reserved() / (1024 ** 3)
        payload = {
            "rank": rank,
            "world_size": world,
            "warmup": a.warmup,
            "window": a.window,
            "repeats": a.repeats,
            "iterations_timed": a.window * a.repeats if a.profile == "none" else a.profile_iters,
            "startup_s": round((state["warmup_end_t"] or time.perf_counter()) - T_START, 4),
            "window_s": [round(x, 6) for x in state["window_s"]],
            "total_s": round(sum(state["window_s"]), 6),
            "peak_memory_allocated_gib": round(peak_alloc, 3),
            "peak_memory_reserved_gib": round(peak_resv, 3),
            "compiles_in_window": (
                (state["compiles_after"] or 0) - (state["compiles_before"] or 0)
                if state["compiles_before"] is not None
                else 0
            ),
            "rng": state["rng"],
            "device_name": torch.musa.get_device_name(0) if torch.musa.is_available() else "unknown",
        }
        Path(f"{out_prefix}.r{rank}.json").write_text(json.dumps(payload, indent=1) + "\n")
        if torch.distributed.is_initialized():
            try:
                torch.distributed.barrier()
            except Exception:
                pass
        kill_descendants()
        os._exit(0)

    def on_step_post(_opt, _args, _kwargs):
        state["iter"] += 1
        it = state["iter"]
        state["rng"].append(rng_fingerprint())

        if a.profile == "torch":
            if it == a.warmup:
                torch.musa.synchronize()
                state["warmup_end_t"] = time.perf_counter()
                state["prof_start_t"] = time.perf_counter()
                state["prof"].start()
            elif it == a.warmup + a.profile_iters:
                torch.musa.synchronize()
                span_s = time.perf_counter() - state["prof_start_t"]
                state["window_s"].append(span_s)
                state["prof"].stop(span_s, a.profile_iters)
                _finish_and_exit()
            return

        if it == a.warmup:
            torch.musa.synchronize()
            torch.musa.reset_peak_memory_stats()
            state["warmup_end_t"] = time.perf_counter()
            state["win_start_t"] = state["warmup_end_t"]
            state["compiles_before"] = compile_count()
        elif it > a.warmup and (it - a.warmup) % a.window == 0:
            torch.musa.synchronize()
            now = time.perf_counter()
            state["window_s"].append(now - state["win_start_t"])
            state["win_start_t"] = now
            if it >= total_iters:
                state["compiles_after"] = compile_count()
                _finish_and_exit()

    register_optimizer_step_post_hook(on_step_post)
    sys.argv = [a.entry] + train_argv
    runpy.run_path(a.entry, run_name="__main__")


if __name__ == "__main__":
    main()
