"""Frozen training recipe for Wan2.2-TI2V-5B + VSA full finetuning on 8x MTT S5000."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

ASSETS = Path(os.environ.get("ASSETS", "./assets"))
MODEL_DIR = ASSETS / "models/Wan2.2-TI2V-5B"
DATA_DIR = ASSETS / "data/wan22_vsa_data_704x1280_121f"

RECIPE = {
    "num_latent_t": 31,
    "latent_height": 44,
    "latent_width": 80,
    "batch_size": 1,
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-5,
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-8,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "flow_shift": 5.0,
    "num_train_timesteps": 1000,
    "vsa_sparsity": 0.8,
    "vsa_decay_rate": 0.0,
    "vsa_decay_interval_steps": 50,
    "vsa_q_chunk_blocks": 32,
    "num_workers": 2,
    "seed": 42,
}


def build_argv(
    max_train_steps: int,
    output_dir: str | Path,
    seed: Optional[int] = None,
    learning_rate: Optional[float] = None,
    extra: Optional[List[str]] = None,
) -> List[str]:
    lr = RECIPE["learning_rate"] if learning_rate is None else learning_rate
    sd = RECIPE["seed"] if seed is None else seed
    argv = [
        "--model_dir", str(MODEL_DIR),
        "--data_dir", str(DATA_DIR),
        "--output_dir", str(output_dir),
        "--num_latent_t", str(RECIPE["num_latent_t"]),
        "--latent_height", str(RECIPE["latent_height"]),
        "--latent_width", str(RECIPE["latent_width"]),
        "--batch_size", str(RECIPE["batch_size"]),
        "--gradient_accumulation_steps", str(RECIPE["gradient_accumulation_steps"]),
        "--max_train_steps", str(max_train_steps),
        "--learning_rate", repr(float(lr)),
        "--beta1", repr(RECIPE["beta1"]),
        "--beta2", repr(RECIPE["beta2"]),
        "--eps", repr(RECIPE["eps"]),
        "--weight_decay", repr(RECIPE["weight_decay"]),
        "--max_grad_norm", repr(RECIPE["max_grad_norm"]),
        "--flow_shift", repr(RECIPE["flow_shift"]),
        "--num_train_timesteps", str(RECIPE["num_train_timesteps"]),
        "--vsa_sparsity", repr(RECIPE["vsa_sparsity"]),
        "--vsa_decay_rate", repr(RECIPE["vsa_decay_rate"]),
        "--vsa_decay_interval_steps", str(RECIPE["vsa_decay_interval_steps"]),
        "--vsa_q_chunk_blocks", str(RECIPE["vsa_q_chunk_blocks"]),
        "--num_workers", str(RECIPE["num_workers"]),
        "--seed", str(sd),
        "--log_steps", "1",
        "--save_steps", "0",
    ]
    if extra:
        argv.extend(extra)
    return argv
