"""Pre-encoded Wan2.2-TI2V-5B latent + UMT5 text embedding dataset and DataLoader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class PreEncodedWanDataset(Dataset):
    """Loads pre-encoded Wan2.2 VAE latents [48, T, H, W] and UMT5 text embeds [512, 4096]."""

    def __init__(
        self,
        data_dir: str | Path,
        num_latent_t: Optional[int] = None,
        latent_height: Optional[int] = None,
        latent_width: Optional[int] = None,
        preload: bool = False,
    ):
        self.data_dir = Path(data_dir)
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        self.entries: List[Dict[str, Any]] = manifest["samples"]
        if not self.entries:
            raise ValueError(f"Empty dataset manifest: {manifest_path}")
        self.num_latent_t = num_latent_t
        self.latent_height = latent_height
        self.latent_width = latent_width
        self._cache: Optional[List[Dict[str, torch.Tensor]]] = None
        if preload:
            self._cache = [self._load_item(i) for i in range(len(self.entries))]

    def _load_item(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.entries[idx]
        p = self.data_dir / entry["file"]
        obj = torch.load(p, map_location="cpu", weights_only=True)
        latents = obj["latents"]  # [48, T, H, W]
        text_embeds = obj["text_embeds"]  # [512, 4096]
        if self.num_latent_t is not None:
            latents = latents[:, : self.num_latent_t]
        if self.latent_height is not None:
            latents = latents[:, :, : self.latent_height]
        if self.latent_width is not None:
            latents = latents[:, :, :, : self.latent_width]
        return {
            "latents": latents.contiguous(),
            "text_embeds": text_embeds.contiguous(),
            "sample_id": torch.tensor(idx, dtype=torch.int64),
        }

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._cache is not None:
            return self._cache[idx]
        return self._load_item(idx)


class InfiniteDistributedSampler(Sampler[int]):
    """Deterministic distributed sampler that cycles indefinitely across epochs."""

    def __init__(
        self,
        dataset_size: int,
        rank: int,
        world_size: int,
        seed: int = 42,
        total_samples_per_rank: int = 100000,
    ):
        self.dataset_size = dataset_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.total_samples_per_rank = total_samples_per_rank

    def __iter__(self):
        epoch = 0
        yielded = 0
        while yielded < self.total_samples_per_rank:
            g = torch.Generator()
            g.manual_seed(self.seed + epoch * 10007)
            perm = torch.randperm(self.dataset_size, generator=g).tolist()
            # Pad to multiple of world_size so every rank gets equal steps
            rem = len(perm) % self.world_size
            if rem != 0:
                perm.extend(perm[: self.world_size - rem])
            rank_indices = perm[self.rank :: self.world_size]
            for idx in rank_indices:
                yield idx
                yielded += 1
                if yielded >= self.total_samples_per_rank:
                    break
            epoch += 1

    def __len__(self) -> int:
        return self.total_samples_per_rank


def build_dataloader(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
    seed: int,
    num_latent_t: Optional[int] = None,
    latent_height: Optional[int] = None,
    latent_width: Optional[int] = None,
    preload: bool = False,
) -> DataLoader:
    dataset = PreEncodedWanDataset(
        data_dir=data_dir,
        num_latent_t=num_latent_t,
        latent_height=latent_height,
        latent_width=latent_width,
        preload=preload,
    )
    sampler = InfiniteDistributedSampler(
        dataset_size=len(dataset),
        rank=rank,
        world_size=world_size,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
