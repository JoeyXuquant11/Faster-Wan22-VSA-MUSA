"""Prepare a public, reproducible pre-encoded Wan2.2-TI2V-5B dataset using the REAL
Wan2.2 VAE (Wan2.2_VAE.pth) and REAL UMT5-XXL encoder (models_t5_umt5-xxl-enc-bf16.pth).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import types
from pathlib import Path

import torch
import torch_musa  # noqa: F401
from transformers import AutoTokenizer

if "imageio" not in sys.modules:
    _im = types.ModuleType("imageio"); _im.__path__ = []; sys.modules["imageio"] = _im; sys.modules["imageio.v3"] = types.ModuleType("imageio.v3")
if os.environ.get("DIFFSYNTH_PATH"):
    sys.path.insert(0, os.environ["DIFFSYNTH_PATH"])
from diffsynth.models.wan_video_text_encoder import WanTextEncoder
from diffsynth.models.wan_video_vae import WanVideoVAE38


PROMPTS = [
    "a bright red sphere drifting smoothly across a deep blue ocean horizon at sunset",
    "golden sunlight rippling over calm turquoise water with gentle camera pan",
    "neon cyan and magenta light waves pulsing rhythmically in a dark geometric tunnel",
    "a rotating emerald crystal cube floating above a misty silver landscape",
    "soft orange autumn leaves gliding diagonally across a warm morning sky",
    "concentric white ripples expanding across a dark obsidian pond in slow motion",
    "a glowing amber orb tracing a figure eight path through indigo twilight clouds",
    "vibrant green northern lights shimmering vertically across a starry arctic night",
    "parallel crimson and gold laser beams sweeping smoothly across a matte black stage",
    "a metallic silver ring spinning steadily in front of a soft pastel gradient",
    "warm candlelight flickering gently over textured wooden waves in close-up view",
    "diagonal bands of sapphire and violet light drifting slowly from left to right",
    "a luminous white comet streaking across a deep navy sky with soft glowing trail",
    "pastel pink and yellow clouds morphing gently during a serene sunrise timelapse",
    "a checkerboard of warm terracotta and ivory tiles sliding smoothly in perspective",
    "electric blue plasma arcs dancing rhythmically across a dark studio backdrop",
    "soft emerald and teal foliage swaying calmly in an afternoon breeze",
    "a floating coral torus bobbing gently over reflective lavender water",
    "crisp white snowflakes drifting diagonally past a warm golden lantern glow",
    "smooth metallic copper waves undulating steadily under directional studio lighting",
    "a bright yellow sun rising slowly behind layered purple mountain silhouettes",
    "interlocking geometric triangles shifting colors between cyan, coral, and gold",
    "gentle ocean foam washing across smooth dark pebbles in steady slow motion",
    "glowing violet fireflies drifting along curved trajectories in a midnight forest",
    "horizontal ribbons of ruby and ivory silk fluttering softly in a steady wind",
    "a translucent glass prism casting slowly rotating rainbow bands across white stone",
    "deep crimson magma veins pulsing slowly beneath dark volcanic rock textures",
    "calm silver mist drifting smoothly over a minimalist turquoise geometric grid",
    "a bright mint satellite orbiting around a shadowed cobalt planet in deep space",
    "warm honey colored light rays sweeping slowly through soft atmospheric haze",
    "layered indigo and aquamarine gradients breathing smoothly in a tranquil loop",
    "a crisp golden compass needle rotating smoothly over a dark slate dial",
]


def generate_procedural_video(
    idx: int,
    num_frames: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate a deterministic, richly structured RGB video [3, T, H, W] in [-1, 1]."""
    g = torch.Generator(device="cpu").manual_seed(202600 + idx * 997)
    freqs_x = torch.rand(3, generator=g).to(device) * 4.0 + 1.0
    freqs_y = torch.rand(3, generator=g).to(device) * 4.0 + 1.0
    speeds = (torch.rand(3, generator=g).to(device) - 0.5) * 6.0
    phases = torch.rand(3, generator=g).to(device) * 2.0 * math.pi
    cx0 = (torch.rand(1, generator=g).item() - 0.5) * 0.8
    cy0 = (torch.rand(1, generator=g).item() - 0.5) * 0.8
    vx = (torch.rand(1, generator=g).item() - 0.5) * 1.2
    vy = (torch.rand(1, generator=g).item() - 0.5) * 1.2
    rad = 0.15 + torch.rand(1, generator=g).item() * 0.25
    blob_col = (torch.rand(3, generator=g).to(device) * 1.6 - 0.8).view(3, 1, 1, 1)

    t = torch.linspace(0.0, 1.0, num_frames, device=device, dtype=torch.float32)
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32)

    # Background traveling 2D wave per RGB channel
    chans = []
    for c in range(3):
        ang = (
            freqs_x[c] * math.pi * x.view(1, 1, width)
            + freqs_y[c] * math.pi * y.view(1, height, 1)
            + speeds[c] * math.pi * t.view(num_frames, 1, 1)
            + phases[c]
        )
        wave = 0.55 * torch.sin(ang) + 0.25 * torch.cos(0.5 * ang + phases[(c + 1) % 3])
        chans.append(wave)
    vid = torch.stack(chans, dim=0)  # [3, T, H, W]

    # Moving foreground glowing Gaussian blob for strong temporal dynamics
    cx = (cx0 + vx * t).view(num_frames, 1, 1)
    cy = (cy0 + vy * t).view(num_frames, 1, 1)
    dist2 = (x.view(1, 1, width) - cx) ** 2 + (y.view(1, height, 1) - cy) ** 2
    blob = torch.exp(-dist2 / (rad * rad)).unsqueeze(0)  # [1, T, H, W]
    vid = (vid * (1.0 - 0.7 * blob) + blob_col * blob).clamp(-1.0, 1.0)
    return vid


def encode_texts(model_dir: Path, prompts: list[str], device: torch.device) -> list[torch.Tensor]:
    print(f"[prepare_dataset] Loading UMT5-XXL tokenizer & encoder from {model_dir}...")
    tok = AutoTokenizer.from_pretrained(str(model_dir / "google/umt5-xxl"), use_fast=True)
    enc = WanTextEncoder()
    sd = torch.load(str(model_dir / "models_t5_umt5-xxl-enc-bf16.pth"), map_location="cpu", weights_only=True)
    enc.load_state_dict(sd, strict=True)
    enc = enc.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)

    embeds = []
    with torch.no_grad():
        for i in range(0, len(prompts), 4):
            batch_prompts = prompts[i : i + 4]
            toks = tok(
                batch_prompts,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            ids = toks.input_ids.to(device)
            mask = toks.attention_mask.to(device)
            out = enc(ids, mask)  # [b, 512, 4096]
            seq_lens = mask.gt(0).sum(dim=1).tolist()
            for j, sl in enumerate(seq_lens):
                emb = out[j].clone()
                emb[sl:] = 0
                embeds.append(emb.cpu().to(torch.bfloat16))
    del enc, sd
    torch.musa.empty_cache()
    print(f"[prepare_dataset] Encoded {len(embeds)} prompts -> shape {tuple(embeds[0].shape)}")
    return embeds


def encode_videos(
    model_dir: Path,
    num_samples: int,
    num_frames: int,
    height: int,
    width: int,
    device: torch.device,
) -> list[torch.Tensor]:
    print(f"[prepare_dataset] Loading Wan2.2 VAE38 from {model_dir / 'Wan2.2_VAE.pth'}...")
    vae = WanVideoVAE38()
    sd = torch.load(str(model_dir / "Wan2.2_VAE.pth"), map_location="cpu", weights_only=True)
    # Wan2.2_VAE.pth keys start with "encoder." / "conv1." etc. -> goes into vae.model
    if any(k.startswith("model.") for k in sd.keys()):
        vae.load_state_dict(sd, strict=True)
    else:
        vae.model.load_state_dict(sd, strict=True)
    vae = vae.to(device=device, dtype=torch.float32).eval().requires_grad_(False)

    latents_list = []
    with torch.no_grad():
        for idx in range(num_samples):
            vid = generate_procedural_video(idx, num_frames, height, width, device=device)
            # Encode directly via vae.model.encode([1, 3, T, H, W], vae.scale)
            lat = vae.model.encode(vid.unsqueeze(0), vae.scale).squeeze(0)  # [48, T_lat, H_lat, W_lat]
            latents_list.append(lat.cpu().to(torch.bfloat16))
            if (idx + 1) % 8 == 0 or idx == num_samples - 1:
                print(
                    f"  encoded video {idx + 1}/{num_samples}: "
                    f"shape={tuple(lat.shape)} mean={lat.mean().item():.4f} std={lat.std().item():.4f}"
                )
    del vae, sd
    torch.musa.empty_cache()
    return latents_list


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, default=os.environ.get("WAN22_MODEL_DIR", "./assets/models/Wan2.2-TI2V-5B"))
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=32)
    p.add_argument("--num_frames", type=int, default=121)
    p.add_argument("--height", type=int, default=704)
    p.add_argument("--width", type=int, default=1280)
    args = p.parse_args()

    dev = torch.device("musa:0")
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.num_samples)]
    text_embeds = encode_texts(model_dir, prompts, dev)
    latents_list = encode_videos(model_dir, args.num_samples, args.num_frames, args.height, args.width, dev)

    samples_meta = []
    for idx in range(args.num_samples):
        fname = f"sample_{idx:05d}.pt"
        fpath = out_dir / fname
        payload = {
            "latents": latents_list[idx],
            "text_embeds": text_embeds[idx],
            "prompt": prompts[idx],
        }
        torch.save(payload, fpath)
        sha = hashlib.sha256(fpath.read_bytes()).hexdigest()[:16]
        samples_meta.append(
            {
                "id": idx,
                "file": fname,
                "latents_shape": list(latents_list[idx].shape),
                "text_shape": list(text_embeds[idx].shape),
                "sha256_16": sha,
            }
        )

    manifest = {
        "model_dir": str(model_dir),
        "num_samples": args.num_samples,
        "num_frames": args.num_frames,
        "height": args.height,
        "width": args.width,
        "samples": samples_meta,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[prepare_dataset] Wrote {args.num_samples} samples to {out_dir}")


if __name__ == "__main__":
    main()
