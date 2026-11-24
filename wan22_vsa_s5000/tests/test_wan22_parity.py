"""Verify Wan22VSAModel loads Wan2.2-TI2V-5B and matches DiffSynth WanModel at sparsity=0."""

import json
import sys
import types
from pathlib import Path

import torch
import torch_musa  # noqa: F401
from einops import rearrange
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if "imageio" not in sys.modules:
    _im = types.ModuleType("imageio")
    _im.__path__ = []
    sys.modules["imageio"] = _im
    sys.modules["imageio.v3"] = types.ModuleType("imageio.v3")
import os
if os.environ.get("DIFFSYNTH_PATH"):
    sys.path.insert(0, os.environ["DIFFSYNTH_PATH"])

from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from wan22_vsa.model import load_wan22_ti2v_5b_vsa

MODEL_DIR = Path(os.environ.get("WAN22_MODEL_DIR", "./assets/models/Wan2.2-TI2V-5B"))


def load_diffsynth_wan22(num_layers=4):
    m = WanModel(
        dim=3072,
        in_dim=48,
        ffn_dim=14336,
        out_dim=48,
        text_dim=4096,
        freq_dim=256,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=24,
        num_layers=num_layers,
        has_image_input=False,
    )
    idx = json.loads((MODEL_DIR / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    sd, by_shard = {}, {}
    for k, fname in idx.items():
        if k.startswith("blocks.") and int(k.split(".")[1]) >= num_layers:
            continue
        by_shard.setdefault(fname, []).append(k)
    for fname, keys in sorted(by_shard.items()):
        with safe_open(str(MODEL_DIR / fname), framework="pt", device="cpu") as sf:
            for k in keys:
                sd[k] = sf.get_tensor(k)
    m.load_state_dict(sd, strict=True)
    return m


def diffsynth_forward(dit, x, timestep, context):
    """Exact DiffSynth model_fn_wan_video forward path."""
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).to(x.dtype))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    ctx = dit.text_embedding(context)
    x = dit.patchify(x)
    f, h, w = x.shape[2:]
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    for blk in dit.blocks:
        x = blk(x, ctx, t_mod, freqs)
    x = dit.head(x, t)
    return dit.unpatchify(x, (f, h, w))


def main():
    dev = torch.device("musa:0")
    vsa_m = load_wan22_ti2v_5b_vsa(MODEL_DIR, dtype=torch.bfloat16, num_layers_override=4).to(dev).eval()
    ref_m = load_diffsynth_wan22(num_layers=4).to(device=dev, dtype=torch.bfloat16).eval()

    torch.manual_seed(7)
    x = torch.randn(1, 48, 5, 16, 20, device=dev, dtype=torch.bfloat16)
    t = torch.tensor([500.0], device=dev, dtype=torch.bfloat16)
    ctx = torch.randn(1, 512, 4096, device=dev, dtype=torch.bfloat16) * 0.1

    with torch.no_grad():
        y_vsa_dense = vsa_m(x, t, ctx, vsa_sparsity=0.0)
        y_ref = diffsynth_forward(ref_m, x, t, ctx)

    rel_l2 = (y_vsa_dense.float() - y_ref.float()).norm().item() / y_ref.float().norm().item()
    cos = torch.nn.functional.cosine_similarity(
        y_vsa_dense.float().reshape(1, -1), y_ref.float().reshape(1, -1)
    ).item()
    print(f"[4-layer Wan2.2 parity vs DiffSynth] rel_l2={rel_l2:.6f} cos={cos:.6f}")
    assert cos > 0.999 and rel_l2 < 3e-2, f"Wan2.2 parity failed: rel_l2={rel_l2}, cos={cos}"

    # Sparse forward + backward on 4 layers
    vsa_m.train()
    y_sparse = vsa_m(x, t, ctx, vsa_sparsity=0.8)
    loss = y_sparse.float().pow(2).mean()
    loss.backward()
    gate_grad = vsa_m.blocks[0].self_attn.to_gate_compress.weight.grad
    assert gate_grad is not None and gate_grad.float().norm().item() > 0, "to_gate_compress grad missing!"
    print(f"[4-layer VSA sparse=0.8 fwd+bwd] loss={loss.item():.6f} blk0.gate_grad_norm={gate_grad.float().norm().item():.6f}")
    print("ALL WAN2.2-TI2V-5B PARITY & VSA GRADIENT TESTS PASSED!")


if __name__ == "__main__":
    main()
