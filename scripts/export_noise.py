import os
from pathlib import Path
from typing import List
import warnings

import torch
import torch.nn as nn
import numpy as np

from t2ue.models.clip_surrogate import OpenAIClipSurrogate
from t2ue.models.generator import T2UEGenerator, GenConfig
from t2ue.utils.checkpoint import load_checkpoint
from t2ue.utils.seed import seed_all

# text-to-unlearnable-examples t2ue zero-contact export stage
# generate delta_u from text prompts with frozen clip plus trained generator
def read_prompts(path: str) -> List[str]:
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]


# warn if bn stats look unstable before eval export
def warn_if_bn_stats_look_unreliable(model: nn.Module) -> None:
    bad_layers = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.BatchNorm2d):
            if mod.running_mean is None or mod.running_var is None:
                bad_layers.append(name)
                continue
            if (not torch.isfinite(mod.running_mean).all()) or (not torch.isfinite(mod.running_var).all()):
                bad_layers.append(name)
                continue
            if (mod.running_var <= 0).any():
                bad_layers.append(name)

    if bad_layers:
        warnings.warn(
            "Generator BatchNorm running stats look unreliable for eval-mode export. "
            "Verify training used sufficiently large/diverse batches before exporting noise.",
            stacklevel=2,
        )

@torch.no_grad()
def main(ckpt: str, prompts_path: str, out_dir: str, model_name: str, seed: int = 123):
    # seed controls z sampling and prompt order effects
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load trained t2ue generator G
    payload = load_checkpoint(ckpt, map_location="cpu")
    gen_cfg = GenConfig(**payload["gen_cfg"])
    G = T2UEGenerator(gen_cfg).to(device)
    G.load_state_dict(payload["state_dict"], strict=True)
    G.eval()
    warn_if_bn_stats_look_unreliable(G)

    # frozen clip text encoder as in t2ue
    clip_model = OpenAIClipSurrogate(model_name, device=device).to(device)

    prompts = read_prompts(prompts_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # one delta per prompt with z ~ N(0 I)
    for i, text in enumerate(prompts):
        emb_t = clip_model.encode_text([text])  # (1,D)
        z = torch.randn(1, gen_cfg.z_dim, device=device) 
        delta = G(emb_t, z).cpu().numpy()[0]  # (3,H,W)

        # save in clip-normalized tensor space
        np.save(out / f"delta_{i:05d}.npy", delta)
        with open(out / f"delta_{i:05d}.txt", "w") as f:
            f.write(text + "\n")

    print(f"Saved {len(prompts)} deltas to {out}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip_model", default="ViT-B/32")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    main(args.ckpt, args.prompts, args.out, args.clip_model, args.seed)
