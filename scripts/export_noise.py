import os
from pathlib import Path
from typing import List

import torch
import numpy as np

from t2ue.models.clip_surrogate import OpenAIClipSurrogate
from t2ue.models.generator import T2UEGenerator, GenConfig
from t2ue.utils.checkpoint import load_checkpoint
from t2ue.utils.seed import seed_all

def read_prompts(path: str) -> List[str]:
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]

@torch.no_grad()
def main(ckpt: str, prompts_path: str, out_dir: str, model_name: str, seed: int = 123):
    """
    Zero-Contact generation stage
    Given a trained generator G checkpoint, and a list of text prompts,
    generate and save CLIP-normalized perturbations delta_u for each prompt.
    """
    
    
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained generator G
    payload = load_checkpoint(ckpt, map_location="cpu")
    gen_cfg = GenConfig(**payload["gen_cfg"])
    G = T2UEGenerator(gen_cfg).to(device)
    G.load_state_dict(payload["state_dict"], strict=True)
    G.eval()

    ## Load frozen CLIP surrogate to compute text embeddings
    clip_model = OpenAIClipSurrogate(model_name, device=device).to(device)

    prompts = read_prompts(prompts_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # TODO: Determinism for class wise noise generation 
    # Script currently seeds globally at start
    # fix z seed per class prompt or uniqye seeds per sample
    for i, text in enumerate(prompts):
        emb_t = clip_model.encode_text([text])  # (1,D)
        z = torch.randn(1, gen_cfg.z_dim, device=device) 
        delta = G(emb_t, z).cpu().numpy()[0]  # (3,H,W)

        # Save as .npy in CLIP-normalized tensor space
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
