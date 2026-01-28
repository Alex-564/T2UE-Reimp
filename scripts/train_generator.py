import os
import math
from pathlib import Path
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from t2ue.utils.misc import load_yaml
from t2ue.utils.seed import seed_all
from t2ue.utils.checkpoint import save_checkpoint
from t2ue.utils.meters import AvgMeter

from t2ue.data.transforms import build_clip_image_transform
from t2ue.data.coco import CocoCaptionPairs

from t2ue.models.clip_surrogate import OpenAIClipSurrogate
from t2ue.models.generator import T2UEGenerator, GenConfig
from t2ue.losses.infonce import symmetric_infonce


def collate_fn(batch):
    images, caps = zip(*batch)
    images = torch.stack(images, dim=0)
    caps = list(caps)
    return images, caps


def main(cfg_path: str):
    """
    Train the T2UE generator G to minimize the InfoNCE loss computed
    by frozen CLIP surrogate (f_I, f_T) on poisoned data.
    
    """
    cfg: Dict[str, Any] = load_yaml(cfg_path)
    seed_all(int(cfg["seed"]))

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data loading
    # Uses COCO Captioneds dataset as per paper specificaiton
    # Explicit CLIP specific expected preprocessing
    tfm = build_clip_image_transform(out_res=int(cfg["gen"]["out_res"]))
    ds = CocoCaptionPairs(root=cfg["coco"]["root"], annFile=cfg["coco"]["ann"], transform=tfm)
    dl = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # Frozen CLIP surrogate (f_I, f_T) 
    clip_model = OpenAIClipSurrogate(cfg["clip"]["model_name"], device=device).to(device)

    # Generator G (learnable) 
    gen_cfg = GenConfig(
        z_dim=int(cfg["gen"]["z_dim"]),
        text_dim=int(cfg["gen"]["text_dim"]),
        base_ch=int(cfg["gen"]["base_ch"]),
        out_res=int(cfg["gen"]["out_res"]),
        eps=float(cfg["gen"]["eps"]),
    )
    G = T2UEGenerator(gen_cfg).to(device)
    G.train()

    # Paper defined optimizer + scheduler:
    opt = Adam(G.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    # cosine scheduler as in implementation details 
    total_steps = int(cfg["train"]["epochs"]) * len(dl)
    sched = CosineAnnealingLR(opt, T_max=total_steps)

    # Checkpointing params
    tau = float(cfg["train"]["tau"])
    log_every = int(cfg["train"]["log_every"])
    save_every_epochs = int(cfg["train"]["save_every_epochs"])

    global_step = 0
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        meter = AvgMeter()
        pbar = tqdm(dl, desc=f"epoch {epoch}/{cfg['train']['epochs']}")

        for images, caps in pbar:
            images = images.to(device, non_blocking=True)

            # Text features from frozen encoder (CLIP text encoder) 
            with torch.no_grad():
                emb_t = clip_model.encode_text(caps)  # (B,D)

            # Random latent z ~ N(0, I) 
            z = torch.randn(images.shape[0], gen_cfg.z_dim, device=device)

            # Generate bounded perturbation delta_u, add to image 
            delta = G(emb_t, z)
            images_poison = torch.clamp(images + delta, min=-10.0, max=10.0)
            # Note: images are already CLIP-normalized; clamp here is mainly for numerical sanity.

            # CLIP image embedding (grad flows w.r.t images_poison => delta => G)
            img_emb = clip_model.encode_image(images_poison)

            # InfoNCE (symmetric) Eq.(3), applied to protected data as Eq.(4) 
            loss = symmetric_infonce(img_emb, emb_t, tau=tau)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

            meter.update(loss.item(), n=images.shape[0])
            global_step += 1

            if global_step % log_every == 0:
                pbar.set_postfix(loss=f"{meter.avg:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")

        if epoch % save_every_epochs == 0 or epoch == int(cfg["train"]["epochs"]):
            ckpt_path = out_dir / f"generator_epoch{epoch:04d}.pt"
            save_checkpoint(
                str(ckpt_path),
                {
                    "epoch": epoch,
                    "gen_cfg": gen_cfg.__dict__,
                    "state_dict": G.state_dict(),
                    "optim": opt.state_dict(),
                    "cfg_path": cfg_path,
                },
            )

    print(f"Done. Checkpoints in: {out_dir}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    args = ap.parse_args()
    main(args.config)
