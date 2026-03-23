import os
import math
from pathlib import Path
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
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

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


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

    # Optional TF32 acceleration on NVIDIA Ampere/Ada GPUs.
    if device.type == "cuda" and bool(cfg["train"].get("allow_tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    amp_cfg = cfg["train"].get("amp", {"enabled": False, "dtype": "bf16"})
    if isinstance(amp_cfg, dict):
        amp_enabled = bool(amp_cfg.get("enabled", False)) and device.type == "cuda"
        amp_dtype_name = str(amp_cfg.get("dtype", "bf16")).lower()
    else:
        amp_enabled = bool(amp_cfg) and device.type == "cuda"
        amp_dtype_name = "bf16"

    if amp_dtype_name not in ("bf16", "fp16"):
        raise ValueError(f"Unsupported amp.dtype={amp_dtype_name}. Use 'bf16' or 'fp16'.")
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bf16" else torch.float16
    use_grad_scaler = amp_enabled and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_grad_scaler else None

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
    opt = AdamW(G.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    # cosine scheduler as in implementation details 
    total_steps = int(cfg["train"]["epochs"]) * len(dl)
    sched = CosineAnnealingLR(opt, T_max=total_steps)

    # CLIP-normalized bounds corresponding to pixel-space [0, 1].
    clip_mean = torch.tensor(CLIP_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    clip_std = torch.tensor(CLIP_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    valid_min = (0.0 - clip_mean) / clip_std
    valid_max = (1.0 - clip_mean) / clip_std

    # Checkpointing params
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

            opt.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                # Generate bounded perturbation delta_u, add to image 
                delta = G(emb_t, z)
                images_poison = torch.max(torch.min(images + delta, valid_max), valid_min)

                # CLIP image embedding (grad flows w.r.t images_poison => delta => G)
                img_emb = clip_model.encode_image(images_poison)
                logit_scale = clip_model.model.logit_scale.exp().detach()

                # InfoNCE (symmetric) Eq.(3), applied to protected data as Eq.(4) 
                loss = symmetric_infonce(img_emb, emb_t, logit_scale=logit_scale)

            if use_grad_scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
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
