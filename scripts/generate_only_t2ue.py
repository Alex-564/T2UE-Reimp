import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms

from t2ue.models.clip_surrogate import OpenAIClipSurrogate
from t2ue.models.generator import T2UEGenerator, GenConfig
from t2ue.utils.checkpoint import load_checkpoint

# hardcode CLIP mean/std.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

def load_checkpoint(path: str, map_location="cpu"):
    return torch.load(path, map_location=map_location)

def save_tensor_as_image(x: torch.Tensor, out_path: str, quality: int = 95):
    """
    x: [3,H,W] in [0,1]
    """
    x = x.detach().clamp(0, 1).cpu()
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(out_path, quality=quality)

def build_image_loader_tf(input_size: int, interpolation: str):
    interp_map = {
        "nearest": transforms.InterpolationMode.NEAREST,
        "bilinear": transforms.InterpolationMode.BILINEAR,
        "bicubic": transforms.InterpolationMode.BICUBIC,
        "lanczos": transforms.InterpolationMode.LANCZOS,
    }
    if interpolation not in interp_map:
        raise ValueError(f"Unknown interpolation={interpolation}. Choose from {list(interp_map.keys())}")

    return transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=interp_map[interpolation]),
        transforms.ToTensor(),  # [0,1]
    ])

def to_clip_normalized(pix: torch.Tensor) -> torch.Tensor:
    """
    pix: (B,3,H,W) in [0,1]
    return: CLIP-normalized
    """
    mean = torch.tensor(CLIP_MEAN, device=pix.device).view(1,3,1,1)
    std  = torch.tensor(CLIP_STD, device=pix.device).view(1,3,1,1)
    return (pix - mean) / std

def delta_norm_to_pixel(delta_norm: torch.Tensor) -> torch.Tensor:
    """
    Convert delta in CLIP-normalized space to pixel-space delta:
      x = (p - mean)/std
      x' = x + delta_norm  => p' = p + delta_norm * std
    So delta_pixel = delta_norm * std
    """
    std = torch.tensor(CLIP_STD, device=delta_norm.device).view(1,3,1,1)
    return delta_norm * std

def load_t2ue_generator(ckpt_path: str, device: torch.device):
    payload = load_checkpoint(ckpt_path, map_location="cpu")
    gen_cfg_dict = payload["gen_cfg"]

    cfg = GenConfig(**gen_cfg_dict)

    G = T2UEGenerator(cfg).to(device)
    G.load_state_dict(payload["state_dict"], strict=True)
    G.eval()
    return G, cfg

def load_openai_clip_surrogate(model_name: str, device: torch.device):
    clip_model = OpenAIClipSurrogate(model_name, device=device).to(device)
    clip_model.eval()
    return clip_model

@torch.no_grad()
def generate_class_deltas(
    class_manifest: Dict[str, Dict],
    G,
    gen_cfg,
    clip_model,
    device: torch.device,
    out_delta_dir: Path,
    force: bool = False,
):
    """
    Generates and saves one delta per class deterministically.
    Saves:
      out_delta_dir/delta_classXXXX.npy  (pixel-space delta, [3,H,W] float32)
      out_delta_dir/delta_classXXXX_meta.json  (prompt, seed)
    """
    out_delta_dir.mkdir(parents=True, exist_ok=True)

    for c_str, entry in tqdm(class_manifest.items(), desc="Generate class deltas"):
        c = int(entry["class_id"])
        prompt = entry["prompt"]
        z_seed = int(entry["z_seed"])

        delta_path = out_delta_dir / f"delta_class{c:05d}.npy"
        meta_path  = out_delta_dir / f"delta_class{c:05d}_meta.json"

        if delta_path.exists() and (not force):
            continue

        # Deterministic z from per-class seed
        g = torch.Generator(device=device)
        g.manual_seed(z_seed)
        z = torch.randn(1, gen_cfg.z_dim, generator=g, device=device)

        # Text embedding (frozen CLIP)
        emb_t = clip_model.encode_text([prompt])  # (1,D), normalized

        # delta in CLIP-normalized space from G (current repo behavior)
        delta_norm = G(emb_t, z)  # (1,3,H,W), bounded by eps in normalized units

        # Convert to pixel-space delta for saving/applying to real images
        delta_pix = delta_norm_to_pixel(delta_norm)  # (1,3,H,W)

        # Save
        np.save(delta_path, delta_pix[0].float().cpu().numpy().astype(np.float32))
        with open(meta_path, "w") as f:
            json.dump({"class_id": c, "prompt": prompt, "z_seed": z_seed}, f, indent=2)

def main():
    ap = argparse.ArgumentParser()

    # Inputs / outputs (matching CUDA style)
    ap.add_argument("--samples-csv", type=str, required=True)
    ap.add_argument("--class-manifest", type=str, required=True)
    ap.add_argument("--t2ue-ckpt", type=str, required=True)
    ap.add_argument("--clip-model", type=str, default="ViT-B/32")

    ap.add_argument("--out-images-dir", type=str, required=True)
    ap.add_argument("--out-poison-map", type=str, required=True)
    ap.add_argument("--out-metrics-json", type=str, default=None)

    # Cache
    ap.add_argument("--out-delta-dir", type=str, required=True, help="Where per-class deltas are cached")
    ap.add_argument("--force-regenerate-deltas", action="store_true")

    # Image handling
    ap.add_argument("--input-size", type=int, default=112, help="Resize images to this before applying poison")
    ap.add_argument("--interpolation", type=str, default="bilinear",
                    choices=["nearest", "bilinear", "bicubic", "lanczos"])
    ap.add_argument("--save-quality", type=int, default=100)

    # Determinism / performance 
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--num-workers", type=int, default=0, help="0 recommended for strict determinism")
    ap.add_argument("--batch-size", type=int, default=1, help="1 is simplest for deterministic ordering")
    ap.add_argument("--seed", type=int, default=0, help="Global seed for any residual randomness")

    args = ap.parse_args()

    # Paths
    os.makedirs(args.out_images_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_poison_map), exist_ok=True)
    if args.out_metrics_json is None:
        args.out_metrics_json = os.path.join(os.path.dirname(args.out_poison_map), "metrics.json")
    os.makedirs(os.path.dirname(args.out_metrics_json), exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Global determinism 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load manifests
    with open(args.class_manifest, "r") as f:
        class_manifest = json.load(f)

    samples: List[Tuple[str, int]] = []
    with open(args.samples_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            samples.append((row["clean_path"], int(row["label"])))
    if not samples:
        raise RuntimeError("No samples provided.")

    # Load generator + CLIP surrogate
    t0 = time.perf_counter()
    G, gen_cfg = load_t2ue_generator(args.t2ue_ckpt, device=device)
    clip_model = load_openai_clip_surrogate(args.clip_model, device=device)
    t_load = time.perf_counter() - t0

    # Generate per-class deltas (cache)
    t1 = time.perf_counter()
    out_delta_dir = Path(args.out_delta_dir)
    generate_class_deltas(
        class_manifest=class_manifest,
        G=G,
        gen_cfg=gen_cfg,
        clip_model=clip_model,
        device=device,
        out_delta_dir=out_delta_dir,
        force=args.force_regenerate_deltas,
    )
    t_delta = time.perf_counter() - t1

    # Image transform (pixel space)
    tf = build_image_loader_tf(args.input_size, args.interpolation)

    poison_rows = []
    total_images = 0
    t_apply = 0.0
    t_save = 0.0
    t_total_start = time.perf_counter()

    for clean_path, label in tqdm(samples, desc="T2UE apply"):
        # Load image deterministically
        img = Image.open(clean_path).convert("RGB")
        pix = tf(img)  # (3,H,W) in [0,1]
        pix = pix.unsqueeze(0).to(device)  # (1,3,H,W)

        # Load cached class delta (pixel space)
        delta_path = out_delta_dir / f"delta_class{label:05d}.npy"
        if not delta_path.exists():
            raise FileNotFoundError(f"Missing class delta for label {label}: {delta_path}")

        delta_pix = np.load(delta_path)
        delta_pix = torch.from_numpy(delta_pix).to(device).unsqueeze(0)  # (1,3,H,W)

        # If input_size differs from generator output size, resize delta to match
        if delta_pix.shape[-1] != pix.shape[-1]:
            delta_pix = F.interpolate(delta_pix, size=pix.shape[-2:], mode="bilinear", align_corners=False)

        # Apply in pixel space
        t_a0 = time.perf_counter()
        poisoned = (pix + delta_pix).clamp(0.0, 1.0)
        t_apply += time.perf_counter() - t_a0

        # Save
        t_s0 = time.perf_counter()
        fname = os.path.basename(clean_path)
        poisoned_path = os.path.join(args.out_images_dir, fname)
        save_tensor_as_image(poisoned[0], poisoned_path, quality=args.save_quality)
        t_save += time.perf_counter() - t_s0

        poison_rows.append((clean_path, poisoned_path))
        total_images += 1

    t_total = time.perf_counter() - t_total_start

    # Write poison map
    with open(args.out_poison_map, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clean_path", "poisoned_path"])
        w.writerows(poison_rows)

    metrics = {
        "total_images": total_images,
        "total_time_sec": t_total,
        "load_models_time_sec": t_load,
        "delta_generation_time_sec": t_delta,
        "apply_time_sec": t_apply,
        "save_time_sec": t_save,
        "throughput_img_per_sec": (total_images / t_total) if t_total > 0 else None,
        "args": vars(args),
        "notes": {
            "delta_space": "pixel_space_cached (converted from CLIP-normalized deltas via *CLIP_STD)",
            "determinism": "order follows CSV; class z derived from manifest seeds",
        }
    }

    with open(args.out_metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved poisoned images: {args.out_images_dir}")
    print(f"Saved poison map: {args.out_poison_map}")
    print(f"Saved metrics: {args.out_metrics_json}")
    print(f"Throughput: {metrics['throughput_img_per_sec']:.2f} img/s")

if __name__ == "__main__":
    main()
