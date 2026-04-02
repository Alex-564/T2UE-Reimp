import argparse
import csv
import json
import os
import sys
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Allow direct execution from repository root without requiring PYTHONPATH tweaks.
SCRIPT_DIR = Path(__file__).resolve().parent
T2UE_REPO_ROOT = SCRIPT_DIR.parent
if str(T2UE_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(T2UE_REPO_ROOT))

from t2ue.models.clip_surrogate import OpenAIClipSurrogate
from t2ue.models.generator import T2UEGenerator, GenConfig
from t2ue.utils.checkpoint import load_checkpoint

# hardcode CLIP mean/std.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

def tensor_to_uint8_hwc(x: torch.Tensor) -> np.ndarray:
    x = x.detach().clamp(0, 1).cpu()
    return (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def save_uint8_hwc(arr_hwc: np.ndarray, out_path: str, image_format: str, save_quality: int) -> None:
    if image_format == "png":
        Image.fromarray(arr_hwc).save(out_path, format="PNG", compress_level=0)
    elif image_format == "jpg":
        Image.fromarray(arr_hwc).save(out_path, format="JPEG", quality=save_quality)
    else:
        raise ValueError(f"Unsupported image_format={image_format}")


def atomic_save_uint8_hwc_as_image(arr_hwc: np.ndarray, out_path: str, image_format: str, save_quality: int) -> None:
    tmp = out_path + ".tmp"
    save_uint8_hwc(arr_hwc, tmp, image_format=image_format, save_quality=save_quality)
    os.replace(tmp, out_path)


def atomic_write_csv(path: str, header: List[str], rows: List[Tuple[str, str]]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)


def atomic_write_json(path: str, obj: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class T2UEImageDataset(Dataset):
    """Deterministic CSV-order image dataset for batched apply."""

    def __init__(self, samples: List[Tuple[str, int, str]], input_size: int, interpolation: str):
        self.samples = samples
        self.tf = build_image_loader_tf(input_size, interpolation)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        clean_path, label, poisoned_rel_path = self.samples[idx]
        with Image.open(clean_path) as img:
            x = self.tf(img.convert("RGB"))
        return x, int(label), clean_path, poisoned_rel_path

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

def delta_norm_to_pixel(delta_norm: torch.Tensor) -> torch.Tensor:
    """
    Convert delta in CLIP-normalized space to pixel-space delta:
      x = (p - mean)/std
      x' = x + delta_norm  => p' = p + delta_norm * std
    So delta_pixel = delta_norm * std
    """
    std = torch.tensor(CLIP_STD, device=delta_norm.device).view(1,3,1,1)
    return delta_norm * std


def resolve_poisoned_output_path(
    out_images_dir: str,
    clean_path: str,
    poisoned_rel_path: str,
    image_format: str,
) -> str:
    ext = ".png" if image_format == "png" else ".jpg"
    fallback_name = os.path.splitext(os.path.basename(clean_path))[0] + ext

    rel = (poisoned_rel_path or "").strip()
    if not rel:
        rel = fallback_name
    else:
        rel = os.path.normpath(rel)
        if os.path.isabs(rel):
            raise RuntimeError(f"poisoned_rel_path must be relative, got absolute path: {poisoned_rel_path}")
        if rel == ".." or rel.startswith(".." + os.sep):
            raise RuntimeError(f"poisoned_rel_path escapes output directory: {poisoned_rel_path}")

    return os.path.abspath(os.path.join(out_images_dir, rel))


def load_class_manifest(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        items = []
        for k, v in raw.items():
            if not isinstance(v, dict):
                raise RuntimeError(f"Invalid class manifest entry for key '{k}': expected object")
            entry = dict(v)
            if "class_id" not in entry:
                entry["class_id"] = k
            items.append(entry)
    elif isinstance(raw, list):
        items = raw
    else:
        raise RuntimeError("class manifest must be a JSON object or list")

    normalized: Dict[int, Dict[str, Any]] = {}
    z_seeds = set()
    for idx, entry in enumerate(items):
        if not isinstance(entry, dict):
            raise RuntimeError(f"Invalid class manifest entry at index {idx}: expected object")

        try:
            class_id = int(entry["class_id"])
        except Exception as e:
            raise RuntimeError(f"Invalid class_id in manifest entry {idx}") from e
        if class_id < 0:
            raise RuntimeError(f"class_id must be non-negative, got {class_id}")
        if class_id in normalized:
            raise RuntimeError(f"Duplicate class_id in manifest: {class_id}")

        prompt = str(entry.get("prompt", "")).strip()
        if not prompt:
            raise RuntimeError(f"Missing or empty prompt for class_id={class_id}")

        try:
            z_seed = int(entry["z_seed"])
        except Exception as e:
            raise RuntimeError(f"Invalid z_seed for class_id={class_id}") from e

        if z_seed in z_seeds:
            raise RuntimeError(f"Duplicate z_seed in class manifest: {z_seed}")
        z_seeds.add(z_seed)

        normalized[class_id] = {
            "class_id": class_id,
            "prompt": prompt,
            "z_seed": z_seed,
        }

    if not normalized:
        raise RuntimeError("Class manifest is empty")

    return [normalized[c] for c in sorted(normalized.keys())]


def build_manifest_index(class_manifest: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {
        int(entry["class_id"]): {
            "class_id": int(entry["class_id"]),
            "prompt": str(entry["prompt"]),
            "z_seed": int(entry["z_seed"]),
        }
        for entry in class_manifest
    }


def expected_delta_meta(
    class_id: int,
    prompt: str,
    z_seed: int,
    clip_model_name: str,
    ckpt_path: str,
    class_manifest_sha256: str,
) -> Dict[str, Any]:
    return {
        "class_id": int(class_id),
        "prompt": str(prompt),
        "z_seed": int(z_seed),
        "clip_model": str(clip_model_name),
        "t2ue_ckpt": os.path.abspath(ckpt_path),
        "class_manifest_sha256": str(class_manifest_sha256),
    }


def validate_cached_delta(delta_path: Path, meta_path: Path, expected_meta: Dict[str, Any]) -> bool:
    if not delta_path.exists() or not meta_path.exists():
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            cached_meta = json.load(f)
    except Exception:
        return False
    return cached_meta == expected_meta


def load_samples(samples_csv: str) -> List[Tuple[str, int, str]]:
    samples: List[Tuple[str, int, str]] = []
    with open(samples_csv, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise RuntimeError("samples CSV has no header")
        required = {"clean_path", "label"}
        if not required.issubset(set(r.fieldnames)):
            raise RuntimeError(f"samples CSV must contain columns: {sorted(required)}")

        for row_idx, row in enumerate(r, start=2):
            clean_path = str(row.get("clean_path", "")).strip()
            if not clean_path:
                raise RuntimeError(f"Missing clean_path at CSV line {row_idx}")
            try:
                label = int(row["label"])
            except Exception as e:
                raise RuntimeError(f"Invalid label at CSV line {row_idx}") from e

            poisoned_rel_path = str(row.get("poisoned_rel_path", "")).strip()
            samples.append((clean_path, label, poisoned_rel_path))

    if not samples:
        raise RuntimeError("No samples provided.")
    return samples

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
    class_manifest: List[Dict[str, Any]],
    G,
    gen_cfg,
    clip_model,
    device: torch.device,
    out_delta_dir: Path,
    clip_model_name: str,
    ckpt_path: str,
    class_manifest_sha256: str,
    force: bool = False,
):
    """
    Generates and saves one delta per class deterministically.
    Saves:
      out_delta_dir/delta_classXXXX.npy  (pixel-space delta, [3,H,W] float32)
      out_delta_dir/delta_classXXXX_meta.json  (prompt, seed)
    """
    out_delta_dir.mkdir(parents=True, exist_ok=True)

    for entry in tqdm(class_manifest, desc="Generate class deltas"):
        c = int(entry["class_id"])
        prompt = entry["prompt"]
        z_seed = int(entry["z_seed"])

        delta_path = out_delta_dir / f"delta_class{c:05d}.npy"
        meta_path  = out_delta_dir / f"delta_class{c:05d}_meta.json"
        meta_obj = expected_delta_meta(
            class_id=c,
            prompt=prompt,
            z_seed=z_seed,
            clip_model_name=clip_model_name,
            ckpt_path=ckpt_path,
            class_manifest_sha256=class_manifest_sha256,
        )

        if (not force) and validate_cached_delta(delta_path, meta_path, meta_obj):
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
        atomic_write_json(str(meta_path), meta_obj)


def load_and_validate_delta_for_label(
    label: int,
    out_delta_dir: Path,
    target_hw: Tuple[int, int],
    expected_manifest: Dict[int, Dict[str, Any]],
    clip_model_name: str,
    ckpt_path: str,
    class_manifest_sha256: str,
) -> torch.Tensor:
    expected = expected_manifest.get(int(label))
    if expected is None:
        raise RuntimeError(
            f"Label {label} from samples CSV is missing in class manifest."
        )

    delta_path = out_delta_dir / f"delta_class{label:05d}.npy"
    meta_path = out_delta_dir / f"delta_class{label:05d}_meta.json"
    meta_obj = expected_delta_meta(
        class_id=int(label),
        prompt=str(expected["prompt"]),
        z_seed=int(expected["z_seed"]),
        clip_model_name=clip_model_name,
        ckpt_path=ckpt_path,
        class_manifest_sha256=class_manifest_sha256,
    )
    if not validate_cached_delta(delta_path, meta_path, meta_obj):
        raise RuntimeError(
            f"Stale or missing class delta cache detected for label={label}. "
            "Re-run without --skip-regenerate-deltas to rebuild cache."
        )

    delta_pix = np.load(delta_path)
    delta_t = torch.from_numpy(delta_pix).unsqueeze(0)
    if tuple(delta_t.shape[-2:]) != tuple(target_hw):
        delta_t = F.interpolate(delta_t, size=target_hw, mode="bilinear", align_corners=False)
    return delta_t


def apply_cached_deltas(
    samples: List[Tuple[str, int, str]],
    out_images_dir: str,
    out_delta_dir: Path,
    image_format: str,
    input_size: int,
    interpolation: str,
    save_quality: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    expected_manifest: Dict[int, Dict[str, Any]],
    clip_model_name: str,
    ckpt_path: str,
    class_manifest_sha256: str,
) -> Tuple[List[Tuple[str, str]], float, float, int, float]:
    ds = T2UEImageDataset(samples=samples, input_size=input_size, interpolation=interpolation)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    poison_rows: List[Tuple[str, str]] = []
    t_apply = 0.0
    t_save = 0.0
    seen_poisoned_paths = set()
    delta_cache: Dict[int, torch.Tensor] = {}

    t_total_start = time.perf_counter()

    for xb, yb, clean_paths, poisoned_rel_paths in tqdm(dl, desc="T2UE apply"):
        xb = xb.to(device, non_blocking=True)
        out_b = torch.empty_like(xb)

        t_a0 = time.perf_counter()
        for i in range(xb.size(0)):
            label = int(yb[i].item())
            if label not in delta_cache:
                delta_cache[label] = load_and_validate_delta_for_label(
                    label=label,
                    out_delta_dir=out_delta_dir,
                    target_hw=(xb.shape[-2], xb.shape[-1]),
                    expected_manifest=expected_manifest,
                    clip_model_name=clip_model_name,
                    ckpt_path=ckpt_path,
                    class_manifest_sha256=class_manifest_sha256,
                ).to(device)
            out_b[i:i+1] = (xb[i:i+1] + delta_cache[label]).clamp(0.0, 1.0)
        t_apply += time.perf_counter() - t_a0

        t_s0 = time.perf_counter()
        for i in range(xb.size(0)):
            clean_path = str(clean_paths[i])
            poisoned_rel_path = str(poisoned_rel_paths[i])
            poisoned_path = resolve_poisoned_output_path(
                out_images_dir=out_images_dir,
                clean_path=clean_path,
                poisoned_rel_path=poisoned_rel_path,
                image_format=image_format,
            )
            if poisoned_path in seen_poisoned_paths:
                raise RuntimeError(
                    f"Duplicate poisoned output path detected: {poisoned_path}. "
                    "Regenerate with unique per-sample relative paths."
                )
            seen_poisoned_paths.add(poisoned_path)

            os.makedirs(os.path.dirname(poisoned_path), exist_ok=True)
            arr = tensor_to_uint8_hwc(out_b[i])
            atomic_save_uint8_hwc_as_image(
                arr_hwc=arr,
                out_path=poisoned_path,
                image_format=image_format,
                save_quality=save_quality,
            )
            poison_rows.append((clean_path, poisoned_path))
        t_save += time.perf_counter() - t_s0

    t_total = time.perf_counter() - t_total_start
    return poison_rows, t_apply, t_save, len(poison_rows), t_total

def main():
    ap = argparse.ArgumentParser()

    # Inputs / outputs: single-pass all workflow
    ap.add_argument("--samples-csv", type=str, required=True)
    ap.add_argument("--class-manifest", type=str, required=True)
    ap.add_argument("--t2ue-ckpt", type=str, required=True)
    ap.add_argument("--clip-model", type=str, default="ViT-B/32")

    ap.add_argument("--out-images-dir", type=str, required=True)
    ap.add_argument("--out-poison-map", type=str, required=True)
    ap.add_argument("--out-metrics-json", type=str, default=None)

    # Cache
    ap.add_argument("--out-delta-dir", type=str, required=True, help="Where per-class deltas are cached")
    ap.add_argument(
        "--skip-regenerate-deltas",
        action="store_true",
        help="Reuse valid cached deltas instead of regenerating (default regenerates).",
    )

    # Image handling
    ap.add_argument("--input-size", type=int, default=112, help="Resize images to this before applying poison")
    ap.add_argument("--interpolation", type=str, default="bilinear",
                    choices=["nearest", "bilinear", "bicubic", "lanczos"])
    ap.add_argument("--image-format", type=str, default="png", choices=["png", "jpg"])
    ap.add_argument("--save-quality", type=int, default=100)

    # Determinism / performance 
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for batched apply")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size for batched apply")
    ap.add_argument("--seed", type=int, default=0, help="Global seed for any residual randomness")

    args = ap.parse_args()

    args.out_images_dir = os.path.abspath(args.out_images_dir)
    args.out_poison_map = os.path.abspath(args.out_poison_map)
    args.out_delta_dir = os.path.abspath(args.out_delta_dir)
    args.class_manifest = os.path.abspath(args.class_manifest)
    args.t2ue_ckpt = os.path.abspath(args.t2ue_ckpt)
    force_regenerate_deltas = (not args.skip_regenerate_deltas)

    if args.out_metrics_json is None:
        args.out_metrics_json = os.path.join(os.path.dirname(args.out_poison_map), "metrics.json")
    args.out_metrics_json = os.path.abspath(args.out_metrics_json)

    os.makedirs(args.out_images_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_poison_map), exist_ok=True)
    os.makedirs(args.out_delta_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_metrics_json), exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Global determinism
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t_load = 0.0
    t_delta = 0.0
    t_apply = 0.0
    t_save = 0.0
    t_apply_total = 0.0
    total_images = 0
    out_delta_dir = Path(args.out_delta_dir)

    class_manifest = load_class_manifest(args.class_manifest)
    class_manifest_index = build_manifest_index(class_manifest)
    class_manifest_sha256 = file_sha256(args.class_manifest)

    # Single pass: load models -> generate deltas -> apply deltas.
    t0 = time.perf_counter()
    G, gen_cfg = load_t2ue_generator(args.t2ue_ckpt, device=device)
    clip_model = load_openai_clip_surrogate(args.clip_model, device=device)
    t_load = time.perf_counter() - t0

    t1 = time.perf_counter()
    generate_class_deltas(
        class_manifest=class_manifest,
        G=G,
        gen_cfg=gen_cfg,
        clip_model=clip_model,
        device=device,
        out_delta_dir=out_delta_dir,
        clip_model_name=args.clip_model,
        ckpt_path=args.t2ue_ckpt,
        class_manifest_sha256=class_manifest_sha256,
        force=force_regenerate_deltas,
    )
    t_delta = time.perf_counter() - t1

    samples = load_samples(args.samples_csv)
    poison_rows, t_apply, t_save, total_images, t_apply_total = apply_cached_deltas(
        samples=samples,
        out_images_dir=args.out_images_dir,
        out_delta_dir=out_delta_dir,
        image_format=args.image_format,
        input_size=args.input_size,
        interpolation=args.interpolation,
        save_quality=args.save_quality,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        expected_manifest=class_manifest_index,
        clip_model_name=args.clip_model,
        ckpt_path=args.t2ue_ckpt,
        class_manifest_sha256=class_manifest_sha256,
    )

    atomic_write_csv(args.out_poison_map, ["clean_path", "poisoned_path"], poison_rows)

    metrics = {
        "total_images": total_images,
        "total_time_sec": t_load + t_delta + t_apply_total,
        "load_models_time_sec": t_load,
        "delta_generation_time_sec": t_delta,
        "apply_time_sec": t_apply,
        "save_time_sec": t_save,
        "throughput_img_per_sec": (total_images / t_apply_total) if t_apply_total > 0 else None,
        "args": vars(args),
        "notes": {
            "delta_space": "pixel_space_cached (converted from CLIP-normalized deltas via *CLIP_STD)",
            "determinism": "delta generation order follows sorted class_id; apply order follows CSV via shuffle=False DataLoader",
            "zero_contact_delta_generation": True,
            "force_regenerate_default": True,
        }
    }

    atomic_write_json(args.out_metrics_json, metrics)

    print(f"Saved class deltas: {args.out_delta_dir}")
    print(f"Saved poisoned images: {args.out_images_dir}")
    print(f"Saved poison map: {args.out_poison_map}")
    print(f"Saved metrics: {args.out_metrics_json}")
    if metrics["throughput_img_per_sec"] is not None:
        print(f"Throughput: {metrics['throughput_img_per_sec']:.2f} img/s")

if __name__ == "__main__":
    main()
