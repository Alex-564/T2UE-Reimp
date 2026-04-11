import importlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from t2ue.utils.misc import load_yaml
from t2ue.utils.seed import seed_all
from t2ue.utils.checkpoint import save_checkpoint, load_checkpoint
from t2ue.utils.meters import AvgMeter

from t2ue.data.transforms import build_clip_image_transform
from t2ue.data.coco import CocoCaptionPairs

from t2ue.models.clip_surrogate import OpenAIClipSurrogate
from t2ue.models.generator import T2UEGenerator, GenConfig
from t2ue.losses.infonce import symmetric_infonce

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _build_compat_signature(cfg: Dict[str, Any], dl_len: int, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "clip_model_name": str(cfg["clip"]["model_name"]),
        "gen": {
            "z_dim": int(cfg["gen"]["z_dim"]),
            "text_dim": int(cfg["gen"]["text_dim"]),
            "base_ch": int(cfg["gen"]["base_ch"]),
            "out_res": int(cfg["gen"]["out_res"]),
            "eps": float(cfg["gen"]["eps"]),
        },
        "train": {
            "batch_size": int(cfg["train"]["batch_size"]),
            "lr": float(cfg["train"]["lr"]),
            "weight_decay": float(cfg["train"]["weight_decay"]),
            "amp": cfg["train"].get("amp", {}),
        },
        "num_workers": int(cfg.get("num_workers", 4)),
        "prefetch_factor": int(cfg.get("prefetch_factor", 2)),
        "dataset": dataset_info,
        "dl_len": int(dl_len),
    }


def _get_rng_state(device: torch.device) -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }


def _set_rng_state(payload: Dict[str, Any], device: torch.device) -> None:
    rng = payload.get("rng_state")
    if rng is None:
        raise ValueError("Resume checkpoint is missing rng_state; cannot guarantee deterministic resume.")
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"])
    if device.type == "cuda":
        if rng.get("torch_cuda") is None:
            raise ValueError("Resume checkpoint is missing CUDA RNG state for CUDA training.")
        torch.cuda.set_rng_state_all(rng["torch_cuda"])


def collate_fn(batch):
    images, caps = zip(*batch)
    images = torch.stack(images, dim=0)
    caps = list(caps)
    return images, caps


def _pick_wds_caption(meta: Dict[str, Any]) -> str:
    captions = meta.get("captions")
    if not isinstance(captions, list) or len(captions) == 0:
        raise ValueError("WebDataset sample metadata must contain a non-empty 'captions' list.")
    return random.choice(captions)


def _build_train_loader(
    cfg: Dict[str, Any],
    tfm,
    coco_root: Optional[str],
    coco_ann: Optional[str],
    wds_pattern: Optional[str],
    wds_num_samples: Optional[int],
    wds_shuffle: int,
) -> Tuple[DataLoader, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    num_workers = int(cfg.get("num_workers", 4))
    batch_size = int(cfg["train"]["batch_size"])

    dl_kwargs: Dict[str, Any] = {}
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))

    if wds_pattern is not None:
        if wds_num_samples is None or int(wds_num_samples) <= 0:
            raise ValueError("--wds-num-samples must be a positive integer when --wds-pattern is used.")

        try:
            wds = importlib.import_module("webdataset")
        except ModuleNotFoundError as exc:
            raise ImportError(
                "WebDataset support requires the 'webdataset' package. Install dependencies from requirements.txt."
            ) from exc

        ds = (
            wds.WebDataset(str(wds_pattern), shardshuffle=True)
            .shuffle(int(wds_shuffle))
            .decode("pil")
            .to_tuple("jpg", "json")
            .map_tuple(tfm, _pick_wds_caption)
            .with_length(int(wds_num_samples))
        )

        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
            **dl_kwargs,
        )

        dataset_info = {
            "kind": "webdataset",
            "pattern": str(wds_pattern),
            "num_samples": int(wds_num_samples),
            "shuffle_buffer": int(wds_shuffle),
        }
        runtime_data = {
            "source": "webdataset",
            "dataloader_shuffle": False,
            "dataset_shuffle_buffer": int(wds_shuffle),
        }
        return dl, dl_kwargs, dataset_info, runtime_data

    if coco_root is None or coco_ann is None:
        raise ValueError("COCO mode requires both --coco-root and --coco-ann.")

    ds = CocoCaptionPairs(root=coco_root, annFile=coco_ann, transform=tfm)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        **dl_kwargs,
    )

    dataset_info = {
        "kind": "coco_captions",
        "coco_root": str(coco_root),
        "coco_ann": str(coco_ann),
    }
    runtime_data = {
        "source": "coco_captions",
        "dataloader_shuffle": True,
    }
    return dl, dl_kwargs, dataset_info, runtime_data


def main(
    cfg_path: str,
    coco_root: Optional[str] = None,
    coco_ann: Optional[str] = None,
    wds_pattern: Optional[str] = None,
    wds_num_samples: Optional[int] = None,
    wds_shuffle: int = 10000,
    resume: Optional[str] = None,
):
    """
    Train the T2UE generator G to minimize the InfoNCE loss computed
    by frozen CLIP surrogate (f_I, f_T) on poisoned data.
    
    """
    cfg: Dict[str, Any] = load_yaml(cfg_path)
    seed_all(int(cfg["seed"]))
    run_started = time.perf_counter()

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
    # Uses either COCO captions dataset or WebDataset shards with the same random-caption semantics.
    tfm = build_clip_image_transform(out_res=int(cfg["gen"]["out_res"]))
    dl, dl_kwargs, dataset_info, runtime_data = _build_train_loader(
        cfg=cfg,
        tfm=tfm,
        coco_root=coco_root,
        coco_ann=coco_ann,
        wds_pattern=wds_pattern,
        wds_num_samples=wds_num_samples,
        wds_shuffle=wds_shuffle,
    )
    num_workers = int(cfg.get("num_workers", 4))
    compat_sig = _build_compat_signature(cfg, dl_len=len(dl), dataset_info=dataset_info)

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
    scheduler_policy = "new_total_steps"

    # CLIP-normalized bounds corresponding to pixel-space [0, 1].
    clip_mean = torch.tensor(CLIP_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    clip_std = torch.tensor(CLIP_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    valid_min = (0.0 - clip_mean) / clip_std
    valid_max = (1.0 - clip_mean) / clip_std

    # Checkpointing params
    log_every = int(cfg["train"]["log_every"])
    save_every_epochs = int(cfg["train"]["save_every_epochs"])
    metrics_path = out_dir / "metrics.jsonl"
    summary_path = out_dir / "summary.json"
    run_id = f"{int(time.time())}_{os.getpid()}"

    start_epoch = 1
    global_step = 0
    resume_from = None

    if resume is not None:
        resume_path = Path(resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")

        payload = load_checkpoint(str(resume_path), map_location=device)
        resume_from = str(resume_path)

        if "cfg_compat" in payload and payload["cfg_compat"] != compat_sig:
            raise ValueError(
                "Checkpoint/config compatibility mismatch for deterministic resume. "
                "Use matching config/data-loader settings for resume."
            )

        G.load_state_dict(payload["state_dict"], strict=True)
        opt.load_state_dict(payload["optim"])
        if "scheduler" not in payload:
            raise ValueError("Resume checkpoint is missing scheduler state.")
        sched.load_state_dict(payload["scheduler"])

        # Policy: keep a strict step-wise schedule shape for the current target epochs.
        sched.T_max = total_steps

        _set_rng_state(payload, device=device)
        if use_grad_scaler and scaler is not None and "scaler" in payload and payload["scaler"] is not None:
            scaler.load_state_dict(payload["scaler"])

        ckpt_epoch = int(payload.get("epoch", 0))
        if ckpt_epoch <= 0:
            raise ValueError("Resume checkpoint has invalid epoch value.")
        if int(cfg["train"]["epochs"]) < ckpt_epoch:
            raise ValueError(
                f"Configured epochs ({cfg['train']['epochs']}) is smaller than checkpoint epoch ({ckpt_epoch})."
            )

        global_step = int(payload.get("global_step", ckpt_epoch * len(dl)))
        start_epoch = ckpt_epoch + 1
        print(f"Resuming from {resume_path} at epoch {ckpt_epoch}, global_step {global_step}.")

    if start_epoch > int(cfg["train"]["epochs"]):
        print("Nothing to do: resume checkpoint already reached configured total epochs.")
        return

    best_epoch_loss = float("inf")
    completed_epochs = 0
    for epoch in range(start_epoch, int(cfg["train"]["epochs"]) + 1):
        epoch_started = time.perf_counter()
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

        checkpoint_saved = False
        checkpoint_path_str = None
        if epoch % save_every_epochs == 0 or epoch == int(cfg["train"]["epochs"]):
            ckpt_path = out_dir / f"generator_epoch{epoch:04d}.pt"
            checkpoint_saved = True
            checkpoint_path_str = str(ckpt_path)
            save_checkpoint(
                str(ckpt_path),
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "gen_cfg": gen_cfg.__dict__,
                    "state_dict": G.state_dict(),
                    "optim": opt.state_dict(),
                    "scheduler": sched.state_dict(),
                    "rng_state": _get_rng_state(device),
                    "scaler": scaler.state_dict() if use_grad_scaler and scaler is not None else None,
                    "cfg_compat": compat_sig,
                    "cfg_resolved": cfg,
                    "cfg_path": cfg_path,
                },
            )

        epoch_time = time.perf_counter() - epoch_started
        completed_epochs += 1
        best_epoch_loss = min(best_epoch_loss, float(meter.avg))
        _append_jsonl(
            metrics_path,
            {
                "timestamp_utc": _iso_now(),
                "run_id": run_id,
                "resume_from": resume_from,
                "epoch": epoch,
                "global_step": global_step,
                "train_loss_avg": float(meter.avg),
                "lr": float(sched.get_last_lr()[0]),
                "epoch_time_sec": float(epoch_time),
                "num_batches": int(len(dl)),
                "num_samples_seen": int(len(dl) * int(cfg["train"]["batch_size"])),
                "checkpoint_saved": checkpoint_saved,
                "checkpoint_path": checkpoint_path_str,
                "runtime": {
                    "device": str(device),
                    "amp_enabled": bool(amp_enabled),
                    "amp_dtype": amp_dtype_name,
                    "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32) if device.type == "cuda" else False,
                    "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32) if device.type == "cuda" else False,
                    "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                    "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                    "num_workers": int(num_workers),
                    "persistent_workers": bool(dl_kwargs.get("persistent_workers", False)),
                    "data_source": str(runtime_data.get("source")),
                    "dataloader_shuffle": bool(runtime_data.get("dataloader_shuffle", False)),
                    "dataset_shuffle_buffer": runtime_data.get("dataset_shuffle_buffer"),
                },
                "scheduler_policy": scheduler_policy,
            },
        )

    total_time = time.perf_counter() - run_started
    _write_json_atomic(
        summary_path,
        {
            "timestamp_utc": _iso_now(),
            "run_id": run_id,
            "resume_from": resume_from,
            "cfg_path": cfg_path,
            "out_dir": str(out_dir),
            "completed_epochs": completed_epochs,
            "final_epoch": int(cfg["train"]["epochs"]),
            "global_step": global_step,
            "best_epoch_loss": None if best_epoch_loss == float("inf") else float(best_epoch_loss),
            "total_time_sec": float(total_time),
            "metrics_jsonl": str(metrics_path),
            "compat_signature": compat_sig,
            "scheduler_policy": scheduler_policy,
            "notes": {
                "checkpoint_cadence": "config-driven save_every_epochs plus final epoch",
                "resume_scope": "epoch-boundary deterministic resume",
            },
        },
    )

    print(f"Done. Checkpoints in: {out_dir}")
    print(f"Metrics JSONL: {metrics_path}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--coco-root", default=None, type=str, help="Path to COCO train images directory")
    ap.add_argument("--coco-ann", default=None, type=str, help="Path to COCO captions annotation JSON")
    ap.add_argument(
        "--wds-pattern",
        default=None,
        type=str,
        help="WebDataset shard URL/pattern, e.g. /path/train-{000000..000011}.tar",
    )
    ap.add_argument(
        "--wds-num-samples",
        default=None,
        type=int,
        help="Total number of samples represented by the WebDataset shards.",
    )
    ap.add_argument(
        "--wds-shuffle",
        default=10000,
        type=int,
        help="WebDataset in-memory shuffle buffer size.",
    )
    ap.add_argument(
        "--resume",
        default=None,
        type=str,
        help="Optional checkpoint path for epoch-boundary resume with deterministic state restoration.",
    )
    args = ap.parse_args()

    using_wds = args.wds_pattern is not None
    if using_wds:
        if args.coco_root is not None or args.coco_ann is not None:
            ap.error("Use either --wds-pattern mode or --coco-root/--coco-ann mode, not both.")
        if args.wds_num_samples is None:
            ap.error("--wds-num-samples is required when --wds-pattern is set.")
    else:
        if args.coco_root is None or args.coco_ann is None:
            ap.error("Provide --coco-root and --coco-ann when --wds-pattern is not used.")

    main(
        cfg_path=args.config,
        coco_root=args.coco_root,
        coco_ann=args.coco_ann,
        wds_pattern=args.wds_pattern,
        wds_num_samples=args.wds_num_samples,
        wds_shuffle=int(args.wds_shuffle),
        resume=args.resume,
    )
