import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


# text-to-unlearnable-examples t2ue trains generator on mscoco pairs
# this helper builds deterministic annotation subsets for controlled ablations
def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Source annotation file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("COCO annotation JSON root must be an object/dict.")
    return data


def _validate_coco_payload(data: Dict[str, Any]) -> None:
    if "images" not in data or "annotations" not in data:
        raise ValueError("COCO annotation JSON must contain 'images' and 'annotations' keys.")
    if not isinstance(data["images"], list) or not isinstance(data["annotations"], list):
        raise ValueError("COCO keys 'images' and 'annotations' must be lists.")


def _stable_subset(data: Dict[str, Any], fraction: float, seed: int) -> Dict[str, Any]:
    images: List[Dict[str, Any]] = data["images"]
    annotations: List[Dict[str, Any]] = data["annotations"]

    n_images_in = len(images)
    if n_images_in == 0:
        raise ValueError("Source annotation JSON has zero images; cannot create a subset.")

    subset_size = max(1, math.floor(n_images_in * fraction))

    rng = random.Random(seed)
    sampled_indices = rng.sample(range(n_images_in), k=subset_size)
    sampled_ids = sorted(images[i]["id"] for i in sampled_indices)
    sampled_id_set = set(sampled_ids)

    image_by_id = {img["id"]: img for img in images}
    subset_images = [image_by_id[iid] for iid in sampled_ids]

    # keep source annotation order while filtering selected image ids
    subset_annotations = [ann for ann in annotations if ann.get("image_id") in sampled_id_set]

    subset_payload: Dict[str, Any] = {}
    if "info" in data:
        subset_payload["info"] = data["info"]
    if "licenses" in data:
        subset_payload["licenses"] = data["licenses"]
    subset_payload["images"] = subset_images
    subset_payload["annotations"] = subset_annotations
    if "categories" in data:
        subset_payload["categories"] = data["categories"]

    return subset_payload


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create a deterministic percentage-based MSCOCO annotation subset by image ID."
    )
    ap.add_argument("--src-ann", required=True, type=str, help="Path to source COCO captions JSON")
    ap.add_argument("--out-ann", required=True, type=str, help="Path to output subset COCO captions JSON")
    ap.add_argument("--fraction", required=True, type=float, help="Subset fraction in (0, 1]")
    ap.add_argument("--seed", type=int, default=123, help="Sampling seed for deterministic subset selection")
    ap.add_argument(
        "--diag-json",
        type=str,
        default=None,
        help="Optional path for diagnostics JSON (default: <out-ann-stem>.diagnostics.json)",
    )
    args = ap.parse_args()

    if not (0.0 < args.fraction <= 1.0):
        raise ValueError(f"--fraction must be in (0, 1], got {args.fraction}")

    src_path = Path(args.src_ann)
    out_path = Path(args.out_ann)
    if args.diag_json is None:
        diag_path = out_path.with_name(f"{out_path.stem}.diagnostics.json")
    else:
        diag_path = Path(args.diag_json)

    t0 = time.perf_counter()

    src = _load_json(src_path)
    _validate_coco_payload(src)

    # subset is deterministic for same source fraction seed
    subset = _stable_subset(src, fraction=args.fraction, seed=args.seed)
    _write_json(out_path, subset)

    duration_sec = time.perf_counter() - t0

    n_images_in = len(src["images"])
    n_images_out = len(subset["images"])
    n_annotations_in = len(src["annotations"])
    n_annotations_out = len(subset["annotations"])

    sampled_preview = [img.get("id") for img in subset["images"][:10]]

    # diagnostics make t2ue data prep fully auditable
    diagnostics = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/subset_coco_annotations.py",
        "src_ann_path": str(src_path),
        "out_ann_path": str(out_path),
        "diag_json_path": str(diag_path),
        "fraction": float(args.fraction),
        "seed": int(args.seed),
        "subset_count_rule": "floor_with_min1",
        "n_images_in": n_images_in,
        "n_images_out": n_images_out,
        "n_annotations_in": n_annotations_in,
        "n_annotations_out": n_annotations_out,
        "images_retained_ratio": (n_images_out / n_images_in) if n_images_in else 0.0,
        "annotations_retained_ratio": (n_annotations_out / n_annotations_in) if n_annotations_in else 0.0,
        "sampled_image_id_preview": sampled_preview,
        "duration_sec": duration_sec,
    }
    _write_json(diag_path, diagnostics)

    print("Created deterministic COCO subset annotation file")
    print(f"source: {src_path}")
    print(f"output: {out_path}")
    print(f"diagnostics: {diag_path}")
    print(f"fraction: {args.fraction}")
    print(f"seed: {args.seed}")
    print(f"images: {n_images_out}/{n_images_in}")
    print(f"annotations: {n_annotations_out}/{n_annotations_in}")


if __name__ == "__main__":
    main()
