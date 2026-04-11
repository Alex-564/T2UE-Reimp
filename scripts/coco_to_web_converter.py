import argparse
import importlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_coco_annotations(annotation_file: Path) -> Dict[str, Any]:
    if not annotation_file.exists():
        raise FileNotFoundError(f"Annotation file does not exist: {annotation_file}")
    with annotation_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("COCO annotation payload root must be an object/dict.")
    if "images" not in payload or "annotations" not in payload:
        raise ValueError("COCO annotation payload must include 'images' and 'annotations' keys.")
    if not isinstance(payload["images"], list) or not isinstance(payload["annotations"], list):
        raise ValueError("COCO 'images' and 'annotations' entries must be lists.")
    return payload


def _build_caption_index(annotations: List[Dict[str, Any]]) -> Dict[int, List[str]]:
    img_to_captions: Dict[int, List[str]] = defaultdict(list)
    for ann in annotations:
        image_id = ann.get("image_id")
        caption = ann.get("caption")
        if image_id is None or not isinstance(caption, str):
            continue
        img_to_captions[int(image_id)].append(caption)
    return dict(img_to_captions)


def convert_coco_to_wds(
    image_dir: Path,
    annotation_file: Path,
    output_prefix: Path,
    maxcount: int,
    maxsize: int,
    allow_empty_captions: bool,
) -> Dict[str, Any]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if maxcount <= 0:
        raise ValueError(f"--maxcount must be > 0, got {maxcount}")
    if maxsize <= 0:
        raise ValueError(f"--maxsize must be > 0, got {maxsize}")

    coco_data = _load_coco_annotations(annotation_file)
    img_to_captions = _build_caption_index(coco_data["annotations"])
    images = sorted(coco_data["images"], key=lambda row: int(row.get("id", -1)))

    try:
        wds = importlib.import_module("webdataset")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Conversion requires the 'webdataset' package. Install dependencies from requirements.txt."
        ) from exc

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pattern = f"{output_prefix}-%06d.tar"
    print(f"Writing WebDataset shards to: {pattern}")

    written = 0
    missing_images = 0
    skipped_empty_captions = 0
    malformed_rows = 0

    with wds.ShardWriter(pattern, maxcount=maxcount, maxsize=maxsize) as sink:
        for row in tqdm(images, total=len(images), desc="Converting"):
            image_id = row.get("id")
            file_name = row.get("file_name")
            if image_id is None or not isinstance(file_name, str):
                malformed_rows += 1
                continue

            image_id = int(image_id)
            captions = img_to_captions.get(image_id, [])
            if not captions and not allow_empty_captions:
                skipped_empty_captions += 1
                continue

            img_path = image_dir / file_name
            if not img_path.exists():
                missing_images += 1
                continue

            with img_path.open("rb") as stream:
                image_bytes = stream.read()

            sample = {
                "__key__": str(image_id).zfill(12),
                "jpg": image_bytes,
                "json": {
                    "captions": captions,
                    "image_id": image_id,
                    "file_name": file_name,
                },
            }
            sink.write(sample)
            written += 1

    shard_paths = sorted(output_prefix.parent.glob(f"{output_prefix.name}-*.tar"))
    return {
        "timestamp_utc": _iso_now(),
        "script": "scripts/coco_to_web_converter.py",
        "image_dir": str(image_dir),
        "annotation_file": str(annotation_file),
        "output_prefix": str(output_prefix),
        "output_pattern": pattern,
        "maxcount": int(maxcount),
        "maxsize": int(maxsize),
        "allow_empty_captions": bool(allow_empty_captions),
        "num_images_in_annotations": int(len(images)),
        "written_samples": int(written),
        "num_shards": int(len(shard_paths)),
        "missing_images": int(missing_images),
        "skipped_empty_captions": int(skipped_empty_captions),
        "malformed_rows": int(malformed_rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert COCO image+caption annotations into WebDataset tar shards.")
    ap.add_argument("--image-dir", required=True, type=str, help="Path to COCO image directory (e.g. train2017)")
    ap.add_argument("--annotation-file", required=True, type=str, help="Path to COCO captions JSON")
    ap.add_argument(
        "--output-prefix",
        required=True,
        type=str,
        help="Output shard prefix (e.g. dataset/mscoco_wds/train -> train-000000.tar)",
    )
    ap.add_argument("--maxcount", type=int, default=10000, help="Maximum samples per tar shard.")
    ap.add_argument(
        "--maxsize",
        type=int,
        default=3_000_000_000,
        help="Maximum shard size in bytes before rolling over to next tar.",
    )
    ap.add_argument(
        "--allow-empty-captions",
        action="store_true",
        help="If set, keep samples with zero captions. By default these are skipped.",
    )
    ap.add_argument(
        "--meta-json",
        type=str,
        default=None,
        help="Optional output path for conversion metadata JSON (default: <output-prefix>.meta.json).",
    )
    args = ap.parse_args()

    image_dir = Path(args.image_dir)
    annotation_file = Path(args.annotation_file)
    output_prefix = Path(args.output_prefix)
    meta_json = Path(args.meta_json) if args.meta_json is not None else output_prefix.with_suffix(".meta.json")

    summary = convert_coco_to_wds(
        image_dir=image_dir,
        annotation_file=annotation_file,
        output_prefix=output_prefix,
        maxcount=int(args.maxcount),
        maxsize=int(args.maxsize),
        allow_empty_captions=bool(args.allow_empty_captions),
    )

    meta_json.parent.mkdir(parents=True, exist_ok=True)
    with meta_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print("Conversion complete")
    print(f"  written samples:       {summary['written_samples']}")
    print(f"  shard count:           {summary['num_shards']}")
    print(f"  skipped empty caption: {summary['skipped_empty_captions']}")
    print(f"  missing images:        {summary['missing_images']}")
    print(f"  malformed rows:        {summary['malformed_rows']}")
    print(f"  metadata json:         {meta_json}")


if __name__ == "__main__":
    main()