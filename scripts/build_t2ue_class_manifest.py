import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


Z_SEED_MAX_DEFAULT = 2**31 - 1


def _normalize_header(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _pick_header(fieldnames: List[str], candidate_headers: List[str], description: str) -> str:
    if not fieldnames:
        raise RuntimeError("CSV has no header")

    norm_to_actual = {_normalize_header(h): h for h in fieldnames}
    for cand in candidate_headers:
        norm = _normalize_header(cand)
        if norm in norm_to_actual:
            return norm_to_actual[norm]

    raise RuntimeError(
        f"Could not find {description} column. "
        f"Accepted headers: {candidate_headers}. Found headers: {fieldnames}"
    )


def clean_identity_name(name: str) -> str:
    text = str(name).replace("_", " ").strip()
    if not text:
        return ""

    kept = []
    for ch in text:
        if ch.isalpha() or ch.isdigit() or ch.isspace() or ch in ".'-":
            kept.append(ch)
        else:
            kept.append(" ")

    text = "".join(kept)

    text = re.sub(r"\s*\.\s*", ".", text)
    text = re.sub(r"\s*'\s*", "'", text)
    text = re.sub(r"\s*-\s*", "-", text)

    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" .'-")
    return text


def load_manifest_identities(manifest_csv: str) -> List[Dict[str, object]]:
    by_label: Dict[int, str] = {}

    with open(manifest_csv, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames or []

        label_col = _pick_header(fieldnames, ["label"], "manifest label")
        raw_id_col = _pick_header(fieldnames, ["raw_id", "class_id", "identity_id", "identity"], "manifest raw identity")

        for row_idx, row in enumerate(r, start=2):
            try:
                class_id = int(row[label_col])
            except Exception as e:
                raise RuntimeError(f"Invalid label at line {row_idx} in manifest CSV") from e

            raw_id = str(row.get(raw_id_col, "")).strip()
            if not raw_id:
                raise RuntimeError(f"Missing raw_id for label={class_id} at line {row_idx}")

            prev = by_label.get(class_id)
            if prev is None:
                by_label[class_id] = raw_id
            elif prev != raw_id:
                raise RuntimeError(
                    f"Inconsistent raw_id for label={class_id}: '{prev}' vs '{raw_id}' (line {row_idx})"
                )

    if not by_label:
        raise RuntimeError("No identities found in manifest CSV")

    return [{"class_id": c, "raw_id": by_label[c]} for c in sorted(by_label.keys())]


def load_annotation_name_map(annotation_csv: str) -> Dict[str, str]:
    name_map: Dict[str, str] = {}

    with open(annotation_csv, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames or []

        id_col = _pick_header(
            fieldnames,
            ["class_id", "raw_id", "identity_id", "identity", "id", "person_id", "personid"],
            "annotation identity key",
        )
        name_col = _pick_header(
            fieldnames,
            ["name", "identity_name", "person_name", "full_name", "fullname"],
            "annotation identity name",
        )

        for row in r:
            identity_key = str(row.get(id_col, "")).strip()
            if not identity_key:
                continue

            cleaned_name = clean_identity_name(str(row.get(name_col, "")))
            if not cleaned_name:
                continue

            # Keep first non-empty cleaned name if duplicates appear.
            if identity_key not in name_map:
                name_map[identity_key] = cleaned_name

    return name_map


def build_manifest(
    classes: List[Dict[str, object]],
    name_map: Dict[str, str],
    prompt_template: str,
    master_seed: int,
    max_z_seed: int,
) -> Tuple[Dict[str, Dict[str, object]], int, int]:
    if max_z_seed <= 0:
        raise RuntimeError("max_z_seed must be positive")

    rng = np.random.default_rng(master_seed)
    used_z = set()

    manifest: Dict[str, Dict[str, object]] = {}
    matched_name_count = 0
    fallback_raw_id_count = 0

    for entry in classes:
        class_id = int(entry["class_id"])
        raw_id = str(entry["raw_id"])

        class_name = name_map.get(raw_id, "")
        if class_name:
            matched_name_count += 1
        else:
            class_name = raw_id
            fallback_raw_id_count += 1

        try:
            prompt = prompt_template.format(
                class_id=class_id,
                raw_id=raw_id,
                class_name=class_name,
            )
        except KeyError as e:
            raise RuntimeError(f"Unknown placeholder in prompt template: {e}") from e

        if not str(prompt).strip():
            raise RuntimeError(f"Rendered prompt is empty for class_id={class_id}")

        z_seed = int(rng.integers(low=0, high=max_z_seed, endpoint=False))
        while z_seed in used_z:
            z_seed = int(rng.integers(low=0, high=max_z_seed, endpoint=False))
        used_z.add(z_seed)

        manifest[str(class_id)] = {
            "class_id": class_id,
            "raw_id": raw_id,
            "class_name": class_name,
            "prompt": prompt,
            "z_seed": z_seed,
        }

    if not manifest:
        raise RuntimeError("No class entries generated")

    return manifest, matched_name_count, fallback_raw_id_count


def parse_args():
    p = argparse.ArgumentParser("Build deterministic class manifest for T2UE from manifest + annotations")

    p.add_argument("--manifest-csv", type=str, required=True,
                   help="Dataset manifest CSV containing at least label and raw_id columns")
    p.add_argument("--annotation-csv", type=str, required=True,
                   help="Identity annotation CSV containing identity key and name columns")

    p.add_argument("--prompt-template", type=str, default="A photo of {class_name}")
    p.add_argument("--master-seed", type=int, default=0)
    p.add_argument("--max-z-seed", type=int, default=Z_SEED_MAX_DEFAULT)

    p.add_argument("--out-class-manifest", type=str, required=True)
    p.add_argument("--out-config-json", type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    classes = load_manifest_identities(args.manifest_csv)
    name_map = load_annotation_name_map(args.annotation_csv)

    manifest, matched_name_count, fallback_raw_id_count = build_manifest(
        classes=classes,
        name_map=name_map,
        prompt_template=args.prompt_template,
        master_seed=args.master_seed,
        max_z_seed=args.max_z_seed,
    )

    out_manifest = Path(args.out_class_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    if args.out_config_json is None:
        out_config_json = out_manifest.with_suffix(out_manifest.suffix + ".config.json")
    else:
        out_config_json = Path(args.out_config_json)
    out_config_json.parent.mkdir(parents=True, exist_ok=True)

    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    config = {
        "manifest_csv": str(Path(args.manifest_csv)),
        "annotation_csv": str(Path(args.annotation_csv)),
        "prompt_template": args.prompt_template,
        "master_seed": args.master_seed,
        "max_z_seed": args.max_z_seed,
        "num_classes": len(classes),
        "out_class_manifest": str(out_manifest),
    }

    with open(out_config_json, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved class manifest: {out_manifest}")
    print(f"Saved config: {out_config_json}")
    print(f"Classes: {len(classes)}")
    print(f"Classes using annotation names: {matched_name_count}")
    print(f"Classes using raw_id fallback: {fallback_raw_id_count}")


if __name__ == "__main__":
    main()
