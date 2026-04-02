# T2UE (Text-to-Unlearnable Examples) — faithful reimplementation (OpenAI CLIP surrogate)

A faithful re-implementation of the T2UE: Generating Unlearnable Examples from Text Descriptions Research Paper
- Generator maps (text embedding, random latent) -> bounded perturbation delta_u
- Trained offline on MSCOCO using a *frozen* pretrained CLIP surrogate and symmetric InfoNCE.

Key paper matches:
- Architecture: FC -> reshape 4x4x512 -> SSACN blocks w/ SSCBN -> Conv -> Tanh
- Constraint: ||delta||_inf <= 8/255
- Training: Adam lr=1e-4, batch=128, cosine schedule, 500 epochs (ViT-B/32) / 300 (ViT-B/16)

## Install
pip install -r requirements.txt

## MSCOCO setup
Download COCO 2017 train images + captions annotations:
- images: train2017/
- annotations: annotations/captions_train2017.json

Use CLI arguments when launching training.

## Build deterministic subset annotations
Create a deterministic percentage subset of MSCOCO annotations (image-ID sampling,
all captions preserved for sampled images):

python scripts/subset_coco_annotations.py \
  --src-ann /path/to/coco/annotations/captions_train2017.json \
  --out-ann /path/to/coco/annotations/captions_train2017_subset50_seed123.json \
  --fraction 0.5 \
  --seed 123

This also writes diagnostics JSON by default next to the output annotation file:
`captions_train2017_subset50_seed123.diagnostics.json`.

Use the subset in training by pointing `--coco-ann` to the subset annotation file:

python scripts/train_generator.py --config configs/vitb32_coco.yaml \
  --coco-root /path/to/coco/images/train2017 \
  --coco-ann /path/to/coco/annotations/captions_train2017_subset50_seed123.json

## Train generator (offline)
python scripts/train_generator.py --config configs/vitb32_coco.yaml \
  --coco-root /path/to/coco/images/train2017 \
  --coco-ann /path/to/coco/annotations/captions_train2017.json

python scripts/train_generator.py --config configs/vitb16_coco.yaml \
  --coco-root /path/to/coco/images/train2017 \
  --coco-ann /path/to/coco/annotations/captions_train2017.json

Checkpoints appear in runs/vitb32/ or runs/vitb16/

## Export text-only noise (zero-contact)
Create a prompts file - Example:
echo "a photo of a cat" > prompts.txt
echo "a photo of a dog" >> prompts.txt

python scripts/export_noise.py \
  --ckpt runs/vitb32/generator_epoch0500.pt \
  --prompts prompts.txt \
  --out exported_noise/ \
  --clip_model "ViT-B/32"

This writes delta_*.npy files (3x224x224) in CLIP-normalized tensor space.

## Class-wise zero-contact pipeline (single pass)

The class-wise workflow remains conceptually split into:
1. Phase 1: deterministic class manifest generation.
2. Delta generation from class prompts and seeds.
3. Poison application to samples from cached class deltas.

Script execution is now single-pass only for steps 2+3:
- `scripts/generate_only_t2ue.py` always runs delta generation followed by apply.
- Phase flags are removed.
- Delta regeneration is the default protocol to avoid stale-cache reuse.

### Phase 1: build class manifest

Create a deterministic `class_manifest.json` that maps each class to one prompt and one latent seed.

```bash
python scripts/build_t2ue_class_manifest.py \
  --manifest-csv /path/to/manifest.csv \
  --annotation-csv /path/to/identity_metadata.csv \
  --prompt-template "A photo of {class_name}" \
  --master-seed 0 \
  --out-class-manifest /path/to/class_manifest.json
```

Notes:
- The builder only uses identities present in the passed manifest CSV.
- Annotation names are matched by identity key overlap and cleaned before templating.
- If an identity is missing in annotation CSV, `raw_id` is used as fallback `class_name`.
- Prompt placeholders: `{class_id}`, `{raw_id}`, `{class_name}`.

### Single-pass run (delta generation + apply)

```bash
python scripts/generate_only_t2ue.py \
  --class-manifest /path/to/class_manifest.json \
  --samples-csv /path/to/samples.csv \
  --t2ue-ckpt /path/to/generator_epoch0500.pt \
  --out-delta-dir /path/to/delta_cache \
  --out-images-dir /path/to/poisoned/images \
  --out-poison-map /path/to/poison_map.csv \
  --clip-model "ViT-B/32" \
  --batch-size 32 \
  --num-workers 0 \
  --image-format png \
  --input-size 112 \
  --device cuda:0
```

`samples.csv` must include:
- `clean_path`
- `label`
- optional `poisoned_rel_path`

Output path convention matches REM/CUDA:
- If `poisoned_rel_path` exists, it is preserved (including identity subfolders).
- If missing, output falls back to `<basename>.<image_format>`.

Notes:
- Default behavior regenerates all class deltas each run.
- To reuse valid cache entries (only when metadata matches), pass `--skip-regenerate-deltas`.

