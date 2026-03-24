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

