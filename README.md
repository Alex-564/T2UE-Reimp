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

Set paths in configs/*.yaml

## Train generator (offline)
python scripts/train_generator.py --config configs/vitb32_coco.yaml
python scripts/train_generator.py --config configs/vitb16_coco.yaml

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

