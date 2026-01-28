from torchvision import transforms

# OpenAI CLIP uses 224x224 and specific normalization; the official clip package provides
# preprocess, but we keep it explicit and stable here.
# can be found under clip/clip.py :> _transform(n_px):
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

def build_clip_image_transform(out_res: int = 224):
    return transforms.Compose([
        transforms.Resize(out_res, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(out_res),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])
