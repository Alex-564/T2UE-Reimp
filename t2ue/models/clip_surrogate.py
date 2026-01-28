from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip  # OpenAI CLIP

class OpenAIClipSurrogate(nn.Module):
    """
    Frozen CLIP (f_I, f_T) surrogate as in the paper. 
    """
    def __init__(self, model_name: str, device: torch.device):
        super().__init__()
        self.model, _ = clip.load(model_name, device=device, jit=False)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.device = device

    @torch.no_grad()
    def tokenize(self, texts: List[str]) -> torch.Tensor:
        return clip.tokenize(texts, truncate=True).to(self.device)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Returns L2-normalized text embeddings.
        Note: CLIP params are frozen; grads are not needed for them anyway.
        """
        tokens = self.tokenize(texts)  # no_grad
        txt = self.model.encode_text(tokens)
        return F.normalize(txt.float(), dim=-1)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        images are already CLIP-normalized tensors (B,3,224,224).
        IMPORTANT: do NOT wrap this in no_grad; we need gradients w.r.t. images
        so loss can train the generator.
        """
        img = self.model.encode_image(images)
        return F.normalize(img.float(), dim=-1)
