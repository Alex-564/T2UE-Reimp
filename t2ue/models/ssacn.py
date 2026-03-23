import torch
import torch.nn as nn
import torch.nn.functional as F
from .sscbn import SSCBN

class SSACNBlock(nn.Module):
    """
    SSACN block: upsample + convs + SSCBN(text) with residual connection.
    The paper: "sequential application of SSACN blocks, incorporating residual connections
    and upsampling operations".
    """
    def __init__(self, in_ch: int, out_ch: int, text_dim: int, upsample: bool = True):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.upsample = upsample

        # text_dim: CLIP text embedding dim (512 for ViT-B/32)

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = SSCBN(out_ch, text_dim)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = SSCBN(out_ch, text_dim)

        # SSCBN after each conv ensures conditioning is pervasive

        self.skip = None
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, emb_t: torch.Tensor) -> torch.Tensor:
        # Backbone: takes from 4x4 seed map to full res perturbation

        # Optional upsample (nearest is common in GAN generators).
        x_up = x
        if self.upsample:
            x_up = F.interpolate(x, scale_factor=2.0, mode="nearest")

        # Residual skip path
        res = x_up
        if self.skip is not None:
            res = self.skip(res)

        # Main path
        h = self.conv1(x_up)
        h = self.bn1(h, emb_t)
        h = F.relu(h, inplace=True)

        h = self.conv2(h)
        h = self.bn2(h, emb_t)

        out = F.relu(h + res, inplace=True)
        return out
