from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ssacn import SSACNBlock

@dataclass
class GenConfig:
    z_dim: int = 128
    text_dim: int = 512
    base_ch: int = 512  # start feature channels at 512
    out_res: int = 224
    eps: float = 8.0 / 255.0

class T2UEGenerator(nn.Module):
    """
    T2UE generator G adapted from SSA-GAN style as described in §3.1 and Fig.3. 
    """
    def __init__(self, cfg: GenConfig):
        super().__init__()
        self.cfg = cfg

        # FC -> 4x4x512 initial map (paper figure). 
        self.fc = nn.Linear(cfg.z_dim, cfg.base_ch * 4 * 4)

        # Upsampling blocks: 4->8->16->32->64->128->256 then crop/resize to 224.
        chs = [cfg.base_ch, 256, 128, 64, 32, 32]  # 5 upsample blocks => 4*2^5 = 128, add 1 more => 256
        blocks = []
        in_ch = chs[0]
        for out_ch in chs[1:]:
            blocks.append(SSACNBlock(in_ch, out_ch, cfg.text_dim))
            in_ch = out_ch
        # One more block to reach 256
        blocks.append(SSACNBlock(in_ch, 32, cfg.text_dim))
        self.blocks = nn.ModuleList(blocks)

        self.to_rgb = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, emb_t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        emb_t: (B, text_dim)
        z:     (B, z_dim)
        returns delta: (B,3, out_res,out_res) in CLIP-normalized space (same scale as input tensor),
        bounded in L_inf by eps after tanh scaling + clamp.
        """
        B = z.shape[0]
        h = self.fc(z).view(B, self.cfg.base_ch, 4, 4) # seed feature map

        # Upsampling blocks with SSACN conditioning
        for blk in self.blocks:
            h = blk(h, emb_t)

        # now roughly 256x256; project to RGB and bound
        delta = torch.tanh(self.to_rgb(h))  # [-1,1]
        delta = self.cfg.eps * delta        # [-eps, eps]

        # Resize/crop to CLIP resolution (224) if needed
        if delta.shape[-1] != self.cfg.out_res:
            delta = F.interpolate(delta, size=(self.cfg.out_res, self.cfg.out_res), mode="bilinear", align_corners=False)

        # Hard L_inf safety clamp (belt + suspenders)
        delta = torch.clamp(delta, -self.cfg.eps, self.cfg.eps)
        return delta
