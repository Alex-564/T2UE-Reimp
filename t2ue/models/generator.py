from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ssacn import SSACNBlock

CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

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

        # 7 SSACN blocks total (paper): one 4x4 refinement block + six upsampling blocks to 256x256.
        # Spatial path: 4 -> 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256.
        chs = [cfg.base_ch, 256, 128, 64, 32, 32, 32]
        self.blocks = nn.ModuleList()
        self.blocks.append(SSACNBlock(cfg.base_ch, cfg.base_ch, cfg.text_dim, upsample=False))
        in_ch = cfg.base_ch
        for out_ch in chs[1:]:
            self.blocks.append(SSACNBlock(in_ch, out_ch, cfg.text_dim, upsample=True))
            in_ch = out_ch

        self.to_rgb = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.register_buffer(
            "clip_std",
            torch.tensor(CLIP_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

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

        # Project to RGB and constrain perturbation.
        delta = torch.tanh(self.to_rgb(h))  # [-1,1]

        # Enforce ||delta_pixel||_inf <= eps while operating in CLIP-normalized space:
        # delta_pixel = delta_norm * std  =>  |delta_norm_c| <= eps / std_c.
        norm_eps = self.cfg.eps / self.clip_std.to(dtype=delta.dtype)
        delta = delta * norm_eps

        # Resize/crop to CLIP resolution (224) if needed
        if delta.shape[-1] != self.cfg.out_res:
            delta = F.interpolate(delta, size=(self.cfg.out_res, self.cfg.out_res), mode="bilinear", align_corners=False)

        # Hard per-channel L_inf safety clamp in normalized space.
        delta = torch.max(torch.min(delta, norm_eps), -norm_eps)
        return delta
