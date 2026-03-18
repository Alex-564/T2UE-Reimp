import torch
import torch.nn as nn

class SSCBN(nn.Module):
    """
    Semantic-Space Conditional BatchNorm as defined in Eq.(2). 
    Predicts gamma and beta from text embeddings -> prompt driven generation.

    """
    def __init__(self, num_features: int, text_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.bn = nn.BatchNorm2d(num_features, affine=False, eps=eps)
        # affline = false, disable internal learned gamma/beta (supplied through text embeddings)

        # Small MLPs mapping emb_t -> gamma/beta (per-channel)
        # Paper Eq.(2): h' = gamma(emb_t) * BN(h) + beta(emb_t)
        self.gamma = nn.Sequential(
            nn.Linear(text_dim, num_features),
        )
        self.beta = nn.Sequential(
            nn.Linear(text_dim, num_features),
        )

        # init: start close to identity transform
        nn.init.zeros_(self.gamma[0].weight)
        nn.init.ones_(self.gamma[0].bias)
        nn.init.zeros_(self.beta[0].weight)
        nn.init.zeros_(self.beta[0].bias)

    def forward(self, x: torch.Tensor, emb_t: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,H,W)
        emb_t: (B,text_dim)
        """
        h = self.bn(x)

        gamma = self.gamma(emb_t).unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
        beta = self.beta(emb_t).unsqueeze(-1).unsqueeze(-1)    # (B,C,1,1)

        return gamma * h + beta
