import torch
import torch.nn.functional as F

def symmetric_infonce(image_emb: torch.Tensor, text_emb: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """
    image_emb: (B, D) L2-normalized
    text_emb:  (B, D) L2-normalized

    Note: Assumes image_emb and text_emb are already L2-normalized.
    """
    B = image_emb.shape[0]
    
    # Similarity Matrix
    # Transposes text matrix to (D, B)
    # Matrix multiplication results in (B, B) similarity scores
    # Temperature scaling with tau
    logits = (image_emb @ text_emb.t()) / tau

    # Ground truth labels
    labels = torch.arange(B, device=logits.device)

    # Symmetric InfoNCE loss
    # Robust to both image-to-text and text-to-image retrieval
    # Standard for multi-modal representation learning
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)

    # Average the two losses
    return 0.5 * (loss_i2t + loss_t2i)
