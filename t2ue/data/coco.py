import random
from typing import Tuple, List
from torchvision.datasets import CocoCaptions

class CocoCaptionPairs(CocoCaptions):
    """
    Returns (image_tensor, caption_string).
    torchvision CocoCaptions returns (PIL, [captions]).
    We sample one caption (random) each time for diversity, which matches the papers
    premise of learning text-semantic guidance. 
    """
    def __init__(self, root: str, annFile: str, transform=None):
        super().__init__(root=root, annFile=annFile, transform=transform)

    def __getitem__(self, idx: int) -> Tuple[object, str]:
        img, captions = super().__getitem__(idx)  # img already transformed if transform provided
        assert isinstance(captions, list) and len(captions) > 0
        cap = random.choice(captions)
        return img, cap
