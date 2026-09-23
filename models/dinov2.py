"""
models/dinov2.py -- per-frame image embedding extractor using Meta's DINOv2 (self-supervised
ViT), used to embed individual frames for adjacent-frame similarity clustering (see
frame_clustering.py). Unlike the other wrappers in this package, this isn't a classifier
plugged into models/registry.py's get_model()/predict_from_path() pattern (there's no
label2id / class to predict) -- it's a plain image encoder run independently per frame, since
DINOv2 has no notion of a "clip" the way the video models do.
"""
import torch
import torchvision.transforms.functional as tvF
from transformers import AutoImageProcessor, AutoModel

CHECKPOINT = "facebook/dinov2-small"


class DINOv2:
    def __init__(self, checkpoint: str = CHECKPOINT, device=None):
        self.processor = AutoImageProcessor.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    @torch.no_grad()
    def embed_frames(self, frames) -> torch.Tensor:
        """frames: (T,C,H,W) uint8 RGB -> (T, D) CLS-token embeddings, one per frame."""
        pil_frames = [tvF.to_pil_image(f) for f in frames]
        pixel_values = self.processor(pil_frames, return_tensors="pt")["pixel_values"]
        out = self.model(pixel_values.to(self.device))
        return out.last_hidden_state[:, 0].cpu()  # (T, D) CLS token per frame
