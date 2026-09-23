"""
models/videomae_ucf101.py -- uniform-interface wrapper around Hugging Face's
nateraw/videomae-base-finetuned-ucf101 (a VideoMAE ViT-based video transformer, finetuned on
the full 101-class UCF101), so it drops into the same get_model() / predict_from_path() pattern
as models/r3d_ucf101.py, models/trn.py, models/trn_official.py and models/ssv2.py -- a
different architecture family (transformer vs. R3D's 3D CNN) evaluated on the same
dataset/paths as r3d.

    label2id                        class name (str) -> class index (int)
    predict_from_path(path)         -> predicted class index (int)
    predict_from_batch_path(paths)  -> (N, 101) logits tensor

"path" is a video file (CONST.UCF101_PATH's .avi clips), matching r3d_ucf101.py -- FRAME_COUNT
frames are sampled directly from it (models/video_utils.sample_frames, the same TSN/TRN-style
"centre of each of N equal segments" protocol trn.py/trn_official.py/r3d_ucf101.py use), then
run through the checkpoint's own VideoMAEImageProcessor.
"""
import torch
import torchvision.transforms.functional as tvF
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from models.video_utils import sample_frames

CHECKPOINT = "nateraw/videomae-base-finetuned-ucf101"


class VideoMAEUCF101:
    def __init__(self):
        self.processor = VideoMAEImageProcessor.from_pretrained(CHECKPOINT)
        self.model = VideoMAEForVideoClassification.from_pretrained(CHECKPOINT)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.FRAME_COUNT = self.model.config.num_frames
        self.label2id = self.model.config.label2id

    def eval(self):
        self.model.eval()
        return self

    def preprocess(self, frames) -> torch.Tensor:
        """frames: (T,C,H,W) uint8 RGB -> (1,T,3,H,W) pixel_values, processor's own
        normalization."""
        pil_frames = [tvF.to_pil_image(f) for f in frames]
        return self.processor(pil_frames, return_tensors="pt")["pixel_values"]

    def video_from_path(self, path) -> torch.Tensor:
        return self.preprocess(sample_frames(path, self.FRAME_COUNT))

    def predict_from_path(self, path):
        with torch.no_grad():
            logits = self.model(self.video_from_path(path).to(self.device)).logits
        return logits.argmax(-1).item()

    def predict_from_batch_path(self, paths, bs: int = 32) -> torch.Tensor:
        chunks = [paths[i:i + bs] for i in range(0, len(paths), bs)]
        out = torch.empty(0)
        for i, chunk in enumerate(chunks):
            print(f"Processing batch {(i + 1) / len(chunks):.2%}", end="\r")
            clips = torch.cat([self.video_from_path(p) for p in chunk], dim=0)
            with torch.no_grad():
                logits = self.model(clips.to(self.device)).logits
            out = torch.cat([out, logits.cpu()], dim=0)
        return out
