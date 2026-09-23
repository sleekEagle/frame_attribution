"""
models/r3d_ucf101.py -- uniform-interface wrapper around func.UCF101_data_model's ResNet3D-50
(finetuned on UCF101, models/r3d/ckpt/save_200.pth), so it drops into the same get_model() /
predict_from_path() pattern as models/ssv2.py (VJEPA2), models/trn.py (TRN) and
models/trn_official.py (TRNOfficial):

    label2id                        class name (str) -> class index (int)
    predict_from_path(path)         -> predicted class index (int)
    predict_from_batch_path(paths)  -> (N, 101) logits tensor

Like the other three, "path" is a video file (CONST.UCF101_PATH's .avi clips) -- FRAME_COUNT
frames are sampled directly from it (models/video_utils.sample_frames, the same
TSN/TRN-style "centre of each of N equal segments" protocol trn.py/trn_official.py use), then
run through the same PIL-based spatial_transform the checkpoint was trained with
(func.UCF101_data_model.transform: Resize -> CenterCrop -> ToTensor -> Normalize).
"""
import torch
import torchvision.transforms.functional as tvF

from func import UCF101_data_model
from models.video_utils import sample_frames


class R3DUCF101:
    FRAME_COUNT = 16  # models/r3d/ucf101.json's sample_duration

    def __init__(self):
        self._dm = UCF101_data_model()
        self.model = self._dm.model
        self.device = next(self.model.parameters()).device
        self.transform = self._dm.transform
        self.label2id = {name: idx for idx, name in self._dm.inference_class_names.items()}

    def eval(self):
        self.model.eval()
        return self

    def preprocess(self, frames) -> torch.Tensor:
        """frames: (T,C,H,W) uint8 RGB -> (1,3,T,H,W) float, model's own normalization."""
        clip = torch.stack([self.transform(tvF.to_pil_image(f)) for f in frames])  # (T,3,H,W)
        return clip.permute(1, 0, 2, 3).unsqueeze(0)

    def video_from_path(self, path) -> torch.Tensor:
        return self.preprocess(sample_frames(path, self.FRAME_COUNT))

    def predict_from_path(self, path):
        with torch.no_grad():
            logits = self.model(self.video_from_path(path).to(self.device))
        return logits.argmax(-1).item()

    def predict_from_batch_path(self, paths, bs: int = 32) -> torch.Tensor:
        chunks = [paths[i:i + bs] for i in range(0, len(paths), bs)]
        out = torch.empty(0)
        for i, chunk in enumerate(chunks):
            print(f"Processing batch {(i + 1) / len(chunks):.2%}", end="\r")
            clips = torch.cat([self.video_from_path(p) for p in chunk], dim=0)
            with torch.no_grad():
                logits = self.model(clips.to(self.device))
            out = torch.cat([out, logits.cpu()], dim=0)
        return out
