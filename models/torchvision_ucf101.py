"""
models/torchvision_ucf101.py -- uniform-interface wrapper for torchvision 3D-CNN video models
fine-tuned on UCF101 split 1 and published by dronefreak/human-action-classification on the
Hugging Face Hub:

    mc3_18   dronefreak/mc3-18-ucf101   MC3-18 (3D convolutions in the first stage, convolutions
                                        over space only afterwards; no temporal downsampling)
    r3d_18   dronefreak/r3d-18-ucf101   R3D-18 (3D ResNet-18; stages 2-4 halve time)

Both: Kinetics-400 pretraining, 101 classes, 16 frames, 112 px. Preprocessing follows the
publisher's inference code (src/hac/video/inference/predictor.py): frames sampled with
linspace(0, n-1, 16), Resize((128, 171)) -> CenterCrop(112) -> ToTensor -> Normalize with the
Kinetics mean/std.

Class order: the checkpoints use Python's sorted() order of the class folder names (the
publisher's training dataset), NOT the classInd.txt order its predictor uses. The two differ only
at HammerThrow/Hammering and JumpRope/JumpingJack. On the 3,782 readable split-1 test videos,
sorted order gives 86.83% (MC3-18) and 82.95% (R3D-18), matching the reported 87.05% / 83.43%;
classInd order gives 0% on those four classes. `class_order` defaults to "sorted".

    label2id                        class name (str) -> class index (int)
    predict_from_path(path)         -> predicted class index (int)
"""
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as tvF
from huggingface_hub import hf_hub_download
from torchcodec.decoders import VideoDecoder

import CONST

CHECKPOINTS = {
    "mc3_18": ("dronefreak/mc3-18-ucf101", "mc318-ufc101-split-1.pth"),
    "r3d_18": ("dronefreak/r3d-18-ucf101", "r3d18-ufc101-split-1.pth"),
}
MEAN, STD = [0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989]


def ucf101_class_names(order: str = "classind"):
    classind = [line.split()[1] for line in open(Path(CONST.UCF101_SPLITS_PATH) / "classInd.txt")
                if line.strip()]
    return classind if order == "classind" else sorted(classind)


class TorchvisionUCF101:
    FRAME_COUNT = 16

    def __init__(self, arch: str = "mc3_18", class_order: str = "sorted"):
        repo, fname = CHECKPOINTS[arch]
        self.arch = arch
        self.model = getattr(torchvision.models.video, arch)(weights=None, num_classes=101)
        # published training checkpoint (optimizer state etc. included), hence weights_only=False
        ckpt = torch.load(hf_hub_download(repo, fname), map_location="cpu", weights_only=False)
        state = {k.removeprefix("backbone."): v for k, v in ckpt["model_state_dict"].items()}
        self.model.load_state_dict(state, strict=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        self.label2id = {name: i for i, name in enumerate(ucf101_class_names(class_order))}
        self.transform = T.Compose([T.Resize((128, 171), antialias=True), T.CenterCrop(112),
                                    T.Normalize(MEAN, STD)])

    def eval(self):
        self.model.eval()
        return self

    @staticmethod
    def frame_indices(n_total: int, n: int):
        """The publisher's sampling: n frames at linspace(0, n_total - 1, n)."""
        return np.linspace(0, n_total - 1, n).astype(int).tolist()

    def preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (T, C, H, W) uint8 RGB -> (1, 3, T, 112, 112) float."""
        x = self.transform(frames.float() / 255.0)  # transforms apply per frame on (T, C, H, W)
        return x.permute(1, 0, 2, 3).unsqueeze(0)

    def video_from_path(self, path) -> torch.Tensor:
        decoder = VideoDecoder(str(path))
        frames = decoder.get_frames_at(indices=self.frame_indices(len(decoder), self.FRAME_COUNT)).data
        return self.preprocess(frames)

    def predict_from_path(self, path):
        with torch.no_grad():
            logits = self.model(self.video_from_path(path).to(self.device))
        return logits.argmax(-1).item()


class MC3_18UCF101(TorchvisionUCF101):
    def __init__(self, class_order: str = "sorted"):
        super().__init__("mc3_18", class_order)


class R3D18UCF101(TorchvisionUCF101):
    def __init__(self, class_order: str = "sorted"):
        super().__init__("r3d_18", class_order)
