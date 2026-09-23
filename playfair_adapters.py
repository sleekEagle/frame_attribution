"""
playfair_adapters.py -- glue between run_playfair.py and the models used in the WACV paper.

Use with run_playfair.py's custom-adapter hook:

    python run_playfair.py <video.avi> --adapter playfair_adapters:build_r3d_ucf101 \
        --frame-indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --approximate

    python run_playfair.py <video.webm> --adapter playfair_adapters:build_vjepa2_ssv2 --num-frames 8

build_* functions are called as fn(device=<torch.device>, **kwargs) and must return an
object with:  preprocess(frames uint8 (T,H,W,3)) -> (T,C,H',W'),
              forward_clips(clips (B,S,C,H',W')) -> logits (B,K),
              num_classes, class_names.

Everything the models need (architecture, checkpoint, normalisation, class order) is read from
the same files the rest of frame_attribution uses, so numbers are comparable to the paper.
"""
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


class R3D50UCF101Adapter:
    """Hara et al. 3D-ResNet-50 fine-tuned on UCF101 (models/r3d/ckpt/save_200.pth).

    Pre-processing reproduces get_inference_utils() in models/resnet3d/main.py:
    Resize(112) [short side] -> CenterCrop(112) -> ToTensor -> Normalize(mean, std).
    The network ends in adaptive avg-pooling, so clips with any number of frames >= 1 run
    (needed because Play Fair evaluates coalitions of frames).
    """

    def __init__(self, device, config="models/r3d/ucf101.json", checkpoint=None,
                 annotation="dataloaders/ucf101/ucf101_01.json", random_weights=False):
        from PIL import Image  # noqa: F401
        from torchvision import transforms as T

        cfg = json.loads((HERE / config).read_text())
        opt = Namespace(**cfg)

        sys.path.insert(0, str(HERE / "models" / "resnet3d"))  # model.py does `from res_models import ...`
        from models.resnet3d.model import generate_model  # noqa: E402

        opt.device = torch.device("cpu")
        opt.distributed = False
        model = generate_model(opt)
        if not random_weights:
            ckpt_path = Path(checkpoint) if checkpoint else HERE / "models" / "r3d" / "ckpt" / "save_200.pth"
            ck = torch.load(ckpt_path, map_location="cpu")
            assert ck["arch"] == opt.arch, (ck["arch"], opt.arch)
            (model.module if hasattr(model, "module") else model).load_state_dict(ck["state_dict"])
        else:
            print("[adapter] WARNING: random weights (smoke test only)")
        self.model = (model.module if hasattr(model, "module") else model).eval().to(device)
        self.device = device

        labels = json.loads((HERE / annotation).read_text())["labels"]
        self.class_names = sorted(labels) if isinstance(labels, list) else labels
        self.num_classes = int(opt.n_classes)
        assert len(self.class_names) == self.num_classes
        self.tf = T.Compose([T.Resize(opt.sample_size), T.CenterCrop(opt.sample_size), T.ToTensor(),
                             T.Normalize(opt.mean, opt.std)])

    def preprocess(self, frames: np.ndarray) -> torch.Tensor:
        from PIL import Image
        return torch.stack([self.tf(Image.fromarray(f)) for f in frames])  # (T,3,112,W')

    @torch.no_grad()
    def forward_clips(self, clips: torch.Tensor) -> torch.Tensor:
        return self.model(clips.permute(0, 2, 1, 3, 4))  # (B,3,S,H,W)


def build_r3d_ucf101(device, **kw):
    return R3D50UCF101Adapter(device, **{k: (v if k != "random_weights" else v in ("1", "true", "True"))
                                         for k, v in kw.items()})


def build_vjepa2_ssv2(device, **kw):
    """V-JEPA 2 ViT-L SSv2 classifier (same HF checkpoint as models/ssv2.py). Reuses the
    adapter in run_playfair.py (handles the tubelet-size-2 constraint for odd coalitions)."""
    import run_playfair
    return run_playfair.VJEPA2Adapter(device, **kw)
