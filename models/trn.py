"""
models/trn.py -- Multiscale TRN (Zhou et al., NeurIPS'18), BN-Inception backbone, trained on
Something-Something v2. This is the *exact* pretrained checkpoint Play Fair [27] (Price & Damen,
ACCV'20 -- the closest baseline the WACV reviewers asked for a direct comparison against)
benchmarks its ESV attribution on. It ships inside the play-fair/ submodule already vendored in
this repo (play-fair/checkpoints/download.sh), we just need to download the two checkpoint files
and wire them up without the rest of play-fair's old (Python 3.7 / pydantic-1 / gulpio) stack.

Mirrors the interface of models/ssv2.py's VJEPA2 class (label2id, sample_frames,
predict_from_path, predict_from_batch_path) so it drops into the same evaluation pattern.

frame_count is fixed at 8: the temporal MLP's input dim (256 * 8) is baked into the
trn_8_frames.pth checkpoint, so this model always samples exactly 8 frames per clip
(TemporalSegmentSampler, test_mode=True -- same protocol the original TRN-pytorch/TSN
papers evaluate with: uniformly divide the clip into 8 segments, take each segment's centre
frame).

Setup (run once, from the frame_attribution folder, in your normal conda env):
    python play-fair/checkpoints/download_trn.py

Usage:
    from models.trn import TRN
    model = TRN()
    model.predict_from_path("some_video.webm")
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.video_utils import sample_frames

HERE = Path(__file__).resolve().parent.parent  # frame_attribution/
PLAYFAIR_SRC = HERE / "play-fair" / "src"
if not PLAYFAIR_SRC.exists():
    raise FileNotFoundError(
        f"Could not find {PLAYFAIR_SRC}. Expected the play-fair submodule to be checked "
        "out at frame_attribution/play-fair (it already is in this repo -- if it's missing, "
        "run `git submodule update --init` or re-clone)."
    )
if str(PLAYFAIR_SRC) not in sys.path:
    sys.path.insert(0, str(PLAYFAIR_SRC))

# NOTE: we deliberately import these three leaf modules directly instead of going through
# play-fair's models.builder / config.model. Those pull in (a) config/model.py, which needs
# pydantic 1.x (`Field(..., const=True)` was removed in pydantic 2), and (b)
# models/backbones/resnet.py, which imports `torchvision.models.utils.load_state_dict_from_url`
# -- removed in modern torchvision. Neither is needed for BN-Inception TRN, so importing around
# them keeps this usable in a normal, current-day env with no extra installs beyond what
# models/ssv2.py (VJEPA2) already needs.
#
# We can't just `from models.backbones... import ...`: this file is itself `models/trn.py`,
# so `models` is already bound in sys.modules to *this* package (frame_attribution/models)
# by the time this runs, and `models.backbones` would resolve there instead of under
# play-fair/src -- there is no frame_attribution/models/backbones. Load play-fair's
# `models` package under an alias instead, so it can't collide.
import importlib.util  # noqa: E402


def _load_playfair_models():
    alias = "playfair_models"
    if alias in sys.modules:
        return sys.modules[alias]
    pkg_dir = PLAYFAIR_SRC / "models"
    spec = importlib.util.spec_from_file_location(
        alias, pkg_dir / "__init__.py", submodule_search_locations=[str(pkg_dir)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


_load_playfair_models()
bninception = importlib.import_module("playfair_models.backbones.bninception").bninception
replace_last_linear = importlib.import_module("playfair_models.utils").replace_last_linear
MLPConsensus = importlib.import_module("playfair_models.components.mlp").MLPConsensus
AggregatedBackboneModel = importlib.import_module(
    "playfair_models.aggregated_backbone_model").AggregatedBackboneModel

CLASSES_CSV = PLAYFAIR_SRC / "datasets" / "metadata" / "something_something_v2" / "classes.csv"
BACKBONE_CKPT = HERE / "play-fair" / "checkpoints" / "backbones" / "trn.pth"
TEMPORAL_CKPT = HERE / "play-fair" / "checkpoints" / "features" / "trn_8_frames.pth"


def _load_state_dict(module: nn.Module, path: Path) -> None:
    ckpt = torch.load(str(path), map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt.get("model", ckpt)
    module.load_state_dict(state_dict)


class TRN(nn.Module):
    FRAME_COUNT = 8
    INPUT_SIZE = 224
    # bninception's "imagenet" pretrained_settings mean/std (BGR, no /255), reused as-is by
    # play-fair/configs/trn_bninception.jsonnet for the SSv2-trained model.
    MEAN_BGR = (104.0, 117.0, 128.0)

    def __init__(self, backbone_checkpoint=BACKBONE_CKPT, temporal_checkpoint=TEMPORAL_CKPT,
                 class_count=174, device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        for p, what, url in [
            (Path(backbone_checkpoint), "backbone (trn.pth)",
             "https://www.dropbox.com/s/u9ajcv13ndljo8i/trn.pth?dl=1"),
            (Path(temporal_checkpoint), "temporal module (trn_8_frames.pth)",
             "https://www.dropbox.com/s/5xxkjtrk4m0ogy0/trn_8_frames.pth?dl=1"),
        ]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing TRN {what} at {p}.\n"
                    f"Download it, e.g.: python play-fair/checkpoints/download_trn.py\n"
                    f"(or manually: {url} -> {p})"
                )

        backbone = bninception(num_classes=1000, pretrained=False)
        replace_last_linear(backbone, 256)  # -> 256-d frame feature, matches backbone_dim
        temporal_module = MLPConsensus(
            input_dim=256 * self.FRAME_COUNT, hidden_dim=256, output_dim=class_count,
            hidden_layers=1, dropout=0.7, batch_norm=False,
        )
        _load_state_dict(backbone, Path(backbone_checkpoint))
        _load_state_dict(temporal_module, Path(temporal_checkpoint))
        self.model = AggregatedBackboneModel(backbone, temporal_module).eval().to(self.device)

        self.label2id = {}
        with open(CLASSES_CSV, newline="") as f:
            for row in csv.DictReader(f):
                self.label2id[row["name"]] = int(row["id"])
        assert len(self.label2id) == class_count, (
            f"Expected {class_count} SSv2 classes in {CLASSES_CSV}, found {len(self.label2id)}"
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 8, 3, 224, 224) preprocessed BGR clips -> (N, 174) logits."""
        return self.model(x.to(self.device))

    def sample_frames(self, video_path) -> torch.Tensor:
        return sample_frames(video_path, self.FRAME_COUNT)  # (T,C,H,W) uint8 RGB

    def preprocess(self, frames) -> torch.Tensor:
        """frames: (T,C,H,W) or (T,H,W,C) uint8 RGB -> (T,3,224,224) float, BGR, mean-subtracted."""
        x = frames if torch.is_tensor(frames) else torch.from_numpy(np.asarray(frames))
        x = x.float()
        if x.shape[-1] == 3 and x.shape[1] != 3:  # (T,H,W,C) -> (T,C,H,W)
            x = x.permute(0, 3, 1, 2)
        x = x.flip(1)  # RGB -> BGR (channel dim = 1)
        h, w = x.shape[-2:]
        scale = (self.INPUT_SIZE / 0.875) / min(h, w)  # resize short side to 256
        nh, nw = max(self.INPUT_SIZE, round(h * scale)), max(self.INPUT_SIZE, round(w * scale))
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
        top, left = (nh - self.INPUT_SIZE) // 2, (nw - self.INPUT_SIZE) // 2
        x = x[:, :, top:top + self.INPUT_SIZE, left:left + self.INPUT_SIZE]
        mean = torch.tensor(self.MEAN_BGR, dtype=x.dtype).view(1, 3, 1, 1)
        return x - mean

    def video_from_path(self, path) -> torch.Tensor:
        frames = self.sample_frames(path)
        return self.preprocess(frames).unsqueeze(0)  # (1,8,3,224,224)

    def predict_from_path(self, path) -> int:
        with torch.no_grad():
            logits = self.forward(self.video_from_path(path))
        return logits.argmax(-1).item()

    def predict_from_batch_path(self, paths, bs: int = 32) -> torch.Tensor:
        chunks = [paths[i:i + bs] for i in range(0, len(paths), bs)]
        out = torch.empty(0)
        for i, chunk in enumerate(chunks):
            print(f"Processing batch {(i + 1) / len(chunks):.2%}", end="\r")
            clips = torch.cat([self.video_from_path(p) for p in chunk], dim=0)
            with torch.no_grad():
                logits = self.forward(clips)
            out = torch.cat([out, logits.cpu()], dim=0)
        return out
