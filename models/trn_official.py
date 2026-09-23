"""
models/trn_official.py -- the ORIGINAL TRN checkpoint from the paper authors
(Zhou et al., "Temporal Relational Reasoning in Videos", ECCV 2018,
https://github.com/zhoubolei/TRN-pytorch), used as a fallback since play-fair's own
repackaged checkpoint (play-fair/checkpoints/download.sh, Dropbox) is dead ("File
Deleted").

Architecture (replicated from TRN-pytorch's models.py + TRNmodule.py, since that repo
targets Python 2-era torch/torchvision and can't be imported as-is):
    BNInception backbone (play-fair's copy: same Caffe-derived layer names, since both
    ultimately descend from the same yjxiong/Cadene BN-Inception port -- see the PR
    "Add BN-Inception by yjxiong" on Cadene/tensorflow-model-zoo.torch) -- global-pooled
    1024-d feature per frame, classifier head removed (dropout->identity at eval time)
      -> new_fc: Linear(1024, 256)               ("img_feature_dim" in the original code)
      -> RelationModuleMultiScale(256, 8, 174)    (TRNmultiscale consensus, verbatim port
                                                    of TRNmodule.py's class of the same name)

Checkpoint: trained on Something-Something **v1** (174 classes, same taxonomy as v2).
See download_trn_official.py for where to get it and why v1-not-v2.

NOTE on determinism: RelationModuleMultiScale randomly subsamples 3 relation-subsets per
scale (all scales except the full 8-frame one) on every forward call -- this is the
original authors' design, present at test time too, not a bug here. Expect the reported
accuracy to vary by a fraction of a point between runs.
"""
import importlib
import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.video_utils import sample_frames

HERE = Path(__file__).resolve().parent.parent  # frame_attribution/
PLAYFAIR_SRC = HERE / "play-fair" / "src"
if str(PLAYFAIR_SRC) not in sys.path:
    sys.path.insert(0, str(PLAYFAIR_SRC))

# `from models.backbones... import ...` would resolve against *this* package
# (frame_attribution/models, already bound to the name "models") instead of
# play-fair/src/models -- there is no frame_attribution/models/backbones. Load
# play-fair's `models` package under an alias to avoid the collision (see
# models/trn.py, which has the same issue).


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

BACKBONE_CKPT = HERE / "play-fair" / "checkpoints" / "backbones" / \
    "TRN_something_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar"
CATEGORIES_TXT = HERE / "play-fair" / "checkpoints" / "backbones" / "something_categories.txt"


# --------------------------------------------------------------------------- #
# Verbatim port of TRNmodule.py's RelationModuleMultiScale (Zhou et al.), the
# "TRNmultiscale" consensus module used by this checkpoint's `consensus.*` weights.
# --------------------------------------------------------------------------- #
class RelationModuleMultiScale(nn.Module):
    def __init__(self, img_feature_dim, num_frames, num_class):
        super().__init__()
        self.subsample_num = 3
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)]

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale)))

        self.num_class = num_class
        self.num_frames = num_frames
        num_bottleneck = 256
        self.fc_fusion_scales = nn.ModuleList()
        for scale in self.scales:
            fc_fusion = nn.Sequential(
                nn.ReLU(),
                nn.Linear(scale * self.img_feature_dim, num_bottleneck),
                nn.ReLU(),
                nn.Linear(num_bottleneck, self.num_class),
            )
            self.fc_fusion_scales.append(fc_fusion)

    def forward(self, input):
        act_all = input[:, self.relations_scales[0][0], :]
        act_all = act_all.view(act_all.size(0), self.scales[0] * self.img_feature_dim)
        act_all = self.fc_fusion_scales[0](act_all)

        for scale_id in range(1, len(self.scales)):
            idx_relations_randomsample = np.random.choice(
                len(self.relations_scales[scale_id]), self.subsample_scales[scale_id], replace=False)
            for idx in idx_relations_randomsample:
                act_relation = input[:, self.relations_scales[scale_id][idx], :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scale_id] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scale_id](act_relation)
                act_all += act_relation
        return act_all

    @staticmethod
    def return_relationset(num_frames, num_frames_relation):
        return list(itertools.combinations(range(num_frames), num_frames_relation))


class TRNOfficial(nn.Module):
    FRAME_COUNT = 8
    INPUT_SIZE = 224
    IMG_FEATURE_DIM = 256
    MEAN_BGR = (104.0, 117.0, 128.0)

    def __init__(self, backbone_checkpoint=BACKBONE_CKPT, categories_file=CATEGORIES_TXT,
                 class_count=174, device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        for p, what in [(Path(backbone_checkpoint), "checkpoint"), (Path(categories_file), "categories file")]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing TRN (official) {what} at {p}.\n"
                    "Download with: python play-fair/checkpoints/download_trn_official.py"
                )

        backbone = bninception(num_classes=1000, pretrained=False)
        backbone.last_linear = nn.Identity()  # classifier removed; dropout(eval)==identity anyway, no params either way
        new_fc = nn.Linear(1024, self.IMG_FEATURE_DIM)
        consensus = RelationModuleMultiScale(self.IMG_FEATURE_DIM, self.FRAME_COUNT, class_count)

        ckpt = torch.load(str(backbone_checkpoint), map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        # original keys look like "module.base_model.conv1_7x7_s2.weight" / "module.new_fc.weight" /
        # "module.consensus.fc_fusion_scales.0.1.weight" -- strip the leading "module." (DataParallel).
        state_dict = {".".join(k.split(".")[1:]) if k.startswith("module.") else k: v
                      for k, v in state_dict.items()}

        backbone_sd = {k[len("base_model."):]: v for k, v in state_dict.items() if k.startswith("base_model.")}
        new_fc_sd = {k[len("new_fc."):]: v for k, v in state_dict.items() if k.startswith("new_fc.")}
        consensus_sd = {k[len("consensus."):]: v for k, v in state_dict.items() if k.startswith("consensus.")}

        missing, unexpected = backbone.load_state_dict(backbone_sd, strict=False)
        if missing or unexpected:
            print(f"[trn_official] backbone load_state_dict: missing={missing} unexpected={unexpected}")
        new_fc.load_state_dict(new_fc_sd)
        consensus.load_state_dict(consensus_sd)

        self.backbone = backbone
        self.new_fc = new_fc
        self.consensus = consensus
        self.to(self.device).eval()

        names = [ln.strip() for ln in Path(categories_file).read_text().splitlines() if ln.strip()]
        names = [n.replace("[", "").replace("]", "").replace("'", "") for n in names]
        assert len(names) == class_count, f"expected {class_count} categories, got {len(names)} in {categories_file}"
        self.label2id = {name: i for i, name in enumerate(names)}

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 8, 3, 224, 224) preprocessed BGR clips -> (N, 174) logits."""
        x = x.to(self.device)
        n, t = x.shape[:2]
        feat = self.backbone(x.reshape((n * t,) + x.shape[2:]))  # (N*T, 1024)
        feat = self.new_fc(feat).view(n, t, self.IMG_FEATURE_DIM)  # (N, T, 256)
        return self.consensus(feat)  # (N, 174)

    def sample_frames(self, video_path) -> torch.Tensor:
        return sample_frames(video_path, self.FRAME_COUNT)  # (T,C,H,W) uint8 RGB

    def preprocess(self, frames) -> torch.Tensor:
        x = frames if torch.is_tensor(frames) else torch.from_numpy(np.asarray(frames))
        x = x.float()
        if x.shape[-1] == 3 and x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)
        x = x.flip(1)  # RGB -> BGR
        h, w = x.shape[-2:]
        scale = (self.INPUT_SIZE / 0.875) / min(h, w)  # resize short side to 256
        nh, nw = max(self.INPUT_SIZE, round(h * scale)), max(self.INPUT_SIZE, round(w * scale))
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
        top, left = (nh - self.INPUT_SIZE) // 2, (nw - self.INPUT_SIZE) // 2
        x = x[:, :, top:top + self.INPUT_SIZE, left:left + self.INPUT_SIZE]
        mean = torch.tensor(self.MEAN_BGR, dtype=x.dtype).view(1, 3, 1, 1)
        return x - mean

    def video_from_path(self, path) -> torch.Tensor:
        return self.preprocess(self.sample_frames(path)).unsqueeze(0)  # (1,8,3,224,224)

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
