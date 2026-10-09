"""
e1_common.py -- shared machinery for experiment E1 (frame-duplication test for attribution
dilution). See the project doc "claude/E1_duplication_protocol.md" for the protocol.

What lives here
---------------
* MODEL_SPECS          which design each model uses (insertion vs reallocation), slot count,
                       tubelet size and the m values to sweep.
* ClipModel adapters   one uniform interface over the repo's model wrappers:
                           load_frames(path)          -> uint8 (T,C,H,W), model's own sampling
                           preprocess(frames_u8)      -> float (T,C,H,W), per-frame ops only
                           forward_clips(x, grad)     -> logits (B,K) for x (B,S,C,H,W)
                           gradcam_features(x)        -> (feature map, to_slots fn) for Grad-CAM
* Layout helpers       a clip is described by two parallel lists: `content` (which distinct
                       frame fills each slot) and `copy` (0 = the original frame, 1.. = extra
                       copies). Layouts are plain dicts so they can be stored in JSONL.
* FrameBank            preprocessed frames + near-duplicate variants, cached; builds clip
                       tensors from layouts.
* Evaluator            batched model evaluation of many layouts, removal operators
                       (drop / freeze-past / freeze-future / late = mean of the two), content
                       importance I(c), prediction statistics for the sensitivity gate.
* Attribution methods  Shapley (exact / permutation) with drop or freeze removal, Play Fair
                       (official code via run_playfair.py), leave-one-out, whole-frame occlusion,
                       Integrated Gradients, Grad-CAM. All return one signed score per SLOT for
                       the explained class.

All repo imports (models/, dataloaders/, run_playfair) are lazy so the "toy" model can be used
for smoke tests without the datasets or checkpoints.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import random
import zlib
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# These scripts live in frame_attribution/motivation/ but use the repo's models/, dataloaders/ and
# run_playfair.py, and some wrappers open files relative to the repo root (e.g. r3d's
# models/r3d/ucf101.json). Put the repo root on sys.path and make it the working directory, so the
# scripts work from anywhere; relative --out paths (default results/e1) resolve under the repo root.
import sys  # noqa: E402
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

# --------------------------------------------------------------------------------------------
# Model specs
# --------------------------------------------------------------------------------------------
# design:  "insert"  -> variable-length model, extra copies are INSERTED (clip gets longer)
#          "realloc" -> fixed-length model, K = slots/2 distinct frames, reference = each x2,
#                       extra copies of the target take slots from other frames (x2 -> x1)
#          "replace" -> fixed-length model, K = slots distinct frames, reference = the original
#                       clip (each x1); the target's copies REPLACE m-1 neighbouring frames, which
#                       disappear from the clip (the control replaces the same frames with the
#                       recipient). For models that the x2 reference takes out of distribution.
# removal: how a content is removed for content importance I(c) and for Shapley/LOO
#          ("drop" shortens the clip; "late" = mean of freeze-past and freeze-future)
MODEL_SPECS: Dict[str, dict] = {
    # realloc, not insert: R3D's output collapses at T > 16 whatever the content (its strided
    # temporal stages leave >1 position before the avg-pool), so the clip must stay at 16 slots
    "r3d":          dict(design="realloc", slots=16, tubelet=1, ms=[2, 4, 8],
                         removal="late", dataset="ucf101_test", gradcam_layer="layer2"),
    "vjepa2":       dict(design="insert",  slots=16, tubelet=2, ms=[1, 3, 5, 9],
                         removal="drop", dataset="ssv2_sampled", gradcam_layer=-1,
                         sampling="linspace"),
    # replace, not realloc: in the x2 reference every tubelet holds two identical frames, which
    # changes VideoMAE's prediction in 63% of videos (results/e1/videomae_diag); inputs shorter
    # than 16 frames carry no accuracy penalty at fixed content, so removal is by deletion
    "videomae":     dict(design="replace", slots=16, tubelet=2, ms=[2, 4, 6, 8],
                         removal="drop", dataset="ucf101_test", gradcam_layer=-1),
    "trn":          dict(design="realloc", slots=8,  tubelet=1, ms=[2, 4],
                         removal="late", dataset="ssv2_sampled", gradcam_layer=None),
    "trn_official": dict(design="realloc", slots=8,  tubelet=1, ms=[2, 4],
                         removal="late", dataset="ssv2_sampled", gradcam_layer=None),
    # torchvision UCF101 checkpoints (models/torchvision_ucf101.py). MC3-18 keeps the temporal
    # resolution after its first stage: accuracy is flat from 3 to 24 frames and input length has
    # no effect at fixed content, so insertion + deletion are valid (results/frame_count_tv_*,
    # length_fixed_content_tv_*). R3D-18 halves time at stages 2-4 and shows steps at 4/5, 8/9 and
    # 16/17 frames, and up to +30 pp for the same frames at length 16 -> realloc + freeze, as r3d.
    "mc3_18":       dict(design="insert",  slots=16, tubelet=1, ms=[1, 3, 5, 9],
                         removal="drop", dataset="ucf101_test", gradcam_layer="layer3",
                         sampling="linspace"),
    "r3d_18":       dict(design="realloc", slots=16, tubelet=1, ms=[2, 4, 6, 8],
                         removal="late", dataset="ucf101_test", gradcam_layer="layer2",
                         sampling="linspace"),
    # synthetic, for smoke tests (max-pool over time -> exact copies are perfect substitutes)
    "toy":          dict(design="insert",  slots=16, tubelet=1, ms=[1, 2, 4, 8],
                         removal="drop", dataset="toy", gradcam_layer=None),
    "toy_realloc":  dict(design="realloc", slots=16, tubelet=2, ms=[2, 4, 8],
                         removal="late", dataset="toy", gradcam_layer=None),
    "toy_replace":  dict(design="replace", slots=16, tubelet=2, ms=[1, 2, 4, 8],
                         removal="drop", dataset="toy", gradcam_layer=None),
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def stable_seed(*parts) -> int:
    """Deterministic 31-bit seed from any printable parts (paths, ints, strings)."""
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


# --------------------------------------------------------------------------------------------
# Model adapters
# --------------------------------------------------------------------------------------------
def _segment_centers(n_total: int, n: int) -> List[int]:
    seg = n_total / n
    idx = np.floor(np.arange(n) * seg + seg / 2).astype(np.int64)
    return np.clip(idx, 0, n_total - 1).tolist()


def frame_indices_for(name: str, n_total: int) -> List[int]:
    """The frames model `name` reads from a video of n_total frames, without loading the model.
    The adapters below use it, and so does code that must see the same frames as the model
    (e.g. the DINOv2 frame clustering, compute_frame_hierarchies.py --model).
      "centers"  (default): centre of each of `slots` equal segments, the TSN/TRN protocol
                 (models/video_utils.sample_segment_centers; r3d, videomae, trn, trn_official)
      "linspace": linspace(0, n_total - 1, slots) truncated to int (models/ssv2.py for vjepa2,
                 models/torchvision_ucf101.py for mc3_18 and r3d_18)"""
    spec = MODEL_SPECS[name]
    n = spec["slots"]
    sampling = spec.get("sampling", "centers")
    if sampling == "linspace":
        return np.linspace(0, n_total - 1, n).astype(int).tolist()
    if sampling == "centers":
        return _segment_centers(n_total, n)
    raise ValueError(f"unknown sampling {sampling!r} for {name}")


def _decode(path, idx: List[int]) -> torch.Tensor:
    from torchcodec.decoders import VideoDecoder
    dec = VideoDecoder(str(path))
    return dec.get_frames_at(indices=idx).data  # (T,C,H,W) uint8


def _n_frames(path) -> int:
    from torchcodec.decoders import VideoDecoder
    return len(VideoDecoder(str(path)))


class ClipModel:
    """Uniform interface. Subclasses set: name, spec, num_classes, label2id."""

    name: str = ""
    num_classes: int = 0
    label2id: Dict[str, int] = {}

    def __init__(self, name: str):
        self.name = name
        self.spec = MODEL_SPECS[name]
        self.device = DEVICE

    # -- data --
    def frame_indices(self, n_total: int) -> List[int]:
        return frame_indices_for(self.name, n_total)

    def load_frames(self, path) -> Tuple[torch.Tensor, List[int]]:
        idx = self.frame_indices(_n_frames(path))
        return _decode(path, idx), idx

    def preprocess(self, frames_u8: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # -- model --
    def forward_clips(self, x: torch.Tensor, grad: bool = False) -> torch.Tensor:
        raise NotImplementedError

    def gradcam_features(self, x: torch.Tensor):
        """Return (A, to_slots) where A is a differentiable feature tensor computed from x,
        `logits` is the model output computed THROUGH A, and to_slots(cam) maps a per-feature
        CAM to a per-slot score vector of length S. Implemented by subclasses that support it."""
        raise NotImplementedError(f"Grad-CAM not implemented for {self.name}")

    def forward_with_features(self, x: torch.Tensor):
        raise NotImplementedError(f"Grad-CAM not implemented for {self.name}")


def _pad_to_tubelet(x: torch.Tensor, tubelet: int) -> torch.Tensor:
    """Repeat the last frame so S is a multiple of the tubelet size (x: (B,S,C,H,W))."""
    s = x.shape[1]
    if tubelet > 1 and s % tubelet:
        pad = tubelet - s % tubelet
        x = torch.cat([x, x[:, -1:].expand(-1, pad, -1, -1, -1)], dim=1)
    return x


class R3DClip(ClipModel):
    def __init__(self, name="r3d"):
        super().__init__(name)
        from models.registry import get_model
        self.m = get_model("r3d")
        self.net = self.m.model.module if hasattr(self.m.model, "module") else self.m.model
        self.net.to(self.device).eval()
        self.label2id = self.m.label2id
        self.num_classes = len(self.label2id)

    def preprocess(self, frames_u8):
        return self.m.preprocess(frames_u8)[0].permute(1, 0, 2, 3).contiguous()  # (T,3,H,W)

    def forward_clips(self, x, grad=False):
        with torch.set_grad_enabled(grad):
            return self.net(x.to(self.device).permute(0, 2, 1, 3, 4))

    def forward_with_features(self, x):
        """Hara ResNet forward, exposing the chosen layer's activation (B,C,T',H',W')."""
        n = self.net
        layer = self.spec["gradcam_layer"]
        h = x.to(self.device).permute(0, 2, 1, 3, 4)
        h = n.relu(n.bn1(n.conv1(h)))
        if not n.no_max_pool:
            h = n.maxpool(h)
        feats = None
        for lname in ["layer1", "layer2", "layer3", "layer4"]:
            h = getattr(n, lname)(h)
            if lname == layer:
                feats = h
        logits = n.fc(torch.flatten(n.avgpool(h), 1))
        s = x.shape[1]

        def to_slots(cam):  # cam: (B,T',H',W') -> (B,S)
            per_t = cam.sum(dim=(2, 3))  # (B,T')
            return F.interpolate(per_t[:, None], size=s, mode="linear", align_corners=False)[:, 0]
        return logits, feats, "channels_first", to_slots


class TorchvisionVideoClip(ClipModel):
    """torchvision VideoResNet checkpoints fine-tuned on UCF101 (models/torchvision_ucf101.py):
    mc3_18 and r3d_18. Frames are sampled as the publisher's predictor does (linspace)."""

    def __init__(self, name):
        super().__init__(name)
        from models.registry import get_model
        self.m = get_model(name)
        self.net = self.m.model.to(self.device).eval()
        self.label2id = self.m.label2id
        self.num_classes = len(self.label2id)

    def preprocess(self, frames_u8):
        return self.m.preprocess(frames_u8)[0].permute(1, 0, 2, 3).contiguous()  # (T,3,H,W)

    def forward_clips(self, x, grad=False):
        with torch.set_grad_enabled(grad):
            return self.net(x.to(self.device).permute(0, 2, 1, 3, 4))

    def forward_with_features(self, x):
        """torchvision VideoResNet forward, exposing the chosen layer's activation (B,C,T',H',W')."""
        n = self.net
        layer = self.spec["gradcam_layer"]
        h = n.stem(x.to(self.device).permute(0, 2, 1, 3, 4))
        feats = None
        for lname in ["layer1", "layer2", "layer3", "layer4"]:
            h = getattr(n, lname)(h)
            if lname == layer:
                feats = h
        logits = n.fc(torch.flatten(n.avgpool(h), 1))
        s = x.shape[1]

        def to_slots(cam):  # cam: (B,T',H',W') -> (B,S)
            per_t = cam.sum(dim=(2, 3))  # (B,T')
            return F.interpolate(per_t[:, None], size=s, mode="linear", align_corners=False)[:, 0]
        return logits, feats, "channels_first", to_slots


@contextmanager
def _videomae_pos_embed(hf_model, n_frames: int):
    """Slice VideoMAE's fixed sinusoidal table to n_frames' token count (see eval_frame_count.py)."""
    emb = hf_model.videomae.embeddings
    cfg = hf_model.config
    full = emb.position_embeddings
    n_tok = (n_frames // cfg.tubelet_size) * (cfg.image_size // cfg.patch_size) ** 2
    emb.position_embeddings = full[:, :n_tok]
    try:
        yield
    finally:
        emb.position_embeddings = full


class _ViTClip(ClipModel):
    """Shared Grad-CAM for token-based video ViTs (tokens ordered t, h, w)."""

    hf = None  # HF model

    def _encoder_layers(self):
        raise NotImplementedError

    def _forward_hf(self, x):
        raise NotImplementedError

    def forward_with_features(self, x):
        s = x.shape[1]
        tub = self.spec["tubelet"]
        store = {}
        layer = self._encoder_layers()[self.spec["gradcam_layer"]]

        def hook(_m, _i, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            store["a"] = o
        hnd = layer.register_forward_hook(hook)
        try:
            logits = self._forward_hf(x, grad=True)
        finally:
            hnd.remove()
        feats = store["a"]  # (B,N,D)

        def to_slots(cam):  # cam: (B,N) -> (B,S)
            b, n = cam.shape
            s_pad = s + (-s) % tub
            t_tok = s_pad // tub
            per_t = cam.reshape(b, t_tok, n // t_tok).sum(-1)  # (B,T_tok)
            return per_t.repeat_interleave(tub, dim=1)[:, :s]  # both frames of a tubelet share it
        return logits, feats, "tokens", to_slots


class VJEPA2Clip(_ViTClip):
    def __init__(self, name="vjepa2"):
        super().__init__(name)
        from models.registry import get_model
        self.m = get_model("vjepa2")
        self.hf = self.m.model.to(self.device).eval()
        self.label2id = self.m.label2id
        self.num_classes = int(self.hf.config.num_labels)

    def preprocess(self, frames_u8):
        return self.m.processor(frames_u8, return_tensors="pt")["pixel_values_videos"][0]

    def _encoder_layers(self):
        return self.hf.vjepa2.encoder.layer

    def _forward_hf(self, x, grad=False):
        x = _pad_to_tubelet(x.to(self.device), self.spec["tubelet"])
        with torch.set_grad_enabled(grad):
            return self.hf(pixel_values_videos=x).logits

    def forward_clips(self, x, grad=False):
        return self._forward_hf(x, grad)


class VideoMAEClip(_ViTClip):
    def __init__(self, name="videomae"):
        super().__init__(name)
        from models.registry import get_model
        self.m = get_model("videomae")
        self.hf = self.m.model.to(self.device).eval()
        self.label2id = self.m.label2id
        self.num_classes = int(self.hf.config.num_labels)

    def preprocess(self, frames_u8):
        return self.m.preprocess(frames_u8)[0]  # (T,3,224,224)

    def _encoder_layers(self):
        return self.hf.videomae.encoder.layer

    def _forward_hf(self, x, grad=False):
        x = _pad_to_tubelet(x.to(self.device), self.spec["tubelet"])
        s = x.shape[1]
        if s > self.hf.config.num_frames:
            raise ValueError(f"VideoMAE's position table only covers {self.hf.config.num_frames} frames")
        with torch.set_grad_enabled(grad), _videomae_pos_embed(self.hf, s):
            return self.hf(pixel_values=x).logits

    def forward_clips(self, x, grad=False):
        return self._forward_hf(x, grad)


class TRNClip(ClipModel):
    """Play Fair's TRN (8 frames, MLP over concatenated per-frame features)."""

    def __init__(self, name="trn"):
        super().__init__(name)
        from models.registry import get_model
        self.m = get_model("trn")
        self.agg = self.m.model.to(self.device).eval()  # AggregatedBackboneModel
        self.label2id = self.m.label2id
        self.num_classes = len(self.label2id)

    def preprocess(self, frames_u8):
        return self.m.preprocess(frames_u8)

    def forward_clips(self, x, grad=False):
        assert x.shape[1] == self.spec["slots"], "TRN needs exactly 8 frames"
        with torch.set_grad_enabled(grad):
            return self.agg.logits(self.agg.features(x.to(self.device)))

    def forward_with_features(self, x):
        feats = self.agg.features(x.to(self.device))  # (B,T,256) per-frame features
        logits = self.agg.logits(feats)
        return logits, feats, "tokens", lambda cam: cam  # one token per slot


class TRNOfficialClip(ClipModel):
    """Original TRN-pytorch multiscale checkpoint. Its consensus randomly subsamples relations
    on every call; we fix numpy's RNG around each call so f(.) is deterministic."""

    def __init__(self, name="trn_official"):
        super().__init__(name)
        from models.registry import get_model
        self.m = get_model("trn_official")
        self.m.to(self.device).eval()
        self.label2id = self.m.label2id
        self.num_classes = len(self.label2id)

    def preprocess(self, frames_u8):
        return self.m.preprocess(frames_u8)

    def _features(self, x):
        n, t = x.shape[:2]
        f = self.m.backbone(x.reshape((n * t,) + x.shape[2:]))
        return self.m.new_fc(f).view(n, t, self.m.IMG_FEATURE_DIM)

    def _consensus(self, feats):
        state = np.random.get_state()
        np.random.seed(0)
        try:
            return self.m.consensus(feats)
        finally:
            np.random.set_state(state)

    def forward_clips(self, x, grad=False):
        assert x.shape[1] == self.spec["slots"], "TRN needs exactly 8 frames"
        with torch.set_grad_enabled(grad):
            return self._consensus(self._features(x.to(self.device)))

    def forward_with_features(self, x):
        feats = self._features(x.to(self.device))
        return self._consensus(feats), feats, "tokens", lambda cam: cam


class ToyClip(ClipModel):
    """Synthetic variable-length model: per-frame linear features -> max over time -> linear.
    Exact duplicates are perfect substitutes (f depends on the SET of frames present), so:
    Shapley gives each of m copies exactly 1/m of the content's value, LOO gives 0 for m>=2."""

    def __init__(self, name="toy", n_classes=5, size=8, seed=0):
        super().__init__(name)
        g = torch.Generator().manual_seed(seed)
        d = 3 * size * size
        self.size = size
        self.w1 = (torch.randn(d, 32, generator=g) / math.sqrt(d)).to(self.device)
        self.w2 = (torch.randn(32, n_classes, generator=g) / math.sqrt(32)).to(self.device)
        self.num_classes = n_classes
        self.label2id = {f"class{i}": i for i in range(n_classes)}

    def load_frames(self, path):
        g = torch.Generator().manual_seed(stable_seed(path))
        n = self.spec["slots"]
        return torch.randint(0, 256, (n, 3, self.size, self.size), generator=g, dtype=torch.uint8), list(range(n))

    def preprocess(self, frames_u8):
        return frames_u8.float() / 255.0 - 0.5

    def forward_clips(self, x, grad=False):
        with torch.set_grad_enabled(grad):
            x = x.to(self.device)
            b, s = x.shape[:2]
            h = torch.relu(x.reshape(b, s, -1) @ self.w1)  # (B,S,32)
            return torch.amax(h, dim=1) @ self.w2 * 5.0


ADAPTERS = {"r3d": R3DClip, "vjepa2": VJEPA2Clip, "videomae": VideoMAEClip, "trn": TRNClip,
            "trn_official": TRNOfficialClip, "toy": ToyClip, "toy_realloc": ToyClip,
            "toy_replace": ToyClip, "mc3_18": TorchvisionVideoClip, "r3d_18": TorchvisionVideoClip}


def load_clip_model(name: str) -> ClipModel:
    if name not in ADAPTERS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(ADAPTERS)}")
    return ADAPTERS[name](name)


# --------------------------------------------------------------------------------------------
# Videos
# --------------------------------------------------------------------------------------------
def get_videos(model: ClipModel, dataset: Optional[str], limit: Optional[int], seed: int = 0):
    """[(class_name, path)], restricted to classes the model knows; if `limit`, a class-stratified
    random subset (round-robin over shuffled classes) of that size."""
    dataset = dataset or model.spec["dataset"]
    if dataset == "toy":
        n = limit or 20
        return [(f"class{i % model.num_classes}", f"toy_video_{i}") for i in range(n)]
    from dataloaders.registry import get_dataloader
    names, paths = get_dataloader(dataset)
    items = [(n, str(p)) for n, p in zip(names, paths) if n in model.label2id]
    if not limit or limit >= len(items):
        return items
    rng = random.Random(seed)
    by_cls: Dict[str, list] = {}
    for it in items:
        by_cls.setdefault(it[0], []).append(it)
    for v in by_cls.values():
        rng.shuffle(v)
    classes = sorted(by_cls)
    rng.shuffle(classes)
    out, i = [], 0
    while len(out) < limit:
        c = classes[i % len(classes)]
        if by_cls[c]:
            out.append(by_cls[c].pop())
        i += 1
        if all(not v for v in by_cls.values()):
            break
    return out


# --------------------------------------------------------------------------------------------
# Layouts
# --------------------------------------------------------------------------------------------
def make_layout(content: Sequence[int]) -> dict:
    """copy index = occurrence number of that content, in slot order."""
    seen: Dict[int, int] = {}
    copy = []
    for c in content:
        copy.append(seen.get(c, 0))
        seen[c] = seen.get(c, 0) + 1
    return {"content": [int(c) for c in content], "copy": copy}


def reference_layout(design: str, n_contents: int) -> dict:
    if design in ("insert", "replace"):
        return make_layout(range(n_contents))
    return make_layout([c for c in range(n_contents) for _ in range(2)])


def insertion_layout(n: int, target: int, m: int, tubelet: int) -> Tuple[dict, bool]:
    """Insert m-1 extra copies of `target` next to it. With tubelet=2 the copies go BEFORE the
    target when it opens its tubelet (even index) and AFTER it when it closes it (odd index), so
    whole tubelets are inserted and every other tubelet keeps its original pair of frames."""
    extra = [target] * (m - 1)
    content = list(range(n))
    if tubelet == 2 and target % 2 == 0:
        content = content[:target] + extra + content[target:]
    else:
        content = content[:target + 1] + extra + content[target + 1:]
    aligned = (m - 1) % tubelet == 0
    lay = make_layout(content)
    # keep the ORIGINAL frame as copy 0 even when copies are inserted before it
    if tubelet == 2 and target % 2 == 0 and m > 1:
        pos = [i for i, c in enumerate(content) if c == target]
        for k, p in enumerate(pos):
            lay["copy"][p] = (k + 1) % m  # copies 1..m-1 first, original (0) last
    return lay, aligned


def spread_control_layout(n: int, target: int, m: int, tubelet: int, rng: random.Random):
    """Same length as insertion_layout(n, target, m) but the m-1 extra frames are single extra
    copies of DIFFERENT non-target frames (tubelet=1), or duplicated whole tubelets not containing
    the target (tubelet=2). Returns (layout, duplicated contents)."""
    k = m - 1
    if k == 0:
        return make_layout(range(n)), []
    if tubelet == 1:
        pool = [c for c in range(n) if c != target]
        dup = sorted(rng.sample(pool, min(k, len(pool))))
        content = []
        for c in range(n):
            content += [c, c] if c in dup else [c]
        return make_layout(content), dup
    assert k % 2 == 0, "tubelet=2 needs an even number of inserted frames"
    t_tub = target // 2
    pool = [t for t in range(n // 2) if t != t_tub]
    dup_tubs = sorted(rng.sample(pool, min(k // 2, len(pool))))
    content = []
    for t in range(n // 2):
        pair = [2 * t, 2 * t + 1]
        content += pair + pair if t in dup_tubs else pair
    return make_layout(content), [c for t in dup_tubs for c in (2 * t, 2 * t + 1)]


def _choose_losers(K, target, recipient, n_losers, tubelet, rng):
    """Frames that drop from x2 to x1. tubelet=2: pick disjoint ADJACENT pairs (c, c+1) so the two
    x1 frames share one tubelet and all tubelets stay aligned; fall back to arbitrary frames."""
    eligible = [c for c in range(K) if c not in (target, recipient)]
    if n_losers > len(eligible):
        return None, False
    if tubelet == 2 and n_losers % 2 == 0:
        pairs = [(c, c + 1) for c in eligible if c + 1 in eligible]
        for _ in range(200):
            rng.shuffle(pairs)
            chosen, used = [], set()
            for a, b in pairs:
                if a not in used and b not in used:
                    chosen.append((a, b))
                    used |= {a, b}
                if len(chosen) == n_losers // 2:
                    return sorted(used), True
    return sorted(rng.sample(eligible, n_losers)), tubelet == 1


def realloc_layouts(K: int, target: int, m: int, recipient: int, tubelet: int, rng: random.Random):
    """Fixed-length design over 2K slots. Reference = every content x2. Target condition: target
    x m, (m-2) other frames drop to x1. Recipient control: SAME losers, but the freed slots go to
    `recipient` (target stays x2). Returns (target_layout, control_layout, losers, aligned)."""
    losers, aligned = _choose_losers(K, target, recipient, m - 2, tubelet, rng)
    if losers is None:
        return None, None, None, False

    def build(receiver):
        counts = {c: 2 for c in range(K)}
        for c in losers:
            counts[c] = 1
        counts[receiver] = 2 + (m - 2)
        return make_layout([c for c in range(K) for _ in range(counts[c])])
    aligned = aligned and (m % tubelet == 0)
    return build(target), build(recipient), losers, aligned


def replacement_layouts(K: int, target: int, m: int, recipient: int, tubelet: int, rng: random.Random):
    """Fixed-length design over K slots. Reference = the original clip, content c in slot c (each
    x1). Target condition: a block of m consecutive slots containing the target's slot is filled
    with the target; the other m-1 contents in the block (the losers) disappear from the clip.
    Recipient control: the SAME loser slots are filled with the recipient instead, and the target
    keeps its single slot, so both clips lose the same contents and contain the same number of
    extra copies. The block is chosen at random (rng) among the blocks that contain the target's
    slot and not the recipient's; with tubelet=2 and even m only tubelet-aligned blocks are used,
    so the target's copies fill whole tubelets. Copy index 0 marks each content's original slot.
    Returns (target_layout, control_layout, losers, aligned); (None, None, None, False) if no block
    fits."""
    aligned = m % tubelet == 0
    step = tubelet if aligned else 1
    starts = [s for s in range(0, K - m + 1, step)
              if s <= target < s + m and not (s <= recipient < s + m)]
    if not starts:
        return None, None, None, False
    s0 = rng.choice(starts)
    block = list(range(s0, s0 + m))
    losers = [c for c in block if c != target]

    def build(receiver):
        content = list(range(K))
        for slot in losers:
            content[slot] = receiver
        n_extra = {}
        copy = []
        for slot, c in enumerate(content):
            if slot == c:  # the content's original frame
                copy.append(0)
            else:
                n_extra[c] = n_extra.get(c, 0) + 1
                copy.append(n_extra[c])
        return {"content": content, "copy": copy}
    return build(target), build(recipient), losers, aligned


# --------------------------------------------------------------------------------------------
# Frame bank (preprocessed frames + near-duplicate variants)
# --------------------------------------------------------------------------------------------
DUP_TYPES = ["exact", "noise", "shift"]


def perturb_u8(frame: torch.Tensor, dup_type: str, seed: int) -> torch.Tensor:
    """Near-duplicate of a uint8 (C,H,W) frame: Gaussian noise sigma=2/255, or a 1-2 px shift
    with edge replication."""
    g = torch.Generator().manual_seed(seed)
    if dup_type == "noise":
        noise = torch.randn(frame.shape, generator=g) * 2.0
        return (frame.float() + noise).round().clamp(0, 255).to(torch.uint8)
    if dup_type == "shift":
        choices = [-2, -1, 1, 2]
        dy = choices[int(torch.randint(0, 4, (1,), generator=g))]
        dx = choices[int(torch.randint(0, 4, (1,), generator=g))]
        p = 2
        padded = F.pad(frame[None].float(), (p, p, p, p), mode="replicate")[0]
        h, w = frame.shape[-2:]
        return padded[:, p + dy:p + dy + h, p + dx:p + dx + w].to(torch.uint8)
    raise ValueError(dup_type)


class FrameBank:
    def __init__(self, model: ClipModel, frames_u8: torch.Tensor, dup_type: str, seed: int):
        self.model, self.frames_u8, self.dup_type, self.seed = model, frames_u8, dup_type, seed
        self.base = model.preprocess(frames_u8).to(model.device)  # (N,C,H,W)
        self._var: Dict[Tuple[int, int], torch.Tensor] = {}

    def get(self, c: int, k: int) -> torch.Tensor:
        if k == 0 or self.dup_type == "exact":
            return self.base[c]
        key = (c, k)
        if key not in self._var:
            v = perturb_u8(self.frames_u8[c], self.dup_type, stable_seed(self.seed, c, k))
            self._var[key] = self.model.preprocess(v[None])[0].to(self.model.device)
        return self._var[key]

    def clip(self, layout: dict) -> torch.Tensor:
        return torch.stack([self.get(c, k) for c, k in zip(layout["content"], layout["copy"])])


# --------------------------------------------------------------------------------------------
# Evaluation, removal operators, content importance, gate statistics
# --------------------------------------------------------------------------------------------
def sub_layout(layout: dict, slots: Sequence[int]) -> dict:
    return {"content": [layout["content"][i] for i in slots], "copy": [layout["copy"][i] for i in slots]}


def removed_layouts(layout: dict, keep: np.ndarray, mode: str) -> List[dict]:
    """keep: bool mask over slots. Returns the layouts whose outputs are averaged:
    drop -> [kept slots]; past/future -> [frozen]; late -> [past, future]. [] if nothing kept."""
    keep = np.asarray(keep, dtype=bool)
    kept = np.flatnonzero(keep)
    if len(kept) == 0:
        return []
    if mode == "drop":
        return [sub_layout(layout, kept)]
    out = []
    for direction in (["past", "future"] if mode == "late" else [mode]):
        src = []
        for i in range(len(keep)):
            if keep[i]:
                src.append(i)
                continue
            before, after = kept[kept < i], kept[kept > i]
            if direction == "past":
                src.append(int(before[-1]) if len(before) else int(after[0]))
            else:
                src.append(int(after[0]) if len(after) else int(before[-1]))
        out.append(sub_layout(layout, src))
    return out


class Evaluator:
    def __init__(self, model: ClipModel, bank: FrameBank, batch_size: int = 16, fp16: bool = False):
        self.model, self.bank, self.bs = model, bank, batch_size
        self.fp16 = fp16 and model.device == "cuda"
        self.n_evals = 0

    @torch.no_grad()
    def logits(self, layouts: List[dict]) -> np.ndarray:
        """(N,K) logits for a list of layouts (grouped by length, batched)."""
        out = [None] * len(layouts)
        by_len: Dict[int, List[int]] = {}
        for i, lay in enumerate(layouts):
            by_len.setdefault(len(lay["content"]), []).append(i)
        for _, idxs in by_len.items():
            for s in range(0, len(idxs), self.bs):
                chunk = idxs[s:s + self.bs]
                x = torch.stack([self.bank.clip(layouts[i]) for i in chunk])
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
                    y = self.model.forward_clips(x).float().cpu().numpy()
                for j, i in enumerate(chunk):
                    out[i] = y[j]
                self.n_evals += len(chunk)
        return np.stack(out) if out else np.zeros((0, self.model.num_classes))

    def value(self, layout: dict, keeps: np.ndarray, mode: str, cls: int, prior: np.ndarray) -> np.ndarray:
        """v(S) = P(cls | clip with only the kept slots, removed ones handled by `mode`);
        v(empty) = prior[cls]. keeps: (M,S) bool."""
        keeps = np.atleast_2d(keeps)
        jobs, owners = [], []
        for r, k in enumerate(keeps):
            for lay in removed_layouts(layout, k, mode):
                jobs.append(lay)
                owners.append(r)
        vals = np.full(len(keeps), np.nan)
        if jobs:
            p = softmax_np(self.logits(jobs))[:, cls]
            acc: Dict[int, List[float]] = {}
            for r, v in zip(owners, p):
                acc.setdefault(r, []).append(float(v))
            for r, v in acc.items():
                vals[r] = float(np.mean(v))
        vals[np.isnan(vals)] = prior[cls]
        return vals


def softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def content_importance(ev: Evaluator, layout: dict, mode: str, cls: int, prior: np.ndarray,
                       contents: Optional[Sequence[int]] = None) -> Dict[int, float]:
    """I(c) = v(all slots) - v(all slots except every copy of c)."""
    cont = np.asarray(layout["content"])
    contents = sorted(set(cont.tolist())) if contents is None else list(contents)
    keeps = [np.ones(len(cont), bool)] + [cont != c for c in contents]
    v = ev.value(layout, np.stack(keeps), mode, cls, prior)
    return {int(c): float(v[0] - v[i + 1]) for i, c in enumerate(contents)}


def margin_k(z: np.ndarray, cls: int, k: int = 3) -> float:
    """logit[cls] - mean of the k best OTHER logits (the paper's m_k, anchored on cls)."""
    others = np.sort(np.delete(z, cls))[::-1][:k]
    return float(z[cls] - others.mean())


def js_div(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    eps = 1e-12
    kl = lambda a, b: float(np.sum(a * (np.log(a + eps) - np.log(b + eps))))  # noqa: E731
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def pred_stats(z_ref: np.ndarray, z: np.ndarray, cls: int) -> dict:
    p_ref, p = softmax_np(z_ref), softmax_np(z)
    return {"pred": int(z.argmax()), "same_pred": bool(z.argmax() == cls),
            "prob": float(p[cls]), "d_prob": float(p[cls] - p_ref[cls]),
            "d_logit": float(z[cls] - z_ref[cls]),
            "d_margin3": margin_k(z, cls) - margin_k(z_ref, cls),
            "js": js_div(p_ref, p)}


# --------------------------------------------------------------------------------------------
# Attribution methods -- all return a float np.array with one score per SLOT
# --------------------------------------------------------------------------------------------
def shapley_exact(n: int, value_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    masks = np.array(list(itertools.product([False, True], repeat=n)))[:, ::-1]  # bit i = slot i
    ids = (masks * (1 << np.arange(n))).sum(1)
    order = np.argsort(ids)
    masks = masks[order]
    v = value_fn(masks)  # index = bitmask integer
    sizes = masks.sum(1)
    w = np.array([math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n) if s < n else 0.0
                  for s in range(n + 1)])
    phi = np.zeros(n)
    idx = np.arange(len(masks))
    for i in range(n):
        without = idx[~masks[:, i]]
        phi[i] = np.sum(w[sizes[without]] * (v[without | (1 << i)] - v[without]))
    return phi


def shapley_permutation(n: int, value_fn, n_perm: int, seed: int) -> np.ndarray:
    """Antithetic permutation sampling (each permutation and its reverse)."""
    rng = np.random.default_rng(seed)
    perms = []
    for _ in range(max(1, n_perm // 2)):
        p = rng.permutation(n)
        perms += [p, p[::-1]]
    keys, index = [], {}
    rows = []
    for p in perms:
        mask = np.zeros(n, bool)
        chain = [mask.copy()]
        for i in p:
            mask[i] = True
            chain.append(mask.copy())
        rows.append(chain)
        for m in chain:
            k = m.tobytes()
            if k not in index:
                index[k] = len(keys)
                keys.append(m)
    v = value_fn(np.stack(keys))
    phi = np.zeros(n)
    for p, chain in zip(perms, rows):
        vals = [v[index[m.tobytes()]] for m in chain]
        for j, i in enumerate(p):
            phi[i] += vals[j + 1] - vals[j]
    return phi / len(perms)


def attr_shapley(ev, layout, cls, prior, removal, exact_max=12, n_perm=64, seed=0):
    n = len(layout["content"])
    fn = lambda keeps: ev.value(layout, keeps, removal, cls, prior)  # noqa: E731
    if n <= exact_max:
        return shapley_exact(n, fn)
    return shapley_permutation(n, fn, n_perm, seed)


def attr_loo(ev, layout, cls, prior, removal):
    n = len(layout["content"])
    keeps = np.ones((n + 1, n), bool)
    for i in range(n):
        keeps[i + 1, i] = False
    v = ev.value(layout, keeps, removal, cls, prior)
    return v[0] - v[1:]


@torch.no_grad()
def attr_occlusion(model: ClipModel, bank: FrameBank, layout, cls, fill: str = "zero"):
    """Whole-frame occlusion: replace ONE slot (all channels) by zeros in the model's normalised
    input space (= the dataset mean colour) and record the drop in P(cls)."""
    x = bank.clip(layout)
    n = x.shape[0]
    xs = x[None].repeat(n + 1, 1, 1, 1, 1)
    for i in range(n):
        xs[i + 1, i] = 0.0
    p = torch.softmax(torch.cat([model.forward_clips(xs[j:j + 8]) for j in range(0, n + 1, 8)]).float(), -1)[:, cls]
    p = p.cpu().numpy()
    return p[0] - p[1:]


def attr_integrated_gradients(model: ClipModel, bank: FrameBank, layout, cls, steps=32, batch=2):
    """IG on the logit of `cls`, zero baseline in normalised input space, midpoint Riemann sum;
    per-slot score = SUM over (C,H,W) (so scores add up to logit(x) - logit(baseline))."""
    x = bank.clip(layout).detach()
    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    total = torch.zeros_like(x)
    for s in range(0, steps, batch):
        a = alphas[s:s + batch].to(x.device).view(-1, 1, 1, 1, 1)
        xi = (a * x[None]).requires_grad_(True)
        y = model.forward_clips(xi, grad=True)[:, cls].sum()
        g, = torch.autograd.grad(y, xi)
        total += g.sum(0)
    attr = x * total / steps
    return attr.sum(dim=(1, 2, 3)).detach().cpu().numpy()


def attr_gradcam(model: ClipModel, bank: FrameBank, layout, cls):
    x = bank.clip(layout).detach()[None]
    logits, feats, kind, to_slots = model.forward_with_features(x)
    g, = torch.autograd.grad(logits[0, cls], feats)
    if kind == "channels_first":  # (B,C,T,H,W)
        w = g.mean(dim=(2, 3, 4), keepdim=True)
        cam = torch.relu((w * feats).sum(1))  # (B,T,H,W)
    else:  # tokens (B,N,D)
        w = g.mean(dim=1, keepdim=True)
        cam = torch.relu((w * feats).sum(-1))  # (B,N)
    return to_slots(cam.detach())[0].cpu().numpy()


def attr_playfair(model: ClipModel, bank: FrameBank, layout, cls, prior, approximate: bool,
                  max_samples: int = 1024, seed: int = 0, batch_size: int = 16, fp16: bool = False):
    """Play Fair ESVs with the OFFICIAL attributor (run_playfair.py / play-fair/src): frames are
    DROPPED (variable-length path), f = softmax, f(empty) = prior. fp16 runs the model evaluations
    under torch.autocast, as the Evaluator does for the other perturbation methods."""
    import run_playfair as rp
    set_all_seeds(seed)
    frames_pp = bank.clip(layout)
    priors = torch.from_numpy(prior.astype(np.float32)).to(model.device)[None]
    char_fn = rp.CharacteristicFn(model, frames_pp, priors, batch_size, fp16=fp16)
    dev = torch.device(model.device)
    sampler = (rp.ConstructiveRandomSamplerPy311(max_samples=max_samples, device=dev) if approximate
               else rp.ExhaustiveSubsetSampler(device=dev))
    attributor = rp.CharacteristicFunctionShapleyAttributor(
        characteristic_fn=char_fn, n_classes=model.num_classes, subset_sampler=sampler, device=dev)
    seq = torch.arange(frames_pp.shape[0], device=dev).unsqueeze(-1)
    esvs, _ = attributor.explain(seq, n_iters=1)
    return esvs[:, cls].cpu().numpy(), char_fn.n_model_evals


METHODS = ["shapley_freeze", "shapley_drop", "playfair", "loo_drop", "loo_freeze",
           "occlusion", "ig", "gradcam"]
STOCHASTIC = {"shapley_freeze", "shapley_drop", "playfair"}


def run_method(method: str, model: ClipModel, bank: FrameBank, ev: Evaluator, layout: dict, cls: int,
               prior: np.ndarray, seed: int = 0, n_perm: int = 64, exact_max: int = 12,
               ig_steps: int = 32, ig_batch: int = 2, pf_max_samples: int = 1024) -> Tuple[np.ndarray, int]:
    """Returns (per-slot scores, number of model evaluations used)."""
    n0 = ev.n_evals
    # insert and replace remove frames by deletion by design (the model accepts shorter inputs)
    variable = model.spec["design"] in ("insert", "replace")
    # r3d is realloc only because it breaks ABOVE 16 frames; dropping frames makes clips shorter,
    # which it accepts (results/design_choice_report.md)
    if method in ("shapley_drop", "loo_drop", "playfair") and not variable and model.name not in ("videomae", "r3d"):
        raise ValueError(f"{method} drops frames; {model.name} needs a fixed number of frames")
    if method == "shapley_freeze":
        s = attr_shapley(ev, layout, cls, prior, "late", exact_max, n_perm, seed)
    elif method == "shapley_drop":
        s = attr_shapley(ev, layout, cls, prior, "drop", exact_max, n_perm, seed)
    elif method == "loo_drop":
        s = attr_loo(ev, layout, cls, prior, "drop")
    elif method == "loo_freeze":
        s = attr_loo(ev, layout, cls, prior, "late")
    elif method == "occlusion":
        s = attr_occlusion(model, bank, layout, cls)
        return s, len(layout["content"]) + 1
    elif method == "ig":
        return attr_integrated_gradients(model, bank, layout, cls, ig_steps, ig_batch), ig_steps
    elif method == "gradcam":
        return attr_gradcam(model, bank, layout, cls), 1
    elif method == "playfair":
        n = len(layout["content"])
        # same precision and batch size as the Evaluator (--fp16, --batch-size)
        s, n_ev = attr_playfair(model, bank, layout, cls, prior, approximate=n > exact_max,
                                max_samples=pf_max_samples, seed=seed, batch_size=ev.bs, fp16=ev.fp16)
        return s, n_ev
    else:
        raise ValueError(method)
    return s, ev.n_evals - n0


# --------------------------------------------------------------------------------------------
# JSONL helpers
# --------------------------------------------------------------------------------------------
def resolve_video_path(path: str) -> str:
    """A video path stored in a conditions file, made openable on this machine. Conditions built
    on one machine (e.g. D:/datasets/UCF-101/<class>/<file>.avi on Windows) are attributed on another
    (Colab): if `path` doesn't exist, look for <class>/<file> under CONST.UCF101_PATH and
    CONST.SSV2_PATH (both datasets are one folder per class). Callers keep the ORIGINAL path as
    the record key, so attributions still join to their conditions."""
    if os.path.exists(path):
        return path
    import CONST
    parts = path.replace("\\", "/").split("/")
    for root in (CONST.UCF101_PATH, CONST.SSV2_PATH):
        cand = os.path.join(root, *parts[-2:])
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"{path} (also not found as <class>/<file> under UCF101_PATH or SSV2_PATH)")


def read_jsonl(path) -> List[dict]:
    """Records of a JSONL file. Lines that aren't valid JSON are skipped with a warning: a write
    cut off by a crash or a Colab disconnect (common on Google Drive, which can leave NUL bytes)
    leaves a broken line, and the resuming scripts then simply redo that record."""
    if not os.path.exists(path):
        return []
    out, bad = [], 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.replace("\x00", "").strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"[warn] {path}: skipped {bad} unreadable line(s); those records will be redone")
    return out


def append_jsonl(path, record: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def uniform_prior(k: int) -> np.ndarray:
    return np.full(k, 1.0 / k)


def load_prior(model: ClipModel, spec: Optional[str]) -> np.ndarray:
    """'uniform' (default) or a .npy file with K class frequencies."""
    if not spec or spec == "uniform":
        return uniform_prior(model.num_classes)
    p = np.load(spec).astype(float)
    return p / p.sum()
