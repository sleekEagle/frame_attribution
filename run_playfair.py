#!/usr/bin/env python3
"""
run_playfair.py -- per-frame Element Shapley Values (ESV) for ONE video.

Implements the Play Fair method (Price & Damen, ACCV 2020) for an *end-to-end*
video classifier that can consume a variable number of frames.

What is reused vs. what is new here
-----------------------------------
* REUSED, UNMODIFIED: the Shapley computation itself, i.e. `play-fair/src/attribution/
  characteristic_function_shapley_value_attributor.py` and the subset samplers. This
  is the official code path Play Fair uses for variable-length models (TSN).
* NEW (glue only): (1) test-time frame sampling identical to Play Fair's
  `sample_uniform_idx`; (2) a characteristic function f(X) that runs an end-to-end
  model on a *subsequence of raw frames* (Play Fair's repo only supports models that sit
  on top of pre-extracted features); (3) IO + plotting; (4) a frame-sensitivity probe.

Characteristic function (as in Play Fair's compute_esvs.py)
    f(X)  = softmax(model(X))          for |X| >= 1   (frames kept in temporal order)
    f({}) = class prior                (uniform unless --priors is given)
    ESV_i = Shapley value of frame i under v(X) = f(X) - f({})
    Efficiency:  sum_i ESV_i[c] = f(full)[c] - prior[c]     (exact mode only)

IMPORTANT: this is the "variable-length" Play Fair path -- frames are DROPPED, the
model sees a shorter clip. It is only meaningful if your model handles shorter clips
sensibly. Run the built-in probe (on by default) and read its output.

Example
-------
    python run_playfair.py video.avi --adapter r3d --checkpoint r3d_ucf101.pth \
        --num-classes 101 --labels ucf101_classes.txt --num-frames 8

    python run_playfair.py video.webm --adapter vjepa2 --num-frames 8 --fp16

    # your own model: my_models.py must define build_adapter(device, **kwargs)
    python run_playfair.py video.avi --adapter my_models:build_adapter --adapter-arg ckpt=x.pth
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Import the official Play Fair code (kept untouched in ./play-fair)
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
PLAYFAIR_SRC = HERE / "play-fair" / "src"
if not PLAYFAIR_SRC.exists():
    sys.exit(
        f"Could not find {PLAYFAIR_SRC}.\n"
        "Clone the official repo next to this script:\n"
        "    git clone https://github.com/willprice/play-fair.git"
    )
sys.path.insert(0, str(PLAYFAIR_SRC))
from attribution.characteristic_function_shapley_value_attributor import (  # noqa: E402
    CharacteristicFunctionShapleyAttributor,
)
from subset_samplers import ConstructiveRandomSampler, ExhaustiveSubsetSampler  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


class ConstructiveRandomSamplerPy311(ConstructiveRandomSampler):
    """Play Fair targets Python 3.7, where `random.sample(<set>, n)` worked. Python >= 3.11
    removed that. This subclass makes the ONE change needed (sample from a list) and is
    otherwise identical to upstream `ConstructiveRandomSampler._sample`."""

    def _sample(self, n_video_frames, n_frames):
        from scipy.special import comb
        import random as _random

        max_possible = comb(n_video_frames, n_frames, exact=True)
        candidates = ConstructiveRandomSampler.compute_candidate_pool(
            self.population, self.previous_subsets)
        n = min(len(candidates), max_possible, self.max_samples)
        return set(_random.sample(list(candidates), n))


# --------------------------------------------------------------------------- #
# Frame sampling + decoding
# --------------------------------------------------------------------------- #
def sample_uniform_idx(n_total: int, n_samples: int) -> np.ndarray:
    """Centre of each of `n_samples` equal segments. Identical to Play Fair's
    `frame_sampling.sample_uniform_idx` (the test-time sampler)."""
    if n_total < n_samples:
        raise ValueError(
            f"Video has {n_total} frames but {n_samples} were requested. "
            "Lower --num-frames or pass --frame-indices."
        )
    seg = n_total / n_samples
    return np.floor(np.arange(n_samples) * seg + seg / 2).astype(np.int64)


def _decode_all(path: Path) -> Tuple[List[np.ndarray], float]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def load_frames(
    path: Path, num_frames: int, explicit_indices: Optional[Sequence[int]]
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """Returns (frames uint8 (n,H,W,3), frame_indices (n,), total_frames, fps)."""
    import cv2

    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMG_EXT)
        if not files:
            raise ValueError(f"No images found in {path}")
        total, fps = len(files), 0.0
        idx = (
            np.asarray(explicit_indices, dtype=np.int64)
            if explicit_indices
            else sample_uniform_idx(total, num_frames)
        )
        frames = np.stack(
            [cv2.cvtColor(cv2.imread(str(files[i])), cv2.COLOR_BGR2RGB) for i in idx]
        )
        return frames, idx, total, fps

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()

    # Fast path: trust the container's frame count, decode sequentially, keep needed.
    if total > 0:
        idx = (
            np.asarray(explicit_indices, dtype=np.int64)
            if explicit_indices
            else sample_uniform_idx(total, num_frames)
        )
        want = {int(i): k for k, i in enumerate(idx)}
        got: Dict[int, np.ndarray] = {}
        cap = cv2.VideoCapture(str(path))
        i, last = 0, int(idx.max())
        while i <= last:
            ok, bgr = cap.read()
            if not ok:
                break
            if i in want:
                got[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            i += 1
        cap.release()
        if len(got) == len(want):
            return np.stack([got[int(j)] for j in idx]), idx, total, fps

    # Slow path: container count was missing/wrong (common for webm) -> decode all.
    frames_all, fps = _decode_all(path)
    total = len(frames_all)
    if total == 0:
        raise ValueError(f"Decoded 0 frames from {path}")
    idx = (
        np.asarray(explicit_indices, dtype=np.int64)
        if explicit_indices
        else sample_uniform_idx(total, num_frames)
    )
    return np.stack([frames_all[int(j)] for j in idx]), idx, total, fps


# --------------------------------------------------------------------------- #
# Model adapters
#   An adapter turns an arbitrary end-to-end video classifier into:
#       preprocess(frames uint8 (T,H,W,3)) -> float tensor (T,C,H',W')   [per-frame ops only]
#       forward_clips(clips (B,S,C,H',W'))  -> logits (B,K)              [any S >= 1]
# --------------------------------------------------------------------------- #
class TorchvisionVideoAdapter:
    """torchvision 3D-CNNs (r3d_18, mc3_18, r2plus1d_18, ...).

    All of them end in adaptive average pooling, so any temporal length >= 1 runs.
    NOTE: if your R3D checkpoint comes from another code base (e.g. Hara's
    3D-ResNets used by Uchiyama et al.), the state-dict keys differ -- write a
    small custom adapter instead (see --adapter module:function).
    """

    def __init__(self, device, arch="r3d_18", num_classes=400, checkpoint=None,
                 resize=128, size=112,
                 mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)):
        from torchvision.models import video as tv

        self.device = device
        self.model = getattr(tv, arch)(weights=None, num_classes=int(num_classes))
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu")
            state = state.get("state_dict", state.get("model", state))
            state = {re.sub(r"^(module\.|model\.)", "", k): v for k, v in state.items()}
            self.model.load_state_dict(state, strict=True)
        else:
            print("[adapter] WARNING: no --checkpoint given, using RANDOM weights "
                  "(fine for smoke tests, meaningless for science).")
        self.model.eval().to(device)
        self.num_classes = int(num_classes)
        self.class_names = None
        self.resize, self.size = int(resize), int(size)
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def preprocess(self, frames: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # (T,3,H,W)
        h, w = x.shape[-2:]
        scale = self.resize / min(h, w)
        x = F.interpolate(x, size=(round(h * scale), round(w * scale)),
                          mode="bilinear", align_corners=False)
        h, w = x.shape[-2:]
        t, l = (h - self.size) // 2, (w - self.size) // 2
        x = x[..., t:t + self.size, l:l + self.size]
        return (x - self.mean) / self.std

    @torch.no_grad()
    def forward_clips(self, clips: torch.Tensor) -> torch.Tensor:
        return self.model(clips.permute(0, 2, 1, 3, 4))  # (B,C,S,H,W)


class VJEPA2Adapter:
    """Hugging Face V-JEPA 2 video classifier (ViT-L, SSv2 head by default).

    V-JEPA 2 uses 3D patch ("tubelet") embedding with tubelet_size=2. An odd number
    of frames cannot be tiled by tubelets; a Conv3d-based embedding would silently
    truncate the last frame. `pad_odd=True` repeats the final frame so every frame in
    the coalition is seen by the model. That is a small deviation from pure dropping;
    it is reported in the output JSON. The frame-sensitivity probe checks the result.
    """

    def __init__(self, device, repo="facebook/vjepa2-vitl-fpc16-256-ssv2",
                 pad_odd=True, **_):
        from transformers import AutoModelForVideoClassification, AutoVideoProcessor

        self.device = device
        self.model = AutoModelForVideoClassification.from_pretrained(repo).to(device).eval()
        self.processor = AutoVideoProcessor.from_pretrained(repo)
        cfg = self.model.config
        self.num_classes = int(cfg.num_labels)
        self.class_names = [cfg.id2label[i] for i in range(self.num_classes)]
        self.tubelet = int(getattr(cfg, "tubelet_size", 2))
        self.pad_odd = pad_odd

    def preprocess(self, frames: np.ndarray) -> torch.Tensor:
        video = torch.from_numpy(frames).permute(0, 3, 1, 2)  # uint8 (T,C,H,W)
        return self.processor(video, return_tensors="pt")["pixel_values_videos"][0]

    @torch.no_grad()
    def forward_clips(self, clips: torch.Tensor) -> torch.Tensor:
        s = clips.shape[1]
        if self.pad_odd and s % self.tubelet:
            pad = self.tubelet - s % self.tubelet
            clips = torch.cat([clips, clips[:, -1:].expand(-1, pad, -1, -1, -1)], dim=1)
        return self.model(pixel_values_videos=clips).logits


def build_adapter(args, device):
    if ":" in args.adapter:  # user-supplied "module:function"
        mod_name, fn_name = args.adapter.split(":")
        sys.path.insert(0, str(Path.cwd()))
        fn = getattr(importlib.import_module(mod_name), fn_name)
        kwargs = dict(kv.split("=", 1) for kv in args.adapter_arg)
        return fn(device=device, **kwargs)
    if args.adapter == "r3d":
        return TorchvisionVideoAdapter(
            device, arch=args.arch, num_classes=args.num_classes,
            checkpoint=args.checkpoint, resize=args.resize, size=args.size,
            mean=args.mean, std=args.std)
    if args.adapter == "vjepa2":
        return VJEPA2Adapter(device, repo=args.vjepa_repo)
    raise ValueError(f"Unknown adapter {args.adapter!r}")


# --------------------------------------------------------------------------- #
# Characteristic function f(X)
# --------------------------------------------------------------------------- #
class CharacteristicFn:
    """Callable handed to Play Fair's attributor.

    The attributor indexes `sequence_features[subset_idxs]`. To avoid materialising
    thousands of clip copies we hand it *frame indices* (T,1) instead of pixels and
    gather pixels lazily, in chunks, here.
    """

    def __init__(self, adapter, frames_pp: torch.Tensor, priors: torch.Tensor,
                 batch_size: int, fp16: bool):
        self.adapter, self.frames = adapter, frames_pp
        self.priors, self.bs, self.fp16 = priors, batch_size, fp16
        self.n_model_evals = 0

    def __call__(self, idx: torch.Tensor) -> torch.Tensor:
        assert idx.ndim == 3, f"expected (N,S,1) index tensor, got {tuple(idx.shape)}"
        n, s = idx.shape[:2]
        if s == 0:  # empty coalition -> class prior
            return self.priors.expand(n, -1)
        idx = idx[..., 0]
        out = []
        for i in range(0, n, self.bs):
            clips = self.frames[idx[i:i + self.bs]]  # (b,S,C,H,W), temporal order kept
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
                logits = self.adapter.forward_clips(clips)
            out.append(torch.softmax(logits.float(), dim=-1))
        self.n_model_evals += n
        return torch.cat(out, dim=0)


# --------------------------------------------------------------------------- #
# Safety check: does the model actually *look at* every frame of a short clip?
# --------------------------------------------------------------------------- #
@torch.no_grad()
def probe_frame_sensitivity(adapter, frames_pp: torch.Tensor, fp16: bool) -> Dict[int, List[int]]:
    """For each clip length S, perturb each position j and check the logits change.
    A model that ignores some frames (temporal stride/tubelet truncation, fixed
    positional embeddings, ...) makes those frames spurious 'dummy players' and
    silently biases the Shapley values. Returns {S: [ignored positions]}."""
    n = frames_pp.shape[0]
    ignored: Dict[int, List[int]] = {}
    for s in range(1, n + 1):
        clip = frames_pp[:s]
        variants = [clip]
        for j in range(s):
            v = clip.clone()
            v[j] = v[j] + 1.0  # large shift in normalised pixel space
            variants.append(v)
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            logits = adapter.forward_clips(torch.stack(variants)).float()
        base, rest = logits[0], logits[1:]
        tol = 1e-4 * (base.abs().max().item() + 1e-8)
        bad = [j for j in range(s) if (rest[j] - base).abs().max().item() <= tol]
        if bad:
            ignored[s] = bad
    return ignored


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def load_priors(path: Optional[Path], k: int) -> Tuple[np.ndarray, str]:
    if path is None:
        return np.full(k, 1.0 / k, dtype=np.float32), "uniform"
    if path.suffix == ".npy":
        p = np.load(path)
    elif path.suffix == ".csv":  # same layout as Play Fair: columns `class,prior`
        import pandas as pd

        p = pd.read_csv(path, index_col="class")["prior"].sort_index().values
    else:
        p = np.loadtxt(path)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if len(p) != k:
        raise ValueError(f"priors has {len(p)} entries but the model has {k} classes")
    return (p / p.sum()).astype(np.float32), str(path)


def load_labels(path: Optional[Path], adapter_names) -> Optional[List[str]]:
    if path:
        return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    return adapter_names


def resolve_class(spec: Optional[str], names: Optional[List[str]], default: int) -> int:
    if spec is None:
        return default
    if spec.lstrip("-").isdigit():
        return int(spec)
    if names and spec in names:
        return names.index(spec)
    raise ValueError(f"--target-class {spec!r} is neither an index nor a known class name")


def plot_result(frames, frame_idx, esv, title, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(frames)
    fig = plt.figure(figsize=(max(8, 1.6 * n), 5))
    gs = fig.add_gridspec(2, n, height_ratios=[1, 1.3], hspace=0.15, wspace=0.03)
    for i in range(n):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(frames[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"#{frame_idx[i]}", fontsize=8)
    ax = fig.add_subplot(gs[1, :])
    ax.bar(np.arange(n), esv, color=["#2a9d4b" if v >= 0 else "#d1495b" for v in esv])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels([str(i) for i in frame_idx], fontsize=8)
    ax.set_xlabel("frame index in video")
    ax.set_ylabel("ESV (prob. mass)")
    fig.suptitle(title, fontsize=10)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Play Fair per-frame Element Shapley Values for one video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("video", type=Path, help="video file OR directory of image frames")
    p.add_argument("--adapter", default="r3d",
                   help="'r3d', 'vjepa2', or 'your_module:build_adapter'")
    p.add_argument("--adapter-arg", action="append", default=[], metavar="KEY=VAL",
                   help="extra kwargs for a custom adapter (repeatable)")
    p.add_argument("--checkpoint", type=Path, help="r3d: state-dict of a fine-tuned model")
    p.add_argument("--num-classes", type=int, default=101, help="r3d: size of the head")
    p.add_argument("--arch", default="r3d_18", help="r3d: torchvision.models.video name")
    p.add_argument("--resize", type=int, default=128, help="r3d: short-side resize")
    p.add_argument("--size", type=int, default=112, help="r3d: centre-crop size")
    p.add_argument("--mean", type=float, nargs=3, default=(0.43216, 0.394666, 0.37645))
    p.add_argument("--std", type=float, nargs=3, default=(0.22803, 0.22145, 0.216989))
    p.add_argument("--vjepa-repo", default="facebook/vjepa2-vitl-fpc16-256-ssv2")
    p.add_argument("--labels", type=Path, help="text file, one class name per line")
    p.add_argument("--priors", type=Path,
                   help=".csv (class,prior) / .npy / .txt. f({}) in the paper. Default uniform. "
                        "Use the TRAIN-set class frequencies to match Play Fair.")
    p.add_argument("--num-frames", type=int, default=8,
                   help="frames sampled from the video = #players in the game")
    p.add_argument("--frame-indices", type=int, nargs="+",
                   help="explicit frame indices (overrides --num-frames)")
    p.add_argument("--target-class", help="class index or name (default: model prediction)")
    p.add_argument("--approximate", action="store_true",
                   help="Play Fair's constructive random subset sampler (needed for >12 frames)")
    p.add_argument("--max-samples-per-scale", type=int, default=128)
    p.add_argument("--n-iters", type=int, default=1,
                   help="approximation repeats (ignored in exact mode)")
    p.add_argument("--force", action="store_true", help="allow exact mode with >12 frames")
    p.add_argument("--batch-size", type=int, default=32, help="clips per forward pass")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fp16", action="store_true", help="CUDA autocast fp16")
    p.add_argument("--skip-probe", action="store_true", help="skip frame-sensitivity probe")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import random

    random.seed(args.seed)
    device = torch.device(args.device)
    if args.fp16 and device.type != "cuda":
        print("[warn] --fp16 needs CUDA; running in fp32.")
        args.fp16 = False

    # ---- data ----------------------------------------------------------------
    frames, frame_idx, total, fps = load_frames(args.video, args.num_frames, args.frame_indices)
    n = len(frames)
    print(f"[data] {args.video.name}: {total} frames, sampled {n}: {frame_idx.tolist()}")

    n_exact_evals = 2 ** n  # 2^n-1 non-empty subsets + 1 repeat of the full clip
    if not args.approximate and n > 12 and not args.force:
        sys.exit(f"Exact ESV over {n} frames needs {n_exact_evals:,} model evaluations. "
                 "Use --approximate (Play Fair's sampler) or --force.")

    # ---- model ---------------------------------------------------------------
    adapter = build_adapter(args, device)
    names = load_labels(args.labels, getattr(adapter, "class_names", None))
    k = adapter.num_classes
    priors_np, priors_src = load_priors(args.priors, k)
    if priors_src == "uniform":
        print("[priors] using UNIFORM f({}) -- pass --priors <train-set frequencies> to match Play Fair")
    priors = torch.from_numpy(priors_np).to(device).unsqueeze(0)  # (1,K)

    frames_pp = adapter.preprocess(frames).to(device)  # (n,C,H',W') -- per-frame ops only
    if not args.skip_probe:
        bad = probe_frame_sensitivity(adapter, frames_pp, args.fp16)
        if bad:
            print("[probe] WARNING: the model output does NOT depend on some frames:")
            for s, pos in bad.items():
                print(f"         clip length {s}: ignored positions {pos}")
            print("        Those frames act as spurious 'dummy players' and bias the ESVs. "
                  "Fix the adapter (padding/stride handling) before trusting results.")
        else:
            print("[probe] OK: model output depends on every frame at every clip length 1..%d" % n)
    else:
        bad = None

    # ---- Play Fair ESV -------------------------------------------------------
    char_fn = CharacteristicFn(adapter, frames_pp, priors, args.batch_size, args.fp16)
    sampler = (ConstructiveRandomSamplerPy311(max_samples=args.max_samples_per_scale, device=device)
               if args.approximate else ExhaustiveSubsetSampler(device=device))
    attributor = CharacteristicFunctionShapleyAttributor(
        characteristic_fn=char_fn, n_classes=k, subset_sampler=sampler, device=device)
    seq = torch.arange(n, device=device).unsqueeze(-1)  # frame *indices* as "features"

    t0 = time.time()
    esvs, full = attributor.explain(seq, n_iters=args.n_iters if args.approximate else 1)
    secs = time.time() - t0
    esvs, full = esvs.cpu().numpy(), full.cpu().numpy()  # (n,K), (K,)

    pred = int(full.argmax())
    tgt = resolve_class(args.target_class, names, pred)
    name = lambda c: (names[c] if names and c < len(names) else str(c))  # noqa: E731
    esv_t = esvs[:, tgt]
    expected = float(full[tgt] - priors_np[tgt])
    gap = float(esv_t.sum() - expected)

    print(f"[result] prediction: {name(pred)} (p={full[pred]:.3f}); explaining: {name(tgt)} "
          f"(p={full[tgt]:.3f}, prior={priors_np[tgt]:.4f})")
    print(f"[result] model evaluations: {char_fn.n_model_evals:,}  ({secs:.1f}s)  "
          f"mode={'approximate' if args.approximate else 'exact'}")
    print(f"[result] efficiency: sum(ESV)={esv_t.sum():+.6f}  vs  f(X)-f({{}})={expected:+.6f}  "
          f"gap={gap:+.2e}" + ("" if not args.approximate else "  (nonzero gap expected when approximating)"))
    print("         frame  ESV")
    for i in range(n):
        print(f"         {int(frame_idx[i]):5d}  {esv_t[i]:+.4f}  " + "#" * int(round(40 * abs(esv_t[i]) / (np.abs(esv_t).max() + 1e-12))))

    # ---- save ----------------------------------------------------------------
    out = args.out_dir or Path("playfair_out") / args.video.stem
    out.mkdir(parents=True, exist_ok=True)
    top = np.argsort(-full)[:5]
    record = {
        "video": str(args.video), "total_frames": total, "fps": fps,
        "frame_indices": frame_idx.tolist(),
        "timestamps_s": (frame_idx / fps).tolist() if fps else None,
        "adapter": args.adapter, "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "removal": "drop (Play Fair variable-length path)",
        "vjepa_pad_odd": getattr(adapter, "pad_odd", None),
        "priors": priors_src,
        "mode": "approximate" if args.approximate else "exact",
        "max_samples_per_scale": args.max_samples_per_scale if args.approximate else None,
        "n_iters": args.n_iters if args.approximate else 1,
        "n_model_evals": char_fn.n_model_evals, "seconds": secs, "seed": args.seed,
        "predicted": {"index": pred, "name": name(pred), "prob": float(full[pred])},
        "target": {"index": tgt, "name": name(tgt), "prob": float(full[tgt]),
                   "prior": float(priors_np[tgt])},
        "esv_target": esv_t.tolist(), "esv_sum": float(esv_t.sum()),
        "efficiency_target": expected, "efficiency_gap": gap,
        "top5": [{"index": int(c), "name": name(int(c)), "prob": float(full[c])} for c in top],
        "probe_ignored_positions": bad,
    }
    (out / "esv.json").write_text(json.dumps(record, indent=2))
    np.savez(out / "esv.npz", esv=esvs, full_probs=full, priors=priors_np, frame_indices=frame_idx)
    plot_result(frames, frame_idx, esv_t,
                f"Play Fair ESV -- {name(tgt)} (p={full[tgt]:.2f}), {n} frames, "
                f"{'approx' if args.approximate else 'exact'}", out / "esv.png")
    print(f"[saved] {out}/esv.json, esv.npz, esv.png")


if __name__ == "__main__":
    main()
