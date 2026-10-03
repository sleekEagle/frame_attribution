"""
eval_frame_count.py -- top-1 accuracy of r3d / vjepa2 / videomae as a function of the number of
input frames. All three are 16-frame models; each clip is re-sampled with N frames spread over
the *whole* video (same temporal coverage, sparser sampling), using each model's own sampling
protocol, so the N=16 row reproduces eval_accuracy.py's numbers.

    python eval_frame_count.py                                   # all 3 models, default N list
    python eval_frame_count.py --models r3d videomae --limit 2000
    python eval_frame_count.py --models r3d --datasets r3d=ucf101_test
    python eval_frame_count.py --frames 16 8 4 2 1

How each model copes with N != 16:
  r3d       fully convolutional (padded strided temporal convs/pools + AdaptiveAvgPool3d) ->
            any N >= 1 runs natively.
  vjepa2    3D RoPE positions are derived from the token index and the classifier is an
            attentive pooler -> any even N runs natively. N=1 is also native: the HF model
            duplicates the frame to fill its 2-frame tubelet (VJEPA2Embeddings.forward).
  videomae  fixed sinusoidal position table sized for 16 frames (1568 tokens). The table is
            1D over tokens flattened in (t, h, w) order, so its first (N/2)*196 rows are exactly
            temporal slots 0..N/2-1 -- we slice it to the token count. N=1 is NOT native
            (tubelet=2); we repeat the frame twice (what vjepa2 does internally) and mark the
            result as `repeated` in the summary.

Per-sample predictions are appended to <out>_preds.csv as they are produced, so an interrupted
run resumes where it stopped; the accuracy table is written to <out>_summary.csv.
"""
import argparse
import csv
import os
import random
from contextlib import contextmanager

import numpy as np
import torch
from torchcodec.decoders import VideoDecoder

from dataloaders.registry import get_dataloader
from models.registry import MODEL_DATASET, get_model
from models.video_utils import sample_segment_centers

DEFAULT_FRAMES = [16, 14, 12, 10, 8, 6, 4, 2, 1]
MODELS = ["r3d", "vjepa2", "videomae"]


# ---------------------------------------------------------------------------------------------
# frame sampling -- each model's own protocol, generalised to N frames
# ---------------------------------------------------------------------------------------------
def frame_indices(model_name: str, n_total: int, n: int) -> list:
    if model_name == "vjepa2":
        # models/ssv2.py's VJEPA2.sample_frames uses linspace(0, L-1, 16); for N=1 that would
        # always pick frame 0, so take the centre frame instead
        if n == 1:
            return [n_total // 2]
        return np.linspace(0, n_total - 1, n, dtype=int).tolist()
    # r3d / videomae: models/video_utils.sample_frames (segment centres)
    return sample_segment_centers(n_total, n).tolist()


# ---------------------------------------------------------------------------------------------
# per-model forward on a (N, C, H, W) uint8 frame tensor
# ---------------------------------------------------------------------------------------------
@contextmanager
def videomae_pos_embed(model, n_frames: int):
    """Temporarily slice VideoMAE's fixed sinusoidal position table to n_frames' token count."""
    emb = model.model.videomae.embeddings
    cfg = model.model.config
    full = emb.position_embeddings
    n_tokens = (n_frames // cfg.tubelet_size) * (cfg.image_size // cfg.patch_size) ** 2
    emb.position_embeddings = full[:, :n_tokens]
    try:
        yield
    finally:
        emb.position_embeddings = full


def predict(model_name: str, model, frames: torch.Tensor) -> int:
    return logits(model_name, model, frames).argmax(-1).item()


def logits(model_name: str, model, frames: torch.Tensor) -> torch.Tensor:
    """(1, num_classes) logits for a (N, C, H, W) uint8 frame tensor."""
    with torch.no_grad():
        if model_name == "r3d":
            out = model.model(model.preprocess(frames).to(model.device))
        elif model_name == "videomae":
            tubelet = model.model.config.tubelet_size
            if frames.shape[0] % tubelet:  # only hit for N=1 with the default frame list
                frames = frames.repeat_interleave(tubelet, dim=0)
            with videomae_pos_embed(model, frames.shape[0]):
                out = model.model(model.preprocess(frames).to(model.device)).logits
        elif model_name == "vjepa2":
            inputs = model.processor(frames, return_tensors="pt").to(model.device)
            out = model.model(**inputs).logits
        else:
            raise ValueError(model_name)
    return out


def support_note(model_name: str, model, n: int) -> str:
    """'native' if the model runs N frames without input modification, else what we did."""
    if model_name == "videomae" and n % model.model.config.tubelet_size:
        return "repeated"
    return "native"


# ---------------------------------------------------------------------------------------------
# experiment loop
# ---------------------------------------------------------------------------------------------
PRED_FIELDS = ["model", "dataset", "path", "gt", "n_frames", "pred"]


def load_done(preds_csv: str) -> set:
    if not os.path.exists(preds_csv):
        return set()
    with open(preds_csv, newline="") as f:
        return {(r["model"], r["dataset"], r["path"], int(r["n_frames"])) for r in csv.DictReader(f)}


def run_model(model_name, dataset_name, frame_counts, limit, seed, preds_csv):
    model = get_model(model_name)
    d_names, paths = get_dataloader(dataset_name)
    samples = [(n, p) for n, p in zip(d_names, paths) if n in model.label2id]
    n_skipped = len(paths) - len(samples)
    if n_skipped:
        print(f"[warn] {model_name}: {n_skipped} samples have no label2id entry -- skipped")
    if limit and limit < len(samples):
        samples = random.Random(seed).sample(samples, limit)

    done = load_done(preds_csv)
    new_file = not os.path.exists(preds_csv)
    with open(preds_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PRED_FIELDS)
        if new_file:
            writer.writeheader()
        for i, (cls_name, path) in enumerate(samples):
            todo = [n for n in frame_counts
                    if (model_name, dataset_name, str(path), n) not in done]
            if not todo:
                continue
            print(f"{model_name}: {i / len(samples) * 100:.2f} % done", end="\r")
            try:
                decoder = VideoDecoder(str(path))
                n_total = len(decoder)
            except Exception as e:  # unreadable video -- skip it for every N
                print(f"\n[warn] could not decode {path}: {e}")
                continue
            gt = model.label2id[cls_name]
            for n in todo:
                frames = decoder.get_frames_at(indices=frame_indices(model_name, n_total, n)).data
                writer.writerow({"model": model_name, "dataset": dataset_name, "path": str(path),
                                 "gt": gt, "n_frames": n, "pred": predict(model_name, model, frames)})
            f.flush()
    print()
    return model


def summarize(preds_csv, models, model_objs, frame_counts):
    with open(preds_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    table = []
    for m, dataset_name in models:
        for n in frame_counts:
            hits = [int(r["gt"]) == int(r["pred"]) for r in rows
                    if r["model"] == m and r["dataset"] == dataset_name and int(r["n_frames"]) == n]
            if not hits:
                continue
            table.append({"model": m, "dataset": dataset_name, "n_frames": n,
                          "accuracy": 100 * sum(hits) / len(hits), "n_samples": len(hits),
                          "support": support_note(m, model_objs[m], n)})
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--frames", nargs="+", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--datasets", nargs="*", default=[],
                        help="per-model dataset overrides, e.g. r3d=ucf101_test videomae=ucf101_test")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate a random subset of this many clips per model")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/frame_count")
    args = parser.parse_args()

    overrides = dict(kv.split("=", 1) for kv in args.datasets)
    models = [(m, overrides.get(m, MODEL_DATASET[m])) for m in args.models]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    preds_csv = f"{args.out}_preds.csv"

    model_objs = {}
    for m, dataset_name in models:
        print(f"=== {m} on {dataset_name}, frames {args.frames}")
        model_objs[m] = run_model(m, dataset_name, args.frames, args.limit, args.seed, preds_csv)

    table = summarize(preds_csv, models, model_objs, args.frames)
    with open(f"{args.out}_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    print(f"\n{'model':<10}{'dataset':<14}{'frames':>7}{'acc %':>9}{'n':>8}  support")
    for r in table:
        print(f"{r['model']:<10}{r['dataset']:<14}{r['n_frames']:>7}{r['accuracy']:>9.2f}"
              f"{r['n_samples']:>8}  {r['support']}")


if __name__ == "__main__":
    main()
