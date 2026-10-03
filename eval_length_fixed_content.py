"""
eval_length_fixed_content.py -- does a model's output depend on input LENGTH when the CONTENT is
held fixed?

eval_frame_count.py varies the number of frames T by resampling the video, so length and
information content change together (every added frame is a new frame). E1's drop-based methods
(Shapley-drop, Play Fair) instead add copies of frames that are already present, so the relevant
question is whether the output changes with length alone. For each video and each k this script
samples k distinct frames (each model's own protocol, as in eval_frame_count.py) and evaluates
two inputs built from exactly those frames:

    native     the k frames on their own                     (input length k)
    repeated   each of the k frames repeated in consecutive   (input length --target-len, 16)
               slots, block sizes differing by at most 1

If the model's output depends only on content, the two variants agree for every k. A gap
between them is an effect of input length at fixed content.

    python eval_length_fixed_content.py --models vjepa2 --datasets vjepa2=ssv2_sampled --limit 500
    python eval_length_fixed_content.py --models r3d --datasets r3d=ucf101_test --limit 500

Per-video results are appended to <out>_preds.csv as they are produced (re-running resumes);
the table is written to <out>_summary.csv:
    accuracy and mean P(ground truth) per (k, variant), and the share of videos where the
    native and repeated inputs give the same predicted class.
"""
import argparse
import csv
import os
import random

import numpy as np
import torch
from torchcodec.decoders import VideoDecoder

from dataloaders.registry import get_dataloader
from eval_frame_count import frame_indices, logits
from models.registry import MODEL_DATASET, get_model

DEFAULT_KS = [1, 2, 4, 6, 8, 10, 12, 14, 16]
MODELS = ["r3d", "vjepa2", "videomae"]
FIELDS = ["model", "dataset", "path", "gt", "k", "variant", "pred", "p_gt"]


def repeat_to_length(frames: torch.Tensor, length: int) -> torch.Tensor:
    """(k, C, H, W) -> (length, C, H, W): frame j fills a block of consecutive slots; block sizes
    differ by at most one (slot s holds frame floor(s * k / length))."""
    k = frames.shape[0]
    idx = np.floor(np.arange(length) * k / length).astype(np.int64)
    return frames[torch.from_numpy(idx)]


def load_done(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {(r["model"], r["dataset"], r["path"], int(r["k"]), r["variant"]) for r in csv.DictReader(f)}


def run_model(model_name, dataset_name, ks, target_len, limit, seed, preds_csv):
    model = get_model(model_name)
    d_names, paths = get_dataloader(dataset_name)
    samples = [(n, p) for n, p in zip(d_names, paths) if n in model.label2id]
    if limit and limit < len(samples):
        samples = random.Random(seed).sample(samples, limit)  # same subset as eval_frame_count.py

    done = load_done(preds_csv)
    new_file = not os.path.exists(preds_csv)
    with open(preds_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for i, (cls_name, path) in enumerate(samples):
            todo = [(k, v) for k in ks for v in ("native", "repeated")
                    if (model_name, dataset_name, str(path), k, v) not in done]
            if not todo:
                continue
            print(f"{model_name}: {i / len(samples) * 100:.1f} % done", flush=True)
            try:
                decoder = VideoDecoder(str(path))
                n_total = len(decoder)
            except Exception as e:  # unreadable video
                print(f"[warn] could not decode {path}: {e}")
                continue
            gt = model.label2id[cls_name]
            frames_k = {}
            for k, variant in todo:
                if k not in frames_k:
                    frames_k[k] = decoder.get_frames_at(indices=frame_indices(model_name, n_total, k)).data
                frames = frames_k[k] if variant == "native" else repeat_to_length(frames_k[k], target_len)
                p = torch.softmax(logits(model_name, model, frames).float(), -1)[0]
                writer.writerow({"model": model_name, "dataset": dataset_name, "path": str(path),
                                 "gt": gt, "k": k, "variant": variant,
                                 "pred": int(p.argmax()), "p_gt": float(p[gt])})
            f.flush()


def summarize(preds_csv, models, ks):
    with open(preds_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    by = {}  # (model, dataset, path, k) -> {variant: row}
    for r in rows:
        by.setdefault((r["model"], r["dataset"], r["path"], int(r["k"])), {})[r["variant"]] = r
    table = []
    for m, dataset_name in models:
        for k in ks:
            pairs = [v for (mm, dd, _, kk), v in by.items()
                     if mm == m and dd == dataset_name and kk == k and len(v) == 2]
            if not pairs:
                continue
            out = {"model": m, "dataset": dataset_name, "k": k, "n": len(pairs)}
            for variant in ("native", "repeated"):
                out[f"acc_{variant}"] = 100 * np.mean([int(v[variant]["pred"]) == int(v[variant]["gt"]) for v in pairs])
                out[f"p_gt_{variant}"] = float(np.mean([float(v[variant]["p_gt"]) for v in pairs]))
            out["same_pred_pct"] = 100 * np.mean([v["native"]["pred"] == v["repeated"]["pred"] for v in pairs])
            table.append(out)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["vjepa2"], choices=MODELS)
    ap.add_argument("--ks", nargs="+", type=int, default=DEFAULT_KS)
    ap.add_argument("--target-len", type=int, default=16, help="length of the 'repeated' input")
    ap.add_argument("--datasets", nargs="*", default=[],
                    help="per-model dataset overrides, e.g. vjepa2=ssv2_sampled r3d=ucf101_test")
    ap.add_argument("--limit", type=int, default=None, help="random subset of this many videos")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/length_fixed_content")
    args = ap.parse_args()

    overrides = dict(kv.split("=", 1) for kv in args.datasets)
    models = [(m, overrides.get(m, MODEL_DATASET[m])) for m in args.models]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    preds_csv = f"{args.out}_preds.csv"

    for m, dataset_name in models:
        print(f"=== {m} on {dataset_name}, k = {args.ks}, repeated to {args.target_len} frames")
        run_model(m, dataset_name, args.ks, args.target_len, args.limit, args.seed, preds_csv)

    table = summarize(preds_csv, models, args.ks)
    with open(f"{args.out}_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    print(f"\n{'model':<9}{'k':>4}{'n':>6}   {'acc native':>10} {'acc repeated':>12}   "
          f"{'P(gt) native':>12} {'P(gt) repeated':>14}   {'same pred %':>11}")
    for r in table:
        print(f"{r['model']:<9}{r['k']:>4}{r['n']:>6}   {r['acc_native']:>10.1f} {r['acc_repeated']:>12.1f}   "
              f"{r['p_gt_native']:>12.3f} {r['p_gt_repeated']:>14.3f}   {r['same_pred_pct']:>11.1f}")


if __name__ == "__main__":
    main()
