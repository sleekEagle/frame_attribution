"""
trn_freeze_check.py -- does TRN react to exact repeats as such, apart from the content they lose?

TRN accepts exactly 8 frames, so eval_length_fixed_content.py's test for freezing (the same k
frames as a short clip and repeated to fill the slots) cannot be run on it: there is no short clip.
This script holds the content (almost) fixed and changes only whether repeated slots are exact
copies. For each video and each k it samples k frames (segment centres, TRN's protocol) and builds
three 8-frame clips:

    frozen      each of the k frames fills a block of consecutive slots, as freezing does
                (eval_length_fixed_content.repeat_to_length)
    neighbour   the same blocks, but the j-th extra slot of a block holds the video's raw frame
                j steps after the block's frame (j steps before it at the end of the video), so
                the clip shows nearly the same content with no exact repeats
    ref         the 8 segment-centre frames, the normal input (k = 8)
    ref_shift   the same 8 frames, each moved one raw frame later: how much the output changes
                when every frame is swapped for its raw neighbour, without any repetition

If frozen and neighbour agree about as closely as ref and ref_shift, TRN does not react to exact
repeats as such, and its accuracy loss under freezing comes from the content that is removed.

Per-video rows go to <out>_preds.csv (re-running resumes); the summary goes to <out>_summary.txt.

    python motivation/trn_freeze_check.py
    python motivation/trn_freeze_check.py --summary-only
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from torchcodec.decoders import VideoDecoder  # noqa: E402

from dataloaders.registry import get_dataloader  # noqa: E402
from eval_frame_count import logits  # noqa: E402
from eval_length_fixed_content import repeat_to_length  # noqa: E402
from models.registry import get_model  # noqa: E402
from models.video_utils import sample_segment_centers  # noqa: E402

MODEL = "trn_official"
SLOTS = 8
FIELDS = ["path", "gt", "ref_pred", "k", "variant", "pred", "p_gt", "logit_ref", "p_ref",
          "n_identical"]


def block_indices(base: np.ndarray, n_total: int, neighbour: bool) -> list:
    """Raw frame index for each of the SLOTS slots. base: the k sampled frame indices. With
    neighbour=False every slot of a block holds the block's frame (freezing); with neighbour=True
    the j-th slot of a block (j = 0, 1, ...) holds the raw frame j steps after it, or j steps
    before it if that would run past the end of the video."""
    k = len(base)
    owner = np.floor(np.arange(SLOTS) * k / SLOTS).astype(np.int64)
    out = []
    for s in range(SLOTS):
        j = s - int(np.argmax(owner == owner[s]))  # position of slot s inside its block
        f = int(base[owner[s]])
        if neighbour and j:
            f = f + j if f + j < n_total else f - j
        out.append(int(np.clip(f, 0, n_total - 1)))
    return out


def clips_for(n_total: int, ks) -> dict:
    """(k, variant) -> list of SLOTS raw frame indices."""
    out = {}
    ref = sample_segment_centers(n_total, SLOTS)
    out[(SLOTS, "ref")] = ref.tolist()
    out[(SLOTS, "ref_shift")] = [int(f + 1) if f + 1 < n_total else int(f - 1) for f in ref]
    for k in ks:
        base = sample_segment_centers(n_total, k)
        out[(k, "frozen")] = block_indices(base, n_total, neighbour=False)
        out[(k, "neighbour")] = block_indices(base, n_total, neighbour=True)
    return out


def load_done(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {r["path"] for r in csv.DictReader(f)}


def run(ks, limit, preds_csv):
    model = get_model(MODEL)
    names, paths = get_dataloader("ssv2_sampled")
    samples = [(n, p) for n, p in zip(names, paths) if n in model.label2id]
    if limit:
        samples = samples[:limit]
    done = load_done(preds_csv)
    new_file = not os.path.exists(preds_csv)
    with open(preds_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for i, (cls_name, path) in enumerate(samples):
            if str(path) in done:
                continue
            try:
                decoder = VideoDecoder(str(path))
                n_total = len(decoder)
            except Exception as e:  # unreadable video
                print(f"[warn] could not decode {path}: {e}")
                continue
            gt = model.label2id[cls_name]
            idx = clips_for(n_total, ks)
            frames = {key: decoder.get_frames_at(indices=v).data for key, v in idx.items()}
            out = {key: logits(MODEL, model, fr).float()[0] for key, fr in frames.items()}
            ref_pred = int(out[(SLOTS, "ref")].argmax())
            for (k, variant), z in out.items():
                fr = frames[(k, variant)]
                # number of slots whose frame is pixel-identical to the slot before it
                n_identical = int(sum(torch.equal(fr[s], fr[s - 1]) for s in range(1, SLOTS)))
                p = torch.softmax(z, -1)
                writer.writerow({"path": str(path), "gt": gt, "ref_pred": ref_pred, "k": k,
                                 "variant": variant, "pred": int(p.argmax()), "p_gt": float(p[gt]),
                                 "logit_ref": float(z[ref_pred]), "p_ref": float(p[ref_pred]),
                                 "n_identical": n_identical})
            f.flush()
            if i % 50 == 0:
                print(f"[{i + 1}/{len(samples)}] {Path(path).name}", flush=True)


def boot_ci(x, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    return np.percentile(means, [2.5, 97.5])


def summarize(preds_csv, ks) -> str:
    with open(preds_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    by = {}
    for r in rows:
        by.setdefault(r["path"], {})[(int(r["k"]), r["variant"])] = r
    lines = []

    def pair_stats(a_key, b_key, subset_name, vids):
        pairs = [(v[a_key], v[b_key]) for v in vids if a_key in v and b_key in v]
        if not pairs:
            return
        acc_a = 100 * np.mean([int(a["pred"]) == int(a["gt"]) for a, _ in pairs])
        acc_b = 100 * np.mean([int(b["pred"]) == int(b["gt"]) for _, b in pairs])
        same = 100 * np.mean([a["pred"] == b["pred"] for a, b in pairs])
        d_logit = np.array([float(a["logit_ref"]) - float(b["logit_ref"]) for a, b in pairs])
        d_p = np.array([float(a["p_ref"]) - float(b["p_ref"]) for a, b in pairs])
        lo, hi = boot_ci(d_p)
        ident_b = np.mean([int(b["n_identical"]) for _, b in pairs])
        lines.append(
            f"  {subset_name:<9} {a_key[1]:>9} vs {b_key[1]:<10} k={min(a_key[0], b_key[0])}  n={len(pairs):>4}  "
            f"acc {acc_a:5.1f} / {acc_b:5.1f}  same pred {same:5.1f}%  "
            f"median |d logit| {np.median(np.abs(d_logit)):.3f}  mean d logit {d_logit.mean():+.3f}  "
            f"mean d p_ref {d_p.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  "
            f"identical slots in 2nd clip {ident_b:.2f}")

    all_v = list(by.values())
    correct = [v for v in all_v if (SLOTS, "ref") in v and int(v[(SLOTS, "ref")]["pred"]) == int(v[(SLOTS, "ref")]["gt"])]
    for name, vids in (("all", all_v), ("correct", correct)):
        lines.append(f"{name}: {len(vids)} videos")
        pair_stats((SLOTS, "ref"), (SLOTS, "ref_shift"), name, vids)
        for k in ks:
            pair_stats((k, "frozen"), (k, "neighbour"), name, vids)
            # each variant against the full 8-frame clip
            for variant in ("frozen", "neighbour"):
                pair_stats((SLOTS, "ref"), (k, variant), name, vids)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", nargs="+", type=int, default=[4, 6])
    ap.add_argument("--limit", type=int, default=None, help="first N videos only (for a quick test)")
    ap.add_argument("--out", default="results/trn_freeze_check")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    preds_csv = f"{args.out}_preds.csv"
    if not args.summary_only:
        run(args.ks, args.limit, preds_csv)
    text = summarize(preds_csv, args.ks)
    with open(f"{args.out}_summary.txt", "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
