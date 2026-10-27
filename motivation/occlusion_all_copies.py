"""
occlusion_all_copies.py -- occlusion of EVERY copy of a content, against its importance I.

Check 1 of occlusion_check.py compares single-slot occlusion with I(c), which removes every copy
of c. On the realloc models (m_ref = 2) those don't match. Here, on each E1 reference clip, every
copy of content c is replaced by the flat frame (0 in normalised space) at once:

    occ_all(c) = P(x) - P(x with every copy of c flat)
    I(c)       = P(x) - P(x with every copy of c removed by the model's removal)   (recomputed)
    D(c)       = occ_all(c) - I(c) = P(removed) - P(flat)

so both sides remove the same content, and D is the effect of the flat frame relative to the
removal operator (for freeze: relative to repeating the neighbouring kept frames).
I is recomputed and compared with the stored I_ref as a check that the clips are rebuilt right.

    python motivation/occlusion_all_copies.py --model r3d
    python motivation/occlusion_all_copies.py --model r3d --summary     # no model, just the table
"""
import argparse
import time
from pathlib import Path

import numpy as np

from e1_common import (MODEL_SPECS, Evaluator, FrameBank, append_jsonl, load_clip_model, load_prior,
                       read_jsonl, resolve_video_path)


def run(args, out_path):
    import torch
    model = load_clip_model(args.model)
    records = read_jsonl(Path(args.out) / f"{args.model}_conditions.jsonl")
    if args.limit:
        records = records[:args.limit]
    done = {r["video"] for r in read_jsonl(out_path)}
    for vi, rec in enumerate(records):
        path = rec["video"]
        if path in done:
            continue
        t0 = time.time()
        prior = load_prior(model, rec.get("prior"))
        frames, _ = model.load_frames(resolve_video_path(path))
        cframes = frames[0::2] if rec["design"] == "realloc" else frames
        cls = rec["pred"]
        layout = next(c for c in rec["conditions"] if c["id"] == "exact|reference")["layout"]
        bank = FrameBank(model, cframes, "exact", rec["bank_seed"])
        ev = Evaluator(model, bank, args.batch_size, False)
        content = np.asarray(layout["content"])
        contents = sorted(set(content.tolist()))

        keeps = np.stack([np.ones(len(content), bool)] + [content != c for c in contents])
        v = ev.value(layout, keeps, rec["removal"], cls, prior)

        x = bank.clip(layout)
        xs = x[None].repeat(len(contents) + 1, 1, 1, 1, 1)
        for j, c in enumerate(contents):
            xs[j + 1, torch.from_numpy(content == c).to(x.device)] = 0.0
        with torch.no_grad():
            logits = torch.cat([model.forward_clips(xs[s:s + args.batch_size])
                                for s in range(0, len(xs), args.batch_size)])
        p = torch.softmax(logits.float(), -1)[:, cls].cpu().numpy()

        append_jsonl(out_path, {
            "model": args.model, "video": path, "cls": cls, "contents": contents,
            "n_copies": [int((content == c).sum()) for c in contents],
            "p_full_removal": float(v[0]), "p_removed": [float(a) for a in v[1:]],
            "p_full_flat": float(p[0]), "p_flat": [float(a) for a in p[1:]],
            "I_stored": [float(rec["I_ref"][str(c)]) for c in contents]})
        print(f"[{vi + 1}/{len(records)}] {Path(path.replace(chr(92), '/')).name} "
              f"({time.time() - t0:.1f}s)")


def summary(out_path, name):
    recs = read_jsonl(out_path)
    I, occ, I_stored = [], [], []
    for r in recs:
        I += [r["p_full_removal"] - a for a in r["p_removed"]]
        occ += [r["p_full_flat"] - a for a in r["p_flat"]]
        I_stored += r["I_stored"]
    I, occ, I_stored = map(np.asarray, (I, occ, I_stored))
    d = occ - I
    low = np.abs(I) <= np.quantile(np.abs(I), 1 / 3)
    from scipy.stats import spearmanr
    print(f"\n{name}: {len(recs)} videos, {len(I)} contents "
          f"(copies per content: {sorted({n for r in recs for n in r['n_copies']})})")
    print(f"  check: max |I recomputed - I stored| = {np.max(np.abs(I - I_stored)):.2e}")
    print(f"  all contents:   median |I| {np.median(np.abs(I)):.4f}   median occ_all {np.median(occ):.4f}"
          f"   median D {np.median(d):.4f}   median |D| {np.median(np.abs(d)):.4f}"
          f"   rho(occ_all, I) {spearmanr(occ, I)[0]:.2f}")
    print(f"  low-|I| third:  median |I| {np.median(np.abs(I[low])):.4f}"
          f"   median occ_all {np.median(occ[low]):.4f}"
          f"   share |occ_all| > |I| {np.mean(np.abs(occ[low]) > np.abs(I[low])):.2f}"
          f"   median D {np.median(d[low]):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--summary", action="store_true", help="only print the table")
    ap.add_argument("--out", default="results/e1")
    args = ap.parse_args()
    out_path = Path(args.out) / f"{args.model}_occ_all_copies.jsonl"
    if not args.summary:
        run(args, out_path)
    summary(out_path, args.model)


if __name__ == "__main__":
    main()
