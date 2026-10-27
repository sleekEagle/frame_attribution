"""
whole_clip_repeat.py -- every frame of a clip shown m times (C = N), on a model that accepts
longer clips (MC3-18). Tests Proposition 2 of the paper: an efficient method (Shapley) gives
r_phi = r_I / m exactly; Integrated Gradients does so on the logit scale; Grad-CAM is expected to
fall by about m through the model's average pooling.

For each video of the E1 records (same predicted class, exact copies, deletion), clips with
m = 1, 2, 3 copies of every frame are attributed. One JSONL line per (video, m, method) with the
per-slot scores and the model outputs needed for r_I (probability) and r_F (logit, IG baseline).

    python motivation/whole_clip_repeat.py --model mc3_18 --limit 100 --methods ig gradcam loo_drop occlusion
    python motivation/whole_clip_repeat.py --model mc3_18 --limit 30 --methods shapley_drop
    python motivation/whole_clip_repeat.py --model mc3_18 --summary

On Kaggle (2 GPUs), one process per GPU; each writes <model>_whole_repeat.shard<s>.jsonl and
--summary reads all shard files (see motivation/kaggle_whole_repeat.sh):
    CUDA_VISIBLE_DEVICES=0 python motivation/whole_clip_repeat.py --model vjepa2 --fp16 --shard 0 --num-shards 2 ...
"""
import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch

from e1_common import (Evaluator, FrameBank, append_jsonl, load_clip_model, load_prior, read_jsonl,
                       resolve_video_path, run_method)


def repeat_layout(n: int, m: int) -> dict:
    return {"content": [c for c in range(n) for _ in range(m)], "copy": [k for _ in range(n) for k in range(m)]}


def summary(out_path: Path):
    paths = [out_path] + sorted(Path(p) for p in glob.glob(str(out_path.with_suffix("")) + ".shard*.jsonl"))
    rows = [r for p in paths for r in read_jsonl(p)]
    print("read", ", ".join(f"{p.name} ({len(read_jsonl(p))})" for p in paths if p.exists()))
    by = {}
    for r in rows:
        by.setdefault((r["video"], r["method"]), {})[r["m"]] = r
    res = {}
    for (video, method), d in by.items():
        if 1 not in d:
            continue
        b = d[1]
        for m, r in d.items():
            if m == 1 or not (r["same_pred"] and b["same_pred"]):
                continue
            dI = b["p_full"] - b["p_empty"]
            dF = b["logit_full"] - b["logit_base"]
            if dI <= 0 or dF <= 0 or np.mean(b["scores"]) <= 0:
                continue
            r_I = (r["p_full"] - r["p_empty"]) / dI
            r_F = (r["logit_full"] - r["logit_base"]) / dF
            r_phi = np.mean(r["scores"]) / np.mean(b["scores"])
            res.setdefault((method, m), []).append((r_phi * m / r_I, r_phi * m / r_F, r_phi / r_I, r_I))
    q = lambda a: f"{np.median(a):6.2f} [{np.quantile(a, .25):5.2f},{np.quantile(a, .75):5.2f}]"  # noqa: E731
    print(f"{'method':<14}{'m':>2}{'n':>5}  {'r_phi/(r_I/m)':>22}  {'r_phi/(r_F/m)':>22}  {'r_phi/r_I':>22}  {'r_I':>22}")
    for (method, m), v in sorted(res.items()):
        v = np.array(v)
        print(f"{method:<14}{m:>2}{len(v):>5}  {q(v[:, 0]):>22}  {q(v[:, 1]):>22}  {q(v[:, 2]):>22}  {q(v[:, 3]):>22}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mc3_18")
    ap.add_argument("--methods", nargs="+", default=["ig", "gradcam", "loo_drop", "occlusion"])
    ap.add_argument("--ms", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--n-perm", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--fp16", action="store_true", help="half precision for the batched evaluations (Shapley, LOO)")
    ap.add_argument("--ig-steps", type=int, default=32)
    ap.add_argument("--ig-batch", type=int, default=2, help="lower it if IG runs out of GPU memory")
    ap.add_argument("--shard", type=int, default=None, help="with --num-shards: only videos i with i %% num-shards == shard")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--conditions", default=None, help="default: results/e1/<model>_conditions.jsonl")
    ap.add_argument("--out", default="results/e1")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    out_path = Path(args.out) / f"{args.model}_whole_repeat.jsonl"
    if args.summary:
        return summary(out_path)
    records = read_jsonl(Path(args.conditions or f"results/e1/{args.model}_conditions.jsonl"))[:args.limit]
    if args.shard is not None:  # split the videos over several GPUs; --summary merges the files
        records = [r for i, r in enumerate(records) if i % args.num_shards == args.shard]
        out_path = out_path.with_suffix(f".shard{args.shard}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[whole repeat] {args.model}: {len(records)} videos, m={args.ms}, methods={args.methods} -> {out_path}", flush=True)

    model = load_clip_model(args.model)
    assert model.spec["design"] in ("insert", "replace"), "needs a model that accepts longer clips"
    done = {(r["video"], r["m"], r["method"]) for r in read_jsonl(out_path)}
    for vi, rec in enumerate(records):
        t_vid = time.time()
        prior = load_prior(model, rec.get("prior"))
        frames, _ = model.load_frames(resolve_video_path(rec["video"]))
        bank = FrameBank(model, frames, "exact", rec["bank_seed"])
        ev = Evaluator(model, bank, args.batch_size, args.fp16)
        cls = rec["pred"]
        n = rec["slots"]
        for m in args.ms:
            todo = [meth for meth in args.methods if (rec["video"], m, meth) not in done]
            if not todo:
                continue
            lay = repeat_layout(n, m)
            logits = ev.logits([lay])[0]
            p = torch.softmax(torch.from_numpy(logits), -1).numpy()
            with torch.no_grad():   # IG baseline: flat frames (zeros in normalised space)
                base = model.forward_clips(torch.zeros_like(bank.clip(lay))[None]).float().cpu().numpy()[0]
            for meth in todo:
                t0 = time.time()
                scores, n_ev = run_method(meth, model, bank, ev, lay, cls, prior, seed=args.seed,
                                          n_perm=args.n_perm, ig_steps=args.ig_steps, ig_batch=args.ig_batch)
                append_jsonl(out_path, {
                    "model": args.model, "video": rec["video"], "m": m, "method": meth, "cls": cls,
                    "same_pred": bool(int(np.argmax(logits)) == cls),
                    "p_full": float(p[cls]), "p_empty": float(prior[cls]),
                    "logit_full": float(logits[cls]), "logit_base": float(base[cls]),
                    "scores": [float(s) for s in scores], "n_evals": int(n_ev),
                    "secs": round(time.time() - t0, 3)})
                done.add((rec["video"], m, meth))
        print(f"[{vi + 1}/{len(records)}] {Path(rec['video']).name} done ({time.time() - t_vid:.1f}s)", flush=True)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
