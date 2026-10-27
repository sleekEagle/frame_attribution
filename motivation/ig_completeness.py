"""
ig_completeness.py -- do Integrated Gradients scores add up to the logit change, and how many steps
does that take? IG (e1_common.attr_integrated_gradients) uses a zero baseline in normalised input
space and a midpoint Riemann sum, so the per-slot scores should add up to
logit(x) - logit(baseline). With too few steps they don't; the 32 steps used in E1 were enough on
MC3-18 (motivation/whole_clip_repeat.py) but not on V-JEPA2 in a 2-video smoke test.

For each video, two clips of E1: the reference clip and the exact-copy target clip with the most
copies that kept the prediction. For each number of steps, one JSONL line per clip with the sum of
the scores, the logit change, and the scores themselves, so the per-copy ratio of E1 can be
recomputed per number of steps.

    python motivation/ig_completeness.py --model mc3_18 --limit 10
    python motivation/ig_completeness.py --model vjepa2 --limit 4 --steps 32 128 256 --shard 0 --num-shards 2
    python motivation/ig_completeness.py --model vjepa2 --summary
"""
import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch

from e1_common import (FrameBank, append_jsonl, attr_integrated_gradients, load_clip_model, read_jsonl,
                       resolve_video_path)


def pick_conditions(rec):
    """The reference clip and the exact-copy target clip with the largest m whose prediction held."""
    conds = [c for c in rec["conditions"] if c["dup"] == "exact"]
    ref = [c for c in conds if c["kind"] == "reference"]
    tgt = [c for c in conds if c["kind"] == "target" and c["stats"]["same_pred"]]
    out = ref[:1]
    if tgt:
        out.append(max(tgt, key=lambda c: (c["m"], -c["target"])))
    return out


def summary(out_path: Path):
    paths = [out_path] + sorted(Path(p) for p in glob.glob(str(out_path.with_suffix("")) + ".shard*.jsonl"))
    rows = [r for p in paths for r in read_jsonl(p)]
    if not rows:
        print("no rows")
        return
    q = lambda a: f"{np.median(a):7.3f} [{np.quantile(a, .25):6.3f},{np.quantile(a, .75):6.3f}]"  # noqa: E731
    print(f"{rows[0]['model']}: sum of scores / logit change (1 = scores add up), and the E1 per-copy ratio")
    print(f"{'steps':>6}{'clips':>7}  {'sum/gap':>24}  {'|sum/gap - 1|':>24}  {'videos':>7}  {'per-copy ratio':>24}")
    for steps in sorted({r["steps"] for r in rows}):
        rs = [r for r in rows if r["steps"] == steps]
        ratio = np.array([r["sum"] / r["gap"] for r in rs if r["gap"] > 0])
        # per-copy ratio: mean score of the target's copies in the target clip / its score in the reference
        by_vid = {}
        for r in rs:
            by_vid.setdefault(r["video"], {})[r["kind"]] = r
        pc = []
        for d in by_vid.values():
            if "reference" in d and "target" in d:
                t, ref = d["target"]["target"], d["reference"]
                s_ref = np.mean([s for s, c in zip(ref["scores"], ref["content"]) if c == t])
                s_tgt = np.mean([s for s, c in zip(d["target"]["scores"], d["target"]["content"]) if c == t])
                if s_ref > 0:
                    pc.append(s_tgt / s_ref)
        pcs = q(pc) if pc else "-"
        print(f"{steps:>6}{len(ratio):>7}  {q(ratio):>24}  {q(np.abs(ratio - 1)):>24}  {len(pc):>7}  {pcs:>24}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", nargs="+", type=int, default=[32, 128, 256])
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--ig-batch", type=int, default=1)
    ap.add_argument("--conditions", default=None, help="default: results/e1/<model>_conditions.jsonl")
    ap.add_argument("--out", default="results/e1")
    ap.add_argument("--shard", type=int, default=None, help="with --num-shards: only videos i with i %% num-shards == shard")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    out_path = Path(args.out) / f"{args.model}_ig_completeness.jsonl"
    if args.summary:
        return summary(out_path)
    records = read_jsonl(Path(args.conditions or f"results/e1/{args.model}_conditions.jsonl"))[:args.limit]
    if args.shard is not None:
        records = [r for i, r in enumerate(records) if i % args.num_shards == args.shard]
        out_path = out_path.with_suffix(f".shard{args.shard}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[ig completeness] {args.model}: {len(records)} videos, steps={args.steps} -> {out_path}", flush=True)

    model = load_clip_model(args.model)
    done = {(r["video"], r["cond_id"], r["steps"]) for r in read_jsonl(out_path)}
    for vi, rec in enumerate(records):
        t_vid = time.time()
        frames, _ = model.load_frames(resolve_video_path(rec["video"]))
        cframes = frames[0::2] if rec["design"] == "realloc" else frames
        bank = FrameBank(model, cframes, "exact", rec["bank_seed"])
        cls = rec["pred"]
        for cond in pick_conditions(rec):
            todo = [s for s in args.steps if (rec["video"], cond["id"], s) not in done]
            if not todo:
                continue
            x = bank.clip(cond["layout"])
            with torch.no_grad():  # full precision, as IG itself
                gap = float(model.forward_clips(x[None])[0, cls] - model.forward_clips(torch.zeros_like(x)[None])[0, cls])
            for steps in todo:
                t0 = time.time()
                scores = attr_integrated_gradients(model, bank, cond["layout"], cls, steps, args.ig_batch)
                append_jsonl(out_path, {
                    "model": args.model, "video": rec["video"], "cond_id": cond["id"], "kind": cond["kind"],
                    "target": cond.get("target"), "m": cond.get("m"), "steps": steps, "cls": cls,
                    "sum": float(np.sum(scores)), "gap": gap, "scores": [float(s) for s in scores],
                    "content": cond["layout"]["content"], "secs": round(time.time() - t0, 3)})
                print(f"  {cond['id']:<28} steps={steps:<4} sum/gap={np.sum(scores) / gap:7.3f} ({time.time() - t0:.0f}s)", flush=True)
        print(f"[{vi + 1}/{len(records)}] {Path(rec['video']).name} done ({time.time() - t_vid:.1f}s)", flush=True)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
