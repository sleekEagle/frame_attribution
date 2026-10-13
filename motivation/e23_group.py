"""
e23_group.py -- group-level Shapley values for the videos of an E2/E3 run, the yardstick for
the dilution tests in e23_metrics.py.

For every video in <out>/<model>_e23.jsonl, and every tree level k in --levels (the cut of the
DINOv2 tree into k segments, e23_run.level_groups), compute exact Shapley values with each
SEGMENT as one player (e1_common.group_shapley): how much each segment matters to the model as
a unit. No credit is divided inside a segment, so these values aren't diluted. Same frames,
class and removal as the E2/E3 run (the model's E1 removal; e23_run records it).

Writes <out>/<model>_e23_group.jsonl, one line per video (re-running resumes):
    {"video", "pred", "pred_here", "removal", "levels": {"4": {"groups": [[...], ...],
     "phi": [...], "n_evals": int}, "8": {...}}}
pred_here is the prediction recomputed in this run (it can differ from the E2/E3 run's in rare
cases because of fp16); the values explain the E2/E3 run's class "pred" either way.

    python motivation/e23_group.py --model mc3_18 --fp16
    python motivation/e23_group.py --model vjepa2 --fp16 --batch-size 64     # Kaggle
Cost: 2^k subsets per level (16 + 256 for k = 4, 8; x2 for late freeze).
"""
import argparse
import time
from pathlib import Path

from e1_common import (MODEL_SPECS, Evaluator, FrameBank, _decode, append_jsonl, group_shapley,
                       load_clip_model, load_prior, make_layout, read_jsonl, resolve_video_path,
                       stable_seed)
from e23_run import level_groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    ap.add_argument("--levels", nargs="+", type=int, default=None, help="default: n/4 and n/2")
    ap.add_argument("--prior", default="uniform")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/e23", help="folder with <model>_e23.jsonl")
    args = ap.parse_args()

    recs = read_jsonl(Path(args.out) / f"{args.model}_e23.jsonl")[:args.limit]
    out_path = Path(args.out) / f"{args.model}_e23_group.jsonl"
    done = {r["video"] for r in read_jsonl(out_path)}
    model = load_clip_model(args.model)
    prior = load_prior(model, args.prior)
    print(f"[e23 group] {args.model}: {len(recs)} videos ({len(done)} done)")

    for vi, rec in enumerate(recs):
        if rec["video"] in done:
            continue
        t0 = time.time()
        n = rec["n"]
        levels = args.levels or sorted({max(2, n // 4), max(2, n // 2)})
        frames = _decode(resolve_video_path(rec["video"]), rec["frame_indices"])
        bank = FrameBank(model, frames, "exact", stable_seed(rec["video"]))
        ev = Evaluator(model, bank, args.batch_size, args.fp16)
        layout = make_layout(range(n))
        pred_here = int(ev.logits([layout])[0].argmax())
        out = {}
        for k in levels:
            groups = level_groups(rec["nodes"], n, k)
            n0 = ev.n_evals
            phi = group_shapley(ev, layout, groups, rec["pred"], prior, rec["removal"])
            out[str(k)] = {"groups": groups, "phi": [float(x) for x in phi], "n_evals": ev.n_evals - n0}
        append_jsonl(out_path, {"model": args.model, "video": rec["video"], "pred": rec["pred"],
                                "pred_here": pred_here, "removal": rec["removal"], "levels": out,
                                "secs": round(time.time() - t0, 2)})
        name = rec["video"].replace("\\", "/").split("/")[-1]
        print(f"[{vi + 1}/{len(recs)}] {name} ({time.time() - t0:.1f}s)")
    print(f"[e23 group] wrote {out_path}")


if __name__ == "__main__":
    main()
