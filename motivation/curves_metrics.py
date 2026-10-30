"""
curves_metrics.py -- readouts of the deletion and insertion curves (curves.py).

Each curve is normalized by the clip's own contribution: y = (P - prior) / (P_full - prior), so
y = 1 is the full clip and y = 0 an empty clip. Clips where P_full - prior < --min-gap are left
out (nothing to explain). Per clip and method:
  deletion AUC    mean of y over the n + 1 deletion steps   (lower = the order finds the needed
                  frames first)
  insertion AUC   mean of y over the n + 1 insertion steps  (higher = the first frames suffice)
  del@25%         y after removing the top quarter of the frames
  ins@25%         y with only the top quarter kept
Means over clips with 95% bootstrap CIs (resampling videos), and the paired difference to
frame-level Shapley on the same clips (negative = better for deletion, positive = better for
insertion).

    python motivation/curves_metrics.py --model r3d_18 --source e23
Reads <out>/<model>_curves_<source>.jsonl and its .shard<s>.jsonl files; writes
<out>/<model>_curves_<source>_summary.csv.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from e1_common import read_jsonl

SHAPLEY = ("shapley_drop", "shapley_freeze")


def boot(per_video, n_boot, seed=0):
    vids = list(per_video)
    allv = [x for v in vids for x in per_video[v]]
    rng = np.random.default_rng(seed)
    bs = [np.mean([x for i in rng.choice(len(vids), len(vids)) for x in per_video[vids[i]]]) for _ in range(n_boot)]
    return float(np.mean(allv)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True, choices=["e23", "fr"])
    ap.add_argument("--out", default="results/curves")
    ap.add_argument("--min-gap", type=float, default=0.05)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    # the unsharded file plus any shard files (Kaggle runs one shard per GPU); a clip in both counts once
    stem = f"{args.model}_curves_{args.source}"
    recs, seen = [], set()
    for p in [Path(args.out) / f"{stem}.jsonl"] + sorted(Path(args.out).glob(f"{stem}.shard*.jsonl")):
        for r in read_jsonl(p):
            if (r["video"], r["clip"]) not in seen:
                seen.add((r["video"], r["clip"]))
                recs.append(r)

    def nest():  # clip -> readout -> method -> video -> [x]
        return defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    vals, diffs = nest(), nest()
    kept = defaultdict(int)
    for r in recs:
        gap = r["p_full"] - r["prior"]
        if gap < args.min_gap:
            continue
        kept[r["clip"]] += 1
        n = r["n"]
        q = max(1, round(n / 4))
        ro = {}
        for meth, c in r["curves"].items():
            d = (np.asarray(c["del"]) - r["prior"]) / gap
            i = (np.asarray(c["ins"]) - r["prior"]) / gap
            ro[meth] = {"deletion_auc": float(d.mean()), "insertion_auc": float(i.mean()),
                        "del_at_25": float(d[q]), "ins_at_25": float(i[q])}
        base = next((m for m in SHAPLEY if m in ro), None)
        for meth, x in ro.items():
            for k, v in x.items():
                vals[r["clip"]][k][meth][r["video"]].append(v)
                if base:
                    diffs[r["clip"]][k][meth][r["video"]].append(v - ro[base][k])

    rows = []
    for clip in sorted(vals):
        for k in ("deletion_auc", "insertion_auc", "del_at_25", "ins_at_25"):
            for meth in sorted(vals[clip][k]):
                mu, lo, hi = boot(vals[clip][k][meth], args.boot)
                dm, dlo, dhi = boot(diffs[clip][k][meth], args.boot)
                rows.append({"clip": clip, "readout": k, "method": meth, "mean": mu, "ci_lo": lo, "ci_hi": hi,
                             "diff_vs_shapley": dm, "diff_lo": dlo, "diff_hi": dhi,
                             "n_clips": sum(len(v) for v in vals[clip][k][meth].values())})
    out = Path(args.out) / f"{args.model}_curves_{args.source}_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[curves] wrote {out}")
    print(f"\n{args.model} {args.source}: clips kept (P_full - prior >= {args.min_gap}): {dict(kept)}")
    for clip in sorted(vals):
        print(f"\n  {clip}: mean [95% CI]   difference to frame Shapley [95% CI]")
        for k in ("deletion_auc", "insertion_auc"):
            print(f"   {k}  ({'lower' if k.startswith('del') else 'higher'} is better)")
            for r in rows:
                if r["clip"] == clip and r["readout"] == k:
                    print(f"     {r['method']:17s}{r['mean']:7.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
                          f"   {r['diff_vs_shapley']:+7.3f} [{r['diff_lo']:+.3f}, {r['diff_hi']:+.3f}]")


if __name__ == "__main__":
    main()
