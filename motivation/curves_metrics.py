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
  deletion_auc_groups, insertion_auc_groups   group-wise curves only: the areas with one step per
                  group (k + 1 points) instead of one per frame
Means over clips with 95% bootstrap CIs (resampling videos), and the paired difference to
frame-level Shapley on the same clips (negative = better for deletion, positive = better for
insertion). For the group-wise curves (sources e23_gw, fr_gw), the difference is to the ranking of
the same groups by mean frame-level Shapley value (sources e23_gwm, fr_gwm; e23_gw, fr_gw from the
first group-wise run).

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
    ap.add_argument("--source", required=True, help="e23, fr, or a group-wise run: e23_gwm, fr_gwm (fr_gw, e23_gw)")
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
            if "del_groups" in c:  # group-wise curves: the same areas with one step per group
                ro[meth]["deletion_auc_groups"] = float(((np.asarray(c["del_groups"]) - r["prior"]) / gap).mean())
                ro[meth]["insertion_auc_groups"] = float(((np.asarray(c["ins_groups"]) - r["prior"]) / gap).mean())
        shap = next((m for m in SHAPLEY if m in ro), None)
        for meth, x in ro.items():
            # group-wise curves (curves.py --groupwise) are compared with the mean-ranking of the
            # same groups, frame-wise ones with frame-level Shapley
            base = next((b for b in ro if b.startswith("gw_") and b.endswith("_mean")
                         and meth.startswith(b[:-len("mean")])), None) if meth.startswith("gw_") else shap
            for k, v in x.items():
                vals[r["clip"]][k][meth][r["video"]].append(v)
                if base in ro:
                    diffs[r["clip"]][k][meth][r["video"]].append(v - ro[base][k])

    rows = []
    for clip in sorted(vals):
        for k in ("deletion_auc", "insertion_auc", "del_at_25", "ins_at_25",
                  "deletion_auc_groups", "insertion_auc_groups"):
            for meth in sorted(vals[clip][k]):
                mu, lo, hi = boot(vals[clip][k][meth], args.boot)
                dm, dlo, dhi = boot(diffs[clip][k][meth], args.boot) if diffs[clip][k][meth] else (np.nan,) * 3
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
        print(f"\n  {clip}: mean [95% CI]   difference to {'mean ranking' if '_gw' in args.source else 'frame Shapley'} [95% CI]")
        for k in ("deletion_auc", "insertion_auc", "deletion_auc_groups", "insertion_auc_groups"):
            if not any(r["clip"] == clip and r["readout"] == k for r in rows):
                continue
            print(f"   {k}  ({'lower' if k.startswith('del') else 'higher'} is better)")
            for r in rows:
                if r["clip"] == clip and r["readout"] == k:
                    print(f"     {r['method']:17s}{r['mean']:7.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
                          f"   {r['diff_vs_shapley']:+7.3f} [{r['diff_lo']:+.3f}, {r['diff_hi']:+.3f}]")


if __name__ == "__main__":
    main()
