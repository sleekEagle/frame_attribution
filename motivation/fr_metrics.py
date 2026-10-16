"""
fr_metrics.py -- readouts of the frame-rate test (fr_run.py): when one stretch of the video is
sampled more densely (natural near-duplicates instead of copies), does each of its frames get
less credit and does it lose its rank, while the model relies on it as much as before?

    python motivation/fr_metrics.py --model vjepa2

Clips whose prediction differs from the base clip are left out (the per-clip gate). For the
TARGET segment A (the most important at base), in each densified clip (target clip: A densified
d times; control clip: the least important segment densified instead), relative to the base:
  I ratio        I(A) / I_base(A): how much the model relies on A (remove the whole segment)
  group ratio    group Shapley value of A / its base value (A as a unit; not divided)
  per-frame      mean frame score of A's frames / the same at base   (dilution: falls with d)
  total          summed frame scores of A / the same at base
Medians over videos with a positive base value for that readout; beta = median per-video slope of
log(ratio) on log(d), through 0 (0 = no change, -1 = falls as 1/d), with a bootstrap 95% CI.
Ranking (rankings only, any method):
  top-1 loss     among videos where A has the highest mean frame score at base, the share where
                 it doesn't in the clip; the same by group value (should stay ~0)
  inversion      over the other segments B with group value below A's in the clip (by more than
                 --delta): share where B's mean frame score beats A's. Compared with the base clip.
Writes <out>/<model>_fr_rows.csv and _fr_summary.csv.

--name fr_practical reads <out>/<model>_fr_practical.jsonl (fr_practical.py: broadcast scores of
practical groupings, "grp_*", next to frame-level Shapley) and writes <model>_fr_practical_*.csv.
There, segments covered by one group share a score, so top-1 counts a tie for first as top.
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from e1_common import read_jsonl

PROB = {"shapley_drop", "loo_drop", "occlusion", "playfair"}


def _write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[fr] wrote {path} ({len(rows)} rows)")


def seg_stats(scores, segs, S):
    s = np.asarray(scores, float)
    segs = np.asarray(segs)
    return [s[segs == j].mean() for j in range(S)], [s[segs == j].sum() for j in range(S)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="results/fr")
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--name", default="fr", help="input <model>_<name>.jsonl, output prefix")
    args = ap.parse_args()
    recs = read_jsonl(Path(args.out) / f"{args.model}_{args.name}.jsonl")
    ties = args.name != "fr"

    def is_top(x, j):
        return x[j] >= max(x) - 1e-12 if ties else int(np.argmax(x)) == j

    rows = []
    for r in recs:
        S, A = r["n_seg"], r["target"]
        base = next(c for c in r["clips"] if c["kind"] == "base")
        methods = {(a["method"], a["seed"]) for a in base["attributions"]}
        for c in r["clips"]:
            if c["kind"] == "base" or not c["stats"]["same_pred"]:
                continue
            row0 = {"video": r["video"], "kind": c["kind"], "d": c["d"], "n_unique": c["n_unique"],
                    "I_ratio": c["I_seg"][A] / base["I_seg"][A] if base["I_seg"][A] > 0 else float("nan"),
                    "group_ratio": c["group_values"][A] / base["group_values"][A] if base["group_values"][A] > 0 else float("nan"),
                    "group_top_base": int(np.argmax(base["group_values"]) == A),
                    "group_top_clip": int(np.argmax(c["group_values"]) == A)}
            for (m, sd) in methods:
                ab = next(a for a in base["attributions"] if a["method"] == m and a["seed"] == sd)
                ac = next((a for a in c["attributions"] if a["method"] == m and a["seed"] == sd), None)
                if ac is None:
                    continue
                mb, sb = seg_stats(ab["scores"], base["segs"], S)
                mc, sc = seg_stats(ac["scores"], c["segs"], S)
                gv = c["group_values"]
                others = [B for B in range(S) if B != A and gv[A] - gv[B] > args.delta]
                gvb = base["group_values"]
                others_b = [B for B in range(S) if B != A and gvb[A] - gvb[B] > args.delta]
                rows.append({**row0, "method": m, "seed": sd,
                             "per_frame": mc[A] / mb[A] if mb[A] > 0 else float("nan"),
                             "total": sc[A] / sb[A] if sb[A] > 0 else float("nan"),
                             "frame_top_base": int(is_top(mb, A)),
                             "frame_top_clip": int(is_top(mc, A)),
                             "inv_clip": float(np.mean([mc[B] > mc[A] for B in others])) if others else float("nan"),
                             "inv_base": float(np.mean([mb[B] > mb[A] for B in others_b])) if others_b else float("nan")})
    _write_csv(Path(args.out) / f"{args.model}_{args.name}_rows.csv", rows)

    def beta(rs, col):
        per = defaultdict(list)
        for x in rs:
            if np.isfinite(x[col]) and x[col] > 0:
                per[x["video"]].append((math.log(x["d"]), math.log(x[col])))
        b = [sum(p * q for p, q in v) / sum(p * p for p, _ in v) for v in per.values()]
        if not b:
            return float("nan"), float("nan"), float("nan"), 0
        rng = np.random.default_rng(0)
        bs = [float(np.median(rng.choice(b, len(b)))) for _ in range(args.boot)]
        return float(np.median(b)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(b)

    ds = sorted({x["d"] for x in rows})
    summ = []
    for (m, sd) in sorted({(x["method"], x["seed"]) for x in rows}):
        for kind in ("target", "control"):
            rs = [x for x in rows if x["method"] == m and x["seed"] == sd and x["kind"] == kind]
            for col in ("I_ratio", "group_ratio", "per_frame", "total"):
                line = {"method": m, "seed": sd, "kind": kind, "readout": col}
                for d in ds:
                    v = [x[col] for x in rs if x["d"] == d and np.isfinite(x[col])]
                    line[f"median_d{d}"] = float(np.median(v)) if v else float("nan")
                    line[f"n_d{d}"] = len(v)
                line["beta"], line["beta_lo"], line["beta_hi"], line["n_beta"] = beta(rs, col)
                summ.append(line)
            for name, tb, tc in (("top1_loss_frame", "frame_top_base", "frame_top_clip"),
                                 ("top1_loss_group", "group_top_base", "group_top_clip")):
                line = {"method": m, "seed": sd, "kind": kind, "readout": name}
                for d in ds:
                    v = [1 - x[tc] for x in rs if x["d"] == d and x[tb] == 1]
                    line[f"median_d{d}"] = float(np.mean(v)) if v else float("nan")
                    line[f"n_d{d}"] = len(v)
                summ.append(line)
            for name, col in (("inversion_base", "inv_base"), ("inversion_clip", "inv_clip")):
                line = {"method": m, "seed": sd, "kind": kind, "readout": name}
                for d in ds:
                    v = [x[col] for x in rs if x["d"] == d and np.isfinite(x[col])]
                    line[f"median_d{d}"] = float(np.mean(v)) if v else float("nan")
                    line[f"n_d{d}"] = len(v)
                summ.append(line)
    _write_csv(Path(args.out) / f"{args.model}_{args.name}_summary.csv", summ)

    kept = len({x["video"] for x in rows})
    print(f"\n{args.model}: {len(recs)} videos with a target segment, {kept} with at least one clip passing the gate")
    print(f"  medians at d = {ds} (top-1 loss and inversion: rates)")
    for line in summ:
        vals = "  ".join(f"{line[f'median_d{d}']:6.2f} (n={line[f'n_d{d}']:3d})" for d in ds)
        b = f"   beta {line['beta']:+.2f} [{line['beta_lo']:+.2f}, {line['beta_hi']:+.2f}]" if "beta" in line else ""
        print(f"  {line['method']:13s} {line['kind']:8s} {line['readout']:16s} {vals}{b}")


if __name__ == "__main__":
    main()
