"""
e1_compare_estimators.py -- do two attribution estimators of the same quantity agree within
sampling noise? Written for Play Fair vs Shapley-drop (both estimate Element Shapley Values with
the same characteristic function, using different Monte Carlo samplers).

    python motivation/e1_compare_estimators.py --model r3d
    python motivation/e1_compare_estimators.py --model vjepa2 --out <folder with vjepa2_* files>

Method A (default playfair, seed 0) is compared with method B (default shapley_drop, seed 0) on
exactly the conditions where both exist. The reference for "sampling noise" is a second,
independent run of B (default shapley_drop, seed 1) on the same conditions: B-seed1 vs B-seed0
differ only by sampling. If A estimates the same quantity, A vs B-seed0 should differ no more
than B-seed1 vs B-seed0 (less, if A uses more samples), and without a systematic shift.

Reported, per copy type:
  1. per-slot agreement over all shared conditions: Pearson correlation and relative RMS
     difference ||a - b|| / ||b|| per condition (medians);
  2. per-target E1 metrics from <model>_rows.csv / <model>_slopes.csv (per-copy ratio at each m,
     total ratio, beta): medians for A, B-seed0, B-seed1 on the same targets, the median paired
     difference A - B0 and B1 - B0 with bootstrap 95% CIs, and the median absolute paired
     differences |A - B0| vs |B1 - B0|.
Requires e1_metrics.py attr to have been run (rows/slopes CSVs).
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def parse_ms(spec):
    method, seed = spec.split(":")
    return method, int(seed)


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(n, len(x)), replace=True), axis=1)
    return float(np.median(x)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def fmt(m, lo, hi):
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="r3d")
    ap.add_argument("--out", default="results/e1", help="folder with <model>_attributions.jsonl etc.")
    ap.add_argument("--a", default="playfair:0", help="method:seed under test")
    ap.add_argument("--b", default="shapley_drop:0", help="method:seed compared against")
    ap.add_argument("--b2", default="shapley_drop:1", help="independent repeat of b (noise reference)")
    args = ap.parse_args()
    out = Path(args.out) if Path(args.out).is_absolute() else REPO / args.out
    A, B, B2 = parse_ms(args.a), parse_ms(args.b), parse_ms(args.b2)

    # ---- 1. per-slot agreement
    scores = defaultdict(dict)  # (video, cond_id) -> {(method, seed): np.array}
    with open(out / f"{args.model}_attributions.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (r["method"], int(r["seed"]))
            if key in (A, B, B2):
                scores[(r["video"], r["cond_id"])][key] = np.asarray(r["scores"], float)
    print(f"=== {args.model}: A = {args.a}, B = {args.b}, noise reference B2 = {args.b2}")
    print("\n1. Per-slot agreement (median over shared conditions)")
    for dup in ("exact", "noise"):
        stats = defaultdict(list)
        for (video, cid), d in scores.items():
            if not cid.startswith(dup + "|") or not all(k in d for k in (A, B, B2)):
                continue
            a, b, b2 = d[A], d[B], d[B2]
            stats["corr_A_B"].append(np.corrcoef(a, b)[0, 1])
            stats["corr_B2_B"].append(np.corrcoef(b2, b)[0, 1])
            stats["rel_A_B"].append(np.linalg.norm(a - b) / np.linalg.norm(b))
            stats["rel_B2_B"].append(np.linalg.norm(b2 - b) / np.linalg.norm(b))
        if not stats:
            continue
        n = len(stats["corr_A_B"])
        print(f"  {dup:<6} n = {n} conditions | correlation A-B {np.nanmedian(stats['corr_A_B']):.3f} vs "
              f"B2-B {np.nanmedian(stats['corr_B2_B']):.3f} | relative RMS difference A-B "
              f"{np.median(stats['rel_A_B']):.3f} vs B2-B {np.median(stats['rel_B2_B']):.3f} | "
              f"A closer to B than B2 is in {100 * np.mean(np.array(stats['rel_A_B']) < np.array(stats['rel_B2_B'])):.0f}% of conditions")

    # ---- 2. per-target E1 metrics
    def load(name, value_cols, key_cols):
        table = defaultdict(dict)
        with open(out / f"{args.model}_{name}.csv", newline="") as f:
            for r in csv.DictReader(f):
                ms = (r["method"], int(float(r["seed"])))
                if ms in (A, B, B2):
                    k = tuple(r[c] for c in key_cols)
                    table[k][ms] = {c: float(r[c]) if r[c] not in ("", "nan") else np.nan for c in value_cols}
        return table

    rows = load("rows", ["per_copy_ratio", "conservation"], ["video", "target", "dup", "kind", "m"])
    slopes = load("slopes", ["beta"], ["video", "target", "dup", "kind"])

    def compare(table, col, select):
        a, b, b2 = [], [], []
        for k, d in table.items():
            if select(k) and all(ms in d for ms in (A, B, B2)):
                a.append(d[A][col]); b.append(d[B][col]); b2.append(d[B2][col])
        a, b, b2 = map(np.asarray, (a, b, b2))
        ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(b2)
        return a[ok], b[ok], b2[ok]

    print("\n2. Per-target E1 metrics on identical targets (target condition)")
    print("   quantity                     n   median A  median B0 median B1   paired A-B0 [95% CI]      paired B1-B0 [95% CI]     |A-B0| vs |B1-B0|")
    for dup in ("exact", "noise"):
        ms_vals = sorted({float(k[4]) for k in rows if k[2] == dup and k[3] == "target"})
        items = [(f"per-copy ratio, m = {m:g}", rows, "per_copy_ratio",
                  (lambda mm: lambda k: k[2] == dup and k[3] == "target" and float(k[4]) == mm)(m)) for m in ms_vals]
        items += [(f"total ratio, m = {m:g}", rows, "conservation",
                   (lambda mm: lambda k: k[2] == dup and k[3] == "target" and float(k[4]) == mm)(m)) for m in ms_vals]
        items += [("beta", slopes, "beta", lambda k: k[2] == dup and k[3] == "target")]
        for label, table, col, sel in items:
            a, b, b2 = compare(table, col, sel)
            if len(a) == 0:
                continue
            print(f"   {dup:<6}{label:<23}{len(a):>4}  {np.median(a):>8.3f} {np.median(b):>9.3f} {np.median(b2):>9.3f}   "
                  f"{fmt(*boot_ci(a - b)):<26}{fmt(*boot_ci(b2 - b)):<26}{np.median(np.abs(a - b)):.3f} vs {np.median(np.abs(b2 - b)):.3f}")


if __name__ == "__main__":
    main()
