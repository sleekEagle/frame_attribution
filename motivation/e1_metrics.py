"""
e1_metrics.py -- E1 step 4 report (sensitivity gate) and step 6 metrics (dilution).

    python motivation/e1_metrics.py gate --model r3d          # read BEFORE running e1_attribute.py
    python motivation/e1_metrics.py attr --model r3d          # after e1_attribute.py

gate  -> <out>/<model>_gate.csv
         one row per (dup, kind, m, aligned): n, agreement %, mean/median d_prob, d_logit,
         d_margin3, JS, and median I^(m)(t)/I_ref(t). Reallocation models also get the
         "original vs x2 reference" row. Use it to fix each model's --max-m (agreement >= 90-95%).

attr  -> <out>/<model>_rows.csv      one row per (video, target, dup, kind, m, method, seed)
         <out>/<model>_slopes.csv    dilution slope beta per (video, target, dup, kind, method, seed)
         <out>/<model>_summary.csv   aggregated per (method, dup, kind, m[, tier])
         <out>/<model>_beta.csv      median beta + bootstrap 95% CI per (method, dup, kind[, tier])

Readouts. For a content c in a clip, its slot scores give: per-copy MEAN, MAX over copies, SUM.
  per_copy_ratio   mean(t, cond) / mean(t, ref)          (ref = same dup type, method, seed)
  max_ratio        max(t, cond)  / max(t, ref)
  conservation     sum(t, cond)  / sum(t, ref)
  collapse         every copy of t has |score| < 10% of |mean(t, ref)|
  beta             slope of log(per_copy_ratio) on log(m / m_ref), through (0, 0), targets with a
                   positive reference score; -1 = full dilution, 0 = none
  inversion        L_t = contents j the method ranked clearly below t at the reference
                   (max readout, margin delta). delta = 2 x median |seed0 - seed1| difference for
                   stochastic methods with 2 seeds, else 5% of the reference score range.
                   raw rate = share of L_t with score(t) < score(j) in the condition;
                   validated rate = same, only over j with I^(m)(t) > I^(m)(j) (model still ranks
                   t above j, so an inversion is an error of the explanation).
  top1_loss        rows where t was the method's top content at the reference (seed A):
                   is t still top in the condition (seed B)? floor = same check on the reference
                   re-run with seed B.
  I_ratio          I^(m)(t) / I_ref(t): does the MODEL still rely on t?

Caveats
  * With EXACT copies, leave-one-out / occlusion give each copy ~0 once m >= 2, so their
    per_copy_ratio is 0 and beta is undefined (NaN): read collapse_rate for those methods.
  * Reallocation design: the reference already holds every content twice, so exact-copy LOO /
    occlusion are collapsed AT THE REFERENCE (ref score ~0 -> rows dropped from beta). Use the
    near-duplicate (noise/shift) rows for those methods, and report the all-distinct original
    clip separately if needed.
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from e1_common import STOCHASTIC, read_jsonl


def _write_csv(path, rows):
    if not rows:
        print(f"[e1] nothing to write for {path}")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[e1] wrote {path} ({len(rows)} rows)")


def _intkeys(d):
    return {int(k): v for k, v in d.items()} if d else {}


def _nanmed(x):
    x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.median(x)) if x else float("nan")


def _nanmean(x):
    x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(x)) if x else float("nan")


def boot_median_ci(x, n=2000, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)])
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(n, len(x)), replace=True), axis=1)
    return float(np.median(x)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


# --------------------------------------------------------------------------------------------
# gate
# --------------------------------------------------------------------------------------------
def gate(args):
    recs = read_jsonl(Path(args.out) / f"{args.model}_conditions.jsonl")
    groups = defaultdict(list)
    orig = []
    for r in recs:
        I_ref = _intkeys(r["I_ref"])
        if r.get("orig_vs_ref"):
            orig.append(r["orig_vs_ref"])
        for c in r["conditions"]:
            if c["kind"] == "reference":
                key = (c["dup"], "reference_vs_exact", r["m_ref"], True)
                groups[key].append((c["vs_exact_ref"], None))
                continue
            t = c["target"]
            ir = None
            if "I" in c and I_ref.get(t, 0) > 0:
                ir = _intkeys(c["I"]).get(t, float("nan")) / I_ref[t]
            groups[(c["dup"], c["kind"], c["m"], c["aligned"])].append((c["stats"], ir))
    rows = []
    if orig:
        rows.append({"dup": "exact", "kind": "original_vs_x2_reference", "m": "", "aligned": "",
                     "n": len(orig),
                     "agreement_pct": 100 * np.mean([o["orig_pred"] == o["pred"] for o in orig]),
                     "mean_d_prob": _nanmean([o["d_prob"] for o in orig]),
                     "median_js": _nanmed([o["js"] for o in orig])})
    for (dup, kind, m, aligned), items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        st = [s for s, _ in items]
        rows.append({"dup": dup, "kind": kind, "m": m, "aligned": aligned, "n": len(st),
                     "agreement_pct": 100 * np.mean([s["same_pred"] for s in st]),
                     "mean_d_prob": _nanmean([s["d_prob"] for s in st]),
                     "median_d_prob": _nanmed([s["d_prob"] for s in st]),
                     "mean_d_logit": _nanmean([s["d_logit"] for s in st]),
                     "mean_d_margin3": _nanmean([s["d_margin3"] for s in st]),
                     "median_js": _nanmed([s["js"] for s in st]),
                     "median_I_ratio_target": _nanmed([ir for _, ir in items])})
    _write_csv(Path(args.out) / f"{args.model}_gate.csv", rows)
    print(f"\n{'dup':<7}{'kind':<26}{'m':>3}{'algn':>6}{'n':>6}{'agree%':>8}{'dProb':>9}{'JS':>9}{'I_t ratio':>10}")
    for r in rows:
        print(f"{r['dup']:<7}{r['kind']:<26}{str(r['m']):>3}{str(r['aligned'])[:1]:>6}{r['n']:>6}"
              f"{r['agreement_pct']:>8.1f}{r.get('mean_d_prob', float('nan')):>9.3f}"
              f"{r.get('median_js', float('nan')):>9.4f}{r.get('median_I_ratio_target', float('nan')):>10.2f}")


# --------------------------------------------------------------------------------------------
# attribution metrics
# --------------------------------------------------------------------------------------------
def readouts(scores, content):
    by = defaultdict(list)
    for s, c in zip(scores, content):
        by[c].append(s)
    return ({c: float(np.mean(v)) for c, v in by.items()},
            {c: float(np.max(v)) for c, v in by.items()},
            {c: float(np.sum(v)) for c, v in by.items()},
            {c: v for c, v in by.items()})


def _ratio(a, b):
    return a / b if b and abs(b) > 1e-12 else float("nan")


def attr(args):
    out = Path(args.out)
    recs = {r["video"]: r for r in read_jsonl(out / f"{args.model}_conditions.jsonl")}
    attrs = read_jsonl(out / f"{args.model}_attributions.jsonl")
    A = {(a["video"], a["cond_id"], a["method"], a["seed"]): a for a in attrs}
    methods = sorted({a["method"] for a in attrs})
    seeds_of = defaultdict(set)
    for a in attrs:
        seeds_of[a["method"]].add(a["seed"])

    rows = []
    for video, rec in recs.items():
        conds = {c["id"]: c for c in rec["conditions"]}
        for method in methods:
            seeds = sorted(seeds_of[method])
            seed_a, seed_b = seeds[0], (seeds[1] if len(seeds) > 1 else seeds[0])
            for dup in {c["dup"] for c in rec["conditions"]}:
                ref_id = f"{dup}|reference"
                ref_by_seed = {s: A.get((video, ref_id, method, s)) for s in seeds}
                if ref_by_seed[seed_a] is None:
                    continue
                ref_cond = conds[ref_id]
                I_ref = _intkeys(ref_cond.get("I") or rec["I_ref"])
                # margin delta for the inversion test
                ra = readouts(ref_by_seed[seed_a]["scores"], ref_by_seed[seed_a]["content"])[1]
                rb_rec = ref_by_seed.get(seed_b)
                if method in STOCHASTIC and rb_rec is not None and seed_b != seed_a:
                    rb = readouts(rb_rec["scores"], rb_rec["content"])[1]
                    delta = 2 * float(np.median([abs(ra[c] - rb[c]) for c in ra]))
                else:
                    rb = ra
                    delta = 0.05 * (max(ra.values()) - min(ra.values()))
                for seed in seeds:
                    ref_a = ref_by_seed.get(seed)
                    if ref_a is None:
                        continue
                    r_mean, r_max, r_sum, _ = readouts(ref_a["scores"], ref_a["content"])
                    for cid, c in conds.items():
                        if c["dup"] != dup or c["kind"] == "reference":
                            continue
                        a = A.get((video, cid, method, seed))
                        if a is None:
                            continue
                        t = c["target"]
                        c_mean, c_max, c_sum, c_all = readouts(a["scores"], a["content"])
                        I_c = _intkeys(c.get("I"))
                        # inversion (method-relative L_t; validated with the model's I^(m))
                        L = [j for j in r_max if j != t and r_max[t] - r_max[j] > delta]
                        inv_raw = [c_max[t] < c_max[j] for j in L if j in c_max]
                        valid = [j for j in L if j in c_max and I_c and I_c.get(t, -1e9) > I_c.get(j, 1e9)]
                        inv_val = [c_max[t] < c_max[j] for j in valid]
                        # top-1 loss (method's own top at the reference with seed A)
                        top1 = top1_floor = None
                        if seed == seed_b and max(ra, key=ra.get) == t:
                            top1 = max(c_max, key=c_max.get) != t
                            top1_floor = max(rb, key=rb.get) != t
                        ref_pc = r_mean[t]
                        rows.append({
                            "video": video, "class": rec["class"], "target": t, "tier": c["tier"],
                            "dup": dup, "kind": c["kind"], "m": c["m"], "m_ref": rec["m_ref"],
                            "aligned": c["aligned"], "method": method, "seed": seed,
                            "n_copies_t": len(c_all[t]),
                            "ref_per_copy": ref_pc, "cond_per_copy": c_mean[t],
                            "per_copy_ratio": _ratio(c_mean[t], ref_pc),
                            "max_ratio": _ratio(c_max[t], r_max[t]),
                            "conservation": _ratio(c_sum[t], r_sum[t]),
                            "collapse": (all(abs(s) < 0.1 * abs(ref_pc) for s in c_all[t])
                                         if abs(ref_pc) > 1e-12 else None),
                            "share_ref": _ratio(r_sum[t], sum(abs(v) for v in r_sum.values())),
                            "share_cond": _ratio(c_sum[t], sum(abs(v) for v in c_sum.values())),
                            "n_L": len(L), "inv_raw": _nanmean(inv_raw) if inv_raw else None,
                            "n_valid": len(valid), "inv_validated": _nanmean(inv_val) if inv_val else None,
                            "top1_loss": top1, "top1_floor": top1_floor,
                            "I_ratio": _ratio(I_c.get(t, float("nan")), I_ref.get(t)) if I_c else None,
                            "n_evals": a["n_evals"], "secs": a["secs"]})
    _write_csv(out / f"{args.model}_rows.csv", rows)

    # ---- slopes -----------------------------------------------------------------------------
    by_curve = defaultdict(list)
    for r in rows:
        by_curve[(r["video"], r["target"], r["tier"], r["dup"], r["kind"], r["method"], r["seed"])].append(r)
    slopes = []
    for (video, t, tier, dup, kind, method, seed), rs in by_curve.items():
        if not rs or not (rs[0]["ref_per_copy"] > 0):
            continue
        xs, ys = [0.0], [0.0]
        for r in rs:
            pr = r["per_copy_ratio"]
            if pr is not None and np.isfinite(pr) and pr > 0:
                xs.append(math.log(r["m"] / r["m_ref"]))
                ys.append(math.log(pr))
        if len(xs) < 2:
            beta = float("nan")
        else:
            x, y = np.array(xs), np.array(ys)
            beta = float((x @ y) / (x @ x))  # least squares through the origin
        slopes.append({"video": video, "target": t, "tier": tier, "dup": dup, "kind": kind,
                       "method": method, "seed": seed, "beta": beta, "n_points": len(xs) - 1,
                       "n_nonpositive": len(rs) - (len(xs) - 1)})
    _write_csv(out / f"{args.model}_slopes.csv", slopes)

    # ---- aggregation ------------------------------------------------------------------------
    gkeys = ["method", "dup", "kind", "m"] + (["tier"] if args.by_tier else [])
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in gkeys)].append(r)
    summary = []
    for key, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        t1 = [r for r in rs if r["top1_loss"] is not None]
        summary.append({**dict(zip(gkeys, key)), "n": len(rs),
                        "median_per_copy_ratio": _nanmed([r["per_copy_ratio"] for r in rs]),
                        "median_max_ratio": _nanmed([r["max_ratio"] for r in rs]),
                        "median_conservation": _nanmed([r["conservation"] for r in rs]),
                        "collapse_rate": _nanmean([float(r["collapse"]) for r in rs if r["collapse"] is not None]),
                        "inv_raw": _nanmean([r["inv_raw"] for r in rs]),
                        "inv_validated": _nanmean([r["inv_validated"] for r in rs]),
                        "n_top1": len(t1),
                        "top1_loss": _nanmean([float(r["top1_loss"]) for r in t1]),
                        "top1_floor": _nanmean([float(r["top1_floor"]) for r in t1]),
                        "median_I_ratio": _nanmed([r["I_ratio"] for r in rs])})
    _write_csv(out / f"{args.model}_summary.csv", summary)

    bkeys = ["method", "dup", "kind"] + (["tier"] if args.by_tier else [])
    bgroups = defaultdict(list)
    for s in slopes:
        bgroups[tuple(s[k] for k in bkeys)].append(s["beta"])
    beta_rows = []
    for key, bs in sorted(bgroups.items(), key=lambda kv: str(kv[0])):
        med, lo, hi = boot_median_ci(bs)
        beta_rows.append({**dict(zip(bkeys, key)), "n": len(bs),
                          "n_finite": int(np.isfinite(bs).sum()),
                          "median_beta": med, "ci_lo": lo, "ci_hi": hi})
    _write_csv(out / f"{args.model}_beta.csv", beta_rows)
    print(f"\n{'method':<16}{'dup':<7}{'kind':<20}{'n':>5}{'beta':>8}{'95% CI':>18}")
    for b in beta_rows:
        print(f"{b['method']:<16}{b['dup']:<7}{b['kind']:<20}{b['n']:>5}{b['median_beta']:>8.2f}"
              f"   [{b['ci_lo']:.2f}, {b['ci_hi']:.2f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["gate", "attr"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="results/e1")
    ap.add_argument("--by-tier", action="store_true", help="split summaries by target tier")
    args = ap.parse_args()
    gate(args) if args.what == "gate" else attr(args)


if __name__ == "__main__":
    main()
