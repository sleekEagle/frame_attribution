"""
e23_metrics.py -- E2 and E3 readouts from e23_run.py's <out>/<model>_e23.jsonl.

    python motivation/e23_metrics.py --model mc3_18
    python motivation/e23_metrics.py --model toy --out <dir> --tau 0.0     # smoke test

Score units. Shapley, leave-one-out, occlusion and Play Fair are differences of P(c*), the same
units as group importance I(G). IG (logit units) and Grad-CAM (arbitrary) are rescaled per
video so their scores add up to P(c*|clip) - prior, which makes their ratios comparable.
Each method is compared with I(G) under its own removal (drop for *_drop and Play Fair, late
freeze for *_freeze, the model's E1 removal for the others).

E2 main readouts: how redundant are natural clips, and how do methods respond?
  -> <model>_e2_visibility.csv, _e2_redundancy.csv, _e2_method_vs_R.csv
  visibility (per video, model only; videos with P(c*|clip) - prior > 0.05)
    sum_loo_share   sum of the single-frame importances I({i}) / (P(c*|clip) - prior): how much
                    of the clip's importance single-frame removal can see. ~0 = the frames can
                    all stand in for each other.
    max_loo_share   the most important single frame / (P(c*|clip) - prior)
    left_after_k    share of P(c*|clip) - prior left after removing the top-k frames (k = 1, 2,
                    4) in each deletion order (each method, cluster orders, random)
  redundancy of a group (model only), for groups with I(G) > --tau and 2 <= |G| <= n/2:
    R(G) = 1 - sum_{i in G} I({i}) / I(G). R ~ 1: no single frame of G is needed (they stand
    in for each other); R ~ 0: each frame adds its own part. By size and grouping (DINOv2
    nodes vs fixed windows), with the mean DINOv2 similarity inside the group.
  method vs redundancy: for the same groups, per method (seed 0) and R bin, the median of
    r_sum = sum_{i in G} s_i / I(G) (does the method keep the group's importance?) and of
    best x |G| / I(G) (the best frame's credit relative to an even share of I(G)).

E2 first readouts (kept for the record; CONFOUNDED, don't interpret the slopes):
  -> <model>_e2_rows.csv, _e2_bins.csv, _e2_slopes.csv
  The slope of r_max on |G| also depends on how I(G) grows with |G|: a fixed window holds more
  distinct content as it grows, so its I(G) grows and r_max falls even without any sharing of
  credit, while a run of similar frames gains little importance with size. So the "dino -
  window" slope difference measures how importance scales, not dilution.
  For every group G with I(G) > --tau and 1 <= |G| <= n/2, in two groupings:
    dino    the nodes of the DINOv2 tree (runs of similar frames)
    window  fixed windows of 2, 4, 8 frames (control: same sizes, similarity ignored)
  readouts
    r_max = max_{i in G} s_i / I(G)   can one frame of G show G's importance?
    r_sum = sum_{i in G} s_i / I(G)   is G's importance there when the scores are added up?
    r_loo = max_{i in G} I({i}) / I(G)   (model only, no method) is any single frame of G
            needed on its own? Small = the frames of G can stand in for each other.
  slope of log(readout) on log|G| (pooled; 95% CI from resampling videos). If the credit of
  G is shared equally by its frames, r_max falls as 1/|G| (slope -1); if one frame carries
  it, the slope is 0. dilution shows as a slope near -1 for dino groups and a flatter slope
  for windows; "dino - window" is the paired difference.

E3: can the method find the important part?  -> _e3_hits.csv, _e3_inversions.csv, _e3_deletion.csv
  At tree levels --levels (number of clusters; default n/4 and n/2):
    hit rate    is the method's top frame inside the most important cluster G*?
                (chance = |G*| / n), by |G*|
    inversions  pairs of clusters (A, B) at a level with I(A) > tau and I(A) - I(B) > --delta:
                does the method rank B above A? "max" compares the best frame of each cluster
                (frame-level detection), "sum" the summed scores. By size ratio |A| / |B|.
                Dilution predicts more max-inversions when A is the LARGER cluster, and few
                sum-inversions. CONFOUNDED by size: the best of more frames is larger by chance
                alone, and I(G) grows with size, so pairs with a larger A are biased towards
                fewer max-inversions. Read hits and inversions with this in mind.
  deletion      normalised area under P(c*) after removing the top-k frames, k = 1..n:
                mean_k (v_k - prior) / (v_0 - prior). Lower = the order finds the important
                frames sooner. By order (each method, cluster order at a level, random).

Dilution tests (needs <model>_e23_group.jsonl from e23_group.py)
  -> <model>_dil_pairs.csv, _dil_contrast.csv, _dil_top.csv
  Dilution: the evidence is spread over frames that stand in for each other, so a frame-level
  method divides the credit, each frame looks unimportant, and the content can rank below
  content shown in fewer frames. The yardstick is the group Shapley value phi(G) of each
  segment of a tree cut with k segments: how much the segment matters as a unit, with no
  division inside it. The frame scores are read as a user would: credit per frame, best frame
  per segment, and the top-ranked frames. Rankings only, so no unit conversion is needed.
    pairs     segments A, B at the same level with phi(A) > --tau and phi(A) - phi(B) > --delta.
              inv_mean: B's frames have a higher AVERAGE score than A's (each frame of the more
              important segment looks less important; the main readout for dilution).
              inv_max: the best frame of B outscores the best frame of A. inv_sum: B's summed
              scores beat A's. By size ratio |A|/|B|.
              chance: the same comparisons with the frame scores rotated by every offset (so
              they keep their smoothness over time but are no longer aligned with the segments).
              For max, chance is lower when A is larger (the best of more frames wins more often
              by chance alone), so max must be read against its chance level; for mean, chance
              is about 0.5 at every size ratio.
    contrast  gain = chance - observed inversion rate (how much better than chance the method
              ranks the more important segment). gain difference = gain when A is at least twice
              as large as B, minus gain when A and B have the same size (within a factor 2); 95%
              CI from resampling videos. Dilution predicts a negative gain difference.
    top       the most important segment G* (largest phi, > --tau): is the method's top frame
              in G* (chance |G*|/n)? is any of its top 4 frames in G* (chance
              1 - C(n-|G*|, 4)/C(n, 4))? median rank of G*'s best frame. By |G*|.

Swap test  -> _swap.csv
  For neighbouring frames (i, i+1): mean |change in P(c*)| when one replaces the other, vs
  their DINOv2 similarity (Spearman correlation), and within- vs across-cluster pairs at each
  level. Small within-cluster changes = DINOv2's clusters are interchangeable for the model.

numpy only, elementwise: no matrix products or least-squares calls (tc_env's MKL crashes on
those; see results/design_choice_report.md).
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from e1_common import read_jsonl
from e23_run import METHOD_REMOVAL, level_groups

PROB_UNITS = {"shapley_drop", "shapley_freeze", "loo_drop", "loo_freeze", "occlusion", "playfair"}


def _write_csv(path, rows):
    if not rows:
        print(f"[e23] nothing to write for {path}")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[e23] wrote {path} ({len(rows)} rows)")


def _slope(x, y):
    """OLS slope of y on x, elementwise (no BLAS)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    dx = x - x.mean()
    den = float((dx * dx).sum())
    return float((dx * (y - y.mean())).sum() / den) if den > 0 else float("nan")


def _rank(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="stable")
    r = np.empty(len(a))
    r[order] = np.arange(len(a))
    # average ties
    for v in np.unique(a):
        m = a == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def _spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def _boot(by_video, stat, B, seed=0):
    """95% CI of stat(list of per-video row lists), resampling videos."""
    vids = list(by_video)
    if len(vids) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(vids), len(vids))
        v = stat([by_video[vids[i]] for i in pick])
        if not math.isnan(v):
            vals.append(v)
    if len(vals) < 10:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _size_bin(s):
    return "1" if s == 1 else "2" if s == 2 else "3-4" if s <= 4 else "5-8" if s <= 8 else ">8"


def method_scores(rec, a):
    """(scores in P(c*) units, removal they are compared under)."""
    s = np.asarray(a["scores"], float)
    if a["method"] not in PROB_UNITS:
        tot = s.sum()
        total_p = rec["v_all"][rec["removal"]] - rec["prior_pred"]
        s = s * (total_p / tot) if abs(tot) > 1e-12 else s * float("nan")
    return s, METHOD_REMOVAL.get(a["method"], rec["removal"])


def groups_with_I(rec, removal):
    """[(grouping, frames, I, height)] for every tree node except the whole clip, and windows."""
    n = rec["n"]
    imp = rec["importance"][removal]
    out = [("dino", nd["frames"], I, nd["height"])
           for nd, I in zip(rec["nodes"], imp["nodes"]) if len(nd["frames"]) < n]
    out += [("window", w, I, float("nan")) for w, I in zip(rec["windows"], imp["windows"])]
    return out


# --------------------------------------------------------------------------------------------
R_BINS = [(-math.inf, 0.25, "R < 0.25"), (0.25, 0.5, "0.25-0.5"), (0.5, 0.75, "0.5-0.75"),
          (0.75, math.inf, "R >= 0.75")]


def _bin_size(s):
    return "2" if s == 2 else "3-4" if s <= 4 else "5-8" if s <= 8 else ">8"


def e2_redundancy(recs, args):
    """The main E2 readouts: visibility per video, redundancy R(G), and method vs R."""
    vis = []
    for rec in recs:
        n, rem = rec["n"], rec["removal"]
        total = rec["v_all"][rem] - rec["prior_pred"]
        if total <= 0.05:
            continue
        leaf = np.asarray(rec["importance"][rem]["nodes"][:n], float)
        row = {"video": rec["video"], "total": total, "sum_loo_share": float(leaf.sum() / total),
               "max_loo_share": float(leaf.max() / total)}
        for d in rec["deletion"]:
            for k in (1, 2, 4):
                row[f"left_after_{k}|{d['order']}"] = float((d["values"][k] - rec["prior_pred"]) / total)
        vis.append(row)
    _write_csv(Path(args.out) / f"{args.model}_e2_visibility.csv", vis)

    groups, mrows = [], []
    for rec in recs:
        n = rec["n"]
        sim = np.asarray(rec["sim"], float)
        # model-only redundancy, under the model's E1 removal
        rem = rec["removal"]
        leaf = rec["importance"][rem]["nodes"][:n]
        for grouping, g, I, _ in groups_with_I(rec, rem):
            if not (2 <= len(g) <= n // 2):
                continue
            sub = sim[np.ix_(g, g)]
            groups.append({"video": rec["video"], "grouping": grouping, "size": len(g), "I": I,
                           "mean_sim": float((sub.sum() - len(g)) / (len(g) * (len(g) - 1))),
                           "R": float(1 - sum(leaf[i] for i in g) / I) if I > args.tau else float("nan")})
        # method vs redundancy, each method under its own removal
        for a in rec["attributions"]:
            if a["seed"] != 0:
                continue
            s, removal = method_scores(rec, a)
            if removal not in rec["importance"]:
                continue
            leaf_m = rec["importance"][removal]["nodes"][:n]
            for grouping, g, I, _ in groups_with_I(rec, removal):
                if not (2 <= len(g) <= n // 2) or I <= args.tau:
                    continue
                mrows.append({"method": a["method"], "R": 1 - sum(leaf_m[i] for i in g) / I,
                              "r_sum": float(s[g].sum() / I), "best_share": float(s[g].max() / I * len(g))})

    red = []
    for b in ("2", "3-4", "5-8"):
        for gp in ("dino", "window"):
            rs = [x for x in groups if _bin_size(x["size"]) == b and x["grouping"] == gp]
            ks = [x for x in rs if x["I"] > args.tau]
            if not rs:
                continue
            red.append({"size": b, "grouping": gp, "n_groups": len(rs), "n_important": len(ks),
                        "median_mean_sim": float(np.median([x["mean_sim"] for x in rs])),
                        "median_R": float(np.median([x["R"] for x in ks])) if ks else float("nan"),
                        "share_R_above_half": float(np.mean([x["R"] > 0.5 for x in ks])) if ks else float("nan")})
    _write_csv(Path(args.out) / f"{args.model}_e2_redundancy.csv", red)

    mvr = []
    for meth in sorted({x["method"] for x in mrows}):
        for lo, hi, lab in R_BINS:
            xs = [x for x in mrows if x["method"] == meth and lo <= x["R"] < hi]
            if xs:
                mvr.append({"method": meth, "R_bin": lab, "n": len(xs),
                            "median_r_sum": float(np.median([x["r_sum"] for x in xs])),
                            "median_best_share": float(np.median([x["best_share"] for x in xs]))})
    _write_csv(Path(args.out) / f"{args.model}_e2_method_vs_R.csv", mvr)
    return vis, red, mvr


def e2(recs, args):
    rows = []
    for rec in recs:
        n = rec["n"]
        sim = np.asarray(rec["sim"], float)
        for a in rec["attributions"]:
            s, removal = method_scores(rec, a)
            if removal not in rec["importance"]:
                continue
            leaf_I = rec["importance"][removal]["nodes"][:n]
            for grouping, g, I, h in groups_with_I(rec, removal):
                if len(g) > n // 2:
                    continue
                sg = s[g]
                if len(g) > 1:
                    sub = sim[np.ix_(g, g)]
                    mean_sim = float((sub.sum() - len(g)) / (len(g) * (len(g) - 1)))
                else:
                    mean_sim = float("nan")
                rows.append({
                    "video": rec["video"], "method": a["method"], "seed": a["seed"],
                    "grouping": grouping, "size": len(g), "height": h, "mean_sim": mean_sim,
                    "I": I, "max_s": float(sg.max()), "sum_s": float(sg.sum()),
                    "max_loo": float(max(leaf_I[i] for i in g)),
                    "r_max": float(sg.max() / I) if I > 0 else float("nan"),
                    "r_sum": float(sg.sum() / I) if I > 0 else float("nan"),
                    "r_loo": float(max(leaf_I[i] for i in g) / I) if I > 0 else float("nan")})
    _write_csv(Path(args.out) / f"{args.model}_e2_rows.csv", rows)

    keep = [r for r in rows if r["I"] > args.tau]
    # bins
    bins = []
    cell = defaultdict(list)
    for r in keep:
        cell[(r["method"], r["seed"], r["grouping"], _size_bin(r["size"]))].append(r)
    for (m, sd, gp, b), rs in sorted(cell.items()):
        bins.append({"method": m, "seed": sd, "grouping": gp, "size_bin": b, "n": len(rs),
                     "median_I": float(np.median([r["I"] for r in rs])),
                     "median_mean_sim": float(np.nanmedian([r["mean_sim"] for r in rs])) if b != "1" else float("nan"),
                     "median_r_max": float(np.median([r["r_max"] for r in rs])),
                     "median_r_sum": float(np.median([r["r_sum"] for r in rs])),
                     "median_r_loo": float(np.median([r["r_loo"] for r in rs]))})
    _write_csv(Path(args.out) / f"{args.model}_e2_bins.csv", bins)

    # slopes, with "model" (r_loo, method-free) taken from the first method's rows
    def slope_of(rlist, readout):
        x = [math.log(r["size"]) for r in rlist if r[readout] > 0]
        y = [math.log(r[readout]) for r in rlist if r[readout] > 0]
        return _slope(x, y)

    slopes = []
    by_ms = defaultdict(list)
    for r in keep:
        by_ms[(r["method"], r["seed"])].append(r)
    first = sorted(by_ms)[0] if by_ms else None
    jobs = [(m, sd, ro) for (m, sd) in sorted(by_ms) for ro in ("r_max", "r_sum")]
    if first:
        jobs.append(("model", None, "r_loo"))
    for m, sd, ro in jobs:
        rs = by_ms[first] if m == "model" else by_ms[(m, sd)]
        per = {}
        for gp in ("dino", "window"):
            sub = [r for r in rs if r["grouping"] == gp]
            byv = defaultdict(list)
            for r in sub:
                byv[r["video"]].append(r)
            per[gp] = (sub, byv)
        for gp in ("dino", "window"):
            sub, byv = per[gp]
            lo, hi = _boot(byv, lambda vs: slope_of([r for v in vs for r in v], ro), args.boot)
            slopes.append({"method": m, "seed": sd, "readout": ro, "grouping": gp,
                           "n_groups": len(sub), "n_videos": len(byv),
                           "excluded_nonpositive": sum(1 for r in sub if not r[ro] > 0),
                           "slope": slope_of(sub, ro), "ci_lo": lo, "ci_hi": hi})
        both = defaultdict(list)
        for r in rs:
            both[r["video"]].append(r)
        diff = lambda vs: (slope_of([r for v in vs for r in v if r["grouping"] == "dino"], ro)  # noqa: E731
                           - slope_of([r for v in vs for r in v if r["grouping"] == "window"], ro))
        lo, hi = _boot(both, diff, args.boot)
        slopes.append({"method": m, "seed": sd, "readout": ro, "grouping": "dino - window",
                       "n_groups": len(rs), "n_videos": len(both), "excluded_nonpositive": "",
                       "slope": diff(list(both.values())), "ci_lo": lo, "ci_hi": hi})
    _write_csv(Path(args.out) / f"{args.model}_e2_slopes.csv", slopes)
    return slopes


# --------------------------------------------------------------------------------------------
def e3(recs, args):
    hits, invs, dels = [], [], []
    for rec in recs:
        n = rec["n"]
        levels = args.levels or sorted({max(2, n // 4), max(2, n // 2)})
        node_I = {r: {tuple(nd["frames"]): I for nd, I in zip(rec["nodes"], rec["importance"][r]["nodes"])}
                  for r in rec["importance"]}
        for a in rec["attributions"]:
            s, removal = method_scores(rec, a)
            if removal not in node_I:
                continue
            top = int(np.argmax(s))
            for k in levels:
                groups = level_groups(rec["nodes"], n, k)
                Is = [node_I[removal][tuple(g)] for g in groups]
                j = int(np.argmax(Is))
                if Is[j] > args.tau:
                    hits.append({"video": rec["video"], "method": a["method"], "seed": a["seed"],
                                 "level": k, "size_top": len(groups[j]), "I_top": Is[j],
                                 "hit": int(top in groups[j]), "chance": len(groups[j]) / n})
                for ia, A in enumerate(groups):
                    for ib, B in enumerate(groups):
                        if ia == ib or not (Is[ia] > args.tau and Is[ia] - Is[ib] > args.delta):
                            continue
                        invs.append({"video": rec["video"], "method": a["method"], "seed": a["seed"],
                                     "level": k, "size_A": len(A), "size_B": len(B),
                                     "log2_ratio": math.log2(len(A) / len(B)),
                                     "inv_max": int(s[A].max() < s[B].max()),
                                     "inv_sum": int(s[A].sum() < s[B].sum())})
        v0, prior = rec["deletion"][0]["values"][0], rec["prior_pred"]
        if v0 - prior > args.tau:
            for d in rec["deletion"]:
                v = np.asarray(d["values"], float)
                dels.append({"video": rec["video"], "order": d["order"],
                             "auc": float(((v[1:] - prior) / (v0 - prior)).mean()),
                             "after_1": float((v[1] - prior) / (v0 - prior)),
                             "after_2": float((v[2] - prior) / (v0 - prior)),
                             "after_4": float((v[4] - prior) / (v0 - prior))})

    def ratio_bin(x):
        return "A smaller" if x <= -1 else "same size" if x < 1 else "A 2-3x" if x < 2 else "A >=4x"

    summ = []
    cell = defaultdict(list)
    for h in hits:
        cell[(h["method"], h["seed"], h["level"], _size_bin(h["size_top"]))].append(h)
    for (m, sd, k, b), hs in sorted(cell.items()):
        summ.append({"method": m, "seed": sd, "level": k, "size_top": b, "n": len(hs),
                     "hit_rate": float(np.mean([h["hit"] for h in hs])),
                     "chance": float(np.mean([h["chance"] for h in hs]))})
    _write_csv(Path(args.out) / f"{args.model}_e3_hits.csv", summ)

    summ_i = []
    cell = defaultdict(list)
    for r in invs:
        cell[(r["method"], r["seed"], r["level"], ratio_bin(r["log2_ratio"]))].append(r)
    for (m, sd, k, b), rs in sorted(cell.items()):
        summ_i.append({"method": m, "seed": sd, "level": k, "size_ratio": b, "n_pairs": len(rs),
                       "n_videos": len({r["video"] for r in rs}),
                       "inv_rate_max": float(np.mean([r["inv_max"] for r in rs])),
                       "inv_rate_sum": float(np.mean([r["inv_sum"] for r in rs]))})
    _write_csv(Path(args.out) / f"{args.model}_e3_inversions.csv", summ_i)

    summ_d = []
    cell = defaultdict(list)
    for r in dels:
        cell[r["order"]].append(r)
    for o, rs in sorted(cell.items()):
        byv = {r["video"]: [r] for r in rs}
        lo, hi = _boot(byv, lambda vs: float(np.mean([v[0]["auc"] for v in vs])), args.boot)
        summ_d.append({"order": o, "n_videos": len(rs), "auc": float(np.mean([r["auc"] for r in rs])),
                       "ci_lo": lo, "ci_hi": hi,
                       "after_1": float(np.mean([r["after_1"] for r in rs])),
                       "after_2": float(np.mean([r["after_2"] for r in rs])),
                       "after_4": float(np.mean([r["after_4"] for r in rs]))})
    _write_csv(Path(args.out) / f"{args.model}_e3_deletion.csv", summ_d)
    return summ, summ_i, summ_d


# --------------------------------------------------------------------------------------------
def _ratio_bin(x):
    return "A smaller" if x <= -1 else "same size" if x < 1 else "A 2-3x" if x < 2 else "A >=4x"


def dilution(recs, grp, args):
    """Dilution tests against group Shapley values (see the module docstring)."""
    by_video = {g["video"]: g for g in grp}
    pairs, tops = [], []
    for rec in recs:
        g = by_video.get(rec["video"])
        if g is None:
            continue
        n = rec["n"]
        for k, lv in g["levels"].items():
            groups, phi = lv["groups"], np.asarray(lv["phi"], float)
            j_top = int(np.argmax(phi))
            for a in rec["attributions"]:
                s = np.asarray(a["scores"], float)
                rots = [np.roll(s, o) for o in range(1, n)]  # chance: scores no longer aligned with segments
                key = dict(video=rec["video"], method=a["method"], seed=a["seed"], level=int(k))
                for ia, A in enumerate(groups):
                    if phi[ia] <= args.tau:
                        continue
                    for ib, B in enumerate(groups):
                        if ia == ib or phi[ia] - phi[ib] <= args.delta:
                            continue
                        pairs.append({**key, "log2_ratio": math.log2(len(A) / len(B)),
                                      "inv_max": int(s[B].max() > s[A].max()),
                                      "inv_mean": int(s[B].mean() > s[A].mean()),
                                      "inv_sum": int(s[B].sum() > s[A].sum()),
                                      "chance_max": float(np.mean([x[B].max() > x[A].max() for x in rots])),
                                      "chance_mean": float(np.mean([x[B].mean() > x[A].mean() for x in rots]))})
                if phi[j_top] > args.tau:
                    G = groups[j_top]
                    order = np.argsort(-s, kind="stable")
                    rank = int(min(np.flatnonzero(np.isin(order, G)))) + 1
                    gs = len(G)
                    tops.append({**key, "size_top": gs, "hit1": int(order[0] in G), "chance1": gs / n,
                                 "top4": int(rank <= 4),
                                 "chance4": 1 - math.comb(n - gs, 4) / math.comb(n, 4) if n - gs >= 4 else 1.0,
                                 "rank": rank})

    out_p = []
    cell = defaultdict(list)
    for p in pairs:
        cell[(p["method"], p["seed"], p["level"], _ratio_bin(p["log2_ratio"]))].append(p)
    for (m, sd, k, b), ps in sorted(cell.items()):
        out_p.append({"method": m, "seed": sd, "level": k, "size_ratio": b, "n_pairs": len(ps),
                      "n_videos": len({p["video"] for p in ps}),
                      "inv_rate_max": float(np.mean([p["inv_max"] for p in ps])),
                      "chance_max": float(np.mean([p["chance_max"] for p in ps])),
                      "inv_rate_mean": float(np.mean([p["inv_mean"] for p in ps])),
                      "chance_mean": float(np.mean([p["chance_mean"] for p in ps])),
                      "inv_rate_sum": float(np.mean([p["inv_sum"] for p in ps]))})
    _write_csv(Path(args.out) / f"{args.model}_dil_pairs.csv", out_p)

    out_c = []
    by_msk = defaultdict(lambda: defaultdict(list))
    for p in pairs:
        by_msk[(p["method"], p["seed"], p["level"])][p["video"]].append(p)

    def gain(vs, readout, which):
        """mean over pairs of (chance - observed) inversion rate: how much better than chance the
        method ranks the more important segment. which: "larger" (A >= 2x B) or "same" (within 2x)."""
        sel = (lambda p: p["log2_ratio"] >= 1) if which == "larger" else (lambda p: abs(p["log2_ratio"]) < 1)
        x = [p[f"chance_{readout}"] - p[f"inv_{readout}"] for v in vs for p in v if sel(p)]
        return float(np.mean(x)) if x else float("nan")

    def rate(vs, col, which):
        sel = (lambda p: p["log2_ratio"] >= 1) if which == "larger" else (lambda p: abs(p["log2_ratio"]) < 1)
        x = [p[col] for v in vs for p in v if sel(p)]
        return float(np.mean(x)) if x else float("nan")
    for (m, sd, k), byv in sorted(by_msk.items()):
        vs = list(byv.values())
        for readout in ("mean", "max"):
            diff = lambda w: gain(w, readout, "larger") - gain(w, readout, "same")  # noqa: E731
            lo, hi = _boot(byv, diff, args.boot)
            out_c.append({"method": m, "seed": sd, "level": k, "readout": readout,
                          "inv_A_larger": rate(vs, f"inv_{readout}", "larger"),
                          "chance_A_larger": rate(vs, f"chance_{readout}", "larger"),
                          "inv_same_size": rate(vs, f"inv_{readout}", "same"),
                          "chance_same_size": rate(vs, f"chance_{readout}", "same"),
                          "gain_A_larger": gain(vs, readout, "larger"), "gain_same_size": gain(vs, readout, "same"),
                          "gain_difference": diff(vs), "ci_lo": lo, "ci_hi": hi})
    _write_csv(Path(args.out) / f"{args.model}_dil_contrast.csv", out_c)

    out_t = []
    cell = defaultdict(list)
    for t in tops:
        cell[(t["method"], t["seed"], t["level"], _size_bin(t["size_top"]))].append(t)
    for (m, sd, k, b), ts in sorted(cell.items()):
        out_t.append({"method": m, "seed": sd, "level": k, "size_top": b, "n": len(ts),
                      "hit1": float(np.mean([t["hit1"] for t in ts])), "chance1": float(np.mean([t["chance1"] for t in ts])),
                      "top4": float(np.mean([t["top4"] for t in ts])), "chance4": float(np.mean([t["chance4"] for t in ts])),
                      "median_rank": float(np.median([t["rank"] for t in ts]))})
    _write_csv(Path(args.out) / f"{args.model}_dil_top.csv", out_t)
    return out_p, out_c, out_t


def swap(recs, args):
    pairs = []
    for rec in recs:
        if not rec.get("swaps"):
            continue
        n = rec["n"]
        levels = args.levels or sorted({max(2, n // 4), max(2, n // 2)})
        cut = {k: level_groups(rec["nodes"], n, k) for k in levels}
        by_i = defaultdict(list)
        for sw in rec["swaps"]:
            by_i[sw["i"]].append(sw)
        for i, sws in by_i.items():
            row = {"video": rec["video"], "i": i, "sim": rec["sim"][i][i + 1],
                   "abs_d_prob": float(np.mean([abs(x["d_prob"]) for x in sws])),
                   "js": float(np.mean([x["js"] for x in sws]))}
            for k, gs in cut.items():
                row[f"same_cluster_k{k}"] = int(any(i in g and i + 1 in g for g in gs))
            pairs.append(row)
    if not pairs:
        return []
    out = [{"what": "spearman(sim, |d_prob|)", "level": "", "n_pairs": len(pairs),
            "value": _spearman([p["sim"] for p in pairs], [p["abs_d_prob"] for p in pairs])},
           {"what": "spearman(sim, js)", "level": "", "n_pairs": len(pairs),
            "value": _spearman([p["sim"] for p in pairs], [p["js"] for p in pairs])}]
    for key in [k for k in pairs[0] if k.startswith("same_cluster_k")]:
        k = key.split("_k")[1]
        for same in (1, 0):
            ps = [p for p in pairs if p[key] == same]
            if ps:
                out.append({"what": f"mean |d_prob|, {'within' if same else 'across'} clusters",
                            "level": k, "n_pairs": len(ps),
                            "value": float(np.mean([p["abs_d_prob"] for p in ps]))})
                out.append({"what": f"mean sim, {'within' if same else 'across'} clusters",
                            "level": k, "n_pairs": len(ps),
                            "value": float(np.mean([p["sim"] for p in ps]))})
    _write_csv(Path(args.out) / f"{args.model}_swap.csv", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="results/e23", help="folder with <model>_e23.jsonl")
    ap.add_argument("--tau", type=float, default=0.02, help="min group importance I(G) (P units)")
    ap.add_argument("--delta", type=float, default=0.02, help="min I(A) - I(B) for an inversion pair")
    ap.add_argument("--levels", nargs="+", type=int, default=None)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    recs = read_jsonl(Path(args.out) / f"{args.model}_e23.jsonl")
    print(f"[e23] {args.model}: {len(recs)} videos")
    if not recs:
        return
    vis, red, mvr = e2_redundancy(recs, args)
    slopes = e2(recs, args)
    hits, invs, dels = e3(recs, args)
    sw = swap(recs, args)
    grp = read_jsonl(Path(args.out) / f"{args.model}_e23_group.jsonl")
    dil = dilution(recs, grp, args) if grp else None

    if dil:
        dp, dc, dt = dil
        print(f"\nDILUTION ({len(grp)} videos with group values)")
        print("  inversion: a frame-level reading ranks the LESS important segment above the more important one")
        print("  'mean' = average credit per frame (each frame of A looks less important); 'max' = best frame")
        print("  gain = chance - observed inversion rate (chance: frame scores rotated, so no longer aligned with segments)")
        print("  dilution predicts a smaller gain when the more important segment A is the larger one (gain diff < 0)")
        print(f"  {'method':15s} {'seed':>4s} {'k':>2s} {'readout':7s} {'A >=2x larger: inv/chance':>26s} {'same size: inv/chance':>22s} {'gain diff [95% CI]':>22s}")
        for r in dc:
            print(f"  {r['method']:15s} {str(r['seed']):>4s} {r['level']:2d} {r['readout']:7s} "
                  f"{r['inv_A_larger']:>15.2f} / {r['chance_A_larger']:.2f} {r['inv_same_size']:>13.2f} / {r['chance_same_size']:.2f}"
                  f"   {r['gain_difference']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]")
        print("  most important segment G*: top frame in G* (chance) | top-4 includes G* (chance) | median rank of G*'s best frame")
        for r in dt:
            print(f"  {r['method']:15s} {str(r['seed']):>4s} k={r['level']:<2d} |G*|={r['size_top']:4s} n={r['n']:4d}  "
                  f"{r['hit1']:.2f} ({r['chance1']:.2f}) | {r['top4']:.2f} ({r['chance4']:.2f}) | {r['median_rank']:.0f}")

    print(f"\nE2  visibility ({len(vis)} videos with P(c*|clip) - prior > 0.05), medians:")
    print(f"  single-frame importances added up / clip importance : {np.median([v['sum_loo_share'] for v in vis]):.3f}")
    print(f"  most important single frame / clip importance       : {np.median([v['max_loo_share'] for v in vis]):.3f}")
    for k in (1, 2, 4):
        cols = [c for c in vis[0] if c.startswith(f"left_after_{k}|")]
        best = np.median([min(v[c] for c in cols if not c.endswith("random") and "cluster" not in c) for v in vis])
        rnd = np.median([v[f"left_after_{k}|random"] for v in vis])
        print(f"  share of clip importance left after removing the top {k} frames: best method {best:.2f}, random {rnd:.2f}")
    print("\nE2  redundancy R(G) = 1 - sum of single-frame importances / I(G), groups with I(G) > tau")
    print(f"  {'size':5s} {'grouping':8s} {'groups':>7s} {'I>tau':>6s} {'mean sim':>9s} {'median R':>9s} {'R>0.5':>6s}")
    for r in red:
        print(f"  {r['size']:5s} {r['grouping']:8s} {r['n_groups']:7d} {r['n_important']:6d} {r['median_mean_sim']:9.3f} "
              f"{r['median_R']:9.2f} {r['share_R_above_half']:6.2f}")
    print("\nE2  method vs redundancy: median r_sum (sum of the group's credit / I(G)) [best frame x |G| / I(G)]")
    for r in mvr:
        print(f"  {r['method']:15s} {r['R_bin']:10s} n={r['n']:5d}  r_sum {r['median_r_sum']:6.2f}  [{r['median_best_share']:6.2f}]")

    print("\nE2  (confounded, for the record) slope of log(readout) on log|G|")
    print(f"{'method':15s} {'seed':>4s} {'readout':7s} {'grouping':13s} {'n':>6s} {'slope':>7s}   95% CI")
    for r in slopes:
        print(f"{r['method']:15s} {str(r['seed']):>4s} {r['readout']:7s} {r['grouping']:13s} "
              f"{r['n_groups']:6d} {r['slope']:7.2f}   [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]")
    print("\nE3  inversion rate (method ranks a LESS important cluster above a more important one)")
    print(f"{'method':15s} {'seed':>4s} {'level':>5s} {'size ratio':10s} {'pairs':>6s} {'max':>6s} {'sum':>6s}")
    for r in invs:
        print(f"{r['method']:15s} {str(r['seed']):>4s} {r['level']:5d} {r['size_ratio']:10s} "
              f"{r['n_pairs']:6d} {r['inv_rate_max']:6.2f} {r['inv_rate_sum']:6.2f}")
    print("\nE3  deletion AUC (lower = finds the important frames sooner)")
    for r in dels:
        print(f"  {r['order']:20s} n={r['n_videos']:4d}  auc={r['auc']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
              f"  after 1/2/4 frames: {r['after_1']:.2f} {r['after_2']:.2f} {r['after_4']:.2f}")
    if sw:
        print("\nSwap test")
        for r in sw:
            print(f"  {r['what']:40s} level={str(r['level']):3s} n={r['n_pairs']:5d}  {r['value']:.4f}")


if __name__ == "__main__":
    main()
