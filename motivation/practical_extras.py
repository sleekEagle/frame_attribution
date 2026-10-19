"""
practical_extras.py -- three analyses of the practical-grouping runs, from existing files only
(no model runs):

  1. group sizes   how coarse each grouping is: number of groups per clip and frames per group,
                   on the E1 reference clips and the frame-rate test base clips
                   (the clips a user would see).
  2. rank CIs      95% bootstrap confidence intervals, resampling videos, for the ranking
                   readouts of e1_metrics.py --practical (validated inversions, top-1 loss) and
                   fr_metrics.py --name fr_practical (inversions, top-1 loss) at the highest level.
  3. cost          model evaluations per clip: each grouping's exact group Shapley against
                   frame-level Shapley (sampled orderings) on the same clips.

    python motivation/practical_extras.py

Writes results/practical_extras_{sizes,ranks,cost}.csv and prints the tables.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

E1 = Path("results/e1")
FR = Path("results/fr")
E1_MODELS = {"trn_official": "4", "r3d_18": "8", "r3d": "8", "mc3_18": "9", "vjepa2": "9"}
FR_MODELS = ["mc3_18", "vjepa2"]
PRACTICAL = ("dino_k", "dino_thr", "dinoavg_", "win_k", "win_w")
SHAPLEY = {"trn_official": "shapley_freeze", "r3d_18": "shapley_freeze", "r3d": "shapley_freeze",
           "mc3_18": "shapley_drop", "vjepa2": "shapley_drop"}
BOOT = 2000


def jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def boot_mean(values_by_video, seed=0):
    """Mean over all values, with a 95% CI from resampling videos."""
    vids = [v for v in values_by_video if values_by_video[v]]
    if not vids:
        return float("nan"), float("nan"), float("nan"), 0, 0
    allv = [x for v in vids for x in values_by_video[v]]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(BOOT):
        pick = rng.choice(len(vids), len(vids))
        xs = [x for i in pick for x in values_by_video[vids[i]]]
        bs.append(np.mean(xs))
    return float(np.mean(allv)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(allv), len(vids)


def f(x):
    return float(x) if x not in ("", None, "nan", "None") else float("nan")


def boolish(x):
    return {"True": 1.0, "False": 0.0}[x] if x in ("True", "False") else f(x)


# ---- 1. group sizes ---------------------------------------------------------------------------
def sizes():
    rows = []
    for m in E1_MODELS:
        by = defaultdict(lambda: {"k": [], "size": []})
        for g in jsonl(E1 / f"{m}_group_attributions.jsonl"):
            if not g["grouping"].startswith(PRACTICAL):
                continue
            n = sum(len(x) for x in g["groups"])
            if g["cond_id"].endswith("|reference"):
                by[g["grouping"]]["k"].append(len(g["groups"]))
                by[g["grouping"]]["size"] += [len(x) for x in g["groups"]]
                by[g["grouping"]]["n"] = n
        for gname, d in sorted(by.items()):
            rows.append({"test": "E1 reference clips", "model": m, "grouping": gname, "frames": d.get("n"),
                         "n_clips": len(d["k"]), "groups_median": float(np.median(d["k"])),
                         "groups_p10": float(np.percentile(d["k"], 10)), "groups_p90": float(np.percentile(d["k"], 90)),
                         "frames_per_group_mean": float(np.mean(d["size"]))})
    for m in FR_MODELS:
        by = defaultdict(lambda: {"k": [], "size": []})
        for r in jsonl(FR / f"{m}_fr_practical.jsonl"):
            for c in r["clips"]:
                if c["kind"] != "base":
                    continue
                for a in c["attributions"]:
                    if a["method"].startswith("grp_"):
                        g = a["method"][4:]
                        by[g]["k"].append(len(a["groups"]))
                        by[g]["size"] += [len(x) for x in a["groups"]]
        for gname, d in sorted(by.items()):
            rows.append({"test": "frame-rate base clips", "model": m, "grouping": gname, "frames": 16,
                         "n_clips": len(d["k"]), "groups_median": float(np.median(d["k"])),
                         "groups_p10": float(np.percentile(d["k"], 10)), "groups_p90": float(np.percentile(d["k"], 90)),
                         "frames_per_group_mean": float(np.mean(d["size"]))})
    write_csv(Path("results/practical_extras_sizes.csv"), rows)
    print(f"\n{'test':22s}{'model':14s}{'grouping':14s}{'groups: median [10-90%]':>26s}{'frames/group':>14s}")
    for r in rows:
        print(f"{r['test']:22s}{r['model']:14s}{r['grouping']:14s}{r['groups_median']:>12.1f} "
              f"[{r['groups_p10']:.0f}-{r['groups_p90']:.0f}]{'':6s}{r['frames_per_group_mean']:>10.2f}")
    return rows


# ---- 2. ranking CIs ---------------------------------------------------------------------------
def ranks():
    rows = []
    for m, mmax in E1_MODELS.items():
        rs = [r for r in csv.DictReader(open(E1 / f"{m}_practical_rows.csv"))
              if r["dup"] == "exact" and r["m"] == mmax]
        for meth in sorted({r["method"] for r in rs}):
            for kind in sorted({r["kind"] for r in rs}):
                sub = [r for r in rs if r["method"] == meth and r["kind"] == kind]
                inv, top = defaultdict(list), defaultdict(list)
                for r in sub:
                    if np.isfinite(f(r["inv_validated"])):
                        inv[r["video"]].append(f(r["inv_validated"]))
                    if np.isfinite(boolish(r["top1_loss"])):
                        top[r["video"]].append(boolish(r["top1_loss"]))
                for name, d in (("inversions_validated", inv), ("top1_loss", top)):
                    mu, lo, hi, n, nv = boot_mean(d)
                    rows.append({"test": "E1", "model": m, "level": f"m={mmax}", "method": meth, "kind": kind,
                                 "readout": name, "mean": mu, "ci_lo": lo, "ci_hi": hi, "n": n, "n_videos": nv})
    for m in FR_MODELS:
        rs = [r for r in csv.DictReader(open(FR / f"{m}_fr_practical_rows.csv")) if r["d"] == "3"]
        for meth in sorted({r["method"] for r in rs}):
            for kind in ("target", "control"):
                sub = [r for r in rs if r["method"] == meth and r["kind"] == kind]
                inv, top = defaultdict(list), defaultdict(list)
                for r in sub:
                    if np.isfinite(f(r["inv_clip"])):
                        inv[r["video"]].append(f(r["inv_clip"]))
                    if f(r["frame_top_base"]) == 1:
                        top[r["video"]].append(1 - f(r["frame_top_clip"]))
                for name, d in (("inversions", inv), ("top1_loss", top)):
                    mu, lo, hi, n, nv = boot_mean(d)
                    rows.append({"test": "frame-rate", "model": m, "level": "d=3", "method": meth, "kind": kind,
                                 "readout": name, "mean": mu, "ci_lo": lo, "ci_hi": hi, "n": n, "n_videos": nv})
    write_csv(Path("results/practical_extras_ranks.csv"), rows)
    print(f"\n{'test':11s}{'model':14s}{'method':17s}{'readout':22s}{'target: mean [95% CI] (n)':>32s}{'control':>26s}")
    for r in rows:
        if r["kind"] != "target":
            continue
        c = next(x for x in rows if x["test"] == r["test"] and x["model"] == r["model"] and x["method"] == r["method"]
                 and x["readout"] == r["readout"] and x["kind"] != "target")
        print(f"{r['test']:11s}{r['model']:14s}{r['method']:17s}{r['readout']:22s}"
              f"{r['mean']:>8.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}] ({r['n']:3d})"
              f"{c['mean']:>10.2f} [{c['ci_lo']:.2f}, {c['ci_hi']:.2f}]")
    return rows


# ---- 3. cost ----------------------------------------------------------------------------------
def cost():
    rows = []
    for m in E1_MODELS:
        ev = defaultdict(list)
        clips = set()
        for g in jsonl(E1 / f"{m}_group_attributions.jsonl"):
            if g["grouping"].startswith(PRACTICAL):
                clips.add((g["video"], g["cond_id"]))
                if g["n_evals"] > 0:  # 0 = values reused from an identical grouping of the same clip
                    ev[g["grouping"]].append(g["n_evals"])
        for a in jsonl(E1 / f"{m}_attributions.jsonl"):
            if a["method"] == SHAPLEY[m] and a["seed"] == 0 and (a["video"], a["cond_id"]) in clips:
                ev["frame Shapley"].append(a["n_evals"])
        for k, v in sorted(ev.items()):
            rows.append({"test": "E1", "model": m, "method": k, "evals_median": float(np.median(v)),
                         "evals_max": int(np.max(v)), "n_clips": len(v)})
    for m in FR_MODELS:
        ev = defaultdict(list)
        for r in jsonl(FR / f"{m}_fr_practical.jsonl"):
            for c in r["clips"]:
                for a in c["attributions"]:
                    k = "frame Shapley" if a["method"] == "shapley_drop" else a["method"][4:]
                    ev[k].append(a["n_evals"])
        for k, v in sorted(ev.items()):
            rows.append({"test": "frame-rate", "model": m, "method": k, "evals_median": float(np.median(v)),
                         "evals_max": int(np.max(v)), "n_clips": len(v)})
    write_csv(Path("results/practical_extras_cost.csv"), rows)
    print(f"\n{'test':11s}{'model':14s}{'method':15s}{'evals/clip median':>18s}{'max':>8s}")
    for r in rows:
        print(f"{r['test']:11s}{r['model']:14s}{r['method']:15s}{r['evals_median']:>18.0f}{r['evals_max']:>8d}")
    return rows


if __name__ == "__main__":
    sizes()
    ranks()
    cost()
