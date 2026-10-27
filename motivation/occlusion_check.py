"""
occlusion_check.py -- does the model respond to the flat occlusion frame itself?

Occlusion replaces one slot with a flat frame (0 in normalised space). Leave-one-out removes the
same slot with the model's own removal (freeze or drop). Both are drops in the predicted class's
probability for the same slot, so D = occlusion - LOO is the effect of inserting the flat frame,
relative to that removal. Reads the stored E1 attributions only; runs no model.

  A. Reference clips: occlusion, LOO and D per slot, overall and for slots whose content has
     low importance |I_ref|. On the realloc models (m_ref = 2) every content is shown twice, so
     removing one copy leaves the other and the evidence term is already small.
  B. Target content: per-copy occlusion, LOO and D against m, for target and control clips.
     If D stays roughly constant while LOO falls, occlusion's per-copy score falls less than LOO's.

    python motivation/occlusion_check.py
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from e1_common import read_jsonl

MODELS = {"r3d": "R3D-50", "r3d_18": "R3D-18", "mc3_18": "MC3-18", "vjepa2": "V-JEPA2",
          "trn_official": "TRN"}
LOO_OF = {"realloc": "loo_freeze", "insert": "loo_drop"}


def _intkeys(d):
    return {int(k): v for k, v in d.items()}


def load(out, model):
    recs = {r["video"]: r for r in read_jsonl(out / f"{model}_conditions.jsonl")}
    design = next(iter(recs.values()))["design"]
    loo = LOO_OF[design]
    scores = defaultdict(dict)
    for a in read_jsonl(out / f"{model}_attributions.jsonl"):
        if a["method"] in ("occlusion", loo) and a["cond_id"].startswith("exact|"):
            scores[(a["video"], a["cond_id"])][a["method"]] = (np.asarray(a["scores"], float),
                                                              a["content"])
    return recs, design, loo, scores


def part_a(recs, loo, scores):
    occ_all, loo_all, imp_all = [], [], []
    for v, r in recs.items():
        s = scores.get((v, "exact|reference"))
        if not s or "occlusion" not in s or loo not in s:
            continue
        I_ref = _intkeys(r["I_ref"])
        occ, content = s["occlusion"]
        lo, _ = s[loo]
        occ_all += list(occ)
        loo_all += list(lo)
        imp_all += [I_ref[c] for c in content]
    occ_all, loo_all, imp_all = map(np.asarray, (occ_all, loo_all, imp_all))
    d = occ_all - loo_all
    low = np.abs(imp_all) <= np.quantile(np.abs(imp_all), 1 / 3)
    return {
        "n_slots": len(d),
        "med_absI": np.median(np.abs(imp_all)),
        "med_occ": np.median(occ_all), "med_loo": np.median(loo_all), "med_D": np.median(d),
        "med_absD": np.median(np.abs(d)),
        "low_med_absI": np.median(np.abs(imp_all[low])),
        "low_med_occ": np.median(occ_all[low]), "low_med_loo": np.median(loo_all[low]),
        "low_frac_occ_gt_I": np.mean(np.abs(occ_all[low]) > np.abs(imp_all[low])),
        "rho_occ_I": spearmanr(occ_all, imp_all)[0], "rho_loo_I": spearmanr(loo_all, imp_all)[0],
    }


def part_b(recs, loo, scores):
    """Per-copy medians of the target content, by condition kind and m (exact copies)."""
    rows = defaultdict(lambda: defaultdict(list))
    for v, r in recs.items():
        for c in r["conditions"]:
            if c["dup"] != "exact":
                continue
            s = scores.get((v, c["id"]))
            if not s or "occlusion" not in s or loo not in s:
                continue
            occ, content = s["occlusion"]
            lo, _ = s[loo]
            if c["kind"] == "reference":
                targets = [t["content"] for t in r["targets"]]
                key_m = r["m_ref"]
            else:
                targets = [c["target"]]
                key_m = c["m"]
            for t in targets:
                idx = [i for i, cc in enumerate(content) if cc == t]
                if not idx:
                    continue
                kind = "reference" if c["kind"] == "reference" else c["kind"].split("_")[0]
                row = rows[(kind, key_m)]
                row["occ"].append(np.mean(occ[idx]))
                row["loo"].append(np.mean(lo[idx]))
                row["D"].append(np.mean(occ[idx] - lo[idx]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/e1")
    args = ap.parse_args()
    out = Path(args.out)

    print("A. Reference clips, per slot (drops in probability; medians)")
    print(f"{'model':<9}{'LOO':<11}{'n':>6}{'|I|':>8}{'occ':>8}{'LOO':>8}{'D':>8}{'|D|':>8}"
          f" | low-|I| third:{'|I|':>7}{'occ':>8}{'LOO':>8}{'|occ|>|I|':>10}"
          f" | rho(occ,I){'':>1}rho(LOO,I)")
    results = {}
    for m, name in MODELS.items():
        recs, design, loo, scores = load(out, m)
        a = part_a(recs, loo, scores)
        results[m] = (recs, loo, scores)
        print(f"{name:<9}{loo:<11}{a['n_slots']:>6}{a['med_absI']:>8.4f}{a['med_occ']:>8.4f}"
              f"{a['med_loo']:>8.4f}{a['med_D']:>8.4f}{a['med_absD']:>8.4f}"
              f" | {'':>14}{a['low_med_absI']:>7.4f}{a['low_med_occ']:>8.4f}{a['low_med_loo']:>8.4f}"
              f"{a['low_frac_occ_gt_I']:>10.2f} | {a['rho_occ_I']:>9.2f}{a['rho_loo_I']:>11.2f}")

    print("\nB. Target content, mean over its copies, medians over targets (exact copies)")
    for m, name in MODELS.items():
        recs, loo, scores = results[m]
        rows = part_b(recs, loo, scores)
        print(f"\n{name} ({loo})")
        print(f"  {'clip':<11}{'m':>3}{'n':>6}{'occ':>9}{'LOO':>9}{'D=occ-LOO':>11}")
        for (kind, mm) in sorted(rows, key=lambda k: (k[0] != "reference", k[0], k[1])):
            r = rows[(kind, mm)]
            print(f"  {kind:<11}{mm:>3}{len(r['occ']):>6}{np.median(r['occ']):>9.4f}"
                  f"{np.median(r['loo']):>9.4f}{np.median(r['D']):>11.4f}")


if __name__ == "__main__":
    main()
