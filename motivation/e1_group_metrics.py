"""
e1_group_metrics.py -- does group-level attribution remove dilution in E1? Compares, on the same
gate-passing targets, frame-level Shapley (from e1_attribute.py) with group Shapley (from
e1_group.py) and with the model's own reliance on the target.

    python motivation/e1_group_metrics.py --model r3d_18

Per target condition (target t, duplication level m) and its control, relative to the
reference clip of the same copy type:
  I_t ratio        I^(m)(t) / I_ref(t): how much the model relies on t (remove every copy of t)
  frame per-copy   mean frame-level Shapley score of t's copies / the same in the reference
  frame total      summed scores of t's copies / the same in the reference
  group <g>        group Shapley value of the group holding t (its first original slot) /
                   the same in the reference, for grouping g = oracle (one group per content),
                   dino (DINOv2, same number of groups), dino_half (DINOv2, half as many)
Only targets with a positive reference value are used for each readout (as in E1).
  beta             slope of log(ratio) on log(m / m_ref), through 0, per target: 0 = the
                   readout doesn't change with m, -1 = it falls as 1/m (credit divided evenly)
  top-1 loss       among targets that were top-ranked in the reference, the share that aren't in
                   the target clip: frame level by best copy among contents, group level by value
                   among groups
Writes <out>/<model>_group_rows.csv and _group_summary.csv, and prints the tables.
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from e1_common import read_jsonl

SHAPLEY = {"realloc": "shapley_freeze", "insert": "shapley_drop", "replace": "shapley_drop"}
READOUTS = ["I_t", "frame_per_copy", "frame_total", "group_oracle", "group_dino", "group_dino_half"]


def _write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[e1 group] wrote {path} ({len(rows)} rows)")


def first_slot(layout, c):
    """The slot of content c's original frame (copy 0)."""
    return next(i for i, (cc, k) in enumerate(zip(layout["content"], layout["copy"])) if cc == c and k == 0)


def group_value_of(ga, slot):
    for g, v in zip(ga["groups"], ga["values"]):
        if slot in g:
            return v
    return float("nan")


def top_content_frame(scores, layout, t):
    """Is t the top content by best copy (frame-level reading)?"""
    best = defaultdict(lambda: -math.inf)
    for s, c in zip(scores, layout["content"]):
        best[c] = max(best[c], s)
    return max(best, key=best.get) == t


def top_group(ga, slot):
    j = int(np.argmax(ga["values"]))
    return slot in ga["groups"][j]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="results/e1")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    d = Path(args.out)
    recs = {r["video"]: r for r in read_jsonl(d / f"{args.model}_conditions.jsonl")}
    design = next(iter(recs.values()))["design"]
    m_ref = 2 if design == "realloc" else 1
    sh = SHAPLEY[design]
    attr = {(a["video"], a["cond_id"]): a for a in read_jsonl(d / f"{args.model}_attributions.jsonl")
            if a["method"] == sh and a["seed"] == 0}
    grp = defaultdict(dict)
    for g in read_jsonl(d / f"{args.model}_group_attributions.jsonl"):
        grp[(g["video"], g["cond_id"])][g["grouping"]] = g
    videos = sorted({v for v, _ in grp})

    rows = []
    for v in videos:
        rec = recs[v]
        conds = {c["id"]: c for c in rec["conditions"]}
        for c in rec["conditions"]:
            if c["kind"] == "reference" or (v, c["id"]) not in grp:
                continue
            ref = conds[f"{c['dup']}|reference"]
            t = c["target"]
            if (v, ref["id"]) not in grp:
                continue
            row = {"video": v, "cond_id": c["id"], "dup": c["dup"], "kind": c["kind"], "m": c["m"],
                   "target": t, "tier": c.get("tier")}
            I_ref, I_c = ref["I"].get(str(t)), c["I"].get(str(t)) if c.get("I") else None
            row["I_t"] = I_c / I_ref if I_ref and I_c is not None and I_ref > 0 else float("nan")
            a_c, a_r = attr.get((v, c["id"])), attr.get((v, ref["id"]))
            if a_c and a_r:
                sc, sr = np.asarray(a_c["scores"]), np.asarray(a_r["scores"])
                tc = [i for i, cc in enumerate(c["layout"]["content"]) if cc == t]
                tr = [i for i, cc in enumerate(ref["layout"]["content"]) if cc == t]
                row["frame_per_copy"] = sc[tc].mean() / sr[tr].mean() if sr[tr].mean() > 0 else float("nan")
                row["frame_total"] = sc[tc].sum() / sr[tr].sum() if sr[tr].sum() > 0 else float("nan")
                row["frame_top_ref"] = int(top_content_frame(sr, ref["layout"], t))
                row["frame_top_cond"] = int(top_content_frame(sc, c["layout"], t))
            s_c, s_r = first_slot(c["layout"], t), first_slot(ref["layout"], t)
            for gname in ("oracle", "dino", "dino_half"):
                gc, gr = grp[(v, c["id"])].get(gname), grp[(v, ref["id"])].get(gname)
                if not (gc and gr):
                    continue
                vr = group_value_of(gr, s_r)
                row[f"group_{gname}"] = group_value_of(gc, s_c) / vr if vr > 0 else float("nan")
                row[f"group_{gname}_top_ref"] = int(top_group(gr, s_r))
                row[f"group_{gname}_top_cond"] = int(top_group(gc, s_c))
                if gname != "oracle":
                    row[f"group_{gname}_recovered"] = gc.get("recovered")
            rows.append(row)
    _write_csv(d / f"{args.model}_group_rows.csv", rows)

    # summaries
    summ = []
    ms = sorted({r["m"] for r in rows})
    for dup in ("exact", "noise"):
        for kind in sorted({r["kind"] for r in rows}):
            for ro in READOUTS:
                line = {"dup": dup, "kind": kind, "readout": ro}
                for m in ms:
                    x = [r[ro] for r in rows if r["dup"] == dup and r["kind"] == kind and r["m"] == m
                         and ro in r and np.isfinite(r[ro])]
                    line[f"median_m{m}"] = float(np.median(x)) if x else float("nan")
                    line[f"n_m{m}"] = len(x)
                # beta per target (video, target), through the origin
                per = defaultdict(list)
                for r in rows:
                    if r["dup"] == dup and r["kind"] == kind and ro in r and np.isfinite(r[ro]) and r[ro] > 0:
                        per[(r["video"], r["target"])].append((math.log(r["m"] / m_ref), math.log(r[ro])))
                betas = [sum(a * b for a, b in p) / sum(a * a for a, _ in p) for p in per.values()
                         if sum(a * a for a, _ in p) > 0]
                if betas:
                    rng = np.random.default_rng(0)
                    bs = [float(np.median(rng.choice(betas, len(betas)))) for _ in range(args.boot)]
                    line.update(beta=float(np.median(betas)), beta_lo=float(np.percentile(bs, 2.5)),
                                beta_hi=float(np.percentile(bs, 97.5)), n_beta=len(betas))
                summ.append(line)
            # top-1 loss
            for ro in ("frame", "group_oracle", "group_dino", "group_dino_half"):
                line = {"dup": dup, "kind": kind, "readout": f"top1_loss_{ro}"}
                for m in ms:
                    x = [1 - r[f"{ro}_top_cond"] for r in rows if r["dup"] == dup and r["kind"] == kind
                         and r["m"] == m and r.get(f"{ro}_top_ref") == 1 and f"{ro}_top_cond" in r]
                    line[f"median_m{m}"] = float(np.mean(x)) if x else float("nan")
                    line[f"n_m{m}"] = len(x)
                summ.append(line)
    _write_csv(d / f"{args.model}_group_summary.csv", summ)

    rec_rate = {g: np.mean([r[f"group_{g}_recovered"] for r in rows if r.get(f"group_{g}_recovered") is not None])
                for g in ("dino", "dino_half") if any(r.get(f"group_{g}_recovered") is not None for r in rows)}
    print(f"\n{args.model}: {len(videos)} videos, {len(rows)} target/control clips; frame-level method {sh}")
    print("DINOv2 puts the target's copies in a group of their own: " +
          ", ".join(f"{g} {v:.0%}" for g, v in rec_rate.items()))
    for dup in ("exact",):
        for kind in sorted({r["kind"] for r in rows}):
            print(f"\n  {dup}, {kind}: medians at m = {ms}  (top-1 loss: share of reference-top targets no longer top)")
            for line in summ:
                if line["dup"] == dup and line["kind"] == kind:
                    vals = "  ".join(f"{line[f'median_m{m}']:6.2f}" for m in ms)
                    b = f"   beta {line['beta']:+.2f} [{line['beta_lo']:+.2f}, {line['beta_hi']:+.2f}] n={line['n_beta']}" if "beta" in line else ""
                    print(f"    {line['readout']:24s} {vals}{b}")


if __name__ == "__main__":
    main()
