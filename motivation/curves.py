"""
curves.py -- deletion and insertion curves: a faithfulness measure that doesn't depend on our
own readouts. Do group explanations (each frame shows its group's undivided Shapley value) pick
out the frames the model needs better than frame-level attributions?

For a clip and an attribution (one score per frame), order the frames by score, highest first
(ties, as within a group, broken at random, seeded per video and method):
  deletion   P(predicted class) after removing the first j frames of the order, j = 0..n
             (j = n leaves an empty clip: the prior). A faithful order makes it fall fast.
  insertion  P(predicted class) with ONLY the first j frames kept, j = 0..n. A faithful order
             makes it rise fast.
Frames are removed with the model's own removal (drop or late freeze), as everywhere else.
Every method gets the same budget: one frame per step, n steps.

Sources (--source):
  e23  the E2/E3 clips (<e23>/<model>_e23.jsonl): natural videos at the model's own n frames.
       Frame-level methods: the stored attributions (seed 0). Groupings (--groupings, as in
       e23_run.practical_groups): the DINOv2 Ward tree is the stored one; the average-linkage
       tree is rebuilt from the stored DINOv2 similarities; group Shapley (exact) is computed
       here. A random order is added as a baseline.
  fr   the frame-rate test clips (<fr>/<model>_fr_practical.jsonl, from fr_practical.py): the
       base clip and the target clip at the highest density, with the stored frame-level
       Shapley and broadcast scores of every grouping, plus a random order.

Group-wise curves (--groupwise G [G ...]): do group values rank the parts of a clip better than
frame-level Shapley values do? For each named grouping, the curves remove (or keep) whole groups,
one per step, and only the ranking of the groups differs:
  gw_<G>_group   by the group's exact Shapley value
  gw_<G>_mean    by the mean frame-level Shapley value of its frames (the ranking dilution affects)
  gw_<G>_sum     by the sum of its frame-level Shapley values
  gw_<G>_random  at random
  gw_<G>_mean_<M>, gw_<G>_sum_<M>   by the mean / sum of the frame scores of every other stored
                 frame-level method M (e23 only: the fr records store Shapley alone)
(ties broken at random). Each curve is stored on two axes. On the frames axis ("del", "ins";
n + 1 points), after j frames the groups completed by j are removed (kept); this axis rewards
importance per frame, which is what a diluted mean score measures, so it favours the mean
ranking. On the groups axis ("del_groups", "ins_groups"; k + 1 points), each step is one part of
the clip; this axis favours ranking by total importance. Both are reported. No frame-wise curves are computed in this mode, and the
output is <model>_curves_<source>_gwm.jsonl (_gw.jsonl in earlier runs, with Shapley alone). In fr, the groups are the runs of equal broadcast
scores (every grouping is contiguous); a clip whose run count differs from the stored number of
groups (two neighbouring groups with the same value) skips that grouping.

Writes <out>/<model>_curves_<source>.jsonl, one line per clip:
    {"video", "clip", "n", "p_full", "prior", "curves": {method: {"del": [...], "ins": [...]}},
     "n_evals"}
Re-running resumes; --shard / --num-shards split the videos. Analysis: curves_metrics.py.

    python motivation/curves.py --model r3d_18 --source e23 --fp16
    python motivation/curves.py --model vjepa2 --source fr --fp16
    python motivation/curves.py --model r3d_18 --source e23 --fp16 --groupwise dinoavg_thr win_k8
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from e1_common import (MODEL_SPECS, Evaluator, FrameBank, _decode, append_jsonl, group_shapley,
                       load_clip_model, load_prior, make_layout, read_jsonl, resolve_video_path,
                       stable_seed)
from e23_run import broadcast_scores, practical_groups, tau_key, tree_linkage, tree_nodes

DEFAULT_GROUPINGS = ["dinoavg_thr", "dino_thr", "dino_k4", "dino_k8", "win_k4", "win_k8"]
SHAPLEY = ("shapley_drop", "shapley_freeze")  # the frame-level Shapley attribution, by removal


def order_of(scores, seed):
    """Frames by score, highest first; ties broken at random (seeded)."""
    s = np.asarray(scores, float)
    tie = np.random.default_rng(seed).random(len(s))
    return [int(i) for i in np.lexsort((tie, -s))]


def curves(ev, layout, order, removal, cls, prior):
    """Deletion and insertion values along `order` (n + 1 points each)."""
    n = len(order)
    dele = np.ones((n + 1, n), bool)
    ins = np.zeros((n + 1, n), bool)
    for j in range(1, n + 1):
        dele[j:, order[j - 1]] = False
        ins[j:, order[j - 1]] = True
    v = ev.value(layout, np.concatenate([dele, ins]), removal, cls, prior)
    return [float(x) for x in v[:n + 1]], [float(x) for x in v[n + 1:]]


def group_curves(ev, layout, groups, values, removal, cls, prior, seed):
    """Deletion and insertion with whole groups as steps, best group first. Returns the curves on
    the frames axis (n + 1 points: after j frames, the groups completed by j are removed / kept)
    and on the groups axis (k + 1 points, one per group)."""
    n = len(layout["content"])
    k = len(groups)
    tie = np.random.default_rng(seed).random(k)
    order = np.lexsort((tie, -np.asarray(values, float)))
    dele = np.ones((k + 1, n), bool)
    ins = np.zeros((k + 1, n), bool)
    for s, gi in enumerate(order, 1):
        dele[s:, groups[gi]] = False
        ins[s:, groups[gi]] = True
    v = ev.value(layout, np.concatenate([dele, ins]), removal, cls, prior)
    done = np.cumsum([0] + [len(groups[gi]) for gi in order])  # frames gone after s groups
    step = np.searchsorted(done, np.arange(n + 1), side="right") - 1
    return ([float(v[s]) for s in step], [float(v[k + 1 + s]) for s in step],
            [float(x) for x in v[:k + 1]], [float(x) for x in v[k + 1:]])


def score_runs(scores):
    """Groups of a contiguous grouping from its broadcast scores: maximal runs of equal scores."""
    groups = [[0]]
    for i in range(1, len(scores)):
        if scores[i] == scores[i - 1]:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def avg_linkage_nodes(sim, n):
    from frame_clustering import hierarchy_from_distances
    D = np.sqrt(np.clip(2 - 2 * np.asarray(sim, float), 0, None))
    np.fill_diagonal(D, 0)
    return tree_nodes(hierarchy_from_distances(D, "average"), n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    ap.add_argument("--source", required=True, choices=["e23", "fr"])
    ap.add_argument("--e23", default="results/e23")
    ap.add_argument("--fr", default="results/fr")
    ap.add_argument("--out", default="results/curves")
    ap.add_argument("--tau-file", default="results/e1/dino_tau.json")
    ap.add_argument("--groupings", nargs="+", default=DEFAULT_GROUPINGS)
    ap.add_argument("--max-groups", type=int, default=8)
    ap.add_argument("--min-groups", type=int, default=2)
    ap.add_argument("--prior", default="uniform")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--groupwise", nargs="+", default=None,
                    help="group-wise curves for these groupings, with four rankings of the same groups")
    args = ap.parse_args()
    if args.groupwise:
        args.groupings = args.groupwise

    src = Path(args.e23) / f"{args.model}_e23.jsonl" if args.source == "e23" else \
        Path(args.fr) / f"{args.model}_fr_practical.jsonl"
    recs = read_jsonl(src)[:args.limit]
    suffix = ""
    if args.shard is not None:
        recs = [r for i, r in enumerate(recs) if i % args.num_shards == args.shard]
        suffix = f".shard{args.shard}"
    gw = "_gwm" if args.groupwise else ""
    out_path = Path(args.out) / f"{args.model}_curves_{args.source}{gw}{suffix}.jsonl"
    done = {(r["video"], r["clip"]) for r in read_jsonl(out_path)}
    model = load_clip_model(args.model)
    prior = load_prior(model, args.prior)
    taus = {}
    if args.source == "e23" and any(g.endswith("_thr") for g in args.groupings):
        tau_json = json.loads(Path(args.tau_file).read_text())
        taus = {g: tau_json[tau_key(args.model, g)]["tau"] for g in args.groupings if g.endswith("_thr")}
    print(f"[curves] {args.model} {args.source}: {len(recs)} videos ({len(done)} clips done), "
          f"groupings={args.groupings if args.source == 'e23' or gw else 'stored'}{', group-wise' if gw else ''}")

    for vi, rec in enumerate(recs):
        path = rec["video"]
        t0 = time.time()
        if args.source == "e23":
            todo = [("clip", None)] if (path, "clip") not in done else []
        else:
            dmax = max(c["d"] for c in rec["clips"])
            todo = [(f"{c['kind']}_d{c['d']}", c) for c in rec["clips"]
                    if (c["kind"] == "base" or (c["kind"] == "target" and c["d"] == dmax))
                    and (path, f"{c['kind']}_d{c['d']}") not in done]
        if not todo:
            continue
        cls = rec["pred"]
        if args.source == "e23":
            if args.model.startswith("toy"):
                frames, _ = model.load_frames(path)
            else:
                frames = _decode(resolve_video_path(path), rec["frame_indices"])
            pos_of = list(range(len(frames)))
        else:
            all_idx = sorted({i for c in rec["clips"] for i in c["frame_indices"]})
            frames = _decode(resolve_video_path(path), all_idx)
            pos = {fi: k for k, fi in enumerate(all_idx)}
        bank = FrameBank(model, frames, "exact", stable_seed(path))
        ev = Evaluator(model, bank, args.batch_size, args.fp16)
        removal = rec.get("removal") or model.spec["removal"]

        for clip_name, c in todo:
            n0 = ev.n_evals
            gw_groups = {}  # grouping -> (groups, group values), for the group-wise curves
            if args.source == "e23":
                n = rec["n"]
                layout = make_layout(pos_of[:n])
                scores = {a["method"]: a["scores"] for a in rec["attributions"] if a["seed"] == 0}
                trees = {"ward": rec["nodes"]}
                if any(tree_linkage(g) == "average" for g in args.groupings):
                    trees["average"] = avg_linkage_nodes(rec["sim"], n)
                for g in args.groupings:
                    groups, _ = practical_groups(g, n, trees.get(tree_linkage(g)), taus.get(g),
                                                 args.max_groups, args.min_groups)
                    vals = group_shapley(ev, layout, groups, cls, prior, removal)
                    scores[f"grp_{g}"] = broadcast_scores(groups, vals, n).tolist()
                    gw_groups[g] = (groups, list(vals))
            else:
                layout = make_layout([pos[i] for i in c["frame_indices"]])
                n = len(c["frame_indices"])
                scores = {a["method"]: a["scores"] for a in c["attributions"] if a["seed"] == 0}
                if gw:
                    scores = {m: v for m, v in scores.items()
                              if m in SHAPLEY or m in {f"grp_{g}" for g in args.groupings}}
                    for g in args.groupings:
                        groups = score_runs(scores[f"grp_{g}"])
                        if len(groups) != c["practical"][g]["k"]:
                            print(f"  [skip] {clip_name} {g}: {len(groups)} runs, {c['practical'][g]['k']} groups")
                            continue
                        gw_groups[g] = (groups, [scores[f"grp_{g}"][grp[0]] for grp in groups])
            p_full = float(ev.value(layout, np.ones((1, n), bool), removal, cls, prior)[0])
            out = {}
            if gw:  # the same groups, removed whole; only their ranking differs
                phi = np.asarray(next(scores[m] for m in SHAPLEY if m in scores), float)
                for g, (groups, psi) in gw_groups.items():
                    ranks = {"group": psi,
                             "mean": [float(phi[grp].mean()) for grp in groups],
                             "sum": [float(phi[grp].sum()) for grp in groups],
                             "random": list(np.random.default_rng(stable_seed(path, clip_name, "gw", g)).random(len(groups)))}
                    for rname, vals in ranks.items():
                        d, i, dg, ig = group_curves(ev, layout, groups, vals, removal, cls, prior,
                                                    stable_seed(path, clip_name, "gw", g, rname))
                        out[f"gw_{g}_{rname}"] = {"del": d, "ins": i, "del_groups": dg, "ins_groups": ig}
                    for m, sc in scores.items():  # the other frame-level methods, mean and sum
                        if m in SHAPLEY or m.startswith("grp_"):
                            continue
                        sc = np.asarray(sc, float)
                        for agg in ("mean", "sum"):
                            vals = [float(getattr(sc[grp], agg)()) for grp in groups]
                            d, i, dg, ig = group_curves(ev, layout, groups, vals, removal, cls, prior,
                                                        stable_seed(path, clip_name, "gw", g, agg, m))
                            out[f"gw_{g}_{agg}_{m}"] = {"del": d, "ins": i, "del_groups": dg, "ins_groups": ig}
            else:
                scores["random"] = list(np.random.default_rng(stable_seed(path, clip_name, "random")).random(n))
                for meth, s in scores.items():
                    d, i = curves(ev, layout, order_of(s, stable_seed(path, clip_name, meth)), removal, cls, prior)
                    out[meth] = {"del": d, "ins": i}
            append_jsonl(out_path, {"model": args.model, "video": path, "clip": clip_name, "n": n,
                                    "p_full": p_full, "prior": float(prior[cls]), "curves": out,
                                    "n_evals": ev.n_evals - n0})
        name = path.replace("\\", "/").split("/")[-1]
        print(f"[{vi + 1}/{len(recs)}] {name}: {len(todo)} clip(s) ({time.time() - t0:.1f}s, {ev.n_evals} evals)")
    print(f"[curves] wrote {out_path}")


if __name__ == "__main__":
    main()
