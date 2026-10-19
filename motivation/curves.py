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

Writes <out>/<model>_curves_<source>.jsonl, one line per clip:
    {"video", "clip", "n", "p_full", "prior", "curves": {method: {"del": [...], "ins": [...]}},
     "n_evals"}
Re-running resumes; --shard / --num-shards split the videos. Analysis: curves_metrics.py.

    python motivation/curves.py --model r3d_18 --source e23 --fp16
    python motivation/curves.py --model vjepa2 --source fr --fp16
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
    args = ap.parse_args()

    src = Path(args.e23) / f"{args.model}_e23.jsonl" if args.source == "e23" else \
        Path(args.fr) / f"{args.model}_fr_practical.jsonl"
    recs = read_jsonl(src)[:args.limit]
    suffix = ""
    if args.shard is not None:
        recs = [r for i, r in enumerate(recs) if i % args.num_shards == args.shard]
        suffix = f".shard{args.shard}"
    out_path = Path(args.out) / f"{args.model}_curves_{args.source}{suffix}.jsonl"
    done = {(r["video"], r["clip"]) for r in read_jsonl(out_path)}
    model = load_clip_model(args.model)
    prior = load_prior(model, args.prior)
    taus = {}
    if args.source == "e23" and any(g.endswith("_thr") for g in args.groupings):
        tau_json = json.loads(Path(args.tau_file).read_text())
        taus = {g: tau_json[tau_key(args.model, g)]["tau"] for g in args.groupings if g.endswith("_thr")}
    print(f"[curves] {args.model} {args.source}: {len(recs)} videos ({len(done)} clips done), "
          f"groupings={args.groupings if args.source == 'e23' else 'stored'}")

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
            else:
                layout = make_layout([pos[i] for i in c["frame_indices"]])
                n = len(c["frame_indices"])
                scores = {a["method"]: a["scores"] for a in c["attributions"] if a["seed"] == 0}
            scores["random"] = list(np.random.default_rng(stable_seed(path, clip_name, "random")).random(n))
            p_full = float(ev.value(layout, np.ones((1, n), bool), removal, cls, prior)[0])
            out = {}
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
