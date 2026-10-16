"""
fr_practical.py -- the frame-rate test (fr_run.py) with PRACTICAL groupings: groupings chosen
without knowing which part of the clip was sampled more densely. Does a group explanation keep
the denser segment's credit, where frame-level Shapley divides it, and do DINOv2 groups do this
better than fixed windows?

For every video in <fr>/<model>_fr.jsonl and every clip of it (base, target, control; the
same frames as in fr_run.py), and every grouping in --groupings (e23_run.practical_groups):
  dino_k<K>   the clip's frames embedded with DINOv2, the neighbours-only tree cut into K groups
  dino_thr    the same tree cut at the merge height tau of <e1>/dino_tau.json (calibrate_tau.py),
              between --min-groups and --max-groups groups
  dinoavg_*   the same with an average-linkage tree
  win_k<K>    K equal windows
  win_w<W>    windows of W frames
compute group Shapley (e1_common.group_shapley; same class, removal and prior as fr_run.py) and
turn it into per-frame BROADCAST scores (each frame shows its group's value, not divided;
e23_run.broadcast_scores). The scores are stored as attributions "grp_<grouping>", next to the
clip's original frame-level Shapley attributions, so fr_metrics.py reads the output unchanged:

    python motivation/fr_practical.py --model vjepa2 --fp16 --max-groups 8
    python motivation/fr_metrics.py --model vjepa2 --name fr_practical

Writes <fr>/<model>_fr_practical.jsonl: the input record with each clip's "attributions"
replaced by [frame-level Shapley] + [grp_*], and "practical": {grouping: {"k", "capped"}}.
Re-running resumes; --shard / --num-shards split the videos (merge the shard files afterwards).
"""
import argparse
import json
import time
from pathlib import Path

from e1_common import (MODEL_SPECS, Evaluator, FrameBank, _decode, append_jsonl, group_shapley,
                       load_clip_model, load_prior, make_layout, read_jsonl, resolve_video_path,
                       stable_seed)
from e23_run import broadcast_scores, embed, practical_groups, tau_key, tree_linkage, tree_nodes
from frame_clustering import hierarchy_from_embeddings

DEFAULT_GROUPINGS = ["dino_k4", "dino_k8", "dino_thr", "dinoavg_thr", "win_k4", "win_k8", "win_w4"]
KEEP_METHODS = {"shapley_drop"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    ap.add_argument("--groupings", nargs="+", default=DEFAULT_GROUPINGS)
    ap.add_argument("--fr", default="results/fr", help="folder with <model>_fr.jsonl")
    ap.add_argument("--tau-file", default="results/e1/dino_tau.json")
    ap.add_argument("--max-groups", type=int, default=12)
    ap.add_argument("--min-groups", type=int, default=2)
    ap.add_argument("--exact-max", type=int, default=12)
    ap.add_argument("--n-perm", type=int, default=64)
    ap.add_argument("--prior", default="uniform")
    ap.add_argument("--clusterer", default="dinov2", choices=["dinov2", "pixel"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    fr = Path(args.fr)
    recs = read_jsonl(fr / f"{args.model}_fr.jsonl")[:args.limit]
    suffix = ""
    if args.shard is not None:
        recs = [r for i, r in enumerate(recs) if i % args.num_shards == args.shard]
        suffix = f".shard{args.shard}"
    out_path = fr / f"{args.model}_fr_practical{suffix}.jsonl"
    done = {r["video"] for r in read_jsonl(out_path)}
    tau_json = json.loads(Path(args.tau_file).read_text())
    taus = {g: tau_json[tau_key(args.model, g)]["tau"] for g in args.groupings if g.endswith("_thr")}
    model = load_clip_model(args.model)
    prior = load_prior(model, args.prior)
    removal = model.spec["removal"]
    print(f"[fr practical] {args.model}: {len(recs)} videos ({len(done)} done), groupings={args.groupings}, "
          f"thresholds {taus}, {args.min_groups}-{args.max_groups} groups")

    for vi, rec in enumerate(recs):
        path = rec["video"]
        if path in done:
            continue
        t0 = time.time()
        all_idx = sorted({i for c in rec["clips"] for i in c["frame_indices"]})
        frames_all = _decode(resolve_video_path(path), all_idx)
        pos = {fi: k for k, fi in enumerate(all_idx)}
        bank = FrameBank(model, frames_all, "exact", stable_seed(path))
        ev = Evaluator(model, bank, args.batch_size, args.fp16)
        clips = []
        for c in rec["clips"]:
            content = [pos[i] for i in c["frame_indices"]]
            lay = make_layout(content)
            n = len(content)
            trees = {}
            links = {tree_linkage(g) for g in args.groupings} - {None}
            if links:
                emb = embed(frames_all[content], args.clusterer)
                trees = {lk: tree_nodes(hierarchy_from_embeddings(emb, lk), n) for lk in links}
            attrs = [a for a in c["attributions"] if a["method"] in KEEP_METHODS]
            info = {}
            for g in args.groupings:
                groups, inf = practical_groups(g, n, trees.get(tree_linkage(g)), taus.get(g),
                                               args.max_groups, args.min_groups)
                n0 = ev.n_evals
                vals = group_shapley(ev, lay, groups, rec["pred"], prior, removal,
                                     exact_max=args.exact_max, n_perm=args.n_perm,
                                     seed=stable_seed(path, c["kind"], c["d"], g))
                attrs.append({"method": f"grp_{g}", "seed": 0,
                              "scores": broadcast_scores(groups, vals, n).tolist(),
                              "groups": groups, "values": [float(v) for v in vals],
                              "n_evals": ev.n_evals - n0, "secs": 0.0})
                info[g] = inf
            clips.append({**c, "attributions": attrs, "practical": info})
        append_jsonl(out_path, {**rec, "clips": clips, "practical_secs": round(time.time() - t0, 1)})
        name = path.replace("\\", "/").split("/")[-1]
        print(f"[{vi + 1}/{len(recs)}] {name}: {len(clips)} clips ({time.time() - t0:.1f}s, {ev.n_evals} evals)")
    print(f"[fr practical] wrote {out_path}")


if __name__ == "__main__":
    main()
