"""
e23_run.py -- E2 and E3: attribution dilution in NATURAL videos, using groups of similar frames.

E1 creates exact copies. E2 and E3 ask whether the same thing happens with the similar frames
that real videos already contain. Per video (one JSONL line in <out>/<model>_e23.jsonl;
re-running resumes):

  1. Load the model's own frames (motivation/e1_common, the same sampling as E1) and predict.
     Misclassified videos are skipped unless --include-wrong. Everything explains the
     predicted class c*. The clip is the natural one: every frame once, no duplication (also
     for the reallocation models r3d / r3d_18, whose E1 reference repeated each frame).
  2. Cluster THESE frames on the fly: DINOv2 embeddings, Ward linkage, only neighbouring frames
     merge (frame_clustering.py). The tree has 2n-1 nodes: the n single frames, every merge,
     and the whole clip. Each node is a contiguous run of similar frames.
     Also stored: the DINOv2 cosine similarity of every frame pair.
  3. Group importance I(G) = P(c*|clip) - P(c*|clip without the frames of G), for every tree
     node and for fixed windows of 2, 4, 8 frames (the control grouping, which ignores
     similarity). Removal as in E1: "drop" for models that accept shorter inputs, "late"
     freeze otherwise; computed for every removal that a chosen method uses. I of a single
     frame is its leave-one-out value.
  4. Per-frame attributions with the E1 methods (e1_common.run_method) on the natural clip.
  5. Deletion curves: P(c*) after removing the top-k frames, k = 0..n, in the order given by
     each method (seed 0), by cluster importance at some tree levels (--deletion-levels), and
     in a random order.
  6. Swap test (do DINOv2 clusters match how the video model sees the frames?): for each pair
     of neighbouring frames, replace one by the other and record the change in the model's
     output. e23_metrics.py relates it to DINOv2 similarity and to cluster membership.

Analysis: motivation/e23_metrics.py.

    python motivation/e23_run.py --model mc3_18 --limit 200 --fp16
    python motivation/e23_run.py --model vjepa2 --limit 100 --fp16
    python motivation/e23_run.py --model r3d --limit 200
    python motivation/e23_run.py --model toy --limit 4 --clusterer pixel     # smoke test
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch

from e1_common import (MODEL_SPECS, STOCHASTIC, Evaluator, FrameBank, append_jsonl, get_videos,
                       js_div, load_clip_model, load_prior, make_layout, read_jsonl, run_method,
                       softmax_np, stable_seed)

DEFAULT_METHODS = {
    "insert": ["shapley_drop", "loo_drop", "occlusion", "ig", "gradcam"],
    "replace": ["shapley_drop", "loo_drop", "occlusion", "ig", "gradcam"],
    "realloc": ["shapley_freeze", "loo_freeze", "occlusion", "ig", "gradcam"],
}
# which removal a method's scores are in (group importance must use the same one)
METHOD_REMOVAL = {"shapley_drop": "drop", "loo_drop": "drop", "playfair": "drop",
                  "shapley_freeze": "late", "loo_freeze": "late"}


# --------------------------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------------------------
def embed(frames_u8: torch.Tensor, clusterer: str) -> np.ndarray:
    """(n,C,H,W) uint8 -> (n,D) embeddings. "dinov2" for real runs; "pixel" (16x16 downsampled
    pixels) for smoke tests without the DINOv2 download."""
    if clusterer == "dinov2":
        from frame_clustering import _get_dino
        return _get_dino().embed_frames(frames_u8).numpy()
    x = torch.nn.functional.interpolate(frames_u8.float(), size=(16, 16), mode="area")
    return x.reshape(len(frames_u8), -1).numpy()


def tree_nodes(Z: np.ndarray, n: int):
    """Every node of the merge tree as {"frames": sorted list, "height": merge distance,
    "children": [a, b]}: the n leaves (height 0, no children) first, then the n-1 merges in merge
    order (node n + j is merge j; the last node is the whole clip)."""
    nodes = [{"frames": [i], "height": 0.0, "children": []} for i in range(n)]
    for a, b, h, _ in Z:
        a, b = int(a), int(b)
        nodes.append({"frames": sorted(nodes[a]["frames"] + nodes[b]["frames"]),
                      "height": float(h), "children": [a, b]})
    return nodes


def level_groups(nodes, n: int, k: int):
    """The k groups of the tree cut with k clusters: apply the first n - k merges."""
    alive = set(range(n))
    for node in range(n, 2 * n - k):
        alive -= set(nodes[node]["children"])
        alive.add(node)
    return sorted((nodes[c]["frames"] for c in alive), key=lambda g: g[0])


def fixed_windows(n: int, sizes):
    return [list(range(s * i, s * (i + 1))) for s in sizes if 1 < s < n for i in range(n // s)]


# --------------------------------------------------------------------------------------------
# Model-side measurements
# --------------------------------------------------------------------------------------------
def group_importance(ev, layout, groups, removal, cls, prior):
    """[I(G) for G in groups] and v(all), with v(S) = P(cls | only S kept, `removal`)."""
    n = len(layout["content"])
    keeps = np.ones((len(groups) + 1, n), bool)
    for r, g in enumerate(groups):
        keeps[r + 1, g] = False
    v = ev.value(layout, keeps, removal, cls, prior)
    return [float(v[0] - x) for x in v[1:]], float(v[0])


def deletion_curve(ev, layout, order, removal, cls, prior):
    """P(cls) after removing the first k frames of `order`, k = 0..n (k = n gives the prior)."""
    n = len(order)
    keeps = np.ones((n + 1, n), bool)
    for k in range(1, n + 1):
        keeps[k:, order[k - 1]] = False
    return [float(x) for x in ev.value(layout, keeps, removal, cls, prior)]


def swap_test(ev, n, cls):
    """For each neighbouring pair (i, i+1), replace frame i by frame i+1 ("fwd") and frame i+1
    by frame i ("bwd"); change in P(cls) and JS divergence vs the original clip."""
    layouts = [make_layout(range(n))]
    keys = []
    for i in range(n - 1):
        for d in ("fwd", "bwd"):
            c = list(range(n))
            if d == "fwd":
                c[i] = i + 1
            else:
                c[i + 1] = i
            layouts.append(make_layout(c))
            keys.append((i, d))
    p = softmax_np(ev.logits(layouts))
    out = []
    for (i, d), q in zip(keys, p[1:]):
        out.append({"i": i, "dir": d, "d_prob": float(q[cls] - p[0][cls]),
                    "js": float(js_div(p[0], q)), "same_pred": bool(q.argmax() == cls)})
    return out


# --------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    ap.add_argument("--dataset", default=None, help="default: the model's E1 dataset (MODEL_SPECS)")
    ap.add_argument("--limit", type=int, default=200, help="number of videos (class-stratified)")
    ap.add_argument("--seed", type=int, default=0, help="video sampling seed (same as E1)")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1], help="for stochastic methods")
    ap.add_argument("--clusterer", default="dinov2", choices=["dinov2", "pixel"])
    ap.add_argument("--window-sizes", nargs="+", type=int, default=[2, 4, 8])
    ap.add_argument("--deletion-levels", nargs="+", type=int, default=None,
                    help="tree levels (number of clusters) for cluster-order deletion curves; "
                         "default n/4 and n/2")
    ap.add_argument("--no-swap", action="store_true", help="skip the swap test")
    ap.add_argument("--include-wrong", action="store_true", help="keep misclassified videos")
    ap.add_argument("--prior", default="uniform")
    ap.add_argument("--n-perm", type=int, default=64)
    ap.add_argument("--exact-max", type=int, default=12)
    ap.add_argument("--pf-max-samples", type=int, default=256, help="as the R3D-50 E1 Play Fair run")
    ap.add_argument("--ig-steps", type=int, default=32)
    ap.add_argument("--ig-batch", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--out", default="results/e23")
    args = ap.parse_args()

    model = load_clip_model(args.model)
    spec = model.spec
    methods = args.methods or DEFAULT_METHODS[spec["design"]]
    removal = spec["removal"]  # for deletion curves and for methods not tied to a removal
    removals = sorted({removal} | {METHOD_REMOVAL[m] for m in methods if m in METHOD_REMOVAL})
    prior = load_prior(model, args.prior)
    n = spec["slots"]
    del_levels = args.deletion_levels or sorted({max(2, n // 4), max(2, n // 2)})

    out_path = Path(args.out) / f"{args.model}_e23.jsonl"
    skip_path = Path(args.out) / f"{args.model}_e23_skipped.jsonl"
    done = {r["video"] for r in read_jsonl(out_path)} | {r["video"] for r in read_jsonl(skip_path)}
    videos = get_videos(model, args.dataset, args.limit, args.seed)
    print(f"[e23] {args.model}: {len(videos)} videos ({len(done)} done), n={n}, methods={methods}, "
          f"removals={removals}, clusterer={args.clusterer}")

    for vi, (cls_name, path) in enumerate(videos):
        if path in done:
            continue
        t0 = time.time()
        try:
            frames, fidx = model.load_frames(path)
        except Exception as e:
            append_jsonl(skip_path, {"video": path, "reason": f"decode: {e}"})
            continue
        if len(frames) != n:
            append_jsonl(skip_path, {"video": path, "reason": f"got {len(frames)} frames"})
            continue

        bank = FrameBank(model, frames, "exact", stable_seed(path))
        ev = Evaluator(model, bank, args.batch_size, args.fp16)
        layout = make_layout(range(n))
        z = ev.logits([layout])[0]
        pred = int(z.argmax())
        gt = int(model.label2id[cls_name])
        if pred != gt and not args.include_wrong:
            append_jsonl(skip_path, {"video": path, "reason": "misclassified", "gt": gt, "pred": pred})
            continue

        # 2. clusters of the model's own frames
        from frame_clustering import hierarchy_from_embeddings
        emb = embed(frames, args.clusterer)
        Z = hierarchy_from_embeddings(emb)
        nodes = tree_nodes(Z, n)
        e = torch.nn.functional.normalize(torch.as_tensor(emb, dtype=torch.float64), dim=1)
        sim = (e @ e.T).numpy().round(4).tolist()  # torch, not numpy: tc_env's MKL crashes on matmul
        windows = fixed_windows(n, args.window_sizes)

        # 3. group importance (nodes, then windows), for every removal in use
        importance, v_all = {}, {}
        for r in removals:
            I, v_all[r] = group_importance(ev, layout, [nd["frames"] for nd in nodes] + windows,
                                           r, pred, prior)
            importance[r] = {"nodes": I[:len(nodes)], "windows": I[len(nodes):]}

        # 4. per-frame attributions
        attrs = []
        for method in methods:
            for seed in (args.seeds if method in STOCHASTIC else [0]):
                t1 = time.time()
                try:
                    # a different sampling seed per video: with one seed for all videos, every
                    # video gets the same permutations, and on models where v(S) depends mostly on
                    # |S| (mc3_18) the scores then repeat the same noise pattern across videos
                    s, n_ev = run_method(method, model, bank, ev, layout, pred, prior,
                                         seed=stable_seed(path, seed),
                                         n_perm=args.n_perm, exact_max=args.exact_max,
                                         ig_steps=args.ig_steps, ig_batch=args.ig_batch,
                                         pf_max_samples=args.pf_max_samples)
                except (ValueError, NotImplementedError) as err:
                    print(f"[skip] {method}: {err}")
                    continue
                attrs.append({"method": method, "seed": seed, "scores": [float(x) for x in s],
                              "n_evals": int(n_ev), "secs": round(time.time() - t1, 3)})

        # 5. deletion curves (spec removal)
        I_nodes = importance[removal]["nodes"]
        node_I = {tuple(nd["frames"]): I for nd, I in zip(nodes, I_nodes)}
        orders = []
        for a in attrs:
            if a["seed"] == 0:
                orders.append((f"{a['method']}:0", [int(i) for i in np.argsort(-np.asarray(a["scores"]), kind="stable")]))
        for k in del_levels:
            groups = sorted(level_groups(nodes, n, k), key=lambda g: -node_I[tuple(g)])
            orders.append((f"cluster:{k}", [i for g in groups for i in g]))
        rnd = list(range(n))
        random.Random(stable_seed(path, "deletion")).shuffle(rnd)
        orders.append(("random", rnd))
        deletion = [{"order": name, "perm": perm,
                     "values": deletion_curve(ev, layout, perm, removal, pred, prior)}
                    for name, perm in orders]

        # 6. swap test
        swaps = [] if args.no_swap else swap_test(ev, n, pred)

        p = softmax_np(z)
        append_jsonl(out_path, {
            "model": args.model, "video": path, "class": cls_name, "gt": gt, "pred": pred,
            "p_pred": float(p[pred]), "prior_pred": float(prior[pred]), "n": n,
            "frame_indices": [int(i) for i in fidx], "removal": removal, "clusterer": args.clusterer,
            "sim": sim, "nodes": nodes, "windows": windows,
            "importance": importance, "v_all": v_all,
            "attributions": attrs, "deletion": deletion, "swaps": swaps,
            "n_evals": int(ev.n_evals), "secs": round(time.time() - t0, 2)})
        name = path.replace("\\", "/").split("/")[-1]
        print(f"[{vi + 1}/{len(videos)}] {name}: pred={pred} p={p[pred]:.2f} "
              f"({time.time() - t0:.1f}s, {ev.n_evals} evals)")
    print(f"[e23] wrote {out_path}")


if __name__ == "__main__":
    main()
