"""
calibrate_tau.py -- choose the merge-height threshold tau of the "dino_thr" grouping, per model,
on videos that are NOT used to test the grouping (not the E1 or frame-rate test videos).

tau is set so that the median number of groups at the model's base clip length n is n / 2
(8 for 16 frames, 4 for TRN's 8): tau = median over videos of the (n - n/2)-th smallest merge
height of the video's DINOv2 tree (contiguous Ward, frame_clustering.hierarchy_from_embeddings).
With tau fixed, a run of near-identical frames merges at a small height and becomes one group,
whatever its length.

Sources of trees:
  * models with an E2/E3 run: the stored trees in <e23>/<model>_e23.jsonl (no GPU needed);
  * other models (TRN): --from-videos N decodes N videos of the model's dataset that are not in
    <e1>/<model>_conditions.jsonl, at the model's frame sampling, and embeds them with DINOv2.

--linkage average calibrates the average-linkage tree instead (grouping "dinoavg_thr"); from the
E2/E3 runs it is rebuilt from the stored DINOv2 cosine similarities, D = sqrt(2 - 2 sim), and
stored under the key "<model>@average".

Writes/updates <e1>/dino_tau.json: {model: {"tau", "n", "target_k", "n_videos", "source",
"k_quantiles": [10, 25, 50, 75, 90 %] of the number of groups at tau}}.

    python motivation/calibrate_tau.py --models r3d r3d_18 mc3_18 vjepa2
    python motivation/calibrate_tau.py --models trn_official --from-videos 150
"""
import argparse
import json
from pathlib import Path

import numpy as np

from e1_common import _decode, _n_frames, frame_indices_for, read_jsonl, resolve_video_path
from e23_run import threshold_k, tree_nodes


def heights_from_e23(path, linkage):
    from frame_clustering import hierarchy_from_distances
    out = []
    for r in read_jsonl(path):
        n = r["n"]
        if linkage == "ward":
            hs = [node["height"] for node in r["nodes"][n:]]
        else:
            D = np.sqrt(np.clip(2 - 2 * np.asarray(r["sim"]), 0, None))
            np.fill_diagonal(D, 0)
            hs = list(hierarchy_from_distances(D, linkage)[:, 2])
        out.append((n, sorted(hs)))
    return out


def heights_from_videos(model_name, n_videos, e1_dir, seed, linkage):
    from e1_common import get_videos, load_clip_model
    from e23_run import embed
    from frame_clustering import hierarchy_from_embeddings
    model = load_clip_model(model_name)
    used = {r["video"] for r in read_jsonl(Path(e1_dir) / f"{model_name}_conditions.jsonl")}
    used |= {Path(p.replace("\\", "/")).name for p in used}
    out = []
    for _, path in get_videos(model, None, None, seed):
        if path in used or Path(path.replace("\\", "/")).name in used:
            continue
        try:
            p = resolve_video_path(path)
            frames = _decode(p, frame_indices_for(model_name, _n_frames(p)))
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {path}: {e}")
            continue
        n = len(frames)
        nodes = tree_nodes(hierarchy_from_embeddings(embed(frames, "dinov2"), linkage), n)
        out.append((n, sorted(node["height"] for node in nodes[n:])))
        if len(out) >= n_videos:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--e23", default="results/e23")
    ap.add_argument("--e1", default="results/e1")
    ap.add_argument("--from-videos", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--linkage", default="ward", choices=["ward", "average"])
    args = ap.parse_args()
    out_path = Path(args.e1) / "dino_tau.json"
    taus = json.loads(out_path.read_text()) if out_path.exists() else {}
    for m in args.models:
        if args.from_videos:
            hs, source = heights_from_videos(m, args.from_videos, args.e1, args.seed, args.linkage), "videos not in E1"
        else:
            hs, source = heights_from_e23(Path(args.e23) / f"{m}_e23.jsonl", args.linkage), "E2/E3 trees"
        key = m if args.linkage == "ward" else f"{m}@average"
        n = int(np.median([h[0] for h in hs]))
        hs = [h for h in hs if h[0] == n]
        target_k = n // 2
        tau = float(np.median([h[1][n - target_k - 1] for h in hs]))
        ks = [threshold_k([None] * n + [{"height": x} for x in h], n, tau) for _, h in hs]
        taus[key] = {"tau": tau, "linkage": args.linkage, "n": n, "target_k": target_k, "n_videos": len(hs), "source": source,
                   "k_quantiles": [float(q) for q in np.percentile(ks, [10, 25, 50, 75, 90])]}
        print(f"{key}: tau = {tau:.4f} from {len(hs)} videos ({source}); groups at n = {n}: "
              f"10/25/50/75/90% = {taus[key]['k_quantiles']}")
    out_path.write_text(json.dumps(taus, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
