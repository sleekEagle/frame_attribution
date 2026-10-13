"""
e1_group.py -- group-level Shapley on the E1 clips: does attributing GROUPS instead of frames
remove dilution? Uses the clips of e1_build_conditions.py and the same conditions that
e1_attribute.py attributed (those that passed the gate).

For every attributed clip (reference, target, control), compute e1_common.group_shapley with:
  oracle     one group per content: all slots that show the same content (we built the clips, so
             we know which slots hold copies). The target's m copies form one group.
  dino       the clip's frames clustered with DINOv2 (frame_clustering; neighbours only), the tree
             cut into the same number of groups as the oracle ("dino") and into half as many
             ("dino_half", coarser groups that may merge the target with similar neighbours).
             For each DINOv2 grouping we record whether the target's copies form exactly one
             group of their own ("recovered").
Same class (the reference prediction), removal and prior as E1. Exact Shapley for up to
--exact-max groups, otherwise --n-perm sampled orderings (seeded per video and clip).

Writes <out>/<model>_group_attributions.jsonl, one line per (video, condition, grouping):
    {"video", "cond_id", "grouping", "groups": [[slots], ...], "contents": [[content ids], ...],
     "values": [...], "recovered": bool | null, "n_evals"}
Re-running resumes. Analysis: e1_group_metrics.py.

    python motivation/e1_group.py --model r3d_18 --fp16
    python motivation/e1_group.py --model mc3_18 --fp16 --limit 60
    python motivation/e1_group.py --model vjepa2 --fp16 --dup-types exact --n-perm 32
"""
import argparse
import time
from pathlib import Path

import numpy as np

from e1_common import (MODEL_SPECS, Evaluator, FrameBank, append_jsonl, group_shapley,
                       load_clip_model, load_prior, perturb_u8, read_jsonl, resolve_video_path,
                       stable_seed)
from e23_run import embed, level_groups, tree_nodes
from frame_clustering import hierarchy_from_embeddings


def oracle_groups(layout):
    """One group per content, ordered by first slot."""
    by_c = {}
    for slot, c in enumerate(layout["content"]):
        by_c.setdefault(c, []).append(slot)
    return sorted(by_c.values(), key=lambda g: g[0])


def slot_frames_u8(layout, cframes, dup, bank_seed):
    """uint8 frame shown in each slot (copies of noise/shift type are perturbed as in FrameBank)."""
    import torch
    out = []
    for c, k in zip(layout["content"], layout["copy"]):
        f = cframes[c]
        if k > 0 and dup != "exact":
            f = perturb_u8(f, dup, stable_seed(bank_seed, c, k))
        out.append(f)
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    ap.add_argument("--groupings", nargs="+", default=["oracle", "dino", "dino_half"])
    ap.add_argument("--dup-types", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N videos")
    ap.add_argument("--clusterer", default="dinov2", choices=["dinov2", "pixel"])
    ap.add_argument("--exact-max", type=int, default=12)
    ap.add_argument("--n-perm", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--out", default="results/e1")
    args = ap.parse_args()

    out_dir = Path(args.out)
    recs = read_jsonl(out_dir / f"{args.model}_conditions.jsonl")[:args.limit]
    attributed = {(a["video"], a["cond_id"]) for a in read_jsonl(out_dir / f"{args.model}_attributions.jsonl")}
    out_path = out_dir / f"{args.model}_group_attributions.jsonl"
    done = {(r["video"], r["cond_id"], r["grouping"]) for r in read_jsonl(out_path)}
    model = load_clip_model(args.model)
    print(f"[e1 group] {args.model}: {len(recs)} videos, groupings={args.groupings}")

    for vi, rec in enumerate(recs):
        path = rec["video"]
        conds = [c for c in rec["conditions"] if (path, c["id"]) in attributed
                 and (not args.dup_types or c["dup"] in args.dup_types)]
        todo = [(c, g) for c in conds for g in args.groupings if (path, c["id"], g) not in done]
        if not todo:
            continue
        t0 = time.time()
        prior = load_prior(model, rec.get("prior"))
        frames, _ = model.load_frames(resolve_video_path(path))
        cframes = frames[0::2] if rec["design"] == "realloc" else frames
        banks, evs, emb_cache, val_cache = {}, {}, {}, {}
        for cond, grouping in todo:
            dup, layout = cond["dup"], cond["layout"]
            if dup not in banks:
                banks[dup] = FrameBank(model, cframes, dup, rec["bank_seed"])
                evs[dup] = Evaluator(model, banks[dup], args.batch_size, args.fp16)
            ev = evs[dup]
            orc = oracle_groups(layout)
            target = cond.get("target")
            recovered = None
            if grouping == "oracle":
                groups = orc
            else:
                key = (dup, tuple(layout["content"]), tuple(layout["copy"]))
                if key not in emb_cache:
                    emb_cache[key] = embed(slot_frames_u8(layout, cframes, dup, rec["bank_seed"]), args.clusterer)
                n = len(layout["content"])
                nodes = tree_nodes(hierarchy_from_embeddings(emb_cache[key]), n)
                k = len(orc) if grouping == "dino" else max(2, len(orc) // 2)
                groups = level_groups(nodes, n, k)
                if target is not None and target >= 0:
                    t_slots = sorted(i for i, c in enumerate(layout["content"]) if c == target)
                    recovered = any(sorted(g) == t_slots for g in groups)
            n0 = ev.n_evals
            vkey = (dup, tuple(layout["content"]), tuple(layout["copy"]), tuple(map(tuple, groups)))
            if vkey not in val_cache:  # e.g. DINOv2 found exactly the oracle groups: same values
                val_cache[vkey] = group_shapley(ev, layout, groups, rec["pred"], prior, rec["removal"],
                                                exact_max=args.exact_max, n_perm=args.n_perm,
                                                seed=stable_seed(path, cond["id"]))
            vals = val_cache[vkey]
            append_jsonl(out_path, {
                "model": args.model, "video": path, "cond_id": cond["id"], "grouping": grouping,
                "groups": groups, "contents": [[layout["content"][i] for i in g] for g in groups],
                "values": [float(v) for v in vals], "recovered": recovered,
                "n_evals": ev.n_evals - n0})
        name = path.replace("\\", "/").split("/")[-1]
        print(f"[{vi + 1}/{len(recs)}] {name}: {len(todo)} group attributions ({time.time() - t0:.1f}s)")
    print(f"[e1 group] wrote {out_path}")


if __name__ == "__main__":
    main()
