"""
fr_run.py -- the frame-rate test: dilution with NATURAL near-duplicates. A natural version of E1:
instead of copying a frame, we sample one part of the video more densely.

Per video (one JSONL line in <out>/<model>_fr.jsonl; re-running resumes):
  1. Base clip: the model's own 16 frames (e1_common.frame_indices_for). Split into 4 time
     segments of 4 frames each. Misclassified videos are skipped; everything explains the
     predicted class.
  2. Group Shapley over the 4 segments (exact) on the base clip. The TARGET segment is the most
     important one (value > --tau); the CONTROL segment is the least important one (smallest
     |value|).
  3. For each density d in --densities: a target clip in which the target segment's 4 frames
     are replaced by 4d frames sampled evenly from the same stretch of the video (from its first
     to its last base frame), and a control clip in which the control segment is densified
     instead. The clip gets longer (16 + 4(d - 1) frames), so only models that accept other
     lengths and deleted frames are valid (insertion design: mc3_18, vjepa2).
  4. For every clip (base, target, control): the prediction statistics against the base clip;
     per-frame attributions (e1_common.run_method); group Shapley over the 4 segments (exact);
     and each segment's importance to the model, I(seg) = P(c*) minus P(c*) with the whole
     segment removed (the analogue of E1's I_t).
Frames shown in several clips are decoded once; if a densified segment repeats a frame (very
short videos), those slots are exact copies, and n_unique records how many distinct frames the
densified segment has.

Analysis: motivation/fr_metrics.py.

    python motivation/fr_run.py --model mc3_18 --limit 150 --fp16
    python motivation/fr_run.py --model vjepa2 --dataset ssv2 --limit 80 --fp16 --n-perm 32
"""
import argparse
import time
from pathlib import Path

import numpy as np

from e1_common import (MODEL_SPECS, STOCHASTIC, Evaluator, FrameBank, _decode, _n_frames,
                       append_jsonl, frame_indices_for, get_videos, group_shapley, load_clip_model,
                       load_prior, make_layout, pred_stats, read_jsonl, resolve_video_path,
                       run_method, stable_seed)

DEFAULT_METHODS = ["shapley_drop", "loo_drop", "occlusion", "ig", "gradcam"]


def segment_importance(ev, layout, seg_of_slot, n_seg, cls, prior):
    """I(seg) for each segment: v(all) - v(all but that segment), deletion."""
    n = len(layout["content"])
    keeps = np.ones((n_seg + 1, n), bool)
    for j in range(n_seg):
        keeps[j + 1] = np.asarray(seg_of_slot) != j
    v = ev.value(layout, keeps, "drop", cls, prior)
    return [float(v[0] - x) for x in v[1:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--densities", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--n-seg", type=int, default=4)
    ap.add_argument("--tau", type=float, default=0.01, help="min group value of the target segment")
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--n-perm", type=int, default=64)
    ap.add_argument("--ig-steps", type=int, default=32)
    ap.add_argument("--prior", default="uniform")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--out", default="results/fr")
    args = ap.parse_args()

    model = load_clip_model(args.model)
    assert model.spec["design"] == "insert", "the frame-rate test needs a model that accepts other lengths (insertion design)"
    prior = load_prior(model, args.prior)
    S = args.n_seg
    out_path = Path(args.out) / f"{args.model}_fr.jsonl"
    skip_path = Path(args.out) / f"{args.model}_fr_skipped.jsonl"
    done = {r["video"] for r in read_jsonl(out_path)} | {r["video"] for r in read_jsonl(skip_path)}
    videos = get_videos(model, args.dataset, args.limit, args.seed)
    print(f"[fr] {args.model}: {len(videos)} videos ({len(done)} done), densities={args.densities}, methods={args.methods}")

    for vi, (cls_name, path) in enumerate(videos):
        if path in done:
            continue
        t0 = time.time()
        try:
            n_total = _n_frames(resolve_video_path(path))
        except Exception as e:
            append_jsonl(skip_path, {"video": path, "reason": f"decode: {e}"})
            continue
        base_idx = frame_indices_for(args.model, n_total)
        n = len(base_idx)
        per = n // S
        seg_base = [j for j in range(S) for _ in range(per)]

        def clip_indices(seg, d):
            """frame indices of the clip with segment `seg` densified d times (d = 1: base)."""
            out, segs = [], []
            for j in range(S):
                part = base_idx[j * per:(j + 1) * per]
                if j == seg and d > 1:
                    part = np.linspace(part[0], part[-1], per * d).round().astype(int).tolist()
                out += part
                segs += [j] * len(part)
            return out, segs

        plan = [("base", None, 1)]
        # decode every frame any clip might need (target/control chosen after the base clip)
        all_idx = sorted(set(base_idx) | {i for s in range(S) for d in args.densities for i in clip_indices(s, d)[0]})
        try:
            frames_all = _decode(resolve_video_path(path), all_idx)
        except Exception as e:
            append_jsonl(skip_path, {"video": path, "reason": f"decode: {e}"})
            continue
        pos = {fi: k for k, fi in enumerate(all_idx)}
        bank = FrameBank(model, frames_all, "exact", stable_seed(path))
        ev = Evaluator(model, bank, args.batch_size, args.fp16)

        def layout_of(idx):
            return make_layout([pos[i] for i in idx])

        base_lay = layout_of(base_idx)
        z_base = ev.logits([base_lay])[0]
        pred = int(z_base.argmax())
        gt = int(model.label2id[cls_name])
        if pred != gt:
            append_jsonl(skip_path, {"video": path, "reason": "misclassified", "gt": gt, "pred": pred})
            continue
        seg_groups = lambda segs: [[i for i, s in enumerate(segs) if s == j] for j in range(S)]  # noqa: E731
        phi_base = group_shapley(ev, base_lay, seg_groups(seg_base), pred, prior, "drop")
        target = int(np.argmax(phi_base))
        if phi_base[target] <= args.tau:
            append_jsonl(skip_path, {"video": path, "reason": f"no segment with value > {args.tau}",
                                     "phi": [float(x) for x in phi_base]})
            continue
        control = int(min((j for j in range(S) if j != target), key=lambda j: abs(phi_base[j])))
        plan += [(kind, seg, d) for d in args.densities for kind, seg in (("target", target), ("control", control))]

        clips = []
        for kind, seg, d in plan:
            idx, segs = clip_indices(seg, d) if kind != "base" else (base_idx, seg_base)
            lay = layout_of(idx)
            z = ev.logits([lay])[0]
            attrs = []
            for method in args.methods:
                for sd in (args.seeds if method in STOCHASTIC else [0]):
                    t1 = time.time()
                    try:
                        s, n_ev = run_method(method, model, bank, ev, lay, pred, prior,
                                             seed=stable_seed(path, kind, d, sd), n_perm=args.n_perm,
                                             ig_steps=args.ig_steps)
                    except (ValueError, NotImplementedError) as err:
                        print(f"[skip] {method}: {err}")
                        continue
                    attrs.append({"method": method, "seed": sd, "scores": [float(x) for x in s],
                                  "n_evals": int(n_ev), "secs": round(time.time() - t1, 2)})
            seg_part = [i for i, s in zip(idx, segs) if s == seg] if seg is not None else []
            clips.append({
                "kind": kind, "seg": seg, "d": d, "frame_indices": [int(i) for i in idx], "segs": segs,
                "n_unique": len(set(seg_part)) if seg_part else None,
                "stats": pred_stats(z_base, z, pred),
                "group_values": [float(x) for x in (phi_base if kind == "base" else
                                                    group_shapley(ev, lay, seg_groups(segs), pred, prior, "drop"))],
                "I_seg": segment_importance(ev, lay, segs, S, pred, prior),
                "attributions": attrs})
        append_jsonl(out_path, {"model": args.model, "video": path, "class": cls_name, "pred": pred,
                                "n_total": n_total, "n_seg": S, "target": target, "control": control,
                                "clips": clips, "n_evals": int(ev.n_evals), "secs": round(time.time() - t0, 1)})
        name = path.replace("\\", "/").split("/")[-1]
        print(f"[{vi + 1}/{len(videos)}] {name}: target seg {target}, control seg {control} "
              f"({time.time() - t0:.1f}s, {ev.n_evals} evals)")
    print(f"[fr] wrote {out_path}")


if __name__ == "__main__":
    main()
