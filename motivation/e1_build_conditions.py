"""
e1_build_conditions.py -- E1 steps 2-4: reference clip, content importance, target selection,
duplication conditions + controls, and the sensitivity gate (model outputs only, no attributions).

    python motivation/e1_build_conditions.py --model r3d --limit 20            # pilot
    python motivation/e1_build_conditions.py --model r3d --limit 300
    python motivation/e1_build_conditions.py --model vjepa2 --limit 200 --fp16
    python motivation/e1_build_conditions.py --model videomae --limit 300
    python motivation/e1_build_conditions.py --model toy --limit 5             # smoke test, no data needed

Per video (one JSONL line in <out>/<model>_conditions.jsonl; re-running resumes):
  1. Load the model's standard frames. Insertion design (vjepa2) and replacement design
     (videomae): the 16 frames are the K=16 contents, reference = original clip (m=1).
     Reallocation design (r3d, trn): every other frame -> K = slots/2 contents, reference = each
     content x2 (m=2).
  2. Predicted class c* on the reference (videos the model gets wrong are skipped unless
     --include-wrong). Everything below explains c*.
  3. Content importance I(c) = P(c*|ref) - P(c*|ref with ALL copies of c removed); removal =
     drop (insertion design) or late freeze = mean(past-fill, future-fill) (reallocation design).
  4. Targets are chosen from I(c) -- never from an attribution method, so every method is tested
     on the same frames: one HIGH target (random among the top-2 contents with I > tau) and one
     MID target (random among ranks 3..K/2 with I > tau). The content with the smallest |I| (not a
     target) is the RECIPIENT used by the reallocation control.
  5. For every duplicate type (exact / noise / shift) and every m: the target condition and its
     control (spread control for insertion, recipient control for reallocation and replacement), each with
     prediction statistics vs. that duplicate type's reference (the sensitivity gate) and, unless
     --no-cond-importance, content importance of every content in that condition, I^(m)(c),
     which e1_metrics.py uses to validate rank inversions.
  6. Reallocation design only: the original all-distinct clip vs the x2 reference (gate step 1).
"""
import argparse
import math
import random
import time
from pathlib import Path

import numpy as np

from e1_common import (MODEL_SPECS, Evaluator, FrameBank, append_jsonl, content_importance,
                       get_videos, insertion_layout, load_clip_model, load_prior, make_layout,
                       pred_stats, read_jsonl, realloc_layouts, reference_layout,
                       replacement_layouts, spread_control_layout, stable_seed)


def pick_targets(I: dict, K: int, tau: float, rng: random.Random):
    ranked = sorted(I, key=lambda c: -I[c])
    eligible = [c for c in ranked if I[c] > tau]
    targets = []
    if eligible:
        targets.append({"content": rng.choice(eligible[:2]), "tier": "high"})
        mid_pool = [c for c in eligible[2:max(3, math.ceil(K / 2))] if c != targets[0]["content"]]
        if mid_pool:
            targets.append({"content": rng.choice(mid_pool), "tier": "mid"})
    chosen = {t["content"] for t in targets}
    rest = [c for c in I if c not in chosen]
    recipient = min(rest, key=lambda c: abs(I[c])) if rest else None
    return targets, recipient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="r3d", choices=sorted(MODEL_SPECS))
    ap.add_argument("--dataset", default=None,
                    help="override the model's default dataset (MODEL_SPECS; r3d -> ucf101_test)")
    ap.add_argument("--limit", type=int, default=None, help="number of videos (class-stratified)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dup-types", nargs="+", default=["exact", "noise"],
                    choices=["exact", "noise", "shift"])
    ap.add_argument("--ms", nargs="+", type=int, default=None, help="override the model's m sweep")
    ap.add_argument("--tau", type=float, default=0.01, help="min content importance for a target")
    ap.add_argument("--prior", default="uniform", help="'uniform' or a .npy of class frequencies")
    ap.add_argument("--include-wrong", action="store_true", help="keep misclassified videos")
    ap.add_argument("--no-cond-importance", action="store_true",
                    help="skip I^(m)(c) per condition (faster; disables validated inversion)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--out", default="results/e1")
    args = ap.parse_args()

    model = load_clip_model(args.model)
    spec = model.spec
    design, tub, slots = spec["design"], spec["tubelet"], spec["slots"]
    removal = spec["removal"]
    m_ref = 2 if design == "realloc" else 1
    ms = [m for m in (args.ms or spec["ms"]) if m != m_ref]
    prior = load_prior(model, args.prior)

    out_path = Path(args.out) / f"{args.model}_conditions.jsonl"
    skip_path = Path(args.out) / f"{args.model}_skipped.jsonl"
    done = {r["video"] for r in read_jsonl(out_path)} | {r["video"] for r in read_jsonl(skip_path)}
    videos = get_videos(model, args.dataset, args.limit, args.seed)
    print(f"[e1] {args.model}: design={design} slots={slots} tubelet={tub} m_ref={m_ref} ms={ms} "
          f"removal={removal} -> {len(videos)} videos ({len(done)} already done)")

    for vi, (cls_name, path) in enumerate(videos):
        if path in done:
            continue
        t0 = time.time()
        vseed = stable_seed(path)
        rng = random.Random(stable_seed(path, args.seed))
        try:
            frames, fidx = model.load_frames(path)
        except Exception as e:  # unreadable video
            append_jsonl(skip_path, {"video": path, "reason": f"decode: {e}"})
            continue
        if len(frames) != slots:
            append_jsonl(skip_path, {"video": path, "reason": f"got {len(frames)} frames"})
            continue

        if design == "realloc":
            cframes, cidx, K = frames[0::2], fidx[0::2], slots // 2
        else:  # insert / replace: every sampled frame is a content, reference = original clip
            cframes, cidx, K = frames, fidx, slots
        ref = reference_layout(design, K)

        bank = FrameBank(model, cframes, "exact", vseed)
        ev = Evaluator(model, bank, args.batch_size, args.fp16)
        z_ref = ev.logits([ref])[0]
        pred = int(z_ref.argmax())
        gt = int(model.label2id[cls_name])
        if pred != gt and not args.include_wrong:
            append_jsonl(skip_path, {"video": path, "reason": "misclassified", "gt": gt, "pred": pred})
            continue

        orig_vs_ref = None
        if design == "realloc":  # gate step 1: all-distinct original vs x2 reference
            full_bank = FrameBank(model, frames, "exact", vseed)
            z_orig = Evaluator(model, full_bank, args.batch_size, args.fp16).logits(
                [make_layout(range(slots))])[0]
            orig_vs_ref = pred_stats(z_orig, z_ref, int(z_orig.argmax()))
            orig_vs_ref["orig_pred"] = int(z_orig.argmax())

        I_ref = content_importance(ev, ref, removal, pred, prior)
        targets, recipient = pick_targets(I_ref, K, args.tau, rng)
        if not targets:
            append_jsonl(skip_path, {"video": path, "reason": f"no content with I > {args.tau}",
                                     "I_ref": I_ref})
            continue

        conditions = []
        for dup in args.dup_types:
            bank_d = bank if dup == "exact" else FrameBank(model, cframes, dup, vseed)
            ev_d = Evaluator(model, bank_d, args.batch_size, args.fp16)
            z_ref_d = ev_d.logits([ref])[0]

            def add(kind, layout, **extra):
                z = ev_d.logits([layout])[0]
                cond = {"id": None, "dup": dup, "kind": kind, "layout": layout,
                        "stats": pred_stats(z_ref_d, z, pred), **extra}
                if not args.no_cond_importance:
                    cond["I"] = content_importance(ev_d, layout, removal, pred, prior)
                tag = f"t{extra.get('target')}|m{extra.get('m')}" if kind != "reference" else ""
                cond["id"] = f"{dup}|{kind}" + (f"|{tag}" if tag else "")
                conditions.append(cond)

            add("reference", ref, m=m_ref, target=None, tier=None, aligned=True,
                vs_exact_ref=pred_stats(z_ref, z_ref_d, pred))
            for t in targets:
                tc, tier = t["content"], t["tier"]
                for m in ms:
                    if design == "insert":
                        if (m - 1) % tub:
                            continue  # keeps tubelets aligned; see MODEL_SPECS ms
                        lay, aligned = insertion_layout(K, tc, m, tub)
                        crng = random.Random(stable_seed(path, tc, m, "spread"))
                        ctrl, dup_c = spread_control_layout(K, tc, m, tub, crng)
                        add("target", lay, target=tc, tier=tier, m=m, aligned=aligned)
                        add("control_spread", ctrl, target=tc, tier=tier, m=m, aligned=aligned,
                            duplicated=dup_c)
                    else:
                        if recipient is None:
                            continue
                        lrng = random.Random(stable_seed(path, tc, m, "losers"))
                        layouts_fn = realloc_layouts if design == "realloc" else replacement_layouts
                        lay, ctrl, losers, aligned = layouts_fn(K, tc, m, recipient, tub, lrng)
                        if lay is None:
                            continue  # no valid set of slots for the target's copies
                        add("target", lay, target=tc, tier=tier, m=m, aligned=aligned, losers=losers)
                        add("control_recipient", ctrl, target=tc, tier=tier, m=m, aligned=aligned,
                            losers=losers, recipient=recipient)

        append_jsonl(out_path, {
            "model": args.model, "video": path, "class": cls_name, "gt": gt, "pred": pred,
            "design": design, "slots": slots, "K": K, "m_ref": m_ref, "removal": removal,
            "tubelet": tub, "bank_seed": vseed, "prior": args.prior,
            "content_frame_idx": [int(i) for i in cidx], "I_ref": I_ref,
            "targets": targets, "recipient": recipient, "orig_vs_ref": orig_vs_ref,
            "conditions": conditions, "n_evals": ev.n_evals, "secs": round(time.time() - t0, 2)})
        n_bad = sum(not c["stats"]["same_pred"] for c in conditions)
        print(f"[{vi + 1}/{len(videos)}] {Path(path).name}: pred={pred} targets="
              f"{[t['content'] for t in targets]} conditions={len(conditions)} "
              f"pred_changed={n_bad} ({time.time() - t0:.1f}s)")

    print(f"[e1] wrote {out_path}")


if __name__ == "__main__":
    main()
