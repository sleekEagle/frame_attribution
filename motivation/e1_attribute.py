"""
e1_attribute.py -- E1 step 5: run frame-attribution methods on every reference / target / control
clip produced by e1_build_conditions.py. Writes one JSONL line per (video, condition, method,
seed) with one signed score per SLOT for the model's predicted class. Re-running resumes.

    python motivation/e1_attribute.py --model r3d --methods loo_drop occlusion ig gradcam
    python motivation/e1_attribute.py --model r3d --methods shapley_drop playfair --seeds 0 1
    python motivation/e1_attribute.py --model r3d --max-m 4          # after reading the gate report
    python motivation/e1_attribute.py --model toy                   # smoke test

Methods (see e1_common.run_method):
  shapley_freeze   Shapley over slots, removed slots filled by late freeze (your WACV fill)
  shapley_drop     Shapley over slots, removed slots dropped (variable-length models only)
  playfair         Play Fair ESVs with the official play-fair attributor, drop path
                   (variable-length models; exact for <= --exact-max slots, else Play Fair's
                   constructive sampler with --pf-max-samples subsets per scale)
  loo_drop / loo_freeze   leave-one-slot-out
  occlusion        whole-frame occlusion (all channels of one slot set to 0 in normalised space)
  ig               Integrated Gradients on the class logit, zero baseline, per-slot SUM
  gradcam          Grad-CAM (r3d: --gradcam-layer, ViTs: last encoder block, TRN: frame features)
Stochastic methods (shapley_*, playfair) are run for every --seeds value; deterministic ones once.

Only conditions that pass the sensitivity gate are attributed: prediction unchanged
(stats.same_pred) and m <= --max-m. References are always attributed. The gate is applied per
PAIR: a target condition and its control (same dup type, target and m) are kept only if BOTH keep
the reference prediction, so every attributed target has its matched control (--unpaired-gate
filters each condition on its own instead).
"""
import argparse
import time
from pathlib import Path

from e1_common import (MODEL_SPECS, STOCHASTIC, Evaluator, FrameBank, append_jsonl,
                       load_clip_model, load_prior, read_jsonl, resolve_video_path, run_method)

DEFAULT_METHODS = {
    "insert": ["shapley_drop", "playfair", "loo_drop", "occlusion", "ig", "gradcam"],
    "realloc": ["shapley_freeze", "loo_freeze", "occlusion", "ig", "gradcam"],
    # deletion-based like insert; Play Fair is opt-in (--methods playfair) because of its cost
    "replace": ["shapley_drop", "loo_drop", "occlusion", "ig", "gradcam"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    ap.add_argument("--conditions", default=None, help="default: <out>/<model>_conditions.jsonl")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--dup-types", nargs="+", default=None, help="subset of dup types to run")
    ap.add_argument("--kinds", nargs="+", default=None,
                    help="subset of reference / target / control_spread / control_recipient")
    ap.add_argument("--max-m", type=int, default=None, help="skip conditions with m above this")
    ap.add_argument("--unpaired-gate", action="store_true",
                    help="gate each condition on its own instead of target/control pairs")
    ap.add_argument("--limit", type=int, default=None, help="only the first N videos")
    ap.add_argument("--n-perm", type=int, default=64, help="permutations for sampled Shapley")
    ap.add_argument("--exact-max", type=int, default=12, help="exact Shapley/ESV up to this many slots")
    ap.add_argument("--pf-max-samples", type=int, default=1024)
    ap.add_argument("--ig-steps", type=int, default=32)
    ap.add_argument("--ig-batch", type=int, default=2)
    ap.add_argument("--gradcam-layer", default=None, help="r3d: layer1..layer4; ViTs: block index")
    ap.add_argument("--prior", default=None, help="default: the prior used when building conditions")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--out", default="results/e1")
    args = ap.parse_args()

    model = load_clip_model(args.model)
    if args.gradcam_layer:  # "layer2" for r3d, or an encoder-block index such as -2 for ViTs
        gl = args.gradcam_layer
        model.spec["gradcam_layer"] = int(gl) if gl.lstrip("-").isdigit() else gl
    methods = args.methods or DEFAULT_METHODS[model.spec["design"]]
    cond_path = Path(args.conditions or Path(args.out) / f"{args.model}_conditions.jsonl")
    out_path = Path(args.out) / f"{args.model}_attributions.jsonl"
    records = read_jsonl(cond_path)
    if args.limit:
        records = records[:args.limit]
    done = {(r["video"], r["cond_id"], r["method"], r["seed"]) for r in read_jsonl(out_path)}
    print(f"[e1] {args.model}: {len(records)} videos, methods={methods}, seeds={args.seeds}")

    for vi, rec in enumerate(records):
        path = rec["video"]
        prior = load_prior(model, args.prior or rec.get("prior"))
        frames, _ = model.load_frames(resolve_video_path(path))  # path stays the record key
        cframes = frames[0::2] if rec["design"] == "realloc" else frames
        cls = rec["pred"]
        banks, evs = {}, {}
        t_vid = time.time()
        pair_key = lambda c: (c["dup"], c["target"], c["m"])  # noqa: E731
        failed = {pair_key(c) for c in rec["conditions"]
                  if c["kind"] != "reference" and not c["stats"]["same_pred"]}
        for cond in rec["conditions"]:
            if args.dup_types and cond["dup"] not in args.dup_types:
                continue
            if args.kinds and cond["kind"] not in args.kinds:
                continue
            if cond["kind"] != "reference":
                if not cond["stats"]["same_pred"]:
                    continue
                if not args.unpaired_gate and pair_key(cond) in failed:
                    continue  # its target/control partner changed the prediction
                if args.max_m and cond["m"] > args.max_m:
                    continue
            dup = cond["dup"]
            if dup not in banks:
                banks[dup] = FrameBank(model, cframes, dup, rec["bank_seed"])
                evs[dup] = Evaluator(model, banks[dup], args.batch_size, args.fp16)
            for method in methods:
                for seed in (args.seeds if method in STOCHASTIC else [0]):
                    key = (path, cond["id"], method, seed)
                    if key in done:
                        continue
                    t0 = time.time()
                    try:
                        scores, n_ev = run_method(
                            method, model, banks[dup], evs[dup], cond["layout"], cls, prior,
                            seed=seed, n_perm=args.n_perm, exact_max=args.exact_max,
                            ig_steps=args.ig_steps, ig_batch=args.ig_batch,
                            pf_max_samples=args.pf_max_samples)
                    except (ValueError, NotImplementedError) as e:
                        print(f"[skip] {method} on {cond['id']}: {e}")
                        done.add(key)
                        continue
                    append_jsonl(out_path, {
                        "model": args.model, "video": path, "cond_id": cond["id"],
                        "method": method, "seed": seed, "cls": cls,
                        "scores": [float(s) for s in scores],
                        "content": cond["layout"]["content"], "copy": cond["layout"]["copy"],
                        "n_evals": int(n_ev), "secs": round(time.time() - t0, 3)})
                    done.add(key)
        name = path.replace("\\", "/").split("/")[-1]  # Windows paths too, on Linux
        print(f"[{vi + 1}/{len(records)}] {name} done ({time.time() - t_vid:.1f}s)")
    print(f"[e1] wrote {out_path}")


if __name__ == "__main__":
    main()
