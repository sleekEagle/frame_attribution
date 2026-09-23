#!/usr/bin/env python3
"""
test_playfair_models.py -- run Play Fair (via run_playfair.py) with the two models and two
datasets used in the paper, and print a pass/fail summary.

    R3D-50 (models/r3d/ckpt/save_200.pth)   on UCF101   D:\\datasets\\UCF-101\\<Class>\\v_*.avi
    V-JEPA 2 ViT-L (HF ssv2 head)           on SSv2     D:\\datasets\\SSV2\\s2s_test\\<label>\\*.webm

Run from the frame_attribution folder (needs a GPU for the V-JEPA2 part):

    python test_playfair_models.py                       # both, 3 videos each, 8 frames, exact
    python test_playfair_models.py --dataset ucf101 --n 5
    python test_playfair_models.py --frames 16 --approximate --max-samples 128

What is checked per video
  * the full-clip prediction from Play Fair's f(X) equals the class the model predicts
  * efficiency axiom: sum(ESV) == f(full) - prior   (exact mode: |gap| < 1e-4)
  * frame-sensitivity probe: model output depends on every frame at every clip length
  * (UCF101 only) whether the prediction matches the ground-truth class from the file name
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(video, adapter, frames, approximate, max_samples, out_dir, extra):
    is_r3d = adapter.startswith("playfair_adapters:build_r3d")
    cmd = [sys.executable, str(HERE / "run_playfair.py"), str(video), "--adapter", adapter,
           "--out-dir", str(out_dir), "--batch-size", "16"]
    if is_r3d:  # R3D was trained on 16 consecutive frames -> use the first `frames` consecutive ones
        cmd += ["--frame-indices"] + [str(i) for i in range(frames)]
    else:       # V-JEPA2 / SSv2: uniformly spread frames, fp16 on GPU
        cmd += ["--num-frames", str(frames), "--fp16"]
    if approximate:
        cmd += ["--approximate", "--max-samples-per-scale", str(max_samples)]
    p = subprocess.run(cmd + extra, capture_output=True, text=True)
    if p.returncode != 0:
        return None, p.stdout + p.stderr
    return json.loads((Path(out_dir) / "esv.json").read_text()), p.stdout


def pick_ucf(root, n):
    """First video of group 01 (a UCF101 split-1 test group) from n spread-out classes."""
    classes = sorted(p for p in root.iterdir() if p.is_dir())
    step = max(1, len(classes) // n)
    vids = []
    for c in classes[::step][:n]:
        v = sorted(c.glob("v_*_g01_c01.avi")) or sorted(c.glob("*.avi"))
        if v:
            vids.append((v[0], c.name))
    return vids


def pick_ssv2(root, n):
    lines = (HERE / "dataloaders" / "ssv2_paths.txt").read_text().splitlines()
    step = max(1, len(lines) // n)
    out = []
    for ln in lines[::step][:n]:
        rel = Path(ln.strip())
        out.append((root / rel, rel.parent.as_posix()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ucf101", "ssv2", "both"], default="both")
    ap.add_argument("--n", type=int, default=3, help="videos per dataset")
    ap.add_argument("--frames", type=int, default=8, help="#players (frames) in the game")
    ap.add_argument("--approximate", action="store_true")
    ap.add_argument("--max-samples", type=int, default=128)
    ap.add_argument("--ucf-root", type=Path, default=Path(r"D:\datasets\UCF-101"))
    ap.add_argument("--ssv2-root", type=Path, default=Path(r"D:\datasets\SSV2\s2s_test"))
    ap.add_argument("--random-weights", action="store_true", help="R3D: skip the checkpoint (plumbing smoke test)")
    ap.add_argument("--out", type=Path, default=HERE / "playfair_out" / "model_tests")
    a = ap.parse_args()

    jobs = []
    if a.dataset in ("ucf101", "both"):
        jobs += [("UCF101/R3D-50", "playfair_adapters:build_r3d_ucf101", v, gt, True)
                 for v, gt in pick_ucf(a.ucf_root, a.n)]
    if a.dataset in ("ssv2", "both"):
        jobs += [("SSv2/V-JEPA2", "playfair_adapters:build_vjepa2_ssv2", v, gt, False)
                 for v, gt in pick_ssv2(a.ssv2_root, a.n)]

    rows, failures = [], 0
    for name, adapter, video, gt, gt_from_class_dir in jobs:
        out_dir = a.out / (name.split("/")[0] + "_" + video.stem)
        extra = ["--adapter-arg", "random_weights=1"] if (a.random_weights and "r3d" in adapter) else []
        rec, log = run(video, adapter, a.frames, a.approximate, a.max_samples, out_dir, extra)
        if rec is None:
            print(f"[FAIL] {name} {video.name}: run_playfair crashed\n{log[-1500:]}")
            failures += 1
            continue
        pred = rec["predicted"]["name"]
        eff_ok = a.approximate or abs(rec["efficiency_gap"]) < 1e-4
        probe_ok = not rec["probe_ignored_positions"]
        gt_ok = (pred.lower() == gt.lower()) if gt_from_class_dir else (pred.replace("[", "").replace("]", "") == gt)
        ok = eff_ok and probe_ok
        failures += 0 if ok else 1
        rows.append((name, video.name, gt, pred, rec["predicted"]["prob"], gt_ok, rec["efficiency_gap"],
                     probe_ok, rec["n_model_evals"], rec["seconds"], "PASS" if ok else "FAIL"))

    hdr = f"{'model/dataset':14} {'video':28} {'ground truth':26} {'prediction':26} {'p':>5} {'pred==gt':>8} {'eff.gap':>9} {'probe':>5} {'evals':>7} {'sec':>6}  result"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for r in rows:
        print(f"{r[0]:14} {r[1][:28]:28} {r[2][:26]:26} {r[3][:26]:26} {r[4]:5.2f} {str(r[5]):>8} {r[6]:+9.1e} "
              f"{'ok' if r[7] else 'BAD':>5} {r[8]:7d} {r[9]:6.1f}  {r[10]}")
    print(f"\n{len(rows) - failures}/{len(jobs)} runs passed the checks. Per-video outputs in {a.out}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
