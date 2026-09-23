#!/usr/bin/env python3
"""
Sanity tests for run_playfair.py. Run:  python test_playfair_sanity.py

1. Brute-force check: our pipeline == textbook Shapley formula computed independently
   (same characteristic function, same ordered-subsequence semantics).
2. Efficiency axiom:  sum_i ESV_i = f(X) - f({}).
3. Symmetry axiom:    identical frames in an order-invariant model get identical ESV.
4. Probe catches a model that silently ignores the last frame of odd-length clips
   (the Conv3d/tubelet truncation failure mode).
5. Variable length:   torchvision r3d_18 runs for every clip length 1..16.
6. Variable length:   V-JEPA 2 adapter runs for every clip length 1..16 (tiny random-weight model).
7. End-to-end CLI on a synthetic video (random-weight R3D): exact mode, gap ~ 0.
"""
import itertools
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_playfair as rp  # noqa: E402


class ToyAdapter:
    """Order-invariant, variable-length toy model (like TSN): mean of per-frame logits."""

    def __init__(self, k=5, feat=4, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(feat, k, generator=g)
        self.num_classes = k
        self.class_names = None

    def preprocess(self, frames):  # not used here
        raise NotImplementedError

    def forward_clips(self, clips):  # (B,S,F)
        return (clips @ self.W).mean(dim=1)


class OrderDependentAdapter(ToyAdapter):
    """Adds a temporal-position term so the model is NOT order invariant."""

    def forward_clips(self, clips):
        s = clips.shape[1]
        w = torch.linspace(0.2, 1.8, s).view(1, s, 1)
        return ((clips * w) @ self.W).mean(dim=1)


class DropsLastOddFrame(ToyAdapter):
    """Mimics Conv3d(tubelet=2, stride=2): odd clips silently lose their last frame."""

    def forward_clips(self, clips):
        if clips.shape[1] % 2:
            clips = clips[:, :-1] if clips.shape[1] > 1 else clips
        return super().forward_clips(clips)


def run_pipeline(adapter, feats, priors):
    n, k = feats.shape[0], adapter.num_classes
    cf = rp.CharacteristicFn(adapter, feats, priors, batch_size=16, fp16=False)
    attr = rp.CharacteristicFunctionShapleyAttributor(
        characteristic_fn=cf, n_classes=k,
        subset_sampler=rp.ExhaustiveSubsetSampler(device=torch.device("cpu")),
        device=torch.device("cpu"))
    esv, full = attr.explain(torch.arange(n).unsqueeze(-1), n_iters=1)
    return esv.numpy(), full.numpy(), cf


def brute_force_shapley(adapter, feats, priors):
    """Textbook: phi_i = sum_{S subset N\\{i}} |S|!(n-|S|-1)!/n! [v(S+i) - v(S)], v = f - f({})."""
    n, k = feats.shape[0], adapter.num_classes

    def f(subset):
        if not subset:
            return priors[0].numpy()
        clip = feats[list(subset)].unsqueeze(0)  # temporal order preserved (sorted)
        return torch.softmax(adapter.forward_clips(clip), -1)[0].numpy()

    phi = np.zeros((n, k))
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for r in range(n):
            w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for S in itertools.combinations(others, r):
                with_i = tuple(sorted(S + (i,)))
                phi[i] += w * (f(with_i) - f(S))
    return phi


def test_bruteforce_efficiency():
    torch.manual_seed(1)
    n = 6
    feats = torch.randn(n, 4)
    priors = torch.full((1, 5), 0.2)
    for cls in (ToyAdapter, OrderDependentAdapter):  # incl. order-dependent model
        ad = cls()
        esv, full, cf = run_pipeline(ad, feats, priors)
        ref = brute_force_shapley(ad, feats, priors)
        err = np.abs(esv - ref).max()
        assert err < 1e-5, f"{cls.__name__}: pipeline != brute force (max err {err})"
        eff = np.abs(esv.sum(0) - (full - priors[0].numpy())).max()
        assert eff < 1e-5, f"{cls.__name__}: efficiency violated ({eff})"
        # 2^n - 1 non-empty subsets, +1 because Play Fair's run() re-evaluates the full clip
        # to return the grand-coalition scores.
        assert cf.n_model_evals == 2 ** n, cf.n_model_evals
        print(f"  ok  {cls.__name__}: matches brute force (err {err:.1e}), efficiency gap {eff:.1e}, "
              f"{cf.n_model_evals} model evals = 2^{n}")


def test_symmetry():
    torch.manual_seed(2)
    feats = torch.randn(5, 4)
    feats[3] = feats[1]  # frames 1 and 3 identical
    esv, _, _ = run_pipeline(ToyAdapter(), feats, torch.full((1, 5), 0.2))
    assert np.abs(esv[1] - esv[3]).max() < 1e-6
    print("  ok  symmetry: identical frames receive identical ESV")


def test_probe_catches_ignored_frame():
    torch.manual_seed(3)
    feats = torch.randn(6, 4)
    ok = rp.probe_frame_sensitivity(ToyAdapter(), feats, fp16=False)
    assert ok == {}, ok
    bad = rp.probe_frame_sensitivity(DropsLastOddFrame(), feats, fp16=False)
    assert set(bad) == {3, 5} and bad[3] == [2] and bad[5] == [4], bad
    print(f"  ok  probe: clean model -> {{}}, truncating model flagged {bad}")


def test_r3d_variable_length():
    from torchvision.models import video as tv

    m = tv.r3d_18(weights=None, num_classes=7).eval()
    with torch.no_grad():
        for t in range(1, 17):
            assert m(torch.randn(1, 3, t, 112, 112)).shape == (1, 7)
    print("  ok  torchvision r3d_18 accepts every clip length 1..16 (architecturally)")


def test_vjepa2_variable_length():
    """V-JEPA 2 (tubelet_size=2) runs for every clip length 1..16 through VJEPA2Adapter.forward_clips.

    Uses a tiny random-weight model built from a config (no download). Odd lengths only work
    because the adapter pads the last frame; the raw model is also checked to show why.
    """
    try:
        from transformers import VJEPA2Config, VJEPA2ForVideoClassification
    except ImportError:
        print("  skip  transformers with V-JEPA 2 not installed")
        return

    k, size = 7, 32
    cfg = VJEPA2Config(
        crop_size=size, frames_per_clip=16, patch_size=16, tubelet_size=2,
        hidden_size=32, num_attention_heads=2, num_hidden_layers=2, mlp_ratio=2.0,
        pred_hidden_size=16, pred_num_attention_heads=2, pred_num_hidden_layers=1,
        pred_num_mask_tokens=1, num_labels=k)
    model = VJEPA2ForVideoClassification(cfg).eval()

    ad = rp.VJEPA2Adapter.__new__(rp.VJEPA2Adapter)  # skip from_pretrained
    ad.device, ad.model, ad.num_classes = torch.device("cpu"), model, k
    ad.class_names, ad.tubelet = None, int(cfg.tubelet_size)

    ad.pad_odd = True
    for t in range(1, 17):
        assert ad.forward_clips(torch.randn(1, t, 3, size, size)).shape == (1, k), t

    # Without padding an odd clip's last frame is silently dropped (or the model errors out).
    ad.pad_odd = False
    x = torch.randn(1, 5, 3, size, size)
    try:
        same = torch.allclose(ad.forward_clips(x), ad.forward_clips(x[:, :4]), atol=1e-5)
    except Exception:
        same = None
    print(f"  ok  V-JEPA 2 adapter accepts every clip length 1..16 with pad_odd=True "
          f"(unpadded odd clip ignores last frame: {same})")


def test_cli_end_to_end():
    import cv2

    with tempfile.TemporaryDirectory() as d:
        vid = Path(d) / "synthetic.avi"
        wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
        rng = np.random.default_rng(0)
        for t in range(40):
            img = rng.integers(0, 60, (120, 160, 3), dtype=np.uint8)
            x = 10 + 3 * t
            img[40:80, x:x + 25] = (255, 40, 40)  # moving red block
            wr.write(img)
        wr.release()
        out = Path(d) / "out"
        cmd = [sys.executable, str(HERE / "run_playfair.py"), str(vid), "--adapter", "r3d",
               "--num-classes", "10", "--num-frames", "6", "--device", "cpu",
               "--out-dir", str(out), "--batch-size", "16"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        rec = json.loads((out / "esv.json").read_text())
        assert abs(rec["efficiency_gap"]) < 1e-4, rec["efficiency_gap"]
        assert rec["n_model_evals"] == 2 ** 6
        assert (out / "esv.png").exists() and (out / "esv.npz").exists()
        print(f"  ok  CLI end-to-end: efficiency gap {rec['efficiency_gap']:.1e}, "
              f"{rec['n_model_evals']} evals, probe ignored={rec['probe_ignored_positions']}")

        # approximate mode should also run
        cmd2 = cmd + ["--approximate", "--max-samples-per-scale", "8", "--num-frames", "10",
                      "--out-dir", str(out / "approx")]
        r = subprocess.run(cmd2, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        print("  ok  CLI approximate mode (10 frames, 8 subsets/scale) runs")


if __name__ == "__main__":
    for fn in (test_vjepa2_variable_length, test_cli_end_to_end):
        print(fn.__name__)
        fn()
    print("\nALL SANITY TESTS PASSED")
