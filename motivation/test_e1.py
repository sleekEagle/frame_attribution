"""
test_e1.py -- sanity tests for the E1 pipeline. Needs only torch/numpy (uses the synthetic toy
model, no datasets or checkpoints).  Run (from anywhere):  python motivation/test_e1.py

1. Layouts: insertion lengths/copies, tubelet alignment, spread-control length match,
   reallocation slot counts and identical losers in target vs recipient control; replacement
   blocks (contiguous, tubelet-aligned, same loser slots in target and control).
2. Removal operators: drop / past / future / late.
3. Theory check on a max-pool toy model (exact copies are perfect substitutes):
     - exact Shapley gives every copy of the target the same value (symmetry),
     - leave-one-out (drop) gives each copy 0 once m >= 2,
     - per-copy Shapley falls as m grows (dilution), sum over copies is roughly kept.
4. Exact Shapley == permutation Shapley (many permutations) on a small clip.
5. Integrated Gradients completeness: sum of per-slot IG == logit(x) - logit(0).
6. End-to-end: build conditions -> attribute -> metrics on the toy model.
"""
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

import e1_common as E

HERE = Path(__file__).resolve().parent


def test_layouts():
    lay, al = E.insertion_layout(16, 5, 4, 1)
    assert len(lay["content"]) == 19 and lay["content"].count(5) == 4 and al
    assert lay["content"][5:9] == [5, 5, 5, 5]
    # tubelet 2: target opening its tubelet (even) -> copies inserted before it
    lay, al = E.insertion_layout(16, 4, 3, 2)
    assert lay["content"][:8] == [0, 1, 2, 3, 4, 4, 4, 5] and al
    assert lay["copy"][4:7] == [1, 2, 0], lay["copy"]  # original keeps copy index 0
    pairs = [lay["content"][i:i + 2] for i in range(0, len(lay["content"]), 2)]
    assert [4, 4] in pairs and [4, 5] in pairs  # (4',4'')(4,5): every original pair intact
    # target closing its tubelet (odd) -> copies after it
    lay, _ = E.insertion_layout(16, 5, 3, 2)
    assert lay["content"][4:8] == [4, 5, 5, 5]
    # spread control: same length, target not duplicated
    for tub, m in [(1, 4), (2, 5)]:
        tgt, _ = E.insertion_layout(16, 6, m, tub)
        ctrl, dup = E.spread_control_layout(16, 6, m, tub, random.Random(0))
        assert len(ctrl["content"]) == len(tgt["content"]) and ctrl["content"].count(6) == 1, (tub, m)
        assert max(ctrl["content"].count(c) for c in range(16)) == 2
    # reallocation
    rng = random.Random(0)
    for tub in (1, 2):
        for m in (2, 4, 8):
            t, c, losers, al = E.realloc_layouts(8, 2, m, 6, tub, rng)
            assert len(t["content"]) == 16 and len(c["content"]) == 16
            assert t["content"].count(2) == m and c["content"].count(6) == m and c["content"].count(2) == 2
            for j in losers:
                assert t["content"].count(j) == 1 and c["content"].count(j) == 1
            if tub == 2 and al:
                pairs = [t["content"][i:i + 2] for i in range(0, 16, 2)]
                assert [2, 2] in pairs
    # replacement: reference = original clip; the target's copies replace m-1 neighbours
    for tub in (1, 2):
        for target, recipient in [(5, 12), (0, 15), (14, 1), (7, 8)]:
            for m in (2, 4, 6, 8):
                rng = random.Random(m)
                t, c, losers, al = E.replacement_layouts(16, target, m, recipient, tub, rng)
                if t is None:  # no block can contain the target and avoid the recipient
                    assert tub == 2 and target // 2 == recipient // 2, (target, recipient, m)
                    continue
                assert len(t["content"]) == len(c["content"]) == 16
                assert t["content"].count(target) == m and c["content"].count(target) == 1
                assert c["content"].count(recipient) == m and len(losers) == m - 1
                for j in losers:  # losers are gone from both clips; same slots in both
                    assert j not in t["content"] and j not in c["content"]
                block = [s for s in range(16) if t["content"][s] == target]
                assert block == list(range(block[0], block[0] + m))  # one contiguous block
                assert [s for s in range(16) if c["content"][s] == recipient and s != recipient] \
                    == [s for s in block if s != target]  # control copies sit in the loser slots
                for s in range(16):  # everything outside the block is unchanged
                    if s not in block:
                        assert t["content"][s] == s and c["content"][s] == s
                assert t["copy"][target] == 0 and c["copy"][recipient] == 0  # originals keep copy 0
                if tub == 2:
                    assert al and block[0] % 2 == 0  # whole tubelets
    print("layouts ok")


def test_removal():
    lay = E.make_layout([0, 1, 2, 3])
    keep = np.array([True, False, False, True])
    assert E.removed_layouts(lay, keep, "drop")[0]["content"] == [0, 3]
    assert E.removed_layouts(lay, keep, "past")[0]["content"] == [0, 0, 0, 3]
    assert E.removed_layouts(lay, keep, "future")[0]["content"] == [0, 3, 3, 3]
    assert len(E.removed_layouts(lay, keep, "late")) == 2
    keep = np.array([False, False, True, True])
    assert E.removed_layouts(lay, keep, "past")[0]["content"] == [2, 2, 2, 3]  # falls back to future
    assert E.removed_layouts(lay, np.zeros(4, bool), "drop") == []
    print("removal ok")


def _toy_setup(n=6):
    model = E.load_clip_model("toy")
    frames, _ = model.load_frames("toy_video_theory")
    frames = frames[:n]
    bank = E.FrameBank(model, frames, "exact", 0)
    ev = E.Evaluator(model, bank)
    ref = E.make_layout(range(n))
    cls = int(ev.logits([ref])[0].argmax())
    prior = E.uniform_prior(model.num_classes)
    return model, bank, ev, ref, cls, prior


def test_theory():
    model, bank, ev, ref, cls, prior = _toy_setup(6)
    I = E.content_importance(ev, ref, "drop", cls, prior)
    t = max(I, key=I.get)
    per_copy, sums = [], []
    for m in (1, 2, 3, 4):
        lay, _ = E.insertion_layout(6, t, m, 1)
        phi = E.attr_shapley(ev, lay, cls, prior, "drop", exact_max=12)
        copies = [phi[i] for i, c in enumerate(lay["content"]) if c == t]
        assert np.allclose(copies, copies[0], atol=1e-6), copies  # symmetry
        loo = E.attr_loo(ev, lay, cls, prior, "drop")
        loo_t = [loo[i] for i, c in enumerate(lay["content"]) if c == t]
        if m >= 2:
            assert np.allclose(loo_t, 0, atol=1e-6), loo_t  # redundancy masking
        per_copy.append(copies[0])
        sums.append(sum(copies))
    assert all(a > b for a, b in zip(per_copy, per_copy[1:])), per_copy
    beta = np.polyfit(np.log([1, 2, 3, 4]), np.log(per_copy), 1)[0]
    print(f"theory ok: per-copy Shapley {np.round(per_copy, 4)}, sum over copies {np.round(sums, 4)}, "
          f"slope {beta:.2f}")


def test_exact_vs_permutation():
    model, bank, ev, ref, cls, prior = _toy_setup(6)
    ex = E.attr_shapley(ev, ref, cls, prior, "drop", exact_max=12)
    pe = E.attr_shapley(ev, ref, cls, prior, "drop", exact_max=0, n_perm=4000, seed=0)
    assert np.allclose(ex, pe, atol=0.02), (ex, pe)
    full = ev.value(ref, np.ones((1, 6), bool), "drop", cls, prior)[0]
    assert abs(ex.sum() - (full - prior[cls])) < 1e-6  # efficiency
    print("exact vs permutation ok")


def test_ig_completeness():
    model, bank, ev, ref, cls, prior = _toy_setup(6)
    ig = E.attr_integrated_gradients(model, bank, ref, cls, steps=256, batch=64)
    x = bank.clip(ref)[None]
    diff = (model.forward_clips(x)[0, cls] - model.forward_clips(torch.zeros_like(x))[0, cls]).item()
    assert abs(ig.sum() - diff) < 0.05 * max(1.0, abs(diff)), (ig.sum(), diff)
    print("IG completeness ok")


def test_vit_adapters():
    """Tiny random-weight V-JEPA 2 / VideoMAE through the E1 adapters (no download): forward for
    every clip length used by E1 (odd lengths padded), Grad-CAM and IG return one score per slot."""
    try:
        from transformers import (VJEPA2Config, VJEPA2ForVideoClassification, VideoMAEConfig,
                                  VideoMAEForVideoClassification)
    except ImportError:
        print("skip ViT adapters (transformers missing)")
        return
    k, size = 7, 32
    vj = E.VJEPA2Clip.__new__(E.VJEPA2Clip)
    vj.name, vj.spec, vj.device, vj.num_classes = "vjepa2", dict(E.MODEL_SPECS["vjepa2"]), "cpu", k
    vj.hf = VJEPA2ForVideoClassification(VJEPA2Config(
        crop_size=size, frames_per_clip=16, patch_size=16, tubelet_size=2, hidden_size=32,
        num_attention_heads=2, num_hidden_layers=2, mlp_ratio=2.0, pred_hidden_size=16,
        pred_num_attention_heads=2, pred_num_hidden_layers=1, pred_num_mask_tokens=1,
        num_labels=k)).eval()
    vm = E.VideoMAEClip.__new__(E.VideoMAEClip)
    vm.name, vm.spec, vm.device, vm.num_classes = "videomae", dict(E.MODEL_SPECS["videomae"]), "cpu", k
    vm.hf = VideoMAEForVideoClassification(VideoMAEConfig(
        image_size=size, patch_size=16, num_frames=16, tubelet_size=2, hidden_size=32,
        num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, num_labels=k)).eval()
    for model, lengths in [(vj, range(1, 25)), (vm, range(1, 17))]:
        for s in lengths:
            assert model.forward_clips(torch.randn(2, s, 3, size, size)).shape == (2, k), (model.name, s)

        class _Bank:  # minimal FrameBank stand-in
            def __init__(self, n):
                self.x = torch.randn(n, 3, size, size)

            def clip(self, lay):
                return torch.stack([self.x[c] for c in lay["content"]])
        for lay in [E.make_layout(range(16)), E.insertion_layout(16, 4, 3, 2)[0] if model is vj
                    else E.make_layout(range(15))]:
            bank = _Bank(16)
            gc = E.attr_gradcam(model, bank, lay, 0)
            ig = E.attr_integrated_gradients(model, bank, lay, 0, steps=4, batch=2)
            assert gc.shape == (len(lay["content"]),) and ig.shape == gc.shape
            assert np.allclose(gc[0::2][:len(gc[1::2])], gc[1::2]), "tubelet frames share a CAM"
    print("ViT adapters ok")


def test_end_to_end():
    tmp = Path(tempfile.mkdtemp())
    try:
        for model in ("toy", "toy_realloc", "toy_replace"):
            run = lambda *a: subprocess.run([sys.executable, str(HERE / a[0]), *a[1:], "--out", str(tmp)], cwd=HERE,  # noqa: E731
                                            check=True, capture_output=True, text=True)
            run("e1_build_conditions.py", "--model", model, "--limit", "3", "--include-wrong",
                "--dup-types", "exact", "noise")
            methods = (["shapley_freeze", "loo_freeze", "occlusion", "ig"] if model == "toy_realloc"
                       else ["shapley_drop", "loo_drop", "occlusion", "ig"])
            run("e1_attribute.py", "--model", model, "--methods", *methods, "--seeds", "0", "1",
                "--n-perm", "16")
            run("e1_metrics.py", "gate", "--model", model)
            r = run("e1_metrics.py", "attr", "--model", model)
            for f in ("conditions.jsonl", "attributions.jsonl", "gate.csv", "rows.csv", "beta.csv"):
                assert (tmp / f"{model}_{f}").exists(), f
            print(f"end-to-end ok ({model})")
            print(r.stdout.strip().split("\n\n")[-1])
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_layouts()
    test_removal()
    test_theory()
    test_exact_vs_permutation()
    test_ig_completeness()
    test_vit_adapters()
    test_end_to_end()
    print("\nall E1 tests passed")
