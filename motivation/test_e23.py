"""
test_e23.py -- sanity tests for E2/E3 (e23_run.py, e23_metrics.py). Uses the toy model and pixel
clustering only (no datasets, checkpoints or DINOv2).  Run:  python motivation/test_e23.py

1. Tree helpers: tree_nodes() has 2n-1 contiguous nodes, level_groups(k) has k groups that
   partition the clip, and every node of the cut is a tree node.
2. Known answer on runs of exact copies (toy max-pool model, so copies are perfect substitutes):
   pixel clustering recovers every run as a tree node; within a run, exact Shapley gives every
   copy the same value; leave-one-out of a copy is 0 (the others stand in for it) while the run's
   group importance is not; e23_metrics' r_loo is therefore 0 and r_max = r_sum / |run|.
3. group_shapley: singleton groups give frame-level Shapley; the values add up to
   v(all) - v(empty); copy invariance (a run's group value doesn't change when the run gets
   longer, while the frame-level credit per copy is divided).
4. End-to-end: e23_run.py -> e23_metrics.py on the toy model.
5. e1_group.pair_groups: the target's copies form one group, the other contents form neighbouring
   pairs that never span the target, the groups partition the clip, and reference, target and
   control clips get the same partition of contents.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

import e1_common as E
from e23_run import group_importance, level_groups, tree_nodes
from frame_clustering import hierarchy_from_embeddings

HERE = Path(__file__).resolve().parent


def test_tree():
    rng = np.random.default_rng(0)
    for n in (2, 5, 8, 16):
        Z = hierarchy_from_embeddings(rng.normal(size=(n, 6)))
        nodes = tree_nodes(Z, n)
        assert len(nodes) == 2 * n - 1 and nodes[-1]["frames"] == list(range(n))
        for nd in nodes:
            f = nd["frames"]
            assert f == list(range(f[0], f[-1] + 1)), "nodes must be contiguous"
        node_set = {tuple(nd["frames"]) for nd in nodes}
        for k in range(1, n + 1):
            gs = level_groups(nodes, n, k)
            assert len(gs) == k and sorted(i for g in gs for i in g) == list(range(n))
            assert all(tuple(g) in node_set for g in gs)
    print("tree ok")


def test_runs_known_answer():
    model = E.load_clip_model("toy")
    runs = [1, 2, 3, 4]  # n = 10 -> exact Shapley (1024 subsets)
    n = sum(runs)
    g = torch.Generator().manual_seed(3)
    distinct = torch.randint(0, 256, (len(runs), 3, model.size, model.size), generator=g, dtype=torch.uint8)
    frames = torch.cat([distinct[j][None].repeat(r, 1, 1, 1) for j, r in enumerate(runs)])
    bounds = np.cumsum([0] + runs)
    run_frames = [list(range(bounds[j], bounds[j + 1])) for j in range(len(runs))]

    emb = frames.float().reshape(n, -1).numpy()
    nodes = tree_nodes(hierarchy_from_embeddings(emb), n)
    node_set = {tuple(nd["frames"]) for nd in nodes}
    for rf in run_frames:
        assert tuple(rf) in node_set, f"run {rf} is not a tree node"

    bank = E.FrameBank(model, frames, "exact", 0)
    ev = E.Evaluator(model, bank)
    layout = E.make_layout(range(n))
    cls = int(ev.logits([layout])[0].argmax())
    prior = E.uniform_prior(model.num_classes)
    I_runs, _ = group_importance(ev, layout, run_frames, "drop", cls, prior)
    phi = E.attr_shapley(ev, layout, cls, prior, "drop", exact_max=12)
    loo = E.attr_loo(ev, layout, cls, prior, "drop")
    for rf, I in zip(run_frames, I_runs):
        if len(rf) < 2:
            continue
        assert np.ptp(phi[rf]) < 1e-6, f"Shapley not equal within run {rf}: {phi[rf]}"
        assert np.abs(loo[rf]).max() < 1e-6, f"LOO of a copy should be 0: {loo[rf]}"
    assert max(abs(I) for I, rf in zip(I_runs, run_frames) if len(rf) > 1) > 1e-3, "runs should matter"
    print("runs known answer ok  (I(run) =", np.round(I_runs, 3).tolist(),
          " Shapley per copy =", [round(float(phi[rf[0]]), 4) for rf in run_frames], ")")


def test_group_shapley():
    """group_shapley: singletons == frame-level Shapley; values add up to v(all) - v(empty);
    copy invariance on the toy max-pool model (exact copies are perfect substitutes)."""
    model = E.load_clip_model("toy")
    g = torch.Generator().manual_seed(5)
    distinct = torch.randint(0, 256, (4, 3, model.size, model.size), generator=g, dtype=torch.uint8)
    prior = E.uniform_prior(model.num_classes)

    def setup(runs):
        frames = torch.cat([distinct[j][None].repeat(r, 1, 1, 1) for j, r in enumerate(runs)])
        bank = E.FrameBank(model, frames, "exact", 0)
        ev = E.Evaluator(model, bank)
        n = sum(runs)
        b = np.cumsum([0] + runs)
        return ev, E.make_layout(range(n)), [list(range(b[j], b[j + 1])) for j in range(len(runs))]

    ev, lay, groups = setup([1, 2, 1, 3])
    cls = int(ev.logits([lay])[0].argmax())
    n = len(lay["content"])
    phi_f = E.attr_shapley(ev, lay, cls, prior, "drop", exact_max=12)
    phi_s = E.group_shapley(ev, lay, [[i] for i in range(n)], cls, prior, "drop")
    assert np.allclose(phi_f, phi_s, atol=1e-6), "singleton groups must equal frame-level Shapley"
    phi_g = E.group_shapley(ev, lay, groups, cls, prior, "drop")
    v_all = ev.value(lay, np.ones((1, n), bool), "drop", cls, prior)[0]
    assert abs(phi_g.sum() - (v_all - prior[cls])) < 1e-6, "efficiency"
    # copy invariance: lengthen run 1 (2 -> 5 copies); its group value must not change
    ev2, lay2, groups2 = setup([1, 5, 1, 3])
    phi_g2 = E.group_shapley(ev2, lay2, groups2, cls, prior, "drop")
    assert np.allclose(phi_g, phi_g2, atol=1e-6), f"copy invariance failed: {phi_g} vs {phi_g2}"
    # ... while the frame-level credit per copy of that run is divided
    phi_f2 = E.attr_shapley(ev2, lay2, cls, prior, "drop", exact_max=12)
    print("group shapley ok  (group value of the run:", round(float(phi_g[1]), 4), "with 2 or 5 copies;",
          "frame-level credit per copy:", round(float(phi_f[1]), 4), "->", round(float(phi_f2[1]), 4), ")")


def test_pair_groups():
    """pair_groups on insertion-design layouts like V-JEPA2's (16 contents)."""
    from e1_group import pair_groups

    def contents(lay, groups):
        return sorted(tuple(sorted({lay["content"][i] for i in g})) for g in groups)

    ref = {"content": list(range(16))}
    for t in (0, 9, 10, 15):
        tgt = {"content": list(range(t + 1)) + [t] * 8 + list(range(t + 1, 16))}  # m = 9
        ctl = {"content": list(range(16)) + [14, 15]}  # control: extra copies of other contents
        per = []
        for lay in (ref, tgt, ctl):
            groups = pair_groups(lay, t)
            n = len(lay["content"])
            assert sorted(i for g in groups for i in g) == list(range(n)), "groups must partition the clip"
            t_slots = [i for i, c in enumerate(lay["content"]) if c == t]
            assert t_slots in groups, "the target's copies must form one group"
            for g in groups:
                cs = {lay["content"][i] for i in g}
                assert len(cs) <= 2 and (cs == {t} or t not in cs), cs
                assert not (min(cs) < t < max(cs)), f"a pair spans the target: {cs}"
            assert len(groups) <= 10
            per.append(contents(lay, groups))
        assert per[0] == per[1] == per[2], "reference, target and control must group contents alike"
    print("pair groups ok  (16 contents ->", len(pair_groups(ref, 10)), "groups for target 10,",
          len(pair_groups(ref, 9)), "for target 9)")


def test_end_to_end():
    tmp = Path(tempfile.mkdtemp())
    try:
        run = lambda *a: subprocess.run([sys.executable, str(HERE / a[0]), *a[1:], "--out", str(tmp)],  # noqa: E731
                                        cwd=HERE, check=True, capture_output=True, text=True)
        run("e23_run.py", "--model", "toy", "--limit", "3", "--clusterer", "pixel", "--include-wrong")
        r = run("e23_metrics.py", "--model", "toy", "--tau", "0", "--boot", "50")
        for f in ("e23.jsonl", "e2_rows.csv", "e2_bins.csv", "e2_slopes.csv", "e3_hits.csv",
                  "e3_inversions.csv", "e3_deletion.csv", "swap.csv"):
            assert (tmp / f"toy_{f}").exists(), f
        print("end-to-end ok")
        print("\n".join(r.stdout.strip().split("\n")[:8]))
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_tree()
    test_runs_known_answer()
    test_group_shapley()
    test_pair_groups()
    test_end_to_end()
    print("\nall E2/E3 tests passed")
