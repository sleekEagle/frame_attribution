"""
fig_groups.py -- groupings against the model's own change, in the same readout as fig_dilution.py.

For every target (E1 target clips, exact copies, largest number of copies) or target segment
(frame-rate test, target segment densified, largest density d), the deviation is the change in the
score of one frame divided by the model's own change: per_copy_ratio / I_ratio (E1) or
per_frame / I_ratio (frame-rate test). Group scores are broadcast: each frame shows the value of its
group, not divided. Median over targets with I_ratio > 0. 1 means that the score follows the model.

One row per attribution method or grouping. Filled markers: E1; open markers: frame-rate test.

    python motivation/fig_groups.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = [  # (prefix, label, largest m, Shapley, marker, colour) -- as in fig_dilution.py
    ("r3d", "R3D-50", 8, "shapley_freeze", "o", "#2a78d6"),
    ("r3d_18", "R3D-18", 8, "shapley_freeze", "s", "#eb6834"),
    ("mc3_18", "MC3-18", 9, "shapley_drop", "^", "#1baf7a"),
    ("vjepa2", "V-JEPA2", 9, "shapley_drop", "D", "#eda100"),
    ("trn_official", "TRN", 4, "shapley_freeze", "v", "#e87ba4"),
]
FR_MODELS = ("mc3_18", "vjepa2")
FR_D = 3
ROWS = ["Shapley values, frames", "windows of fixed width", "windows, fixed number",
        "DINOv2, fixed number", "DINOv2, Ward threshold", "DINOv2, average-linkage threshold"]
MS = {"o": 5, "s": 4.6, "D": 4.4, "^": 6, "v": 6}
FLOOR = 0.05
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def grouping(row, prefix, shapley, fr):
    """Method name in the rows files; None if the grouping was not run."""
    if row == "Shapley values, frames":
        return shapley
    if fr:
        return {"windows of fixed width": "grp_win_w4", "windows, fixed number": "grp_win_k8",
                "DINOv2, fixed number": "grp_dino_k8", "DINOv2, Ward threshold": "grp_dino_thr",
                "DINOv2, average-linkage threshold": "grp_dinoavg_thr"}[row]
    small = prefix in ("trn_official", "vjepa2")    # K = 4 (TRN: 8 frames; V-JEPA2 ran only K = 4)
    return {"windows of fixed width": {"trn_official": "grp_win_w2", "r3d": "grp_win_k4",
                                        "r3d_18": "grp_win_k4"}.get(prefix, "grp_win_w4"),
            "windows, fixed number": "grp_win_k4" if small else "grp_win_k8",
            "DINOv2, fixed number": "grp_dino_k4" if small else "grp_dino_k8",
            "DINOv2, Ward threshold": "grp_dino_thr",
            "DINOv2, average-linkage threshold": "grp_dinoavg_thr"}[row]


def deviation(d, col):
    x = (d[col] / d.I_ratio).replace([np.inf, -np.inf], np.nan).dropna()
    return (float(np.median(x)) if len(x) else np.nan), len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", default="results/e1")
    ap.add_argument("--fr", default="results/fr")
    ap.add_argument("--out", default="paper/figures/fig_groups.pdf")
    args = ap.parse_args()

    e1 = {}
    for p, _, m, *_ in MODELS:
        d = pd.read_csv(Path(args.e1) / f"{p}_practical_rows.csv")
        e1[p] = d[(d.kind == "target") & (d.dup == "exact") & (d.m == m) & (d.I_ratio > 0)]
    fr = {}
    for p in FR_MODELS:
        d = pd.read_csv(Path(args.fr) / f"{p}_fr_practical_rows.csv")
        fr[p] = d[(d.kind == "target") & (d.d == FR_D) & (d.I_ratio > 0)]

    plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    slots = [(p, False) for p, *_ in MODELS] + [(p, True) for p in FR_MODELS]
    offsets = np.linspace(-0.3, 0.3, len(slots))
    style = {p: (mk, c) for p, _, _, _, mk, c in MODELS}
    shapley = {p: s for p, _, _, s, *_ in MODELS}
    label = {p: lab for p, lab, *_ in MODELS}
    table = []
    for r, row in enumerate(ROWS):
        ax.axhline(r, color=GRID, lw=0.5, zorder=1)
        for (p, is_fr), dy in zip(slots, offsets):
            meth = grouping(row, p, shapley[p], is_fr)
            src = fr[p] if is_fr else e1[p]
            d = src[src.method == meth]
            x, n = deviation(d, "per_frame" if is_fr else "per_copy_ratio")
            table.append((row, label[p] + (" (frame rate)" if is_fr else ""), meth, n, x))
            if not np.isfinite(x):
                continue
            mk, c = style[p]
            st = dict(mfc="white", mec=c, mew=1.1) if is_fr else dict(mfc=c, mec="white", mew=0.8)
            ax.plot([max(x, FLOOR)], [r + dy], marker=mk, ms=MS[mk], ls="none", zorder=5, **st)
    ax.axvline(1, color=INK, lw=0.9, zorder=2)
    ax.text(0.93, -0.68, "$\\leftarrow$ dilutes", fontsize=6.5, color=MUTED, ha="right", va="bottom")
    ax.text(1.0, -0.68, "1", fontsize=6.5, color=INK, ha="center", va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(0.1, 1.6)
    ax.set_xticks([0.1, 0.2, 0.5, 1])
    ax.set_xticklabels(["0.1", "0.2", "0.5", "1"])
    ax.set_ylim(len(ROWS) - 0.6, -0.8)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels(ROWS)
    ax.tick_params(length=2, width=0.6)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("change per frame / model's change  ($r_\\phi / r_I$)")
    handles = [plt.Line2D([], [], marker=mk, ls="none", ms=MS[mk], mfc=c, mec="white", mew=0.8,
                          label=lab) for _, lab, _, _, mk, c in MODELS]
    handles += [plt.Line2D([], [], marker=style[p][0], ls="none", ms=MS[style[p][0]], mfc="white",
                           mec=style[p][1], mew=1.1, label=f"{label[p]}, frame rate") for p in FR_MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=6.5, bbox_to_anchor=(0.5, -0.03), handletextpad=0.25, columnspacing=0.9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(str(Path(args.out).with_suffix(".png")), dpi=200, bbox_inches="tight")

    print(f"{'row':<36}{'model':<24}{'method':<18}{'n':>5}{'r_phi/r_I':>11}")
    for row, lab, meth, n, x in table:
        print(f"{row:<36}{lab:<24}{meth:<18}{n:>5}{x:>11.3f}")


if __name__ == "__main__":
    main()
