"""
fig_dilution.py -- E1: how much each attribution method dilutes a repeated frame.

For every target in the E1 target clips (exact copies, largest number of copies tested), the
deviation is per_copy_ratio / I_ratio: the change in the score of one copy divided by the model's
own change. An attribution that follows the model sits at 1; below 1 it dilutes.

One row per method, one point per model (median over targets, both seeds pooled). Leave-one-out
uses deletion where the model allows it (R3D-50, R3D-18 on the same 120 videos as the freezing run,
MC3-18, V-JEPA2) and freezing on TRN (grey marker: partly fixed by the operator).

    python motivation/fig_dilution.py --r3d18-loo-drop <dir with r3d_18_rows.csv incl. loo_drop>
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = [  # (file prefix, label, largest m, Shapley method, marker)
    ("r3d", "R3D-50", 8, "shapley_freeze", "o"),
    ("r3d_18", "R3D-18", 8, "shapley_freeze", "s"),
    ("mc3_18", "MC3-18", 9, "shapley_drop", "^"),
    ("vjepa2", "V-JEPA2", 9, "shapley_drop", "D"),
    ("trn_official", "TRN", 4, "shapley_freeze", "v"),
]
ROWS = ["Shapley", "Leave-one-out", "Occlusion", "Integrated Gradients", "Grad-CAM"]
FLOOR = 0.005           # values at or below this are drawn at the floor (log axis)
BLUE = "#2a78d6"        # categorical slot 1 (one series)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def method_for(row, model, shapley):
    if row == "Shapley":
        return shapley
    if row == "Leave-one-out":
        return "loo_freeze" if model == "trn_official" else "loo_drop"
    return {"Occlusion": "occlusion", "Integrated Gradients": "ig", "Grad-CAM": "gradcam"}[row]


def point(rows, method, m):
    d = rows[(rows.method == method) & (rows.dup == "exact") & (rows.kind == "target") & (rows.m == m)]
    d = d[d.I_ratio > 0]
    return (d.per_copy_ratio / d.I_ratio).median(), len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", default="results/e1")
    ap.add_argument("--r3d18-loo-drop", required=True,
                    help="dir whose r3d_18_rows.csv contains loo_drop on the first 120 videos")
    ap.add_argument("--out", default="paper/figures/fig_dilution.pdf")
    args = ap.parse_args()

    rows = {p: pd.read_csv(Path(args.e1) / f"{p}_rows.csv") for p, *_ in MODELS}
    r18_drop = pd.read_csv(Path(args.r3d18_loo_drop) / "r3d_18_rows.csv")

    plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    offsets = np.linspace(-0.24, 0.24, len(MODELS))   # one sub-row per model, so markers never overlap
    table = []
    for r, row in enumerate(ROWS):
        ax.axhline(r, color=GRID, lw=0.5, zorder=1)
        for (prefix, label, m, shapley, marker), dy in zip(MODELS, offsets):
            method = method_for(row, prefix, shapley)
            src = r18_drop if (prefix == "r3d_18" and method == "loo_drop") else rows[prefix]
            x, n = point(src, method, m)
            table.append((row, label, method, m, n, x))
            if not np.isfinite(x):
                continue
            ax.plot([max(x, FLOOR)], [r + dy], marker=marker, ms=5.5, mec="white", mew=0.9,
                    mfc=MUTED if method == "loo_freeze" else BLUE, ls="none", zorder=5)
    ax.axvline(1, color=INK, lw=0.9, zorder=2)
    ax.text(0.9, -0.62, "$\\leftarrow$ dilutes", fontsize=6.5, color=MUTED, ha="right", va="bottom")
    ax.text(1.0, -0.62, "1", fontsize=6.5, color=INK, ha="center", va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(FLOOR * 0.8, 1.6)
    ax.set_xticks([0.01, 0.1, 1])
    ax.set_xticklabels(["0.01", "0.1", "1"])
    ax.set_ylim(len(ROWS) - 0.6, -0.75)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels(ROWS)
    ax.tick_params(length=2, width=0.6)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("change in the score of one copy / model's change  ($r_\\phi / r_I$)")
    handles = [plt.Line2D([], [], marker=mk, ls="none", ms=5.5, mfc=BLUE, mec="white", mew=0.9,
                          label=lab) for _, lab, _, _, mk in MODELS]
    handles += [plt.Line2D([], [], marker="v", ls="none", ms=5.5, mfc=MUTED, mec="white", mew=0.9,
                           label="TRN, leave-one-out with freezing")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=6.8, bbox_to_anchor=(0.5, -0.03), handletextpad=0.3, columnspacing=1.1)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(str(Path(args.out).with_suffix(".png")), dpi=200, bbox_inches="tight")

    print(f"{'method':<22}{'model':<9}{'source':<16}{'m':>3}{'n':>5}{'r_phi/r_I':>11}")
    for row, label, method, m, n, x in table:
        print(f"{row:<22}{label:<9}{method:<16}{m:>3}{n:>5}{x:>11.3f}")


if __name__ == "__main__":
    main()
