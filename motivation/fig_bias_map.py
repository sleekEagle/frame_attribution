"""
fig_bias_map.py -- E1: where each attribution method sits between dilution and inflation.

For every target in the E1 target clips (exact copies, largest number of copies tested), the
per-frame deviation is per_copy_ratio / I_ratio and the total deviation is conservation / I_ratio
(attribution change divided by the model's own change). A frame-count invariant attribution sits
at (1, 1). For a frame-level method, total = m x per-frame, so each point lies on a line of slope 1
(log axes) shifted by the number of copies m, and (1, 1) is out of reach when m > 1.

One panel per method, one point per model (median over targets, both seeds pooled). Leave-one-out
uses deletion where the model allows it (R3D-50, R3D-18 on the same 120 videos as the freezing run,
MC3-18, V-JEPA2) and freezing on TRN (hollow marker: partly fixed by the operator).

    python motivation/fig_bias_map.py --r3d18-loo-drop <dir with r3d_18_rows.csv incl. loo_drop>
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
PANELS = ["Shapley", "Leave-one-out", "Occlusion", "Integrated Gradients", "Grad-CAM"]
FLOOR = 0.01            # values at or below this are drawn at the floor (log axes)
BLUE = "#2a78d6"        # categorical slot 1 (one series per panel)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def method_for(panel, model, shapley):
    if panel == "Shapley":
        return shapley
    if panel == "Leave-one-out":
        return "loo_freeze" if model == "trn_official" else "loo_drop"
    return {"Occlusion": "occlusion", "Integrated Gradients": "ig", "Grad-CAM": "gradcam"}[panel]


def point(rows, method, m):
    d = rows[(rows.method == method) & (rows.dup == "exact") & (rows.kind == "target") & (rows.m == m)]
    d = d[d.I_ratio > 0]
    x = (d.per_copy_ratio / d.I_ratio).median()
    y = (d.conservation / d.I_ratio).median()
    return x, y, len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", default="results/e1")
    ap.add_argument("--r3d18-loo-drop", required=True,
                    help="dir whose r3d_18_rows.csv contains loo_drop on the first 120 videos")
    ap.add_argument("--out", default="paper/figures/fig_bias_map.pdf")
    args = ap.parse_args()

    rows = {p: pd.read_csv(Path(args.e1) / f"{p}_rows.csv") for p, *_ in MODELS}
    r18_drop = pd.read_csv(Path(args.r3d18_loo_drop) / "r3d_18_rows.csv")

    plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "serif"})
    fig, axes = plt.subplots(1, len(PANELS), figsize=(11.0, 2.75), sharex=True, sharey=True)
    lo, hi = FLOOR * 0.6, 40
    table = []
    for ax, panel in zip(axes, PANELS):
        ax.axvline(1, color=GRID, lw=0.8, zorder=1)
        ax.axhline(1, color=GRID, lw=0.8, zorder=1)
        ax.plot([1], [1], marker="*", ms=9, color=INK, zorder=4, clip_on=False)
        for prefix, label, m, shapley, marker in MODELS:
            method = method_for(panel, prefix, shapley)
            src = r18_drop if (prefix == "r3d_18" and method == "loo_drop") else rows[prefix]
            x, y, n = point(src, method, m)
            table.append((panel, label, method, m, n, x, y))
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            xc, yc = max(x, FLOOR), max(y, FLOOR)
            freeze_loo = method == "loo_freeze"   # partly fixed by the operator
            ax.plot([xc], [yc], marker=marker, ms=5.5, mec="white", mew=0.9,
                    mfc=MUTED if freeze_loo else BLUE, ls="none", zorder=5)
        quad = dict(fontsize=6.3, color=MUTED, transform=ax.transAxes)
        ax.text(0.03, 0.97, "both", ha="left", va="top", **quad)
        ax.text(0.97, 0.97, "inflation", ha="right", va="top", **quad)
        ax.text(0.03, 0.03, "dilution", ha="left", va="bottom", **quad)
        ax.text(0.97, 0.03, "impossible", ha="right", va="bottom", style="italic", **quad)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, 3)
        ax.set_ylim(lo, hi)
        ax.set_title(panel, fontsize=8, color=INK)
        ax.set_xticks([0.01, 0.1, 1])
        ax.set_xticklabels(["$\\leq$0.01", "0.1", "1"])
        ax.set_yticks([0.01, 0.1, 1, 10])
        ax.set_yticklabels(["$\\leq$0.01", "0.1", "1", "10"])
        ax.tick_params(length=2, width=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("score of one copy / model's change", fontsize=7)
    axes[0].set_ylabel("summed score / model's change")
    handles = [plt.Line2D([], [], marker=mk, ls="none", ms=5.5, mfc=BLUE, mec="white", mew=0.9,
                          label=lab) for _, lab, _, _, mk in MODELS]
    handles += [plt.Line2D([], [], marker="v", ls="none", ms=5.5, mfc=MUTED, mec="white", mew=0.9,
                           label="TRN, leave-one-out with freezing"),
                plt.Line2D([], [], marker="*", ls="none", ms=8, color=INK,
                           label="frame-count invariant")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, -0.02), handletextpad=0.3, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.6)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(str(Path(args.out).with_suffix(".png")), dpi=200, bbox_inches="tight")

    print(f"{'panel':<22}{'model':<9}{'method':<16}{'m':>3}{'n':>5}{'per-frame':>11}{'total':>9}")
    for panel, label, method, m, n, x, y in table:
        print(f"{panel:<22}{label:<9}{method:<16}{m:>3}{n:>5}{x:>11.3f}{y:>9.3f}")


if __name__ == "__main__":
    main()
