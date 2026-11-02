"""
fig_e1_design.py -- E1: the reference, target and control clips of the two designs.

Schematic (no data). Letters are contents; the target B is filled. Layouts follow
motivation/e1_common.py: insertion_layout / spread_control_layout (insertion, m = 3) and
realloc_layouts (reallocation, m = 4, losers A and D, recipient C = the content the model
depends on least).

    python motivation/fig_e1_design.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, MUTED, EDGE = "#0b0b0b", "#52514e", "#8a8984"
TARGET, EXTRA = "#2a78d6", "#eb6834"   # categorical slots 1 and 2 of fig_dilution.py

# (row label, contents, which slots are extra copies relative to the reference)
INSERTION = [
    ("reference", "ABCDE", ""),
    ("target", "ABBBCDE", "  ++   "),
    ("control", "AABCDDE", " +   + "),
]
REALLOCATION = [
    ("reference", "AABBCCDD", ""),
    ("target", "ABBBBCCD", "  ++    "),
    ("control", "ABBCCCCD", "    ++  "),
]


def draw(ax, rows, title, note):
    w, gap = 0.82, 0.18
    for r, (label, contents, extra) in enumerate(rows):
        y = -r
        ax.text(-0.35, y, label, ha="right", va="center", fontsize=7.5, color=INK)
        for i, c in enumerate(contents):
            is_extra = i < len(extra) and extra[i] == "+"
            fill = TARGET if c == "B" else "white"
            edge = EXTRA if is_extra else EDGE
            ax.add_patch(FancyBboxPatch((i * (w + gap), y - w / 2), w, w,
                                        boxstyle="round,pad=0,rounding_size=0.08",
                                        fc=fill, ec=edge, lw=1.6 if is_extra else 0.7))
            ax.text(i * (w + gap) + w / 2, y, c, ha="center", va="center", fontsize=7.5,
                    color="white" if c == "B" else INK)
    ax.set_title(title, fontsize=8, color=INK, loc="left", x=-0.02)
    ax.text(0, -len(rows) + 0.25, note, fontsize=6.6, color=MUTED, va="top")
    ax.set_xlim(-0.4, 8 * (w + gap))
    ax.set_ylim(-len(rows) - 0.35, 0.6)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/figures/fig_e1_design.pdf")
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "serif"})
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 1.55))
    draw(axes[0], INSERTION, "Insertion (MC3-18, V-JEPA2)",
         "each content once; $m = 3$ copies of B")
    draw(axes[1], REALLOCATION, "Reallocation (R3D-50, R3D-18, TRN)",
         "each content twice; $m = 4$ copies of B")
    fig.subplots_adjust(left=0.1, right=0.99, top=0.86, bottom=0.12, wspace=0.35)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(str(Path(args.out).with_suffix(".png")), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
