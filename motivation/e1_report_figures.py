"""
e1_report_figures.py -- figures for results/E1_report.md, from results/e1/<model>_summary.csv and
<model>_beta.csv (written by e1_metrics.py attr).

    python motivation/e1_report_figures.py          # needs only numpy + matplotlib

fig5_e1_credit.png  per-copy ratio (top) and total ratio (bottom) of the target, against the number
                    of copies relative to the reference, for each model and method; with the model's
                    own reliance (I_t ratio) and an even split for reference.
fig6_e1_beta.png    dilution slope beta of the target vs its control, per model and method.

In tc_env, numpy's MKL crashes inside matplotlib; use another environment (e.g. torch_plot).
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
E1 = ROOT / "results" / "e1"
OUT = ROOT / "results" / "figures"

# model -> (label, m_ref, Shapley variant, leave-one-out variant)
MODELS = {
    "r3d":          ("R3D-50 (UCF101)", 2, "shapley_freeze", "loo_freeze"),
    "r3d_18":       ("R3D-18 (UCF101)", 2, "shapley_freeze", "loo_freeze"),
    "mc3_18":       ("MC3-18 (UCF101)", 1, "shapley_drop", "loo_drop"),
    "vjepa2":       ("V-JEPA2 (SSv2)", 1, "shapley_drop", "loo_drop"),
    "trn_official": ("TRN (SSv2)", 2, "shapley_freeze", "loo_freeze"),
}
STYLE = {  # method key -> (label, colour, marker)
    "shapley": ("Shapley", "#1f77b4", "o"),
    "playfair": ("Play Fair", "#17becf", "D"),
    "ig": ("Integrated Gradients", "#2ca02c", "s"),
    "loo": ("Leave-one-out", "#d62728", "v"),
    "occlusion": ("Occlusion", "#9467bd", "^"),
    "gradcam": ("Grad-CAM", "#ff7f0e", "P"),
}


def load(model):
    rows = list(csv.DictReader(open(E1 / f"{model}_summary.csv")))
    beta = list(csv.DictReader(open(E1 / f"{model}_beta.csv")))
    return rows, beta


def method_map(model):
    _, _, sh, loo = MODELS[model]
    return {"shapley": sh, "playfair": "playfair", "ig": "ig", "loo": loo,
            "occlusion": "occlusion", "gradcam": "gradcam"}


def series(rows, method, kind, col):
    pts = sorted((int(r["m"]), float(r[col])) for r in rows
                 if r["method"] == method and r["dup"] == "exact" and r["kind"] == kind)
    return pts


def fig_credit():
    fig, axes = plt.subplots(2, len(MODELS), figsize=(3.2 * len(MODELS), 6.4), sharey="row")
    for j, (model, (label, m_ref, sh, _)) in enumerate(MODELS.items()):
        rows, _ = load(model)
        mm = method_map(model)
        I = series(rows, sh, "target", "median_I_ratio")
        xs = [1] + [m / m_ref for m, _ in I]
        for i, col in enumerate(("median_per_copy_ratio", "median_conservation")):
            ax = axes[i, j]
            # references: even split, and the model's own reliance
            if i == 0:
                ax.plot(xs, [1 / x for x in xs], ":", color="grey", label="credit split evenly")
                ax.plot(xs, [1] + [v / (m / m_ref) for m, v in I], "--", color="black", lw=1.6,
                        label="model's reliance, shared by the copies")
            else:
                ax.plot(xs, [1] * len(xs), ":", color="grey")
                ax.plot(xs, [1] + [v for _, v in I], "--", color="black", lw=1.6,
                        label="model's reliance (I_t ratio)")
            for key, (mlabel, colour, marker) in STYLE.items():
                pts = series(rows, mm[key], "target", col)
                if not pts or (key == "playfair" and model == "r3d"):  # r3d drop: length-confounded
                    continue
                ax.plot([1] + [m / m_ref for m, _ in pts], [1] + [v for _, v in pts], marker=marker,
                        color=colour, label=mlabel, ms=5, lw=1.2,
                        ls="-" if key != "playfair" else (0, (1, 1)))
            ax.set_xscale("log")
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{x:g}" for x in xs])
            ax.minorticks_off()
            ax.axhline(0, color="#ddd", lw=0.8, zorder=0)
            if i == 0:
                ax.set_title(label, fontsize=10)
            ax.set_xlabel("copies, relative to the reference" if i == 1 else "")
        axes[0, 0].set_ylabel("credit of one copy\n(relative to the reference)")
        axes[1, 0].set_ylabel("credit of all copies\n(relative to the reference)")
    h, lab = [], []
    for ax in axes.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in lab:
                h.append(hh)
                lab.append(ll)
    fig.legend(h, lab, loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "fig5_e1_credit.png", dpi=160)
    print("wrote", OUT / "fig5_e1_credit.png")


def fig_beta():
    fig, ax = plt.subplots(figsize=(10, 4.8))
    keys = list(STYLE)
    width = 0.8 / len(keys)
    for j, model in enumerate(MODELS):
        _, beta = load(model)
        mm = method_map(model)
        for k, key in enumerate(keys):
            if key == "playfair" and model == "r3d":
                continue
            b = {r["kind"].startswith("control"): r for r in beta
                 if r["method"] == mm[key] and r["dup"] == "exact"}
            if True not in b or False not in b:
                continue
            x = j + (k - (len(keys) - 1) / 2) * width
            t, c = b[False], b[True]
            mlabel, colour, marker = STYLE[key]
            ax.errorbar(x, max(float(t["median_beta"]), -1.6),
                        yerr=[[max(0, float(t["median_beta"]) - max(float(t["ci_lo"]), -1.6))],
                              [float(t["ci_hi"]) - float(t["median_beta"])]],
                        fmt=marker, color=colour, ms=6, capsize=2)
            ax.plot(x, float(c["median_beta"]), marker, mfc="white", color=colour, ms=6)
            if float(t["median_beta"]) < -1.6:
                ax.annotate(f"{float(t['median_beta']):.1f}", (x, -1.6), textcoords="offset points",
                            xytext=(0, -12), ha="center", fontsize=7, color=colour)
    ax.axhline(0, color="grey", lw=0.8)
    ax.axhline(-1, color="grey", lw=0.8, ls=":")
    ax.text(-0.45, 0.02, "β = 0: each copy keeps its credit", fontsize=8, color="grey", va="bottom")
    ax.text(-0.45, -0.98, "β = −1: credit split evenly", fontsize=8, color="grey", va="bottom")
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([v[0] for v in MODELS.values()])
    ax.set_xlim(-0.5, len(MODELS) - 0.5)
    ax.set_ylim(-1.85, 0.45)
    ax.set_ylabel("dilution slope β")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="", marker=mk, color=c, ms=6, label=lb) for lb, c, mk in STYLE.values()]
    handles += [Line2D([], [], ls="", marker="o", color="black", ms=6, label="target (filled, 95% range)"),
                Line2D([], [], ls="", marker="o", mfc="white", color="black", ms=6, label="control (open)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "fig6_e1_beta.png", dpi=160)
    print("wrote", OUT / "fig6_e1_beta.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_credit()
    fig_beta()
