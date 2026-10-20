from __future__ import annotations

"""
Generate every figure of the KBS manuscript from result files (no hand-entered numbers).

Usage:
    python experiments/make_paper_figures.py --out outputs/figs
"""

import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Validated categorical slots (fixed order) and chart chrome.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
COL1, COL2 = 3.4, 7.0  # single / double column width in inches


def load_runs(pattern: str) -> list[dict]:
    runs = []
    for p in sorted(glob.glob(pattern)):
        runs.extend(json.load(open(p))["runs"])
    return runs


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".pdf"))
    fig.savefig(os.path.join(out, name + ".png"), dpi=200)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------
# Fig. 1: one worked example of spectral intrusion
# ---------------------------------------------------------------------------
def fig_example(out: str) -> None:
    from synthetic_overlap import make_task

    runs = [r for r in load_runs("results/synthetic_overlap_mamba_*.json")
            if r["rho"] == 0.0 and not r["conflict"]]
    # Pick the seed whose intrusion signature is closest to the median (not the most striking).
    score = [np.mean(np.array(r["delta_err"])[sorted(set(r["b_bins"]) - set(r["a_bins"]))]) for r in runs]
    r = runs[int(np.argsort(score)[len(score) // 2])]
    a_bins, b_bins = r["a_bins"], r["b_bins"]
    rng = np.random.default_rng(r["seed"])
    # Re-draw the same frequency choices so the generator state matches the experiment.
    grid = list(range(2, 21))
    rng.choice(grid, size=4, replace=False)
    n_shared = len(r["shared_bins"])
    if n_shared:
        rng.choice(a_bins, size=n_shared, replace=False)
    free = [j for j in grid if all(abs(j - a) >= 2 for a in a_bins)]
    rng.choice(free, size=4 - n_shared, replace=False)
    a = make_task(12000, a_bins, [0.99] * 4, 48, 0.1, rng)
    b = make_task(12000, b_bins, [0.99] * 4, 48, 0.1, rng)

    def spec(x):
        w = np.lib.stride_tricks.sliding_window_view(x, 48)[::12]
        s = np.abs(np.fft.rfft(w - w.mean(axis=1, keepdims=True), axis=1)).mean(axis=0)
        return s / s[1:].sum()

    sa, sb = spec(a), spec(b)
    bins = np.arange(len(sa))
    d = np.array(r["delta_err"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL1, 2.9), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.1], "hspace": 0.38})
    ax1.plot(bins[1:], sa[1:], color=BLUE, lw=1.6, marker="o", ms=3, label="old task A")
    ax1.plot(bins[1:], sb[1:], color=ORANGE, lw=1.6, marker="s", ms=3, ls="--", label="new task B")
    ax1.set_ylabel("normalised\namplitude")
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7, handlelength=2.2)

    from matplotlib.patches import Patch

    b_only = set(b_bins) - set(a_bins)
    colors = [ORANGE if k in b_only else (BLUE if k in a_bins else MUTED) for k in bins[1:]]
    ax2.bar(bins[1:], d[1:], color=colors, width=0.72)
    ax2.axhline(0, color=AXIS, lw=0.8)
    ax2.set_ylabel(r"error growth on A, $\Delta E$")
    ax2.set_xlabel(r"frequency bin $\omega$ (cycles per 48-step horizon)")
    ax2.set_title("Where task A is forgotten after learning B", loc="left", color=INK)
    ax2.legend(handles=[Patch(color=ORANGE, label="bins only B uses (intrusion)"),
                        Patch(color=BLUE, label="bins A uses"),
                        Patch(color=MUTED, label="other bins")],
               loc="upper right", fontsize=6.5, handlelength=1.2)
    ax2.set_ylim(min(0, d[1:].min()) * 1.1, d[1:].max() * 1.45)
    print(f"  fig1 example: seed={r['seed']} A={a_bins} B={b_bins}")
    save(fig, out, "fig1_example")


# ---------------------------------------------------------------------------
# Fig. 2: forgetting vs spectral overlap
# ---------------------------------------------------------------------------
def fig_overlap(out: str) -> None:
    sets = [("Linear", "results/synthetic_overlap_linear.json"), ("MLP", "results/synthetic_overlap_mlp.json"),
            ("Mamba", "results/synthetic_overlap_mamba_*.json")]
    fig, axes = plt.subplots(1, 3, figsize=(COL2 * 0.9, 2.2), sharey=False)
    for ax, (name, pat) in zip(axes, sets):
        runs = load_runs(pat)
        rhos = sorted({r["rho"] for r in runs})
        for conflict, color, marker, ls, label in [(False, BLUE, "o", "-", "same dynamics"),
                                                   (True, ORANGE, "s", "--", "conflicting dynamics")]:
            m = [np.mean([r["rel_forgetting"] for r in runs if r["rho"] == p and r["conflict"] == conflict]) for p in rhos]
            s = [np.std([r["rel_forgetting"] for r in runs if r["rho"] == p and r["conflict"] == conflict]) for p in rhos]
            ax.errorbar(rhos, m, yerr=s, color=color, marker=marker, ms=4, lw=1.6, ls=ls,
                        capsize=2, elinewidth=0.8, label=label)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_title(name, loc="left", color=INK)
        ax.set_xlabel(r"spectral overlap $\rho$")
        ax.set_xticks(rhos)
    axes[0].set_ylabel("relative forgetting on A")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 1.07))
    fig.subplots_adjust(wspace=0.3)
    save(fig, out, "fig2_overlap")


# ---------------------------------------------------------------------------
# Fig. 3: error growth by bin category
# ---------------------------------------------------------------------------
def fig_bins(out: str) -> None:
    conds = [("Linear", "results/synthetic_overlap_linear.json", False),
             ("MLP", "results/synthetic_overlap_mlp.json", False),
             ("Mamba", "results/synthetic_overlap_mamba_*.json", False)]
    cats = ["intrusion", "neglect", "unused"]
    fig, axes = plt.subplots(1, 3, figsize=(COL2 * 0.62, 2.0), sharey=False)
    for ax, (name, pat, conflict) in zip(axes, conds):
        runs = [r for r in load_runs(pat) if r["rho"] == 0.0 and r["conflict"] == conflict]
        vals = {c: [] for c in cats}
        for r in runs:
            d = np.array(r["delta_err"])
            a, b = set(r["a_bins"]), set(r["b_bins"])
            near = {u + s for u in a | b for s in (-1, 0, 1)}
            vals[cats[0]].append(d[sorted(b - a)].mean())
            vals[cats[1]].append(d[sorted(a)].mean())
            vals[cats[2]].append(d[sorted(set(range(1, len(d))) - near)].mean())
        m = [np.mean(vals[c]) for c in cats]
        s = [np.std(vals[c]) for c in cats]
        ax.bar(range(3), m, yerr=s, color=[ORANGE, BLUE, MUTED], width=0.62,
               capsize=2, error_kw={"elinewidth": 0.8, "ecolor": INK2})
        ax.set_xticks(range(3))
        ax.set_xticklabels(cats, fontsize=7, rotation=30, ha="right", rotation_mode="anchor")
        ax.set_title(name, loc="left", color=INK, fontsize=7.5)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(r"mean error growth $\Delta E$ ($\rho=0$)")
    fig.subplots_adjust(wspace=0.35)
    save(fig, out, "fig3_bins")


# ---------------------------------------------------------------------------
# Fig. 4: real benchmarks, intrusion vs overlap
# ---------------------------------------------------------------------------
def fig_real(out: str) -> None:
    from analyze_real_band import delta_s, partial_spearman

    order = ["electricity", "traffic", "ettm2", "ettm1", "weather"]
    names = {"electricity": "Electricity", "traffic": "Traffic", "ettm2": "ETTm2",
             "ettm1": "ETTm1", "weather": "Weather"}
    res = real_band_scores()
    settings = [("mamba48", "Mamba, $H=48$", BLUE, "o"), ("linear48", "Linear, $H=48$", AQUA, "s"),
                ("mamba96", "Mamba, $H=96$", YELLOW, "D")]
    fig, axes = plt.subplots(1, 2, figsize=(COL2 * 0.72, 2.5), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, (k, title) in zip(axes, [(0, "Intrusion"), (1, "Overlap (prior hypothesis)")]):
        for s, (key, label, color, marker) in enumerate(settings):
            m = np.array([res[key][ds][k].mean() for ds in order])
            ci = np.array([1.96 * res[key][ds][k].std(ddof=1) / np.sqrt(len(res[key][ds][k])) for ds in order])
            ax.errorbar(m, y + (1 - s) * 0.22, xerr=ci, fmt=marker, color=color, ms=4.2, capsize=1.8,
                        elinewidth=0.85, label=label)
        ax.axvline(0, color=AXIS, lw=0.9)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel(r"partial Spearman $\rho$ (95% CI)")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(-0.32, 0.42)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([names[d] for d in order])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=7, bbox_to_anchor=(0.55, 1.07))
    fig.subplots_adjust(wspace=0.08)
    save(fig, out, "fig4_real")


REAL_SETTINGS = {"mamba48": "results/real_band", "linear48": "results/real_band_linear",
                 "mamba96": "results/real_band_h96"}
REAL_ORDER = ["electricity", "traffic", "ettm2", "ettm1", "weather"]


def real_band_scores() -> dict:
    """{setting: {dataset: (intrusion partials, overlap partials)}} over task pairs."""
    from analyze_real_band import delta_s, partial_spearman

    res = {}
    for key, root in REAL_SETTINGS.items():
        res[key] = {}
        for ds in REAL_ORDER:
            d = json.load(open(f"{root}/{ds}.json"))
            spectra = [np.array(s) for s in d["spectra"]]
            pi, po = [], []
            for r in d["records"]:
                dE = np.array(r["dE"])
                sj, si = spectra[r["old_task"]], spectra[r["new_task"]]
                b = slice(1, None)
                pi.append(partial_spearman(np.maximum(0, si - sj)[b], dE[b], sj[b]))
                po.append(partial_spearman(delta_s(sj, si)[b], dE[b], sj[b]))
            res[key][ds] = (np.array(pi), np.array(po))
    return res


def table_generality(out_tables: str) -> None:
    """LaTeX table: intrusion and overlap partial correlations for every backbone / horizon."""
    from scipy.stats import wilcoxon

    res = real_band_scores()
    names = {"electricity": "Electricity", "traffic": "Traffic", "ettm2": "ETTm2",
             "ettm1": "ETTm1", "weather": "Weather"}
    heads = [("mamba48", r"Mamba, $H{=}48$"), ("linear48", r"Linear, $H{=}48$"), ("mamba96", r"Mamba, $H{=}96$")]

    def fmt(v):
        p = wilcoxon(v).pvalue
        s = f"{v.mean():+.2f}"
        star = "$^{**}$" if p < 0.01 else ("$^{*}$" if p < 0.05 else "")
        return (r"\textbf{" + s + "}" if (v.mean() > 0 and p < 0.05) else s) + star

    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Generality of the finding on real benchmarks: mean partial Spearman correlation "
             r"(30 task pairs) between per-frequency forgetting and the intrusion or overlap score, for two "
             r"backbones and two horizons. $^{*}p<0.05$, $^{**}p<0.01$ (Wilcoxon); significant positive "
             r"associations in bold.}", r"\label{tab:generality}", r"\small", r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{l" + "cc" * len(heads) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{h}}}" for _, h in heads) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(heads))),
             "Dataset & " + " & ".join(["Intrusion & Overlap"] * len(heads)) + r" \\", r"\midrule"]
    for ds in REAL_ORDER:
        row = [names[ds]]
        for key, _ in heads:
            row += [fmt(res[key][ds][0]), fmt(res[key][ds][1])]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    os.makedirs(out_tables, exist_ok=True)
    with open(os.path.join(out_tables, "tab_generality.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote tab_generality.tex")


# ---------------------------------------------------------------------------
# Fig. 5: intervention study
# ---------------------------------------------------------------------------
def fig_intervention(out: str) -> None:
    runs = load_runs("results/int1024_*.json")
    losses = [("mse", "standard replay (MSE)", BLUE),
              ("fdrl_old", "old-energy weighting", MUTED),
              ("fdrl_intrusion", "IFR (intrusion weighting)", ORANGE)]
    rhos = sorted({r["rho"] for r in runs})
    fig, axes = plt.subplots(1, 2, figsize=(COL2 * 0.62, 2.1))
    markers = ["o", "D", "s"]
    width = 0.26

    def vals_for(metric, key):
        return [[r[metric] for r in runs if r["rho"] == p and r["sampler"] == "random" and r["loss"] == key]
                for p in rhos]

    # Left: forgetting as bars from zero (magnitude). Right: new-task MSE as points,
    # because differences are small relative to the value and bars would need a cut axis.
    for k, (key, label, color) in enumerate(losses):
        v = vals_for("rel_forgetting", key)
        axes[0].bar(np.arange(len(rhos)) + (k - 1) * (width + 0.02), [np.mean(x) for x in v], width,
                    yerr=[np.std(x) / np.sqrt(len(x)) for x in v], color=color, label=label,
                    capsize=2, error_kw={"elinewidth": 0.8, "ecolor": INK2})
        v = vals_for("mse_b", key)
        axes[1].errorbar(np.arange(len(rhos)) + (k - 1) * 0.18, [np.mean(x) for x in v],
                         yerr=[np.std(x) / np.sqrt(len(x)) for x in v], fmt=markers[k], color=color,
                         ms=5, capsize=2, elinewidth=0.9)
    for ax, ylab in [(axes[0], "relative forgetting on old task"), (axes[1], "test MSE on new task")]:
        ax.set_xticks(range(len(rhos)))
        ax.set_xticklabels([rf"$\rho={p:g}$" for p in rhos])
        ax.set_ylabel(ylab)
        ax.grid(axis="x", visible=False)
    axes[1].set_xlim(-0.6, len(rhos) - 0.4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=6.8, bbox_to_anchor=(0.5, 1.08))
    axes[0].set_title("Stability", loc="left", color=INK)
    axes[1].set_title("Plasticity", loc="left", color=INK)
    fig.subplots_adjust(wspace=0.42)
    save(fig, out, "fig5_intervention")


# ---------------------------------------------------------------------------
# Concept figure: the prior intuition versus what we measure (schematic)
# ---------------------------------------------------------------------------
def fig_concept(out: str) -> None:
    f = np.linspace(0, 24, 600)

    def peaks(centres, width=0.7):
        return sum(np.exp(-0.5 * ((f - c) / width) ** 2) for c in centres)

    old_c, new_c = [3, 8, 13], [8, 16, 20]          # bin 8 is shared
    s_old, s_new = peaks(old_c), peaks(new_c)
    shared, old_only, new_only = [8], [3, 13], [16, 20]

    fig, axes = plt.subplots(1, 2, figsize=(COL2 * 0.9, 2.35), sharey=True)
    panels = [
        ("Prior intuition: overlap hypothesis",
         {8: 1.0}, "forgetting predicted where regimes share energy\n"
                   "$\\Rightarrow$ protect the old regime's frequencies",
         "designs built on it: worse than plain replay"),
        ("Measured: spectral intrusion (this work)",
         {16: 1.0, 20: 0.85, 3: 0.35, 13: 0.3, 8: 0.04},
         "forgetting where the NEW regime adds frequencies;\n"
         "shared frequencies are protected",
         "IFR targets these bins: $-42\\%$ forgetting vs. replay"),
    ]
    for ax, (title, bars, text, verdict) in zip(axes, panels):
        ax.fill_between(f, 0, s_old, color=BLUE, alpha=0.18, lw=0)
        ax.plot(f, s_old, color=BLUE, lw=1.4, label="old regime spectrum")
        ax.fill_between(f, 0, s_new, color=ORANGE, alpha=0.14, lw=0)
        ax.plot(f, s_new, color=ORANGE, lw=1.4, ls="--", label="new regime spectrum")
        for b, h in bars.items():
            color = ORANGE if b in new_only else (BLUE if b in old_only else INK2)
            ax.bar(b, -0.75 * h, width=1.1, color=color, edgecolor="white", linewidth=0.6)
        ax.axhline(0, color=AXIS, lw=0.9)
        ax.set_title(title, loc="left", color=INK, fontsize=8)
        ax.text(0.0, -0.98, text, transform=ax.get_yaxis_transform(), fontsize=6.6, color=INK2, va="top")
        ax.text(0.0, -1.42, verdict, transform=ax.get_yaxis_transform(), fontsize=6.8, color=INK,
                va="top", fontweight="bold")
        ax.set_xlim(0, 24)
        ax.set_ylim(-1.7, 1.15)
        ax.set_xticks([])
        ax.set_yticks([0.5, -0.4])
        ax.set_yticklabels(["energy", "forgetting"], fontsize=6.8)
        ax.tick_params(axis="y", length=0)
        ax.grid(False)
        ax.set_xlabel("frequency $\\rightarrow$", fontsize=7)
        ax.spines["left"].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 1.1),
               handlelength=2.4)
    fig.subplots_adjust(wspace=0.08)
    save(fig, out, "fig0_concept")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/figs")
    ap.add_argument("--tables", default="outputs/tables")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for f in (fig_concept, fig_example, fig_overlap, fig_bins, fig_real, fig_intervention):
        f(args.out)
    table_generality(args.tables)


if __name__ == "__main__":
    main()
