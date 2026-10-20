from __future__ import annotations

"""
Summarise synthetic_overlap.py results.

For each backbone x condition x overlap rho, reports mean +- std over seeds of:
    rel_forgetting   (MSE_after - MSE_before) / MSE_before on task A
    dE_shared        error-energy growth at frequencies shared by A and B
    dE_a_only        error-energy growth at frequencies only A uses
    dE_other         error-energy growth at frequencies A does not use
    spearman         rank correlation of Delta_S(f) with per-bin error growth
Also tests, pooled over rho in (0, 1), whether dE_shared > dE_a_only within runs
(paired Wilcoxon), which is the direct prediction of the hypothesis.

Usage:
    python experiments/analyze_synthetic.py results/synthetic_overlap_*.json
"""

import json
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, wilcoxon


def fmt(vals: list[float]) -> str:
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    if not vals:
        return "      -       "
    return f"{np.mean(vals):+.3f}+-{np.std(vals):.3f}"


def main(paths: list[str]) -> None:
    runs = []
    for p in paths:
        with open(p) as f:
            runs.extend(json.load(f)["runs"])

    groups = defaultdict(list)
    for r in runs:
        groups[(r["backbone"], r["conflict"], r["rho"])].append(r)

    print(f"{'backbone':8} {'conflict':8} {'rho':>4}  {'rel_forget':>14} {'dE_shared':>14} "
          f"{'dE_a_only':>14} {'dE_other':>14} {'spearman':>14}  n")
    for key in sorted(groups):
        g = groups[key]
        print(f"{key[0]:8} {str(key[1]):8} {key[2]:4.2f}  "
              f"{fmt([r['rel_forgetting'] for r in g]):>14} "
              f"{fmt([r['derr_shared'] for r in g]):>14} "
              f"{fmt([r['derr_a_only'] for r in g]):>14} "
              f"{fmt([r['derr_other'] for r in g]):>14} "
              f"{fmt([r['spearman_deltaS_vs_derr'] for r in g]):>14}  {len(g)}")

    # Split error growth at bins A does not use into B-only bins (possible
    # intrusion of B's frequencies into A's forecasts) and bins neither task
    # uses (excluding +-1 neighbours of any used bin to avoid leakage).
    print(f"\n{'backbone':8} {'conflict':8} {'rho':>4}  {'dE_B_only':>14} {'dE_neither':>14}")
    for key in sorted(groups):
        b_only, neither = [], []
        for r in groups[key]:
            d = np.array(r["delta_err"])
            a, b = set(r["a_bins"]), set(r["b_bins"])
            near = {u + s for u in a | b for s in (-1, 0, 1)}
            if b - a:
                b_only.append(float(d[sorted(b - a)].mean()))
            free = sorted(set(range(1, len(d))) - near)
            if free:
                neither.append(float(d[free].mean()))
        print(f"{key[0]:8} {str(key[1]):8} {key[2]:4.2f}  {fmt(b_only):>14} {fmt(neither):>14}")

    print("\nOverlap level vs forgetting (Spearman over all runs) and paired shared-vs-A-only test:")
    for backbone in sorted({r["backbone"] for r in runs}):
        for conflict in (True, False):
            sub = [r for r in runs if r["backbone"] == backbone and r["conflict"] == conflict]
            if not sub:
                continue
            rs = spearmanr([r["rho"] for r in sub], [r["rel_forgetting"] for r in sub])
            mixed = [r for r in sub if 0 < r["rho"] < 1]
            diffs = [r["derr_shared"] - r["derr_a_only"] for r in mixed]
            line = (f"  {backbone:6} conflict={conflict!s:5}  rho~forgetting: "
                    f"r={rs.statistic:+.3f} (p={rs.pvalue:.3g})")
            if len(diffs) >= 5:
                w = wilcoxon(diffs, alternative="greater")
                line += (f" | shared>a_only in {sum(d > 0 for d in diffs)}/{len(diffs)} runs, "
                         f"Wilcoxon p={w.pvalue:.3g}")
            print(line)


if __name__ == "__main__":
    main(sys.argv[1:])
