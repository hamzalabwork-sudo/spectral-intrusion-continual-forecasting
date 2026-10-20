from __future__ import annotations

"""
Summarise synthetic_intervention.py results: forgetting on task A and MSE on
task B for each sampler x loss, per overlap level, with paired Wilcoxon tests
(over seeds) against plain random replay with time-domain MSE.

Usage:
    python experiments/analyze_intervention.py results/synthetic_intervention_*.json
"""

import json
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon


def main(paths: list[str]) -> None:
    runs = []
    for p in paths:
        runs.extend(json.load(open(p))["runs"])

    for rho in sorted({r["rho"] for r in runs}):
        sub = [r for r in runs if r["rho"] == rho]
        by = defaultdict(dict)
        for r in sub:
            by[(r["sampler"], r["loss"])][r["seed"]] = r
        ft = np.mean([r["finetune_rel_forgetting"] for r in sub])
        print(f"\nrho={rho:.2f}   finetune (no replay) rel. forgetting = {ft:+.3f}")
        print(f"  {'sampler':10} {'loss':14} {'rel_forgetting':>16} {'MSE_B':>16}   vs random+mse (forgetting)")
        ref = by[("random", "mse")]
        for key in sorted(by):
            g = by[key]
            f_vals = [g[s]["rel_forgetting"] for s in sorted(g)]
            b_vals = [g[s]["mse_b"] for s in sorted(g)]
            line = (f"  {key[0]:10} {key[1]:14} {np.mean(f_vals):+.3f}+-{np.std(f_vals):.3f}   "
                    f"{np.mean(b_vals):.4f}+-{np.std(b_vals):.4f}")
            if key != ("random", "mse"):
                seeds = sorted(set(g) & set(ref))
                diff = [g[s]["rel_forgetting"] - ref[s]["rel_forgetting"] for s in seeds]
                better = sum(d < 0 for d in diff)
                p = wilcoxon(diff).pvalue if len(diff) >= 5 and any(diff) else float("nan")
                line += f"   delta={np.mean(diff):+.3f}  better in {better}/{len(diff)}  p={p:.3g}"
            print(line)


if __name__ == "__main__":
    main(sys.argv[1:])
