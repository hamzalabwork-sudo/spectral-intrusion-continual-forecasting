from __future__ import annotations

"""
Sensitivity of IFR to kappa and the replay weight lambda (validation split only).

Reads results/tuning/crossdomain_<order>/{er_w<lambda>, ifr_w<lambda>_k<kappa>}/ and
prints validation MSE and forgetting (mean over seeds), plus a LaTeX table fragment.

Usage:
    python experiments/analyze_sensitivity.py --order A --tex outputs/tables/tab_sensitivity.tex
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_benchmark import cl_metrics


def collect(order: str) -> dict:
    out = {}
    for d in glob.glob(f"results/tuning/crossdomain_{order}/*"):
        name = os.path.basename(d)
        m = re.match(r"(er|ifr)_w([\d.]+)(?:_k([\d.]+))?$", name)
        if not m:
            continue
        kappa = 0.0 if m.group(1) == "er" else float(m.group(3))
        runs = [cl_metrics(e["results_matrix"]) for p in glob.glob(d + "/continual_results_*.json")
                for e in json.load(open(p))]
        if runs:
            out[(float(m.group(2)), kappa)] = runs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", default="A")
    ap.add_argument("--tex", default=None)
    args = ap.parse_args()
    res = collect(args.order)
    lams = sorted({k[0] for k in res})
    kaps = sorted({k[1] for k in res})
    print(f"Validation, cross-domain order {args.order}: MSE (forgetting), mean over seeds")
    print("lambda \\ kappa " + "".join(f"{k:>18g}" for k in kaps))
    for lam in lams:
        cells = []
        for k in kaps:
            r = res.get((lam, k))
            cells.append(f"{np.mean([x['avg_mse'] for x in r]):.4f} ({np.mean([x['avg_forgetting'] for x in r]):.3f})"
                         if r else "-")
        print(f"{lam:<15g}" + "".join(f"{c:>18}" for c in cells))

    if args.tex:
        best = min(res, key=lambda k: np.mean([x["avg_mse"] for x in res[k]]))
        lines = [r"\begin{table}[t]", r"\centering",
                 r"\caption{Sensitivity to $\kappa$ and the replay weight $\lambda$: validation MSE (forgetting in "
                 rf"parentheses) on the cross-domain benchmark, order {args.order}, mean over "
                 rf"{len(next(iter(res.values())))} seeds. $\kappa=0$ is ER. Best in bold.}}",
                 r"\label{tab:sensitivity}", r"\small",
                 r"\begin{tabular}{l" + "c" * len(kaps) + "}", r"\toprule",
                 r"$\lambda$ & " + " & ".join(rf"$\kappa={k:g}$" for k in kaps) + r" \\", r"\midrule"]
        for lam in lams:
            row = [f"{lam:g}"]
            for k in kaps:
                r = res.get((lam, k))
                if not r:
                    row.append("--")
                    continue
                s = f"{np.mean([x['avg_mse'] for x in r]):.3f} ({np.mean([x['avg_forgetting'] for x in r]):.3f})"
                row.append(r"\textbf{" + s + "}" if (lam, k) == best else s)
            lines.append(" & ".join(row) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        with open(args.tex, "w") as f:
            f.write("\n".join(lines) + "\n")
        print("wrote", args.tex)


if __name__ == "__main__":
    main()
