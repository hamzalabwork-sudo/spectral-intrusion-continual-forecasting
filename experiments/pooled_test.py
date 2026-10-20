from __future__ import annotations

"""
Pooled paired comparison of IFR against plain replay (ER, same code path with kappa=0).

Per-dataset tests have few seeds; pooling the paired (benchmark, seed) differences and
counting benchmarks where IFR is better on average gives a test across the whole suite.

Usage:
    python experiments/pooled_test.py
"""

import glob
import os
import sys

import numpy as np
from scipy.stats import binomtest, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_benchmark import load


def merged(patterns):
    data = {}
    for pat in patterns:
        for ds, methods in load(glob.glob(pat)).items():
            for m, seeds in methods.items():
                data.setdefault(ds, {}).setdefault(m, {}).update(seeds)
    return data


def report(name, data, ref="ifr", base="ifr_uniform"):
    rel_mse, rel_fgt, bench_better = [], [], 0
    for ds, methods in sorted(data.items()):
        a, b = methods[ref], methods[base]
        seeds = sorted(set(a) & set(b))
        r = [(a[s]["avg_mse"] - b[s]["avg_mse"]) / b[s]["avg_mse"] for s in seeds]
        # Forgetting can be exactly zero, so compare it in absolute terms.
        f = [a[s]["avg_forgetting"] - b[s]["avg_forgetting"] for s in seeds]
        rel_mse += r
        rel_fgt += f
        bench_better += int(np.mean(r) < 0)
    w = wilcoxon(rel_mse, alternative="less").pvalue
    wf = wilcoxon(rel_fgt, alternative="less").pvalue
    sign = binomtest(bench_better, len(data), alternative="greater").pvalue
    print(f"{name}: {len(rel_mse)} paired runs | mean dMSE {100 * np.mean(rel_mse):+.2f}% "
          f"(median {100 * np.median(rel_mse):+.2f}%), IFR better in {sum(x < 0 for x in rel_mse)}/{len(rel_mse)} "
          f"runs, one-sided Wilcoxon p={w:.3g} | better on {bench_better}/{len(data)} benchmarks, "
          f"sign test p={sign:.3g} | forgetting lower in {sum(x < 0 for x in rel_fgt)}/{len(rel_fgt)} runs, "
          f"mean dForgetting {np.mean(rel_fgt):+.4f}, Wilcoxon p={wf:.3g}")


def versus_all(data, ref="ifr"):
    """IFR against every other method over all benchmarks: benchmarks won, mean relative
    MSE difference over paired runs, and a two-sided Wilcoxon test."""
    methods = sorted({m for ds in data.values() for m in ds} - {ref})
    print(f"\n{ref} vs each method (negative = {ref} lower MSE):")
    for m in methods:
        rel, won, n_b = [], 0, 0
        for ds, ms in data.items():
            if m not in ms or ref not in ms:
                continue
            seeds = sorted(set(ms[ref]) & set(ms[m]))
            if not seeds:
                continue
            r = [(ms[ref][s]["avg_mse"] - ms[m][s]["avg_mse"]) / ms[m][s]["avg_mse"] for s in seeds]
            rel += r
            n_b += 1
            won += int(np.mean([ms[ref][s]["avg_mse"] for s in seeds]) < np.mean([ms[m][s]["avg_mse"] for s in seeds]))
        if len(rel) >= 10:
            print(f"  vs {m:15} benchmarks won {won}/{n_b} | runs {len(rel):3d} | mean dMSE {100 * np.mean(rel):+6.2f}% "
                  f"| median {100 * np.median(rel):+6.2f}% | Wilcoxon p={wilcoxon(rel).pvalue:.3g}")


def main():
    within = merged(["results/pilot/*/continual_results_*.json", "results/main/*/continual_results_*.json"])
    within = {k: v for k, v in within.items() if "ifr" in v and "ifr_uniform" in v}
    cross = merged(["results/crossdomain/*/continual_results_*.json"])
    report("within-dataset", within)
    report("cross-domain", cross)
    report("all benchmarks", {**within, **cross})
    versus_all({**within, **cross})


if __name__ == "__main__":
    main()
