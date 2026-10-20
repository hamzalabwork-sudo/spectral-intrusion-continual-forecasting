from __future__ import annotations

"""
Post-hoc analysis of real_band_forgetting.py outputs.

For each (old task j, new task i) pair, relates per-bin error growth dE_j(f)
to scores built from the normalised target spectra s_j (old) and s_i (new):

    overlap      Delta_S(f)                 (paper's original score)
    neglect      max(0, s_j - s_i)          old-task energy the new task lacks
    intrusion    max(0, s_i - s_j)          new-task energy the old task lacks
    discrepancy  |s_j - s_i|

Spearman correlations are computed over non-DC bins, raw and partial
(controlling for s_j). Also reports how much of the total error growth sits
in the DC bin (level shift) and how different consecutive task spectra are.

Usage:
    python experiments/analyze_real_band.py results/real_band/*.json
"""

import json
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr, wilcoxon


def partial_spearman(x, y, z) -> float:
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def delta_s(a, b, eps=1e-8):
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    return (lo / (hi + eps)) * (lo / (0.5 * (a + b) + eps))


def main(paths: list[str]) -> None:
    names = ["overlap", "neglect", "intrusion", "discrepancy"]
    print(f"{'dataset':12} {'pairs':>5} {'DC share':>9} {'TV dist':>8}  " +
          "  ".join(f"{n + ' (raw/partial)':>26}" for n in names))
    for p in paths:
        d = json.load(open(p))
        spectra = [np.array(s) for s in d["spectra"]]
        raw = {n: [] for n in names}
        par = {n: [] for n in names}
        dc_share, tv = [], []
        for r in d["records"]:
            dE = np.array(r["dE"])
            sj, si = spectra[r["old_task"]], spectra[r["new_task"]]
            scores = {
                "overlap": delta_s(sj, si),
                "neglect": np.maximum(0, sj - si),
                "intrusion": np.maximum(0, si - sj),
                "discrepancy": np.abs(sj - si),
            }
            b = slice(1, None)
            for n, s in scores.items():
                raw[n].append(spearmanr(s[b], dE[b]).statistic)
                par[n].append(partial_spearman(s[b], dE[b], sj[b]))
            pos = np.clip(dE, 0, None)
            dc_share.append(pos[0] / (pos.sum() + 1e-12))
            tv.append(0.5 * np.abs(sj - si)[b].sum())

        def cell(n):
            pr = np.array(par[n])
            pval = wilcoxon(pr).pvalue if len(pr) >= 5 else float("nan")
            return f"{np.nanmean(raw[n]):+.2f} / {np.nanmean(pr):+.2f} (p={pval:.2g})"

        print(f"{os.path.basename(p)[:-5]:12} {len(d['records']):5d} {np.mean(dc_share):9.2f} "
              f"{np.mean(tv):8.3f}  " + "  ".join(f"{cell(n):>26}" for n in names))


if __name__ == "__main__":
    main(sys.argv[1:])
