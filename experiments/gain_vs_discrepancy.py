from __future__ import annotations

"""
Does IFR's benefit over plain replay grow with spectral intrusion?

For every benchmark (a result directory plus the config that produced it) and
every old task j < K, compute
    intrusion_j  = mean over later tasks i > j of sum_f max(0, s_i(f) - s_j(f))
                   (stored-spectrum quantity, known before final evaluation)
    gain_j       = mean over shared seeds of (MSE_ER[K, j] - MSE_IFR[K, j]) / MSE_ER[K, j]
and report the Spearman correlation across all (benchmark, task) points.

Usage:
    python experiments/gain_vs_discrepancy.py \
        --bench crossA results/crossdomain/order_A configs/crossdomain/crossdomain_A.yaml data \
        --bench elec   results/pilot/electricity configs/within/electricity.yaml data/electricity.csv
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_continual import build_multidomain_task_data, build_task_data, load_config
from spectracl.continual.intrusion_rehearsal import target_spectrum


def task_spectra(cfg_path: str, data: str, seq_len: int, pred_len: int) -> list[np.ndarray]:
    cfg = load_config(cfg_path)
    dc = cfg["data"]
    if dc.get("domains"):
        loaders, _, _ = build_multidomain_task_data(dc["domains"], seq_len, pred_len, 64, data_root=data)
    else:
        loaders, _, _ = build_task_data(
            data, seq_len, pred_len, 5, 64, feature_cols=dc.get("feature_cols"),
            target_cols=dc.get("target_cols"), date_col=dc.get("date_col", "date"),
            task_split=dc.get("regime_split", dc.get("task_split", "equal")))
    return [target_spectrum(l).numpy() for l in loaders]


def final_mse(result_dir: str, method: str) -> dict[int, list[float]]:
    out = {}
    for p in glob.glob(os.path.join(result_dir, "continual_results_*.json")):
        for e in json.load(open(p)):
            if e["method"] == method:
                out[e.get("seed", 0)] = [r["mse"] for r in e["results_matrix"][-1]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs=4, action="append", required=True,
                    metavar=("NAME", "RESULT_DIR", "CONFIG", "DATA"))
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--pred_len", type=int, default=48)
    ap.add_argument("--out", default="results/gain_vs_discrepancy.json")
    args = ap.parse_args()

    points = []
    for name, rdir, cfg, data in args.bench:
        spectra = task_spectra(cfg, data, args.seq_len, args.pred_len)
        er, ifr = final_mse(rdir, "ifr_uniform"), final_mse(rdir, "ifr")
        seeds = sorted(set(er) & set(ifr))
        k = len(spectra)
        for j in range(k - 1):
            intr = float(np.mean([np.clip(spectra[i] - spectra[j], 0, None).sum() for i in range(j + 1, k)]))
            gains = [(er[s][j] - ifr[s][j]) / er[s][j] for s in seeds]
            points.append({"bench": name, "task": j, "intrusion": intr,
                           "gain": float(np.mean(gains)), "gain_std": float(np.std(gains)),
                           "n_seeds": len(seeds)})
            print(f"{name:8} task {j}: intrusion={intr:.3f}  gain={np.mean(gains):+.3%} "
                  f"(+-{np.std(gains):.3%}, n={len(seeds)})")

    x = [p["intrusion"] for p in points]
    y = [p["gain"] for p in points]
    rs = spearmanr(x, y)
    lo = [p["gain"] for p in points if p["intrusion"] <= np.median(x)]
    hi = [p["gain"] for p in points if p["intrusion"] > np.median(x)]
    print(f"\nSpearman(intrusion, gain) over {len(points)} tasks: r={rs.statistic:+.3f}, p={rs.pvalue:.3g}")
    print(f"Mean gain, low-intrusion half: {np.mean(lo):+.2%}   high-intrusion half: {np.mean(hi):+.2%}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"points": points, "spearman": [rs.statistic, rs.pvalue]}, f, indent=2)


if __name__ == "__main__":
    main()
