from __future__ import annotations

"""
Frequency-resolved forgetting on real benchmarks (Finetune, no CL protection).

After training on task i, the forecast-error spectrum of every earlier task j
is compared with its spectrum after task i-1. The per-bin error growth
dE_j(f) is then related to two candidate interference scores computed from
the target spectra of task j (old) and task i (new):

    overlap  Delta_S(f)  -- the paper's original score, Eq. (1)
    neglect  N(f) = max(0, s_j(f) - s_i(f))   with s = normalised amplitude spectrum,
             i.e. energy the old task needs that the new task does not provide

Because bins where task j has more energy may show more error growth whatever
task i contains, we also report partial Spearman correlations controlling
for s_j(f).

Usage:
    python experiments/real_band_forgetting.py --config configs/within/traffic.yaml \
        --data data/traffic.csv --name traffic --seeds 0 1 2
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_continual import (build_task_data, create_model, load_config,
                           resolve_target_indices, set_seed)


@torch.no_grad()
def error_spectrum(model, loader, device) -> tuple[float, np.ndarray]:
    """(MSE, mean over windows and target channels of |FFT_H(error)|^2 / H)."""
    model.eval()
    sq, spec, n, h = 0.0, None, 0, None
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        err = model(x)["y_pred"] - y                      # (B, H, C)
        h = err.shape[1]
        sq += err.pow(2).sum().item()
        e = torch.fft.rfft(err, dim=1).abs().pow(2).sum(dim=(0, 2)) / h
        spec = e if spec is None else spec + e
        n += err.shape[0] * err.shape[2]
    return sq / (n * h), (spec / n).cpu().numpy()


@torch.no_grad()
def target_spectrum(loader) -> np.ndarray:
    """Normalised mean amplitude spectrum of the H-step targets (DC excluded)."""
    acc, n = None, 0
    for _, y in loader:
        a = torch.fft.rfft(y, dim=1).abs().sum(dim=(0, 2))
        acc = a if acc is None else acc + a
        n += y.shape[0] * y.shape[2]
    s = (acc / n).numpy()
    s[0] = 0.0
    return s / (s.sum() + 1e-12)


def delta_s(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    return (lo / (hi + eps)) * (lo / (0.5 * (a + b) + eps))


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman correlation of x and y controlling for z (rank residuals)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a):
        beta = np.polyfit(rz, a, 1)
        return a - np.polyval(beta, rz)
    return float(np.corrcoef(resid(rx), resid(ry))[0, 1])


def train_task(model, loader, epochs, lr, wd, clip, device):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = F.mse_loss(model(x)["y_pred"], y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=48)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out_dir", default="results/real_band")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--backbone", choices=["mamba", "linear"], default="mamba")
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg["data"]["seq_len"], cfg["data"]["pred_len"] = args.seq_len, args.pred_len
    if args.backbone == "linear":
        cfg.setdefault("model", {})["backbone"] = "linear"
    dc, tc = cfg["data"], cfg["training"]
    task_split = dc.get("regime_split", dc.get("task_split", "equal"))
    dc["target_indices"] = resolve_target_indices(
        args.data, dc.get("feature_cols"), dc.get("target_cols"), dc.get("date_col", "date"))

    train_loaders, test_loaders, _ = build_task_data(
        csv_path=args.data, seq_len=args.seq_len, pred_len=args.pred_len, n_tasks=5,
        batch_size=tc["batch_size"], feature_cols=dc.get("feature_cols"),
        target_cols=dc.get("target_cols"), date_col=dc.get("date_col", "date"),
        task_split=task_split,
    )
    n_tasks = len(train_loaders)
    spectra = [target_spectrum(l) for l in train_loaders]
    sx, sy = next(iter(train_loaders[0]))
    n_in, n_out = sx.shape[-1], sy.shape[-1]

    os.makedirs(args.out_dir, exist_ok=True)
    records, pooled = [], {"dE": [], "overlap": [], "neglect": [], "s_old": []}
    t0 = time.time()
    for seed in args.seeds:
        set_seed(seed)
        model = create_model(cfg, n_in, n_out, args.device)
        prev = None
        for i in range(n_tasks):
            train_task(model, train_loaders[i], args.epochs, args.lr,
                       tc["weight_decay"], tc["grad_clip"], args.device)
            curr = [error_spectrum(model, l, args.device) for l in test_loaders]
            if prev is not None:
                for j in range(i):
                    dE = curr[j][1] - prev[j][1]
                    ov = delta_s(spectra[j], spectra[i])
                    ng = np.maximum(0.0, spectra[j] - spectra[i])
                    b = slice(1, None)  # exclude DC
                    rec = {
                        "seed": seed, "new_task": i, "old_task": j,
                        "mse_before": prev[j][0], "mse_after": curr[j][0],
                        "spearman_overlap": float(spearmanr(ov[b], dE[b]).statistic),
                        "spearman_neglect": float(spearmanr(ng[b], dE[b]).statistic),
                        "spearman_s_old": float(spearmanr(spectra[j][b], dE[b]).statistic),
                        "partial_overlap": partial_spearman(ov[b], dE[b], spectra[j][b]),
                        "partial_neglect": partial_spearman(ng[b], dE[b], spectra[j][b]),
                        "dE": dE.tolist(), "overlap": ov.tolist(), "neglect": ng.tolist(),
                    }
                    records.append(rec)
                    for k, v in (("dE", dE), ("overlap", ov), ("neglect", ng), ("s_old", spectra[j])):
                        pooled[k].extend(v[b].tolist())
            prev = curr
            print(f"[{time.time() - t0:7.1f}s] {args.name} seed={seed} task={i} "
                  f"mse={[round(c[0], 4) for c in curr]}", flush=True)

    keys = ["spearman_overlap", "spearman_neglect", "spearman_s_old", "partial_overlap", "partial_neglect"]
    summary = {k: [float(np.nanmean([r[k] for r in records])), float(np.nanstd([r[k] for r in records]))]
               for k in keys}
    summary["pooled_partial_overlap"] = partial_spearman(
        np.array(pooled["overlap"]), np.array(pooled["dE"]), np.array(pooled["s_old"]))
    summary["pooled_partial_neglect"] = partial_spearman(
        np.array(pooled["neglect"]), np.array(pooled["dE"]), np.array(pooled["s_old"]))
    summary["n_pairs"] = len(records)

    out = os.path.join(args.out_dir, f"{args.name}.json")
    with open(out, "w") as f:
        json.dump({"args": vars(args), "spectra": [s.tolist() for s in spectra],
                   "summary": summary, "records": records}, f)
    print("SUMMARY", args.name, json.dumps(summary))


if __name__ == "__main__":
    main()
