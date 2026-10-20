from __future__ import annotations

"""
Intervention test on the synthetic AR(2) tasks.

If forgetting is driven by spectral discrepancy (mainly intrusion of the new
task's frequencies), replay designed around it should beat replay designed
around spectral overlap, at the same memory and compute budget.

Protocol per (rho, seed): train on A; store a fixed random buffer of A
windows; then train on B with one replay batch per step, for every
combination of
    sampler: random | similar (softmax(+cos/tau), current SIAR) | complement (softmax(-cos/tau))
    loss:    mse | fdrl_old (weights ~ s_A, current FDRL) | fdrl_intrusion (weights ~ 1 + k*max(0, s_B - s_A))
Frequency weights are normalised to mean 1 over bins, so uniform weights equal MSE.

Usage:
    python experiments/synthetic_intervention.py --out results/synthetic_intervention.json
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_overlap import build_model, error_spectrum, make_task, split_loaders, train

SAMPLERS = ["random", "similar", "complement"]
LOSSES = ["mse", "fdrl_old", "fdrl_intrusion"]


def fingerprint(x: torch.Tensor) -> torch.Tensor:
    """Unit-norm amplitude spectrum of each input window (DC dropped)."""
    a = torch.fft.rfft(x.squeeze(-1), dim=1).abs()[:, 1:]
    return F.normalize(a, dim=1)


def target_spectrum(y: torch.Tensor) -> torch.Tensor:
    s = torch.fft.rfft(y.squeeze(-1), dim=1).abs().mean(dim=0)
    s[0] = 0.0
    return s / s.sum()


def freq_weights(kind: str, s_old: torch.Tensor, s_new: torch.Tensor, kappa: float) -> torch.Tensor | None:
    if kind == "mse":
        return None
    if kind == "fdrl_old":
        w = s_old.clone()
    elif kind == "fdrl_intrusion":
        intr = torch.clamp(s_new - s_old, min=0)
        w = 1.0 + kappa * intr / (intr.max() + 1e-12)
    else:
        raise ValueError(kind)
    w[0] = w[1:].mean()  # keep the DC (level) term at average weight
    return w / w.mean()


def replay_loss(pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor | None) -> torch.Tensor:
    if w is None:
        return F.mse_loss(pred, target)
    h = pred.shape[1]
    diff = torch.fft.rfft((pred - target).squeeze(-1), dim=1).abs().pow(2) / h
    return (diff * w.unsqueeze(0)).mean()


def train_with_replay(model, loader, buf_x, buf_y, buf_fp, sampler, w, args, device):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    model.train()
    n = buf_x.shape[0]
    for _ in range(args.epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if sampler == "random":
                idx = torch.randint(0, n, (args.replay_bs,), device=device)
            else:
                sim = buf_fp @ fingerprint(x).mean(dim=0)
                sign = 1.0 if sampler == "similar" else -1.0
                probs = torch.softmax(sign * sim / args.tau, dim=0)
                idx = torch.multinomial(probs, args.replay_bs, replacement=True)
            loss = F.mse_loss(model(x)["y_pred"], y)
            loss = loss + args.replay_weight * replay_loss(model(buf_x[idx])["y_pred"], buf_y[idx], w)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/synthetic_intervention.json")
    p.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.5])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--backbone", default="mamba")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--n", type=int, default=12000)
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=48)
    p.add_argument("--min_bin", type=int, default=2)
    p.add_argument("--max_bin", type=int, default=20)
    p.add_argument("--r_a", type=float, default=0.99)
    p.add_argument("--r_b", type=float, default=0.90)
    p.add_argument("--noise", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--buffer", type=int, default=128)
    p.add_argument("--replay_bs", type=int, default=32)
    p.add_argument("--replay_weight", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--kappa", type=float, default=4.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    dev = args.device

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    runs, t0 = [], time.time()
    for rho in args.rhos:
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            torch.manual_seed(seed)
            grid = list(range(args.min_bin, args.max_bin + 1))
            a_bins = sorted(rng.choice(grid, size=args.k, replace=False).tolist())
            n_shared = int(round(rho * args.k))
            shared = sorted(rng.choice(a_bins, size=n_shared, replace=False).tolist()) if n_shared else []
            free = [j for j in grid if all(abs(j - a) >= 2 for a in a_bins)]
            b_new = sorted(rng.choice(free, size=args.k - n_shared, replace=False).tolist())
            a_series = make_task(args.n, a_bins, [args.r_a] * args.k, args.pred_len, args.noise, rng)
            b_series = make_task(args.n, shared + b_new, [args.r_b] * n_shared + [args.r_a] * len(b_new),
                                 args.pred_len, args.noise, rng)
            a_tr, a_te, a_ytr = split_loaders(a_series, args.seq_len, args.pred_len, args.batch_size)
            b_tr, b_te, b_ytr = split_loaders(b_series, args.seq_len, args.pred_len, args.batch_size)

            base = build_model(args.backbone, args.seq_len, args.pred_len).to(dev)
            train(base, a_tr, args.epochs, args.lr, dev)
            mse_a0, _ = error_spectrum(base, a_te, dev)

            ds = a_tr.dataset
            pick = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))[:args.buffer]
            buf_x, buf_y = ds.tensors[0][pick].to(dev), ds.tensors[1][pick].to(dev)
            buf_fp = fingerprint(buf_x)
            s_a, s_b = target_spectrum(a_ytr).to(dev), target_spectrum(b_ytr).to(dev)

            # Lower bound without replay
            ft = copy.deepcopy(base)
            train(ft, b_tr, args.epochs, args.lr, dev)
            ft_mse_a, _ = error_spectrum(ft, a_te, dev)
            ft_mse_b, _ = error_spectrum(ft, b_te, dev)

            for sampler in SAMPLERS:
                for loss_kind in LOSSES:
                    torch.manual_seed(seed + 1000)
                    m = copy.deepcopy(base)
                    w = freq_weights(loss_kind, s_a, s_b, args.kappa)
                    train_with_replay(m, b_tr, buf_x, buf_y, buf_fp, sampler, w, args, dev)
                    mse_a, _ = error_spectrum(m, a_te, dev)
                    mse_b, _ = error_spectrum(m, b_te, dev)
                    runs.append({
                        "rho": rho, "seed": seed, "sampler": sampler, "loss": loss_kind,
                        "mse_a_before": mse_a0, "mse_a_after": mse_a, "mse_b": mse_b,
                        "rel_forgetting": (mse_a - mse_a0) / mse_a0,
                        "finetune_rel_forgetting": (ft_mse_a - mse_a0) / mse_a0,
                        "finetune_mse_b": ft_mse_b,
                    })
                    print(f"[{time.time() - t0:7.1f}s] rho={rho:.2f} seed={seed} {sampler:10s} {loss_kind:14s} "
                          f"forget={runs[-1]['rel_forgetting']:+.3f} (ft {runs[-1]['finetune_rel_forgetting']:+.3f}) "
                          f"mseB={mse_b:.4f}", flush=True)
                    with open(args.out, "w") as f:
                        json.dump({"args": vars(args), "runs": runs}, f)
    print(f"Saved {len(runs)} runs to {args.out}")


if __name__ == "__main__":
    main()
