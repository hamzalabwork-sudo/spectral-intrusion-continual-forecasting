from __future__ import annotations

"""
Controlled test of the frequency-selective forgetting hypothesis.

Each task is a sum of independent AR(2) oscillators. An AR(2) component with
pole r * exp(+-i*omega) has a spectral peak at omega; the pole radius r sets
its damping, i.e. how the component's future evolves given its past.

Task A uses K frequencies. Task B shares round(rho*K) of them and draws the
rest from frequencies A does not use. At shared frequencies B either has a
different pole radius (conflicting dynamics) or the same one (overlap without
conflict, the control condition).

Protocol per run: train on A -> evaluate on A -> train on B -> evaluate on A.
Forgetting is resolved per frequency bin of the H-step forecast error, so we
can compare error growth at shared frequencies with growth at A-only
frequencies, and correlate it with the paper's interference score Delta_S(f).

Usage:
    python experiments/synthetic_overlap.py --out results/synthetic_overlap.json
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectracl.model import MambaForecaster


# ===================================================================
# Data generation
# ===================================================================
def ar2_component(n: int, cycles_per_sample: float, radius: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance AR(2) process with a spectral peak at the given frequency."""
    omega = 2 * math.pi * cycles_per_sample
    a1, a2 = 2 * radius * math.cos(omega), -radius ** 2
    burn = 500
    eps = rng.standard_normal(n + burn)
    x = np.zeros(n + burn)
    for t in range(2, n + burn):
        x[t] = a1 * x[t - 1] + a2 * x[t - 2] + eps[t]
    x = x[burn:]
    return (x - x.mean()) / (x.std() + 1e-8)


def make_task(n: int, freq_bins: list[int], radii: list[float], horizon: int,
              noise: float, rng: np.random.Generator) -> np.ndarray:
    """Sum of AR(2) components; bin j means j cycles per forecast horizon."""
    x = sum(ar2_component(n, j / horizon, r, rng) for j, r in zip(freq_bins, radii))
    x = x / math.sqrt(len(freq_bins)) + noise * rng.standard_normal(n)
    return x.astype(np.float32)


def windows(series: np.ndarray, seq_len: int, pred_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    s = torch.from_numpy(series)
    idx = torch.arange(len(s) - seq_len - pred_len + 1)
    x = torch.stack([s[i:i + seq_len] for i in idx]).unsqueeze(-1)
    y = torch.stack([s[i + seq_len:i + seq_len + pred_len] for i in idx]).unsqueeze(-1)
    return x, y


def split_loaders(series: np.ndarray, seq_len: int, pred_len: int, batch_size: int,
                  test_ratio: float = 0.2) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    cut = int(len(series) * (1 - test_ratio))
    xtr, ytr = windows(series[:cut], seq_len, pred_len)
    xte, yte = windows(series[cut:], seq_len, pred_len)
    train = DataLoader(TensorDataset(xtr, ytr), batch_size=batch_size, shuffle=True, drop_last=True)
    test = DataLoader(TensorDataset(xte, yte), batch_size=512, shuffle=False)
    return train, test, ytr


# ===================================================================
# Backbones
# ===================================================================
class MLPForecaster(nn.Module):
    """Backbone-agnostic control: two-layer MLP over the lookback window."""

    def __init__(self, seq_len: int, pred_len: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, pred_len),
        )

    def forward(self, x: torch.Tensor, update_memory: bool = False) -> dict:
        mu = x.mean(dim=1, keepdim=True)
        sd = x.std(dim=1, keepdim=True) + 1e-5
        y = self.net(((x - mu) / sd).squeeze(-1)).unsqueeze(-1)
        return {"y_pred": y * sd + mu}


class PureLinearForecaster(nn.Module):
    """Linear time-invariant control: one linear map over the window, no normalisation
    (instance normalisation divides by the window's std and is therefore nonlinear).
    Proposition 1 predicts that such a model cannot produce intrusion."""

    def __init__(self, seq_len: int, pred_len: int):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len, bias=False)

    def forward(self, x: torch.Tensor, update_memory: bool = False) -> dict:
        return {"y_pred": self.linear(x.squeeze(-1)).unsqueeze(-1)}


def build_model(name: str, seq_len: int, pred_len: int) -> nn.Module:
    if name == "linear":
        return PureLinearForecaster(seq_len, pred_len)
    if name == "mamba":
        return MambaForecaster(num_features=1, output_features=1, seq_len=seq_len,
                               pred_len=pred_len, d_model=64, d_state=16, n_layers=2,
                               dropout=0.0, use_revin=True)
    if name == "mlp":
        return MLPForecaster(seq_len, pred_len)
    raise ValueError(name)


# ===================================================================
# Training / spectral evaluation
# ===================================================================
def train(model: nn.Module, loader: DataLoader, epochs: int, lr: float, device: str) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = F.mse_loss(model(x)["y_pred"], y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


@torch.no_grad()
def error_spectrum(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, np.ndarray]:
    """Return (MSE, per-bin error energy mean_n |FFT_H(y_hat - y)|^2 / H)."""
    model.eval()
    sq, spec, n = 0.0, None, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        err = (model(x)["y_pred"] - y).squeeze(-1)
        sq += err.pow(2).sum().item()
        e = torch.fft.rfft(err, dim=1).abs().pow(2).sum(dim=0) / err.shape[1]
        spec = e if spec is None else spec + e
        n += err.shape[0]
    return sq / (n * err.shape[1]), (spec / n).cpu().numpy()


def amplitude_spectrum(y: torch.Tensor) -> np.ndarray:
    return torch.fft.rfft(y.squeeze(-1), dim=1).abs().mean(dim=0).numpy()


def delta_s(s_prev: np.ndarray, s_curr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Spectral interference score, Eq. (1) of the paper."""
    lo, hi = np.minimum(s_prev, s_curr), np.maximum(s_prev, s_curr)
    return (lo / (hi + eps)) * (lo / (0.5 * (s_prev + s_curr) + eps))


# ===================================================================
# One run
# ===================================================================
def run_one(args, rho: float, conflict: bool, backbone: str, seed: int, device: str) -> dict:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    grid = list(range(args.min_bin, args.max_bin + 1))
    a_bins = sorted(rng.choice(grid, size=args.k, replace=False).tolist())
    n_shared = int(round(rho * args.k))
    shared = sorted(rng.choice(a_bins, size=n_shared, replace=False).tolist()) if n_shared else []
    # Keep B-only frequencies at least 2 bins from every A frequency so leakage
    # does not blur the shared / A-only distinction.
    free = [j for j in grid if all(abs(j - a) >= 2 for a in a_bins)]
    b_new = sorted(rng.choice(free, size=args.k - n_shared, replace=False).tolist())
    b_bins = shared + b_new

    r_shared_b = args.r_b if conflict else args.r_a
    a_series = make_task(args.n, a_bins, [args.r_a] * args.k, args.pred_len, args.noise, rng)
    b_series = make_task(args.n, b_bins, [r_shared_b] * n_shared + [args.r_a] * len(b_new),
                         args.pred_len, args.noise, rng)

    a_tr, a_te, a_ytr = split_loaders(a_series, args.seq_len, args.pred_len, args.batch_size)
    b_tr, _, b_ytr = split_loaders(b_series, args.seq_len, args.pred_len, args.batch_size)

    model = build_model(backbone, args.seq_len, args.pred_len).to(device)
    train(model, a_tr, args.epochs, args.lr, device)
    mse_before, spec_before = error_spectrum(model, a_te, device)
    train(model, b_tr, args.epochs, args.lr, device)
    mse_after, spec_after = error_spectrum(model, a_te, device)

    d_err = spec_after - spec_before
    ds = delta_s(amplitude_spectrum(a_ytr), amplitude_spectrum(b_ytr))
    a_only = [j for j in a_bins if j not in shared]
    rho_s = spearmanr(ds[1:], d_err[1:]).statistic if len(ds) > 2 else float("nan")

    return {
        "rho": rho, "conflict": conflict, "backbone": backbone, "seed": seed,
        "a_bins": a_bins, "b_bins": b_bins, "shared_bins": shared,
        "mse_before": mse_before, "mse_after": mse_after,
        "forgetting": mse_after - mse_before,
        "rel_forgetting": (mse_after - mse_before) / mse_before,
        "derr_shared": float(np.mean(d_err[shared])) if shared else None,
        "derr_a_only": float(np.mean(d_err[a_only])) if a_only else None,
        "derr_other": float(np.mean(np.delete(d_err, a_bins + [0]))),
        "spearman_deltaS_vs_derr": float(rho_s),
        "delta_err": d_err.tolist(), "delta_s": ds.tolist(),
        "err_before": spec_before.tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/synthetic_overlap.json")
    p.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--backbones", nargs="+", default=["mamba", "mlp"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
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
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    runs = []
    t0 = time.time()
    for backbone in args.backbones:
        for conflict in (True, False):
            for rho in args.rhos:
                for seed in args.seeds:
                    r = run_one(args, rho, conflict, backbone, seed, args.device)
                    runs.append(r)
                    print(f"[{time.time() - t0:7.1f}s] {backbone:5s} conflict={conflict!s:5s} "
                          f"rho={rho:.2f} seed={seed} forget={r['forgetting']:+.4f} "
                          f"rel={r['rel_forgetting']:+.3f} shared={r['derr_shared']} "
                          f"a_only={r['derr_a_only']} spearman={r['spearman_deltaS_vs_derr']:+.3f}",
                          flush=True)
                    with open(args.out, "w") as f:
                        json.dump({"args": vars(args), "runs": runs}, f)
    print(f"Saved {len(runs)} runs to {args.out}")


if __name__ == "__main__":
    main()
