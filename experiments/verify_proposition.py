from __future__ import annotations

"""
Numerical check of Proposition 1 (frequency decomposition of forgetting for linear forecasters).

For a linear forecaster with lookback L and horizon H, y_hat_{t+h} = sum_j w_{h,j} x_{t-j},
let W_A be the least-squares (Wiener) solution for a stationary regime A. Orthogonality of
the optimal error to the lookback window gives, for any other W_B,

    MSE_A(W_B) - MSE_A(W_A) = (1/H) sum_h int_0^{1/2} |H_{B,h}(f) - H_{A,h}(f)|^2 S_A(f) df,

with S_A the one-sided spectral density of A and H_{.,h} the frequency response of row h.
This script fits W_A and W_B on synthetic AR(2) regimes, compares the measured increase of
A's test error with the right-hand side, and splits the right-hand side by frequency band.

Usage:
    python experiments/verify_proposition.py
"""

import os
import sys

import numpy as np
from scipy.signal import welch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic_overlap import make_task

L, H, N_FFT = 96, 48, 8192


def windows(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(x) - L - H + 1)
    X = np.stack([x[i:i + L] for i in idx])
    Y = np.stack([x[i + L:i + L + H] for i in idx])
    return X, Y


def fit(x: np.ndarray) -> np.ndarray:
    X, Y = windows(x)
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)      # (L, H)
    return W


def mse(W: np.ndarray, x: np.ndarray) -> float:
    X, Y = windows(x)
    return float(np.mean((X @ W - Y) ** 2))


def run(seed: int, rho: float, conflict: bool) -> dict:
    rng = np.random.default_rng(seed)
    grid = list(range(2, 21))
    a_bins = sorted(rng.choice(grid, size=4, replace=False).tolist())
    n_shared = int(round(rho * 4))
    shared = sorted(rng.choice(a_bins, size=n_shared, replace=False).tolist()) if n_shared else []
    free = [j for j in grid if all(abs(j - a) >= 2 for a in a_bins)]
    b_new = sorted(rng.choice(free, size=4 - n_shared, replace=False).tolist())
    r_shared = 0.90 if conflict else 0.99
    a = make_task(60000, a_bins, [0.99] * 4, H, 0.1, rng).astype(np.float64)
    b = make_task(60000, shared + b_new, [r_shared] * n_shared + [0.99] * len(b_new), H, 0.1, rng).astype(np.float64)
    a_tr, a_te = a[:40000], a[40000:]
    W_A, W_B = fit(a_tr), fit(b[:40000])

    measured = mse(W_B, a_te) - mse(W_A, a_te)

    f, S_A = welch(a_tr, fs=1.0, nperseg=4096, return_onesided=True, scaling="density")
    dW = W_B - W_A                                            # (L, H): column h is a filter over lags
    dH = np.fft.rfft(dW, n=N_FFT, axis=0)                     # response on [0, 1/2]
    freq = np.fft.rfftfreq(N_FFT)
    S_interp = np.interp(freq, f, S_A)
    density = (np.abs(dH) ** 2).mean(axis=1) * S_interp       # averaged over horizon steps
    df = freq[1] - freq[0]
    predicted = float(np.sum(density) * df)

    def band(bins):
        if not bins:
            return 0.0
        m = np.zeros_like(freq, dtype=bool)
        for k in bins:
            m |= np.abs(freq - k / H) <= 0.5 / H
        return float(np.sum(density[m]) * df)

    parts = {
        "intrusion (B-only)": band(sorted(set(b_new))),
        "neglect (A-only)": band(sorted(set(a_bins) - set(shared))),
        "shared": band(shared),
    }
    parts["rest"] = predicted - sum(parts.values())
    return {"measured": measured, "predicted": predicted, "parts": parts, "mse_A": mse(W_A, a_te)}


def main():
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out_json", default="results/proposition_check.json")
    ap.add_argument("--out_tex", default=None)
    args = ap.parse_args()

    settings = [(0.0, False), (0.5, False), (0.5, True), (1.0, False), (1.0, True)]
    summary = []
    print(f"{'setting':28} {'measured dMSE':>14} {'predicted':>10} {'ratio':>7}   decomposition of predicted (share)")
    for rho, conflict in settings:
        rows = [run(s, rho, conflict) for s in range(args.seeds)]
        m = float(np.mean([r["measured"] for r in rows]))
        p = float(np.mean([r["predicted"] for r in rows]))
        parts = {k: float(np.mean([r["parts"][k] for r in rows])) for k in rows[0]["parts"]}
        share = "  ".join(f"{k}: {v / p:5.1%}" for k, v in parts.items()) if p > 0 else ""
        label = f"rho={rho:.2f} {'conflict' if conflict else 'same dyn.'}"
        print(f"{label:28} {m:14.4f} {p:10.4f} {m / p if p else float('nan'):7.3f}   {share}")
        summary.append({"rho": rho, "conflict": conflict, "measured": m, "predicted": p,
                        "mse_A": float(np.mean([r["mse_A"] for r in rows])), "parts": parts,
                        "n_seeds": args.seeds})

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    if args.out_tex:
        lines = [r"\begin{table}[t]", r"\centering",
                 r"\caption{Numerical check of Proposition~\ref{prop:decomp} on the controlled AR(2) tasks "
                 rf"(least-squares linear forecasters, $L=96$, $H=48$, mean over {args.seeds} seeds). "
                 r"\emph{Measured}: increase of task A's test MSE when A's forecaster is replaced by B's; "
                 r"\emph{predicted}: right-hand side of Eq.~(\ref{eq:decomp}); the last three columns split "
                 r"the prediction by the input frequencies that drive it.}",
                 r"\label{tab:prop}", r"\small", r"\resizebox{\linewidth}{!}{%",
                 r"\begin{tabular}{lccccc}", r"\toprule",
                 r"Setting & Measured & Predicted & A-only bins & shared bins & B-only bins \\", r"\midrule"]
        for s in summary:
            name = rf"$\rho={s['rho']:g}$, {'conflicting' if s['conflict'] else 'same'} dynamics"
            p = s["predicted"]
            sh = s["parts"]
            cells = [f"{s['measured']:.3f}", f"{p:.3f}"]
            if p > 1e-2:
                cells += [f"{100 * sh['neglect (A-only)'] / p:.0f}\\%", f"{100 * sh['shared'] / p:.0f}\\%",
                          f"{100 * sh['intrusion (B-only)'] / p:.0f}\\%"]
            else:
                cells += ["--", "--", "--"]
            lines.append(name + " & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
        with open(args.out_tex, "w") as f:
            f.write("\n".join(lines) + "\n")
        print("wrote", args.out_tex)


if __name__ == "__main__":
    main()
