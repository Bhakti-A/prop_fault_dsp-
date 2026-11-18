"""Per-motor order tracking driven by the motors' own eRPM telemetry (pure DSP, no learning).

For motor k, the shaft angle is theta_k(t) = 2*pi * integral f_k(t) dt.  The h-th order
component of the gyro signal x(t) inside a window is the tracked (speed-following) DFT
    X_k,h = sum_t w(t) x(t) exp(-j h theta_k(t)) / sum_t w(t)
with a Hann window w.  This is a short-time Vold-Kalman-style demodulation: it follows the
motor's speed changes, so its frequency resolution is set by the window length (about 2/T Hz
main lobe for a Hann window), not smeared by speed changes.  Tones of the OTHER motors only
leak in when their speed differs from motor k's by less than about 2/T Hz.
"""
from __future__ import annotations

import numpy as np


def shaft_phase(f_motor: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(n,4) Hz at sample times t [s] -> (n,4) shaft angle [rad], trapezoidal integration over
    the TRUE sample times (the 808 Hz logs are not evenly sampled)."""
    inc = 0.5 * (f_motor[1:] + f_motor[:-1]) * np.diff(t)[:, None]
    return 2 * np.pi * np.vstack([np.zeros((1, f_motor.shape[1])), np.cumsum(inc, axis=0)])


def spectrum_true_time(x: np.ndarray, t: np.ndarray, win: int, freqs: np.ndarray) -> np.ndarray:
    """Mean power spectrum of x (n, axes) at the given frequencies, summed over axes, from
    Hann-windowed non-overlapping segments, using the true sample times (a direct Fourier sum,
    so uneven sampling does not smear the lines).  Returns amplitude^2 per frequency."""
    w = np.hanning(win)
    acc, n = np.zeros(len(freqs)), 0
    for s in range(0, len(t) - win + 1, win):
        seg = x[s:s + win] - x[s:s + win].mean(axis=0)
        E = np.exp(-2j * np.pi * np.outer(t[s:s + win] - t[s], freqs))       # (win, n_freq)
        X = (w[:, None] * seg).T @ E / w.sum()                               # (axes, n_freq)
        acc += (np.abs(2 * X) ** 2).sum(axis=0)
        n += 1
    return acc / max(n, 1)


def window_starts(n: int, win: int, hop: int) -> np.ndarray:
    return np.arange(0, n - win + 1, hop)


def tracked_orders(x: np.ndarray, theta: np.ndarray, starts: np.ndarray, win: int,
                   orders=(1.0, 2.0, 3.0)) -> np.ndarray:
    """x (n, n_axes) signal, theta (n, n_motors). Returns amplitude (n_win, n_motors, n_orders)
    summed in power over axes: A = sqrt(sum_axes |X|^2) * 2 (single-sided sine amplitude)."""
    w = np.hanning(win)
    out = np.zeros((len(starts), theta.shape[1], len(orders)))
    for i, s in enumerate(starts):
        seg = x[s:s + win] - x[s:s + win].mean(axis=0)            # (win, axes)
        ws = w[:, None] * seg
        for k in range(theta.shape[1]):
            th = theta[s:s + win, k]
            for j, h in enumerate(orders):
                X = (np.exp(-1j * h * th)[:, None] * ws).sum(axis=0) / w.sum()
                out[i, k, j] = 2 * np.sqrt(np.sum(np.abs(X) ** 2))
    return out
