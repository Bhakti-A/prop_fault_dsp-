"""Fault monitor built only from signal processing and healthy-flight statistics (no learning).

Per window and per motor k we measure the motor's tracked 1x order level L_k (dB, from
order_tracking.tracked_orders).  A healthy baseline gives, for each motor, the median and the
MAD of L_k over all healthy in-flight windows.  (Speed-binned statistics were tried: on these
flights the 1x level correlates only weakly with speed and the bins mostly separated flights, so
they were slightly worse; a single set of statistics per motor is used.)  Then
    z_k = (L_k - median) / (1.4826 * MAD)              robust "how unusual is this level"
    window flag : max_k z_k > T, with T = 99th percentile of the baseline windows' max_k z_k
    flight alarm: the flight-median z of any motor exceeds FLIGHT_Z (a sustained shift; the
                  median over a flight is far steadier than single windows)
    localization: the motor with the largest flight-median z
    severity    : change in that motor's typical 1x energy,
                  10*log10(median A^2 / baseline median A^2) [dB]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BIN_HZ = 1000.0          # one bin covering all in-flight speeds (see docstring)
MIN_PER_BIN = 10
MIN_SPREAD_DB = 1.0
THRESH_PCTL = 99.0
FLIGHT_Z = 1.5                # flight-level alarm if a motor's flight-median z exceeds this.
                              # Chosen after looking at these 16 flights (largest value in a flight
                              # as flown: 1.11), so it still needs confirming on new flights.


@dataclass
class Baseline:
    edges: np.ndarray            # speed-bin edges [Hz]
    med: np.ndarray              # (n_motors, n_bins) dB
    spread: np.ndarray           # (n_motors, n_bins) dB, robust sigma
    g_med: np.ndarray            # (n_motors,) global fallback
    g_spread: np.ndarray
    med_power: np.ndarray        # (n_motors,) baseline median A^2 (for severity)
    threshold: float
    n_windows: int


def _stats(v):
    m = np.median(v)
    return m, max(1.4826 * np.median(np.abs(v - m)), MIN_SPREAD_DB)


def fit_baseline(A1: np.ndarray, fwin: np.ndarray) -> Baseline:
    """A1 (n_win, n_motors) tracked 1x amplitude; fwin (n_win, n_motors) window-mean speed."""
    L = 20 * np.log10(A1)
    nm = A1.shape[1]
    edges = np.arange(np.floor(fwin.min() / BIN_HZ) * BIN_HZ, fwin.max() + BIN_HZ, BIN_HZ)
    nb = len(edges) - 1
    med = np.full((nm, nb), np.nan)
    spread = np.full((nm, nb), np.nan)
    g_med, g_spread = np.zeros(nm), np.zeros(nm)
    for k in range(nm):
        g_med[k], g_spread[k] = _stats(L[:, k])
        b = np.clip(np.digitize(fwin[:, k], edges) - 1, 0, nb - 1)
        for j in range(nb):
            if (b == j).sum() >= MIN_PER_BIN:
                med[k, j], spread[k, j] = _stats(L[b == j, k])
    bl = Baseline(edges, med, spread, g_med, g_spread, np.median(A1 ** 2, axis=0), np.inf, len(A1))
    bl.threshold = float(np.percentile(scores(bl, A1, fwin).max(axis=1), THRESH_PCTL))
    return bl


def scores(bl: Baseline, A1: np.ndarray, fwin: np.ndarray) -> np.ndarray:
    """Robust z-score per window and motor, (n_win, n_motors)."""
    L = 20 * np.log10(A1)
    nb = len(bl.edges) - 1
    z = np.zeros_like(L)
    for k in range(L.shape[1]):
        b = np.clip(np.digitize(fwin[:, k], bl.edges) - 1, 0, nb - 1)
        m, s = bl.med[k, b], bl.spread[k, b]
        miss = np.isnan(m)
        m = np.where(miss, bl.g_med[k], m)
        s = np.where(miss, bl.g_spread[k], s)
        z[:, k] = (L[:, k] - m) / s
    return z


def decide(bl: Baseline, A1: np.ndarray, fwin: np.ndarray) -> dict:
    z = scores(bl, A1, fwin)
    flagged = z.max(axis=1) > bl.threshold
    motor_win = z.argmax(axis=1)
    zmed = np.median(z, axis=0)
    motor_flight = int(np.argmax(zmed))
    severity_db = 10 * np.log10(np.median(A1 ** 2, axis=0) / bl.med_power)  # per motor
    return {"z": z, "flagged": flagged, "motor_win": motor_win,
            "frac_flagged": float(flagged.mean()),
            "flight_score": float(zmed.max()), "flight_alarm": bool(zmed.max() > FLIGHT_Z),
            "motor_flight": motor_flight, "severity_db": severity_db}
