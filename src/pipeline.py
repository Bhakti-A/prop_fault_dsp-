"""Windowing and per-flight order tracking."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bblog import Flight
from order_tracking import shaft_phase, tracked_orders, window_starts

WIN_S = 1.0             # 1 s windows -> ~2 Hz main lobe (Hann): separates motors >~2-4 Hz apart
HOP_FRAC = 0.5
MIN_FLIGHT_HZ = 150.0   # a window is "in flight" if every motor's mean speed > 150 Hz (armed idle ~105 Hz)
ORDERS = (1.0, 2.0, 3.0)


@dataclass
class Prepared:
    flight: Flight
    theta: np.ndarray        # (n, 4) shaft angle
    starts: np.ndarray       # in-flight window starts
    win: int
    fwin: np.ndarray         # (n_win, 4) mean speed per window
    orders_unfilt: np.ndarray    # (n_win, 4, 3) tracked 1x/2x/3x amplitude, gyroUnfilt
    orders_filt: np.ndarray      # (n_win, 4, 3) same on gyroADC (after Betaflight filters)
    line_order: np.ndarray       # (4,) where each motor's 1x line actually sits, in orders of its eRPM

    def a1(self):
        return self.orders_unfilt[:, :, 0]


LINE_SEARCH = np.round(np.arange(0.990, 1.0151, 0.0005), 4)   # +-~3 Hz around the eRPM frequency


def locate_lines(gyro, theta, starts, win):
    """Each motor's 1x line sits slightly above the frequency computed from its eRPM telemetry
    (0.7-2.3 Hz here, different per motor, steady across flights).  With 1 s windows that is enough
    to miss most of the line, so first find, per motor, the order in LINE_SEARCH where the
    flight's median tracked power peaks, and track the line there."""
    best = np.ones(theta.shape[1])
    for k in range(theta.shape[1]):
        A = tracked_orders(gyro, theta[:, k:k + 1], starts, win, LINE_SEARCH)[:, 0, :]
        best[k] = LINE_SEARCH[np.argmax(np.median(A ** 2, axis=0))]
    return best


def _track_at(gyro, theta, starts, win, line):
    out = np.zeros((len(starts), theta.shape[1], len(ORDERS)))
    for k in range(theta.shape[1]):
        out[:, k, :] = tracked_orders(gyro, theta[:, k:k + 1], starts, win,
                                      tuple(h * line[k] for h in ORDERS))[:, 0, :]
    return out


def prepare(F: Flight) -> Prepared:
    win = int(round(WIN_S * F.fs))
    st = window_starts(len(F.t), win, int(win * HOP_FRAC))
    fwin = np.array([F.f_motor[s:s + win].mean(axis=0) for s in st])
    ok = fwin.min(axis=1) > MIN_FLIGHT_HZ
    st, fwin = st[ok], fwin[ok]
    th = shaft_phase(F.f_motor, F.t)
    line = locate_lines(F.gyro_unfilt, th, st, win) if len(st) else np.ones(4)
    return Prepared(F, th, st, win, fwin,
                    _track_at(F.gyro_unfilt, th, st, win, line),
                    _track_at(F.gyro_filt, th, st, win, line), line)


def decimated(F: Flight, n: int) -> Flight:
    """Keep every n-th sample, exactly what the blackbox does at a lower log rate (no filtering)."""
    return Flight(F.name, F.fs / n, F.t[::n], F.gyro_unfilt[::n], F.gyro_filt[::n],
                  F.f_motor[::n], F.motor_cmd[::n], F.header)
