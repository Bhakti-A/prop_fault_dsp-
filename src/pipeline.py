"""Windowing, per-flight order tracking and the leave-one-flight-out evaluation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bblog import Flight
from inject import a0_for_ratio, inject_imbalance
from monitor import decide, fit_baseline
from order_tracking import shaft_phase, spectrum_true_time, tracked_orders, window_starts

WIN_S = 1.0             # 1 s windows -> ~2 Hz main lobe (Hann): separates motors >~2-4 Hz apart
HOP_FRAC = 0.5
MIN_FLIGHT_HZ = 200.0   # a window is "in flight" if every motor's mean speed > 200 Hz (idle ~105 Hz)
ORDERS = (1.0, 2.0, 3.0)
MOTORS = ["M1", "M2", "M3", "M4"]          # Betaflight motor[0..3]


@dataclass
class Prepared:
    flight: Flight
    theta: np.ndarray        # (n, 4) shaft angle
    starts: np.ndarray       # in-flight window starts
    win: int
    fwin: np.ndarray         # (n_win, 4) mean speed per window
    orders_unfilt: np.ndarray    # (n_win, 4, 3) tracked 1x/2x/3x amplitude, gyroUnfilt
    orders_filt: np.ndarray      # (n_win, 4, 3) same on gyroADC (after Betaflight filters)

    def a1(self, gyro=None):
        if gyro is None:
            return self.orders_unfilt[:, :, 0]
        return tracked_orders(gyro, self.theta, self.starts, self.win, orders=(1.0,))[:, :, 0]


def prepare(F: Flight) -> Prepared:
    win = int(round(WIN_S * F.fs))
    st = window_starts(len(F.t), win, int(win * HOP_FRAC))
    fwin = np.array([F.f_motor[s:s + win].mean(axis=0) for s in st])
    ok = fwin.min(axis=1) > MIN_FLIGHT_HZ
    st, fwin = st[ok], fwin[ok]
    th = shaft_phase(F.f_motor, F.t)
    return Prepared(F, th, st, win, fwin,
                    tracked_orders(F.gyro_unfilt, th, st, win, ORDERS),
                    tracked_orders(F.gyro_filt, th, st, win, ORDERS))


def baseline_from(preps):
    A = np.vstack([p.a1() for p in preps])
    f = np.vstack([p.fwin for p in preps])
    return fit_baseline(A, f), A, f


def nearest_sep(fwin, k):
    others = np.delete(fwin, k, axis=1)
    return np.abs(others - fwin[:, [k]]).min(axis=1)


def evaluate(preps, ratios, n_phase=3, seed=0):
    """Leave-one-flight-out: the baseline never contains the flight under test.
    Returns (clean results, injected flight-level results, injected window-level results)."""
    rng = np.random.default_rng(seed)
    clean, inj_rows, win_rows = [], [], []
    for i, test in enumerate(preps):
        train = [p for j, p in enumerate(preps) if j != i]
        bl, A_tr, f_tr = baseline_from(train)
        d = decide(bl, test.a1(), test.fwin)
        clean.append({"flight": test.flight.name, "windows": len(test.starts),
                      "baseline_windows": bl.n_windows, "threshold_z": bl.threshold,
                      "false_alarm_rate_windows": d["frac_flagged"],
                      "flight_score": d["flight_score"],
                      "flight_alarm": d["flight_alarm"],
                      **{f"excess_dB_{m}": d["severity_db"][k] for k, m in enumerate(MOTORS)}})
        for k in range(4):
            amp_med = np.median(A_tr[:, k])
            f_ref = np.median(f_tr[:, k])
            sep = nearest_sep(test.fwin, k)
            for r in ratios:
                for p in range(n_phase):
                    phi = rng.uniform(0, 2 * np.pi)
                    g = inject_imbalance(test.flight.gyro_unfilt, test.theta, test.flight.f_motor,
                                         k, a0_for_ratio(r, amp_med), f_ref, phi)
                    di = decide(bl, test.a1(g), test.fwin)
                    inj_rows.append({"flight": test.flight.name, "motor": MOTORS[k], "ratio": r,
                                     "phase": p, "frac_flagged": di["frac_flagged"], "flight_score": di["flight_score"],
                                     "flight_alarm": di["flight_alarm"],
                                     "located": MOTORS[di["motor_flight"]],
                                     "severity_est_dB": di["severity_db"][k],
                                     "severity_nominal_dB": 10 * np.log10(1 + r ** 2)})
                    win_rows.append(pd.DataFrame({
                        "flight": test.flight.name, "motor": MOTORS[k], "ratio": r, "phase": p,
                        "flagged": di["flagged"], "located": np.array(MOTORS)[di["motor_win"]],
                        "sep_hz": sep}))
    return pd.DataFrame(clean), pd.DataFrame(inj_rows), pd.concat(win_rows, ignore_index=True)


def decimated(F: Flight, n: int) -> Flight:
    """Keep every n-th sample, exactly what the blackbox does at a lower log rate (no filtering)."""
    return Flight(F.name, F.fs / n, F.t[::n], F.gyro_unfilt[::n], F.gyro_filt[::n],
                  F.f_motor[::n], F.motor_cmd[::n], F.header)


BAND_HZ = (280.0, 400.0)       # covers every motor's in-flight rotation speed in these logs


def band_power_db(F: Flight) -> float:
    """Plain (untracked) gyro power in the motor-speed band, 3 axes summed: what a simple
    band-energy detector would look at."""
    air = F.f_motor.min(axis=1) > MIN_FLIGHT_HZ
    freqs = np.arange(BAND_HZ[0], BAND_HZ[1] + 0.5, 1.0)
    P = spectrum_true_time(F.gyro_unfilt[air], F.t[air], int(round(F.fs)), freqs)
    return float(10 * np.log10(P.sum() / 1.5 / 2))      # /1.5: Hann noise bandwidth in 1 Hz bins
