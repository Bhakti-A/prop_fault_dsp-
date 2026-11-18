"""Fault diagnosis by comparing each motor with the other three (no learning, no training data).

The four motors are the same model on the same frame, so in every 1 s window they act as each
other's reference.  For motor k:

    peer_delta_k = L_k - median(L_j, j != k)        [dB]

where L is the motor's tracked 1x vibration level (order_tracking.tracked_orders).  Speed,
throttle, battery and manoeuvres change all four motors together and cancel out of this
difference; a fault on one motor does not.

    detection    : the flight-median peer_delta of a motor is >= ALERT_DB
    localization : the motor(s) that cross it; the largest one is the worst motor
    severity     : the flight-median peer_delta in dB, graded
                   ALERT_DB  (6 dB)  = twice the vibration amplitude of the other motors
                   SEVERE_DB (12 dB) = four times
"""
from __future__ import annotations

import numpy as np

ALERT_DB = 6.0
SEVERE_DB = 12.0
MOTORS = ["M1", "M2", "M3", "M4"]


def peer_delta(A1: np.ndarray) -> np.ndarray:
    """A1 (n_win, 4) tracked 1x amplitude -> (n_win, 4) level minus median of the other three [dB]."""
    L = 20 * np.log10(A1)
    return np.stack([L[:, k] - np.median(np.delete(L, k, axis=1), axis=1)
                     for k in range(L.shape[1])], axis=1)


def grade(db: float) -> str:
    if db >= SEVERE_DB:
        return "severe"
    if db >= ALERT_DB:
        return "fault"
    return "normal"


def diagnose(A1: np.ndarray) -> dict:
    d = peer_delta(A1)
    med = np.median(d, axis=0)
    worst = int(np.argmax(med))
    flagged = [MOTORS[k] for k in range(len(med)) if med[k] >= ALERT_DB]
    return {"delta": d, "median_delta": med,
            "frac_windows_alert": (d >= ALERT_DB).mean(axis=0),
            "flagged": flagged, "fault": bool(flagged),
            "worst_motor": MOTORS[worst], "severity_db": float(med[worst]),
            "grade": grade(float(med[worst]))}
