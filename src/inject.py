"""SYNTHETIC fault injection (clearly not real fault data).

A damaged/unbalanced propeller on motor k adds a once-per-revolution force that rotates with
the shaft; its magnitude grows with speed squared (centrifugal force m*e*omega^2).  We model
its effect on the body rates as a rotating vector in roll/pitch:
    roll  += a(t) * cos(theta_k(t) + phi)
    pitch += a(t) * sin(theta_k(t) + phi)
    a(t)   = a0 * (f_k(t) / f_ref)^2
theta_k comes from motor k's REAL eRPM, so the tone follows the real speed profile and sits among
the real vibration of all four motors.  a0 is set so that the injected tracked 1x amplitude
equals r times motor k's healthy median 1x amplitude (r = "imbalance ratio").

What this tests: whether the DSP chain (order tracking + healthy statistics) can find and
attribute a 1x imbalance of a given size on this airframe's real noise.  What it does NOT test:
whether a real damaged AIR75 prop produces exactly this signature (needs real damaged-prop flights).
"""
from __future__ import annotations

import numpy as np


def inject_imbalance(gyro: np.ndarray, theta: np.ndarray, f_motor: np.ndarray, motor: int,
                     a0: float, f_ref: float, phi: float) -> np.ndarray:
    a = a0 * (f_motor[:, motor] / f_ref) ** 2
    th = theta[:, motor] + phi
    out = gyro.copy()
    out[:, 0] += a * np.cos(th)
    out[:, 1] += a * np.sin(th)
    return out


def a0_for_ratio(r: float, healthy_median_amp: float) -> float:
    # tracked amplitude of the rotating vector = sqrt(2) * a (roll and pitch each carry a)
    return r * healthy_median_amp / np.sqrt(2)
