"""Load a Betaflight blackbox CSV export (blackbox-tools / Blackbox Explorer CSV)."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FIELDS = (["loopIteration", "time", "time (us)"] + [f"gyroUnfilt[{i}]" for i in range(3)] + [f"gyroADC[{i}]" for i in range(3)]
          + [f"eRPM[{i}]" for i in range(4)] + [f"motor[{i}]" for i in range(4)])


@dataclass
class Flight:
    name: str
    fs: float
    t: np.ndarray              # s, from log start
    gyro_unfilt: np.ndarray    # (n, 3) deg/s, roll/pitch/yaw, BEFORE Betaflight filters
    gyro_filt: np.ndarray      # (n, 3) deg/s, gyroADC, AFTER Betaflight LPF + RPM + dynamic notches
    f_motor: np.ndarray        # (n, 4) mechanical rotation frequency per motor [Hz] from eRPM
    motor_cmd: np.ndarray      # (n, 4) motor output command
    header: dict = field(default_factory=dict)


def load_flight(path: str) -> Flight:
    header, skip = {}, None
    with open(path, errors="replace") as fh:
        for i, line in enumerate(fh):
            if line.startswith('"loopIteration"') or line.startswith("loopIteration"):
                skip = i
                break
            m = re.match(r'^"([^"]+)",(.*)$', line.strip())
            if m:
                header[m.group(1)] = m.group(2).strip('"')
    if skip is None:
        raise ValueError(f"{path}: no data header line")
    df = pd.read_csv(path, skiprows=skip, usecols=lambda c: c.strip() in FIELDS)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"time (us)": "time"})        # blackbox_decode naming
    # True sample times.  When the blackbox logs every n-th loop, each periodic I-frame shifts
    # the frame grid, so about 1 step in 26 is half-length: the samples are NOT evenly spaced.
    # Each frame carries its own microsecond timestamp, and everything downstream uses those.
    # (loopIteration is not used for timing: blackbox_decode renumbers it in decimated logs.)
    us = df["time"].to_numpy(float)
    t = (us - us[0]) / 1e6
    if (np.diff(t) <= 0).any():
        raise ValueError(f"{path}: timestamps are not strictly increasing")
    fs = (len(t) - 1) / t[-1]                      # mean sample rate
    dt = np.diff(t)
    header["_uneven_steps_frac"] = float((np.abs(dt - np.median(dt)) > 0.25 * np.median(dt)).mean())
    poles = int(header.get("motor_poles", 12))        # AIR75 motors: 12 poles
    # Betaflight logs eRPM/100; mechanical Hz = eRPM / (poles/2) / 60
    f_motor = df[[f"eRPM[{k}]" for k in range(4)]].to_numpy(float) * 100 / (poles / 2) / 60
    return Flight(
        name=os.path.splitext(os.path.basename(path))[0], fs=fs, t=t,
        gyro_unfilt=df[[f"gyroUnfilt[{i}]" for i in range(3)]].to_numpy(float),
        gyro_filt=df[[f"gyroADC[{i}]" for i in range(3)]].to_numpy(float),
        f_motor=f_motor, motor_cmd=df[[f"motor[{k}]" for k in range(4)]].to_numpy(float),
        header=header)
