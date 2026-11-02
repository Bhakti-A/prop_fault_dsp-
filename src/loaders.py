"""Dataset loaders -> common schema.

Each loader returns a list of Recording objects:
    signals        : float array (n_channels, n_samples)
    ch_names       : list[str]
    fs             : sampling rate [Hz]
    label_health   : 0 healthy / 1 faulty
    label_motor    : PADRE: 4-char string of per-rotor fault digits 'ABCD' (e.g. '0102');
                     DronePropB: None (rotor not encoded in the dataset)
    label_severity : DronePropB: 0..3 (SV code); PADRE: None (no severity in dataset)
    experiment_id  : unique id of the physical experiment (used as CV group)
    meta           : dict of extra info (fault type, speed, config code, date ...)
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np
import scipy.io as sio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")


@dataclass
class Recording:
    signals: np.ndarray
    ch_names: list
    fs: float
    label_health: int
    label_motor: str | None
    label_severity: int | None
    experiment_id: str
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- PADRE
PADRE_FAULT_NAMES = {0: "none", 1: "chipped edge", 2: "bent tip", 3: "cut-off tip",
                     4: "bent tip + cut-off tip"}
ACC_LSB_PER_G = 2048.0     # +-16 g  (assumed InvenSense sensitivity)
GYR_LSB_PER_DPS = 32.8     # +-1000 dps (assumed InvenSense sensitivity)


def _decode_padre_file(path: str) -> np.ndarray:
    with open(path) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    raw = np.frombuffer(bytes.fromhex("".join(lines)), dtype=">i2").reshape(len(lines), 24)
    return raw.astype(np.float64)


def load_padre(trim_start_s: float = 10.0, trim_end_s: float = 5.0, cache: bool = True):
    """Parrot Bebop 2 raw IMU flights (500 Hz, 4 IMUs x (acc xyz, gyro xyz))."""
    folder = os.path.join(RAW, "padre", "Parrot_Bebop_2", "Raw_data")
    cache_f = os.path.join(ROOT, "data", "processed", "padre_raw.npz")
    files = sorted(glob.glob(os.path.join(folder, "Bebop2_*_raw_*.csv")))
    if not files:
        raise FileNotFoundError(folder)
    fs = 500.0
    if cache and os.path.exists(cache_f):
        z = np.load(cache_f)
        arrays = {k: z[k] for k in z.files}
    else:
        arrays = {re.search(r"_(\d{4})\.csv$", f).group(1): _decode_padre_file(f) for f in files}
        os.makedirs(os.path.dirname(cache_f), exist_ok=True)
        np.savez_compressed(cache_f, **arrays)
    ch_names, scale = [], []
    for rot in "ABCD":
        for kind, s in (("acc", ACC_LSB_PER_G), ("gyr", GYR_LSB_PER_DPS)):
            for ax in "xyz":
                ch_names.append(f"{rot}_{kind}_{ax}")
                scale.append(s)
    scale = np.array(scale)[:, None]
    recs = []
    for code, raw in sorted(arrays.items()):
        x = (raw.T / scale)
        i0, i1 = int(trim_start_s * fs), x.shape[1] - int(trim_end_s * fs)
        digits = [int(c) for c in code]
        recs.append(Recording(
            signals=x[:, i0:i1], ch_names=ch_names, fs=fs,
            label_health=int(any(digits)), label_motor=code, label_severity=None,
            experiment_id=f"padre_{code}",
            meta={"dataset": "PADRE", "code": code,
                  "fault_types": [PADRE_FAULT_NAMES[d] for d in digits],
                  "n_faulty_rotors": int(sum(d > 0 for d in digits)),
                  "saturation_frac": float(np.mean(np.abs(raw[i0:i1]) >= 32767))}))
    return recs


# ----------------------------------------------------------------------- DronePropB
DPB_RE = re.compile(r"F(\d)_SV(\d)_SP(\d)_C(\d)(?:_R(\d))?\.mat$")


def _active_region(x: np.ndarray, fs: float, frac: float = 0.3, margin_s: float = 1.0):
    """Indices of the steady 'motors running' part, from a moving-RMS envelope."""
    n = int(0.5 * fs)
    nb = len(x) // n
    rms = x[: nb * n].reshape(nb, n).std(axis=1)
    thr = frac * np.median(np.sort(rms)[nb // 2:])
    act = np.where(rms > thr)[0]
    if len(act) == 0:
        return 0, len(x)
    m = int(margin_s * fs)
    return act[0] * n + m, (act[-1] + 1) * n - m


def load_dronepropb(sensor: str = "piezo", include_background: bool = False,
                    trim_s: float = 1.0):
    """DronePropB ground tests.

    sensor='piezo' -> 1 ch at 10 kHz (AccData); sensor='imu' -> 3 ch at 1 kHz (IMUData).
    Duplicate files (identical data stored under two names, e.g. F0 C1 == F0 C2) are dropped.
    SP0 (motors off, piezo only) files are excluded unless include_background=True.
    """
    folder = os.path.join(RAW, "dronepropb")
    files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    recs, seen = [], {}
    for f in files:
        name = os.path.basename(f)
        m = DPB_RE.search(name)
        F, SV, SP, C = (int(m.group(i)) for i in range(1, 5))
        R = int(m.group(5)) if m.group(5) else 1
        if SP == 0 and not include_background:
            continue
        mat = sio.loadmat(f, squeeze_me=True)
        acc = mat["AccData"]
        piezo = np.asarray(acc["Vib_Data"].item(), float).ravel()
        fp = (round(float(piezo[:2000].sum()), 9), len(piezo))   # duplicate fingerprint
        if fp in seen:
            continue
        seen[fp] = name
        date = str(acc["DATE"].item())
        if sensor == "piezo":
            fs = float(acc["samplingRate"].item())
            i0, i1 = int(trim_s * fs), len(piezo) - int(trim_s * fs)
            sig, chn = piezo[None, i0:i1], ["piezo"]
        elif sensor == "imu":
            if "IMUData" not in mat:
                continue
            imu = mat["IMUData"]
            fs = float(imu["samplingRate"].item())
            v = np.asarray(imu["Vib_Data"].item(), float)
            i0, i1 = _active_region(v[2] - v[2].mean(), fs)
            sig, chn = v[:, i0:i1], ["imu_x", "imu_y", "imu_z"]
        else:
            raise ValueError(sensor)
        recs.append(Recording(
            signals=sig, ch_names=chn, fs=fs,
            label_health=int(F > 0), label_motor=None, label_severity=SV,
            experiment_id=name[:-4],
            meta={"dataset": "DronePropB", "F": F, "SV": SV, "SP": SP, "C": C, "R": R,
                  "date": date, "file": name}))
    return recs
