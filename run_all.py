"""Reproduce every number and figure in README.md. Pure signal processing, no ML.

    .venv/bin/python -I run_all.py

Inputs (not in git, see README "How to run"):
    data/air75/*.csv            Betaflight blackbox CSV exports, 20-22 Sept (main analysis)
    data/bbl/<date>/*.csv       optional: earlier logs decoded from .BBL files (trend over dates)
Outputs: results/tables/*.csv|*.md, results/figures/*.png, results/summary.json
"""
import glob
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import numpy as np
import pandas as pd

import figures as FG
from bblog import load_flight
from monitor import ALERT_DB, MOTORS, SEVERE_DB, diagnose, grade, peer_delta
from pipeline import decimated, prepare

MAIN_DIR = os.path.join(HERE, "data", "air75")
BBL_DIR = os.path.join(HERE, "data", "bbl")
MAIN_DATE = "2026-09-20/22"
MIN_WINDOWS = 20                  # at least ~10 s airborne (1 s windows, 50 % overlap)
TAB = os.path.join(HERE, "results", "tables")
os.makedirs(TAB, exist_ok=True)
T0 = time.time()
summary = {"fault_level_dB": ALERT_DB, "severe_level_dB": SEVERE_DB}


def log(msg):
    print(f"[{time.time() - T0:5.1f}s] {msg}", flush=True)


def save(df, name, index=False):
    df.to_csv(os.path.join(TAB, name + ".csv"), index=index)
    with open(os.path.join(TAB, name + ".md"), "w") as fh:
        fh.write(df.to_markdown(index=index))


def drive_per_speed(F):
    """Each motor's drive command per unit speed, relative to the median motor (1.00 = typical)."""
    spin = F.f_motor.min(axis=1) > 150
    r = np.median(F.motor_cmd[spin] / F.f_motor[spin], axis=0)
    return r / np.median(r)


def per_log_row(P, date):
    d = diagnose(P.a1())
    dps = drive_per_speed(P.flight)
    spd = np.median(P.fwin, axis=0)
    return {"date": date, "flight": P.flight.name, "log_rate_hz": round(P.flight.fs),
            "airborne_windows": len(P.starts), "speed_hz_median": round(float(np.median(spd))),
            **{f"peer_dB_{m}": round(float(d["median_delta"][k]), 1) for k, m in enumerate(MOTORS)},
            **{f"windows_at_fault_{m}": round(float(d["frac_windows_alert"][k]), 2) for k, m in enumerate(MOTORS)},
            "flagged": ", ".join(d["flagged"]) or "-", "worst_motor": d["worst_motor"],
            "severity_dB": round(d["severity_db"], 1), "grade": d["grade"],
            **{f"drive_per_speed_{m}": round(float(dps[k]), 3) for k, m in enumerate(MOTORS)},
            **{f"speed_offset_hz_{m}": round(float(spd[k] - np.median(spd)), 1) for k, m in enumerate(MOTORS)}}


# ------------------------------------------------------------------ load
log("loading logs")
seen, main, inventory = {}, [], []
for path in sorted(glob.glob(os.path.join(MAIN_DIR, "*.csv"))):
    with open(path, "rb") as fh:
        h = hashlib.sha1(fh.read()).hexdigest()
    name = os.path.basename(path)
    if h in seen:
        inventory.append({"date": MAIN_DATE, "file": name, "used": False, "reason": f"copy of {seen[h]}"})
        continue
    seen[h] = name
    P = prepare(load_flight(path))
    main.append(P)
    inventory.append({"date": MAIN_DATE, "file": name, "used": True, "reason": "",
                      "log_rate_hz": round(P.flight.fs), "duration_s": round(P.flight.t[-1], 1),
                      "airborne_windows": len(P.starts)})
if len(main) < 2:
    sys.exit(f"no logs in {MAIN_DIR} (see README)")

extra = []
for ddir in sorted(glob.glob(os.path.join(BBL_DIR, "*"))):
    date = os.path.basename(ddir)
    for path in sorted(glob.glob(os.path.join(ddir, "*.csv"))):
        F = load_flight(path)
        F.name = f"{date[5:]}_{os.path.basename(path)[:-4]}"
        P = prepare(F) if F.t[-1] >= 4 else None
        n = 0 if P is None else len(P.starts)
        ok = n >= MIN_WINDOWS
        inventory.append({"date": date, "file": F.name, "used": ok,
                          "reason": "" if ok else f"under ~10 s airborne ({n} windows)",
                          "log_rate_hz": round(F.fs), "duration_s": round(F.t[-1], 1), "airborne_windows": n})
        if ok:
            extra.append((date, P))
inv = pd.DataFrame(inventory)
save(inv, "logs_inventory")
summary["main_flights"] = len(main)
summary["main_airborne_windows"] = int(sum(len(p.starts) for p in main))
summary["extra_logs"] = {"decoded": int((inv.date != MAIN_DATE).sum()), "used": len(extra)}

# ------------------------------------------------------------------ vibration levels (main)
log("order levels")
rows = []
for P in main:
    Lu, Lf = 20 * np.log10(P.orders_unfilt[:, :, 0]), 20 * np.log10(P.orders_filt[:, :, 0])
    for k, m in enumerate(MOTORS):
        rows.append({"flight": P.flight.name, "motor": m, "speed_hz_median": round(float(np.median(P.fwin[:, k])), 1),
                     "L1x_dB": round(float(np.median(Lu[:, k])), 1),
                     "betaflight_filter_attenuation_dB": round(float(np.median(Lu[:, k] - Lf[:, k])), 1)})
levels = pd.DataFrame(rows)
save(levels, "order_levels_by_flight")
att = levels.betaflight_filter_attenuation_dB
summary["betaflight_filter_attenuation_dB"] = {"min": float(att.min()), "median": float(att.median()),
                                               "max": float(att.max())}

# ------------------------------------------------------------------ diagnosis (main)
log("diagnosis: each motor vs the other three")
diag = pd.DataFrame([per_log_row(P, MAIN_DATE) for P in main])
save(diag[["flight", "airborne_windows"] + [f"peer_dB_{m}" for m in MOTORS] + ["windows_at_fault_M1",
           "windows_at_fault_M3", "flagged", "worst_motor", "severity_dB", "grade"]], "diagnosis_per_flight")
srows = []
for m in MOTORS:
    v = diag[f"peer_dB_{m}"]
    srows.append({"motor": m, "flights_flagged": f"{int((v >= ALERT_DB).sum())} / {len(v)}",
                  "flights_severe": int((v >= SEVERE_DB).sum()),
                  "median_dB": round(float(v.median()), 1), "min_dB": round(float(v.min()), 1),
                  "max_dB": round(float(v.max()), 1), "verdict": grade(float(v.median()))})
dsum = pd.DataFrame(srows)
save(dsum, "diagnosis_summary")
summary["diagnosis_summary"] = dsum.to_dict(orient="records")
summary["M1_windows_at_fault_median"] = float(diag.windows_at_fault_M1.median())
summary["M3_windows_at_fault_median"] = float(diag.windows_at_fault_M3.median())
summary["unflagged_motors_M2_M4_peer_dB_range"] = [float(diag[["peer_dB_M2", "peer_dB_M4"]].min().min()),
                                                   float(diag[["peer_dB_M2", "peer_dB_M4"]].max().max())]

# log rate check: the 3.2 kHz flights cut down to 808 Hz give the same verdict
chk = []
for P in main:
    if P.flight.fs > 2400:
        Q = prepare(decimated(P.flight, 4))
        chk.append({"flight": P.flight.name,
                    "M1_peer_dB_3200Hz": round(float(np.median(peer_delta(P.a1())[:, 0])), 2),
                    "M1_peer_dB_808Hz": round(float(np.median(peer_delta(Q.a1())[:, 0])), 2)})
chk = pd.DataFrame(chk)
chk["difference_dB"] = (chk.M1_peer_dB_808Hz - chk.M1_peer_dB_3200Hz).round(2)
save(chk, "log_rate_check")
summary["log_rate_check_max_abs_diff_dB"] = float(chk.difference_dB.abs().max())

# ------------------------------------------------------------------ trend and drive (all dates)
log("trend over dates")
trend = pd.DataFrame([per_log_row(P, d) for d, P in extra] + diag.to_dict(orient="records"))
save(trend, "all_logs_by_date")
tby = trend.groupby("date", sort=False).agg(
    logs=("flight", "size"),
    **{f"{m}_peer_dB_median": (f"peer_dB_{m}", "median") for m in MOTORS},
    M3_drive_per_speed_median=("drive_per_speed_M3", "median"),
    M3_speed_offset_hz_median=("speed_offset_hz_M3", "median")).round(3)
tby.insert(0, "date", tby.index)
save(tby, "trend_by_date")
summary["trend_by_date"] = tby.set_index("date").to_dict(orient="index")
drv = trend.groupby("date", sort=False)[[f"drive_per_speed_{m}" for m in MOTORS] +
                                        [f"speed_offset_hz_{m}" for m in MOTORS]].median().round(3)
drv.insert(0, "date", drv.index)
save(drv, "drive_and_speed_by_date")
summary["M3_drive_per_speed_main_range"] = [float(diag.drive_per_speed_M3.min()), float(diag.drive_per_speed_M3.max())]
summary["M3_slowest_in_main_flights"] = int((diag[[f"speed_offset_hz_{m}" for m in MOTORS]].idxmin(axis=1)
                                             == "speed_offset_hz_M3").sum())

# ------------------------------------------------------------------ figures
log("figures")
ex = next(P for P in main if P.flight.name == "log_3")
FG.spectrogram_with_erpm(ex.flight)
FG.unfilt_vs_filtered_psd(ex.flight)
FG.order_spectrum_per_motor(ex)
FG.levels_by_flight(main)
FG.peer_heatmap(diag)
if extra:
    FG.trend_plot(trend)
    FG.load_trend(trend)

with open(os.path.join(HERE, "results", "summary.json"), "w") as fh:
    json.dump(summary, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
log("done")
pd.set_option("display.width", 250)
print(dsum.to_string(index=False))
print(diag[["flight"] + [f"peer_dB_{m}" for m in MOTORS] + ["windows_at_fault_M1", "flagged", "grade"]].to_string(index=False))
print(tby.to_string(index=False))
print(drv.to_string(index=False))
print(chk.to_string(index=False))
print(json.dumps({k: summary[k] for k in ("betaflight_filter_attenuation_dB", "unflagged_motors_M2_M4_peer_dB_range",
                                          "M3_drive_per_speed_main_range", "M3_slowest_in_main_flights",
                                          "extra_logs", "M1_windows_at_fault_median",
                                          "M3_windows_at_fault_median")}))
