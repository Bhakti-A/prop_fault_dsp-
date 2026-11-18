"""Survey a folder of extra blackbox CSVs (e.g. decoded from a .BBL with blackbox_decode) against
the 16-flight baseline.  Screening only: these logs are not part of the main results.

    .venv/bin/python -I survey_logs.py <dir with */*.csv or *.csv>

For every log with at least ~5 s airborne (all motors above 150 Hz) it reports each motor's 1x
level, motor 4 relative to the others, command needed per speed^2 (relative), sudden speed drops,
and, when the motors run in the baseline's speed range (median >= 290 Hz), the monitor's verdict.
"""
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import numpy as np
import pandas as pd

import pipeline
from bblog import load_flight
from monitor import decide
from pipeline import baseline_from, prepare

lab = pd.read_csv(os.path.join(HERE, "flight_labels.csv"))
bl, _, _ = baseline_from([prepare(load_flight(os.path.join(HERE, "data", "air75", n))) for n in lab.file])
pipeline.MIN_FLIGHT_HZ = 150.0
paths = sorted(glob.glob(os.path.join(sys.argv[1], "*", "*.csv")) + glob.glob(os.path.join(sys.argv[1], "*.csv")))
rows, short, grounded = [], 0, 0
for path in paths:
    F = load_flight(path)
    if F.t[-1] < 4:
        short += 1
        continue
    P = prepare(F)
    if len(P.starts) < 8:
        grounded += 1
        continue
    fm = F.f_motor
    spin = fm.min(axis=1) > 150
    d = np.diff(fm, axis=0) / np.maximum(fm[:-1], 1)
    L = np.median(20 * np.log10(P.a1()), axis=0)
    eff = np.median(F.motor_cmd[spin] / fm[spin] ** 2, axis=0)
    eff /= eff.mean()
    sp = float(np.median(P.fwin))
    r = {"log": os.path.basename(path)[:-4], "log_rate_hz": round(F.fs), "duration_s": round(F.t[-1], 1),
         "airborne_windows": len(P.starts), "speed_hz_median": round(sp),
         **{f"L1x_dB_M{k + 1}": round(L[k], 1) for k in range(4)},
         "M4_minus_median_of_others_dB": round(L[3] - np.median(L[:3]), 1),
         **{f"cmd_per_speed2_M{k + 1}": round(eff[k], 2) for k in range(4)},
         "sudden_speed_drops": int(((d < -0.3) & (fm[:-1] > 150)).sum()),
         "flight_score": "", "alarm": "", "worst_motor": ""}
    if sp >= 290:
        dd = decide(bl, P.a1(), P.fwin)
        r.update(flight_score=round(dd["flight_score"], 2), alarm=dd["flight_alarm"],
                 worst_motor=f"M{dd['motor_flight'] + 1}")
    rows.append(r)
df = pd.DataFrame(rows)
out = os.path.join(HERE, "results", "tables", "extra_logs_survey")
df.to_csv(out + ".csv", index=False)
with open(out + ".md", "w") as fh:
    fh.write(df.to_markdown(index=False))
print(f"{len(paths)} logs: {short} under 4 s, {grounded} with under ~5 s airborne, {len(df)} surveyed")
print(df.to_string(index=False))
