"""Reproduce every number and figure in README.md. Pure signal processing, no ML.

    .venv/bin/python -I run_all.py                 # logs in data/air75/*.csv
    .venv/bin/python -I run_all.py --data <dir>    # another folder of Betaflight blackbox CSVs

flight_labels.csv says which propeller set each log was flown on (pilot's record):
"new" = after all four propellers were replaced (healthy reference), "old" = before.
Runs in about 30 s on a laptop (single core).
Outputs: results/tables/*.csv|*.md, results/figures/*.png, results/summary.json
"""
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
from inject import a0_for_ratio, inject_imbalance
from monitor import FLIGHT_Z, decide
from pipeline import BAND_HZ, MOTORS, band_power_db, baseline_from, decimated, evaluate, prepare

DATA = sys.argv[sys.argv.index("--data") + 1] if "--data" in sys.argv else os.path.join(HERE, "data", "air75")
TAB = os.path.join(HERE, "results", "tables")
os.makedirs(TAB, exist_ok=True)
RATIOS = [0.25, 0.5, 1, 2, 4, 8]      # injected amplitude / healthy median 1x amplitude
N_PHASE = 3
T0 = time.time()
summary = {}


def log(msg):
    print(f"[{time.time() - T0:5.1f}s] {msg}", flush=True)


def save(df, name, index=True):
    df.to_csv(os.path.join(TAB, name + ".csv"), index=index)
    with open(os.path.join(TAB, name + ".md"), "w") as fh:
        fh.write(df.to_markdown(index=index))


# ------------------------------------------------------------------ load
log(f"loading logs from {DATA}")
labels = pd.read_csv(os.path.join(HERE, "flight_labels.csv"))
seen, new, old, inventory = {}, [], [], []
for name, props in zip(labels.file, labels.props):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        sys.exit(f"missing {path} (copy the blackbox CSVs into {DATA})")
    with open(path, "rb") as fh:
        digest = hashlib.sha1(fh.read()).hexdigest()
    if digest in seen:
        sys.exit(f"{name} is a byte-identical copy of {seen[digest]}")
    seen[digest] = name
    F = load_flight(path)
    P = prepare(F)
    (new if props == "new" else old).append(P)
    inventory.append({"file": name, "props": props, "log_rate_hz": round(F.fs),
                      "duration_s": round(F.t[-1], 1), "in_flight_windows": len(P.starts),
                      "hover_speed_hz_median": round(float(np.median(P.fwin)))})
inv = pd.DataFrame(inventory)
save(inv, "flights_used", index=False)
summary["flights"] = {p: {"flights": int(n), "windows": int(w), "log_time_s": float(d)} for p, n, w, d in
                      inv.groupby("props").agg(n=("file", "size"), w=("in_flight_windows", "sum"),
                                               d=("duration_s", "sum")).itertuples()}

# ------------------------------------------------------------------ order levels (real)
log("order levels per flight")
rows = []
for props, ps in (("new", new), ("old", old)):
    for p in ps:
        Lu, Lf = 20 * np.log10(p.orders_unfilt[:, :, 0]), 20 * np.log10(p.orders_filt[:, :, 0])
        for k, m in enumerate(MOTORS):
            rows.append({"props": props, "flight": p.flight.name, "motor": m,
                         "speed_hz_median": np.median(p.fwin[:, k]), "L1x_dB": np.median(Lu[:, k]),
                         "L1x_spread_dB": 1.4826 * np.median(np.abs(Lu[:, k] - np.median(Lu[:, k]))),
                         "betaflight_filter_attenuation_dB": np.median(Lu[:, k] - Lf[:, k])})
levels = pd.DataFrame(rows).round(2)
save(levels, "order_levels_by_flight", index=False)
lv = levels.groupby(["props", "motor"]).L1x_dB.agg(["min", "median", "max"]).round(1)
save(lv, "order_levels_by_props")
att = levels.groupby("props").betaflight_filter_attenuation_dB.agg(["min", "median", "max"]).round(1)
save(att, "betaflight_filter_attenuation")
summary["L1x_dB"] = {f"{a} {b}": v for (a, b), v in lv.to_dict(orient="index").items()}
summary["betaflight_filter_attenuation_dB"] = att.to_dict(orient="index")

# ------------------------------------------------------------------ healthy reference: new props
log("new props: leave-one-flight-out false alarms + SYNTHETIC imbalance injection")
clean, inj, win = evaluate(new, RATIOS, n_phase=N_PHASE, seed=0)
save(clean.round(3), "new_props_false_alarms", index=False)
summary["new_props_as_flown"] = {"flights": len(clean), "flights_alarmed": int(clean.flight_alarm.sum()),
                                 "flight_score_max": float(clean.flight_score.max()),
                                 "windows_flagged": float(np.average(clean.false_alarm_rate_windows, weights=clean.windows)),
                                 "change_dB_min": float(clean.filter(like="excess_dB").to_numpy().min()),
                                 "change_dB_max": float(clean.filter(like="excess_dB").to_numpy().max())}
inj["located_ok"] = inj.located == inj.motor
inj["severity_err_dB"] = inj.severity_est_dB - inj.severity_nominal_dB
save(inj.round(3), "synthetic_all_cases", index=False)
by_r = inj.groupby("ratio").agg(injected_increase_dB=("severity_nominal_dB", "first"), cases=("flight", "size"),
                                flights_alarmed=("flight_alarm", "mean"), correct_motor=("located_ok", "mean"),
                                measured_increase_dB=("severity_est_dB", "median"),
                                abs_error_dB=("severity_err_dB", lambda v: v.abs().median())).round(3)
save(by_r, "synthetic_by_size")
summary["synthetic_by_ratio"] = {f"r={r:g}": v for r, v in by_r.to_dict(orient="index").items()}
d2 = inj[inj.ratio == 2]
cm = pd.crosstab(d2.motor, d2.located).reindex(index=MOTORS, columns=MOTORS, fill_value=0).to_numpy()
summary["synthetic_confusion_r2"] = cm.tolist()

# ------------------------------------------------------------------ the check: old props (REAL)
log("old props: diagnosis against the new-prop baseline (real data, nothing injected)")
bl, A_new, f_new = baseline_from(new)
rows, zs = [], {}
for p in old:
    d = decide(bl, p.a1(), p.fwin)
    dd = decide(bl, (q := prepare(decimated(p.flight, 4))).a1(), q.fwin)      # same flight at 808 Hz
    zs[p.flight.name] = np.median(d["z"], axis=0)
    rows.append({"flight": p.flight.name, "props": "old", "windows_flagged": d["frac_flagged"],
                 "flight_score": d["flight_score"], "alarm": d["flight_alarm"],
                 "worst_motor": MOTORS[d["motor_flight"]],
                 **{f"change_dB_{m}": d["severity_db"][k] for k, m in enumerate(MOTORS)},
                 "alarm_at_808Hz": dd["flight_alarm"], "worst_motor_at_808Hz": MOTORS[dd["motor_flight"]],
                 "change_dB_M1_at_808Hz": dd["severity_db"][0]})
diag_old = pd.DataFrame(rows)
diag_new = clean.rename(columns={"false_alarm_rate_windows": "windows_flagged", "flight_alarm": "alarm"})
diag_new = diag_new.assign(props="new", worst_motor="-").rename(
    columns={f"excess_dB_{m}": f"change_dB_{m}" for m in MOTORS})
cols = ["flight", "props", "windows_flagged", "flight_score", "alarm", "worst_motor"] + [f"change_dB_{m}" for m in MOTORS]
diagnosis = pd.concat([diag_new[cols], diag_old[cols]], ignore_index=True)
diagnosis["raw_band_power_dB"] = [band_power_db(p.flight) for p in new + old]
diagnosis = diagnosis.round(2)
bp = diagnosis.groupby("props").raw_band_power_dB.agg(["min", "median", "max"]).round(1)
save(bp, "raw_band_power_by_props")
summary["raw_band_power_dB"] = bp.to_dict(orient="index")
save(diagnosis, "diagnosis_per_flight", index=False)
save(diag_old.round(2), "old_props_diagnosis_detail", index=False)
summary["old_props_diagnosis"] = {
    "flights": len(diag_old), "flights_alarmed": int(diag_old.alarm.sum()),
    "flight_score_min": float(diag_old.flight_score.min()), "flight_score_max": float(diag_old.flight_score.max()),
    "worst_motor": diag_old.worst_motor.value_counts().to_dict(),
    **{f"median_change_dB_{m}": float(diag_old[f"change_dB_{m}"].median()) for m in MOTORS},
    "alarmed_at_808Hz": int(diag_old.alarm_at_808Hz.sum()),
    "worst_motor_at_808Hz": diag_old.worst_motor_at_808Hz.value_counts().to_dict(),
    "max_abs_change_in_M1_reading_at_808Hz_dB": float((diag_old.change_dB_M1_at_808Hz - diag_old.change_dB_M1).abs().max())}
conf = pd.DataFrame({"alarm": [int(diag_old.alarm.sum()), int(clean.flight_alarm.sum())],
                     "no alarm": [int((~diag_old.alarm).sum()), int((~clean.flight_alarm).sum())]},
                    index=["old props (5 flights)", "new props (11 flights)"])
save(conf, "detection_summary")
summary["detection_summary"] = conf.to_dict(orient="index")

# ------------------------------------------------------------------ figures
log("figures")
FG.spectrogram_with_erpm(old[2].flight)
FG.unfilt_vs_filtered_psd(old[2].flight)
FG.raw_vs_tracked(old[2], max(new, key=lambda p: len(p.starts)), BAND_HZ)
FG.levels_by_flight(new, old)
FG.flight_scores(diagnosis, FLIGHT_Z)
FG.change_by_motor(diagnosis)
FG.detection_curve(by_r)
FG.severity_plot(inj)
FG.confusion(cm, "SYNTHETIC imbalance, r = 2, new-prop flights", "synthetic_localization_confusion_r2.png")
ex = max(new, key=lambda p: len(p.starts))
tr = [p for p in new if p is not ex]
bl_ex, A_tr, f_tr = baseline_from(tr)
g = inject_imbalance(ex.flight.gyro_unfilt, ex.theta, ex.flight.f_motor, 2,
                     a0_for_ratio(4, np.median(A_tr[:, 2])), np.median(f_tr[:, 2]), 0.0)
FG.example_scores(ex.flight.t[ex.starts], decide(bl_ex, ex.a1(), ex.fwin)["z"],
                  decide(bl_ex, ex.a1(g), ex.fwin)["z"], bl_ex.threshold, 2, 4, ex.flight.name)

summary["settings"] = {"window_s": 1.0, "overlap": 0.5, "ratios": RATIOS, "phases_per_case": N_PHASE,
                       "flight_alarm_median_z": FLIGHT_Z}
with open(os.path.join(HERE, "results", "summary.json"), "w") as fh:
    json.dump(summary, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
log("done")
pd.set_option("display.width", 250)
print(diagnosis.to_string(index=False)); print(diag_old.round(2).iloc[:, -3:].to_string())
print(conf.to_string()); print(bp.to_string()); print(by_r.to_string()); print(lv.to_string()); print(att.to_string())
