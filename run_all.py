"""Reproduce every result in README.md (PADRE dataset only; light on CPU).

    nice -n 10 .venv/bin/python -I run_all.py           # uses cached features in data/processed
    nice -n 10 .venv/bin/python -I run_all.py --force   # recompute features from data/raw

Load control: Random Forest uses 4 cores; the leave-one-flight-out runs use RF only and train on
every 2nd window (50 %-overlap windows are redundant; test windows are never subsampled).
Outputs: results/tables/*.csv|*.md, results/figures/*.png, results/summary.json
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import numpy as np
import pandas as pd

import figures as FG
from loaders import load_padre
from stage1 import (TAB, padre_detection, padre_energy_by_fault_type, padre_features,
                    padre_leakage_demo, padre_localization)

FORCE = "--force" in sys.argv
summary = {}
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def save(df, name, index=True):
    df.to_csv(os.path.join(TAB, name + ".csv"), index=index)
    with open(os.path.join(TAB, name + ".md"), "w") as fh:
        fh.write(df.to_markdown(index=index))


# ------------------------------------------------------------------ features
log("features (cached unless --force)")
pf, pfn = padre_features(False, FORCE), padre_features(True, FORCE)
summary["n_windows"] = len(pf)
summary["n_flights"] = int(pf.experiment_id.nunique())

# ------------------------------------------------------------------ signal figures
log("signal figures")
recs = load_padre()
FG.padre_psd(recs)
FG.padre_spectrograms(recs)
FG.padre_notch_effect(recs)
del recs

# ------------------------------------------------------------------ (1) detection
log("detection: threshold + SVM/RF/kNN, time-block CV")
det_raw, cms = padre_detection(pf, "raw")
det_notch, _ = padre_detection(pfn, "notched")
save(det_raw, "padre_detection_raw")
save(det_notch, "padre_detection_notched")
FG.confusion(cms["SVM-RBF"], ["healthy", "faulty"],
             "PADRE detection, SVM, time-block CV (windows)", "cm_padre_detection_svm.png")
summary["detection_raw"] = det_raw.to_dict(orient="index")
summary["detection_notched"] = det_notch.to_dict(orient="index")

# ------------------------------------------------------------------ (2) localization
log("localization: RF, leave-one-flight-out (raw)")
loc, types, fl = padre_localization(pf, "LOFO", "raw", models=["RandomForest"], train_stride=2)
save(loc, "padre_localization_LOFO_raw", index=False)
save(types, "padre_localization_LOFO_by_fault_type", index=False)
summary["localization_LOFO_raw"] = loc.to_dict(orient="records")
summary["localization_LOFO_by_fault_type"] = types.to_dict(orient="records")
fp_, ft_ = fl["RandomForest"]
cm = np.zeros((2, 2), int)
for a, b in zip(ft_.loc[fp_.index].to_numpy().ravel(), fp_.to_numpy().ravel()):
    cm[a, b] += 1
FG.confusion(cm, ["rotor healthy", "rotor faulty"],
             "PADRE per-rotor localization, RF, leave-one-flight-out\n(28 flights x 4 rotors, majority vote)",
             "cm_padre_localization_rf_lofo.png")
per_flight = fp_.copy()
per_flight.columns = [f"pred_{r}" for r in "ABCD"]
per_flight.insert(0, "true_code", [i.split("_")[1] for i in per_flight.index])
per_flight["pred_code"] = per_flight[[f"pred_{r}" for r in "ABCD"]].astype(str).agg("".join, axis=1)
per_flight["true_faulty_pattern"] = ["".join("1" if c != "0" else "0" for c in s) for s in per_flight.true_code]
per_flight["exact_match"] = per_flight.pred_code == per_flight.true_faulty_pattern
save(per_flight, "padre_localization_LOFO_rf_per_flight")

log("localization: RF, leave-one-flight-out (notch-filtered)")
loc_n, _, _ = padre_localization(pfn, "LOFO", "notched", models=["RandomForest"], train_stride=2)
save(loc_n, "padre_localization_LOFO_notched", index=False)
summary["localization_LOFO_notched"] = loc_n.to_dict(orient="records")

# ------------------------------------------------------------------ (3) severity proxy
log("spectral energy change by fault type (descriptive; PADRE has no severity labels)")
et, eby = padre_energy_by_fault_type(pf)
save(et, "padre_energy_change_single_fault_flights", index=False)
save(eby, "padre_energy_change_by_fault_type")
summary["energy_change_by_fault_type"] = eby.to_dict(orient="index")
FG.padre_energy_by_type(et)

# ------------------------------------------------------------------ raw vs notch
log("raw vs notch summary")
rotor_mean = lambda t: t[t.rotor.isin(list("ABCD"))].balanced_acc.mean()
cmp_tab = pd.DataFrame({
    "Detection, threshold on 1x level": (det_raw.loc["threshold_H1(acc_z)", "balanced_acc"],
                                         det_notch.loc["threshold_H1(acc_z)", "balanced_acc"]),
    "Detection, threshold on RMS": (det_raw.loc["threshold_RMS(acc)", "balanced_acc"],
                                    det_notch.loc["threshold_RMS(acc)", "balanced_acc"]),
    "Detection, SVM": (det_raw.loc["SVM-RBF", "balanced_acc"], det_notch.loc["SVM-RBF", "balanced_acc"]),
    "Detection, RF": (det_raw.loc["RandomForest", "balanced_acc"], det_notch.loc["RandomForest", "balanced_acc"]),
    "Localization, RF (LOFO, mean of 4 rotors)": (rotor_mean(loc), rotor_mean(loc_n)),
}, index=["raw", "notched"]).T.round(3)
save(cmp_tab, "raw_vs_notch_summary")
summary["raw_vs_notch"] = cmp_tab.to_dict(orient="index")
FG.raw_vs_notch_bars(cmp_tab, "raw_vs_notch.png", "PADRE: raw vs software notch-filtered (balanced accuracy)")

# ------------------------------------------------------------------ leakage demo
log("leakage demo (deliberately invalid random-window split)")
lk = padre_leakage_demo(pf)
save(lk, "leakage_demo_random_window_split", index=False)
summary["leakage_demo"] = lk.to_dict(orient="records")

with open(os.path.join(HERE, "results", "summary.json"), "w") as fh:
    json.dump(summary, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
log("done")
