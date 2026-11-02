"""Stage 1 (+ notch comparison): detection, localization, severity."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import recordings_to_features
from loaders import load_dronepropb, load_padre
from models import classifiers, cv_predict, feature_cols, majority_vote, summarize, SEED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
TAB = os.path.join(ROOT, "results", "tables")
os.makedirs(PROC, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# ----------------------------------------------------------------- configuration
PADRE_CFG = dict(win_s=1.0, overlap=0.5, nperseg=256, frot_band=(60, 245), block_count=6,
                 bands=[(1, 10), (10, 25), (25, 50), (50, 80), (80, 110), (110, 140),
                        (140, 170), (170, 200), (200, 250)])
DPB_SP_BAND = {1: (40, 70), 2: (80, 115), 3: (115, 150)}   # throttle(set-point)-guided search band
DPB_PIEZO_CFG = dict(win_s=1.0, overlap=0.5, nperseg=4096,
                     bands=[(10, 40), (40, 80), (80, 160), (160, 320), (320, 640),
                            (640, 1280), (1280, 2560), (2560, 5000)])
DPB_IMU_CFG = dict(win_s=1.0, overlap=0.5, nperseg=512,
                   bands=[(5, 40), (40, 80), (80, 160), (160, 320), (320, 500)])
ROTORS = "ABCD"
N_REPEATS_DPB = 5


def _cached(name, fn, force=False):
    p = os.path.join(PROC, name + ".pkl")
    if os.path.exists(p) and not force:
        return pd.read_pickle(p)
    df = fn()
    df.to_pickle(p)
    return df


def padre_features(notch=False, force=False):
    c = PADRE_CFG
    return _cached(f"padre_feat{'_notch' if notch else ''}", lambda: recordings_to_features(
        load_padre(), c["win_s"], c["overlap"], c["bands"], c["frot_band"], c["nperseg"],
        notch=notch, block_count=c["block_count"]), force)


def dpb_features(sensor="piezo", notch=False, force=False):
    c = DPB_PIEZO_CFG if sensor == "piezo" else DPB_IMU_CFG
    return _cached(f"dpb_{sensor}_feat{'_notch' if notch else ''}", lambda: recordings_to_features(
        load_dronepropb(sensor), c["win_s"], c["overlap"], c["bands"],
        lambda r: DPB_SP_BAND[r.meta["SP"]], c["nperseg"], notch=notch), force)


def _fmt(res, keys=("accuracy", "balanced_acc", "macro_F1")):
    return {k: round(float(res[k]), 3) for k in keys}


# ======================================================================== PADRE
def threshold_detector(df, train_idx, test_idx, score_cols, healthy_mask, q=99):
    """Healthy-baseline threshold: z-score each scalar vs healthy TRAIN windows, take max |z|
    over the given channels; flag faulty if above the q-th percentile of healthy-train max|z|."""
    S = df[score_cols].to_numpy()
    h = train_idx[healthy_mask[train_idx]]
    if len(h) < 2:                       # no healthy baseline in this training fold
        return np.full(len(test_idx), -1)
    mu, sd = S[h].mean(0), S[h].std(0) + 1e-12
    z = np.abs((S - mu) / sd).max(axis=1)
    thr = np.percentile(z[h], q)
    return (z[test_idx] > thr).astype(int)


def padre_detection(df, tag="raw"):
    X = np.nan_to_num(df[feature_cols(df)].to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    y = df["label_health"].to_numpy()
    blocks = df["block"].to_numpy()
    folds = [(np.where(blocks != b)[0], np.where(blocks == b)[0]) for b in np.unique(blocks)]
    out, cms = {}, {}
    healthy = y == 0
    for name, cols in {"threshold_H1(acc_z)": [f"{r}_acc_z_H1" for r in ROTORS],
                       "threshold_RMS(acc)": [f"{r}_acc_{a}_rms" for r in ROTORS for a in "xyz"]}.items():
        pred = np.full(len(y), -1)
        for tr, te in folds:
            pred[te] = threshold_detector(df, tr, te, cols, healthy)
        res = summarize(y, pred, [0, 1]); out[name] = _fmt(res) | {"recall_healthy": round(res["per_class_recall"][0], 3), "recall_faulty": round(res["per_class_recall"][1], 3)}
    for name, model in classifiers().items():
        pred = cv_predict(model, X, y, folds)
        res = summarize(y, pred, [0, 1]); cms[name] = res["cm"]
        out[name] = _fmt(res) | {"recall_healthy": round(res["per_class_recall"][0], 3), "recall_faulty": round(res["per_class_recall"][1], 3)}
    t = pd.DataFrame(out).T
    t.index.name = f"PADRE detection, time-block CV (6 blocks), {tag}"
    return t, cms


def padre_localization(df, protocol="LOFO", tag="raw", models=None, train_stride=1):
    """Per-rotor binary 'is this rotor's propeller faulty?' classifiers."""
    X = np.nan_to_num(df[feature_cols(df)].to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    codes = df["label_motor"].to_numpy()
    groups = df["experiment_id"].to_numpy()
    if protocol == "LOFO":
        folds = [(np.where(groups != g)[0], np.where(groups == g)[0]) for g in np.unique(groups)]
    else:
        b = df["block"].to_numpy()
        folds = [(np.where(b != k)[0], np.where(b == k)[0]) for k in np.unique(b)]
    mods = classifiers()
    if models:
        mods = {k: v for k, v in mods.items() if k in models}
    rows, type_rows, preds_all = [], [], {}
    for name, model in mods.items():
        P = np.zeros((len(df), 4), int)
        for j, r in enumerate(ROTORS):
            digit = np.array([int(c[j]) for c in codes])
            y = (digit > 0).astype(int)
            P[:, j] = cv_predict(model, X, y, folds, train_stride)
            res = summarize(y, P[:, j], [0, 1])
            rows.append({"model": name, "rotor": r, **_fmt(res),
                         "recall_healthy_rotor": round(res["per_class_recall"][0], 3),
                         "recall_faulty_rotor": round(res["per_class_recall"][1], 3)})
            for t in range(5):
                m = digit == t
                if m.any():
                    type_rows.append({"model": name, "rotor": r, "fault_type": t,
                                      "n": int(m.sum()), "pred_faulty": int((P[m, j] == 1).sum())})
        Y = np.array([[int(c[j]) > 0 for j in range(4)] for c in codes]).astype(int)
        exact_w = float((P == Y).all(1).mean())
        flight_pred = pd.DataFrame(P).groupby(groups).agg(lambda v: int(np.mean(v) > 0.5))
        flight_true = pd.DataFrame(Y).groupby(groups).first()
        exact_f = float((flight_pred.values == flight_true.loc[flight_pred.index].values).all(1).mean())
        rotor_f = float((flight_pred.values == flight_true.loc[flight_pred.index].values).mean())
        rows.append({"model": name, "rotor": "ALL (exact 4-bit match, windows)", "accuracy": round(exact_w, 3)})
        rows.append({"model": name, "rotor": "ALL (exact 4-bit match, flights, majority vote)", "accuracy": round(exact_f, 3)})
        rows.append({"model": name, "rotor": "ALL (per-rotor acc, flights, majority vote)", "accuracy": round(rotor_f, 3)})
        preds_all[name] = (flight_pred, flight_true)
    tt = pd.DataFrame(type_rows).groupby(["model", "fault_type"])[["n", "pred_faulty"]].sum()
    tt["rate_pred_faulty"] = (tt["pred_faulty"] / tt["n"]).round(3)
    return pd.DataFrame(rows), tt.reset_index(), preds_all


# =================================================================== DronePropB
def dpb_folds(df, seed):
    rec = df.groupby("experiment_id").first()
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    g = df["experiment_id"].to_numpy()
    folds = []
    for tr_r, te_r in sgk.split(rec.index.to_numpy(), rec["label_severity"].to_numpy(), rec.index.to_numpy()):
        te_ids = set(rec.index[te_r])
        te = np.where(np.isin(g, list(te_ids)))[0]
        tr = np.where(~np.isin(g, list(te_ids)))[0]
        folds.append((tr, te))
    return folds


def sev3(sv):
    """Our grouping (not from the dataset): SV0 none, SV1 minor, SV2-SV3 major."""
    return np.where(sv == 0, 0, np.where(sv == 1, 1, 2))


DPB_TASKS = {
    "detection (healthy/faulty)": lambda d: d["label_health"].to_numpy(),
    "fault type (F0-F3)": lambda d: d["meta_F"].to_numpy(),
    "severity (SV0-SV3)": lambda d: d["label_severity"].to_numpy(),
    "severity 3-class (none/minor/major)": lambda d: sev3(d["label_severity"].to_numpy()),
}


def dpb_threshold(df, folds):
    """Healthy baseline per speed set-point; two-sided max|z| on H1 and RMS of the sensor."""
    y = df["label_health"].to_numpy()
    sp = df["meta_SP"].to_numpy()
    chans = [c[:-3] for c in df.columns if c.endswith("_H1")]
    out = {}
    for name, cols in {"threshold_H1": [f"{c}_H1" for c in chans],
                       "threshold_RMS": [f"{c}_rms" for c in chans]}.items():
        pred = np.full(len(y), -1)
        for tr, te in folds:
            for s in np.unique(sp):
                trs, tes = tr[sp[tr] == s], te[sp[te] == s]
                pred[tes] = threshold_detector(df, trs, tes, cols, y == 0)
        out[name] = pred
    return out


DPB_EXCLUDE = ("f_rot",)   # f_rot is confounded with battery voltage / test order -> not an input


def dpb_X(df):
    return np.nan_to_num(df[feature_cols(df, extra_exclude=DPB_EXCLUDE)].to_numpy(), nan=0.0,
                         posinf=0.0, neginf=0.0)


def dpb_classification(df, tag, models=None, repeats=N_REPEATS_DPB, tasks=None,
                       with_threshold=True):
    X = dpb_X(df)
    groups = df["experiment_id"].to_numpy()
    mods = classifiers()
    if models:
        mods = {k: v for k, v in mods.items() if k in models}
    tasks = tasks or list(DPB_TASKS)
    rows, cms, rec_preds = [], {}, {}
    for seed in range(repeats):
        folds = dpb_folds(df, SEED + seed)
        if with_threshold and "detection (healthy/faulty)" in tasks:
            y = DPB_TASKS["detection (healthy/faulty)"](df)
            for name, pred in dpb_threshold(df, folds).items():
                res = summarize(y, pred, [0, 1])
                rv = majority_vote(groups, pred); ry = pd.Series(y, index=groups).groupby(level=0).first()
                rres = summarize(ry.loc[rv.index].to_numpy(), rv.to_numpy(), [0, 1])
                rows.append({"task": "detection (healthy/faulty)", "model": name, "seed": seed,
                             **{f"win_{k}": v for k, v in _fmt(res).items()},
                             **{f"rec_{k}": v for k, v in _fmt(rres).items()},
                             "win_recall_healthy": res["per_class_recall"][0]})
        for task in tasks:
            y = DPB_TASKS[task](df)
            labels = sorted(np.unique(y))
            for name, model in mods.items():
                pred = cv_predict(model, X, y, folds)
                res = summarize(y, pred, labels)
                rv = majority_vote(groups, pred); ry = pd.Series(y, index=groups).groupby(level=0).first()
                rres = summarize(ry.loc[rv.index].to_numpy(), rv.to_numpy(), labels)
                row = {"task": task, "model": name, "seed": seed,
                       **{f"win_{k}": v for k, v in _fmt(res).items()},
                       **{f"rec_{k}": v for k, v in _fmt(rres).items()}}
                for l, v in res["per_class_recall"].items():
                    row[f"win_recall_class{l}"] = v
                rows.append(row)
                if seed == 0:
                    cms[(task, name)] = (res["cm"], labels, rres["cm"])
                    rec_preds[(task, name)] = pd.DataFrame({"true": ry.loc[rv.index], "pred": rv})
    R = pd.DataFrame(rows)
    num = [c for c in R.columns if c.startswith(("win_", "rec_"))]
    agg = R.groupby(["task", "model"], sort=False)[num].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return R, agg.round(3), cms, rec_preds


def dpb_session_check(df):
    """Healthy R2/R3 runs were recorded ~6 weeks after the faulty runs. Check whether detection
    relies on that: (a) healthy-recording recall split by repetition; (b) re-run detection with
    only the same-session healthy runs (R1)."""
    out = []
    groups = df["experiment_id"].to_numpy()
    X = dpb_X(df)
    y = df["label_health"].to_numpy()
    model = classifiers()["RandomForest"]
    hits = {}
    for seed in range(N_REPEATS_DPB):
        pred = cv_predict(model, X, y, dpb_folds(df, SEED + seed))
        rv = majority_vote(groups, pred)
        meta = df.groupby("experiment_id")[["meta_F", "meta_R", "meta_date"]].first()
        for rid, p in rv.items():
            if meta.loc[rid, "meta_F"] == 0:
                hits.setdefault(int(meta.loc[rid, "meta_R"]), []).append(int(p == 0))
    for r, v in sorted(hits.items()):
        out.append({"check": f"healthy recordings R{r} correctly called healthy (RF, 5 repeats)",
                    "value": f"{np.mean(v):.3f} ({sum(v)}/{len(v)})"})
    keep = ~((df["meta_F"] == 0) & (df["meta_R"] > 1))
    d2 = df[keep].reset_index(drop=True)
    R, agg, _, _ = dpb_classification(d2, "R1-only", models=["RandomForest"],
                                      tasks=["detection (healthy/faulty)"],
                                      with_threshold=False)
    a = agg.loc[("detection (healthy/faulty)", "RandomForest")]
    out.append({"check": "detection RF using only same-session healthy runs (R1, 6 recs): rec balanced acc",
                "value": f"{a['rec_balanced_acc_mean']:.3f} +- {a['rec_balanced_acc_std']:.3f}"})
    out.append({"check": "  ... window balanced acc / macro-F1",
                "value": f"{a['win_balanced_acc_mean']:.3f} / {a['win_macro_F1_mean']:.3f}"})
    return pd.DataFrame(out)


def dpb_severity_regression(df, tag="raw"):
    """Severity from spectral-energy change: Delta band power vs the healthy baseline at the
    same speed (baseline estimated on training folds only), then Ridge / RF regression of SV."""
    bp = [c for c in df.columns if "_bp_" in c or c.endswith(("_rms", "_H1", "_H2", "_H3"))]
    sv = df["label_severity"].to_numpy().astype(float)
    sp = df["meta_SP"].to_numpy()
    groups = df["experiment_id"].to_numpy()
    rows = []
    preds = {}
    for seed in range(N_REPEATS_DPB):
        folds = dpb_folds(df, SEED + seed)
        for name, model in {"Ridge(dE)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
                            "RF-reg(dE)": RandomForestRegressor(300, min_samples_leaf=5, n_jobs=-1,
                                                                random_state=SEED)}.items():
            p = np.full(len(df), np.nan)
            for tr, te in folds:
                D = df[bp].to_numpy().copy()
                for s in np.unique(sp):
                    h = tr[(sp[tr] == s) & (sv[tr] == 0)]
                    m = sp == s
                    D[m] -= D[h].mean(0)          # log-power difference = dB/10 change
                model.fit(D[tr], sv[tr])
                p[te] = model.predict(D[te])
            rp = pd.Series(p, index=groups).groupby(level=0).mean()
            rt = pd.Series(sv, index=groups).groupby(level=0).first().loc[rp.index]
            rows.append({"model": name, "seed": seed,
                         "rec_MAE": float(np.abs(rp - rt).mean()),
                         "rec_spearman": float(spearmanr(rp, rt).statistic),
                         "win_MAE": float(np.nanmean(np.abs(p - sv)))})
            if seed == 0:
                preds[name] = pd.DataFrame({"true": rt, "pred": rp})
    R = pd.DataFrame(rows)
    agg = R.groupby("model")[["rec_MAE", "rec_spearman", "win_MAE"]].agg(["mean", "std"]).round(3)
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg, preds


def dpb_energy_table(df):
    """Descriptive: recording-level total spectral energy change (dB) vs healthy mean at the same
    SP and config group, per severity. Uses all healthy runs as baseline (no model fitted)."""
    ch = [c[:-4] for c in df.columns if c.endswith("_rms")][0]
    bp = [c for c in df.columns if c.startswith(ch + "_bp_")]
    rec = df.groupby("experiment_id").agg({**{c: "mean" for c in bp + [f"{ch}_rms", f"{ch}_H1"]},
                                            "meta_SP": "first", "meta_F": "first",
                                            "label_severity": "first"})
    tot = 10 * np.log10((10 ** rec[bp]).sum(axis=1))
    rec["E_total_dB"] = tot
    rec["H1_dB"] = 10 * rec[f"{ch}_H1"]
    for col in ["E_total_dB", "H1_dB"]:
        base = rec[rec.label_severity == 0].groupby("meta_SP")[col].mean()
        rec["d" + col] = rec[col] - rec["meta_SP"].map(base)
    rho = {c: spearmanr(rec["d" + c], rec["label_severity"]).statistic for c in ["E_total_dB", "H1_dB"]}
    t = rec.groupby(["meta_SP", "label_severity"])[["dE_total_dB", "dH1_dB"]].agg(["mean", "std", "count"]).round(2)
    return t, rec, rho


def frot_confound(df):
    """How much of the DronePropB labels can rotor frequency ALONE predict? (should be ~chance)"""
    from sklearn.ensemble import RandomForestClassifier
    X = df[["f_rot"]].to_numpy()
    rows = []
    for task in ["detection (healthy/faulty)", "fault type (F0-F3)", "severity (SV0-SV3)"]:
        y = DPB_TASKS[task](df)
        r = [summarize(y, cv_predict(RandomForestClassifier(200, min_samples_leaf=20, n_jobs=-1,
                                                            random_state=SEED), X, y,
                                     dpb_folds(df, SEED + s)))["balanced_acc"] for s in range(3)]
        rows.append({"task": task, "f_rot-only balanced acc": round(float(np.mean(r)), 3),
                     "chance": round(1 / len(np.unique(y)), 3)})
    return pd.DataFrame(rows)


def leakage_demo(padre_df, dpb_df):
    """INVALID protocol on purpose: random window-level 5-fold split (overlapping windows of the
    same recording land in train and test). Shows how much leakage inflates scores."""
    from sklearn.model_selection import KFold
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced_subsample",
                                n_jobs=-1, random_state=SEED)
    rows = []
    kf = list(KFold(5, shuffle=True, random_state=SEED).split(dpb_df))
    for task in ["detection (healthy/faulty)", "severity (SV0-SV3)"]:
        y = DPB_TASKS[task](dpb_df)
        res = summarize(y, cv_predict(rf, dpb_X(dpb_df), y, kf))
        rows.append({"dataset": "DronePropB piezo", "task": task, "split": "random windows (LEAKY)",
                     **_fmt(res)})
    X = np.nan_to_num(padre_df[feature_cols(padre_df)].to_numpy())
    y = padre_df["label_motor"].astype("category").cat.codes.to_numpy()
    kf = list(KFold(5, shuffle=True, random_state=SEED).split(padre_df))
    res = summarize(y, cv_predict(rf, X, y, kf))
    rows.append({"dataset": "PADRE", "task": "28-way rotor-configuration ID", "split": "random windows (LEAKY)", **_fmt(res)})
    return pd.DataFrame(rows)


def padre_energy_by_fault_type(df):
    """Descriptive 'severity' proxy (PADRE has NO severity labels): for the 16 single-fault
    flights, change of the faulty rotor's own IMU vibration vs the healthy flight 0000 on the
    same IMU: 1x-harmonic level and total band energy (acc z), in dB."""
    names = {1: "chipped edge", 2: "bent tip", 3: "cut-off tip", 4: "bent tip + cut-off tip"}
    flight = df.groupby("experiment_id").mean(numeric_only=True)
    base = flight.loc["padre_0000"]
    rows = []
    for eid, r in flight.iterrows():
        code = eid.split("_")[1]
        d = [int(c) for c in code]
        if sum(x > 0 for x in d) != 1:
            continue
        j = [k for k, x in enumerate(d) if x > 0][0]
        rot = ROTORS[j]
        bp = [c for c in df.columns if c.startswith(f"{rot}_acc_z_bp_")]
        e = lambda s: 10 * np.log10((10 ** s[bp]).sum())
        rows.append({"flight": code, "faulty_rotor": rot, "fault_type": names[d[j]],
                     "f_rot_Hz": round(r["f_rot"], 1),
                     "dH1_dB (own IMU acc z)": round(10 * (r[f"{rot}_acc_z_H1"] - base[f"{rot}_acc_z_H1"]), 1),
                     "dE_total_dB (own IMU acc z)": round(e(r) - e(base), 1),
                     "dRMS_dB (own IMU acc z)": round(20 * (r[f"{rot}_acc_z_rms"] - base[f"{rot}_acc_z_rms"]), 1),
                     # dead/stuck channel check: window RMS below 0.05 g is not a flying accelerometer
                     "sensor_ok": bool(r[f"{rot}_acc_z_rms"] > np.log10(0.05))})
    t = pd.DataFrame(rows)
    by = t[t.sensor_ok].groupby("fault_type")[["dH1_dB (own IMU acc z)", "dE_total_dB (own IMU acc z)",
                                  "dRMS_dB (own IMU acc z)"]].agg(["mean", "min", "max", "count"]).round(1)
    by.columns = [f"{a.split(' (')[0]} {b}" for a, b in by.columns]
    return t, by


def padre_leakage_demo(df):
    """INVALID protocol on purpose: random window-level split for 28-way configuration ID."""
    from sklearn.model_selection import KFold
    X = np.nan_to_num(df[feature_cols(df)].to_numpy())
    y = df["label_motor"].astype("category").cat.codes.to_numpy()
    kf = list(KFold(5, shuffle=True, random_state=SEED).split(df))
    res = summarize(y, cv_predict(classifiers()["RandomForest"], X, y, kf, train_stride=2))
    return pd.DataFrame([{"task": "28-way rotor-configuration ID (which flight config)",
                          "split": "random windows (LEAKY, invalid)", **_fmt(res)}])
