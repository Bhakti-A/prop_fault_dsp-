"""Figures (static PNG for the report). Palette: validated default categorical order
(blue, orange, aqua, yellow, magenta) used in fixed order; sequential blue ramp for matrices."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import spectrogram, welch

from features import make_windows, notch_windows, estimate_frot_robust

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5",
                                                     "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "font.size": 9, "axes.titlesize": 10, "legend.frameon": False, "figure.dpi": 130,
})


def _save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def _psd(x, fs, nperseg):
    f, p = welch(x - x.mean(), fs, nperseg=nperseg)
    return f, 10 * np.log10(p + 1e-20)


def padre_psd(recs, codes=("0000", "1000", "2000", "3000", "4000")):
    by = {r.meta["code"]: r for r in recs}
    names = {"0000": "0000 healthy", "1000": "1000 A chipped edge", "2000": "2000 A bent tip",
             "3000": "3000 A cut-off tip", "4000": "4000 A bent+cut"}
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for j, rot in enumerate("ABCD"):
        ax = axes[j]
        ch = by["0000"].ch_names.index(f"{rot}_acc_z")
        for k, c in enumerate(codes):
            f, p = _psd(by[c].signals[ch], 500, 512)
            ax.plot(f, p, color=SERIES[k], label=names[c], lw=1.2)
        ax.set_title(f"IMU {rot}, acc z")
        ax.set_xlabel("Frequency [Hz]")
    axes[0].set_ylabel("PSD [dB re g²/Hz]")
    axes[-1].legend(loc="upper right", fontsize=7)
    fig.suptitle("PADRE (Bebop 2): Welch PSD, healthy vs single-rotor faults on rotor A", y=1.03)
    return _save(fig, "padre_psd_by_imu.png")


def spectrograms(padre_recs, dpb_recs):
    by = {r.meta["code"]: r for r in padre_recs}
    dp = {r.experiment_id: r for r in dpb_recs}
    panels = [(by["0000"], "A_acc_z", "PADRE 0000 healthy, IMU A acc z", 256, 250),
              (by["3000"], "A_acc_z", "PADRE 3000 (A cut-off tip), IMU A acc z", 256, 250),
              (dp["F0_SV0_SP2_C1_R1"], "piezo", "DronePropB F0 SP2 C1 R1 healthy, piezo", 4096, 600),
              (dp["F1_SV3_SP2_C1"], "piezo", "DronePropB F1 SV3 SP2 C1, piezo", 4096, 600)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    for ax, (r, ch, title, nps, fmax) in zip(axes.ravel(), panels):
        x = r.signals[r.ch_names.index(ch)]
        f, t, S = spectrogram(x - x.mean(), r.fs, nperseg=nps, noverlap=nps // 2)
        m = f <= fmax
        im = ax.pcolormesh(t, f[m], 10 * np.log10(S[m] + 1e-20), shading="auto", cmap=SEQ)
        ax.set_title(title); ax.set_ylabel("Hz"); ax.set_xlabel("time [s]"); ax.grid(False)
        fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    return _save(fig, "spectrograms.png")


def dpb_psd_severity(dpb_recs, F=1, SP=2, C=1):
    dp = {r.experiment_id: r for r in dpb_recs}
    ids = [f"F0_SV0_SP{SP}_C{C}_R1"] + [f"F{F}_SV{s}_SP{SP}_C{C}" for s in (1, 2, 3)]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for k, i in enumerate(ids):
        f, p = _psd(dp[i].signals[0], dp[i].fs, 16384)
        m = f <= 700
        ax.plot(f[m], p[m], color=SERIES[k], lw=1.1, label=i)
    ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("PSD [dB]")
    ax.set_title(f"DronePropB piezo PSD, F{F} severity SV0..SV3 at SP{SP}, C{C}")
    ax.legend(fontsize=7)
    return _save(fig, "dpb_psd_severity.png")


def notch_effect(padre_recs, dpb_recs):
    by = {r.meta["code"]: r for r in padre_recs}
    dp = {r.experiment_id: r for r in dpb_recs}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, (r, ch, band, nps, fmax, title) in zip(axes, [
            (by["3000"], "A_acc_z", (60, 245), 256, 250, "PADRE 3000, IMU A acc z"),
            (dp["F1_SV3_SP2_C1"], "piezo", (80, 115), 4096, 600, "DronePropB F1 SV3 SP2 C1, piezo")]):
        W, _ = make_windows(r.signals, r.fs, 1.0, 0.5)
        f, P = welch(W, r.fs, nperseg=nps, axis=-1)
        fr = estimate_frot_robust(f, P, band)
        Wn = notch_windows(W, r.fs, fr)
        _, Pn = welch(Wn, r.fs, nperseg=nps, axis=-1)
        j = r.ch_names.index(ch)
        m = f <= fmax
        ax.plot(f[m], 10 * np.log10(P[:, j].mean(0)[m]), color=SERIES[0], label="raw")
        ax.plot(f[m], 10 * np.log10(Pn[:, j].mean(0)[m]), color=SERIES[1],
                label="notched at 1x, 2x, 3x f_rot (Q=5)")
        for k in (1, 2, 3):
            fk = k * np.median(fr)
            fk = fk if fk <= r.fs / 2 else r.fs - fk
            if fk <= fmax:
                ax.axvline(fk, color=INK2, lw=0.6, ls=":")
        ax.set_title(title); ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("mean PSD [dB]")
        ax.legend(fontsize=7)
    fig.suptitle("Software RPM/notch filtering removes the rotor-harmonic lines that carry fault signatures", y=1.03)
    return _save(fig, "notch_effect.png")


def confusion(cm, labels, title, name, xlabel="predicted", ylabel="true"):
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * len(labels), 0.8 + 0.7 * len(labels)))
    ax.imshow(cmn, cmap=SEQ, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=7,
                    color="#fcfcfb" if cmn[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=9)
    return _save(fig, name)


def energy_vs_severity(rec):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), sharey=False)
    for ax, col, lab in zip(axes, ["dE_total_dB", "dH1_dB"],
                            ["total band energy change [dB]", "1x-harmonic level change [dB]"]):
        for k, sp in enumerate(sorted(rec.meta_SP.unique())):
            d = rec[rec.meta_SP == sp]
            jitter = (k - 1) * 0.18
            ax.scatter(d.label_severity + jitter, d[col], s=22, color=SERIES[k], label=f"SP{sp}",
                       edgecolor=SURF, linewidth=0.8)
            means = d.groupby("label_severity")[col].mean()
            ax.plot(means.index + jitter, means.values, color=SERIES[k], lw=1.0)
        ax.axhline(0, color=INK2, lw=0.7)
        ax.set_xticks([0, 1, 2, 3], ["SV0", "SV1", "SV2", "SV3"])
        ax.set_ylabel(lab); ax.set_xlabel("severity level")
    axes[0].legend(fontsize=7)
    fig.suptitle("DronePropB piezo: spectral energy change vs healthy mean at the same speed (one dot = one recording)", y=1.03)
    return _save(fig, "dpb_energy_vs_severity.png")


def raw_vs_notch_bars(table, name, title):
    """table: DataFrame with index=task/model rows and columns ['raw','notched']."""
    fig, ax = plt.subplots(figsize=(8, 0.5 + 0.45 * len(table)))
    y = np.arange(len(table))
    ax.barh(y + 0.2, table["raw"], height=0.38, color=SERIES[0], label="raw")
    ax.barh(y - 0.2, table["notched"], height=0.38, color=SERIES[1], label="notch-filtered")
    for yy, (a, b) in zip(y, table[["raw", "notched"]].to_numpy()):
        ax.text(a + 0.01, yy + 0.2, f"{a:.2f}", va="center", fontsize=7, color=INK2)
        ax.text(b + 0.01, yy - 0.2, f"{b:.2f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y, table.index); ax.invert_yaxis()
    ax.set_xlim(0, 1.1); ax.set_xlabel("balanced accuracy"); ax.set_title(title, pad=22)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7)
    return _save(fig, name)


# ----------------------------------------------------------------- PADRE-only figures
def padre_spectrograms(recs):
    by = {r.meta["code"]: r for r in recs}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.3), sharey=True)
    for ax, code, title in [(axes[0], "0000", "0000 healthy"), (axes[1], "3000", "3000 (rotor A cut-off tip)")]:
        r = by[code]
        x = r.signals[r.ch_names.index("A_acc_z")]
        f, t, S = spectrogram(x - x.mean(), r.fs, nperseg=256, noverlap=128)
        im = ax.pcolormesh(t, f, 10 * np.log10(S + 1e-20), shading="auto", cmap=SEQ, vmin=-60, vmax=20)
        ax.set_title(f"PADRE {title}, IMU A acc z"); ax.set_xlabel("time [s]"); ax.grid(False)
    axes[0].set_ylabel("Frequency [Hz]")
    fig.colorbar(im, ax=axes, label="dB")
    return _save(fig, "padre_spectrograms.png")


def padre_notch_effect(recs, code="3000"):
    r = {x.meta["code"]: x for x in recs}[code]
    W, _ = make_windows(r.signals, r.fs, 1.0, 0.5)
    f, P = welch(W, r.fs, nperseg=256, axis=-1)
    fr = estimate_frot_robust(f, P, (60, 245))
    _, Pn = welch(notch_windows(W, r.fs, fr), r.fs, nperseg=256, axis=-1)
    j = r.ch_names.index("A_acc_z")
    fig, ax = plt.subplots(figsize=(7, 3.3))
    ax.plot(f, 10 * np.log10(P[:, j].mean(0)), color=SERIES[0], label="raw")
    ax.plot(f, 10 * np.log10(Pn[:, j].mean(0)), color=SERIES[1], label="notched at 1x, 2x, 3x f_rot (Q=5)")
    for k in (1, 2, 3):
        fk = k * np.median(fr)
        fk = fk if fk <= r.fs / 2 else r.fs - fk          # aliased harmonic
        ax.axvline(fk, color=INK2, lw=0.6, ls=":")
    ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("mean PSD [dB re g²/Hz]")
    ax.set_title(f"PADRE {code}, IMU A acc z: software RPM/notch filter removes the fault line", fontsize=9)
    ax.legend(fontsize=7)
    return _save(fig, "padre_notch_effect.png")


def padre_energy_by_type(t):
    order = ["chipped edge", "bent tip", "cut-off tip", "bent tip + cut-off tip"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    for ax, col, lab in zip(axes, ["dH1_dB (own IMU acc z)", "dE_total_dB (own IMU acc z)"],
                            ["1x-harmonic level change [dB]", "total band energy change [dB]"]):
        t = t[t.sensor_ok]
        for k, rot in enumerate("ABCD"):
            d = t[t.faulty_rotor == rot]
            xs = [order.index(v) + (k - 1.5) * 0.12 for v in d.fault_type]
            ax.scatter(xs, d[col], s=30, color=SERIES[k], label=f"fault on rotor {rot}",
                       edgecolor=SURF, linewidth=0.8, zorder=3)
        m = t.groupby("fault_type")[col].mean().reindex(order)
        ax.plot(range(4), m.values, color=INK2, lw=1.0, label="mean")
        ax.axhline(0, color=INK2, lw=0.7)
        ax.set_xticks(range(4), order, rotation=15)
        ax.set_ylabel(lab)
    axes[0].legend(fontsize=7)
    fig.suptitle("PADRE single-fault flights: vibration change on the faulty rotor's IMU vs healthy flight 0000\n(flight 0040 excluded: IMU C acc z dead/stuck)", y=1.06)
    return _save(fig, "padre_energy_by_fault_type.png")
