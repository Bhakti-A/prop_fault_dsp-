"""Static PNG figures for the report. Categorical palette in fixed order (blue, orange, aqua,
yellow); a diverging blue-white-orange ramp for the peer-comparison heatmap."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.signal import spectrogram, welch

from monitor import ALERT_DB, MOTORS, SEVERE_DB
from order_tracking import tracked_orders

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MOTOR_COL = {"M1": "#e34948", "M2": "#2a78d6", "M3": "#eda100", "M4": "#1baf7a"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5",
                                                     "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#86b6ef", "#fcfcfb", "#f5b48f", "#eb6834",
                                                "#a3360f"])
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


def spectrogram_with_erpm(F, name="spectrogram_erpm_tracks.png", fmax=800):
    """Raw roll-rate spectrogram (top) and the same with each motor's 1x/2x from eRPM (bottom)."""
    f, t, S = spectrogram(F.gyro_unfilt[:, 0] - F.gyro_unfilt[:, 0].mean(), F.fs,
                          nperseg=2048, noverlap=1536)
    m = f <= fmax
    SdB = 10 * np.log10(S[m] + 1e-12)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    step = max(1, len(F.t) // 1500)
    for ax in axes:
        ax.pcolormesh(t, f[m], SdB, cmap=SEQ, shading="auto", vmin=np.percentile(SdB, 50))
        ax.grid(False); ax.set_ylim(0, fmax); ax.set_ylabel("Frequency [Hz]")
    for k, mo in enumerate(MOTORS):
        for h, ls in ((1, "-"), (2, "--")):
            axes[1].plot(F.t[::step], h * F.f_motor[::step, k], ls, color=MOTOR_COL[mo], lw=0.7,
                         alpha=0.9, label=mo if h == 1 else None)
    axes[0].set_title(f"{F.name}: gyroUnfilt roll spectrogram (as logged)", fontsize=9)
    axes[1].set_title("Same, with each motor's rotation frequency from eRPM: 1x (solid), 2x (dashed)",
                      fontsize=9)
    axes[1].set_xlabel("Time [s]")
    axes[1].legend(loc="upper right", fontsize=7, ncol=4, frameon=True, facecolor=SURF)
    return _save(fig, name)


def unfilt_vs_filtered_psd(F, name="betaflight_unfilt_vs_filtered_psd.png"):
    air = F.f_motor.min(axis=1) > 200
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for j, (x, lab) in enumerate(((F.gyro_unfilt[air, 0], "gyroUnfilt roll (before Betaflight filters)"),
                                  (F.gyro_filt[air, 0], "gyroADC roll (after Betaflight filters)"))):
        f, p = welch(x - x.mean(), F.fs, nperseg=4096)
        mm = f <= 1200
        ax.plot(f[mm], 10 * np.log10(p[mm] + 1e-12), color=SERIES[j], label=lab, lw=1.2)
    fm = F.f_motor[air]
    lo, hi = np.percentile(fm, [5, 95])
    for h, a in ((1, 0.18), (2, 0.12), (3, 0.08)):
        ax.axvspan(h * lo, h * hi, color=SERIES[2], alpha=a, lw=0,
                   label="motor 1x / 2x / 3x range (eRPM, p5-p95)" if h == 1 else None)
    ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("PSD [dB, gyro log units²/Hz]")
    ax.set_title(f"{F.name}: Betaflight's filters remove the motor lines the monitor needs", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    return _save(fig, name)


def order_spectrum_per_motor(P, name="order_spectrum_per_motor.png"):
    """Tracked level around each motor's own rotation frequency (order 1.00 = once per revolution)."""
    orders = np.round(np.arange(0.97, 1.0301, 0.001), 3)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for k, mo in enumerate(MOTORS):
        A = tracked_orders(P.flight.gyro_unfilt, P.theta[:, k:k + 1], P.starts, P.win, orders)[:, 0, :]
        ax.plot(orders, 10 * np.log10(np.median(A ** 2, axis=0)), color=MOTOR_COL[mo], label=mo,
                lw=2.0 if mo in ("M1", "M3") else 1.2)
        ax.axvline(P.line_order[k], color=MOTOR_COL[mo], ls=":", lw=0.8)
    ax.axvline(1.0, color=INK2, ls="-", lw=0.6)
    ax.set_xlabel("Order relative to each motor's eRPM frequency (dotted: where its line was found)")
    ax.set_ylabel("Tracked vibration level [dB]")
    ax.set_title(f"{P.flight.name}: each motor's once-per-revolution line; M3 and M1 stand above M2 and M4",
                 fontsize=9)
    ax.legend(fontsize=8)
    return _save(fig, name)


def levels_by_flight(preps, name="1x_levels_by_flight.png"):
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for k, mo in enumerate(MOTORS):
        med = [np.median(20 * np.log10(p.a1()[:, k])) for p in preps]
        ax.plot(range(len(preps)), med, "o-", color=MOTOR_COL[mo], label=mo, lw=1.4, ms=4)
    ax.set_xticks(range(len(preps)))
    ax.set_xticklabels([p.flight.name for p in preps], rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("Flight-median 1x level [dB]")
    ax.set_title("Once-per-revolution vibration of each motor, 16 flights of 20-22 Sept", fontsize=9)
    ax.legend(fontsize=8, ncol=4, loc="lower left")
    return _save(fig, name)


def peer_heatmap(diag, name="diagnosis_peer_heatmap.png"):
    cols = [f"peer_dB_{m}" for m in MOTORS]
    M = diag[cols].to_numpy(float)
    fig, ax = plt.subplots(figsize=(5.6, 0.32 * len(diag) + 1.4))
    lim = max(abs(M.min()), M.max())
    ax.imshow(M, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=7,
                    color=SURF if abs(M[i, j]) > 0.6 * lim else INK,
                    fontweight="bold" if M[i, j] >= ALERT_DB else "normal")
    ax.set_xticks(range(4)); ax.set_xticklabels(MOTORS)
    ax.set_yticks(range(len(diag))); ax.set_yticklabels(diag.flight, fontsize=7)
    ax.set_title(f"Each motor vs the other three [dB]\n(fault at >= {ALERT_DB:g} dB, severe at >= "
                 f"{SEVERE_DB:g} dB)", fontsize=9)
    return _save(fig, name)


def trend_plot(tr, name="peer_trend_by_date.png"):
    dates = list(dict.fromkeys(tr.date))
    x = {d: i for i, d in enumerate(dates)}
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for k, mo in enumerate(MOTORS):
        col = f"peer_dB_{mo}"
        off = (k - 1.5) * 0.12
        xs = np.array([x[d] for d in tr.date]) + off
        ax.scatter(xs, tr[col], s=14, color=MOTOR_COL[mo], alpha=0.55, lw=0)
        med = tr.groupby("date", sort=False)[col].median()
        ax.plot(np.arange(len(dates)) + off, med.to_numpy(), "-", color=MOTOR_COL[mo], marker="_", ms=14,
                lw=2.2 if mo in ("M1", "M3") else 1.0, label=mo)
    ax.axhspan(ALERT_DB, SEVERE_DB, color=SERIES[1], alpha=0.07, lw=0)
    ax.axhspan(SEVERE_DB, max(SEVERE_DB + 6, tr.filter(like="peer_dB").max().max() + 1), color=SERIES[7],
               alpha=0.07, lw=0)
    ax.axhline(ALERT_DB, color=INK2, ls="--", lw=0.8)
    ax.axhline(SEVERE_DB, color=INK2, ls=":", lw=0.8)
    ax.text(len(dates) - 0.55, ALERT_DB + 0.3, "fault", fontsize=8, color=INK2, ha="right")
    ax.text(len(dates) - 0.55, SEVERE_DB + 0.3, "severe", fontsize=8, color=INK2, ha="right")
    ax.set_xticks(range(len(dates)))
    n = tr.groupby("date", sort=False).size()
    ax.set_xticklabels([f"{d}\n({n[d]} logs)" for d in dates])
    ax.set_ylabel("1x vibration vs other three motors [dB]")
    ax.set_title("1x vibration vs the other three motors over three weeks (dots = logs, bars = median)",
                 fontsize=9)
    ax.legend(fontsize=8, ncol=4, loc="upper left")
    return _save(fig, name)


def load_trend(tr, name="motor_load_trend.png"):
    dates = list(dict.fromkeys(tr.date))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for k, mo in enumerate(MOTORS):
        col = f"drive_per_speed_{mo}"
        med = tr.groupby("date", sort=False)[col].median().to_numpy()
        ax.plot(range(len(dates)), (med - 1) * 100, "o-", color=MOTOR_COL[mo], label=mo,
                lw=2.2 if mo == "M3" else 1.2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(len(dates))); ax.set_xticklabels(dates)
    ax.set_ylabel("Drive command per unit speed\nvs median motor [%]")
    ax.set_title("Drive needed per unit speed: M3 rises over three weeks", fontsize=9)
    ax.legend(fontsize=8, ncol=4, loc="upper left")
    return _save(fig, name)
