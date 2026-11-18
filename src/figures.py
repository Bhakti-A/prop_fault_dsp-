"""Static PNG figures for the report. Categorical palette in fixed order (blue, orange, aqua,
yellow, magenta); sequential blue ramp for matrices."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import spectrogram, welch

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
MOTORS = ["M1", "M2", "M3", "M4"]


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
    for k in range(4):
        for h, ls in ((1, "-"), (2, "--")):
            axes[1].plot(F.t[::step], h * F.f_motor[::step, k], ls, color=SERIES[k + 1], lw=0.7,
                         alpha=0.85, label=f"{MOTORS[k]}" if h == 1 else None)
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


def levels_by_flight(new, old, name="1x_levels_by_flight.png"):
    """One dot per 1 s window, one column per flight; new-prop flights first, then old-prop."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    for k, ax in enumerate(axes):
        i = 0
        for gi, (lab, ps) in enumerate((("new props", new), ("old props", old))):
            for p in ps:
                L = 20 * np.log10(p.orders_unfilt[:, k, 0])
                ax.scatter(np.full(len(L), i), L, s=4, alpha=0.3, color=SERIES[gi], lw=0,
                           label=lab if p is ps[0] else None)
                ax.scatter([i], [np.median(L)], s=30, marker="_", color=INK, lw=1.5)
                i += 1
        ax.set_title(f"{MOTORS[k]}"); ax.set_xlabel("Flight")
        ax.set_xticks(range(i)); ax.set_xticklabels([])
    axes[0].set_ylabel("1x vibration level [dB]")
    axes[0].legend(fontsize=7, markerscale=3, loc="lower left")
    fig.suptitle("Each motor's once-per-revolution vibration, per 1 s window (black tick = flight median)",
                 y=1.03)
    return _save(fig, name)


def raw_vs_tracked(P_old, P_new, band, name="raw_spectrum_vs_order_tracking.png"):
    """Left: raw gyro spectrum (cannot tell the flights apart). Right: order spectrum around
    motor M1's rotation speed (order 1.00 = exactly once per revolution of M1)."""
    from order_tracking import spectrum_true_time, tracked_orders
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    ax = axes[0]
    freqs = np.arange(200.0, 400.5, 1.0)
    for j, (P, lab) in enumerate(((P_new, f"new props ({P_new.flight.name})"),
                                  (P_old, f"old props ({P_old.flight.name})"))):
        F = P.flight
        air = F.f_motor.min(axis=1) > 200
        p = spectrum_true_time(F.gyro_unfilt[air], F.t[air], int(round(F.fs)), freqs)
        ax.plot(freqs, 10 * np.log10(p + 1e-12), color=SERIES[j], label=lab, lw=1.2)
    ax.axvspan(band[0], band[1], color=SERIES[2], alpha=0.13, lw=0,
               label=f"motor-speed band ({band[0]:g}-{band[1]:g} Hz)")
    ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("gyroUnfilt spectrum, 3 axes [dB]")
    ax.set_title("Raw spectrum: four motor lines smeared into one hump", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax = axes[1]
    orders = np.round(np.arange(0.90, 1.1001, 0.005), 3)
    for j, (P, lab) in enumerate(((P_new, "new props"), (P_old, "old props"))):
        A = tracked_orders(P.flight.gyro_unfilt, P.theta[:, :1], P.starts, P.win, orders)[:, 0, :]
        ax.plot(orders, 10 * np.log10(np.median(A ** 2, axis=0)), color=SERIES[j], label=lab, lw=1.4)
    ax.axvline(1.0, color=INK2, ls=":", lw=0.8)
    ax.set_xlabel("Order relative to motor M1 (1.00 = once per M1 revolution)")
    ax.set_ylabel("Tracked level [dB]")
    ax.set_title("Order tracking on M1: the same sharp line at exactly 1.00 with both prop sets", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    return _save(fig, name)


def flight_scores(diag, thr, name="diagnosis_flight_scores.png"):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    x = np.arange(len(diag))
    col = [SERIES[0] if p == "new" else SERIES[1] for p in diag.props]
    ax.bar(x, diag.flight_score, color=col, width=0.65)
    ax.axhline(thr, color=INK2, ls="--", lw=1)
    ax.text(len(diag) - 0.5, thr + 0.06, f"alarm above {thr:g}", ha="right", fontsize=8, color=INK2)
    for i, (v, a, m) in enumerate(zip(diag.flight_score, diag.alarm, diag.worst_motor)):
        ax.text(i, max(v, 0) + 0.06, f"{v:.1f}" + (f"\n{m}" if a else ""), ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(diag.flight, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("Flight score (highest motor's median z)")
    ax.set_ylim(min(0, diag.flight_score.min()) - 0.2, diag.flight_score.max() + 0.7)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SERIES[0], label="new props (each tested against the other 10)"),
                       Patch(color=SERIES[1], label="old props (tested against all 11 new-prop flights)")],
              fontsize=7, loc="upper left")
    ax.set_title("Flight score on real flights: no flight reaches the alarm level, with either prop set",
                 fontsize=9)
    return _save(fig, name)


def change_by_motor(diag, name="diagnosis_change_by_motor.png"):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    new, old = diag[diag.props == "new"], diag[diag.props == "old"]
    cols = [f"change_dB_{m}" for m in MOTORS]
    x = np.arange(4)
    lo, hi = new[cols].min().to_numpy(), new[cols].max().to_numpy()
    ax.bar(x, hi - lo, bottom=lo, width=0.8, color=GRID, label="range over the 11 new-prop flights", zorder=1)
    w = 0.8 / len(old)
    for i, (_, r) in enumerate(old.iterrows()):
        ax.bar(x - 0.4 + (i + 0.5) * w, r[cols].to_numpy(float), width=w * 0.85, color=SERIES[1],
               alpha=0.55 + 0.45 * i / max(1, len(old) - 1), zorder=2,
               label="old-prop flights (log_1 ... log_6, left to right)" if i == 0 else None)
    med = old[cols].median().to_numpy()
    for k in range(4):
        ax.text(k, max(old[cols[k]].max(), hi[k]) + 0.8, f"median {med[k]:+.0f} dB", ha="center", fontsize=8)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(MOTORS)
    ax.set_ylabel("Change in 1x vibration energy [dB]\nvs new-prop baseline")
    ax.set_ylim(min(lo.min(), old[cols].min().min()) - 2, old[cols].max().max() + 4)
    ax.set_title("Old-prop flights vs new-prop baseline: change in each motor's 1x vibration",
                 fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    return _save(fig, name)


def example_scores(t, z_clean, z_inj, thr, motor, ratio, flight, name="synthetic_example.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.3), sharey=True)
    for ax, z, title in ((axes[0], z_clean, f"{flight}: as flown"),
                         (axes[1], z_inj, f"{flight}: + SYNTHETIC imbalance on {MOTORS[motor]}, "
                                          f"r = {ratio:g}")):
        for k in range(4):
            ax.plot(t, z[:, k], color=SERIES[k], lw=1.1, label=MOTORS[k])
        ax.axhline(thr, color=INK2, ls="--", lw=1, label="alarm threshold")
        ax.set_title(title, fontsize=9); ax.set_xlabel("Window start [s]")
    axes[0].set_ylabel("Robust z-score of 1x level")
    axes[1].legend(fontsize=7, ncol=5, loc="upper left")
    return _save(fig, name)


def detection_curve(by_r, name="synthetic_detection_vs_size.png"):
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x = by_r.index.to_numpy()
    ax.plot(x, by_r["flights_alarmed"], "o-", color=SERIES[0], label="flights alarmed")
    ax.plot(x, by_r["correct_motor"], "^-", color=SERIES[2], label="correct motor named")
    ax.set_xscale("log", base=2); ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" for v in x])
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Injected imbalance ratio r (x normal 1x amplitude)"); ax.set_ylabel("Rate")
    ax.set_title("SYNTHETIC imbalance added to new-prop flights (leave-one-flight-out)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    return _save(fig, name)


def confusion(cm, title, name):
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(cm, cmap=SEQ, vmin=0)
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=SURF if cm[i, j] > cm.max() * 0.55 else INK, fontsize=9)
    ax.set_xticks(range(4)); ax.set_xticklabels(MOTORS)
    ax.set_yticks(range(4)); ax.set_yticklabels(MOTORS)
    ax.set_xlabel("Located motor"); ax.set_ylabel("Motor with injected imbalance")
    ax.set_title(title, fontsize=9)
    return _save(fig, name)


def severity_plot(inj, name="synthetic_severity_estimate.png"):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for k, m in enumerate(MOTORS):
        d = inj[inj.motor == m]
        jit = (k - 1.5) * 0.25
        ax.scatter(d.severity_nominal_dB + jit, d.severity_est_dB, s=10, alpha=0.6, color=SERIES[k],
                   label=m, lw=0)
    lim = [min(-6, inj.severity_est_dB.min() - 1), inj.severity_nominal_dB.max() + 2]
    ax.plot(lim, lim, color=INK2, lw=1, ls="--", label="estimate = injected")
    ax.set_xlabel("Injected 1x energy increase, 10·log10(1+r²) [dB]")
    ax.set_ylabel("Measured 1x energy increase [dB]")
    ax.set_title("Severity = change in the motor's 1x energy vs healthy", fontsize=9)
    ax.legend(fontsize=7)
    return _save(fig, name)
