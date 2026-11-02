"""Stage 3 (descriptive only, NOT used for classification): user's own Betaflight blackbox CSV.
Compare gyroUnfilt (pre-filter) vs gyroADC (after Betaflight RPM + dynamic notch + LPF) and mark
motor rotation frequencies from bidirectional-DShot eRPM.
Usage: python -I scripts/betaflight_filter_demo.py <blackbox.csv> <out.png> [motor_poles=12]"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import figures as FG
import matplotlib.pyplot as plt
from scipy.signal import welch

path, out = sys.argv[1], sys.argv[2]
poles = int(sys.argv[3]) if len(sys.argv) > 3 else 12
with open(path, errors="replace") as fh:
    for i, line in enumerate(fh):
        if line.startswith('"loopIteration"'):
            break
df = pd.read_csv(path, skiprows=i)
df.columns = [c.strip() for c in df.columns]
t = df["time"].to_numpy() / 1e6
fs = 1 / np.median(np.diff(t))
thr = df[[f"motor[{k}]" for k in range(4)]].mean(axis=1)
air = thr > thr.quantile(0.5)            # use the upper half of throttle samples (in flight)
x_u = df.loc[air, "gyroUnfilt[0]"].to_numpy(float)
x_f = df.loc[air, "gyroADC[0]"].to_numpy(float)
# eRPM field is eRPM/100 in Betaflight logs -> mechanical Hz = eRPM*100 / (poles/2) / 60
erpm = df.loc[air, [f"eRPM[{k}]" for k in range(4)]].to_numpy(float)
f_mech = erpm * 100 / (poles / 2) / 60
f, Pu = welch(x_u - x_u.mean(), fs, nperseg=4096)
_, Pf = welch(x_f - x_f.mean(), fs, nperseg=4096)
fig, ax = plt.subplots(figsize=(8, 3.4))
m = f <= min(fs / 2, 1000)
ax.plot(f[m], 10 * np.log10(Pu[m] + 1e-12), color=FG.SERIES[0], label="gyroUnfilt[0] (roll, raw)")
ax.plot(f[m], 10 * np.log10(Pf[m] + 1e-12), color=FG.SERIES[1], label="gyroADC[0] (after BF filters)")
lo, hi = np.percentile(f_mech, [10, 90])
ax.axvspan(lo, hi, color=FG.SERIES[2], alpha=0.15, label="motor 1x range from eRPM (p10-p90)")
ax.axvspan(2 * lo, 2 * hi, color=FG.SERIES[3], alpha=0.12, label="2x")
ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("PSD [dB]")
ax.set_title(f"Own AIR75 blackbox ({os.path.basename(path)}, {fs:.0f} Hz log): Betaflight filtering removes motor lines", fontsize=9)
ax.legend(fontsize=7)
fig.savefig(out, bbox_inches="tight"); plt.close(fig)
s = (f >= lo) & (f <= hi)
print(f"fs={fs:.0f}Hz in-air samples={air.sum()} motor 1x p10-p90 = {lo:.0f}-{hi:.0f} Hz; "
      f"band power reduction (unfilt->filtered) in 1x band = {10*np.log10(Pu[s].sum()/Pf[s].sum()):.1f} dB")
