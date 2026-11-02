"""Stage 2 add-on (NON-DRONE data): envelope-spectrum bearing-fault features on 3 CWRU drive-end
files (12 kHz, 0 hp, ~1797 rpm, 0.007" faults): 105 inner race, 118 ball, 130 outer race @6:00.
Shows the envelope-spectrum feature that would be added for motor-bearing faults; NOT a drone result.
Usage: python -I scripts/bearing_envelope_demo.py <cwru_folder> <out_png> <out_csv>"""
import glob, os, sys
import numpy as np, pandas as pd, scipy.io as sio
from scipy.signal import butter, sosfiltfilt, hilbert
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import figures as FG
import matplotlib.pyplot as plt

folder, out_png, out_csv = sys.argv[1:4]
fs = 12000.0
# SKF 6205-2RS JEM defect-frequency multipliers of shaft speed (CWRU documentation)
MULT = {"BPFI": 5.4152, "BPFO": 3.5848, "BSF": 2.3570 * 2}   # BSF line: 2x ball spin freq commonly dominant
files = {"105": ("inner race", "BPFI"), "118": ("ball", "BSF"), "130": ("outer race", "BPFO")}
rows = []
fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
for ax, (num, (name, key)) in zip(axes, files.items()):
    m = sio.loadmat(os.path.join(folder, f"{num}.mat"))
    de = [k for k in m if k.endswith("DE_time")][0]
    rpm = [k for k in m if k.endswith("RPM")]
    rpm = float(np.ravel(m[rpm[0]])[0]) if rpm else 1797.0
    x = np.ravel(m[de]).astype(float)
    fr = rpm / 60
    sos = butter(4, [2500, 5500], btype="bandpass", fs=fs, output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, x)))
    env -= env.mean()
    E = np.abs(np.fft.rfft(env * np.hanning(len(env)))) / len(env)
    f = np.fft.rfftfreq(len(env), 1 / fs)
    sel = (f > 5) & (f < 400)
    fpk = f[sel][np.argmax(E[sel])]
    r = {"file": num, "fault": name, "rpm": rpm, "fs_assumed": fs, "dominant_env_peak_Hz": round(fpk, 1)}
    for k, mlt in MULT.items():
        r[f"{k}_theory_Hz"] = round(mlt * fr, 1)
        s = np.abs(f - mlt * fr) < 3
        r[f"{k}_env_amp_rel_median"] = round(float(E[s].max() / np.median(E[sel])), 1)
    rows.append(r)
    ax.plot(f[sel], E[sel], color=FG.SERIES[0], lw=1.0)
    for j, (k, mlt) in enumerate(MULT.items()):
        ax.axvline(mlt * fr, color=FG.SERIES[j + 1], lw=1.0, ls="--", label=f"{k} {mlt*fr:.0f} Hz")
    ax.set_title(f"CWRU {num}: {name} fault (NON-DRONE)", fontsize=9); ax.set_xlabel("envelope frequency [Hz]")
    ax.legend(fontsize=6)
axes[0].set_ylabel("envelope spectrum [a.u.]")
fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)
t = pd.DataFrame(rows); t.to_csv(out_csv, index=False); print(t.to_string())
