# DSP-based propeller fault detection for quadcopters (PADRE dataset)

Classical DSP features + classical ML (SVM, Random Forest, kNN) on the public **PADRE** dataset (Parrot Bebop 2, 4 on-arm IMUs, propeller faults on 1–4 rotors) for three diagnoses:
(1) **detection** healthy vs faulty, (2) **localization** of the faulty rotor(s), (3) **severity**, as far as the data allow; PADRE has fault *types*, not severity levels (see Stage 1.3).
No new flight data were collected.

Reproduce everything: `nice -n 10 .venv/bin/python -I run_all.py` (~12 min, limited to 4 cores).

## Headline results (all numbers produced by `run_all.py`)

| Diagnosis | Protocol | Result |
|---|---|---|
| Detection, healthy vs faulty | time-block CV (within-flight, see caveat) | SVM balanced acc **0.992**, macro-F1 0.994; RMS-threshold baseline 0.973 |
| Localization, which rotor(s) are faulty | **leave-one-flight-out** (28 flights), RF per rotor | per-rotor balanced acc A **0.95**, B **0.91**, C **0.90**, D **0.73**. Flight level: **90 %** of the 112 rotor states correct, exact 4-rotor pattern in **18 / 28** flights |
| Fault type, rate the faulty rotor is flagged | same | chipped edge 0.89, bent tip 0.68, cut-off tip 0.75, bent+cut 1.00; healthy rotors wrongly flagged 0.05 |
| Severity proxy, spectral energy change | descriptive (no severity labels) | faulty rotor's own IMU vs healthy flight: cut-off tip **+17.5 dB**, chipped edge **+13.9 dB**, bent tip **≈ 0 dB** (mean 1×-harmonic level). Large spread between rotors |
| Effect of notch/RPM filtering | same protocols, notch at 1×/2×/3× f_rot | threshold detector 0.925 → **0.683**, RMS threshold 0.973 → 0.786, RF detection 0.977 → 0.907, RF localization 0.876 → 0.811, exact pattern 18 → 12 / 28 flights |
| Leakage check | random window split (**invalid**, for comparison only) | 0.992 for 28-way configuration ID, which only measures recognising the same flight |

**Main caveat:** PADRE has a single healthy flight, so detection can only be tested within flights (time-disjoint blocks). That number is optimistic. Localization is tested strictly on unseen flights.

---

## Run log

### Stage 0 — Data acquisition

| # | Dataset | Status | Folder |
|---|---|---|---|
| 1 | **PADRE**, github.com/AeroLabPUT/UAV_measurement_data | **Used.** Downloaded the raw IMU folders only (Bebop 2 `Raw_data`, 28 flights, 233 MB; 3DR Solo raw IMU; all Readmes). The full repo is 4.8 GB, mostly derived FFT/DWT/EMD products that we recompute ourselves. | `data/raw/padre/` |
| 2 | **DronePropB**, Mendeley 10.17632/xkvfjmm8zg.1 | Downloaded (111 `.mat`, all SHA-256 verified) and inspected, then **dropped from scope** by decision to keep the project to one dataset and keep the laptop load low. Exploratory runs had shown its faults barely separable under a recording-grouped split. | `data/raw/dronepropb/` |
| 3 | DronePropA (PMC12136912) | Not downloaded: 4.22 GB, and the disk has ~2 GB free. | — |
| 4 | PHM-AP 2025 UT indoor UAV dataset | Not downloaded: ~3.6 GB on Google Drive, and the disk is full. Per its paper the logs are subsampled to ≈200 Hz. | — |

#### PADRE inspection (from the files, not assumed)

| Item | Finding |
|---|---|
| Format | Text file. One line = 96 hex chars = 24 big-endian int16 registers. |
| Flights | 28 files, one flight each, 86 016 samples = 172 s at **500 Hz** (rate stated in the Readme) |
| Channels | 4 IMUs labelled A, B, C, D (layout `C A / D B` in the Readme), each with acc xyz + gyro xyz = 24 channels. The rotor letter in the channel names implies one IMU per arm/rotor. |
| Units | Raw counts. Converted with ±16 g → 2048 LSB/g and ±1000 dps → 32.8 LSB/dps. These standard InvenSense sensitivities are **assumed**; only relative spectra matter here. |
| Labels | Filename code `ABCD`, one digit per rotor: 0 none, 1 chipped edge, 2 bent tip, 3 cut-off tip, 4 bent tip + cut-off tip. Faults occur on 1, 2, 3 or 4 rotors. **Only flight `0000` is fully healthy.** Each rotor is faulty in 11 of the 28 flights. |
| Severity | **No severity levels.** Fault *types* only. |
| RPM | **None.** No motor speed or ESC telemetry. |
| Data quality | Several channels saturate in heavy-fault flights (e.g. 12–15 % of samples on IMU A/C acc z in 3000/4000). **IMU C acc z is dead/stuck in flight 0040** (std 0.00 g). IMU C is noisier than A/B even in the healthy flight (acc z std 3.5 g vs 0.9 g). Take-off at ~1 s. |
| 3DR Solo raw | Binary with an empty Readme, so the format is undocumented. **Not used.** |

### Stage 1 — Propeller fault detection (main deliverable)

#### Methodology

* **Loader → common schema** (`src/loaders.py`): `Recording(signals[24, n], ch_names, fs=500, label_health, label_motor='ABCD', label_severity=None, experiment_id='padre_ABCD', meta)`. Labels come only from the filename.
* **Pre-processing** (`src/features.py`): trim the first 10 s and last 5 s (take-off/landing), then 1.0 s windows with 50 % overlap and linear detrend. That gives 306 windows per flight, 8 568 in total. Each flight is also cut into 6 contiguous time blocks, and windows never cross a block boundary.
* **Rotor frequency f_rot:** there's no RPM, so we use the **dominant spectral peak**. Per window, the Welch PSD (256-point, 1.95 Hz bins) is normalised per channel, summed over channels, and the peak is found in 60–245 Hz. A segment-level estimate with a sub-harmonic check stops it locking onto 2×; each window is then refined within ±15 %. Median f_rot per flight is 127–163 Hz. Flights with a cut-off tip on rotor A run faster (≈160 Hz), consistent with the controller compensating for lost thrust. The 2× and 3× harmonics lie above Nyquist (250 Hz), so they're evaluated at their **aliased** frequency (fs − k·f_rot).
* **Features** per window and channel: log RMS, kurtosis, crest factor, spectral centroid, normalised spectral entropy, PSD peak at 1×/2×/3× f_rot (±2.5 bins, log), 2×/1× ratio, and 9 log band powers (1–10, 10–25, 25–50, 50–80, 80–110, 110–140, 140–170, 170–200, 200–250 Hz). That's 24 ch × 18 + f_rot = **433 features**.
* **Models** (`src/models.py`, fixed settings, no tuning): SVM-RBF (C = 10, balanced, standardised), Random Forest (200 trees, min_samples_leaf = 2, balanced_subsample, 4 cores), kNN (k = 7, standardised).
* **Threshold baseline (healthy-baseline detector):** z-score one DSP quantity against the healthy *training* windows: the 1× level on acc z of the 4 IMUs, or the RMS of the 12 acc channels. A window is flagged faulty if max|z| over channels exceeds the 99th percentile of that statistic on healthy training windows.
* **Evaluation (never a random window split):**
  * **Localization: leave-one-flight-out (28 folds).** For each rotor, an RF answers "is this rotor's propeller faulty?". Training uses every 2nd window to halve CPU; test windows aren't subsampled. Window-level metrics, plus a flight-level majority vote.
  * **Detection: time-block CV (6 folds; fold *k* tests block *k* of every flight).** This is the only option with one healthy flight, since leave-one-flight-out would leave no healthy training data when 0000 is held out. Train and test windows never overlap in time but come from the same flights, so these numbers are **optimistic** for a new flight or a new day.
  * **Notch comparison:** every window is filtered with zero-phase IIR notches at 1×, 2×, 3× f_rot (aliased where needed, Q = 5, similar to Betaflight's RPM filter). f_rot is estimated on the raw window first, standing in for the RPM a real RPM filter would know. The identical feature and model pipeline is then re-run.

#### 1.1 Detection (healthy vs faulty), time-block CV, windows (`results/tables/padre_detection_*.md`)

| Model | Balanced acc (raw) | Macro-F1 (raw) | Healthy recall | Faulty recall | Balanced acc (notched) |
|---|---|---|---|---|---|
| Threshold, 1× level (acc z) | 0.925 | 0.637 | 0.987 | 0.864 | **0.683** |
| Threshold, RMS (acc) | 0.973 | 0.839 | 0.977 | 0.969 | **0.786** |
| SVM-RBF | **0.992** | **0.994** | 0.984 | 1.000 | 0.975 |
| Random Forest | 0.977 | 0.987 | 0.954 | 1.000 | 0.907 |
| kNN | 0.965 | 0.956 | 0.935 | 0.996 | 0.887 |

Healthy = 306 windows (one flight), faulty = 8 262 windows. Confusion matrix: `results/figures/cm_padre_detection_svm.png`.

#### 1.2 Localization, leave-one-flight-out, Random Forest (`results/tables/padre_localization_LOFO_*.md`)

| Rotor | Balanced acc | Macro-F1 | Healthy-rotor recall | Faulty-rotor recall | Balanced acc (notched) |
|---|---|---|---|---|---|
| A | 0.954 | 0.962 | 1.000 | 0.909 | 0.945 |
| B | 0.910 | 0.910 | 0.931 | 0.889 | 0.866 |
| C | 0.904 | 0.917 | 0.996 | 0.812 | 0.841 |
| D | 0.734 | 0.741 | 0.878 | 0.589 | 0.592 |
| Exact 4-bit pattern, windows | 0.597 | | | | 0.411 |
| Exact 4-bit pattern, **flights** (majority vote) | **0.643 (18/28)** | | | | 0.429 (12/28) |
| Per-rotor state, **flights** (majority vote) | **0.902 (101/112)** | | | | 0.848 |

Per fault type (rate at which a rotor with that fault is flagged faulty, windows pooled over rotors): none **0.049** (false alarm), chipped edge **0.888**, bent tip **0.679**, cut-off tip **0.749**, bent + cut-off tip **0.997**.
Missed flights (`padre_localization_LOFO_rf_per_flight.md`): 0002, 0010, 0030, 0200 and 2000 were predicted all-healthy. Note that **0002, 0200 and 2000 are bent-tip faults** (0010 is a chipped edge on C and 0030 a cut-off tip on C). Multi-fault flights miss rotor D most often (1002, 1022, 1102). In 0001 a fault on D was attributed to B. In 1120, D was flagged although healthy.

#### 1.3 Severity, from spectral energy change (descriptive)

PADRE has **no severity labels**, so no severity classifier is trained. Mapping fault types onto "minor/major" would mean inventing labels. Instead we measure how much the vibration on the **faulty rotor's own IMU** (acc z) changes against the healthy flight 0000, for the 16 single-fault flights (`padre_energy_change_*.md`, `results/figures/padre_energy_by_fault_type.png`). Flight 0040 is excluded because its IMU C acc z is dead.

| Fault type | Δ 1×-harmonic level, mean [min, max] dB | Δ total band energy, mean dB | n flights |
|---|---|---|---|
| chipped edge | +13.9 [+0.8, +19.9] | +11.4 | 4 |
| bent tip | +0.2 [−12.9, +14.3] | +2.4 | 4 |
| cut-off tip | +17.5 [−1.1, +26.3] | +16.2 | 4 |
| bent tip + cut-off tip | +23.3 [+18.7, +26.4] | +19.3 | 3 (0040 excluded) |

Material-removal faults (chipped edge, cut-off tip) raise the rotor-frequency vibration by roughly 14–26 dB on rotors A, B and D. A bent tip changes it little, which matches it being the hardest fault to detect in 1.2. Rotor C shows small or negative changes because its IMU is already noisy in the healthy flight. This is a descriptive energy measure, not a validated severity estimator.

#### 1.4 Raw vs software notch-filtered (`results/tables/raw_vs_notch_summary.md`, `results/figures/raw_vs_notch.png`, `results/figures/padre_notch_effect.png`)

| Task | Raw | Notched |
|---|---|---|
| Detection, threshold on 1× level | 0.925 | **0.683** |
| Detection, threshold on RMS | 0.973 | **0.786** |
| Detection, SVM | 0.992 | 0.975 |
| Detection, RF | 0.977 | 0.907 |
| Localization, RF (LOFO, mean of 4 rotors) | 0.876 | 0.811 |

Notching the rotor harmonics removes most of the fault energy: on flight 3000, the 1× line on IMU A drops by ~30 dB (figure). Simple DSP detectors built on that line lose most of their power. The ML models degrade less because faults also change the broadband and off-harmonic spectrum, but localization loses 6 flights' exact patterns (18 → 12 of 28). **Implication:** fault monitoring should tap the gyro/accelerometer *before* Betaflight's RPM and dynamic notch filters (`gyroUnfilt`), not the filtered `gyroADC`. Stage 3 shows the same effect on your own AIR75 logs.

#### 1.5 Leakage check (`results/tables/leakage_demo_random_window_split.md`)

A random window-level 5-fold split, which is **invalid** because overlapping windows from the same flight land in both train and test, gives **0.992** balanced accuracy on the 28-way "which configuration" task. With one flight per configuration, that only measures re-recognising the same flight. It shows why all results above use flight- or time-block-grouped splits.

#### Figures (`results/figures/`)

| File | Content |
|---|---|
| `padre_psd_by_imu.png` | Welch PSD of acc z on IMUs A–D: healthy 0000 vs the four single-rotor-A fault types |
| `padre_spectrograms.png` | Spectrograms, IMU A acc z, healthy 0000 vs 3000 (A cut-off tip) |
| `padre_notch_effect.png` | PSD before/after the software 1×/2×/3× notch (flight 3000, IMU A) |
| `cm_padre_detection_svm.png` | Detection confusion matrix (SVM, time-block CV) |
| `cm_padre_localization_rf_lofo.png` | Per-rotor localization confusion matrix (RF, leave-one-flight-out, flight level) |
| `padre_energy_by_fault_type.png` | Δ 1×-level / Δ band energy on the faulty rotor's IMU, by fault type |
| `raw_vs_notch.png` | Raw vs notched balanced accuracy, all tasks |
| `own_betaflight_unfilt_vs_filtered.png` | Stage 3: your AIR75 log, gyroUnfilt vs gyroADC with eRPM motor bands (descriptive) |
| `nondrone_cwru_envelope_spectrum.png` | Stage 2: envelope spectrum on CWRU bearing data (non-drone) |

### Stage 2 — Motor faults

* **None of the downloaded drone datasets contains motor faults** (bearing wear, winding or magnet faults). PADRE and DronePropB only contain *propeller* damage. DronePropA and the PHM-AP UT dataset weren't downloaded, and per their papers they also contain only propeller faults (edge cut / crack / surface cut; imbalance / severe imbalance / crack).
* Nearest proxy: propeller damage produces a **mass/aerodynamic imbalance**, a 1× rotor-frequency vibration, which is also the first symptom of a bent shaft or loose bell. The Stage 1 results on the imbalance-type faults are that proxy. **They are propeller-fault results and must not be read as motor-wear results.** Bearing and winding faults produce different signatures (bearing defect frequencies, envelope modulation, electrical-frequency sidebands) that none of these datasets exercise.
* Non-drone add-on (`scripts/bearing_envelope_demo.py`): the envelope-spectrum feature (band-pass 2.5–5.5 kHz → Hilbert envelope → FFT) on 3 files of the **CWRU bearing dataset** (industrial test rig, 12 kHz drive end, 0.007" faults, ~1797 rpm). Inner-race fault: envelope peak 161.7 Hz vs theoretical BPFI 162.2 Hz (192× median floor). Outer-race fault: peak 107.6 Hz vs BPFO 107.4 Hz (847× floor). Ball fault: **not identified** (dominant peak at the 30 Hz shaft rate, BSF line only 5×), a known hard case on CWRU. Figure: `results/figures/nondrone_cwru_envelope_spectrum.png`; table: `results/tables/nondrone_cwru_envelope.csv`. **Not drone data; not used anywhere else.**

### Stage 3 — Audit of own drone data on this machine

Searched `~` (incl. Downloads, Documents; skipped system dirs, conda, snap, caches, `.config` firmware blobs) for `.bbl/.bfl`, blackbox CSV exports, `.ulg/.tlog` and other drone logs. Audit script: `scripts/audit_own_logs.py`.

| File(s) | What it is | Raw gyro | Log rate | Per-motor RPM | Fault ground truth | Usable? |
|---|---|---|---|---|---|---|
| `~/real_data/sessions/log_1…log_6.csv` (6 files; `log_4` is a byte-identical copy of `log_3`) | Betaflight 2026.6 blackbox CSV, craft "AIR75 R", BETAFPVG473 | yes: `gyroUnfilt[0..2]` + `gyroADC` | 3.2 kHz (looptime 312 µs, P-denom 1) | yes: `eRPM[0..3]` (bidir DShot, 12 poles) | **none** | **healthy-baseline-only** (and only if props/motors were known good) |
| `~/real_data/sessions/8_1…8_6.csv`, `FC dataflash{,2..5}.csv`, `flight5.csv` (12 files; `flight5` is a copy of `FC dataflash`) | same craft, blackbox CSV | yes | 824 Hz (P-denom 4) | yes | **none** | **healthy-baseline-only** |
| `~/Downloads/BTFL_BLACKBOX_LOG_AIR75_R_20260915_185723_BETAFPVG473_V2.BBL` (13 logs), `~/Documents/BTFL_BLACKBOX_LOG_20260901_130455_BETAFPVG473_V2_BMI270.BBL` (65 logs) | raw dataflash dumps, same craft (`rpm_filter_harmonics=3`, `dshot_bidir=1`) | yes (`gyroUnfilt` in field list) | 3.2 kHz loop | eRPM enabled | **none** | **healthy-baseline-only** (needs `blackbox_decode`, not installed) |
| `~/real_data/tests/synthetic_*blackbox.csv` | synthetic test fixtures, 100 Hz | no | 100 Hz | no | n/a | **no** (synthetic) |
| `~/crazyflie/logs/ranges_*.csv` | Crazyflie Multiranger distances, baro, vbat | no | — | no | none | **no** (no vibration signal) |
| `~/air75_raptor/examples/room_pose_*.csv` | camera/ArUco pose log | no | — | no | none | **no** |
| `~/Downloads/toolbox/uav/uavmatlab/flight.{ulg,tlog}`, PX4 `flight.tlog` test fixture | MATLAB / PX4 example files, not your drone | — | — | — | none | **no** |

**None of the own logs has a healthy-vs-faulty label, so none is used for supervised classification.** They're technically ideal for this project (unfiltered gyro at 3.2 kHz plus per-motor eRPM, which would enable true RPM-based order tracking). The only thing missing is ground truth: flights with a deliberately damaged propeller on a known motor. One descriptive figure (not a classification result) uses `log_1.csv` to show what Betaflight's filtering does on your real craft. In flight, the motor 1× band (299–359 Hz from eRPM) is **32.8 dB lower in `gyroADC` than in `gyroUnfilt`**. This includes Betaflight's LPFs as well as the RPM/dynamic notches. Figure: `results/figures/own_betaflight_unfilt_vs_filtered.png`.

---

## Limitations

* **One dataset, one airframe.** All results are from PADRE: one Parrot Bebop 2 and 28 flights, with each fault configuration flown once. Nothing here shows transfer to another drone, such as your AIR75.
* **One healthy flight.** Detection can only be evaluated within flights (time-disjoint blocks), which is optimistic. Localization is tested strictly on unseen flights.
* **No severity labels in PADRE.** Severity is reported only as a descriptive spectral-energy change per fault type; no severity classifier or regression was validated.
* **No worn-motor ground truth.** All labels are propeller damage. Motor (bearing, winding) faults are out of scope; see Stage 2.
* **No RPM.** f_rot is estimated from the dominant spectral peak, so true order tracking wasn't possible. The 500 Hz sampling aliases the 2× and 3× harmonics, which are handled by evaluating them at their aliased frequency.
* **No labelled Betaflight data.** Your own AIR75 logs have the right signals (gyroUnfilt at 3.2 kHz + eRPM) but no fault labels, so they are healthy-baseline-only. The Betaflight effect on the public data is *simulated* with software notches.
* **Sensor issues in PADRE:** clipping at ±16 g in heavy-fault flights (saturation itself becomes a crude cue), a dead IMU C z-axis in flight 0040, a noisier IMU C overall, and assumed LSB scaling.
* Fixed hyper-parameters, no nested tuning. RF localization trains on every 2nd window to limit CPU load.

## Manual steps for you

1. **Free disk space** (the root partition is ~98 % full). Only needed if you later want a second dataset:
   * DronePropA: <https://data.mendeley.com/datasets/ftdyxrr3c5/1>, put the `.mat` files into `data/raw/dronepropa/` (4.22 GB). Not wired into the pipeline yet.
   * PHM-AP 2025 UT dataset: <https://github.com/ailab-utokyo/ut-drone-dataset>; the zips are on Google Drive (`v3.1_normal`, `v3.1_anomaly_imbalanced_propeller`, `v3.2_*`, `v3.3_*`, ≈3.6 GB). Target: `data/raw/ut_drone/`. Note that per the paper the released logs are aligned/subsampled to ≈200 Hz, which limits harmonic analysis.
2. To make your own AIR75 logs usable for supervised work: fly short hover + throttle-sweep logs with a deliberately damaged propeller (e.g. chipped edge, then a larger cut) on one known motor, plus healthy flights on the same day and battery. Keep the 3.2 kHz logging with gyroUnfilt + eRPM, and record which motor and which damage in the file name.

## How to run

```bash
cd prop_fault_dsp
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -I scripts/download_padre.py      # PADRE raw IMU files (~290 MB) -> data/raw/padre/
nice -n 10 .venv/bin/python -I run_all.py           # all Stage 1 tables/figures, ~12 min, 4 cores
nice -n 10 .venv/bin/python -I run_all.py --force   # also recompute cached features (+~1 min)
# Stage 2 / 3 helpers (seconds each):
.venv/bin/python -I scripts/bearing_envelope_demo.py data/raw/cwru_bearing results/figures/nondrone_cwru_envelope_spectrum.png results/tables/nondrone_cwru_envelope.csv
.venv/bin/python -I scripts/audit_own_logs.py ~/real_data/sessions/*.csv
.venv/bin/python -I scripts/betaflight_filter_demo.py ~/real_data/sessions/log_1.csv results/figures/own_betaflight_unfilt_vs_filtered.png
```

Layout: `src/loaders.py` (PADRE → common schema; a DronePropB loader is kept but unused), `src/features.py` (windowing, f_rot, notch, features), `src/models.py` (classifiers, metrics), `src/stage1.py` (experiments), `src/figures.py`, `run_all.py`, `scripts/` (Stage 2/3 helpers), `results/tables`, `results/figures`, `results/summary.json`, `results/run_all.log`.

## References

* Cabahug, J., & Eslamiat, H. (2022). *Failure Detection in Quadcopter UAVs Using K-Means Clustering*. Sensors, 22(16), 6037. doi:10.3390/s22166037. (Cited in the brief as "Eslamiat et al. 2022".)
* Baldini, A., Felicetti, R., Ferracuti, F., Freddi, A., Iarlori, S., & Monteriù, A. (2023). *Real-time propeller fault detection for multirotor drones based on vibration data analysis*. Engineering Applications of Artificial Intelligence, 123, 106343. doi:10.1016/j.engappai.2023.106343. (Cited in the brief as "Freddi & Iarlori 2023".)
* Ghalamchi, B., Jia, Z., & Mueller, M. W. (2020). *Real-time vibration-based propeller fault diagnosis for multicopters*. IEEE/ASME Transactions on Mechatronics, 25(1), 395–405 (please double-check the volume and pages).
* AeroLab, Poznan University of Technology. PADRE – UAV_measurement_data, github.com/AeroLabPUT/UAV_measurement_data; JINT 2024, doi:10.1007/s10846-024-02101-7.
* Ismail, M. A. A., Elshaar, M. E., Abdalla, A., & Quan, Q. (2024). *DronePropB: Ground Testing Dataset for Commercial Drones with Defective Propellers*. Mendeley Data, v1, doi:10.17632/xkvfjmm8zg.1.
* Ismail, M. A. A., et al. (2025). *DronePropA: Motion trajectories dataset for defective drones*. Data in Brief 60, 111589, doi:10.1016/j.dib.2025.111589.
* Liew, C. F., et al. (2025). *Development and Sharing of a Multi-Modal Indoor UAV Dataset for PHM Research*. PHM Asia Pacific 2025 (paper 4701).
* Case Western Reserve University Bearing Data Center (CWRU) — non-drone bearing data used only for the envelope-spectrum demo.
