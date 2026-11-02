"""Windowing, notch filtering and spectral feature extraction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import detrend, filtfilt, iirnotch, welch
from scipy.stats import kurtosis


def make_windows(x: np.ndarray, fs: float, win_s: float = 1.0, overlap: float = 0.5,
                 start: int = 0, stop: int | None = None):
    """x: (n_ch, n_samp) -> windows (n_win, n_ch, n_win_samp), start indices."""
    stop = x.shape[1] if stop is None else stop
    n = int(round(win_s * fs))
    hop = int(round(n * (1 - overlap)))
    starts = np.arange(start, stop - n + 1, hop)
    if len(starts) == 0:
        return np.empty((0, x.shape[0], n)), starts
    idx = starts[:, None] + np.arange(n)[None, :]
    w = x[:, idx].transpose(1, 0, 2)
    return detrend(w, axis=-1, type="linear"), starts


def alias(f, fs):
    """Apparent frequency of a tone at f after sampling at fs (folding)."""
    f = np.mod(f, fs)
    return np.where(f > fs / 2, fs - f, f)


def estimate_frot(f, P, band):
    """Rotor frequency per window = dominant peak of the channel-summed, per-channel-normalised
    PSD inside `band` (no RPM available). P: (n_win, n_ch, n_f)."""
    sel = (f >= band[0]) & (f <= band[1])
    Pn = P[..., sel] / (P[..., sel].sum(axis=-1, keepdims=True) + 1e-30)
    S = Pn.sum(axis=1)
    k = np.argmax(S, axis=-1)
    # parabolic interpolation for sub-bin accuracy
    fsel = f[sel]
    df = fsel[1] - fsel[0]
    k = np.clip(k, 1, len(fsel) - 2)
    a, b, c = (S[np.arange(len(k)), k + d] for d in (-1, 0, 1))
    delta = 0.5 * (a - c) / (a - 2 * b + c + 1e-30)
    return fsel[k] + np.clip(delta, -0.5, 0.5) * df


def harmonic_sum_frot(f, P, band, fs, n_harm=3):
    """Segment-level fundamental via harmonic summation on the mean, channel-normalised PSD:
    score(f0) = sum_k S(alias(k f0)) / k, f0 in band. Robust to the 2x line dominating."""
    Pn = P / (P.sum(axis=-1, keepdims=True) + 1e-30)
    S = Pn.sum(axis=1).mean(axis=0)                      # (n_f,)
    sel = (f >= band[0]) & (f <= band[1])
    fp = f[sel][np.argmax(S[sel])]
    # sub-harmonic check: if a clear line exists at fp/2 (within 6 dB of the peak), the
    # dominant line was the 2nd harmonic -> take fp/2 as the fundamental.
    half = (np.abs(f - fp / 2) <= 0.08 * fp / 2) & (f >= band[0])
    if half.any() and S[half].max() >= S[sel].max() * 10 ** (-6 / 10):
        fp = f[half][np.argmax(S[half])]
    return float(fp)


def estimate_frot_robust(f, P, band, rel=0.15, fs=None):
    """Two-pass estimate: (1) segment-level fundamental by harmonic summation;
    (2) per-window dominant peak within +-rel of it (tracks throttle changes)."""
    fs = fs if fs is not None else 2 * f[-1]
    med = harmonic_sum_frot(f, P, band, fs)
    lo, hi = max(band[0], med * (1 - rel)), min(band[1], med * (1 + rel))
    return estimate_frot(f, P, (lo, hi))


def notch_windows(W: np.ndarray, fs: float, frot: np.ndarray, n_harm: int = 3, Q: float = 5.0):
    """Software 'RPM filter': per window, notch 1x..n_harm x of f_rot (aliased if above Nyquist),
    like Betaflight's RPM/dynamic notch filters (Q ~ 5). Zero-phase filtfilt."""
    out = W.copy()
    for i, f0 in enumerate(frot):
        for k in range(1, n_harm + 1):
            fk = float(alias(k * f0, fs))
            if 2.0 < fk < fs / 2 - 2.0:
                b, a = iirnotch(fk, Q, fs)
                out[i] = filtfilt(b, a, out[i], axis=-1, padlen=min(3 * max(len(a), len(b)), W.shape[-1] - 1))
    return out


def harmonic_amp(f, P, frot, fs, k, tol_hz):
    """Peak PSD within +-tol of the (aliased) k-th harmonic, per window and channel."""
    fk = alias(k * frot, fs)                                     # (n_win,)
    m = np.abs(f[None, :] - fk[:, None]) <= tol_hz               # (n_win, n_f)
    Pm = np.where(m[:, None, :], P, 0.0)
    return Pm.max(axis=-1)


def window_features(W, fs, ch_names, bands, frot_band, nperseg, frot=None, tol_hz=None):
    """Feature table for windows W (n_win, n_ch, n). Returns DataFrame, f, P, frot."""
    f, P = welch(W, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, axis=-1, detrend="constant")
    if frot is None:
        frot = estimate_frot_robust(f, P, frot_band)
    tol_hz = tol_hz if tol_hz is not None else 2.5 * (f[1] - f[0])
    feats = {"f_rot": frot}
    Psum = P.sum(axis=-1) + 1e-30
    pn = P / Psum[..., None]
    centroid = (pn * f).sum(axis=-1)
    entropy = -(pn * np.log(pn + 1e-30)).sum(axis=-1) / np.log(P.shape[-1])
    rms = np.sqrt((W ** 2).mean(axis=-1))
    kurt = kurtosis(W, axis=-1)
    crest = np.abs(W).max(axis=-1) / (rms + 1e-30)
    df = f[1] - f[0]
    harm = [harmonic_amp(f, P, frot, fs, k, tol_hz) for k in (1, 2, 3)]
    for j, ch in enumerate(ch_names):
        feats[f"{ch}_rms"] = np.log10(rms[:, j] + 1e-12)
        feats[f"{ch}_kurt"] = kurt[:, j]
        feats[f"{ch}_crest"] = crest[:, j]
        feats[f"{ch}_centroid"] = centroid[:, j]
        feats[f"{ch}_entropy"] = entropy[:, j]
        for k, h in zip((1, 2, 3), harm):
            feats[f"{ch}_H{k}"] = np.log10(h[:, j] + 1e-30)
        feats[f"{ch}_H2/H1"] = feats[f"{ch}_H2"] - feats[f"{ch}_H1"]
        for lo, hi in bands:
            sel = (f >= lo) & (f < hi)
            feats[f"{ch}_bp_{lo:g}_{hi:g}"] = np.log10(P[:, j, sel].sum(axis=-1) * df + 1e-30)
    return pd.DataFrame(feats), f, P, frot


def recordings_to_features(recs, win_s, overlap, bands, frot_band, nperseg,
                           notch=False, notch_Q=5.0, n_harm=3, block_count=None,
                           keep_psd=False):
    """Window every recording and build one feature table (+ label/group columns).

    block_count: if set, each recording is cut into that many contiguous time blocks and
    windows never straddle a block boundary (column 'block' added) -> allows time-block CV
    without overlapping windows between train and test.
    """
    rows, psds = [], []
    for r in recs:
        x = r.signals
        fb = frot_band(r) if callable(frot_band) else frot_band
        n = x.shape[1]
        if block_count:
            edges = np.linspace(0, n, block_count + 1).astype(int)
            segs = [(b, edges[b], edges[b + 1]) for b in range(block_count)]
        else:
            segs = [(0, 0, n)]
        for b, s0, s1 in segs:
            W, starts = make_windows(x, r.fs, win_s, overlap, s0, s1)
            if len(W) == 0:
                continue
            frot = None
            if notch:
                # estimate f_rot on the raw window first (an RPM filter knows RPM; we don't)
                fr, Pr = welch(W, fs=r.fs, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
                frot = estimate_frot_robust(fr, Pr, fb)
                W = notch_windows(W, r.fs, frot, n_harm=n_harm, Q=notch_Q)
            df, f, P, frot = window_features(W, r.fs, r.ch_names, bands, fb, nperseg, frot=frot)
            df["experiment_id"] = r.experiment_id
            df["label_health"] = r.label_health
            df["label_severity"] = -1 if r.label_severity is None else r.label_severity
            df["label_motor"] = r.label_motor if r.label_motor is not None else ""
            df["block"] = b
            df["t_start_s"] = starts / r.fs
            for k, v in r.meta.items():
                if isinstance(v, (int, float, str, np.integer, np.floating)):
                    df[f"meta_{k}"] = v
            rows.append(df)
            if keep_psd:
                psds.append((r.experiment_id, b, f, P.mean(axis=0)))
    out = pd.concat(rows, ignore_index=True)
    return (out, psds) if keep_psd else out
