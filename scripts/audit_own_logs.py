"""Stage 3: audit the user's own drone logs for usability in fault classification.
Usage: python -I scripts/audit_own_logs.py <file> [<file> ...]  -> prints a table (markdown)."""
import os, re, sys
import numpy as np

KEYS = ["Craft name", "looptime", "gyro_sync_denom", "pid_process_denom", "frameIntervalPDenom",
        "frameIntervalPNum", "debug_mode", "dshot_bidir", "rpm_filter_harmonics", "motor_poles",
        "fields_disabled_mask", "blackbox_high_resolution", "gyro_lpf1_static_hz", "dyn_notch_count",
        "Log start datetime"]

def audit_csv(path):
    meta, header, nrows, first, last = {}, None, 0, None, None
    with open(path, errors="replace") as fh:
        for line in fh:
            if header is None:
                if line.startswith('"loopIteration"') or line.startswith("loopIteration"):
                    header = [c.strip().strip('"') for c in line.strip().split(",")]
                    continue
                m = re.match(r'^"([^"]+)",(.*)$', line.strip())
                if m and m.group(1) in KEYS:
                    meta[m.group(1)] = m.group(2).strip('"')
            else:
                nrows += 1
                if first is None: first = line
                last = line
    if header is None:
        return {"file": path, "error": "no data header"}
    ti = header.index("time") if "time" in header else None
    dur = rate = None
    if ti is not None and first and last:
        t0, t1 = float(first.split(",")[ti]), float(last.split(",")[ti])
        dur = (t1 - t0) / 1e6; rate = nrows / dur if dur > 0 else None
    has = lambda pat: [h for h in header if re.search(pat, h)]
    return {"file": path, "rows": nrows, "duration_s": dur, "log_rate_hz": rate, "meta": meta,
            "gyroUnfilt": has(r"gyroUnfilt"), "gyroADC": has(r"gyroADC"), "debug": has(r"^debug\[0\]"),
            "eRPM": has(r"eRPM"), "motor": has(r"^motor\[0\]"), "acc": has(r"accSmooth")}

def audit_bbl(path):
    with open(path, "rb") as fh: data = fh.read()
    n_logs = data.count(b"H Product:Blackbox flight data recorder")
    hdr = {}
    for m in re.finditer(rb"H ([^:\n]+):([^\n]*)\n", data[:200000]):
        k = m.group(1).decode(errors="replace")
        if k in KEYS or k.startswith("Field I name"):
            hdr.setdefault(k, m.group(2).decode(errors="replace")[:400])
    used = len(data.rstrip(b"\xff"))
    return {"file": path, "size": len(data), "bytes_used": used, "n_logs": n_logs, "hdr": hdr}

for p in sys.argv[1:]:
    if p.lower().endswith(".bbl"):
        r = audit_bbl(p)
        f = r["hdr"].get("Field I name", "")
        print(f"BBL {os.path.basename(p)} logs={r['n_logs']} used={r['bytes_used']/1e6:.1f}MB "
              f"gyroUnfilt={'gyroUnfilt' in f} eRPM={'eRPM' in f} debug={'debug' in f}",
              {k: v for k, v in r["hdr"].items() if not k.startswith("Field")})
    else:
        r = audit_csv(p)
        if "error" in r: print(p, r["error"]); continue
        print(f"CSV {os.path.basename(p)} rows={r['rows']} dur={r['duration_s']:.1f}s rate={r['log_rate_hz']:.0f}Hz "
              f"gyroUnfilt={len(r['gyroUnfilt'])>0} gyroADC={len(r['gyroADC'])>0} debug={len(r['debug'])>0} "
              f"eRPM={len(r['eRPM'])>0} motor={len(r['motor'])>0} acc={len(r['acc'])>0}", r["meta"])
