"""Download only the PADRE raw IMU files + Readmes (~290 MB) into data/raw/padre/.
The full repository is ~4.8 GB (mostly derived FFT/DWT/EMD products we do not use)."""
import json, os, urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "padre")
REPO = "AeroLabPUT/UAV_measurement_data"
tree = json.load(urllib.request.urlopen(f"https://api.github.com/repos/{REPO}/git/trees/HEAD?recursive=1"))
paths = [t["path"] for t in tree["tree"] if t["type"] == "blob" and (
    ("Raw_data" in t["path"] and "MIC" not in t["path"]) or t["path"].lower().endswith(("readme.txt", "readme.md")))]
for i, p in enumerate(paths, 1):
    out = os.path.join(ROOT, p)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        continue
    os.makedirs(os.path.dirname(out), exist_ok=True)
    urllib.request.urlretrieve(f"https://raw.githubusercontent.com/{REPO}/main/{p.replace(' ', '%20')}", out)
    print(f"[{i}/{len(paths)}] {p}", flush=True)
print("done:", len(paths), "files in", os.path.abspath(ROOT))
