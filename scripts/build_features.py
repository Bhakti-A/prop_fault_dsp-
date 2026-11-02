"""Compute and cache the PADRE feature tables (raw + notch-filtered). run_all.py does this too."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from stage1 import padre_features
force = "--force" in sys.argv
for notch in (False, True):
    t = time.time(); df = padre_features(notch, force)
    print(f"PADRE notch={notch}: {df.shape[0]} windows, {df.experiment_id.nunique()} flights, {time.time()-t:.0f}s", flush=True)
