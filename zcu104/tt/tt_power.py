"""Do cong suat tt-CCA va FBCCA ban A tren PS (cach do nhu muc 16 / g3_power_rt.py). Python PYNQ (root).
Thu tu moi lap: idle, tt100, idle, tt250, idle, fb100, idle, fb250, idle. REPS lap, moi doan DUR s, lay mau cac rail moi DT s.
Duong co so mot dieu kien = TB 2 doan idle ke truoc/sau. Ghi tt_power_samples.csv + tt_power_segments.json.
Dung: python tt_power.py [REPS=3] [DUR=60] [DT=0.1]
"""
import csv
import json
import re
import subprocess
import sys
import threading
import time

import numpy as np
from pynq import get_rails

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 60
DT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
rails = {k: v.power for k, v in get_rails().items() if v.power is not None}
print("rail co cong suat:", list(rails), flush=True)
samples, lab, stop = [], ["idle"], threading.Event()


def sampler():
    while not stop.is_set():
        samples.append([time.monotonic(), lab[0]] + [rails[k].value for k in rails])
        time.sleep(DT)


th = threading.Thread(target=sampler, daemon=True)
th.start()
segs = []


def segment(label, T=None, m=None):
    lab[0] = label
    t0 = time.monotonic()
    n, el = 0, 0.0
    if T is None:
        time.sleep(DUR)
        el = time.monotonic() - t0
    else:
        out = subprocess.run(["./ttcca_A", f"T{T}", "loop", m, str(DUR)], capture_output=True, text=True)
        m = re.search(r"n=(\d+) elapsed=([\d.]+)", out.stdout)
        if out.returncode != 0 or not m:
            raise SystemExit(f"ttcca_A loi: {out.stdout} {out.stderr}")
        n, el = int(m.group(1)), float(m.group(2))
    segs.append(dict(label=label, start=t0, end=time.monotonic(), n=n, elapsed=el))
    print(f"doan {label}: {el:.1f} s, {n} quyet dinh", flush=True)


for rep in range(REPS):
    segment("idle")
    for m in ("tt", "fb"):
        for T in (100, 250):
            segment(f"{m}{T}", T, m)
            segment("idle")
stop.set()
th.join()
names = list(rails)
with open("tt_power_samples.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t", "seg"] + names)
    w.writerows(samples)
ts = np.array([s[0] for s in samples])
P = np.array([s[2:] for s in samples], dtype=float)
for s in segs:
    m = (ts >= s["start"] + 2) & (ts <= s["end"] - 1)
    s["mean_W"] = {k: float(P[m, j].mean()) for j, k in enumerate(names)} if m.any() else {}
    s["n_samples"] = int(m.sum())
    s.pop("start"); s.pop("end")
for i, s in enumerate(segs):
    if s["label"] != "idle":
        a, b = segs[i - 1], segs[i + 1]
        s["base_W"] = {k: (a["mean_W"][k] + b["mean_W"][k]) / 2 for k in names}
        s["delta_W"] = {k: s["mean_W"][k] - s["base_W"][k] for k in names}
        s["t_per_decision_ms"] = 1e3 * s["elapsed"] / s["n"]
        s["energy_per_decision_mJ"] = {k: s["delta_W"][k] * s["t_per_decision_ms"] for k in names}
json.dump(dict(rails=names, reps=REPS, dur=DUR, dt=DT, segments=segs), open("tt_power_segments.json", "w"), indent=1)
print("TT_POWER_DONE", flush=True)
