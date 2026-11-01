"""Xac minh du lieu (khong lien quan mo hinh): pha dap ung SSVEP o tan so co ban cua tung muc tieu,
Benchmark so voi BETA (anh xa theo tan so). Trung binh tin hieu theo nguoi + block, kenh Oz, cua so 1 s
bat dau o [kich thich + tre] cua tung bo. Neu pha kich thich hai bo trung nhau, do lech pha
(BETA - Benchmark) chi con phan do khac tre (0,13 vs 0,14 s): 2*pi*f*(-0,01) -> lech nho va doi theo f.
Neu lech them hang so ~pi/2 thi cach hieu dinh chinh pha BETA sai."""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ssvep.data import load_subject, window  # noqa: E402

out = {}
for ds, n in (("benchmark", 35), ("beta", 70)):
    acc = None
    for s in range(1, n + 1):
        X, meta = load_subject(ds, s)
        Xw = window(X, meta, 1.0)                         # block, target, ch, T
        oz = [c.upper() for c in meta["channels"]].index("OZ")
        m = Xw[:, :, oz, :].mean(axis=0)                  # target, T
        acc = m if acc is None else acc + m
    acc /= n
    fs = meta["srate"]
    t = np.arange(acc.shape[-1]) / fs
    ph = {}
    for k, f in enumerate(meta["freqs"]):
        z = np.sum((acc[k] - acc[k].mean()) * np.exp(-2j * np.pi * f * t))
        ph[round(f, 2)] = (float(np.angle(z)), float(np.abs(z)))
    out[ds] = (ph, meta["latency_s"])
bph, bl = out["benchmark"]
tph, tl = out["beta"]
res = []
for f in sorted(bph):
    d = tph[f][0] - bph[f][0]
    d_lat = 2 * math.pi * f * (tl - bl)                  # anh huong cua khac tre cua so
    r = (d - d_lat + math.pi) % (2 * math.pi) - math.pi
    res.append((f, r))
r = np.array([x[1] for x in res])
print("do lech pha BETA - Benchmark sau khi tru anh huong tre (rad), theo tan so:")
for f, x in res:
    print(f"  {f:5.1f} Hz: {x:+.2f}")
print(f"TB vong (circular mean) = {np.angle(np.mean(np.exp(1j * r))):+.3f} rad; "
      f"do tap trung R = {np.abs(np.mean(np.exp(1j * r))):.3f}; pi/2 = {math.pi / 2:.3f}")
