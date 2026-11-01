"""Do cong suat kernel G3 v2 (cach do nhu muc 13). Python PYNQ (root), SAU g3_pl_rt.py.
Vong PL bang driver C (g3_loop, xoay vong 8 bo dem, tham so fold 1).
Dieu kien (moi doan DUR s, lay mau cac rail PYNQ moi DT s trong mot luong rieng):
  idle  : overlay da nap, khong tinh
  pl100 / pl250 : vong giai ma lien tuc tren PL bang driver C (start + polling), dem so quyet dinh
  ps100 / ps250 : golden C++ 1 luong tren PS (../golden_T<T> ... che do do, R chon ~DUR s)
Thu tu moi lap: idle, pl100, idle, pl250, idle, ps100, idle, ps250, idle. REPS lap.
Duong co so cua mot dieu kien = TB 2 doan idle ngay truoc va sau. Ghi g3_power_samples_<overlay>.csv + g3_power_segments_<overlay>.json.
Dung: python g3_power.py [REPS=3] [DUR=60] [DT=0.1] [OVERLAY=g3_T100.bit]  (pl<T> chi voi IP co trong overlay)
"""
import csv
import ctypes
import json
import subprocess
import sys
import threading
import time

import numpy as np
from pynq import Overlay, allocate, get_rails

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 60
DT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
OV = sys.argv[4] if len(sys.argv) > 4 else "g3rt_T100.bit"
STEM = OV.rsplit(".", 1)[0]
K = 40
REG = json.load(open("regs.json"))
drv = ctypes.CDLL("./libg3drv.so")
drv.g3_loop.argtypes = [ctypes.c_uint64, ctypes.c_uint32, ctypes.c_double, ctypes.c_int, ctypes.POINTER(ctypes.c_uint64),
                        ctypes.POINTER(ctypes.c_double)]
drv.g3_loop.restype = ctypes.c_long
PS_US = {100: 6123.0, 250: 13149.0}       # trung vi do o muc 12, de chon R ~ DUR

ov = Overlay(OV)
TS = [T for T in (100, 250) if f"g3_T{T}" in ov.ip_dict]
print("overlay", OV, "IP G3:", TS, flush=True)
rails = {k: v.power for k, v in get_rails().items() if v.power is not None}
print("rail co cong suat:", list(rails), flush=True)

samples, seg_label, stop = [], ["idle"], threading.Event()


def sampler():
    while not stop.is_set():
        samples.append([time.monotonic(), seg_label[0]] + [rails[k].value for k in rails])   # thoi gian tuyet doi
        time.sleep(DT)


th = threading.Thread(target=sampler, daemon=True)
th.start()

bufs = {}


def setaddr(ip, name, buf):
    ip.write(REG[name], buf.device_address & 0xFFFFFFFF)
    ip.write(REG[name] + 4, buf.device_address >> 32)


for T in TS:
    man = json.load(open(f"T{T}/manifest.json"))
    C, TT = man["C"], man["S0"] + man["T"]
    ip = getattr(ov, f"g3_T{T}")
    pb_ = allocate((man["PRM_LEN"],), dtype=np.int32)
    sb, prb, stb = allocate((K,), dtype=np.int64), allocate((1,), dtype=np.int32), allocate((1,), dtype=np.int32)
    ins = [allocate((C * TT,), dtype=np.int16) for _ in range(8)]
    x = np.fromfile(f"T{T}/fold1_input.bin", dtype="<i2").reshape(-1, C * TT)[:8]
    for k in range(8):
        ins[k][:] = x[k]
        ins[k].flush()
    pb_[:] = np.fromfile(f"T{T}/fold1_params.bin", dtype="<i4")
    pb_.flush()
    for name, buf in (("prm", pb_), ("in", ins[0]), ("score", sb), ("pred", prb), ("status", stb)):
        setaddr(ip, name, buf)
    ip.write(REG["mode"], 0)
    ip.write(REG["ctrl"], 1)
    while not (ip.read(REG["ctrl"]) & 0x2):
        pass
    stb.invalidate()
    assert int(stb[0]) == 1, f"nap tham so T{T} loi"
    ip.write(REG["mode"], 1)
    bufs[T] = (ov.ip_dict[f"g3_T{T}"]["phys_addr"], (ctypes.c_uint64 * 8)(*[b.device_address for b in ins]), ins, pb_, sb, prb, stb)


def run_pl(T):
    phys, addrs = bufs[T][0], bufs[T][1]
    el = ctypes.c_double(0)
    n = drv.g3_loop(phys, REG["in"], DUR, 8, addrs, ctypes.byref(el))
    if n < 0:
        raise SystemExit("g3_loop loi")
    return int(n), el.value


def run_ps(T):
    R = int(DUR * 1e6 / PS_US[T])
    t0 = time.monotonic()
    out = subprocess.run(["../golden_T%d" % T, "../T%d" % T, "8", str(R)], capture_output=True, text=True)
    el = time.monotonic() - t0
    if out.returncode != 0:
        raise SystemExit(f"golden_T{T} loi rc={out.returncode}: {out.stdout} {out.stderr}")
    return R, el     # gom ca lan chay lo dau (8 trial) va lam nong: khong dang ke so voi R


segs = []


def segment(label, fn=None):
    seg_label[0] = label
    t0 = time.monotonic()
    if fn is None:
        time.sleep(DUR)
        n, el = 0, time.monotonic() - t0
    else:
        n, el = fn()
    segs.append(dict(label=label, start=t0, end=time.monotonic(), n=n, elapsed=el))
    print(f"doan {label}: {el:.1f} s, {n} quyet dinh", flush=True)


for rep in range(1, REPS + 1):
    segment("idle")
    conds = [(f"pl{T}", (lambda T=T: run_pl(T))) for T in TS] + [(f"ps{T}", (lambda T=T: run_ps(T))) for T in (100, 250)]
    for lab, fn in conds:
        segment(lab, fn)
        segment("idle")
stop.set()
th.join()
names = list(rails)
with open(f"g3_power_samples_{STEM}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t", "seg"] + names)
    w.writerows(samples)
# gan chi so doan cho moi mau theo thoi gian
ts = np.array([s[0] for s in samples])
P = np.array([s[2:] for s in samples], dtype=float)
res = []
for i, s in enumerate(segs):
    m = (ts >= s["start"] + 2) & (ts <= s["end"] - 1)       # bo 2 s dau, 1 s cuoi moi doan
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
json.dump(dict(overlay=OV, windows_pl=TS, rails=names, reps=REPS, dur=DUR, dt=DT, segments=segs), open(f"g3_power_segments_{STEM}.json", "w"), indent=1)
print("G3_POWER_DONE", flush=True)
