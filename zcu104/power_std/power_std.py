"""Do nang luong chuan hoa. Python PYNQ (root), chay tren board, ghim loi 0 (taskset -c 0).

Khac muc 16/20/23: PL chay auto-restart (ap_ctrl bit 7) va PS NGU (time.sleep), khong tham do ban; moi tac vu PS 1 luong ghim
loi 1. Cung phien cho moi bo giai ma. Moi lap r (0..REPS-1), moi overlay T (100, 250): nap overlay, on dinh 30 s (khong tinh),
roi doan rảnh + cac dieu kien {pl, g31ps, ens, tt, fb} theo thu tu xao tron (random.Random(1000 * r + T)), moi dieu kien kep giua
hai doan ranh. Lay mau rail PYNQ + nhiet do PS/PL (AMS) moi DT s.
So quyet dinh doan pl = thoi gian doan / do tre kernel do trong phien (g3_bench, 2 000 lan, trung vi). Cuoi doan pl: tat
auto-restart, cho ap_idle, chay mot lan, so score voi golden fold 1 luot 0 -> pl_ok.
Dung: python power_std.py [REPS=5] [DUR=60] [DT=0.1]   -> power_std_samples.csv, power_std_segments.json
"""
import csv
import ctypes
import json
import random
import re
import subprocess
import sys
import threading
import time

import numpy as np
from pynq import Overlay, allocate, get_rails

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 60
DT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
B = "/home/xilinx/ssvep_lite_pkg_g31"
K = 40
REG = json.load(open(f"{B}/pl_rt/regs.json"))
drv = ctypes.CDLL(f"{B}/pl_rt/libg3drv.so")
drv.g3_bench.argtypes = [ctypes.c_uint64, ctypes.c_uint32, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint64),
                         ctypes.POINTER(ctypes.c_double)]
drv.g3_bench.restype = ctypes.c_int
PS_US = {100: 6123.0, 250: 13149.0}          # trung vi G3/G3.1 tren PS (muc 12, 20), chi de chon R ~ DUR
PIN = ["taskset", "-c", "1"]

rails = {k: v.power for k, v in get_rails().items() if v.power is not None}
AMS = "/sys/bus/iio/devices/iio:device0/"


def temp(ch):
    rd = lambda f: float(open(AMS + f).read())  # noqa: E731
    return (rd(f"in_{ch}_raw") + rd(f"in_{ch}_offset")) * rd(f"in_{ch}_scale") / 1000.0


samples, lab, stop = [], ["idle"], threading.Event()


def sampler():
    while not stop.is_set():
        samples.append([time.monotonic(), lab[0]] + [rails[k].value for k in rails] +
                       [temp("temp0_ps_temp"), temp("temp2_pl_temp")])
        time.sleep(DT)


def ps_cmd(meth, T):
    if meth == "g31ps":
        return PIN + [f"{B}/golden_T{T}", f"{B}/T{T}", "8", str(int(DUR * 1e6 / PS_US[T]))], B, None
    if meth == "ens":
        return PIN + ["./ensdnn_A", f"T{T}", "loop", str(DUR)], f"{B}/ens", r"n=(\d+) elapsed=([\d.]+)"
    return PIN + ["./ttcca_A", f"T{T}", "loop", meth, str(DUR)], f"{B}/tt", r"n=(\d+) elapsed=([\d.]+)"


def run_ps(meth, T):
    cmd, cwd, pat = ps_cmd(meth, T)
    t0 = time.monotonic()
    out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    el = time.monotonic() - t0
    if out.returncode != 0:
        raise SystemExit(f"{meth}{T} loi rc={out.returncode}: {out.stdout[-300:]} {out.stderr[-300:]}")
    if pat is None:
        return int(cmd[-1]), el
    m = re.search(pat, out.stdout)
    return int(m.group(1)), float(m.group(2))


class PL:
    def __init__(self, T):
        self.T = T
        self.ov = Overlay(f"{B}/pl_rt/g31rt_T{T}.bit")
        self.ip = getattr(self.ov, f"g3_T{T}")
        self.phys = self.ov.ip_dict[f"g3_T{T}"]["phys_addr"]
        d = f"{B}/pl_rt/T{T}"
        man = json.load(open(f"{d}/manifest.json"))
        C, TT = man["C"], man["S0"] + man["T"]
        self.pb = allocate((man["PRM_LEN"],), dtype=np.int32)
        self.sb, self.prb, self.stb = allocate((K,), dtype=np.int64), allocate((1,), dtype=np.int32), allocate((1,), dtype=np.int32)
        self.ins = [allocate((C * TT,), dtype=np.int16) for _ in range(8)]
        x = np.fromfile(f"{d}/fold1_input.bin", dtype="<i2").reshape(-1, C * TT)[:8]
        for k in range(8):
            self.ins[k][:] = x[k]
            self.ins[k].flush()
        self.es = np.fromfile(f"{d}/fold1_score.bin", dtype="<i8").reshape(-1, K)[0]
        self.pb[:] = np.fromfile(f"{d}/fold1_params.bin", dtype="<i4")
        self.pb.flush()
        for name, buf in (("prm", self.pb), ("in", self.ins[0]), ("score", self.sb), ("pred", self.prb), ("status", self.stb)):
            self.setaddr(name, buf)
        self.ip.write(REG["mode"], 0)
        self.call()
        self.stb.invalidate()
        assert int(self.stb[0]) == 1, f"nap tham so T{T} loi"
        self.ip.write(REG["mode"], 1)
        addrs = (ctypes.c_uint64 * 8)(*[b.device_address for b in self.ins])
        us = (ctypes.c_double * 2000)()
        assert drv.g3_bench(self.phys, REG["in"], 2000, 8, addrs, us) == 0
        self.t_us = float(np.median(np.ctypeslib.as_array(us)))
        self.setaddr("in", self.ins[0])

    def setaddr(self, name, buf):
        self.ip.write(REG[name], buf.device_address & 0xFFFFFFFF)
        self.ip.write(REG[name] + 4, buf.device_address >> 32)

    def call(self):
        self.ip.write(REG["ctrl"], 1)
        while not (self.ip.read(REG["ctrl"]) & 0x2):
            pass

    def run(self):
        t0 = time.monotonic()
        self.ip.write(REG["ctrl"], 0x81)                 # ap_start + auto_restart
        time.sleep(DUR)
        self.ip.write(REG["ctrl"], 0x00)                 # tat auto_restart; lan dang chay ket thuc roi idle
        el = time.monotonic() - t0
        while not (self.ip.read(REG["ctrl"]) & 0x4):
            pass
        self.call()
        self.sb.invalidate()
        ok = bool((np.array(self.sb) == self.es).all())
        return int(el * 1e6 / self.t_us), el, ok


segs = []


def segment(label, fn=None):
    lab[0] = label
    t0 = time.monotonic()
    rec = dict(label=label)
    if fn is None:
        time.sleep(DUR)
        rec.update(n=0, elapsed=time.monotonic() - t0)
    else:
        r = fn()
        rec.update(n=r[0], elapsed=r[1])
        if len(r) > 2:
            rec["pl_ok"] = r[2]
    rec.update(start=t0, end=time.monotonic())
    segs.append(rec)
    print(f"doan {label}: {rec['elapsed']:.1f} s, {rec['n']} quyet dinh{'' if 'pl_ok' not in rec else ', khop golden ' + str(rec['pl_ok'])}",
          flush=True)


th = threading.Thread(target=sampler, daemon=True)
th.start()
kern = {}
for rep in range(REPS):
    for T in (100, 250):
        lab[0] = "settle"
        pl = PL(T)
        kern.setdefault(T, []).append(pl.t_us)
        print(f"lap {rep} overlay T{T}: do tre kernel trung vi {pl.t_us:.2f} us", flush=True)
        time.sleep(30)
        conds = [("pl", lambda pl=pl: pl.run())] + [(m, (lambda m=m, T=T: run_ps(m, T))) for m in ("g31ps", "ens", "tt", "fb")]
        random.Random(1000 * rep + T).shuffle(conds)
        segment("idle")
        for m, fn in conds:
            segment(f"{m}{T}", fn)
            segment("idle")
        del pl
stop.set()
th.join()
names = list(rails) + ["ps_temp_C", "pl_temp_C"]
with open("power_std_samples.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t", "seg"] + names)
    w.writerows(samples)
ts = np.array([s[0] for s in samples])
P = np.array([s[2:] for s in samples], dtype=float)
for s in segs:
    m = (ts >= s["start"] + 2) & (ts <= s["end"] - 1)
    s["mean"] = {k: float(P[m, j].mean()) for j, k in enumerate(names)} if m.any() else {}
    s["n_samples"] = int(m.sum())
    s.pop("start"); s.pop("end")
for i, s in enumerate(segs):
    if s["label"] != "idle":
        a, b = segs[i - 1], segs[i + 1]
        assert a["label"] == "idle" and b["label"] == "idle"
        s["base"] = {k: (a["mean"][k] + b["mean"][k]) / 2 for k in rails}
        s["delta_W"] = {k: s["mean"][k] - s["base"][k] for k in rails}
        s["t_per_decision_ms"] = 1e3 * s["elapsed"] / s["n"]
        s["energy_per_decision_mJ"] = {k: s["delta_W"][k] * s["t_per_decision_ms"] for k in rails}
json.dump(dict(rails=list(rails), reps=REPS, dur=DUR, dt=DT, kernel_us=kern, segments=segs),
          open("power_std_segments.json", "w"), indent=1)
print("POWER_STD_DONE", flush=True)
