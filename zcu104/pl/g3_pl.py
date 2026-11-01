"""Chay bo giai ma G3 tren PL ZCU104 (overlay g3.bit/g3.hwh). Chay bang Python cua PYNQ (root).
1. Khop tung bit: 8 trial moi cua so (T100, T250), so score/pred voi expected_* (golden).
2. Do tre (mo ta): moi lan goi = chep trial vao bo dem + flush + start + cho done + invalidate + doc score/pred.
   Ghi rieng thoi gian start->done (chi kernel + polling). N_RUNS luot x R lan, nghi COOLDOWN s.
Dung: python g3_pl.py [N_RUNS=7] [R=2000] [COOLDOWN=10] [OVERLAY=g3_T100.bit]
"""
import json
import sys
import time

import numpy as np
from pynq import Overlay, allocate, Clocks

N_RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 7
R = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
CD = float(sys.argv[3]) if len(sys.argv) > 3 else 10
OV = sys.argv[4] if len(sys.argv) > 4 else "g3_T100.bit"
K = 40
REG = dict(ctrl=0x00, in_lo=0x10, in_hi=0x14, score_lo=0x1C, score_hi=0x20, pred_lo=0x28, pred_hi=0x2C)  # csynth.rpt

ov = Overlay(OV)
TS = [T for T in (100, 250) if f"g3_T{T}" in ov.ip_dict]
print(f"overlay nap xong; fclk0 = {Clocks.fclk0_mhz:.3f} MHz; IP: {[k for k in ov.ip_dict]}", flush=True)
ok_all = True
out = {"overlay": OV, "fclk0_mhz": Clocks.fclk0_mhz, "windows": TS, "runs": {}}
for T in TS:
    man = json.load(open(f"T{T}/manifest.json"))
    B, C, TT = man["B"], man["C"], man["Ttot"]
    x = np.fromfile(f"T{T}/input.bin", dtype="<i2").reshape(B, C * TT)
    es = np.fromfile(f"T{T}/expected_score.bin", dtype="<i8").reshape(B, K)
    ep = np.fromfile(f"T{T}/expected_pred.bin", dtype="<i8")
    ip = getattr(ov, f"g3_T{T}")
    ib, sb, pb = allocate((C * TT,), dtype=np.int16), allocate((K,), dtype=np.int64), allocate((1,), dtype=np.int32)
    for name, buf in (("in", ib), ("score", sb), ("pred", pb)):
        a = buf.device_address
        ip.write(REG[name + "_lo"], a & 0xFFFFFFFF)
        ip.write(REG[name + "_hi"], a >> 32)

    def run(b):
        ib[:] = x[b]
        ib.flush()
        t0 = time.perf_counter_ns()
        ip.write(REG["ctrl"], 1)
        while not (ip.read(REG["ctrl"]) & 0x2):
            pass
        t1 = time.perf_counter_ns()
        sb.invalidate()
        pb.invalidate()
        return np.array(sb), int(pb[0]), t1 - t0

    sai_s = sai_p = 0
    for b in range(B):
        s, p, _ = run(b)
        sai_s += int((s != es[b]).sum())
        sai_p += int(p != ep[b])
    ok = sai_s == 0 and sai_p == 0
    ok_all &= ok
    print(f"T{T}: sai_score={sai_s} sai_pred={sai_p} -> KHOP TUNG BIT: {'DAT' if ok else 'KHONG DAT'}", flush=True)
    if not ok:
        continue
    pred_ref = [run(b)[1] for b in range(B)]
    out["runs"][T] = []
    time.sleep(CD)
    for i in range(1, N_RUNS + 1):
        e2e, kern, lech = [], [], 0
        for r in range(R):
            b = r % B
            t0 = time.perf_counter_ns()
            s, p, dk = run(b)
            e2e.append((time.perf_counter_ns() - t0) / 1e3)
            kern.append(dk / 1e3)
            lech += p != pred_ref[b]
        e2e, kern = np.array(e2e), np.array(kern)
        np.save(f"T{T}/pl_e2e_us_run{i}.npy", e2e)
        np.save(f"T{T}/pl_kernel_us_run{i}.npy", kern)
        load = open("/proc/loadavg").read().strip()
        row = dict(run=i, e2e_median_us=float(np.median(e2e)), e2e_p95_us=float(np.percentile(e2e, 95)),
                   e2e_max_us=float(e2e.max()), kern_median_us=float(np.median(kern)),
                   kern_p95_us=float(np.percentile(kern, 95)), kern_max_us=float(kern.max()), pred_lech=int(lech), loadavg=load)
        out["runs"][T].append(row)
        print(f"BENCH T={T} " + " ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        if lech:
            ok_all = False
        time.sleep(CD)
    for buf in (ib, sb, pb):
        buf.freebuffer()
json.dump(out, open("g3_pl_result.json", "w"), indent=1)
print("G3_PL_DONE" if ok_all else "G3_PL_KHONG_DAT", flush=True)
