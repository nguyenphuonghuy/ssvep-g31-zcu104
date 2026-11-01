"""Kernel G3 v2 tren PL ZCU104. Chay bang Python PYNQ (root) trong ~/ssvep_lite_pkg/pl_rt.
1. Khop tung bit: moi cua so co trong overlay, 35 fold x 240 luot thu: nap tham so (mode 0, kiem status + 5 checksum
   doc nguoc cua kernel v2.1), giai ma
   (mode 1), so score/pred voi golden; accuracy theo fold.
2. Do tre (fold 1): (a) tron chu trinh tu Python nhu muc 13; (b') start -> done bang driver C (libg3drv.so),
   xoay vong 8 bo dem dau vao. N_RUNS luot x R lan, nghi COOLDOWN s.
Dung: python g3_pl_rt.py [N_RUNS=7] [R=2000] [COOLDOWN=10] [OVERLAY=g3rt_T100.bit]  -> g3_pl_rt_result_<overlay>.json
Thanh ghi: regs.json (sinh tu header HLS xg3_decode_rt_hw.h).
"""
import ctypes
import json
import sys
import time

import numpy as np
from pynq import Clocks, Overlay, allocate

N_RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 7
R = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
CD = float(sys.argv[3]) if len(sys.argv) > 3 else 10
OV = sys.argv[4] if len(sys.argv) > 4 else "g3rt_T100.bit"
REG = json.load(open("regs.json"))
drv = ctypes.CDLL("./libg3drv.so")
drv.g3_bench.argtypes = [ctypes.c_uint64, ctypes.c_uint32, ctypes.c_int, ctypes.c_int,
                         ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_double)]
drv.g3_bench.restype = ctypes.c_int

ov = Overlay(OV)
TS = [T for T in (100, 250) if f"g3_T{T}" in ov.ip_dict]
print(f"overlay {OV}; fclk0 = {Clocks.fclk0_mhz:.3f} MHz; IP G3: {TS}", flush=True)
out = {"overlay": OV, "fclk0_mhz": Clocks.fclk0_mhz, "windows": TS, "folds": {}, "runs": {}}
ok_all = True


def setaddr(ip, name, buf):
    ip.write(REG[name], buf.device_address & 0xFFFFFFFF)
    ip.write(REG[name] + 4, buf.device_address >> 32)


def expected_cs(blob, man):
    """5 checksum doc nguoc (kernel v2.1): doan [0, OFF_W8), W8, A8, P8, R8; cs = sum v * ((i % 1021) + 1)."""
    NB, K, F, C, J, D, T = (man[k] for k in ("NB", "K", "F", "C", "J", "D", "T"))
    b0 = 9 + NB + NB * 7 * 6
    b = [0, b0, b0 + F * C, b0 + F * C + D, b0 + F * C + D + NB * K * F * T, len(blob)]
    v = blob.astype(np.int64)
    v8 = ((v + 128) % 256 - 128)
    w = (np.arange(len(v)) % 1021) + 1
    return [int(((v if g == 0 else v8)[b[g]:b[g + 1]] * w[b[g]:b[g + 1]]).sum()) for g in range(5)]


def call(ip):
    ip.write(REG["ctrl"], 1)
    while not (ip.read(REG["ctrl"]) & 0x2):
        pass


for T in TS:
    man = json.load(open(f"T{T}/manifest.json"))
    C, TT, K, PL = man["C"], man["S0"] + man["T"], man["K"], man["PRM_LEN"]
    ip = getattr(ov, f"g3_T{T}")
    phys = ov.ip_dict[f"g3_T{T}"]["phys_addr"]
    pb_, ib, sb, prb, stb = (allocate((PL,), dtype=np.int32), allocate((C * TT,), dtype=np.int16),
                             allocate((K,), dtype=np.int64), allocate((1,), dtype=np.int32), allocate((1,), dtype=np.int32))
    for name, buf in (("prm", pb_), ("in", ib), ("score", sb), ("pred", prb), ("status", stb)):
        setaddr(ip, name, buf)

    def load(fold):
        blob = np.fromfile(f"T{T}/fold{fold}_params.bin", dtype="<i4")
        pb_[:] = blob
        pb_.flush()
        ip.write(REG["mode"], 0)
        call(ip)
        stb.invalidate()
        sb.invalidate()
        ip.write(REG["mode"], 1)
        hw, exp = [int(v) for v in np.array(sb[:5])], expected_cs(blob, man)
        if hw != exp:
            print(f"T{T} fold {fold}: checksum nap SAI hw={hw} ky vong={exp}", flush=True)
            return -2
        return int(stb[0])

    # ---------- 1. khop tung bit tren moi fold ----------
    rows, t0 = [], time.time()
    NF = man.get("NFOLD", 35)                                  # BETA 70 fold
    for fold in range(1, NF + 1):
        st = load(fold)
        x = np.fromfile(f"T{T}/fold{fold}_input.bin", dtype="<i2").reshape(-1, C * TT)
        es = np.fromfile(f"T{T}/fold{fold}_score.bin", dtype="<i8").reshape(-1, K)
        ep = np.fromfile(f"T{T}/fold{fold}_pred.bin", dtype="<i8")
        lab = np.fromfile(f"T{T}/fold{fold}_label.bin", dtype="<i8")
        ss = sp = dung = 0
        for b in range(len(x)):
            ib[:] = x[b]
            ib.flush()
            call(ip)
            sb.invalidate()
            prb.invalidate()
            ss += int((np.array(sb) != es[b]).sum())
            sp += int(prb[0] != ep[b])
            dung += int(prb[0] == lab[b])
        rows.append(dict(fold=fold, status=st, sai_score=ss, sai_pred=sp, acc=dung / len(x), n=len(x)))
        if st != 1 or ss or sp:
            ok_all = False
    sai = sum(r["sai_score"] + r["sai_pred"] for r in rows) + sum(r["status"] != 1 for r in rows)
    print(f"T{T}: {NF} fold x {rows[0]['n']} luot thu; loi nap={sum(r['status'] != 1 for r in rows)} sai_score="
          f"{sum(r['sai_score'] for r in rows)} sai_pred={sum(r['sai_pred'] for r in rows)} -> KHOP TUNG BIT: "
          f"{'DAT' if sai == 0 else 'KHONG DAT'}; acc TB theo fold {100 * np.mean([r['acc'] for r in rows]):.2f}% "
          f"({time.time() - t0:.0f} s)", flush=True)
    out["folds"][T] = rows
    if sai:
        continue

    # ---------- 2. do tre (fold 1) ----------
    load(1)
    x = np.fromfile(f"T{T}/fold1_input.bin", dtype="<i2").reshape(-1, C * TT)[:8]
    ins = [allocate((C * TT,), dtype=np.int16) for _ in range(8)]
    for k in range(8):
        ins[k][:] = x[k]
        ins[k].flush()
    addrs = (ctypes.c_uint64 * 8)(*[b.device_address for b in ins])
    us = (ctypes.c_double * R)()
    out["runs"][T] = []
    time.sleep(CD)
    for i in range(1, N_RUNS + 1):
        e2e = []
        setaddr(ip, "in", ib)
        for r in range(R):
            t = time.perf_counter_ns()
            ib[:] = x[r % 8]
            ib.flush()
            call(ip)
            sb.invalidate()
            prb.invalidate()
            _ = int(prb[0])
            e2e.append((time.perf_counter_ns() - t) / 1e3)
        rc = drv.g3_bench(phys, REG["in"], R, 8, addrs, us)
        if rc != 0:
            raise SystemExit(f"g3_bench loi rc={rc}")
        setaddr(ip, "in", ib)
        kc = np.ctypeslib.as_array(us).copy()
        e2e = np.array(e2e)
        np.save(f"T{T}/rt_e2e_us_run{i}.npy", e2e)
        np.save(f"T{T}/rt_cker_us_run{i}.npy", kc)
        row = dict(run=i, e2e_median_us=float(np.median(e2e)), e2e_p95_us=float(np.percentile(e2e, 95)),
                   e2e_max_us=float(e2e.max()), c_median_us=float(np.median(kc)), c_p95_us=float(np.percentile(kc, 95)),
                   c_min_us=float(kc.min()), c_max_us=float(kc.max()), loadavg=open("/proc/loadavg").read().strip())
        out["runs"][T].append(row)
        print(f"BENCH T={T} " + " ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
              flush=True)
        time.sleep(CD)
    # kiem lai sau do: giai ma 8 luot thu qua bo dem xoay vong van khop golden
    es = np.fromfile(f"T{T}/fold1_score.bin", dtype="<i8").reshape(-1, K)
    for k in range(8):
        setaddr(ip, "in", ins[k])
        call(ip)
        sb.invalidate()
        if (np.array(sb) != es[k]).any():
            ok_all = False
            print(f"T{T}: bo dem xoay vong {k} KHONG khop golden", flush=True)
    setaddr(ip, "in", ib)
STEM = OV.rsplit(".", 1)[0]
json.dump(out, open(f"g3_pl_rt_result_{STEM}.json", "w"), indent=1)
print("G3_PL_RT_DONE" if ok_all else "G3_PL_RT_KHONG_DAT", flush=True)
