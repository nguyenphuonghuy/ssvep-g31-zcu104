"""Chan doan 2: (A) prm bo dem cacheable=False; (B) nap fold 2 roi giai ma luot 0 cua fold 1; (C) khong nap."""
import json, sys
import numpy as np
from pynq import Overlay, allocate
T = int(sys.argv[1]) if len(sys.argv) > 1 else 100
REG = json.load(open("regs.json"))
man = json.load(open(f"T{T}/manifest.json")); C, TT, K, PL = man["C"], man["S0"] + man["T"], man["K"], man["PRM_LEN"]
x = np.fromfile(f"T{T}/fold1_input.bin", dtype="<i2").reshape(-1, C * TT)
es = np.fromfile(f"T{T}/fold1_score.bin", dtype="<i8").reshape(-1, K)

def run(tag, fold_prm, cacheable_prm=True, load=True):
    ov = Overlay(f"g3rt_T{T}.bit")
    ip = getattr(ov, f"g3_T{T}")
    kw = {} if cacheable_prm else {"cacheable": False}
    pb_ = allocate((PL,), np.int32, **kw)
    ib, sb, prb, stb = allocate((C * TT,), np.int16), allocate((K,), np.int64), allocate((1,), np.int32), allocate((1,), np.int32)
    for n, b in (("prm", pb_), ("in", ib), ("score", sb), ("pred", prb), ("status", stb)):
        ip.write(REG[n], b.device_address & 0xFFFFFFFF); ip.write(REG[n] + 4, b.device_address >> 32)
    def call():
        ip.write(REG["ctrl"], 1)
        while not (ip.read(REG["ctrl"]) & 0x2):
            pass
    if load:
        pb_[:] = np.fromfile(f"T{T}/fold{fold_prm}_params.bin", dtype="<i4"); pb_.flush()
        ip.write(REG["mode"], 0); call(); stb.invalidate()
    ip.write(REG["mode"], 1)
    ib[:] = x[0]; ib.flush(); call(); sb.invalidate(); prb.invalidate()
    print(f"{tag}: cacheable_prm={cacheable_prm} load={load} fold_prm={fold_prm} status_nap={int(stb[0]) if load else None} "
          f"pred={int(prb[0])} score[:4]={np.array(sb[:4])} golden(fold1)[:4]={es[0, :4]}", flush=True)
    del ov
run("A0 (nhu cu)", 1)
run("A1 khong cache", 1, cacheable_prm=False)
run("B nap fold 2", 2)
run("C khong nap", 1, load=False)
