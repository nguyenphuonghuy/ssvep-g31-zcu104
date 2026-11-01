"""Chan doan kernel v2 tren board: nap fold 1, giai ma luot thu 0..2, in score/pred/status so voi golden."""
import json, sys
import numpy as np
from pynq import Overlay, allocate
T = int(sys.argv[1]) if len(sys.argv) > 1 else 100
REG = json.load(open("regs.json"))
ov = Overlay(f"g3rt_T{T}.bit")
ip = getattr(ov, f"g3_T{T}")
print("ip_dict keys:", list(ov.ip_dict), "phys", hex(ov.ip_dict[f"g3_T{T}"]["phys_addr"]))
print("register_map:", ip.register_map)
man = json.load(open(f"T{T}/manifest.json")); C, TT, K, PL = man["C"], man["S0"] + man["T"], man["K"], man["PRM_LEN"]
pb_, ib, sb, prb, stb = allocate((PL,), np.int32), allocate((C * TT,), np.int16), allocate((K,), np.int64), allocate((1,), np.int32), allocate((1,), np.int32)
def setaddr(name, buf):
    ip.write(REG[name], buf.device_address & 0xFFFFFFFF); ip.write(REG[name] + 4, buf.device_address >> 32)
for n, b in (("prm", pb_), ("in", ib), ("score", sb), ("pred", prb), ("status", stb)):
    setaddr(n, b)
    print(n, "addr", hex(b.device_address), "doc lai", hex(ip.read(REG[n])), hex(ip.read(REG[n] + 4)))
def call():
    ip.write(REG["ctrl"], 1)
    k = 0
    while not (ip.read(REG["ctrl"]) & 0x2):
        k += 1
    return k
pb_[:] = np.fromfile(f"T{T}/fold1_params.bin", dtype="<i4"); pb_.flush()
ip.write(REG["mode"], 0); print("mode doc lai", ip.read(REG["mode"])); n = call(); stb.invalidate(); print("nap: status", int(stb[0]), "poll", n)
ip.write(REG["mode"], 1); print("mode doc lai", ip.read(REG["mode"]))
x = np.fromfile(f"T{T}/fold1_input.bin", dtype="<i2").reshape(-1, C * TT)
es = np.fromfile(f"T{T}/fold1_score.bin", dtype="<i8").reshape(-1, K); ep = np.fromfile(f"T{T}/fold1_pred.bin", dtype="<i8")
for b in range(3):
    ib[:] = x[b]; ib.flush(); sb[:] = -7; sb.flush(); stb[:] = -9; stb.flush()
    n = call(); sb.invalidate(); prb.invalidate(); stb.invalidate()
    print(f"luot {b}: poll={n} status={int(stb[0])} pred={int(prb[0])} (golden {ep[b]}) score[:6]={np.array(sb[:6])} golden[:6]={es[b, :6]}")
