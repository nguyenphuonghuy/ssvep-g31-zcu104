"""Chan doan 3: doc phi tren phan cung bang cach dat A8 = e_d, BINT = 0 (score_k = phi[k][d]); so voi golden_model.
Dung tren board: python dbg_rt3.py <T>   (can golden_model.py cung thu muc)"""
import json, sys
import numpy as np
from pynq import Overlay, allocate
import golden_model as G
T = int(sys.argv[1]) if len(sys.argv) > 1 else 100
REG = json.load(open("regs.json"))
man = json.load(open(f"T{T}/manifest.json"))
NB, K, F, C, J, D, S0 = (man[k] for k in ("NB", "K", "F", "C", "J", "D", "S0"))
TT, PL, MAXSEC = S0 + T, man["PRM_LEN"], 7
blob = np.fromfile(f"T{T}/fold1_params.bin", dtype="<i4").astype(np.int64)

def unpack(v):
    p = 0
    def take(n):
        nonlocal p
        r = v[p:p + n]; p += n; return r
    magic, tt, mx, shx, ktm, shtm, ke, she, bint = take(9)
    nsec = take(NB)
    sb = take(NB * MAXSEC * 3).reshape(NB, MAXSEC, 3); sa = take(NB * MAXSEC * 2).reshape(NB, MAXSEC, 2)
    sf = take(NB * MAXSEC).reshape(NB, MAXSEC)
    w8 = take(F * C).reshape(F, C); a8 = take(D); p8 = take(NB * K * F * T).reshape(NB, K, F, T); r8 = take(K * J * T).reshape(K, J, T)
    assert p == len(v)
    P = dict(T=T, a_frac=15, mx=int(mx), shx=int(shx), ktm=int(ktm), shtm=int(shtm), ke=int(ke), she=int(she), bint=int(bint),
             w8=w8, a8=a8, p8=p8, r8=r8,
             sos_b=[[tuple(int(x) for x in sb[n, s]) for s in range(nsec[n])] for n in range(NB)],
             sos_a=[[tuple(int(x) for x in sa[n, s]) for s in range(nsec[n])] for n in range(NB)],
             sos_fb=[[int(sf[n, s]) for s in range(nsec[n])] for n in range(NB)])
    return P
P = unpack(blob)
x = np.fromfile(f"T{T}/fold1_input.bin", dtype="<i2").reshape(-1, C, TT)[:1].astype(np.int64)
es = np.fromfile(f"T{T}/fold1_score.bin", dtype="<i8").reshape(-1, K)
pg, sg, inter = G.decide(x, P, return_all=True)
print("golden tren board tu blob: score khop file =", bool((sg[0] == es[0]).all()), flush=True)
phi = inter["phi"][0]                                       # (K, D)
A8_OFF = 9 + NB + NB * MAXSEC * 6 + F * C                  # vi tri A8 trong blob
ov = Overlay(f"g3rt_T{T}.bit")
ip = getattr(ov, f"g3_T{T}")
pb_, ib, sb_, prb, stb = allocate((PL,), np.int32), allocate((C * TT,), np.int16), allocate((K,), np.int64), allocate((1,), np.int32), allocate((1,), np.int32)
for n, b in (("prm", pb_), ("in", ib), ("score", sb_), ("pred", prb), ("status", stb)):
    ip.write(REG[n], b.device_address & 0xFFFFFFFF); ip.write(REG[n] + 4, b.device_address >> 32)
def call():
    ip.write(REG["ctrl"], 1)
    while not (ip.read(REG["ctrl"]) & 0x2):
        pass
ib[:] = x[0].reshape(-1); ib.flush()
NF = NB * F
ds = list(range(0, NF)) + [NF + i * 5 for i in range(NF)]     # 20 dac trung template + h=0 cua 20 (n,f) sin/cos
bad = {}
for d in ds:
    v = blob.copy(); v[8] = 0; v[A8_OFF:A8_OFF + D] = 0; v[A8_OFF + d] = 1
    pb_[:] = v.astype(np.int32); pb_.flush()
    ip.write(REG["mode"], 0); call(); ip.write(REG["mode"], 1); call(); sb_.invalidate()
    hw = np.array(sb_)
    bad[d] = int((hw != phi[:, d]).sum())
    if d in (0, 1, NF, NF + 5, NF + 95):
        print(f"d={d}: hw[:6]={hw[:6]} golden[:6]={phi[:6, d]}", flush=True)
print("so lop sai theo d (template 0..19):", [bad[d] for d in ds[:NF]])
print("so lop sai theo d (sin/cos h=0):", [bad[d] for d in ds[NF:]])
