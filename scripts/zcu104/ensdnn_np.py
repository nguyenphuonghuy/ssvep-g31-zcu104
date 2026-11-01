"""Ensemble-DNN — ban NumPy (C, mo ta). Cung du lieu/che do voi ensdnn.cpp.
Tich chap viet thanh nhan ma tran (BLAS cua NumPy); loc bang scipy.signal.sosfiltfilt; ep 1 luong bang bien moi truong
(OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 dat TRUOC khi chay python).
Dung: python ensdnn_np.py <dir T> check | bench <R>
"""
import json
import os
import sys
import time

import numpy as np
from scipy.signal import sosfiltfilt

NCH, KW, PADL = 120, 10, 4


def load(d):
    m = json.load(open(os.path.join(d, "manifest.json")))
    N, C, K, T, M, R = m["N"], m["C"], m["K"], m["T"], m["M"], 2 * m["NH"]
    T2 = T // 2
    W = np.fromfile(os.path.join(d, "weights.f32"), "<f4").reshape(N, -1)
    sizes = [3, NCH * C, NCH, NCH * NCH * 2, NCH, NCH * NCH * KW, NCH, K * NCH * T2, K]
    assert sum(sizes) == W.shape[1]
    parts = np.split(W, np.cumsum(sizes)[:-1], axis=1)
    P = dict(sub=parts[0], chw=parts[1].reshape(N, NCH, C), chb=parts[2], dw=parts[3].reshape(N, NCH, NCH, 2), db=parts[4],
             cw=parts[5].reshape(N, NCH, NCH, KW), cb=parts[6], fw=parts[7].reshape(N, K, NCH * T2), fb=parts[8])
    sos = np.fromfile(os.path.join(d, "sos.f64"), "<f8").reshape(3, 1, 6)
    Q = np.fromfile(os.path.join(d, "Q.f64"), "<f8").reshape(K, T, R)
    tm = np.fromfile(os.path.join(d, "tmpl.f32"), "<f4").reshape(N, K, 3, C, T).astype(np.float64)
    # tinh truoc tu template (khong tinh vao do tre)
    ws = P["sub"].astype(np.float64)                                             # N, 3
    Wd = P["chw"].astype(np.float64)                                             # N, 120, C
    tk = np.einsum("nb,nkbct->nkct", ws, tm)
    b = np.einsum("njc,nkct->nkjt", Wd, tk)
    b -= b.mean(-1, keepdims=True)
    P.update(ws=ws, Wd=Wd, bc=b, bb=(b * b).sum(-1), sos=sos, Q=Q, m=m)
    return P


def decide(P, raw, times=None):
    m = P["m"]
    N, C, K, T = m["N"], m["C"], m["K"], m["T"]
    T2 = T // 2
    t0 = time.perf_counter()
    xf = np.stack([sosfiltfilt(P["sos"][b], raw, axis=-1, padlen=m["padlen"]) for b in range(3)]).astype(np.float32)  # 3,C,T
    t1 = time.perf_counter()
    s = np.einsum("nb,bct->nct", P["sub"], xf)                                   # N, C, T
    h1 = np.matmul(P["chw"], s) + P["chb"][:, :, None]                           # N, 120, T
    h1r = h1[:, :, :2 * T2].reshape(N, NCH, T2, 2)
    h2 = np.einsum("noik,niuk->nou", P["dw"], h1r, optimize=True) + P["db"][:, :, None]
    np.maximum(h2, 0, out=h2)
    hp = np.zeros((N, NCH, T2 + KW - 1), np.float32)
    hp[:, :, PADL:PADL + T2] = h2
    cols = np.stack([hp[:, :, k:k + T2] for k in range(KW)], axis=-1)            # N, 120(in), T2, KW
    h3 = np.einsum("noik,niuk->nou", P["cw"], cols, optimize=True) + P["cb"][:, :, None]
    logits = np.einsum("nkf,nf->nk", P["fw"], h3.reshape(N, -1)) + P["fb"]
    preds = logits.argmax(-1)
    t2 = time.perf_counter()
    xs = np.einsum("nb,bct->nct", P["ws"], xf.astype(np.float64))
    a = np.matmul(P["Wd"], xs)                                                   # N, 120, T
    a -= a.mean(-1, keepdims=True)
    ar = np.arange(N)
    bsel = P["bc"][ar, preds]                                                    # N, 120, T
    saa = (a * a).sum(-1)
    corr2 = (a * bsel).sum(-1) ** 2 / (saa * P["bb"][ar, preds])
    cca2 = (np.matmul(a, P["Q"][preds]) ** 2).sum(-1) / saa
    sc = corr2 + cca2
    sc[np.isnan(sc)] = -np.inf
    sims = sc.max(-1)
    order = np.argsort(-sims, kind="stable")
    oh = np.zeros((N, K))
    oh[np.arange(N), preds[order]] = sims[order]
    V = np.cumsum(oh, axis=0)
    top = -np.sort(-V, axis=1)
    k = int(np.argmax(top[:, 0] - top[:, 1]))
    out = int(np.argmax(V[k]))
    t3 = time.perf_counter()
    if times is not None:
        times[0] += t1 - t0
        times[1] += t2 - t1
        times[2] += t3 - t2
    return out, k + 1, xf, logits, preds, sims


def main():
    d, mode = sys.argv[1], sys.argv[2]
    P = load(d)
    m = P["m"]
    N, C, K, T, M = m["N"], m["C"], m["K"], m["T"], m["M"]
    raw = np.fromfile(os.path.join(d, "raw.f64"), "<f8").reshape(M, C, T)
    print(f"numpy {np.__version__}; OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')} "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}", flush=True)
    if mode == "check":
        rx = np.fromfile(os.path.join(d, "ref_xfilt.f32"), "<f4").reshape(M, 3, C, T)
        rl = np.fromfile(os.path.join(d, "ref_logits_cpu.f32"), "<f4").reshape(N, M, K)      # cong 2: chuan CPU
        rp = np.fromfile(os.path.join(d, "ref_preds_cpu.i32"), "<i4").reshape(N, M)
        rf = np.fromfile(os.path.join(d, "ref_pred.i32"), "<i4")
        lab = np.fromfile(os.path.join(d, "label.i32"), "<i4")
        fmax = lmax = 0.0
        pm = fs = cor = 0
        for i in range(M):
            out, _, xf, lg, pr, _ = decide(P, raw[i])
            fmax = max(fmax, float(np.abs(xf - rx[i]).max()))
            lmax = max(lmax, float(np.abs(lg - rl[:, i]).max()))
            pm += int((pr == rp[:, i]).sum())
            fs += int(out == rf[i])
            cor += int(out == lab[i])
        frel = fmax / float(np.abs(rx).max())
        g1, g2, g3 = frel <= 1e-4, lmax <= 1e-3 and pm >= np.ceil(0.999 * N * M), fs == M
        print(f"T={T} NUMPY CONG 1 loc {frel:.3g}: {'DAT' if g1 else 'KHONG DAT'}; CONG 2 logit (chuan CPU) {lmax:.3g}, du doan tung mang "
              f"{pm}/{N * M}: {'DAT' if g2 else 'KHONG DAT'}; CONG 3 du doan cuoi {fs}/{M}: {'DAT' if g3 else 'KHONG DAT'}; "
              f"acc {cor / M:.4f}", flush=True)
        sys.exit(0 if (g1 and g2 and g3) else 3)
    if mode == "bench":
        R = int(sys.argv[3]) if len(sys.argv) > 3 else 200
        for i in range(8):
            decide(P, raw[i])
        us, tt = [], [0.0, 0.0, 0.0]
        for r in range(R):
            t = time.perf_counter()
            decide(P, raw[r % 8], tt)
            us.append((time.perf_counter() - t) * 1e6)
        u = np.sort(us)
        print(f"BENCH_NP T={T} N={N} R={R} median_us={np.median(u):.1f} p95_us={np.percentile(u, 95):.1f} min_us={u[0]:.1f} "
              f"max_us={u[-1]:.1f} mean_filt_us={1e6 * tt[0] / R:.1f} mean_dnn_us={1e6 * tt[1] / R:.1f} "
              f"mean_sim_us={1e6 * tt[2] / R:.1f}", flush=True)


if __name__ == "__main__":
    main()
