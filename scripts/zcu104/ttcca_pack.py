"""Xuat goi cho ttcca.cpp: tt-CCA + bank loc + tru TB (ttcca_fb5cd) va FBCCA fb5cd tren Benchmark fold 1.

Chuan float64 xuyen suot (khong qua cache float32 cua prepare_causal): tru TB [0, onset) -> lfilter(b, a) 5 dai tren [0, s0 + N)
-> cat cua so; template, w_bar, diem tinh bang ham cua baselines_tf.py (cca, ttcca_scores, BETA_W, sincos).
Cong 0 (exit 3): acc fold 1 cua chuan trung results/baselines_dc_2026_10_28 (ttcca_fb5cd, FBCCA fb5cd) o ca hai cua so.
Dau ra <out_dir>/T<N>/: meta.json, raw.f64 (M, C, s0+N), filt.f64, xbar.f64 (NB, K, C, N), wxbar.f64 (NB, K, C),
  refs.f64 (K, 2NH, N), fbw.f64 (NB), ref_xfilt.f64 (M, NB, C, N), ref_tt.f64 (M, K), ref_fb.f64 (M, K), label.i32 (M).
Dung: python ttcca_pack.py --out_dir <dir>
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import baselines_tf as B  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(HERE))
BL = os.path.join(PROJ, "results", "baselines_dc_2026_10_28")
DS, FOLD, NB, NH = "benchmark", 1, 5, 5


def prep64(X, meta, win, filt):
    """X (Nb, K, C, Ttot) tho -> (Nb, K, NB, C, N) float64, nhu prepare_causal(..., 'fb5cd') nhung khong ep float32."""
    from scipy.signal import lfilter
    fs, on = meta["srate"], meta["onset_in_array"]
    s0, n = on + int(round(meta["latency_s"] * fs)), int(round(win * fs))
    seg = X[..., :s0 + n].astype(np.float64)
    seg = seg - seg[..., :on].mean(axis=-1, keepdims=True)
    return np.stack([lfilter(b, a, seg, axis=-1)[..., s0:s0 + n] for b, a in filt], axis=-3), s0, n


def ref_acc(path, win, tag=None):
    for r in csv.DictReader(open(path)):
        if int(r["subject"]) == FOLD and round(float(r["window"]), 2) == win and (tag is None or r["tag"] == tag):
            return float(r["acc"])
    raise SystemExit(f"khong co fold {FOLD} w={win} trong {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    from ssvep import cca as C_
    from ssvep.data import load_cfg, load_subject_pre
    nsub = load_cfg(DS)["n_subjects"]
    raw = [load_subject_pre(DS, s) for s in range(1, nsub + 1)]
    meta = raw[0][1]
    fs, on = meta["srate"], meta["onset_in_array"]
    filt = C_.subband_filters(fs, NB)
    fbw = np.array([B.BETA_W(j) for j in range(NB)])
    ok = True
    for win in (0.4, 1.0):
        Xs = []
        for X, _ in raw:
            Xp, s0, N = prep64(X, meta, win, filt)
            Xs.append(Xp)
        X = np.stack(Xs)                                                           # (S, Nb, K, NB, C, N)
        S, Nb, K = X.shape[:3]
        Cc = X.shape[4]
        t, src = FOLD - 1, [s for s in range(S) if s != FOLD - 1]
        refs = B.sincos(meta["freqs"], fs, N, NH)
        xbar = np.stack([X[src, :, :, j].mean(axis=(0, 1)) for j in range(NB)])        # (NB, K, C, N)
        wxbar = np.stack([np.stack([B.cca(xbar[j, k], refs[k])[1] for k in range(K)]) for j in range(NB)])
        M = Nb * K
        xt = X[t].reshape(M, NB, Cc, N)
        tt, fb = np.zeros((M, K)), np.zeros((M, K))
        for i in range(M):
            for j in range(NB):
                tt[i] += fbw[j] * B.ttcca_scores(xt[i, j], xbar[j], refs, wxbar[j])
                fb[i] += fbw[j] * np.array([B.cca(xt[i, j], refs[k])[0] ** 2 for k in range(K)])
        lab = np.tile(np.arange(K), Nb).astype(np.int32)
        acc_tt, acc_fb = float((tt.argmax(1) == lab).mean()), float((fb.argmax(1) == lab).mean())
        w_tt = ref_acc(os.path.join(BL, "ttcca_fb5cd_benchmark_per_subject.csv"), win)
        w_fb = ref_acc(os.path.join(BL, "fbcca_benchmark_fb5cd_per_subject.csv"), win, "fb5cd")
        g0 = abs(acc_tt - w_tt) <= 5e-7 and abs(acc_fb - w_fb) <= 5e-7          # tep luu 6 chu so thap phan
        ok &= g0
        print(f"w={win}: chuan float64 fold {FOLD}: tt-CCA {acc_tt:.6f} (tep {w_tt:.6f}), FBCCA {acc_fb:.6f} (tep {w_fb:.6f}) "
              f"-> cong 0 {'DAT' if g0 else 'KHONG DAT'}", flush=True)
        d = os.path.join(a.out_dir, f"T{N}")
        os.makedirs(d, exist_ok=True)
        rt = raw[t][0].reshape(M, Cc, -1)[..., :s0 + N].astype(np.float64)
        fl = [(len(b), len(aa)) for b, aa in filt]
        json.dump(dict(dataset=DS, fold=FOLD, window=win, C=Cc, K=K, NB=NB, NH=NH, N=N, T0=s0 + N, onset=on, s0=s0, M=M,
                       filt_len=fl, acc_tt=acc_tt, acc_fb=acc_fb), open(os.path.join(d, "meta.json"), "w"), indent=1)
        for name, arr, dt in (("raw.f64", rt, "<f8"), ("filt.f64", np.concatenate([np.r_[b, aa] for b, aa in filt]), "<f8"),
                              ("xbar.f64", xbar, "<f8"), ("wxbar.f64", wxbar, "<f8"), ("refs.f64", np.stack(refs), "<f8"),
                              ("fbw.f64", fbw, "<f8"), ("ref_xfilt.f64", xt, "<f8"), ("ref_tt.f64", tt, "<f8"),
                              ("ref_fb.f64", fb, "<f8"), ("label.i32", lab, "<i4")):
            np.ascontiguousarray(arr, dtype=dt).tofile(os.path.join(d, name))
        print(f"  -> {d}: M={M} C={Cc} K={K} N={N} s0={s0} loc {fl}", flush=True)
    print("CONG 0: DAT" if ok else "CONG 0: KHONG DAT")
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
