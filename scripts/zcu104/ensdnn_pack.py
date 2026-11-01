"""Dong goi du lieu Ensemble-DNN cho ban C++/NumPy tren PS ZCU104. Chay trong job (can numpy).

Vao: results/ensdnn_export_2026_10_21/{ens,filt}_benchmark_w<win>_*_f1.npz (job 17473).
Ra:  <out>/T<T>/ gom file nhi phan little-endian + manifest.json:
  weights.f32   : N mo hinh noi tiep, moi mo hinh [sub(3), ch_w(120*C), ch_b(120), down_w(120*120*2), down_b(120),
                  conv_w(120*120*10), conv_b(120), fc_w(K*120*T2), fc_b(K)] (thu tu chi so nhu PyTorch)
  tmpl.f32      : (N, K, 3, C, T) template TB moi luot thu cua nguoi nguon
  Q.f64         : (K, T, 2NH) co so truc chuan cua tham chieu sin/cos da tru TB
  raw.f64       : (M, C, T) luot thu tho (chua loc) cua nguoi kiem tra
  sos.f64 (3, 6), zi.f64 (3, 2) : bo loc 3 dai (1 khau bac hai moi dai), padlen trong manifest
  ref_xfilt.f32 (M, 3, C, T), ref_logits.f32 (N, M, K), ref_preds.i32 (N, M), ref_sims.f64 (N, M),
  ref_kstar.i32 (M), ref_pred.i32 (M), label.i32 (M)  : chuan PyTorch (job 17473)
Dung: python ensdnn_pack.py --exp_dir <dir> --out <dir>
"""
import argparse
import json
import os

import numpy as np

NAMES = ["sub.weight", "ch.weight", "ch.bias", "down.weight", "down.bias", "conv.weight", "conv.bias", "fc.weight", "fc.bias"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for win in ("0.40", "1.00"):
        E = np.load(os.path.join(a.exp_dir, f"ens_benchmark_w{win}_s0_f1.npz"))
        Fz = np.load(os.path.join(a.exp_dir, f"filt_benchmark_w{win}_f1.npz"))
        W = {k: E[f"w_{k}"] for k in NAMES}
        N = W["sub.weight"].shape[0]
        C = W["ch.weight"].shape[3]
        K = W["fc.bias"].shape[1]
        raw = Fz["raw"]
        M, _, T = raw.shape
        T2 = T // 2
        assert W["fc.weight"].shape == (N, K, 120 * T2), W["fc.weight"].shape
        d = os.path.join(a.out, f"T{T}")
        os.makedirs(d, exist_ok=True)
        per = np.concatenate([np.concatenate([W[k][n].reshape(-1) for k in NAMES]) for n in range(N)]).astype("<f4")
        per.tofile(os.path.join(d, "weights.f32"))
        E["tmpl"].astype("<f4").tofile(os.path.join(d, "tmpl.f32"))
        E["Q"].astype("<f8").tofile(os.path.join(d, "Q.f64"))
        raw.astype("<f8").tofile(os.path.join(d, "raw.f64"))
        Fz["sos"].reshape(3, 6).astype("<f8").tofile(os.path.join(d, "sos.f64"))
        Fz["zi"].reshape(3, 2).astype("<f8").tofile(os.path.join(d, "zi.f64"))
        E["x_filt"].astype("<f4").tofile(os.path.join(d, "ref_xfilt.f32"))
        E["logits"].astype("<f4").tofile(os.path.join(d, "ref_logits.f32"))
        E["preds"].astype("<i4").tofile(os.path.join(d, "ref_preds.i32"))
        E["sims"].astype("<f8").tofile(os.path.join(d, "ref_sims.f64"))
        E["kstar"].astype("<i4").tofile(os.path.join(d, "ref_kstar.i32"))
        E["pred_final"].astype("<i4").tofile(os.path.join(d, "ref_pred.i32"))
        E["y"].astype("<i4").tofile(os.path.join(d, "label.i32"))
        per_model = per.size // N
        man = dict(N=int(N), C=int(C), K=int(K), T=int(T), M=int(M), NH=int(E["Q"].shape[2] // 2), padlen=int(Fz["padlen"]),
                   fs=float(Fz["fs"]), per_model=int(per_model), acc_ref=float(E["acc"]),
                   n_params_total=int(per.size))
        json.dump(man, open(os.path.join(d, "manifest.json"), "w"), indent=1)
        print(f"T{T}: {man}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
