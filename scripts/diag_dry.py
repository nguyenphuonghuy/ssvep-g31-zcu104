"""Chan doan D1 (hau kiem, mo ta): thang luong tu QAT cua G3.1 tren dien cuc kho/uot (Zhu) va Benchmark.

Moi bo du lieu, moi cua so (0,4; 1,0 s), checkpoint G3.1 seed 0 moi fold:
  amax cua q_x, q_z (TB, trung vi, min, max qua fold);
  tren du lieu nguoi kiem thu cua fold: ty le mau |x| < s/2 (bi luong tu ve 0; s = amax_x / 127), ty le luot thu co max|x| < s;
  phan vi (50/90/99/100) cua max|x| theo luot thu (moi nguoi cua bo).
Ra <out_dir>/d1_quant.csv, d1_report.md. Dung: python diag_dry.py --out_dir <dir>
"""
import argparse
import csv
import glob
import os
import re
import statistics as st
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = {"zhu_dry": "checkpoints/zhu_g31_ckpt_2026_10_29", "zhu_wet": "checkpoints/zhu_g31_ckpt_2026_10_29",
      "benchmark": "checkpoints/g3dc_ckpt_benchmark_2026_10_18"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    import torch
    from ssvep.prep import prepare_causal
    os.makedirs(a.out_dir, exist_ok=True)
    rows = []
    for ds, ck in CK.items():
        for win in (0.4, 1.0):
            X, _, _ = prepare_causal(ds, win, "fb5cd")                         # (S, n luot, 5, C, N)
            tmax = np.abs(X).reshape(X.shape[0], X.shape[1], -1).max(-1)        # (S, n luot)
            files = sorted(glob.glob(os.path.join(PROJ, ck, f"wp3_g3_dc_{ds}_w{win:.2f}_s0_f*.pt")),
                           key=lambda p: int(re.search(r"_f(\d+)\.pt$", p).group(1)))
            ax, az, z0, t0 = [], [], [], []
            for p in files:
                f = int(re.search(r"_f(\d+)\.pt$", p).group(1))
                sd = torch.load(p, map_location="cpu")
                amx, amz = float(sd["q_x.amax"]), float(sd["q_z.amax"])
                s = amx / 127.0
                x = np.abs(X[f - 1])
                ax.append(amx); az.append(amz)
                z0.append(float((x < s / 2).mean()))
                t0.append(float((tmax[f - 1] < s).mean()))
            q = np.percentile(tmax, [50, 90, 99, 100])
            r = dict(dataset=ds, window=win, n_fold=len(files), amax_x_mean=st.mean(ax), amax_x_med=st.median(ax),
                     amax_x_min=min(ax), amax_x_max=max(ax), amax_z_mean=st.mean(az), frac_zero=st.mean(z0),
                     frac_trial_below_s=st.mean(t0), trialmax_p50=q[0], trialmax_p90=q[1], trialmax_p99=q[2], trialmax_max=q[3])
            rows.append(r)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    with open(os.path.join(a.out_dir, "d1_quant.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    L = ["# D1: thang lượng tử QAT của G3.1 (hậu kiểm, mô tả)", "",
         "Đơn vị tín hiệu: µV sau trừ TB và bank lọc nhân quả (fb5cd). s = amax_x / 127. Seed 0, mọi fold.", "",
         "| Bộ | Cửa sổ | Fold | amax_x TB (min–max) | amax_z TB | Mẫu về 0 (%) | Lượt thử có max|x| < s (%) | max|x| theo lượt thử: p50 / p90 / p99 / max |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['dataset']} | {r['window']} | {r['n_fold']} | {r['amax_x_mean']:.1f} ({r['amax_x_min']:.1f}–{r['amax_x_max']:.1f}) | "
                 f"{r['amax_z_mean']:.1f} | {100 * r['frac_zero']:.1f} | {100 * r['frac_trial_below_s']:.1f} | "
                 f"{r['trialmax_p50']:.1f} / {r['trialmax_p90']:.1f} / {r['trialmax_p99']:.1f} / {r['trialmax_max']:.1f} |")
    open(os.path.join(a.out_dir, "d1_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("DONE")


if __name__ == "__main__":
    main()
