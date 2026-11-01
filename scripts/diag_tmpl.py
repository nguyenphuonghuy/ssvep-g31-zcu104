"""Chan doan D4a (hau kiem, mo ta): chat luong template nguoi nguon, trung binh so voi trung vi.
Moi bo (zhu_dry, zhu_wet, benchmark), cua so 1,0 s, fb5cd, dai con 1, template tu MOI nguoi:
  rho CCA (Nh = 5) giua template va tham chieu sin/cos cua chinh muc tieu (TB theo muc tieu), template TB va template trung vi;
  ty phan nang luong template TB den tu 1% luot thu co max|x| lon nhat.
Ra <out_dir>/d4a_report.md. Dung: python diag_tmpl.py --out_dir <dir>"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baselines_tf as B  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    from ssvep.prep import prepare_causal
    L = ["# D4a: chất lượng template người nguồn (hậu kiểm, mô tả; 1,0 s, dải con 1)", "",
         "| Bộ | ρ CCA template TB | ρ CCA template trung vị | Năng lượng template TB từ 1% lượt thử nhiễu nhất (%) |",
         "|---|---|---|---|"]
    for ds in ("zhu_dry", "zhu_wet", "benchmark"):
        X, y, meta = prepare_causal(ds, 1.0, "fb5cd")                  # (S, n, 5, C, N)
        K = len(meta["freqs"])
        x = X[:, :, 0].astype(np.float64)                               # dai con 1: (S, n, C, N)
        S, n, C, N = x.shape
        x = x.reshape(S, n // K, K, C, N).transpose(2, 0, 1, 3, 4).reshape(K, -1, C, N)   # (K, S*Nb, C, N)
        refs = B.sincos(meta["freqs"], meta["srate"], N, 5)
        tm_mean, tm_med = x.mean(1), np.median(x, axis=1)
        r_mean = np.mean([B.cca(tm_mean[k], refs[k])[0] for k in range(K)])
        r_med = np.mean([B.cca(tm_med[k], refs[k])[0] for k in range(K)])
        mx = np.abs(x).reshape(K, x.shape[1], -1).max(-1)                 # (K, trials)
        fr = []
        for k in range(K):
            top = mx[k] >= np.percentile(mx[k], 99)
            e_all = (tm_mean[k] ** 2).sum()
            e_top = ((x[k][top].sum(0) / x.shape[1]) ** 2).sum()
            fr.append(e_top / e_all)
        L.append(f"| {ds} | {r_mean:.3f} | {r_med:.3f} | {100 * np.mean(fr):.1f} |")
        print(L[-1], flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    open(os.path.join(a.out_dir, "d4a_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("DONE")


if __name__ == "__main__":
    main()
