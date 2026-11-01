"""Phan tich chan doan G3 tren Nakanishi (chot truoc khi chay).

n = 10 nguoi, accuracy = TB 5 seed. Cua so chinh 0,4 va 1,0 s: (i) G3-dc - G3, (ii) G3-pha-khong (wp3_qat8) - G3: Wilcoxon
hai phia, Holm trong ho 2 moi cua so; KTC 95% bootstrap theo nguoi (10 000). Mo ta: FBCCA (fb5, fb5c, fb5cd), tt-CCA goc,
DG-transfer (muc 17).
Dung: python analyze_naka_dc.py --dc_dir <dir> --out_dir <dir>   (OVERWRITE=1 de ghi de)
"""
import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_h1 import boot_ci, fail, holm, read_rows  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJ, "results")
G3 = os.path.join(RES, "g3_independent_2026_10_13")
BL = os.path.join(RES, "baselines_tf_2026_10_15")
WINDOWS, PRIMARY, SUBS = [0.2, 0.4, 0.6, 0.8, 1.0], [0.4, 1.0], list(range(1, 11))


def load(paths, n_seed, method=None, tag=None):
    acc = defaultdict(lambda: defaultdict(list))
    for p in paths:
        for r in read_rows(p):
            if (method and r["method"] != method) or (tag and r.get("tag") != tag):
                continue
            acc[round(float(r["window"]), 2)][int(r["subject"])].append(float(r["acc"]))
    out = {}
    for w in WINDOWS:
        if sorted(acc[w]) != SUBS or any(len(acc[w][s]) != n_seed for s in SUBS):
            fail(f"{paths[:1]} {method} {tag} w={w}: thieu nguoi/seed")
        out[w] = np.array([100 * np.mean(acc[w][s]) for s in SUBS])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dc_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    out_csv = os.path.join(a.out_dir, "naka_dc_tests.csv")
    if os.path.exists(out_csv) and os.environ.get("OVERWRITE") != "1":
        fail(f"{out_csv} da ton tai; OVERWRITE=1 de ghi de")
    os.makedirs(a.out_dir, exist_ok=True)
    g = lambda pat: sorted(glob.glob(os.path.join(a.dc_dir, pat)))  # noqa: E731
    M = {"G3 (fb5c, mục 10)": load(sorted(glob.glob(os.path.join(G3, "wp3_g3_nakanishi_w*_s*_per_subject.csv"))), 5),
         "G3-dc (fb5cd)": load(g("wp3_g3_dc_nakanishi_w*_s*_per_subject.csv"), 5),
         "G3 lọc pha không (wp3_qat8, fb5)": load(g("wp3_qat8_nakanishi_w*_s*_per_subject.csv"), 5),
         "FBCCA pha không (fb5)": load(g("fbcca_nakanishi_fb5_per_subject.csv"), 1, tag="fb5"),
         "FBCCA nhân quả (fb5c)": load([os.path.join(G3, "fbcca_nakanishi_fb5c_per_subject.csv")], 1, tag="fb5c"),
         "FBCCA nhân quả trừ DC (fb5cd)": load(g("fbcca_nakanishi_fb5cd_per_subject.csv"), 1, tag="fb5cd"),
         "tt-CCA gốc (mục 17)": load([os.path.join(BL, "ttcca_nakanishi_per_subject.csv")], 1, method="ttcca"),
         "DG-transfer pha không (mục 17)": load([os.path.join(BL, "huang_fb5_nakanishi_per_subject.csv")], 1, method="huang_fb5")}
    print("[tu kiem] du 10 nguoi x 5 cua so cho moi phuong phap; G3/G3-dc/G3-pha-khong du 5 seed", flush=True)
    base = M["G3 (fb5c, mục 10)"]
    rng = np.random.default_rng(20261016)
    rows = []
    for w in PRIMARY:
        fam = []
        for name in ("G3-dc (fb5cd)", "G3 lọc pha không (wp3_qat8, fb5)"):
            d = M[name][w] - base[w]
            lo, hi = boot_ci(d, rng)
            p = 1.0 if np.allclose(d, 0) else float(wilcoxon(M[name][w], base[w], alternative="two-sided").pvalue)
            fam.append(dict(window=w, so_sanh=f"{name} − G3", a=M[name][w].mean(), b=base[w].mean(), chenh=d.mean(),
                            lo=lo, hi=hi, a_hon=int((d > 0).sum()), p=p))
        for r, pa in zip(fam, holm([r["p"] for r in fam])):
            r.update(p_holm=pa, dat=pa < 0.05)
        rows += fam
    keys = ["window", "so_sanh", "a", "b", "chenh", "lo", "hi", "a_hon", "p", "p_holm", "dat"]
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{r[k]:.6g}" if isinstance(r[k], float) else r[k]) for k in keys})
    L = ["# Chẩn đoán G3 trên Nakanishi", "", "Accuracy TB theo người (%), n = 10; mức ngẫu nhiên 8,33%.", "",
         "| Phương pháp | " + " | ".join(f"{w:.1f} s" for w in WINDOWS) + " |", "|---" * 6 + "|"]
    for name, d in M.items():
        L.append(f"| {name} | " + " | ".join(f"{d[w].mean():.2f}" for w in WINDOWS) + " |")
    L += ["", "## Kiểm định (Wilcoxon hai phía, Holm trong họ 2 mỗi cửa sổ; n = 10, lực thấp)", "",
          "| Cửa sổ | So sánh | A | G3 | Chênh [KTC 95%] | A > G3 | p | p Holm | Đạt |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['window']:.1f} | {r['so_sanh']} | {r['a']:.2f} | {r['b']:.2f} | {r['chenh']:+.2f} [{r['lo']:+.2f}, "
                 f"{r['hi']:+.2f}] | {r['a_hon']}/10 | {r['p']:.3g} | {r['p_holm']:.3g} | {'có' if r['dat'] else 'không'} |")
    open(os.path.join(a.out_dir, "naka_dc_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("DONE")


if __name__ == "__main__":
    main()
