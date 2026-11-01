"""Phan tich G3-dc vs G3 tren Benchmark + BETA (chot truoc khi chay).
Chinh: khong-kem-hon G3-dc - G3 > -1 diem (Wilcoxon mot phia tren chenh + 1), 2 bo x 2 cua so chinh, Holm trong ho 4.
Quy tac: ca 4 dat -> G3.1 = G3-dc. Mo ta: Wilcoxon hai phia moi cua so.
Dung: python analyze_g3dc.py --dc_dir <dir> --out_dir <dir>   (OVERWRITE=1 de ghi de)
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_g3 import load  # noqa: E402  (load(paths, n_sub, ...) theo nguoi, TB 5 seed, kiem du)
from analyze_h1 import boot_ci, fail, holm  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G3 = os.path.join(PROJ, "results", "g3_independent_2026_10_13")
WINDOWS, PRIMARY, NSUB = [0.2, 0.4, 0.6, 0.8, 1.0], [0.4, 1.0], {"benchmark": 35, "beta": 70}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dc_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    out_csv = os.path.join(a.out_dir, "g3dc_tests.csv")
    if os.path.exists(out_csv) and os.environ.get("OVERWRITE") != "1":
        fail(f"{out_csv} da ton tai; OVERWRITE=1 de ghi de")
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(20261017)
    rows, desc, prim = [], [], []
    for ds, n in NSUB.items():
        g3, _ = load(sorted(glob.glob(os.path.join(G3, f"wp3_g3_{ds}_w*_s*_per_subject.csv"))), n)
        dc, _ = load(sorted(glob.glob(os.path.join(a.dc_dir, f"wp3_g3_dc_{ds}_w*_s*_per_subject.csv"))), n)
        for w in WINDOWS:
            d = dc[w] - g3[w]
            lo, hi = boot_ci(d, rng)
            p2 = 1.0 if np.allclose(d, 0) else float(wilcoxon(dc[w], g3[w], alternative="two-sided").pvalue)
            r = dict(dataset=ds, window=w, g3=g3[w].mean(), g3dc=dc[w].mean(), chenh=d.mean(), lo=lo, hi=hi,
                     dc_hon=int((d > 0).sum()), n=n, p_hai_phia=p2,
                     p_noninf=float(wilcoxon(d + 1.0, alternative="greater").pvalue))
            desc.append(r)
            if w in PRIMARY:
                prim.append(r)
    for r, pa in zip(prim, holm([r["p_noninf"] for r in prim])):
        r.update(p_noninf_holm=pa, noninf_dat=pa < 0.05)
    g31 = all(r["noninf_dat"] for r in prim)
    print("[tu kiem] du nguoi x 5 seed x 5 cua so cho G3 (17026) va G3-dc; 2 bo", flush=True)
    keys = ["dataset", "window", "n", "g3", "g3dc", "chenh", "lo", "hi", "dc_hon", "p_hai_phia", "p_noninf", "p_noninf_holm",
            "noninf_dat"]
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in desc:
            wr.writerow({k: (f"{r[k]:.6g}" if isinstance(r.get(k), float) else r.get(k, "")) for k in keys})
    L = ["# G3-dc so với G3 trên Benchmark và BETA", "",
         "Chênh = G3-dc − G3 (điểm %), TB 5 seed theo người; KTC 95% bootstrap. Chính: không-kém-hơn (biên 1 điểm), Holm họ 4.", "",
         "| Bộ | Cửa sổ | G3 | G3-dc | Chênh [KTC 95%] | G3-dc hơn | p hai phía | p không-kém-hơn | p Holm (chính) | Đạt |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in desc:
        prim_s = f"{r['p_noninf_holm']:.3g}" if "p_noninf_holm" in r else "—"
        dat = ("có" if r["noninf_dat"] else "không") if "noninf_dat" in r else "(mô tả)"
        L.append(f"| {r['dataset']} | {r['window']:.1f} | {r['g3']:.2f} | {r['g3dc']:.2f} | {r['chenh']:+.2f} [{r['lo']:+.2f}, "
                 f"{r['hi']:+.2f}] | {r['dc_hon']}/{r['n']} | {r['p_hai_phia']:.3g} | {r['p_noninf']:.3g} | {prim_s} | {dat} |")
    L += ["", f"**Quyết định theo quy tắc chốt trước:** {'G3.1 = G3-dc (cả 4 phép không-kém-hơn đạt)' if g31 else 'giữ G3 (không đủ 4 phép đạt)'}."]
    open(os.path.join(a.out_dir, "g3dc_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("DONE")


if __name__ == "__main__":
    main()
