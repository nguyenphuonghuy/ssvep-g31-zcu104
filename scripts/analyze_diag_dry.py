"""Tong hop D2/D3 (hau kiem, MO TA): so voi G3.1 seed 0 cung fold (results/zhu_g3_2026_10_29).
Dung: python analyze_diag_dry.py   -> results/zhu_diag_2026_10_30/d23_report.md"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_h1 import boot_ci  # noqa: E402

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
D, G = os.path.join(R, "zhu_diag_2026_10_30"), os.path.join(R, "zhu_g3_2026_10_29")


def acc(d, m, ds, w):
    p = os.path.join(d, f"{m}_{ds}_w{w}_s0_per_subject.csv")
    if not os.path.exists(p):
        return None
    r = {int(x["fold"]): 100 * float(x["acc"]) for x in csv.DictReader(open(p)) if x["tag"] == "base"}
    return np.array([r[f] for f in range(1, 103)]) if len(r) == 102 else None


def tr(d, m, ds, w):
    p = os.path.join(d, f"{m}_{ds}_w{w}_s0_timing.csv")
    return 100 * np.mean([float(x["train_acc"]) for x in csv.DictReader(open(p))]) if os.path.exists(p) else float("nan")


rng = np.random.default_rng(20261030)
L = ["# D2/D3/D4b: G3.1 trên Zhu (hậu kiểm, mô tả; seed 0; n = 102)", "",
     "| Điện cực | Cửa sổ | Mô hình | Acc kiểm thử (%) | Acc huấn luyện (%) | − G3.1 seed 0 [KTC 95%] |", "|---|---|---|---|---|---|"]
for ds in ("zhu_dry", "zhu_wet"):
    for w in ("0.40", "1.00"):
        g = acc(G, "wp3_g3_dc", ds, w)
        L.append(f"| {ds} | {w} | G3.1 (QAT max) | {g.mean():.2f} | {tr(G, 'wp3_g3_dc', ds, w):.1f} | – |")
        for m, nm in (("wp3_g3_dc_fp", "không lượng tử (D2)"), ("wp3_g3_dc_p999", "QAT phân vị 99,9% (D3)"),
                      ("wp3_g3_dc_fp_med", "không lượng tử + template trung vị (D4b)")):
            a = acc(D, m, ds, w)
            if a is None:
                continue
            lo, hi = boot_ci(a - g, rng)
            L.append(f"| {ds} | {w} | {nm} | {a.mean():.2f} | {tr(D, m, ds, w):.1f} | {(a - g).mean():+.2f} [{lo:+.2f}, {hi:+.2f}] |")
open(os.path.join(D, "d23_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
