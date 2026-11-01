""": G3.1 so voi G3 khoi tao trang thai loc xac lap (wp3_g3_ss) va luong noi (wp3_g3_st) tren Nakanishi, Zhu uot.

Ho H29a: G3.1 − G3-ss; ho H29b: G3.1 − G3-st; moi ho 4 phep (2 bo x 2 cua so), Wilcoxon hai phia ghep theo nguoi, Holm trong 4,
KTC 95% bootstrap theo nguoi. Mo ta: G3-ss − G3, G3-st − G3. Quy tac dien giai (chot truoc) in ra cuoi bao cao.
Dung: python analyze_filtinit.py --out_dir results/filtinit_analysis_2026_11_01   (exit 3 neu thieu file/nguoi/seed)
"""
import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_h1 import boot_ci, fail, holm, read_rows  # noqa: E402
from analyze_review26 import test, write_csv  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJ, "results")
NSUB = {"nakanishi": 10, "zhu_wet": 102}
WINDOWS = [0.4, 1.0]
SRC = {("G3.1", "nakanishi"): ("naka_dc_2026_10_15", "wp3_g3_dc"), ("G3", "nakanishi"): ("g3_independent_2026_10_13", "wp3_g3"),
       ("G3.1", "zhu_wet"): ("zhu_g3_2026_10_29", "wp3_g3_dc"), ("G3", "zhu_wet"): ("zhu_g3_2026_10_29", "wp3_g3")}


def load(d, model, ds):
    acc = defaultdict(lambda: defaultdict(list))
    for p in sorted(glob.glob(os.path.join(RES, d, f"{model}_{ds}_w*_s*_per_subject.csv"))):
        for r in read_rows(p):
            if r["method"] == model and r["tag"] == "base":
                acc[round(float(r["window"]), 2)][int(r["subject"])].append(float(r["acc"]))
    out = {}
    for w in WINDOWS:
        if sorted(acc[w]) != list(range(1, NSUB[ds] + 1)) or any(len(v) != 5 for v in acc[w].values()):
            fail(f"{d}/{model} {ds} w={w}: thieu nguoi/seed")
        out[w] = np.array([100 * np.mean(acc[w][s]) for s in range(1, NSUB[ds] + 1)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--new_dir", default="filtinit_2026_11_01")
    a = ap.parse_args()
    out_csv = os.path.join(a.out_dir, "filtinit_tests.csv")
    if os.path.exists(out_csv) and os.environ.get("OVERWRITE") != "1":
        fail(f"{out_csv} da ton tai; OVERWRITE=1 de ghi de")
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(20261101)
    M = {}
    for ds in NSUB:
        for k, (d, m) in SRC.items():
            if k[1] == ds:
                M[(k[0], ds)] = load(d, m, ds)
        M[("G3-ss", ds)] = load(a.new_dir, "wp3_g3_ss", ds)
        M[("G3-st", ds)] = load(a.new_dir, "wp3_g3_st", ds)
    rows = []
    for fam, b in (("H29a", "G3-ss"), ("H29b", "G3-st")):
        fr = [dict(ho=fam, so_sanh=f"G3.1 − {b}", dataset=ds, window=w, **test(M[("G3.1", ds)][w], M[(b, ds)][w], rng))
              for ds in NSUB for w in WINDOWS]
        for r, ph in zip(fr, holm([r["p"] for r in fr])):
            r["p_holm"] = ph
        rows += fr
    for b in ("G3-ss", "G3-st"):                               # mo ta, khong hieu chinh
        for ds in NSUB:
            for w in WINDOWS:
                rows.append(dict(ho="mo ta", so_sanh=f"{b} − G3", dataset=ds, window=w, p_holm=float("nan"),
                                 **test(M[(b, ds)][w], M[("G3", ds)][w], rng)))
    keys = ["ho", "so_sanh", "dataset", "window", "mean_g3", "mean_b", "chenh", "lo", "hi", "g3_hon", "b_hon", "n", "p", "p_holm"]
    write_csv(out_csv, rows, keys)
    L = ["# khởi tạo trạng thái lọc so với trừ trung bình\n", "Độ chính xác TB theo người (%), 5 seed.\n",
         "| Bộ | Cửa sổ | G3 | G3.1 | G3-ss | G3-st |", "|---|---|---|---|---|---|"]
    for ds in NSUB:
        for w in WINDOWS:
            L.append(f"| {ds} | {w} | " + " | ".join(f"{M[(k, ds)][w].mean():.2f}" for k in ("G3", "G3.1", "G3-ss", "G3-st")) + " |")
    L += ["\n| Họ | So sánh | Bộ | Cửa sổ | Chênh [KTC 95%] | A>B | B>A | p | p Holm |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['ho']} | {r['so_sanh']} | {r['dataset']} | {r['window']} | {r['chenh']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] | "
                 f"{r['g3_hon']} | {r['b_hon']} | {r['p']:.3g} | {r['p_holm']:.3g} |")
    wet = [r for r in rows if r["ho"] in ("H29a", "H29b") and r["dataset"] == "zhu_wet"]
    sig_hon = [r["p_holm"] < 0.05 and r["chenh"] > 0 for r in wet]
    sig_kem = [r["p_holm"] < 0.05 and r["chenh"] < 0 for r in rows if r["ho"] in ("H29a", "H29b")]
    rule = 3 if any(sig_kem) else (1 if all(sig_hon) else 2)
    L.append(f"\n**Quy tắc diễn giải áp dụng: {rule}** (1: giữ phát biểu; 2: thu hẹp đóng góp 1; 3: phương án khác tốt hơn).")
    open(os.path.join(a.out_dir, "filtinit_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
