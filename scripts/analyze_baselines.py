"""Phan tich doi chung tt-CCA / DG-transfer (chot truoc khi chay).

Moi bo du lieu, moi cua so chinh (0,4; 1,0 s): G3 (LOSO, TB 5 seed; job 17026) - ttcca va G3 - huang_fb5c: Wilcoxon hai phia,
Holm trong ho 2; KTC 95% bootstrap theo nguoi (10 000). Mo ta: moi phuong phap, moi cua so (accuracy, ITR Tg 0,5 s).
Tu kiem (exit 3): du nguoi x cua so cho moi phuong phap; G3 du 5 seed.
Dung: python analyze_baselines.py --bl_dir <dir> --out_dir <dir> [--model wp3_g3_dc]   (OVERWRITE=1 de ghi de)
--model wp3_g3_dc (G3.1): mo hinh de xuat = G3.1 (job 17200 Benchmark/BETA, job 17121 Nakanishi); cung phep kiem.
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
from ssvep.metrics import itr_bpm  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G3 = os.path.join(PROJ, "results", "g3_independent_2026_10_13")
G31 = {"benchmark": os.path.join(PROJ, "results", "g3dc_2026_10_17"), "beta": os.path.join(PROJ, "results", "g3dc_2026_10_17"),
       "nakanishi": os.path.join(PROJ, "results", "naka_dc_2026_10_15")}
WINDOWS, PRIMARY = [0.2, 0.4, 0.6, 0.8, 1.0], [0.4, 1.0]
NSUB = {"benchmark": 35, "beta": 70, "nakanishi": 10}
MT = {"benchmark": 40, "beta": 40, "nakanishi": 12}
METHODS = {"ttcca": "tt-CCA (gốc)", "ttcca_fb5c": "tt-CCA + bank lọc", "huang_fb5c": "DG-transfer (nhân quả)",
           "huang_fb5": "DG-transfer (pha không)"}


def load(paths, ds, n_seed, method=None, tag=None):
    acc = defaultdict(lambda: defaultdict(list))
    for p in paths:
        for r in read_rows(p):
            if (method and r["method"] != method) or (tag and r.get("tag") != tag):
                continue
            acc[round(float(r["window"]), 2)][int(r["subject"])].append(float(r["acc"]))
    subs = list(range(1, NSUB[ds] + 1))
    out = {}
    for w in WINDOWS:
        if sorted(acc[w]) != subs or any(len(acc[w][s]) != n_seed for s in subs):
            fail(f"{ds} {method or paths[0]} w={w}: thieu nguoi/seed")
        out[w] = np.array([100 * np.mean(acc[w][s]) for s in subs])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bl_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="wp3_g3")
    a = ap.parse_args()
    g31 = a.model == "wp3_g3_dc"
    NAME = "G3.1" if g31 else "G3"
    out_csv = os.path.join(a.out_dir, "baselines_tests.csv")
    if os.path.exists(out_csv) and os.environ.get("OVERWRITE") != "1":
        fail(f"{out_csv} da ton tai; OVERWRITE=1 de ghi de")
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(20261015)
    rows, desc = [], []
    for ds in NSUB:
        gdir = G31[ds] if g31 else G3
        g3 = load(sorted(glob.glob(os.path.join(gdir, f"{a.model}_{ds}_w*_s*_per_subject.csv"))), ds, 5, method=a.model)
        fbc = load([os.path.join(G3, f"fbcca_{ds}_fb5c_per_subject.csv")], ds, 1, tag="fb5c")
        bl = {m: load([os.path.join(a.bl_dir, f"{m}_{ds}_per_subject.csv")], ds, 1, method=m) for m in METHODS}
        allm = {f"{NAME} (LOSO)": g3, "FBCCA nhân quả": fbc, **{METHODS[m]: bl[m] for m in METHODS}}
        for name, d in allm.items():
            for w in WINDOWS:
                itr = np.mean([itr_bpm(p / 100, MT[ds], w, 0.5) for p in d[w]])
                desc.append(dict(dataset=ds, method=name, window=w, acc=d[w].mean(), itr=itr))
        for w in PRIMARY:
            fam = []
            for m in ("ttcca", "huang_fb5c"):
                dd = g3[w] - bl[m][w]
                lo, hi = boot_ci(dd, rng)
                p = 1.0 if np.allclose(dd, 0) else float(wilcoxon(g3[w], bl[m][w], alternative="two-sided").pvalue)
                fam.append(dict(dataset=ds, window=w, so_sanh=f"{NAME} − {METHODS[m]}", mean_g3=g3[w].mean(),
                                mean_b=bl[m][w].mean(), chenh=dd.mean(), lo=lo, hi=hi, g3_hon=int((dd > 0).sum()),
                                n=len(dd), p=p))
            for r, pa in zip(fam, holm([r["p"] for r in fam])):
                r.update(p_holm=pa, dat=pa < 0.05)
            rows += fam
    print(f"[tu kiem] du nguoi x cua so cho 4 doi chung, {NAME} (5 seed), FBCCA nhan qua; 3 bo du lieu", flush=True)
    keys = ["dataset", "window", "so_sanh", "n", "mean_g3", "mean_b", "chenh", "lo", "hi", "g3_hon", "p", "p_holm", "dat"]
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{r[k]:.6g}" if isinstance(r[k], float) else r[k]) for k in keys})
    with open(os.path.join(a.out_dir, "baselines_desc.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["dataset", "method", "window", "acc", "itr"])
        wr.writeheader()
        for r in desc:
            wr.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    L = ["# Đối chứng không hiệu chuẩn: tt-CCA và DG-transfer", "",
         f"Accuracy TB theo người (%); ITR Tg = 0,5 s (bits/phút) trong ngoặc. {NAME} = TB 5 seed; đối chứng tất định.", ""]
    for ds in NSUB:
        L += [f"## {ds}", "", "| Phương pháp | " + " | ".join(f"{w:.1f} s" for w in WINDOWS) + " |", "|---" * 6 + "|"]
        for name in [f"{NAME} (LOSO)", "FBCCA nhân quả"] + list(METHODS.values()):
            cells = [next(f"{r['acc']:.2f} ({r['itr']:.1f})" for r in desc if r["dataset"] == ds and r["method"] == name
                          and r["window"] == w) for w in WINDOWS]
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L.append("")
    L += [f"## Kiểm định ({NAME} − đối chứng, Wilcoxon hai phía, Holm trong họ 2 mỗi bộ × cửa sổ)", "",
          f"| Bộ | Cửa sổ | So sánh | {NAME} | Đối chứng | Chênh [KTC 95%] | {NAME} hơn | p | p Holm | Đạt |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['dataset']} | {r['window']:.1f} | {r['so_sanh']} | {r['mean_g3']:.2f} | {r['mean_b']:.2f} | "
                 f"{r['chenh']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] | {r['g3_hon']}/{r['n']} | {r['p']:.3g} | "
                 f"{r['p_holm']:.3g} | {'có' if r['dat'] else 'không'} |")
    open(os.path.join(a.out_dir, "baselines_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("DONE")


if __name__ == "__main__":
    main()
