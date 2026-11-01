"""Phan tich muc 27 (chot truoc khi chay): kiem dinh doc lap tren Zhu 2021 (dien cuc kho/uot).

Ho chinh: moi loai dien cuc x cua so chinh (0,4; 1,0 s), 4 phep G3.1 (TB 5 seed theo nguoi) - {tt-CCA goc, ttcca_fb5cd,
  huang_fb5cd, FBCCA fb5cd}: Wilcoxon hai phia ghep theo nguoi, Holm trong ho 4; KTC 95% bootstrap (10 000). Do nhay: Holm tren 16.
Ho dong gop 2: G3.1 - G3 moi loai dien cuc x cua so (4 phep), Holm trong 4.
Mo ta: uot - kho cua G3.1 (KTC bootstrap ghep theo nguoi); Spearman acc G3.1 (kho) voi tro khang TB cua nguoi (Impedance.mat, kho);
  golden 16/24 bit; moi phuong phap moi cua so (accuracy, ITR Tg 0,5 s).
Tu kiem (exit 3): du 102 nguoi x cua so; G3.1, G3 du 5 seed.
Dung: python analyze_zhu.py --out_dir <dir>   (OVERWRITE=1 de ghi de)
"""
import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_h1 import boot_ci, fail, holm, read_rows  # noqa: E402
from analyze_review26 import test, write_csv  # noqa: E402
from ssvep.metrics import itr_bpm  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJ, "results")
DS = {"zhu_dry": "khô", "zhu_wet": "ướt"}
NSUB, MT, PRIMARY, WINDOWS = 102, 12, [0.4, 1.0], [0.2, 0.4, 0.6, 0.8, 1.0]
IMP = os.path.join(os.environ.get("SSVEP_DATA", os.path.join(PROJ, "datasets")), "wearable_zhu2021", "Impedance.mat")
NAMES = {"ttcca": "tt-CCA (gốc)", "ttcca_fb5cd": "tt-CCA + bank lọc, trừ TB", "huang_fb5cd": "DG-transfer (nhân quả, trừ TB)",
         "fbcca_fb5cd": "FBCCA (nhân quả, trừ TB)"}
DESC_ONLY = {"fbcca_fb5": "FBCCA (pha không)", "fbcca_fb5c": "FBCCA (nhân quả)"}


def load(paths, n_seed, windows, method=None, tag=None):
    acc = defaultdict(lambda: defaultdict(list))
    for p in paths:
        for r in read_rows(p):
            if (method and r["method"] != method) or (tag and r.get("tag") != tag):
                continue
            acc[round(float(r["window"]), 2)][int(r["subject"])].append(float(r["acc"]))
    out = {}
    for w in windows:
        if sorted(acc[w]) != list(range(1, NSUB + 1)) or any(len(acc[w][s]) != n_seed for s in acc[w]):
            fail(f"{method or paths[0]} w={w}: thieu nguoi/seed")
        out[w] = np.array([100 * np.mean(acc[w][s]) for s in range(1, NSUB + 1)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--g_dir", default=os.path.join(RES, "zhu_g3_2026_10_29"))
    ap.add_argument("--bl_dir", default=os.path.join(RES, "zhu_baselines_2026_10_29"))
    ap.add_argument("--golden_dir", default=os.path.join(RES, "zhu_golden_2026_10_29"))
    a = ap.parse_args()
    out_csv = os.path.join(a.out_dir, "zhu_tests.csv")
    if os.path.exists(out_csv) and os.environ.get("OVERWRITE") != "1":
        fail(f"{out_csv} da ton tai; OVERWRITE=1 de ghi de")
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(20261029)
    rows, rows_g3, desc, g31 = [], [], [], {}
    for ds in DS:
        g = load(sorted(glob.glob(os.path.join(a.g_dir, f"wp3_g3_dc_{ds}_w*_s*_per_subject.csv"))), 5, PRIMARY, method="wp3_g3_dc")
        g3 = load(sorted(glob.glob(os.path.join(a.g_dir, f"wp3_g3_{ds}_w*_s*_per_subject.csv"))), 5, PRIMARY, method="wp3_g3")
        g31[ds] = g
        bl = {m: load([os.path.join(a.bl_dir, f"{m}_{ds}_per_subject.csv")], 1, WINDOWS, method=m)
              for m in ("ttcca", "ttcca_fb5cd", "huang_fb5cd")}
        for m in ("fbcca_fb5cd", "fbcca_fb5", "fbcca_fb5c"):
            bl[m] = load([os.path.join(a.bl_dir, f"fbcca_{ds}_{m.split('_')[1]}_per_subject.csv")], 1, WINDOWS, tag=m.split("_")[1])
        for name, d, ws in [("G3.1 (LOSO)", g, PRIMARY), ("G3 (LOSO)", g3, PRIMARY)] + \
                [({**NAMES, **DESC_ONLY}[m], bl[m], WINDOWS) for m in bl]:
            for w in ws:
                desc.append(dict(dataset=ds, method=name, window=w, acc=d[w].mean(),
                                 itr=np.mean([itr_bpm(p / 100, MT, w, 0.5) for p in d[w]])))
        for w in PRIMARY:
            fam = [dict(dataset=ds, window=w, so_sanh=f"G3.1 − {NAMES[m]}", **test(g[w], bl[m][w], rng)) for m in NAMES]
            for r, pa in zip(fam, holm([r["p"] for r in fam])):
                r.update(p_holm=pa, dat=bool(pa < 0.05 and r["chenh"] > 0), b_hon_yn=bool(pa < 0.05 and r["chenh"] < 0))
            rows += fam
            rows_g3.append(dict(dataset=ds, window=w, so_sanh="G3.1 − G3", **test(g[w], g3[w], rng)))
    for r, pa in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm16"] = pa
    for r, pa in zip(rows_g3, holm([r["p"] for r in rows_g3])):
        r.update(p_holm=pa, dat=bool(pa < 0.05 and r["chenh"] > 0), b_hon_yn=bool(pa < 0.05 and r["chenh"] < 0))
    print("[tu kiem] du 102 nguoi x cua so; G3.1, G3 du 5 seed; doi chung du 5 cua so", flush=True)

    L = ["# Mục 27: kiểm định độc lập trên Zhu 2021 (điện cực khô và ướt)", "",
         "Accuracy TB theo người (%); ITR Tg = 0,5 s (bits/phút) trong ngoặc. G3.1, G3 = TB 5 seed; đối chứng tất định. n = 102.", ""]
    for ds, lbl in DS.items():
        L += [f"## Điện cực {lbl} ({ds})", "", "| Phương pháp | " + " | ".join(f"{w:.1f} s" for w in WINDOWS) + " |", "|---" * 6 + "|"]
        for name in ["G3.1 (LOSO)", "G3 (LOSO)"] + list(NAMES.values()) + list(DESC_ONLY.values()):
            cells = []
            for w in WINDOWS:
                m = [r for r in desc if r["dataset"] == ds and r["method"] == name and r["window"] == w]
                cells.append(f"{m[0]['acc']:.2f} ({m[0]['itr']:.1f})" if m else "–")
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L.append("")
    n_pass, n_bl = sum(r["dat"] for r in rows), sum(r["b_hon_yn"] for r in rows)
    hdr = "| Điện cực | Cửa sổ | So sánh | G3.1 | Đối chứng | Chênh [KTC 95%] | G3.1 hơn | p | p Holm |"
    L += ["## Họ chính (Wilcoxon hai phía; Holm trong họ 4 mỗi điện cực × cửa sổ)", "",
          f"**G3.1 cao hơn có ý nghĩa ở {n_pass}/{len(rows)} phép; đối chứng cao hơn có ý nghĩa ở {n_bl}/{len(rows)} phép.**", "",
          hdr + " p Holm (16) | Đạt |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {DS[r['dataset']]} | {r['window']:.1f} | {r['so_sanh']} | {r['mean_g3']:.2f} | {r['mean_b']:.2f} | "
                 f"{r['chenh']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] | {r['g3_hon']}/{r['n']} | {r['p']:.3g} | {r['p_holm']:.3g} | "
                 f"{r['p_holm16']:.3g} | {'có' if r['dat'] else ('ngược' if r['b_hon_yn'] else 'không')} |")
    n16 = sum(r["p_holm16"] < 0.05 and r["chenh"] > 0 for r in rows)
    L += ["", f"Độ nhạy (mô tả): Holm trên cả 16 phép, G3.1 cao hơn có ý nghĩa ở {n16}/16 phép.", "",
          "## Họ đóng góp 2: G3.1 − G3 (Holm trong 4)", "", hdr + " Đạt |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows_g3:
        L.append(f"| {DS[r['dataset']]} | {r['window']:.1f} | {r['so_sanh']} | {r['mean_g3']:.2f} | {r['mean_b']:.2f} | "
                 f"{r['chenh']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] | {r['g3_hon']}/{r['n']} | {r['p']:.3g} | {r['p_holm']:.3g} | "
                 f"{'có' if r['dat'] else ('ngược' if r['b_hon_yn'] else 'không')} |")
    # mo ta: uot - kho; tro khang
    L += ["", "## Mô tả", ""]
    from scipy.io import loadmat
    imp = loadmat(IMP)["Impedance"]                                   # [8, 10, 2, 102], 1: uot, 2: kho (Readme)
    imp_dry = np.nanmean(imp[:, :, 1, :], axis=(0, 1))
    for w in PRIMARY:
        dd = g31["zhu_wet"][w] - g31["zhu_dry"][w]
        lo, hi = boot_ci(dd, rng)
        rho, p = spearmanr(g31["zhu_dry"][w], imp_dry)
        L.append(f"- {w:.1f} s: G3.1 ướt − khô = {dd.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] điểm; Spearman(acc khô, trở kháng khô) "
                 f"ρ = {rho:.3f}, p = {p:.3g} (mô tả).")
    L += ["", "## Golden (mô tả; tiêu chí chênh TB trong ±1 điểm)", ""]
    for b in (16, 24):
        for ds in DS:
            f = os.path.join(a.golden_dir, f"in{b}", ds, "golden_summary.md")
            if os.path.exists(f):
                L += [f"**{b} bit, {DS[ds]}**", ""] + [x for x in open(f, encoding="utf-8").read().splitlines() if x.startswith("|")] + [""]
            else:
                L.append(f"- {b} bit, {DS[ds]}: CHƯA CÓ")
    keys = ["dataset", "window", "so_sanh", "n", "mean_g3", "mean_b", "chenh", "lo", "hi", "g3_hon", "b_hon", "p", "p_holm"]
    write_csv(out_csv, rows, keys + ["p_holm16", "dat", "b_hon_yn"])
    write_csv(os.path.join(a.out_dir, "zhu_tests_g3.csv"), rows_g3, keys + ["dat", "b_hon_yn"])
    write_csv(os.path.join(a.out_dir, "zhu_desc.csv"), desc, ["dataset", "method", "window", "acc", "itr"])
    open(os.path.join(a.out_dir, "zhu_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("DONE")


if __name__ == "__main__":
    main()
