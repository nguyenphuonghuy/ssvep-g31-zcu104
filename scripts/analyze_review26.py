"""Phan tich muc 26 (chot truoc khi chay): doi chung co tru TB doan truoc kich thich + golden moi bo.

26a. Moi bo du lieu x cua so chinh (0,4; 1,0 s): ho 4 phep G3.1 (TB 5 seed theo nguoi) - {tt-CCA goc, ttcca_fb5cd, huang_fb5cd,
     FBCCA fb5cd}: Wilcoxon hai phia ghep theo nguoi, Holm trong ho 4; KTC 95% bootstrap theo nguoi (10 000).
     Do nhay (mo ta): Holm tren ca 24 phep. Mo ta: moi phuong phap, moi cua so (accuracy, ITR Tg 0,5 s).
26b. Golden so voi QAT moi bo x cua so (chenh TB trong +-1 diem; ty le trung quyet dinh), bao hoa dau vao int16, cong tai lap R.
Tu kiem (exit 3): du nguoi x cua so; G3.1 du 5 seed; G3.1 - tt-CCA goc tai tao dung muc 21b (results/g31_baselines_analysis_2026_10_18).
Dung: python analyze_review26.py --out_dir <dir> [--bl_dir ...] [--golden_dir ...] [--ckpt_res ...]   (OVERWRITE=1 de ghi de)
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
from analyze_baselines import G31, MT, NSUB, PRIMARY, WINDOWS, load  # noqa: E402
from analyze_h1 import boot_ci, fail, holm, read_rows  # noqa: E402
from ssvep.metrics import itr_bpm  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJ, "results")
OLD_BL = os.path.join(RES, "baselines_tf_2026_10_15")
OLD_21B = os.path.join(RES, "g31_baselines_analysis_2026_10_18", "baselines_tests.csv")
FBCD = {"benchmark": None, "beta": None, "nakanishi": os.path.join(RES, "naka_dc_2026_10_15")}   # None = --bl_dir
NAMES = {"ttcca": "tt-CCA (gốc)", "ttcca_fb5cd": "tt-CCA + bank lọc, trừ TB", "huang_fb5cd": "DG-transfer (nhân quả, trừ TB)",
         "fbcca_fb5cd": "FBCCA (nhân quả, trừ TB)"}


def test(g, b, rng):
    dd = g - b
    lo, hi = boot_ci(dd, rng)
    p = 1.0 if np.allclose(dd, 0) else float(wilcoxon(g, b, alternative="two-sided").pvalue)
    return dict(mean_g3=g.mean(), mean_b=b.mean(), chenh=dd.mean(), lo=lo, hi=hi, g3_hon=int((dd > 0).sum()),
                b_hon=int((dd < 0).sum()), n=len(dd), p=p)


def write_csv(path, rows, keys):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{r[k]:.6g}" if isinstance(r[k], float) else r[k]) for k in keys})


def part_a(bl_dir, rng, L):
    rows, desc = [], []
    for ds in NSUB:
        g31 = load(sorted(glob.glob(os.path.join(G31[ds], f"wp3_g3_dc_{ds}_w*_s*_per_subject.csv"))), ds, 5, method="wp3_g3_dc")
        bl = {"ttcca": load([os.path.join(OLD_BL, f"ttcca_{ds}_per_subject.csv")], ds, 1, method="ttcca")}
        for m in ("ttcca_fb5cd", "huang_fb5cd"):
            bl[m] = load([os.path.join(bl_dir, f"{m}_{ds}_per_subject.csv")], ds, 1, method=m)
        bl["fbcca_fb5cd"] = load([os.path.join(FBCD[ds] or bl_dir, f"fbcca_{ds}_fb5cd_per_subject.csv")], ds, 1, tag="fb5cd")
        for name, d in {"G3.1 (LOSO)": g31, **{NAMES[m]: bl[m] for m in bl}}.items():
            for w in WINDOWS:
                itr = np.mean([itr_bpm(p / 100, MT[ds], w, 0.5) for p in d[w]])
                desc.append(dict(dataset=ds, method=name, window=w, acc=d[w].mean(), itr=itr))
        for w in PRIMARY:
            fam = [dict(dataset=ds, window=w, so_sanh=f"G3.1 − {NAMES[m]}", **test(g31[w], bl[m][w], rng)) for m in NAMES]
            for r, pa in zip(fam, holm([r["p"] for r in fam])):
                r.update(p_holm=pa, dat=bool(pa < 0.05 and r["chenh"] > 0), b_hon_yn=bool(pa < 0.05 and r["chenh"] < 0))
            rows += fam
    # tu kiem: G3.1 - tt-CCA goc trung muc 21b (chenh, p; KTC khac vi thu tu bootstrap)
    old = {(r["dataset"], round(float(r["window"]), 1)): r for r in read_rows(OLD_21B) if "gốc" in r["so_sanh"]}
    for r in rows:
        if r["so_sanh"].endswith(NAMES["ttcca"]):
            o = old[(r["dataset"], round(r["window"], 1))]
            if abs(float(o["chenh"]) - r["chenh"]) > 1e-4 or abs(float(o["p"]) - r["p"]) > 1e-4 * max(r["p"], 1e-12):
                fail(f"tai tao 21b {r['dataset']} {r['window']}: {o['chenh']} vs {r['chenh']}, p {o['p']} vs {r['p']}")
    print("[tu kiem] G3.1 − tt-CCA gốc tái tạo đúng mục 21b (chênh, p)", flush=True)
    for r, pa in zip(rows, holm([r["p"] for r in rows])):                                      # do nhay: Holm tren 24
        r["p_holm24"] = pa
    L += ["# Mục 26a: G3.1 so với đối chứng có trừ trung bình đoạn trước kích thích", "",
          "Accuracy TB theo người (%); ITR Tg = 0,5 s (bits/phút) trong ngoặc. G3.1 = TB 5 seed; đối chứng tất định.", ""]
    for ds in NSUB:
        L += [f"## {ds}", "", "| Phương pháp | " + " | ".join(f"{w:.1f} s" for w in WINDOWS) + " |", "|---" * 6 + "|"]
        for name in ["G3.1 (LOSO)"] + list(NAMES.values()):
            cells = [next(f"{r['acc']:.2f} ({r['itr']:.1f})" for r in desc if r["dataset"] == ds and r["method"] == name
                          and r["window"] == w) for w in WINDOWS]
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L.append("")
    n_pass, n_bl = sum(r["dat"] for r in rows), sum(r["b_hon_yn"] for r in rows)
    L += ["## Họ kiểm định chính (G3.1 − đối chứng; Wilcoxon hai phía; Holm trong họ 4 mỗi bộ × cửa sổ)", "",
          f"**G3.1 cao hơn có ý nghĩa ở {n_pass}/{len(rows)} phép; đối chứng cao hơn có ý nghĩa ở {n_bl}/{len(rows)} phép.**", "",
          "| Bộ | Cửa sổ | So sánh | G3.1 | Đối chứng | Chênh [KTC 95%] | G3.1 hơn | p | p Holm (4) | p Holm (24) | Đạt |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['dataset']} | {r['window']:.1f} | {r['so_sanh']} | {r['mean_g3']:.2f} | {r['mean_b']:.2f} | "
                 f"{r['chenh']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] | {r['g3_hon']}/{r['n']} | {r['p']:.3g} | "
                 f"{r['p_holm']:.3g} | {r['p_holm24']:.3g} | {'có' if r['dat'] else ('ngược' if r['b_hon_yn'] else 'không')} |")
    n24 = sum(r["p_holm24"] < 0.05 and r["chenh"] > 0 for r in rows)
    L += ["", f"Độ nhạy (mô tả): Holm trên cả 24 phép, G3.1 cao hơn có ý nghĩa ở {n24}/24 phép.", ""]
    return rows, desc


def part_b(golden_dir, ckpt_res, L):
    L += ["# Mục 26b: mô hình số nguyên (golden) so với QAT, mọi bộ dữ liệu", ""]
    gate = os.path.join(ckpt_res, "gate_R.csv")
    if os.path.exists(gate):
        gr = read_rows(gate)
        bad = [r for r in gr if r["khop"] != "1"]
        L += [f"Cổng tái lập R: {len(gr) - len(bad)}/{len(gr)} tệp (bộ × cửa sổ × seed) trùng kết quả đã báo "
              f"(|chênh| tối đa {max(float(r['max_abs_diff']) for r in gr):.2g}).", ""]
    else:
        L += ["Cổng tái lập R: CHƯA CÓ gate_R.csv.", ""]
    rows = []
    L += ["| Bộ | Cửa sổ | Fold × seed | QAT (%) | Golden (%) | Chênh golden − QAT | Trong ±1 điểm | Khớp quyết định (%) | "
          "SNR IIR min (dB) | Mẫu bão hòa (%) | Lượt thử có bão hòa (%) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds in NSUB:
        f = os.path.join(golden_dir, ds, "golden_eval.csv")
        if not os.path.exists(f):
            L.append(f"| {ds} | – | chưa có | | | | | | | | |")
            continue
        g, sat = read_rows(f), read_rows(os.path.join(golden_dir, ds, "saturation.csv"))
        for w in sorted({r["window"] for r in g}):
            rr = [r for r in g if r["window"] == w]
            by = defaultdict(lambda: defaultdict(list))
            for r in rr:
                for k in ("acc_qat", "acc_golden", "agree"):
                    by[k][int(r["fold"])].append(float(r[k]))
            subj = {k: np.array([np.mean(v) for _, v in sorted(by[k].items())]) * 100 for k in by}
            if len(subj["acc_qat"]) != NSUB[ds] or any(len(v) != 5 for v in by["acc_qat"].values()):
                fail(f"golden {ds} w={w}: thieu nguoi/seed")
            d = subj["acc_golden"].mean() - subj["acc_qat"].mean()
            sr = [r for r in sat if r["window"] == w]
            ps = 100 * sum(int(r["n_sat"]) for r in sr) / sum(int(r["n_samples"]) for r in sr)
            pt = 100 * sum(int(r["n_trials_sat"]) for r in sr) / sum(int(r["n_trials"]) for r in sr)
            row = dict(dataset=ds, window=w, n=len(rr), qat=subj["acc_qat"].mean(), golden=subj["acc_golden"].mean(), chenh=d,
                       trong_1=bool(abs(d) <= 1.0), agree=subj["agree"].mean(),
                       snr_min=min(float(r["snr_min_db"]) for r in rr), sat_pct=ps, sat_trial_pct=pt)
            rows.append(row)
            L.append(f"| {ds} | {w} s | {row['n']} | {row['qat']:.2f} | {row['golden']:.2f} | {d:+.2f} | "
                     f"{'có' if row['trong_1'] else 'KHÔNG'} | {row['agree']:.1f} | {row['snr_min']:.1f} | {ps:.4f} | {pt:.2f} |")
    L += ["", "Tiêu chí mô tả (chốt trước, như mục 11): chênh TB trong ±1 điểm; không kiểm định thống kê. "
          "Golden ở BETA, Nakanishi là mô hình tham chiếu, không phải bo mạch.", ""]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--bl_dir", default=os.path.join(RES, "baselines_dc_2026_10_28"))
    ap.add_argument("--golden_dir", default=os.path.join(RES, "golden_g31_all_2026_10_28"))
    ap.add_argument("--ckpt_res", default=os.path.join(RES, "g31_ckpt_all_2026_10_28"))
    a = ap.parse_args()
    out_csv = os.path.join(a.out_dir, "review26_tests.csv")
    if os.path.exists(out_csv) and os.environ.get("OVERWRITE") != "1":
        fail(f"{out_csv} da ton tai; OVERWRITE=1 de ghi de")
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(20261028)
    L = []
    rows, desc = part_a(a.bl_dir, rng, L)
    grows = part_b(a.golden_dir, a.ckpt_res, L)
    write_csv(out_csv, rows, ["dataset", "window", "so_sanh", "n", "mean_g3", "mean_b", "chenh", "lo", "hi", "g3_hon", "b_hon",
                              "p", "p_holm", "p_holm24", "dat", "b_hon_yn"])
    write_csv(os.path.join(a.out_dir, "review26_desc.csv"), desc, ["dataset", "method", "window", "acc", "itr"])
    if grows:
        write_csv(os.path.join(a.out_dir, "review26_golden.csv"), grows, list(grows[0]))
    open(os.path.join(a.out_dir, "review26_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("DONE")


if __name__ == "__main__":
    main()
