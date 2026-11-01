"""Phan tich H2  (chot 2026-10-11).

H2a (khong-kem-hon, mot phia): lai - FBCCA > -1 diem %, Wilcoxon signed-rank mot phia tren (chenh + 1).
H2b (hai phia): lai so voi ung vien thuan hoc sau (H-D).
Holm: ho = {H2a, H2b} trong cung mot cua so. Cua so chinh 0,4 va 1,0 s; cac cua so khac mo ta.
Mo ta them: lai so voi 4 doi chuan WP1 (job 15799), Holm trong ho 4 moi cua so.
Tu kiem (exit 3): tai tao FBCCA @1,0 s cua G0 (66,04 +- 3,72); du 35 nguoi x 5 seed x 5 cua so cho
moi ung vien; n khop FBCCA.

Dung: python analyze_h2.py --loso_dir <ket_qua_loso> --out_dir <thu_muc>   (OVERWRITE=1 de ghi de)
"""
import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_h1 import boot_ci, fail, holm, load_fbcca, mean_sem, read_rows  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WP1 = os.path.join(PROJ, "results", "wp1_baselines_benchmark_2026_10_02")
CANDS = ["wp2_hybrid", "wp2_pure"]
BASE = ["eegnet", "compactcnn", "fbtcnn", "guney"]
SEEDS, WINDOWS, PRIMARY = [0, 1, 2, 3, 4], [0.2, 0.4, 0.6, 0.8, 1.0], {0.4, 1.0}
MARGIN, ALPHA, N_SUBJ = 1.0, 0.05, 35
import analyze_h1  # noqa: E402

analyze_h1.N_BOOT = 10000


def load_dir(d, model):
    acc, n, cost = defaultdict(lambda: defaultdict(dict)), defaultdict(dict), {}
    for fn in sorted(os.listdir(d)):
        if fn.startswith(f"{model}_benchmark_w") and fn.endswith("_per_subject.csv"):
            for r in read_rows(os.path.join(d, fn)):
                if r["tag"] != "base":
                    continue
                w, s, sd = round(float(r["window"]), 2), int(r["subject"]), int(r["seed"])
                if sd in acc[w][s]:
                    fail(f"trung lap {model} w={w} s={s} seed={sd}")
                acc[w][s][sd], n[w][s] = float(r["acc"]), int(r["n"])
        if fn.startswith(f"{model}_benchmark_w") and fn.endswith("_timing.csv"):
            for r in read_rows(os.path.join(d, fn)):
                cost[round(float(r["window"]), 2)] = (int(r["params"]), int(r["macs"]))
    return acc, n, cost


def per_sub(acc_w):
    subs = list(range(1, N_SUBJ + 1))
    return np.array([100 * np.mean([acc_w[s][sd] for sd in SEEDS]) for s in subs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loso_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    out_csv, out_md = os.path.join(a.out_dir, "h2_tests.csv"), os.path.join(a.out_dir, "h2_report.md")
    if os.path.exists(out_csv) and os.environ.get("OVERWRITE") != "1":
        fail(f"{out_csv} da ton tai; OVERWRITE=1 de ghi de")
    os.makedirs(a.out_dir, exist_ok=True)

    # ---------- tu kiem ----------
    fb, fbn = load_fbcca("benchmark")
    m1, s1 = mean_sem([100 * fb[1.0][s] for s in fb[1.0]])
    print(f"[tu kiem] FBCCA @1,0 s = {m1:.2f} +- {s1:.2f} (G0 66.04 +- 3.72)")
    if abs(m1 - 66.04) > 0.006 or abs(s1 - 3.72) > 0.006:
        fail("khong tai tao duoc FBCCA")
    subs = list(range(1, N_SUBJ + 1))
    data, cost = {}, {}
    for m in CANDS + BASE:
        acc, n, c = load_dir(a.loso_dir if m in CANDS else WP1, m)
        for w in WINDOWS:
            if sorted(acc[w]) != subs or any(sorted(acc[w][s]) != SEEDS for s in subs):
                fail(f"{m} w={w}: thieu nguoi/seed")
            if any(n[w][s] != fbn[w][s] for s in subs):
                fail(f"{m} w={w}: n khong khop FBCCA")
        data[m], cost[m] = acc, c
    print("[tu kiem] DAT: 2 ung vien + 4 doi chuan x 5 cua so x 35 nguoi x 5 seed; n khop FBCCA", flush=True)

    from ssvep.wp2_models import cca_macs
    rng = np.random.default_rng(20261011)
    rows = []
    for w in WINDOWS:
        fbv = np.array([100 * fb[w][s] for s in subs])
        hy, pu = per_sub(data["wp2_hybrid"][w]), per_sub(data["wp2_pure"][w])
        T = int(round(w * 250))
        common = dict(window=f"{w:.1f}", vai_tro="H2 (xac nhan)" if w in PRIMARY else "mo ta phu",
                      hybrid_mean=hy.mean(), hybrid_sem=hy.std(ddof=1) / math.sqrt(N_SUBJ),
                      hybrid_params=cost["wp2_hybrid"][w][0], hybrid_macs=cost["wp2_hybrid"][w][1],
                      fbcca_macs_uoc_luong=cca_macs(T, 9, 40, 5, False))
        # H2a
        d = hy - fbv
        lo, hi = boot_ci(d, rng)
        r_a = wilcoxon(d + MARGIN, alternative="greater")
        # H2b
        d2 = hy - pu
        lo2, hi2 = boot_ci(d2, rng)
        r_b = wilcoxon(hy, pu, alternative="two-sided")
        fam = [dict(common, kiem_dinh="H2a lai vs FBCCA (khong-kem-hon, bien 1)", doi_chieu="fbcca",
                    doi_chieu_mean=fbv.mean(), chenh_tb=d.mean(), ktc95_lo=lo, ktc95_hi=hi,
                    so_nguoi_lai_hon=int((d > 0).sum()), p=float(r_a.pvalue)),
               dict(common, kiem_dinh="H2b lai vs thuan hoc sau (hai phia)", doi_chieu="wp2_pure",
                    doi_chieu_mean=pu.mean(), chenh_tb=d2.mean(), ktc95_lo=lo2, ktc95_hi=hi2,
                    so_nguoi_lai_hon=int((d2 > 0).sum()), p=float(r_b.pvalue))]
        for r, pa in zip(fam, holm([r["p"] for r in fam])):
            r["p_holm"], r["dat"] = pa, pa < ALPHA
        rows += fam
        # mo ta: lai vs 4 doi chuan WP1
        desc = []
        for b in BASE:
            bv = per_sub(data[b][w])
            db = hy - bv
            lo3, hi3 = boot_ci(db, rng)
            desc.append(dict(common, kiem_dinh="mo ta: lai vs doi chuan WP1", doi_chieu=b,
                             doi_chieu_mean=bv.mean(), chenh_tb=db.mean(), ktc95_lo=lo3, ktc95_hi=hi3,
                             so_nguoi_lai_hon=int((db > 0).sum()),
                             p=float(wilcoxon(hy, bv, alternative="two-sided").pvalue)))
        for r, pa in zip(desc, holm([r["p"] for r in desc])):
            r["p_holm"], r["dat"] = pa, pa < ALPHA
        rows += desc

    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})

    L = ["# Phân tích H2 — ứng viên WP2 trên Benchmark (LOSO không hiệu chuẩn, 5 seed)", "",
         "Script: `scripts/analyze_h2.py`.",
         "Lai = H-C #21; thuần học sâu = H-D #34 (results/wp2_search_2026_10_08/selection.json).",
         "H2a: Wilcoxon một phía trên (lai − FBCCA + 1); H2b: Wilcoxon hai phía lai vs thuần học sâu;",
         "Holm trong {H2a, H2b} mỗi cửa sổ. KTC 95% bootstrap theo người (10 000 lần). Chênh lệch = lai − đối chiếu.",
         "Lưu ý: ứng viên được chọn trên chính 35 người Benchmark → số LOSO có thiên lệch lạc quan.", ""]
    for w in WINDOWS:
        sub = [r for r in rows if r["window"] == f"{w:.1f}"]
        c = sub[0]
        L += [f"## {w:.1f} s — {c['vai_tro']}", "",
              f"Lai: {c['hybrid_mean']:.2f} ± {c['hybrid_sem']:.2f}% ; {c['hybrid_params']} tham số; "
              f"{c['hybrid_macs'] / 1e6:.3f} M MACs (FBCCA ước lượng {c['fbcca_macs_uoc_luong'] / 1e6:.2f} M).", "",
              "| Kiểm định | Đối chiếu | Acc đối chiếu | Chênh TB [KTC 95%] | Số người lai hơn | p | p Holm | Đạt |",
              "|---|---|---|---|---|---|---|---|"]
        for r in sub:
            L.append(f"| {r['kiem_dinh']} | {r['doi_chieu']} | {r['doi_chieu_mean']:.2f} | {r['chenh_tb']:+.2f} "
                     f"[{r['ktc95_lo']:+.2f}, {r['ktc95_hi']:+.2f}] | {r['so_nguoi_lai_hon']}/{N_SUBJ} | "
                     f"{r['p']:.3g} | {r['p_holm']:.3g} | {'có' if r['dat'] else 'không'} |")
        L.append("")
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"ghi {out_csv} ({len(rows)} dong, dich 30) va {out_md}")
    print("DONE")


if __name__ == "__main__":
    main()
