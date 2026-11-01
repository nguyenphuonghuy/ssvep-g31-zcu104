"""Phan tich moc G3 theo  (chot 2026-10-13, truoc khi chay job 17026).

Phan: G3-dev (Benchmark LOSO), G3-BETA (LOSO), G3-cross (Benchmark -> BETA), G3-Nakanishi (LOSO).
Moi phan, moi cua so: (i) khong-kem-hon G3 - FBCCA nhan qua > -1 diem (Wilcoxon mot phia tren chenh + 1);
(ii) G3 vs FBCCA nhan qua hai phia; Holm trong {(i), (ii)}. Cua so chinh 0,4 va 1,0 s.
Mo ta: G3-dev - lai FP32 filtfilt (job 17000); G3-BETA - 4 doi chuan WP1 tren BETA (job 15907), Holm trong ho 4.
Tu kiem (exit 3): du nguoi x 5 seed x 5 cua so moi phan; FBCCA filtfilt trung G0 tung nguoi; n khop.
Dung: python analyze_g3.py --g3_dir <dir> --out_dir <dir>   (OVERWRITE=1 de ghi de)
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
FP32 = os.path.join(RES, "wp2_loso_benchmark_2026_10_11")
WP1_BETA = os.path.join(RES, "wp1_baselines_beta_2026_10_04")
G0 = os.path.join(RES, "g0_trainfree_2026_10_02")
SEEDS, WINDOWS, PRIMARY = [0, 1, 2, 3, 4], [0.2, 0.4, 0.6, 0.8, 1.0], {0.4, 1.0}
NSUB = {"benchmark": 35, "beta": 70, "nakanishi": 10}
PARTS = [("G3-dev", "benchmark", "wp3_g3_benchmark_w*_s*_per_subject.csv"),
         ("G3-BETA", "beta", "wp3_g3_beta_w*_s*_per_subject.csv"),
         ("G3-cross", "beta", "g3_cross_beta_w*_s*_per_subject.csv"),
         ("G3-Nakanishi", "nakanishi", "wp3_g3_nakanishi_w*_s*_per_subject.csv")]
BASE = ["eegnet", "compactcnn", "fbtcnn", "guney"]


def load(paths, n_sub, method=None, seeds=SEEDS, tag="base"):
    acc, n = defaultdict(lambda: defaultdict(dict)), defaultdict(dict)
    for p in paths:
        for r in read_rows(p):
            if (tag and r.get("tag", "base") != tag) or (method and r["method"] != method):
                continue
            w, s, sd = round(float(r["window"]), 2), int(r["subject"]), int(r["seed"])
            if sd in acc[w][s]:
                fail(f"trung lap {p} w={w} s={s} sd={sd}")
            acc[w][s][sd], n[w][s] = float(r["acc"]), int(r["n"])
    subs = list(range(1, n_sub + 1))
    out = {}
    for w in WINDOWS:
        if sorted(acc[w]) != subs or any(sorted(acc[w][s]) != seeds for s in subs):
            fail(f"{paths[:1]}... w={w}: thieu nguoi/seed")
        out[w] = np.array([100 * np.mean([acc[w][s][sd] for sd in seeds]) for s in subs])
    return out, n


def fbcca(g3_dir, ds, prep):
    p = os.path.join(g3_dir, f"fbcca_{ds}_{prep}_per_subject.csv")
    return load([p], NSUB[ds], seeds=[0], tag=prep)


def test_pair(a, b, rng, noninf=False):
    d = a - b
    lo, hi = boot_ci(d, rng)
    if noninf:
        p = float(wilcoxon(d + 1.0, alternative="greater").pvalue)
    else:
        p = 1.0 if np.allclose(d, 0) else float(wilcoxon(a, b, alternative="two-sided").pvalue)
    return dict(mean_a=a.mean(), mean_b=b.mean(), chenh_tb=d.mean(), ktc95_lo=lo, ktc95_hi=hi,
                so_nguoi_a_hon=int((d > 0).sum()), n=len(d), p=p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g3_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    out_csv, out_md = os.path.join(a.out_dir, "g3_tests.csv"), os.path.join(a.out_dir, "g3_report.md")
    if os.path.exists(out_csv) and os.environ.get("OVERWRITE") != "1":
        fail(f"{out_csv} da ton tai; OVERWRITE=1 de ghi de")
    os.makedirs(a.out_dir, exist_ok=True)

    # ---------- tu kiem ----------
    fb = {}
    for ds in NSUB:
        ff, nf = fbcca(a.g3_dir, ds, "fb5")
        g = defaultdict(dict)
        for r in read_rows(os.path.join(G0, f"fbcca_{ds}_per_subject.csv")):
            g[round(float(r["window"]), 2)][int(r["subject"])] = 100 * float(r["acc"])
        for w in WINDOWS:
            if not np.allclose(ff[w], [g[w][s] for s in range(1, NSUB[ds] + 1)], atol=1e-3):
                fail(f"FBCCA filtfilt {ds} w={w} khong trung G0")
        fc, nc = fbcca(a.g3_dir, ds, "fb5c")
        fb[ds] = (ff, fc, nc)
    parts = {}
    for name, ds, pat in PARTS:
        acc, n = load(sorted(glob.glob(os.path.join(a.g3_dir, pat))), NSUB[ds])
        for w in WINDOWS:
            if any(n[w][s] != fb[ds][2][w][s] for s in range(1, NSUB[ds] + 1)):
                fail(f"{name} w={w}: n khong khop FBCCA")
        parts[name] = (ds, acc)
    fp32, _ = load(sorted(glob.glob(os.path.join(FP32, "wp2_hybrid_benchmark_w*_per_subject.csv"))), 35)
    base = {m: load(sorted(glob.glob(os.path.join(WP1_BETA, f"{m}_beta_w*_per_subject.csv"))), 70)[0] for m in BASE}
    print("[tu kiem] DAT: 4 phan G3 du nguoi x 5 seed x 5 cua so; FBCCA filtfilt trung G0 (3 bo); n khop", flush=True)

    rng = np.random.default_rng(20261014)
    rows = []
    for name, (ds, acc) in parts.items():
        for w in WINDOWS:
            role = "chinh" if w in PRIMARY else "mo ta"
            fc = fb[ds][1][w]
            fam = [dict(test_pair(acc[w], fc, rng, noninf=True), kiem_dinh="khong-kem-hon (bien 1)"),
                   dict(test_pair(acc[w], fc, rng), kiem_dinh="hai phia")]
            for r, pa in zip(fam, holm([r["p"] for r in fam])):
                r.update(phan=name, window=f"{w:.1f}", vai_tro=role, b="FBCCA nhan qua", p_holm=pa,
                         dat=pa < 0.05, fbcca_filtfilt=fb[ds][0][w].mean())
            rows += fam
            if name == "G3-dev":
                r = test_pair(acc[w], fp32[w], rng)
                r.update(phan=name, window=f"{w:.1f}", vai_tro="mo ta", kiem_dinh="hai phia",
                         b="lai FP32 filtfilt (job 17000)", p_holm=r["p"], dat=r["p"] < 0.05,
                         fbcca_filtfilt=fb[ds][0][w].mean())
                rows.append(r)
            if name == "G3-BETA":
                fam = [dict(test_pair(acc[w], base[m][w], rng), b=f"{m} (WP1 BETA, FP32 filtfilt)") for m in BASE]
                for r, pa in zip(fam, holm([r["p"] for r in fam])):
                    r.update(phan=name, window=f"{w:.1f}", vai_tro="mo ta", kiem_dinh="hai phia", p_holm=pa,
                             dat=pa < 0.05, fbcca_filtfilt=fb[ds][0][w].mean())
                rows += fam
    keys = ["phan", "window", "vai_tro", "kiem_dinh", "b", "n", "mean_a", "mean_b", "fbcca_filtfilt", "chenh_tb",
            "ktc95_lo", "ktc95_hi", "so_nguoi_a_hon", "p", "p_holm", "dat"]
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{r[k]:.6g}" if isinstance(r[k], float) else r[k]) for k in keys})

    L = ["# Mốc G3 — kiểm tra cấu hình đóng băng (lai H-C #21 + QAT 8 bit + lọc nhân quả)", "",
         "Script `scripts/analyze_g3.py`.",
         "A = G3; chênh = A − B (điểm %), TB 5 seed theo người; KTC 95% bootstrap theo người (10 000 lần).",
         "Không-kém-hơn: Wilcoxon một phía trên (chênh + 1); hai phía: Wilcoxon; Holm trong 2 phép mỗi (phần, cửa sổ).",
         "G3-cross: lưu ý lệch mốc thời gian/pha ~40 ms giữa BETA và Benchmark (job 17025).", "",
         "## Accuracy TB theo người (%)", "", "| Phần | " + " | ".join(f"{w:.1f} s" for w in WINDOWS) + " |",
         "|---" * 6 + "|"]
    for name, (ds, acc) in parts.items():
        L.append(f"| {name} (n={NSUB[ds]}) | " + " | ".join(f"{acc[w].mean():.2f}" for w in WINDOWS) + " |")
    for ds in NSUB:
        L.append(f"| FBCCA nhân quả {ds} | " + " | ".join(f"{fb[ds][1][w].mean():.2f}" for w in WINDOWS) + " |")
        L.append(f"| FBCCA filtfilt {ds} | " + " | ".join(f"{fb[ds][0][w].mean():.2f}" for w in WINDOWS) + " |")
    L += ["", "## Kiểm định", "", "| Phần | Cửa sổ | Kiểm định | B | Acc A | Acc B | Chênh [KTC 95%] | Số người A > B | p | p Holm | Đạt |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['phan']} | {r['window']} ({r['vai_tro']}) | {r['kiem_dinh']} | {r['b']} | {r['mean_a']:.2f} | "
                 f"{r['mean_b']:.2f} | {r['chenh_tb']:+.2f} [{r['ktc95_lo']:+.2f}, {r['ktc95_hi']:+.2f}] | "
                 f"{r['so_nguoi_a_hon']}/{r['n']} | {r['p']:.3g} | {r['p_holm']:.3g} | {'có' if r['dat'] else 'không'} |")
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"ghi {out_csv} ({len(rows)} dong) va {out_md}")
    print("DONE")


if __name__ == "__main__":
    main()
