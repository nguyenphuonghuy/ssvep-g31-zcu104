"""Phan tich PTQ INT8 cua 4 doi chuan WP1 theo  (chot 2026-10-07).

- Accuracy moi nguoi = TB 5 seed, cho FP32 chay lai (tag base) va moi bien the PTQ
  (base_ptq_int8, base_ptq_int8_pot), cung mo hinh FP32 cua tung fold.
- Do sut = FP32 chay lai - PTQ (diem %), ghep cap theo nguoi; KTC 95% bootstrap 10 000 lan;
  Wilcoxon ghep cap hai phia; Holm trong ho 4 mo hinh moi (cua so, bien the).
- Do lap lai: FP32 chay lai - FP32 job 15799 (so chinh thuc), ghep cap theo nguoi.
Tu kiem (exit 3): du 35 nguoi x 5 seed x 3 tag cho moi (mo hinh, cua so); n khop job 15799.
--smoke: chi in TB, bo kiem dinh (dung cho ket qua smoke 1 fold).

Dung: python analyze_ptq.py --in_dir <ket_qua_ptq> --out_dir <thu_muc> [--smoke]
"""
import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIG = os.path.join(PROJ, "results", "wp1_baselines_benchmark_2026_10_02")
MODELS = ["eegnet", "compactcnn", "fbtcnn", "guney"]
WINDOWS = ["0.40", "1.00"]
TAGS = ["base", "base_ptq_int8", "base_ptq_int8_pot"]
SEEDS = [0, 1, 2, 3, 4]
N_SUBJ = 35
N_BOOT, BOOT_SEED, ALPHA, REF_DROP = 10000, 20261008, 0.05, 1.0


def fail(msg):
    print("TU KIEM KHONG DAT:", msg, flush=True)
    sys.exit(3)


def load(d, tags):
    """acc[(model, window, tag)][subject][seed], n[(model, window)][subject]"""
    acc = defaultdict(lambda: defaultdict(dict))
    n = defaultdict(dict)
    for fn in sorted(os.listdir(d)):
        if not (fn.endswith("_per_subject.csv") and fn.split("_")[0] in MODELS):
            continue
        with open(os.path.join(d, fn), newline="") as f:
            for r in csv.DictReader(f):
                if r["tag"] not in tags or r["window"] not in WINDOWS:
                    continue
                k = (r["method"], r["window"], r["tag"])
                s, sd = int(r["subject"]), int(r["seed"])
                if sd in acc[k][s]:
                    fail(f"trung lap {k} subject={s} seed={sd}")
                acc[k][s][sd] = float(r["acc"])
                n[(r["method"], r["window"])][s] = int(r["n"])
    return acc, n


def per_subject(acc_k, subs):
    return np.array([100 * np.mean([acc_k[s][sd] for sd in SEEDS]) for s in subs])


def holm(p):
    m, order, run, adj = len(p), sorted(range(len(p)), key=lambda i: p[i]), 0.0, [0.0] * len(p)
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def boot_ci(d, rng):
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    return np.percentile(d[idx].mean(axis=1), [2.5, 97.5])


def wilcoxon_p(a, b):
    from scipy.stats import wilcoxon
    if np.allclose(a, b):
        return float("nan"), 1.0          # moi chenh lech = 0: khong co bang chung khac biet
    r = wilcoxon(a, b, alternative="two-sided")
    return float(r.statistic), float(r.pvalue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    acc, n = load(a.in_dir, TAGS)

    if a.smoke:
        for k in sorted(acc):
            v = [x for s in acc[k] for x in acc[k][s].values()]
            print(f"[smoke] {k}: {len(acc[k])} nguoi, {len(v)} gia tri, TB {100 * np.mean(v):.2f}")
        print("DONE (smoke)")
        return

    out_csv = os.path.join(a.out_dir, "ptq_tests.csv")
    if os.path.exists(out_csv) and os.environ.get("OVERWRITE") != "1":
        fail(f"{out_csv} da ton tai; OVERWRITE=1 de ghi de")
    os.makedirs(a.out_dir, exist_ok=True)

    # ---------- 1. tu kiem ----------
    orig, orig_n = load(ORIG, ["base"])
    subs = list(range(1, N_SUBJ + 1))
    for m in MODELS:
        for w in WINDOWS:
            for t in TAGS:
                k = (m, w, t)
                if sorted(acc[k]) != subs or any(sorted(acc[k][s]) != SEEDS for s in subs):
                    fail(f"{k}: thieu nguoi/seed")
            if sorted(orig[(m, w, "base")]) != subs:
                fail(f"job 15799 {m} w={w}: thieu nguoi")
            if any(n[(m, w)][s] != orig_n[(m, w)][s] for s in subs):
                fail(f"{m} w={w}: n khong khop job 15799")
    print("[tu kiem] DAT: 4 mo hinh x 2 cua so x 3 tag x 35 nguoi x 5 seed, n khop job 15799", flush=True)

    # ---------- 2. kiem dinh ----------
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for w in WINDOWS:
        rep = {}
        for m in MODELS:
            fp = per_subject(acc[(m, w, "base")], subs)
            o = per_subject(orig[(m, w, "base")], subs)
            d = fp - o
            lo, hi = boot_ci(d, rng)
            rep[m] = (fp, o, d, lo, hi)
        for t in TAGS[1:]:
            fam = []
            for m in MODELS:
                fp, o, d_rep, rlo, rhi = rep[m]
                q = per_subject(acc[(m, w, t)], subs)
                drop = fp - q
                lo, hi = boot_ci(drop, rng)
                W, p = wilcoxon_p(fp, q)
                seed_drop = [100 * np.mean([acc[(m, w, "base")][s][sd] - acc[(m, w, t)][s][sd]
                                            for s in subs]) for sd in SEEDS]
                fam.append(dict(
                    window=w, variant=t.replace("base_", ""), model=m, n_nguoi=len(subs),
                    fp32_goc_mean=o.mean(), fp32_chay_lai_mean=fp.mean(),
                    lap_lai_chenh_tb=d_rep.mean(), lap_lai_ktc95_lo=rlo, lap_lai_ktc95_hi=rhi,
                    ptq_mean=q.mean(), ptq_sem=q.std(ddof=1) / math.sqrt(len(q)),
                    sut_tb=drop.mean(), sut_ktc95_lo=lo, sut_ktc95_hi=hi,
                    sut_sd_giua_seed=float(np.std(seed_drop, ddof=1)),
                    sut_max_nguoi=drop.max(), so_nguoi_sut=int((drop > 0).sum()),
                    W=W, p=p, duoi_nguong_1diem=bool(hi < REF_DROP)))
            for r, pa in zip(fam, holm([r["p"] for r in fam])):
                r["p_holm"], r["co_y_nghia_holm"] = pa, pa < ALPHA
            rows += fam

    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})

    L = ["# WP1 — PTQ INT8 của 4 đối chuẩn học sâu (Benchmark, LOSO không hiệu chuẩn)", "",
         "Script: `scripts/analyze_ptq.py`.",
         "Sụt = FP32 chạy lại − PTQ (điểm %), ghép cặp theo người (TB 5 seed). KTC 95% bootstrap "
         f"({N_BOOT} lần, seed {BOOT_SEED}); Wilcoxon hai phía; Holm trong 4 mô hình mỗi (cửa sổ, biến thể).",
         "Cột \"< 1 điểm\": cận trên KTC 95% của sụt < 1 điểm % (mô tả, không phải kiểm định không-kém-hơn).",
         "Lặp lại = FP32 chạy lại − FP32 job 15799 (số chính thức của WP1).", ""]
    for w in WINDOWS:
        for v in ("ptq_int8", "ptq_int8_pot"):
            L += [f"## Cửa sổ {w} s — {v}", "",
                  "| Mô hình | FP32 (15799) | FP32 chạy lại | Lặp lại [KTC] | PTQ ± SEM | Sụt TB [KTC 95%] | "
                  "SD sụt giữa seed | Số người sụt | p Holm | < 1 điểm |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for r in rows:
                if r["window"] == w and r["variant"] == v:
                    L.append(f"| {r['model']} | {r['fp32_goc_mean']:.2f} | {r['fp32_chay_lai_mean']:.2f} | "
                             f"{r['lap_lai_chenh_tb']:+.2f} [{r['lap_lai_ktc95_lo']:+.2f}, {r['lap_lai_ktc95_hi']:+.2f}] | "
                             f"{r['ptq_mean']:.2f} ± {r['ptq_sem']:.2f} | {r['sut_tb']:+.2f} "
                             f"[{r['sut_ktc95_lo']:+.2f}, {r['sut_ktc95_hi']:+.2f}] | {r['sut_sd_giua_seed']:.2f} | "
                             f"{r['so_nguoi_sut']}/{r['n_nguoi']} | {r['p_holm']:.3g}"
                             f"{' *' if r['co_y_nghia_holm'] else ''} | {'có' if r['duoi_nguong_1diem'] else 'không'} |")
            L.append("")
    L.append("`*`: p Holm < 0,05.")
    open(os.path.join(a.out_dir, "ptq_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"ghi {out_csv} ({len(rows)} dong, dich 16)")
    print("DONE")


if __name__ == "__main__":
    main()
