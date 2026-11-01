"""Phan tich H1.

H1 (xac nhan): Benchmark, LOSO khong hieu chuan, accuracy tung mo hinh hoc sau KHAC FBCCA o 0,4 s
va 1,0 s. BETA chay cung phuong phap nhung la phan tich BO SUNG, NGOAI H1 (H1 chi phat bieu
H1 cho Benchmark). Cac cua so khac 0,4 s va 1,0 s chi la mo ta phu.

Theo
- don vi quan sat: nguoi; accuracy hoc sau cua moi nguoi = trung binh 5 seed (0-4);
- Wilcoxon signed-rank ghep cap theo nguoi, hai phia, alpha = 0,05;
- Holm cho 4 mo hinh trong cung mot bang (moi dataset x moi cua so la mot ho kiem dinh, vi
  "moi do dai cua so duoc kiem dinh rieng");
- co hieu ung: chenh lech trung binh (diem %) + KTC 95% bootstrap theo nguoi, 10 000 lan;
- bao cao them do lech chuan giua cac seed cua accuracy trung binh toan bo.

Buoc tu kiem (exit 3 neu khong dat): tai tao FBCCA @1,0 s cua G0_REPORT.md (Benchmark 66,04 +- 3,72;
BETA 56,62 +- 2,62, trung binh +- SEM, diem %); moi (mo hinh, cua so, nguoi) du dung 5 seed; so trial
n khop giua hoc sau va FBCCA.

Dung: python analyze_h1.py --out_dir <thu_muc>   (OVERWRITE=1 de ghi de)
"""
import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJ, "results")
DL_DIR = {"benchmark": os.path.join(RES, "wp1_baselines_benchmark_2026_10_02"),
          "beta": os.path.join(RES, "wp1_baselines_beta_2026_10_04")}
FB_CSV = {d: os.path.join(RES, "g0_trainfree_2026_10_02", f"fbcca_{d}_per_subject.csv")
          for d in DL_DIR}
N_SUBJ = {"benchmark": 35, "beta": 70}
G0_FBCCA_1S = {"benchmark": (66.04, 3.72), "beta": (56.62, 2.62)}   # G0_REPORT.md, diem %
MODELS = ["eegnet", "compactcnn", "fbtcnn", "guney"]
SEEDS = [0, 1, 2, 3, 4]
WINDOWS = [0.2, 0.4, 0.6, 0.8, 1.0]
PRIMARY = {0.4, 1.0}
N_BOOT, BOOT_SEED, ALPHA = 10000, 20261007, 0.05


def fail(msg):
    print("TU KIEM KHONG DAT:", msg, flush=True)
    sys.exit(3)


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_fbcca(ds):
    acc, n = defaultdict(dict), defaultdict(dict)
    for r in read_rows(FB_CSV[ds]):
        w = round(float(r["window"]), 2)
        acc[w][int(r["subject"])] = float(r["acc"])
        n[w][int(r["subject"])] = int(r["n"])
    return acc, n


def load_dl(ds, model):
    """acc[w][subject][seed], n[w][subject]"""
    acc = defaultdict(lambda: defaultdict(dict))
    n = defaultdict(dict)
    files = [f for f in os.listdir(DL_DIR[ds])
             if f.startswith(f"{model}_{ds}_w") and f.endswith("_per_subject.csv")]
    for fn in sorted(files):
        for r in read_rows(os.path.join(DL_DIR[ds], fn)):
            w, s, sd = round(float(r["window"]), 2), int(r["subject"]), int(r["seed"])
            if sd in acc[w][s]:
                fail(f"trung lap {ds} {model} w={w} subject={s} seed={sd}")
            acc[w][s][sd] = float(r["acc"])
            n[w][s] = int(r["n"])
    return acc, n


def mean_sem(v):
    v = np.asarray(v, float)
    return v.mean(), v.std(ddof=1) / math.sqrt(len(v))


def holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, run = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[i]))
        adj[i] = run
    return adj


def boot_ci(diff, rng):
    idx = rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
    means = diff[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    out_csv = os.path.join(a.out_dir, "h1_tests.csv")
    out_md = os.path.join(a.out_dir, "h1_report.md")
    if os.path.exists(out_csv) and os.environ.get("OVERWRITE") != "1":
        fail(f"{out_csv} da ton tai; dat OVERWRITE=1 neu muon ghi de")
    os.makedirs(a.out_dir, exist_ok=True)

    # ---------- 1. tu kiem ----------
    data = {}
    for ds in DL_DIR:
        fb, fbn = load_fbcca(ds)
        m1, s1 = mean_sem([100 * fb[1.0][s] for s in fb[1.0]])
        exp_m, exp_s = G0_FBCCA_1S[ds]
        print(f"[tu kiem] FBCCA {ds} @1,0 s = {m1:.2f} +- {s1:.2f} (G0_REPORT {exp_m} +- {exp_s})")
        if abs(m1 - exp_m) > 0.006 or abs(s1 - exp_s) > 0.006:
            fail(f"khong tai tao duoc FBCCA {ds}")
        dl = {}
        for mdl in MODELS:
            acc, n = load_dl(ds, mdl)
            for w in WINDOWS:
                subs = sorted(acc[w])
                if subs != list(range(1, N_SUBJ[ds] + 1)) or sorted(fb[w]) != subs:
                    fail(f"{ds} {mdl} w={w}: danh sach nguoi khong du/khong khop")
                for s in subs:
                    if sorted(acc[w][s]) != SEEDS:
                        fail(f"{ds} {mdl} w={w} subject={s}: seed {sorted(acc[w][s])}")
                    if n[w][s] != fbn[w][s]:
                        fail(f"{ds} {mdl} w={w} subject={s}: n {n[w][s]} != FBCCA {fbn[w][s]}")
            dl[mdl] = acc
        data[ds] = (fb, dl)
        print(f"[tu kiem] {ds}: {len(MODELS)} mo hinh x {len(WINDOWS)} cua so x "
              f"{N_SUBJ[ds]} nguoi x 5 seed day du, n khop FBCCA")
    print("[tu kiem] DAT", flush=True)

    # ---------- 2. kiem dinh ----------
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for ds, (fb, dl) in data.items():
        for w in WINDOWS:
            subs = sorted(fb[w])
            fbv = np.array([100 * fb[w][s] for s in subs])
            fam = []
            for mdl in MODELS:
                per_sub = np.array([100 * np.mean([dl[mdl][w][s][sd] for sd in SEEDS]) for s in subs])
                seed_means = [100 * np.mean([dl[mdl][w][s][sd] for s in subs]) for sd in SEEDS]
                diff = per_sub - fbv
                res = wilcoxon(per_sub, fbv, alternative="two-sided")
                lo, hi = boot_ci(diff, rng)
                dm, dsem = mean_sem(per_sub)
                fm, fsem = mean_sem(fbv)
                fam.append(dict(
                    dataset=ds, window=f"{w:.1f}", model=mdl,
                    vai_tro=("H1 (xac nhan)" if ds == "benchmark" and w in PRIMARY else
                             "bo sung ngoai H1" if w in PRIMARY else "mo ta phu"),
                    n_nguoi=len(subs), dl_mean=dm, dl_sem=dsem,
                    dl_sd_giua_seed=float(np.std(seed_means, ddof=1)),
                    fbcca_mean=fm, fbcca_sem=fsem,
                    chenh_tb=float(diff.mean()), ktc95_lo=float(lo), ktc95_hi=float(hi),
                    chenh_trung_vi=float(np.median(diff)),
                    so_nguoi_dl_hon=int((diff > 0).sum()), so_nguoi_bang=int((diff == 0).sum()),
                    W=float(res.statistic), p=float(res.pvalue)))
            for r, padj in zip(fam, holm([r["p"] for r in fam])):
                r["p_holm"] = padj
                r["co_y_nghia_holm"] = padj < ALPHA
            rows.extend(fam)

    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})

    # ---------- 3. bao cao ----------
    L = ["# Phân tích H1 — WP1 đối chuẩn học sâu so với FBCCA",
         "",
         "Script: `scripts/analyze_h1.py`.",
         "Đơn vị: người; accuracy học sâu mỗi người = trung bình 5 seed. Wilcoxon signed-rank ghép cặp,",
         "hai phía; Holm cho 4 mô hình trong từng (dataset, cửa sổ); KTC 95% bootstrap theo người",
         f"({N_BOOT} lần, seed {BOOT_SEED}). Chênh lệch = học sâu − FBCCA, điểm %.",
         "",
         "- **Benchmark, 0,4 s và 1,0 s: H1 (kiểm định xác nhận).**",
         "- BETA: cùng phương pháp, **bổ sung, ngoài H1** .",
         "- Cửa sổ 0,2 / 0,6 / 0,8 s: mô tả phụ.",
         "- Lưu ý chưa xác minh: FBCCA dùng 5 dải con; số dải con trong bài gốc Chen 2015 chưa đối chiếu.",
         ""]
    for ds in DL_DIR:
        for w in WINDOWS:
            sub = [r for r in rows if r["dataset"] == ds and r["window"] == f"{w:.1f}"]
            L += [f"## {ds} — {w:.1f} s — {sub[0]['vai_tro']}", "",
                  f"FBCCA: {sub[0]['fbcca_mean']:.2f} ± {sub[0]['fbcca_sem']:.2f} (TB ± SEM, n = {sub[0]['n_nguoi']})",
                  "",
                  "| Mô hình | Acc TB ± SEM | SD giữa seed | Chênh TB [KTC 95%] | Trung vị chênh | Số người DL > FBCCA | W | p | p Holm |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for r in sub:
                L.append(f"| {r['model']} | {r['dl_mean']:.2f} ± {r['dl_sem']:.2f} | {r['dl_sd_giua_seed']:.2f} | "
                         f"{r['chenh_tb']:+.2f} [{r['ktc95_lo']:+.2f}, {r['ktc95_hi']:+.2f}] | "
                         f"{r['chenh_trung_vi']:+.2f} | {r['so_nguoi_dl_hon']}/{r['n_nguoi']} | {r['W']:.1f} | "
                         f"{r['p']:.3g} | {r['p_holm']:.3g}{' *' if r['co_y_nghia_holm'] else ''} |")
            L.append("")
    L.append("`*`: p Holm < 0,05.")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"ghi {out_csv} ({len(rows)} dong, dich {len(DL_DIR) * len(WINDOWS) * len(MODELS)}) va {out_md}")
    print("DONE")


if __name__ == "__main__":
    main()
