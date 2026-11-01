"""Phan tich Ensemble-DNN loc nhan qua (chot truoc khi chay).

Moi bo (Benchmark 35, BETA 70, Nakanishi 10), moi cua so chinh 0,4 / 1,0 s (accuracy moi nguoi = TB 5 seed):
  (a) G3.1 - Ensemble-DNN nhan qua; (b) Ensemble-DNN pha khong (muc 22) - Ensemble-DNN nhan qua;
  Wilcoxon hai phia, Holm trong ho 2; KTC 95% bootstrap theo nguoi (10 000).
Mo ta: khoang cach G3.1 - Ensemble-DNN (muc 22) va phan con lai khi Ensemble-DNN loc nhan qua; ITR (Tg 0,5 s).
Tu kiem (exit 3): du 30 tep, du nguoi x 5 seed; muc 22 va G3.1 du du lieu.
Dung: python analyze_ensdnn_causal.py --ens_dir <dir> --out_dir <dir>   (OVERWRITE=1 de ghi de)
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
RES = os.path.join(PROJ, "results")
ENS0 = os.path.join(RES, "ensemble_dnn_2026_10_18")
G31 = {"benchmark": os.path.join(RES, "g3dc_2026_10_17"), "beta": os.path.join(RES, "g3dc_2026_10_17"),
       "nakanishi": os.path.join(RES, "naka_dc_2026_10_15")}
PRIMARY, SEEDS = [0.4, 1.0], [0, 1, 2, 3, 4]
NSUB = {"benchmark": 35, "beta": 70, "nakanishi": 10}
MT = {"benchmark": 40, "beta": 40, "nakanishi": 12}


def load(paths, ds, method):
    acc = defaultdict(lambda: defaultdict(dict))
    for p in paths:
        for r in read_rows(p):
            if r["method"] == method:
                acc[round(float(r["window"]), 2)][int(r["subject"])][int(r["seed"])] = float(r["acc"])
    subs = list(range(1, NSUB[ds] + 1))
    out = {}
    for w in PRIMARY:
        if sorted(acc[w]) != subs or any(sorted(acc[w][s]) != SEEDS for s in subs):
            fail(f"{ds} {method} w={w}: thieu nguoi/seed")
        out[w] = np.array([100 * np.mean([acc[w][s][k] for k in SEEDS]) for s in subs])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ens_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    out_csv = os.path.join(a.out_dir, "ensdnnc_tests.csv")
    if os.path.exists(out_csv) and os.environ.get("OVERWRITE") != "1":
        fail(f"{out_csv} da ton tai; OVERWRITE=1 de ghi de")
    files = sorted(glob.glob(os.path.join(a.ens_dir, "ensdnnc_*_per_subject.csv")))
    if len(files) != 30:
        fail(f"can 30 tep ensdnnc, co {len(files)}")
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(20261024)
    rows, desc = [], []
    for ds in NSUB:
        EC = load(sorted(glob.glob(os.path.join(a.ens_dir, f"ensdnnc_{ds}_w*_s*_per_subject.csv"))), ds, "ensemble_dnn_c")
        E0 = load(sorted(glob.glob(os.path.join(ENS0, f"ensdnn_{ds}_w*_s*_per_subject.csv"))), ds, "ensemble_dnn")
        G = load(sorted(glob.glob(os.path.join(G31[ds], f"wp3_g3_dc_{ds}_w*_s*_per_subject.csv"))), ds, "wp3_g3_dc")
        for w in PRIMARY:
            fam = []
            for name, A, B in (("G3.1 − Ensemble-DNN nhân quả", G[w], EC[w]),
                               ("Ensemble-DNN pha không − Ensemble-DNN nhân quả", E0[w], EC[w])):
                d = A - B
                lo, hi = boot_ci(d, rng)
                p = 1.0 if np.allclose(d, 0) else float(wilcoxon(A, B, alternative="two-sided").pvalue)
                fam.append(dict(dataset=ds, window=w, so_sanh=name, n=len(d), mean_a=A.mean(), mean_b=B.mean(), chenh=d.mean(),
                                lo=lo, hi=hi, a_hon=int((d > 0).sum()), p=p))
            for r, pa in zip(fam, holm([r["p"] for r in fam])):
                r.update(p_holm=pa, dat=pa < 0.05)
            rows += fam
            gap0, gapc = (G[w] - E0[w]).mean(), (G[w] - EC[w]).mean()
            desc.append(dict(dataset=ds, window=w, g31=G[w].mean(), ens_zero=E0[w].mean(), ens_causal=EC[w].mean(),
                             gap_zero=gap0, gap_causal=gapc, frac_explained=(gap0 - gapc) / gap0 if gap0 else float("nan"),   # >0: phan khoang cach do loc
                             itr_g31=np.mean([itr_bpm(v / 100, MT[ds], w, 0.5) for v in G[w]]),
                             itr_ens_zero=np.mean([itr_bpm(v / 100, MT[ds], w, 0.5) for v in E0[w]]),
                             itr_ens_causal=np.mean([itr_bpm(v / 100, MT[ds], w, 0.5) for v in EC[w]])))
    print("[tu kiem] 30 tep; du nguoi x 5 seed: Ensemble-DNN nhan qua, Ensemble-DNN pha khong (muc 22), G3.1", flush=True)

    def dump(path, rr):
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rr[0]))
            wr.writeheader()
            for r in rr:
                wr.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    dump(out_csv, rows)
    dump(os.path.join(a.out_dir, "ensdnnc_desc.csv"), desc)
    L = ["# Ensemble-DNN lọc nhân quả", "", "Accuracy TB theo người (%), TB 5 seed; ITR Tg 0,5 s trong ngoặc.", "",
         "| Bộ | Cửa sổ | G3.1 | Ensemble-DNN pha không | Ensemble-DNN nhân quả | Khoảng cách G3.1 − pha không | "
         "G3.1 − nhân quả | Phần khoảng cách do lọc |", "|---|---|---|---|---|---|---|---|"]
    for r in desc:
        L.append(f"| {r['dataset']} | {r['window']:.1f} | {r['g31']:.2f} ({r['itr_g31']:.1f}) | {r['ens_zero']:.2f} "
                 f"({r['itr_ens_zero']:.1f}) | {r['ens_causal']:.2f} ({r['itr_ens_causal']:.1f}) | {r['gap_zero']:+.2f} | "
                 f"{r['gap_causal']:+.2f} | {100 * r['frac_explained']:.0f}% |")
    L += ["", "## Kiểm định (Wilcoxon hai phía, Holm trong họ 2 mỗi bộ × cửa sổ)", "",
          "| Bộ | Cửa sổ | So sánh | A | B | Chênh [KTC 95%] | A hơn | p | p Holm | Đạt |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['dataset']} | {r['window']:.1f} | {r['so_sanh']} | {r['mean_a']:.2f} | {r['mean_b']:.2f} | "
                 f"{r['chenh']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] | {r['a_hon']}/{r['n']} | {r['p']:.3g} | "
                 f"{r['p_holm']:.3g} | {'có' if r['dat'] else 'không'} |")
    open(os.path.join(a.out_dir, "ensdnnc_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("DONE")


if __name__ == "__main__":
    main()
