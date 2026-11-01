"""Khong dong nhat theo nguoi cua khoang cach G3.1 - Ensemble-DNN (chot truoc khi chay).

d_i = acc G3.1 - acc Ensemble-DNN (TB 5 seed, %). Do kho = do chinh xac FBCCA loc pha khong (tag fb5).
 (i)  Spearman rho(d, FBCCA) moi bo x cua so chinh, Holm trong ho 6; mo ta: d TB theo 3 nhom do kho, so nguoi d >= 0.
 (ii) Delta_i = d_i(1,0 s) - d_i(0,4 s), Wilcoxon hai phia moi bo, Holm trong ho 3; KTC 95% bootstrap.
Mo ta them: lap (i), (ii) voi Ensemble-DNN loc nhan qua (muc 24), khong kiem dinh.
Tu kiem (exit 3): du nguoi x 5 seed; d TB trung chenh trong ensdnn_tests.csv (muc 22).
Dung: python analyze_heterogeneity.py --out_dir <dir>   (OVERWRITE=1 de ghi de)
"""
import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_h1 import boot_ci, fail, holm, read_rows  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJ, "results")
G31 = {"benchmark": "g3dc_2026_10_17", "beta": "g3dc_2026_10_17", "nakanishi": "naka_dc_2026_10_15"}
PRIMARY, SEEDS = [0.4, 1.0], [0, 1, 2, 3, 4]
NSUB = {"benchmark": 35, "beta": 70, "nakanishi": 10}


def load(pattern, ds, method, tag=None, seeds=SEEDS):
    acc = defaultdict(lambda: defaultdict(dict))
    paths = sorted(glob.glob(os.path.join(RES, pattern)))
    for p in paths:
        for r in read_rows(p):
            if r["method"] == method and (tag is None or r.get("tag") == tag):
                acc[round(float(r["window"]), 2)][int(r["subject"])][int(r["seed"])] = float(r["acc"])
    subs = list(range(1, NSUB[ds] + 1))
    out = {}
    for w in PRIMARY:
        if sorted(acc[w]) != subs or any(sorted(acc[w][s]) != seeds for s in subs):
            fail(f"{pattern} {method} w={w}: thieu nguoi/seed")
        out[w] = np.array([100 * np.mean([acc[w][s][k] for k in seeds]) for s in subs])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    out_csv = os.path.join(a.out_dir, "heterogeneity_tests.csv")
    if os.path.exists(out_csv) and os.environ.get("OVERWRITE") != "1":
        fail(f"{out_csv} da ton tai; OVERWRITE=1 de ghi de")
    os.makedirs(a.out_dir, exist_ok=True)
    ref = {(r["dataset"], round(float(r["window"]), 2)): float(r["chenh"])
           for r in read_rows(os.path.join(RES, "ensdnn_analysis_2026_10_18", "ensdnn_tests.csv")) if r["so_sanh"].startswith("G3.1")}
    rng = np.random.default_rng(20261027)
    rho_rows, win_rows, desc = [], [], []
    for ds in NSUB:
        G = load(f"{G31[ds]}/wp3_g3_dc_{ds}_w*_s*_per_subject.csv", ds, "wp3_g3_dc")
        E = {"pha không": load(f"ensemble_dnn_2026_10_18/ensdnn_{ds}_w*_s*_per_subject.csv", ds, "ensemble_dnn"),
             "nhân quả": load(f"ensemble_dnn_causal_2026_10_24/ensdnnc_{ds}_w*_s*_per_subject.csv", ds, "ensemble_dnn_c")}
        F = load(f"g3_independent_2026_10_13/fbcca_{ds}_fb5_per_subject.csv", ds, "fbcca", tag="fb5", seeds=[0])
        for w in PRIMARY:
            if abs((G[w] - E["pha không"][w]).mean() - ref[(ds, w)]) > 1e-4:
                fail(f"{ds} w={w}: d TB {(G[w] - E['pha không'][w]).mean()} != ensdnn_tests {ref[(ds, w)]}")
        for kind, Ek in E.items():
            d = {w: G[w] - Ek[w] for w in PRIMARY}
            for w in PRIMARY:
                rho, p = spearmanr(d[w], F[w])
                order = np.argsort(F[w], kind="stable")
                terc = [d[w][idx].mean() for idx in np.array_split(order, 3)]
                rec = dict(dataset=ds, window=w, ensemble=kind, n=len(d[w]), d_mean=d[w].mean(), rho=float(rho), p=float(p),
                           d_kho=terc[0], d_vua=terc[1], d_de=terc[2], n_g31_ge=int((d[w] >= 0).sum()),
                           fbcca_mean=F[w].mean(), frac_ens_ge95=float((Ek[w] >= 95).mean()), frac_g31_ge95=float((G[w] >= 95).mean()))
                (rho_rows if kind == "pha không" else desc).append(rec)
            delta = d[1.0] - d[0.4]
            lo, hi = boot_ci(delta, rng)
            pw = 1.0 if np.allclose(delta, 0) else float(wilcoxon(delta, alternative="two-sided").pvalue)
            rec = dict(dataset=ds, ensemble=kind, n=len(delta), d_04=d[0.4].mean(), d_10=d[1.0].mean(), delta=delta.mean(), lo=lo, hi=hi,
                       n_thu_hep=int((delta > 0).sum()), p=pw)
            (win_rows if kind == "pha không" else desc).append(rec)
    for r, pa in zip(rho_rows, holm([r["p"] for r in rho_rows])):
        r.update(p_holm=pa, dat=pa < 0.05)
    for r, pa in zip(win_rows, holm([r["p"] for r in win_rows])):
        r.update(p_holm=pa, dat=pa < 0.05)
    print("[tu kiem] du nguoi x 5 seed; d TB trung ensdnn_tests.csv (muc 22)", flush=True)

    def dump(path, rr):
        keys = sorted({k for r in rr for k in r}, key=lambda k: list(rr[0]).index(k) if k in rr[0] else 99)
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            for r in rr:
                wr.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    dump(out_csv, rho_rows)
    dump(os.path.join(a.out_dir, "heterogeneity_window.csv"), win_rows)
    dump(os.path.join(a.out_dir, "heterogeneity_desc_causal.csv"), desc)
    L = ["# Không đồng nhất theo người: G3.1 − Ensemble-DNN", "",
         "d = G3.1 − Ensemble-DNN lọc pha không (điểm %); độ khó = FBCCA lọc pha không. Nhóm khó/vừa/dễ: chia ba theo FBCCA.", "",
         "## (i) Khoảng cách theo độ khó (Spearman, Holm họ 6)", "",
         "| Bộ | Cửa sổ | n | d TB | ρ | p Holm | Đạt | d nhóm khó | vừa | dễ | Số người G3.1 ≥ Ens | Tỷ lệ Ens ≥ 95% |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rho_rows:
        L.append(f"| {r['dataset']} | {r['window']:.1f} | {r['n']} | {r['d_mean']:+.2f} | {r['rho']:+.3f} | {r['p_holm']:.3g} | "
                 f"{'có' if r['dat'] else 'không'} | {r['d_kho']:+.2f} | {r['d_vua']:+.2f} | {r['d_de']:+.2f} | "
                 f"{r['n_g31_ge']}/{r['n']} | {100 * r['frac_ens_ge95']:.0f}% |")
    L += ["", "## (ii) Khoảng cách theo độ dài cửa sổ (Δ = d(1,0 s) − d(0,4 s); Wilcoxon, Holm họ 3)", "",
          "| Bộ | d 0,4 s | d 1,0 s | Δ [KTC 95%] | Số người thu hẹp | p Holm | Đạt |", "|---|---|---|---|---|---|---|"]
    for r in win_rows:
        L.append(f"| {r['dataset']} | {r['d_04']:+.2f} | {r['d_10']:+.2f} | {r['delta']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] | "
                 f"{r['n_thu_hep']}/{r['n']} | {r['p_holm']:.3g} | {'có' if r['dat'] else 'không'} |")
    L += ["", "## Mô tả: Ensemble-DNN lọc nhân quả (không kiểm định)", ""]
    for r in desc:
        if "rho" in r:
            L.append(f"- {r['dataset']} {r['window']:.1f} s: d TB {r['d_mean']:+.2f}, ρ {r['rho']:+.3f}, nhóm khó/vừa/dễ "
                     f"{r['d_kho']:+.2f} / {r['d_vua']:+.2f} / {r['d_de']:+.2f}, G3.1 ≥ Ens {r['n_g31_ge']}/{r['n']}")
        else:
            L.append(f"- {r['dataset']}: Δ {r['delta']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}], thu hẹp {r['n_thu_hep']}/{r['n']}")
    open(os.path.join(a.out_dir, "heterogeneity_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("DONE")


if __name__ == "__main__":
    main()
