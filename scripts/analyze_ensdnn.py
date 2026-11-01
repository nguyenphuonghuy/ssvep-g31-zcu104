"""Phan tich doi chung Ensemble-DNN (Guney & Ozkan 2023) theo  (chot truoc khi chay).

Moi bo (Benchmark 35, BETA 70, Nakanishi 10), moi cua so chinh 0,4 / 1,0 s (accuracy moi nguoi = TB 5 seed):
  G3.1 - Ensemble-DNN va G3 - Ensemble-DNN: Wilcoxon hai phia, Holm trong ho 2; KTC 95% bootstrap theo nguoi (10 000).
Mo ta: accuracy + ITR (Tg 0,5 s; muc 15) moi cua so (Ensemble-DNN seed 0 o 0,2/0,6/0,8 s); so tham so / MACs mot quyet dinh
(GuneyDNN x N mo hinh nguoi nguon + do tuong dong); doi chieu voi ket qua goc cua tac gia (Results/*.mat, repo chinh thuc).
Tu kiem (exit 3): du file/nguoi/seed; dem tham so/MACs giai tich trung ssvep.models.count_params/count_macs.
Dung: python analyze_ensdnn.py --ens_dir <dir> --out_dir <dir>   (OVERWRITE=1 de ghi de)
"""
import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_h1 import boot_ci, fail, holm, read_rows  # noqa: E402
from ssvep.metrics import itr_bpm  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJ, "results")
G3 = os.path.join(RES, "g3_independent_2026_10_13")
G31 = {"benchmark": os.path.join(RES, "g3dc_2026_10_17"), "beta": os.path.join(RES, "g3dc_2026_10_17"),
       "nakanishi": os.path.join(RES, "naka_dc_2026_10_15")}
OFFICIAL = os.path.join(os.path.dirname(os.path.dirname(PROJ)), "Tham khao", "_dl_guney", "repo", "Ensemble-of-DNNs-main",
                        "Results")
OFF_FILE = {"benchmark": "Benchmark_ensemble_results.mat", "beta": "Beta_ensemble_results.mat"}
WINDOWS, PRIMARY, SEEDS = [0.2, 0.4, 0.6, 0.8, 1.0], [0.4, 1.0], [0, 1, 2, 3, 4]
NSUB = {"benchmark": 35, "beta": 70, "nakanishi": 10}
MT = {"benchmark": 40, "beta": 40, "nakanishi": 12}
FS = {"benchmark": 250, "beta": 250, "nakanishi": 256}
C, NB, NCH, NH = 9, 3, 120, 5


def load(paths, ds, seeds, method):
    acc = defaultdict(lambda: defaultdict(dict))
    for p in paths:
        for r in read_rows(p):
            if r["method"] == method:
                acc[round(float(r["window"]), 2)][int(r["subject"])][int(r["seed"])] = float(r["acc"])
    subs = list(range(1, NSUB[ds] + 1))
    out = {}
    for w, sd in seeds.items():
        if sorted(acc[w]) != subs or any(sorted(acc[w][s]) != sd for s in subs):
            fail(f"{ds} {method} w={w}: thieu nguoi/seed (can seed {sd})")
        out[w] = np.array([100 * np.mean([acc[w][s][k] for k in sd]) for s in subs])
    return out


def cost(ds, win):
    """Tham so va MACs (giai tich) cua GuneyDNN; do tuong dong / mo hinh nguoi nguon (cac phep nhan-cong trong testing.m,
    phan tinh truoc duoc tu template khong tinh): ket hop dai con x (NB*C*T), chieu 120 cot (120*C*T), corr (120*T),
    chieu len co so CCA (120*T*2NH)."""
    T = int(round(win * FS[ds]))
    K, T2 = MT[ds], T // 2
    params = NB + (NCH * C + NCH) + (NCH * NCH * 2 + NCH) + (NCH * NCH * 10 + NCH) + (NCH * T2 * K + K)
    macs = NB * C * T + NCH * C * T + NCH * NCH * 2 * T2 + NCH * NCH * 10 * T2 + NCH * T2 * K
    sim = NB * C * T + NCH * C * T + NCH * T + NCH * T * 2 * NH
    return T, params, macs, sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ens_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    out_csv = os.path.join(a.out_dir, "ensdnn_tests.csv")
    if os.path.exists(out_csv) and os.environ.get("OVERWRITE") != "1":
        fail(f"{out_csv} da ton tai; OVERWRITE=1 de ghi de")
    os.makedirs(a.out_dir, exist_ok=True)
    # tu kiem dem tham so / MACs
    import torch
    from ssvep.models import GuneyDNN, count_macs, count_params
    for ds in NSUB:
        for w in WINDOWS:
            T, p, m, _ = cost(ds, w)
            net = GuneyDNN(C, T, MT[ds], FS[ds])
            if count_params(net) != p or count_macs(net, torch.zeros(1, NB, C, T)) != m:
                fail(f"dem tham so/MACs {ds} w={w}: {count_params(net)}/{p}, {count_macs(net, torch.zeros(1, NB, C, T))}/{m}")
    print("[tu kiem] tham so/MACs giai tich == count_params/count_macs", flush=True)
    ens_seeds = {w: (SEEDS if w in PRIMARY else [0]) for w in WINDOWS}
    rng = np.random.default_rng(20261018)
    rows, desc, cst, off = [], [], [], []
    for ds in NSUB:
        E = load(sorted(glob.glob(os.path.join(a.ens_dir, f"ensdnn_{ds}_w*_s*_per_subject.csv"))), ds, ens_seeds,
                 "ensemble_dnn")
        E0 = load(sorted(glob.glob(os.path.join(a.ens_dir, f"ensdnn_{ds}_w*_s0_per_subject.csv"))), ds,
                  {w: [0] for w in WINDOWS}, "ensemble_dnn")
        g31 = load(sorted(glob.glob(os.path.join(G31[ds], f"wp3_g3_dc_{ds}_w*_s*_per_subject.csv"))), ds,
                   {w: SEEDS for w in WINDOWS}, "wp3_g3_dc")
        g3 = load(sorted(glob.glob(os.path.join(G3, f"wp3_g3_{ds}_w*_s*_per_subject.csv"))), ds,
                  {w: SEEDS for w in WINDOWS}, "wp3_g3")
        for name, d in (("Ensemble-DNN", E), ("G3.1", g31), ("G3", g3)):
            for w in WINDOWS:
                desc.append(dict(dataset=ds, method=name, window=w, n_seed=len(ens_seeds[w]) if name == "Ensemble-DNN" else 5,
                                 acc=d[w].mean(), itr=np.mean([itr_bpm(v / 100, MT[ds], w, 0.5) for v in d[w]])))
        for w in WINDOWS:
            T, p, m, s = cost(ds, w)
            N = NSUB[ds] - 1
            cst.append(dict(dataset=ds, window=w, T=T, n_models=N, params_one=p, params_total=N * p, macs_one=m,
                            macs_total=N * (m + s)))
        for w in PRIMARY:
            fam = []
            for name, d in (("G3.1", g31), ("G3", g3)):
                dd = d[w] - E[w]
                lo, hi = boot_ci(dd, rng)
                pv = 1.0 if np.allclose(dd, 0) else float(wilcoxon(d[w], E[w], alternative="two-sided").pvalue)
                fam.append(dict(dataset=ds, window=w, so_sanh=f"{name} − Ensemble-DNN", n=len(dd), mean_a=d[w].mean(),
                                mean_ens=E[w].mean(), chenh=dd.mean(), lo=lo, hi=hi, a_hon=int((dd > 0).sum()), p=pv))
            for r, pa in zip(fam, holm([r["p"] for r in fam])):
                r.update(p_holm=pa, dat=pa < 0.05)
            rows += fam
        if ds in OFF_FILE:
            from scipy.io import loadmat
            mat = loadmat(os.path.join(OFFICIAL, OFF_FILE[ds]))
            arr = [v for k, v in mat.items() if not k.startswith("__")][0]          # nguoi x block x 9 cua so (0,2..1,0)
            if arr.shape[0] != NSUB[ds] or arr.shape[2] != 9:
                fail(f"ket qua goc {ds}: shape {arr.shape}")
            for w in WINDOWS:
                o = 100 * arr[:, :, int(round((w - 0.2) / 0.1))].mean(axis=1)
                r0 = float(pearsonr(o, E0[w])[0])
                off.append(dict(dataset=ds, window=w, official=o.mean(), ours_s0=E0[w].mean(), ours=E[w].mean(),
                                chenh_s0=(E0[w] - o).mean(), r_s0=r0,
                                itr_official=np.mean([itr_bpm(v / 100, MT[ds], w, 0.5) for v in o])))
    print("[tu kiem] du nguoi x seed: Ensemble-DNN (5 seed cua so chinh, seed 0 cua so khac), G3.1, G3; 3 bo", flush=True)

    def dump(path, rr):
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rr[0]))
            wr.writeheader()
            for r in rr:
                wr.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    dump(out_csv, rows)
    dump(os.path.join(a.out_dir, "ensdnn_desc.csv"), desc)
    dump(os.path.join(a.out_dir, "ensdnn_cost.csv"), cst)
    dump(os.path.join(a.out_dir, "ensdnn_official.csv"), off)
    L = ["# Đối chứng Ensemble-DNN (Güney & Özkan 2023)", "",
         "Accuracy TB theo người (%), ITR Tg = 0,5 s (bits/phút) trong ngoặc. Ensemble-DNN: TB 5 seed ở 0,4 / 1,0 s, seed 0 "
         "ở cửa sổ khác; G3/G3.1: TB 5 seed.", ""]
    for ds in NSUB:
        L += [f"## {ds}", "", "| Phương pháp | " + " | ".join(f"{w:.1f} s" for w in WINDOWS) + " |", "|---" * 6 + "|"]
        for name in ("Ensemble-DNN", "G3.1", "G3"):
            cells = [next(f"{r['acc']:.2f} ({r['itr']:.1f})" for r in desc if r["dataset"] == ds and r["method"] == name
                          and r["window"] == w) for w in WINDOWS]
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Kiểm định (Wilcoxon hai phía, Holm trong họ 2 mỗi bộ × cửa sổ)", "",
          "| Bộ | Cửa sổ | So sánh | A | Ensemble-DNN | Chênh [KTC 95%] | A hơn | p | p Holm | Đạt |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['dataset']} | {r['window']:.1f} | {r['so_sanh']} | {r['mean_a']:.2f} | {r['mean_ens']:.2f} | "
                 f"{r['chenh']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] | {r['a_hon']}/{r['n']} | {r['p']:.3g} | "
                 f"{r['p_holm']:.3g} | {'có' if r['dat'] else 'không'} |")
    L += ["", "## Chi phí một quyết định (giải tích)", "", "| Bộ | Cửa sổ | N mô hình | Tham số / mô hình | Tham số tổng | "
          "MACs / mô hình | MACs tổng (gồm độ tương đồng) |", "|---|---|---|---|---|---|---|"]
    for r in cst:
        L.append(f"| {r['dataset']} | {r['window']:.1f} | {r['n_models']} | {r['params_one']:,} | {r['params_total']:,} | "
                 f"{r['macs_one']:,} | {r['macs_total']:,} |")
    L += ["", "## Đối chiếu với kết quả gốc của tác giả (Results/*.mat; mô tả)", "",
          "| Bộ | Cửa sổ | Gốc (%) | Của ta seed 0 (%) | Chênh seed 0 | r theo người (seed 0) | Của ta TB seed | ITR gốc |",
          "|---|---|---|---|---|---|---|---|"]
    for r in off:
        L.append(f"| {r['dataset']} | {r['window']:.1f} | {r['official']:.2f} | {r['ours_s0']:.2f} | {r['chenh_s0']:+.2f} | "
                 f"{r['r_s0']:.3f} | {r['ours']:.2f} | {r['itr_official']:.1f} |")
    open(os.path.join(a.out_dir, "ensdnn_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("DONE")


if __name__ == "__main__":
    main()
