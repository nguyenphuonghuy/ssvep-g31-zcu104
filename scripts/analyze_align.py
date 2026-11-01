"""Phan tich chan doan/hieu chinh lech thoi gian-pha BETA (chot truoc khi chay).

Nguoi BETA S11-S70 (n = 60), accuracy = TB 5 seed. Cua so chinh 0,4 va 1,0 s:
 (i) V1 - V0, V2 - V0: Wilcoxon hai phia, Holm trong ho 2 moi cua so;
 (ii) khong-kem-hon V1, V2 so voi FBCCA nhan qua (cua so V1), bien 1 diem (Wilcoxon mot phia tren chenh + 1), Holm ho 2.
KTC 95% bootstrap theo nguoi (10 000). Mo ta: moi cua so; G3-BETA LOSO (job 17026) tren S11-S70.
Dung: python analyze_align.py --align_dir <dir> --out_dir <dir> [--model wp3_g3_dc]   (OVERWRITE=1 de ghi de)
--model wp3_g3_dc (G3.1): cung phep kiem; FBCCA doi chung (ii) = FBCCA nhan qua fb5c V1 cua muc 14
(job 17065, G3_ALIGN); FBCCA fb5cd V0/V1 chi mo ta; LOSO = G3.1-BETA (job 17200); mo ta them G3 V0/V1/V2 (job 17065).
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_h1 import boot_ci, fail, holm, read_rows  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G3 = os.path.join(PROJ, "results", "g3_independent_2026_10_13")
G3DC = os.path.join(PROJ, "results", "g3dc_2026_10_17")
G3_ALIGN = os.path.join(PROJ, "results", "g3_align_beta_2026_10_15")
EVAL = list(range(11, 71))
WINDOWS, PRIMARY, SEEDS = [0.2, 0.4, 0.6, 0.8, 1.0], [0.4, 1.0], [0, 1, 2, 3, 4]


def collect(paths, method, seeds):
    acc = defaultdict(lambda: defaultdict(dict))
    for p in paths:
        for r in read_rows(p):
            if r["method"] != method:
                continue
            w, s, sd = round(float(r["window"]), 2), int(r["subject"]), int(r["seed"])
            if s in EVAL:
                acc[w][s][sd] = float(r["acc"])
    out = {}
    for w in WINDOWS:
        if sorted(acc[w]) != EVAL or any(sorted(acc[w][s]) != seeds for s in EVAL):
            fail(f"{method} w={w}: thieu nguoi/seed")
        out[w] = np.array([100 * np.mean([acc[w][s][sd] for sd in seeds]) for s in EVAL])
    return out


def test(a, b, rng, noninf=False):
    d = a - b
    lo, hi = boot_ci(d, rng)
    if noninf:
        p = float(wilcoxon(d + 1.0, alternative="greater").pvalue)
    else:
        p = 1.0 if np.allclose(d, 0) else float(wilcoxon(a, b, alternative="two-sided").pvalue)
    return dict(mean_a=a.mean(), mean_b=b.mean(), chenh=d.mean(), lo=lo, hi=hi, a_hon=int((d > 0).sum()), p=p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--align_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="wp3_g3")
    a = ap.parse_args()
    g31 = a.model == "wp3_g3_dc"
    out_csv = os.path.join(a.out_dir, "align_tests.csv")
    if os.path.exists(out_csv) and os.environ.get("OVERWRITE") != "1":
        fail(f"{out_csv} da ton tai; OVERWRITE=1 de ghi de")
    os.makedirs(a.out_dir, exist_ok=True)
    import glob
    files = sorted(glob.glob(os.path.join(a.align_dir, "align_beta_w*_s*_per_subject.csv")))
    V = {v: collect(files, v, SEEDS) for v in ("V0", "V1", "V2")}
    fbp = [os.path.join(G3_ALIGN if g31 else a.align_dir, "fbcca_align_beta_per_subject.csv")]     # FBCCA fb5c (muc 14)
    F = {v: collect(fbp, f"fbcca_{v}", [0]) for v in ("V0", "V1")}
    extra = []
    if g31:
        loso = collect(sorted(glob.glob(os.path.join(G3DC, "wp3_g3_dc_beta_w*_s*_per_subject.csv"))), "wp3_g3_dc", SEEDS)
        fdc = [os.path.join(a.align_dir, "fbcca_align_beta_per_subject.csv")]
        extra = [(f"FBCCA nhân quả trừ DC (fb5cd) {v}", collect(fdc, f"fbcca_{v}", [0])) for v in ("V0", "V1")]
        gf = sorted(glob.glob(os.path.join(G3_ALIGN, "align_beta_w*_s*_per_subject.csv")))
        extra += [(f"G3 {v} (mục 14, job 17065)", collect(gf, v, SEEDS)) for v in ("V0", "V1", "V2")]
        old = None
    else:
        loso = collect(sorted(glob.glob(os.path.join(G3, "wp3_g3_beta_w*_s*_per_subject.csv"))), "wp3_g3", SEEDS)
        old = collect(sorted(glob.glob(os.path.join(G3, "g3_cross_beta_w*_s*_per_subject.csv"))), "g3_cross", SEEDS)
    par = json.load(open(os.path.join(a.align_dir, "align_params.json")))
    print(f"[tu kiem] mo hinh {a.model}: du 60 nguoi x 5 seed x 5 cua so cho V0/V1/V2; FBCCA V0/V1; LOSO; doi chieu", flush=True)
    rng = np.random.default_rng(20261015)
    rows = []
    for w in PRIMARY:
        fam = [dict(test(V["V1"][w], V["V0"][w], rng), so_sanh="V1 − V0", kiem_dinh="hai phía"),
               dict(test(V["V2"][w], V["V0"][w], rng), so_sanh="V2 − V0", kiem_dinh="hai phía")]
        for r, pa in zip(fam, holm([r["p"] for r in fam])):
            r.update(window=w, p_holm=pa, dat=pa < 0.05)
        rows += fam
        fam = [dict(test(V["V1"][w], F["V1"][w], rng, True), so_sanh="V1 − FBCCA nhân quả (V1)", kiem_dinh="không-kém-hơn"),
               dict(test(V["V2"][w], F["V1"][w], rng, True), so_sanh="V2 − FBCCA nhân quả (V1)", kiem_dinh="không-kém-hơn")]
        for r, pa in zip(fam, holm([r["p"] for r in fam])):
            r.update(window=w, p_holm=pa, dat=pa < 0.05)
        rows += fam
    keys = ["window", "so_sanh", "kiem_dinh", "mean_a", "mean_b", "chenh", "lo", "hi", "a_hon", "p", "p_holm", "dat"]
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{r[k]:.6g}" if isinstance(r[k], float) else r[k]) for k in keys})
    L = [f"# Hiệu chỉnh lệch thời gian–pha BETA cho {'G3.1' if g31 else 'G3'}-cross", "",
         f"Tập hiệu chỉnh BETA S1–S10: τ = {par['tau_ms']} ms, φ0 = {par['phi0_deg']:+.1f}°, R = {par['R']:.3f}; dịch cửa sổ "
         f"BETA {par['o_samples']} mẫu (τ' = {par['tau_after_ms']} ms); dịch template dk ∈ [{min(par['d_bench'])}, "
         f"{max(par['d_bench'])}] mẫu (phần dư {par['resid_before']:.3f} → {par['resid_after']:.3f} rad).",
         "Đánh giá: BETA S11–S70 (n = 60), TB 5 seed.", "", "## Accuracy TB (%)", "",
         "| | " + " | ".join(f"{w:.1f} s" for w in WINDOWS) + " |", "|---" * 6 + "|"]
    lines = [("V0 (không hiệu chỉnh)", V["V0"]), ("V1 (dịch cửa sổ)", V["V1"]), ("V2 (V1 + dịch template)", V["V2"]),
             ("FBCCA nhân quả V0", F["V0"]), ("FBCCA nhân quả V1", F["V1"])]
    lines += extra if g31 else [("G3-cross job 17026 (S11–S70)", old)]
    lines += [(("G3.1-BETA LOSO job 17200" if g31 else "G3-BETA LOSO job 17026") + " (S11–S70)", loso)]
    for lab, d in lines:
        L.append(f"| {lab} | " + " | ".join(f"{d[w].mean():.2f}" for w in WINDOWS) + " |")
    L += ["", "## Kiểm định (cửa sổ chính)", "", "| Cửa sổ | So sánh | Kiểm định | A | B | Chênh [KTC 95%] | A > B | p | p Holm | Đạt |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['window']:.1f} | {r['so_sanh']} | {r['kiem_dinh']} | {r['mean_a']:.2f} | {r['mean_b']:.2f} | "
                 f"{r['chenh']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] | {r['a_hon']}/60 | {r['p']:.3g} | {r['p_holm']:.3g} | "
                 f"{'có' if r['dat'] else 'không'} |")
    open(os.path.join(a.out_dir, "align_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("DONE")


if __name__ == "__main__":
    main()
