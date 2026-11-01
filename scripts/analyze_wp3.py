"""Phan tich WP3 theo  (chot 2026-10-12, truoc khi chay job 17017).

- H3 / QAT: sut = FP32 (job 17000, wp2_hybrid) - QAT b bit (b = 8, 6, 4), ghep cap theo nguoi (TB 5 seed).
  H3 duoc ung ho o mot cua so neu can tren KTC 95% bootstrap cua sut (8 bit) < 1 diem %; kem Wilcoxon hai phia.
- Loc nhan qua (Holm trong ho 3 moi cua so): lai nhan qua - lai filtfilt; FBCCA nhan qua - FBCCA filtfilt;
  lai nhan qua - FBCCA nhan qua.
- Ablation (Holm trong ho 3 moi cua so): lai day du - {chi template, chi sin/cos, bo bank loc}.
Tu kiem (exit 3): FBCCA filtfilt cua job WP3 trung G0 tung nguoi; du 35 nguoi x 5 seed x 5 cua so.
Dung: python analyze_wp3.py --wp3_dir <dir> --out_dir <dir>   (OVERWRITE=1 de ghi de)
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_h1 import boot_ci, fail, holm, load_fbcca, read_rows  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FP32_DIR = os.path.join(PROJ, "results", "wp2_loso_benchmark_2026_10_11")
SEEDS, WINDOWS, PRIMARY, N = [0, 1, 2, 3, 4], [0.2, 0.4, 0.6, 0.8, 1.0], {0.4, 1.0}, 35
SUBS = list(range(1, N + 1))
VARS = ["wp3_qat8", "wp3_qat6", "wp3_qat4", "wp3_causal", "wp3_abl_tmpl", "wp3_abl_sc", "wp3_abl_nofb"]
NAME = {"wp2_hybrid": "lai FP32 (job 17000)", "wp3_qat8": "QAT 8 bit", "wp3_qat6": "QAT 6 bit",
        "wp3_qat4": "QAT 4 bit", "wp3_causal": "lai, lọc nhân quả", "wp3_abl_tmpl": "chỉ template",
        "wp3_abl_sc": "chỉ sin/cos", "wp3_abl_nofb": "bỏ bank lọc", "fbcca": "FBCCA filtfilt",
        "fbcca_c": "FBCCA nhân quả"}


def load_model(d, model):
    acc, cost = defaultdict(lambda: defaultdict(dict)), {}
    for fn in sorted(os.listdir(d)):
        if not fn.startswith(f"{model}_benchmark_w"):
            continue
        p = os.path.join(d, fn)
        if fn.endswith("_per_subject.csv"):
            for r in read_rows(p):
                w, s, sd = round(float(r["window"]), 2), int(r["subject"]), int(r["seed"])
                if r["tag"] != "base":
                    continue
                if sd in acc[w][s]:
                    fail(f"trung lap {model} w={w} s={s} sd={sd}")
                acc[w][s][sd] = float(r["acc"])
        elif fn.endswith("_timing.csv"):
            for r in read_rows(p):
                cost[round(float(r["window"]), 2)] = (int(r["params"]), int(r["macs"]))
    out = {}
    for w in WINDOWS:
        if sorted(acc[w]) != SUBS or any(sorted(acc[w][s]) != SEEDS for s in SUBS):
            fail(f"{model} w={w}: thieu nguoi/seed")
        out[w] = np.array([100 * np.mean([acc[w][s][sd] for sd in SEEDS]) for s in SUBS])
    return out, cost


def load_fb(path):
    acc = defaultdict(dict)
    for r in read_rows(path):
        acc[round(float(r["window"]), 2)][int(r["subject"])] = 100 * float(r["acc"])
    return {w: np.array([acc[w][s] for s in SUBS]) for w in WINDOWS}


def compare(a, b, rng):
    d = a - b
    lo, hi = boot_ci(d, rng)
    p = 1.0 if np.allclose(d, 0) else float(wilcoxon(a, b, alternative="two-sided").pvalue)
    return dict(mean_a=a.mean(), mean_b=b.mean(), chenh_tb=d.mean(), ktc95_lo=lo, ktc95_hi=hi,
                so_nguoi_a_hon=int((d > 0).sum()), p=p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wp3_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    out_csv, out_md = os.path.join(a.out_dir, "wp3_tests.csv"), os.path.join(a.out_dir, "wp3_report.md")
    if os.path.exists(out_csv) and os.environ.get("OVERWRITE") != "1":
        fail(f"{out_csv} da ton tai; OVERWRITE=1 de ghi de")
    os.makedirs(a.out_dir, exist_ok=True)

    # ---------- tu kiem ----------
    g0, _ = load_fbcca("benchmark")
    fb = load_fb(os.path.join(a.wp3_dir, "fbcca_fb5_per_subject.csv"))
    for w in WINDOWS:
        if not np.allclose(fb[w], [100 * g0[w][s] for s in SUBS], atol=1e-3):
            fail(f"FBCCA filtfilt cua WP3 khong trung G0 o w={w}")
    fbc = load_fb(os.path.join(a.wp3_dir, "fbcca_fb5c_per_subject.csv"))
    data, cost = {}, {}
    data["wp2_hybrid"], cost["wp2_hybrid"] = load_model(FP32_DIR, "wp2_hybrid")
    for v in VARS:
        data[v], cost[v] = load_model(a.wp3_dir, v)
    data["fbcca"], data["fbcca_c"] = fb, fbc
    print("[tu kiem] DAT: FBCCA filtfilt trung G0 tung nguoi; 8 mo hinh x 5 cua so x 35 nguoi x 5 seed", flush=True)

    rng = np.random.default_rng(20261013)
    rows = []
    for w in WINDOWS:
        role = "chinh" if w in PRIMARY else "mo ta"
        # H3 / QAT (khong Holm, theo )
        for v in ("wp3_qat8", "wp3_qat6", "wp3_qat4"):
            r = compare(data["wp2_hybrid"][w], data[v][w], rng)
            r.update(window=f"{w:.1f}", vai_tro=role, nhom="QAT (sut = FP32 - QAT)", a="wp2_hybrid", b=v,
                     p_holm=r["p"], ket_luan=("H3 DAT" if r["ktc95_hi"] < 1.0 else "H3 KHONG DAT")
                     if v == "wp3_qat8" else ("can tren < 1" if r["ktc95_hi"] < 1.0 else "can tren >= 1"))
            rows.append(r)
        for nhom, pairs in (("Loc nhan qua", [("wp3_causal", "wp2_hybrid"), ("fbcca_c", "fbcca"),
                                              ("wp3_causal", "fbcca_c")]),
                            ("Ablation (lai day du - bien the)", [("wp2_hybrid", "wp3_abl_tmpl"),
                                                                  ("wp2_hybrid", "wp3_abl_sc"),
                                                                  ("wp2_hybrid", "wp3_abl_nofb")])):
            fam = []
            for x, y in pairs:
                r = compare(data[x][w], data[y][w], rng)
                r.update(window=f"{w:.1f}", vai_tro=role, nhom=nhom, a=x, b=y)
                fam.append(r)
            for r, pa in zip(fam, holm([r["p"] for r in fam])):
                r["p_holm"] = pa
                r["ket_luan"] = "co y nghia" if pa < 0.05 else "khong co y nghia"
            rows += fam

    keys = ["window", "vai_tro", "nhom", "a", "b", "mean_a", "mean_b", "chenh_tb", "ktc95_lo", "ktc95_hi",
            "so_nguoi_a_hon", "p", "p_holm", "ket_luan"]
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{r[k]:.6g}" if isinstance(r[k], float) else r[k]) for k in keys})

    L = ["# WP3 — QAT, lọc nhân quả, ablation của lai H-C #21 (Benchmark, LOSO, 5 seed)", "",
         "Script `scripts/analyze_wp3.py`.",
         "Accuracy theo người = TB 5 seed; chênh = A − B (điểm %), KTC 95% bootstrap theo người (10 000 lần);",
         "Wilcoxon hai phía; Holm trong họ 3 mỗi cửa sổ cho lọc nhân quả và ablation (QAT: không Holm).",
         "H3 đạt ở một cửa sổ nếu cận trên KTC 95% của sụt (FP32 − QAT 8 bit) < 1 điểm %.", "",
         "## Chi phí", "", "| Mô hình | Tham số | MACs 0,4 s (M) | MACs 1,0 s (M) |", "|---|---|---|---|"]
    for m in ["wp2_hybrid"] + VARS:
        L.append(f"| {NAME[m]} | {cost[m][0.4][0]} | {cost[m][0.4][1] / 1e6:.3f} | {cost[m][1.0][1] / 1e6:.3f} |")
    L += ["", "## Accuracy TB theo người (%)", "",
          "| Mô hình | " + " | ".join(f"{w:.1f} s" for w in WINDOWS) + " |", "|---" * (len(WINDOWS) + 1) + "|"]
    for m in ["fbcca", "fbcca_c", "wp2_hybrid"] + VARS:
        L.append(f"| {NAME[m]} | " + " | ".join(f"{data[m][w].mean():.2f}" for w in WINDOWS) + " |")
    for nhom in ("QAT (sut = FP32 - QAT)", "Loc nhan qua", "Ablation (lai day du - bien the)"):
        L += ["", f"## {nhom}", "", "| Cửa sổ | A | B | Chênh A − B [KTC 95%] | Số người A > B | p | p Holm | Kết luận |",
              "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["nhom"] == nhom:
                L.append(f"| {r['window']} ({r['vai_tro']}) | {NAME[r['a']]} | {NAME[r['b']]} | {r['chenh_tb']:+.2f} "
                         f"[{r['ktc95_lo']:+.2f}, {r['ktc95_hi']:+.2f}] | {r['so_nguoi_a_hon']}/{N} | {r['p']:.3g} | "
                         f"{r['p_holm']:.3g} | {r['ket_luan']} |")
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"ghi {out_csv} ({len(rows)} dong, dich 45) va {out_md}")
    print("DONE")


if __name__ == "__main__":
    main()
