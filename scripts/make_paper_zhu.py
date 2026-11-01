"""Sinh macro cho bang do chinh xac gop (Benchmark, BETA, Nakanishi, Zhu uot/kho) va so lieu muc 27 (-27h).

O bang: \\acTab<Row><DS><W> = gia tri (macro san co hoac so Zhu) + dau $^{+}$ (G3.1 cao hon co y nghia, p Holm < 0,05 trong ho
cua o do) hoac $^{-}$ (thap hon co y nghia). Row: Fd, Tf, Hd (doi chung tru TB), Tt (tt-CCA goc), Gt (G3), Go (G3.1), Ensc, Ens.
DS: Bench, Beta, Naka, Wet, Dry; W: A (0,4 s), B (1,0 s). Ho kiem dinh: 26a (Holm 4), 27c (Holm 4), G3: Nakanishi (
Holm 2), Zhu (27c, Holm 4), Benchmark/BETA la kiem dinh khong-kem-hon (khong dau); Ensemble-DNN:  (Holm 2).
Van ban: \\rvZ...  Dung: python3 make_paper_zhu.py --out <dir>
"""
import argparse
import csv
import os
import statistics as st

from make_paper_stats import num, pval, rows

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
DSK = {"benchmark": "Bench", "beta": "Beta", "nakanishi": "Naka", "zhu_wet": "Wet", "zhu_dry": "Dry"}
CMP = {"tt-CCA (gốc)": "Tt", "tt-CCA + bank lọc, trừ TB": "Tf", "DG-transfer (nhân quả, trừ TB)": "Hd",
       "FBCCA (nhân quả, trừ TB)": "Fd"}
ZROW = {"G3.1 (LOSO)": "Go", "G3 (LOSO)": "Gt", **{v: k for k, v in {"Tt": "tt-CCA (gốc)", "Tf": "tt-CCA + bank lọc, trừ TB",
        "Hd": "DG-transfer (nhân quả, trừ TB)", "Fd": "FBCCA (nhân quả, trừ TB)"}.items()}}
OLD = {"Fd": "acFd", "Tf": "acTf", "Hd": "acHd", "Tt": "extt", "Gt": "exgt", "Go": "exgto", "Ens": "exens", "Ensc": "exenscAcc"}


def wk(w):
    return {0.4: "A", 1.0: "B"}[round(float(w), 1)]


def mark(chenh, p_holm):
    if p_holm >= 0.05:
        return ""
    return "$^{+}$" if chenh > 0 else "$^{-}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    mk = {}                                                         # (row, DS, W) -> dau
    for r in rows(os.path.join(RES, "review26_analysis_2026_10_28", "review26_tests.csv")):
        mk[(CMP[r["so_sanh"].split(" − ", 1)[1]], DSK[r["dataset"]], wk(r["window"]))] = mark(float(r["chenh"]), float(r["p_holm"]))
    for r in rows(os.path.join(RES, "zhu_analysis_2026_10_29", "zhu_tests.csv")):
        mk[(CMP[r["so_sanh"].split(" − ", 1)[1]], DSK[r["dataset"]], wk(r["window"]))] = mark(float(r["chenh"]), float(r["p_holm"]))
    for r in rows(os.path.join(RES, "zhu_analysis_2026_10_29", "zhu_tests_g3.csv")):
        mk[("Gt", DSK[r["dataset"]], wk(r["window"]))] = mark(float(r["chenh"]), float(r["p_holm"]))
    for r in rows(os.path.join(RES, "naka_dc_analysis_2026_10_15", "naka_dc_tests.csv")):
        if r["so_sanh"].startswith("G3-dc"):
            mk[("Gt", "Naka", wk(r["window"]))] = mark(float(r["chenh"]), float(r["p_holm"]))
    for f, row in (("ensdnn_analysis_2026_10_18/ensdnn_tests.csv", "Ens"), ("ensdnnc_analysis_2026_10_24/ensdnnc_tests.csv", "Ensc")):
        for r in rows(os.path.join(RES, f)):
            if r["so_sanh"].startswith("G3.1 "):
                mk[(row, DSK[r["dataset"]], wk(r["window"]))] = mark(float(r["chenh"]), float(r["p_holm"]))
    zd = {(ZROW[r["method"]], DSK[r["dataset"]], wk(r["window"])): float(r["acc"])
          for r in rows(os.path.join(RES, "zhu_analysis_2026_10_29", "zhu_desc.csv")) if r["method"] in ZROW and round(float(r["window"]), 1) in (0.4, 1.0)}
    zt = {(r["so_sanh"].split(" − ", 1)[1], r["dataset"], wk(r["window"])): r for r in rows(os.path.join(RES, "zhu_analysis_2026_10_29", "zhu_tests.csv"))}
    zg = {(r["dataset"], wk(r["window"])): r for r in rows(os.path.join(RES, "zhu_analysis_2026_10_29", "zhu_tests_g3.csv"))}
    ztests = list(zt.values())

    def golden(bits):
        out, sat = {}, []
        for ds in ("zhu_wet", "zhu_dry"):
            g = rows(os.path.join(RES, "zhu_golden_2026_10_29", f"in{bits}", ds, "golden_eval.csv"))
            for w in ("0.4", "1.0"):
                by = {}
                for r in g:
                    if r["window"] == w:
                        by.setdefault(int(r["fold"]), []).append(float(r["acc_golden"]) - float(r["acc_qat"]))
                out[(ds, w)] = 100 * st.mean(st.mean(v) for v in by.values())
            s = rows(os.path.join(RES, "zhu_golden_2026_10_29", f"in{bits}", ds, "saturation.csv"))
            sat.append((ds, 100 * sum(int(r["n_sat"]) for r in s) / sum(int(r["n_samples"]) for r in s)))
        return out, dict(sat)
    g16, s16 = golden(16)
    g24, s24 = golden(24)
    d1 = {(r["dataset"], wk(r["window"])): r for r in rows(os.path.join(RES, "zhu_diag_2026_10_30", "d1_quant.csv"))}

    def mean_acc(f):
        r = [float(x["acc"]) for x in rows(os.path.join(RES, "zhu_diag_2026_10_30", f)) if x["tag"] == "base"]
        assert len(r) == 102
        return 100 * st.mean(r)
    fpA, fpB = mean_acc("wp3_g3_dc_fp_zhu_dry_w0.40_s0_per_subject.csv"), mean_acc("wp3_g3_dc_fp_zhu_dry_w1.00_s0_per_subject.csv")
    mdA = mean_acc("wp3_g3_dc_fp_med_zhu_dry_w0.40_s0_per_subject.csv")
    os.makedirs(a.out, exist_ok=True)
    for lang in ("en", "vi"):
        L = ["% Sinh boi scripts/make_paper_zhu.py; KHONG sua tay."]
        for row in ("Fd", "Tf", "Hd", "Tt", "Gt", "Go", "Ensc", "Ens"):
            for ds in ("Bench", "Beta", "Naka", "Wet", "Dry"):
                for w in "AB":
                    if ds in ("Wet", "Dry"):
                        v = "--" if row in ("Ens", "Ensc") else num(zd[(row, ds, w)], 2, lang)
                    else:
                        v = f"\\{OLD[row]}{ds}{w}"
                    if row == "Go":
                        v = f"\\textbf{{{v}}}"
                    L.append(f"\\newcommand{{\\acTab{row}{ds}{w}}}{{{v}{mk.get((row, ds, w), '')}}}")
        n_pass = sum(r["dat"] == "True" for r in ztests)
        n_rev = sum(r["b_hon_yn"] == "True" for r in ztests)
        L += [f"\\newcommand{{\\rvZNpass}}{{{n_pass}}}", f"\\newcommand{{\\rvZNrev}}{{{n_rev}}}", f"\\newcommand{{\\rvZNtests}}{{{len(ztests)}}}"]
        for ds, k in (("zhu_wet", "Wet"), ("zhu_dry", "Dry")):
            for w in "AB":
                r = zg[(ds, w)]
                L += [f"\\newcommand{{\\rvZdc{k}{w}}}{{{num(float(r['chenh']), 1, lang)}}}",
                      f"\\newcommand{{\\rvZdcl{k}{w}}}{{{num(float(r['lo']), 1, lang, True)}}}",
                      f"\\newcommand{{\\rvZdch{k}{w}}}{{{num(float(r['hi']), 1, lang, True)}}}"]
        r = zt[("tt-CCA + bank lọc, trừ TB", "zhu_wet", "B")]
        L += [f"\\newcommand{{\\rvZTfWetB}}{{{num(float(r['chenh']), 2, lang, True)}}}",
              f"\\newcommand{{\\rvZTfWetBl}}{{{num(float(r['lo']), 2, lang, True)}}}",
              f"\\newcommand{{\\rvZTfWetBh}}{{{num(float(r['hi']), 2, lang, True)}}}",
              f"\\newcommand{{\\rvZTfWetBp}}{{{pval(float(r['p_holm']), lang)}}}",
              f"\\newcommand{{\\rvZSatMin}}{{{num(min(s16.values()), 1, lang)}}}",
              f"\\newcommand{{\\rvZWMax}}{{{num(max(abs(v) for v in g24.values()), 2, lang)}}}",
              f"\\newcommand{{\\rvZSatDryW}}{{{num(s24['zhu_dry'], 1, lang)}}}",
              f"\\newcommand{{\\rvZZeroDryB}}{{{num(100 * float(d1[('zhu_dry', 'B')]['frac_zero']), 0, lang)}}}",
              f"\\newcommand{{\\rvZZeroWetB}}{{{num(100 * float(d1[('zhu_wet', 'B')]['frac_zero']), 0, lang)}}}",
              f"\\newcommand{{\\rvZFpDryA}}{{{num(fpA, 1, lang)}}}", f"\\newcommand{{\\rvZFpDryB}}{{{num(fpB, 1, lang)}}}",
              f"\\newcommand{{\\rvZMedDryA}}{{{num(mdA, 1, lang)}}}",
              f"\\newcommand{{\\rvZGoDryA}}{{{num(zd[('Go', 'Dry', 'A')], 1, lang)}}}",
              f"\\newcommand{{\\rvZGoDryB}}{{{num(zd[('Go', 'Dry', 'B')], 1, lang)}}}",
              f"\\newcommand{{\\rvZFdDryB}}{{{num(zd[('Fd', 'Dry', 'B')], 1, lang)}}}"]
        for r in rows(os.path.join(RES, "naka_dc_analysis_2026_10_15", "naka_dc_tests.csv")):
            if r["so_sanh"].startswith("G3-dc"):
                L.append(f"\\newcommand{{\\rvZdcNaka{wk(r['window'])}}}{{{num(float(r['chenh']), 1, lang)}}}")
        for r in rows(os.path.join(RES, "review26_analysis_2026_10_28", "review26_tests.csv")):
            c = CMP[r["so_sanh"].split(" − ", 1)[1]]
            if c == "Tf" and wk(r["window"]) == "B":
                L.append(f"\\newcommand{{\\rvQTf{DSK[r['dataset']]}B}}{{{num(float(r['chenh']), 2, lang, True)}}}")
        open(os.path.join(a.out, f"zhu_{lang}.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"Zhu: dat {n_pass}/{len(ztests)}, nguoc {n_rev}; o bang co dau: {sum(1 for v in mk.values() if v)}")


if __name__ == "__main__":
    main()
