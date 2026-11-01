"""Sinh macro so lieu cac muc 26, 28 cho bai TBioCAS (ban Anh va ban Viet, cung ten macro). Chi thu vien chuan.

Vao (chi doc ket qua da chot):
  results/review26_analysis_2026_10_28/review26_tests.csv, review26_desc.csv   (Holm trong ho 4)
  results/golden_g31_all_2026_10_28/<ds>/golden_eval.csv, saturation.csv      (dau vao 16 bit)
  results/golden_g31_w24_2026_10_28/<ds>/golden_eval.csv                      (dau vao 24 bit)
  results/zcu104_g31_pl_beta_2026_10_28/g3_pl_rt_result_g31rt_T*.json        (BETA tren bo)
  results/zcu104_power_std_2026_10_29/summary_run2.json                       (so chinh run2)
  results/zcu104_ttcca_ps_2026_10_29/summary.json                             (do tre tt-CCA/FBCCA)
Ra: <out>/rev_en.tex, <out>/rev_vi.tex voi
  \\qd/\\ql/\\qh/\\qp<Cmp><DS><W>: chenh G3.1 - doi chung, KTC, p Holm (ho 4); dau * neu p Holm < 0,05 (moi chieu)
     Cmp: Tt (tt-CCA goc), Tf (tt-CCA + bank loc, tru TB), Hd (DG-transfer tru TB), Fd (FBCCA tru TB)
  \\ac<Cmp><DS><W>: accuracy (%) cua Tf, Hd, Fd
  \\rv...: so dem, golden, BETA tren bo, nang luong chuan hoa, do tre tt-CCA/FBCCA (xem cuoi file sinh ra)
Dung: python3 make_paper_rev.py --out <dir>
"""
import argparse
import csv
import json
import os
import statistics as st
from collections import defaultdict

from make_paper_stats import DSK, num, pval, rows, wk

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
CMP = {"tt-CCA (gốc)": "Tt", "tt-CCA + bank lọc, trừ TB": "Tf", "DG-transfer (nhân quả, trừ TB)": "Hd",
       "FBCCA (nhân quả, trừ TB)": "Fd"}


def thou(n, lang):
    s = f"{int(round(n)):,}"
    return s.replace(",", "{,}") if lang == "en" else s.replace(",", "\\,")


def golden(d):
    """chenh TB (golden - QAT) theo nguoi cua TB seed, moi cua so."""
    out = {}
    for ds in DSK:
        g = rows(os.path.join(RES, d, ds, "golden_eval.csv"))
        for w in ("0.4", "1.0"):
            by = defaultdict(list)
            for r in g:
                if r["window"] == w:
                    by[int(r["fold"])].append(float(r["acc_golden"]) - float(r["acc_qat"]))
            out[(ds, w)] = 100 * st.mean(st.mean(v) for v in by.values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    T = rows(os.path.join(RES, "review26_analysis_2026_10_28", "review26_tests.csv"))
    D = rows(os.path.join(RES, "review26_analysis_2026_10_28", "review26_desc.csv"))
    assert len(T) == 24
    g16, g24 = golden("golden_g31_all_2026_10_28"), golden("golden_g31_w24_2026_10_28")
    sat = rows(os.path.join(RES, "golden_g31_all_2026_10_28", "nakanishi", "saturation.csv"))
    sat_naka = 100 * sum(int(r["n_sat"]) for r in sat) / sum(int(r["n_samples"]) for r in sat)
    beta_trials = 0
    for t in (100, 250):
        j = json.load(open(os.path.join(RES, "zcu104_g31_pl_beta_2026_10_28", f"g3_pl_rt_result_g31rt_T{t}.json")))
        rr = j["folds"][str(t)]
        assert sum(r["sai_score"] + r["sai_pred"] for r in rr) == 0 and all(r["status"] == 1 for r in rr)
        beta_trials += sum(r["n"] for r in rr)
    fa = rows(os.path.join(RES, "hw_vectors_g31_beta_2026_10_28", "frame_acc.csv"))     # chen 3 mau
    pad_diff = max(abs(100 * st.mean(float(r["acc_hw_frame"]) - float(r["acc_std"]) for r in fa if r["window"] == w))
                   for w in ("0.4", "1.0"))
    P = json.load(open(os.path.join(RES, "zcu104_power_std_2026_10_29", "summary_run2.json")))
    L28 = json.load(open(os.path.join(RES, "zcu104_ttcca_ps_2026_10_29", "summary.json")))
    os.makedirs(a.out, exist_ok=True)
    for lang in ("en", "vi"):
        L = ["% Sinh boi scripts/make_paper_rev.py; KHONG sua tay."]
        for r in T:
            k = f"{CMP[r['so_sanh'].split(' − ', 1)[1]]}{DSK[r['dataset']]}{wk(r['window'])}"
            sig = float(r["p_holm"]) < 0.05
            L.append(f"\\newcommand{{\\qd{k}}}{{{num(float(r['chenh']), 2, lang, True)}{'$^{*}$' if sig else ''}}}")
            L.append(f"\\newcommand{{\\ql{k}}}{{{num(float(r['lo']), 2, lang, True)}}}")
            L.append(f"\\newcommand{{\\qh{k}}}{{{num(float(r['hi']), 2, lang, True)}}}")
            L.append(f"\\newcommand{{\\qp{k}}}{{{pval(float(r['p_holm']), lang)}}}")
        for r in D:
            m = {v: k for k, v in CMP.items()}
            for c in ("Tf", "Hd", "Fd"):
                if r["method"] == m[c] and wk(r["window"]):
                    L.append(f"\\newcommand{{\\ac{c}{DSK[r['dataset']]}{wk(r['window'])}}}{{{num(float(r['acc']), 2, lang)}}}")
        n_pass = sum(r["dat"] == "True" for r in T)
        n_rev = sum(r["b_hon_yn"] == "True" for r in T)
        n24 = sum(float(r["p_holm24"]) < 0.05 and float(r["chenh"]) > 0 for r in T)
        L += [f"\\newcommand{{\\rvNpass}}{{{n_pass}}}", f"\\newcommand{{\\rvNtests}}{{{len(T)}}}",
              f"\\newcommand{{\\rvNpassAll}}{{{n24}}}", f"\\newcommand{{\\rvNrev}}{{{n_rev}}}"]
        gmax = max(abs(g16[(ds, w)]) for ds in ("benchmark", "beta") for w in ("0.4", "1.0"))
        L += [f"\\newcommand{{\\rvGoldMax}}{{{num(gmax, 2, lang)}}}",
              f"\\newcommand{{\\rvGoldNakaA}}{{{num(g16[('nakanishi', '0.4')], 1, lang, True)}}}",
              f"\\newcommand{{\\rvGoldNakaB}}{{{num(g16[('nakanishi', '1.0')], 1, lang, True)}}}",
              f"\\newcommand{{\\rvWNakaA}}{{{num(g24[('nakanishi', '0.4')], 2, lang, True)}}}",
              f"\\newcommand{{\\rvWNakaB}}{{{num(g24[('nakanishi', '1.0')], 2, lang, True)}}}",
              f"\\newcommand{{\\rvWMax}}{{{num(max(abs(v) for v in g24.values()), 2, lang)}}}",
              f"\\newcommand{{\\rvSatNaka}}{{{num(sat_naka, 1, lang)}}}",
              f"\\newcommand{{\\rvBetaTrials}}{{{thou(beta_trials, lang)}}}",
              f"\\newcommand{{\\rvPadDiff}}{{{num(pad_diff, 2, lang)}}}"]
        for w, kk in (("A", "100"), ("B", "250")):
            e = lambda m: P[m + kk]["E_mJ"]  # noqa: E731
            L += [f"\\newcommand{{\\rvEpl{w}}}{{{num(e('pl'), 3, lang)}}}", f"\\newcommand{{\\rvEps{w}}}{{{num(e('g31ps'), 2, lang)}}}",
                  f"\\newcommand{{\\rvdPpl{w}}}{{{num(P['pl' + kk]['dP_W'], 3, lang)}}}",
                  f"\\newcommand{{\\rvdPps{w}}}{{{num(P['g31ps' + kk]['dP_W'], 3, lang)}}}",
                  f"\\newcommand{{\\rvEens{w}}}{{{thou(e('ens'), lang)}}}", f"\\newcommand{{\\rvEtt{w}}}{{{num(e('tt'), 1, lang)}}}",
                  f"\\newcommand{{\\rvEfb{w}}}{{{num(e('fb'), 1, lang)}}}",
                  f"\\newcommand{{\\rveRatio{w}}}{{{num(e('g31ps') / e('pl'), 1, lang)}}}",
                  f"\\newcommand{{\\rvEnsEPl{w}}}{{{thou(e('ens') / e('pl'), lang)}}}",
                  f"\\newcommand{{\\rvEnsEPs{w}}}{{{thou(e('ens') / e('g31ps'), lang)}}}",
                  f"\\newcommand{{\\rvTtEPl{w}}}{{{thou(e('tt') / e('pl'), lang)}}}",
                  f"\\newcommand{{\\rvTtEPs{w}}}{{{num(e('tt') / e('g31ps'), 1, lang)}}}",
                  f"\\newcommand{{\\rvTtMs{w}}}{{{num(L28['latency']['ttA' + kk]['median_ms'], 1, lang)}}}",
                  f"\\newcommand{{\\rvFbMs{w}}}{{{num(L28['latency']['fbA' + kk]['median_ms'], 1, lang)}}}",
                  f"\\newcommand{{\\rvTtLatPs{w}}}{{{num(L28['ratio']['tt' + kk]['lat_vs_g31ps'], 1, lang)}}}",
                  f"\\newcommand{{\\rvTtLatPl{w}}}{{{thou(L28['ratio']['tt' + kk]['lat_vs_g31pl'], lang)}}}",
                  f"\\newcommand{{\\rvTtMac{w}}}{{{num(L28['macs'][kk]['tt'] / 1e6, 1, lang)}}}"]
        open(os.path.join(a.out, f"rev_{lang}.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"ho 4: dat {n_pass}/{len(T)}, nguoc {n_rev}; Holm 24: {n24}; BETA tren bo {beta_trials} luot; golden max {gmax:.3f}")


if __name__ == "__main__":
    main()
