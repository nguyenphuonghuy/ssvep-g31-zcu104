"""Sinh macro chenh lech kiem dinh cho bai bao TBioCAS (ban Anh va ban Viet, cung ten macro). Chi thu vien chuan.

Vao (khong tinh lai, chi doc ket qua da chot):
  results/g31_baselines_analysis_2026_10_18/baselines_tests.csv   (Holm trong ho 2 moi bo x cua so)
  results/ensdnn_analysis_2026_10_18/ensdnn_tests.csv              (Holm trong ho 2)
  results/ensdnnc_analysis_2026_10_24/ensdnnc_tests.csv            (Holm trong ho 2)
  results/g3dc_analysis_2026_10_17/g3dc_tests.csv                  (khong-kem-hon, Holm ho 4)
  results/naka_dc_analysis_2026_10_15/naka_dc_tests.csv            (hai phia, Holm ho 2)
  results/heterogeneity_2026_10_27/heterogeneity_{tests,window}.csv (Spearman Holm ho 6, Wilcoxon Holm ho 3)
Ra: <out>/stats_en.tex, <out>/stats_vi.tex voi
  \\pd<Cmp><DS><W>  chenh G3.1 - doi chung (diem %), them dau * neu p Holm < 0,05
  \\pl<Cmp><DS><W>, \\ph<Cmp><DS><W>  can duoi/tren KTC 95%;  \\pp<Cmp><DS><W>  p Holm
  Cmp: Tt (tt-CCA goc), Dg (DG-transfer nhan qua), Ens (Ensemble-DNN), Ensc (Ensemble-DNN nhan qua), Dc (G3.1 - G3)
  DS: Bench, Beta, Naka;  W: A (0,4 s), B (1,0 s)
  Muc 25 (Ensemble-DNN pha khong): \\htd<DS><W> d TB, \\htr<DS><W> rho Spearman, \\htq<DS><W> p Holm cua rho;
  \\htw<DS> Delta = d(1,0) - d(0,4), \\htwl/\\htwh<DS> KTC 95%, \\htwn<DS> so nguoi thu hep / n, \\htwp<DS> p Holm,
  \\htc<DS> % nguoi Ensemble-DNN >= 95% o 1,0 s
Dung: python3 make_paper_stats.py --out <dir>
"""
import argparse
import csv
import math
import os

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
DSK = {"benchmark": "Bench", "beta": "Beta", "nakanishi": "Naka"}


def rows(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def wk(w):
    return {0.4: "A", 1.0: "B"}.get(round(float(w), 2))


def num(x, nd, lang, sign=False):
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    if lang == "vi":
        s = s.replace(".", ",")
    return s.replace("-", "$-$")


def pval(p, lang):
    if p < 1e-3:
        e = math.floor(math.log10(p))
        m = p / 10 ** e
        if round(m, 1) >= 10:
            m, e = m / 10, e + 1
        mm = f"{m:.1f}".replace(".", "{,}" if lang == "vi" else ".")
        return f"${mm} \\times 10^{{{e}}}$"
    s = f"{p:.3f}"
    return "$" + (s.replace(".", "{,}") if lang == "vi" else s) + "$"


def collect():
    out = []   # (cmp, ds, w, chenh, lo, hi, p_holm, dat)
    for r in rows(os.path.join(RES, "g31_baselines_analysis_2026_10_18", "baselines_tests.csv")):
        if wk(r["window"]):
            c = "Tt" if "tt-CCA" in r["so_sanh"] else "Dg"
            out.append((c, r["dataset"], wk(r["window"]), float(r["chenh"]), float(r["lo"]), float(r["hi"]),
                        float(r["p_holm"]), r["dat"] == "True"))
    for f, c in (("ensdnn_analysis_2026_10_18/ensdnn_tests.csv", "Ens"),
                 ("ensdnnc_analysis_2026_10_24/ensdnnc_tests.csv", "Ensc")):
        for r in rows(os.path.join(RES, f)):
            if r["so_sanh"].startswith("G3.1 ") and wk(r["window"]):
                out.append((c, r["dataset"], wk(r["window"]), float(r["chenh"]), float(r["lo"]), float(r["hi"]),
                            float(r["p_holm"]), r["dat"] == "True"))
    for r in rows(os.path.join(RES, "g3dc_analysis_2026_10_17", "g3dc_tests.csv")):
        if r["p_noninf_holm"]:                       # chi 2 cua so chinh co Holm
            out.append(("Dc", r["dataset"], wk(r["window"]), float(r["chenh"]), float(r["lo"]), float(r["hi"]),
                        float(r["p_noninf_holm"]), r["noninf_dat"] == "True"))
    for r in rows(os.path.join(RES, "naka_dc_analysis_2026_10_15", "naka_dc_tests.csv")):
        if r["so_sanh"].startswith("G3-dc") and wk(r["window"]):
            out.append(("Dc", "nakanishi", wk(r["window"]), float(r["chenh"]), float(r["lo"]), float(r["hi"]),
                        float(r["p_holm"]), r["dat"] == "True"))
    keys = [(c, d, w) for c, d, w, *_ in out]
    assert len(keys) == len(set(keys)) == 30, (len(keys), len(set(keys)))
    return out


def hetero(lang):
    hd = os.path.join(RES, "heterogeneity_2026_10_27")
    ht = [r for r in rows(os.path.join(hd, "heterogeneity_tests.csv")) if r["ensemble"] == "pha không"]
    hw = [r for r in rows(os.path.join(hd, "heterogeneity_window.csv")) if r["ensemble"] == "pha không"]
    assert len(ht) == 6 and len(hw) == 3
    L = []
    for r in ht:
        k = f"{DSK[r['dataset']]}{wk(r['window'])}"
        L.append(f"\\newcommand{{\\htd{k}}}{{{num(float(r['d_mean']), 2, lang, True)}}}")
        L.append(f"\\newcommand{{\\htr{k}}}{{{num(float(r['rho']), 2, lang, True)}}}")
        L.append(f"\\newcommand{{\\htq{k}}}{{{pval(float(r['p_holm']), lang)}}}")
        if r["window"] in ("1", "1.0"):
            L.append(f"\\newcommand{{\\htc{DSK[r['dataset']]}}}{{{num(100 * float(r['frac_ens_ge95']), 0, lang)}}}")
    for r in hw:
        k = DSK[r["dataset"]]
        L.append(f"\\newcommand{{\\htw{k}}}{{{num(float(r['delta']), 2, lang, True)}}}")
        L.append(f"\\newcommand{{\\htwl{k}}}{{{num(float(r['lo']), 2, lang, True)}}}")
        L.append(f"\\newcommand{{\\htwh{k}}}{{{num(float(r['hi']), 2, lang, True)}}}")
        L.append(f"\\newcommand{{\\htwn{k}}}{{{r['n_thu_hep']}/{r['n']}}}")
        L.append(f"\\newcommand{{\\htwp{k}}}{{{pval(float(r['p_holm']), lang)}}}")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = collect()
    os.makedirs(a.out, exist_ok=True)
    for lang in ("en", "vi"):
        L = ["% Sinh boi scripts/make_paper_stats.py; KHONG sua tay."]
        for c, d, w, ch, lo, hi, p, dat in data:
            k = f"{c}{DSK[d]}{w}"
            L.append(f"\\newcommand{{\\pd{k}}}{{{num(ch, 2, lang, True)}{'$^{*}$' if dat else ''}}}")
            L.append(f"\\newcommand{{\\pl{k}}}{{{num(lo, 2, lang, True)}}}")
            L.append(f"\\newcommand{{\\ph{k}}}{{{num(hi, 2, lang, True)}}}")
            L.append(f"\\newcommand{{\\pp{k}}}{{{pval(p, lang)}}}")
        n_sig = {c: sum(1 for cc, *_, dat in data if cc == c and dat) for c in ("Tt", "Dg", "Ens", "Ensc", "Dc")}
        for c, n in n_sig.items():
            L.append(f"\\newcommand{{\\pn{c}}}{{{n}}}")
        L += hetero(lang)
        open(os.path.join(a.out, f"stats_{lang}.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"{len(data)} so sanh; so dat: {n_sig}")


if __name__ == "__main__":
    main()
