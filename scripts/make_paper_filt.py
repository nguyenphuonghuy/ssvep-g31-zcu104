"""Sinh macro so lieu muc 29 (khoi tao trang thai loc) cho bai TBioCAS: \\rvFa... (G3.1 − G3-ss), \\rvFb... (G3.1 − G3-st).
Vao: results/filtinit_analysis_2026_11_01/filtinit_tests.csv. Dung: python3 make_paper_filt.py --out <dir>
"""
import argparse
import math
import os

from make_paper_stats import num, rows

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
DSK = {"nakanishi": "Naka", "zhu_wet": "Wet"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = rows(os.path.join(RES, "filtinit_analysis_2026_11_01", "filtinit_tests.csv"))
    fa = [r for r in R if r["ho"] == "H29a"]
    fb = [r for r in R if r["ho"] == "H29b"]
    assert len(fa) == len(fb) == 4 and all(float(r["p_holm"]) < 0.05 for r in fb) and all(float(r["p_holm"]) >= 0.05 for r in fa)
    for lang in ("en", "vi"):
        L = ["% Sinh boi scripts/make_paper_filt.py; KHONG sua tay."]
        d = [float(r["chenh"]) for r in fa]
        L += [f"\\newcommand{{\\rvFaMin}}{{{num(min(d), 2, lang, True)}}}", f"\\newcommand{{\\rvFaMax}}{{{num(max(d), 2, lang, True)}}}",
              f"\\newcommand{{\\rvFaPmin}}{{{num(math.floor(min(float(r['p_holm']) for r in fa) * 100) / 100, 2, lang)}}}"]
        for r in fb:
            w = {0.4: "A", 1.0: "B"}[round(float(r["window"]), 1)]
            L.append(f"\\newcommand{{\\rvFb{DSK[r['dataset']]}{w}}}{{{num(float(r['chenh']), 2, lang)}}}")
        open(os.path.join(a.out, f"filt_{lang}.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
