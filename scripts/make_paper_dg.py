"""Sinh macro muc 30 (chan doan DG-transfer tren Zhu uot) cho bai TBioCAS: \\rvDg..., so phep thang khi bo DG-transfer, va so phep thang tren BETA (tap doc lap).
Vao: results/dg_diag_2026_11_01/{S tu log, V0_zhu_wet_w1.00.csv}, review26/zhu tests. Dung: python3 make_paper_dg.py --out <dir>
"""
import argparse
import os

from make_paper_stats import num, rows

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
DG = "DG-transfer (nhân quả, trừ TB)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    v0 = rows(os.path.join(RES, "dg_diag_2026_11_01", "V0_zhu_wet_w1.00.csv"))
    assert len(v0) == 102
    l1 = 100 * sum(float(r["l1"]) for r in v0) / 102
    l2 = 100 * sum(float(r["l2"]) for r in v0) / 102
    lab = [r for r in rows(os.path.join(RES, "review26_analysis_2026_10_28", "review26_tests.csv")) if not r["so_sanh"].endswith(DG)]
    wear = [r for r in rows(os.path.join(RES, "zhu_analysis_2026_10_29", "zhu_tests.csv")) if not r["so_sanh"].endswith(DG)]
    assert len(lab) == 18 and len(wear) == 12, (len(lab), len(wear))
    beta = [r for r in rows(os.path.join(RES, "review26_analysis_2026_10_28", "review26_tests.csv")) if r["dataset"] == "beta"]
    assert len(beta) == 8
    win = lambda R: sum(float(r["p_holm"]) < 0.05 and float(r["chenh"]) > 0 for r in R)  # noqa: E731
    for lang in ("en", "vi"):
        L = ["% Sinh boi scripts/make_paper_dg.py; KHONG sua tay.",
             "\\newcommand{\\rvDgSyn}{100}",                                        # log job 18232: S 100,0%
             "\\newcommand{\\rvDgLone}{%s}" % num(l1, 1, lang), "\\newcommand{\\rvDgLtwo}{%s}" % num(l2, 1, lang),
             "\\newcommand{\\rvDgLabWin}{%d}" % win(lab), "\\newcommand{\\rvDgLabN}{%d}" % len(lab),
             "\\newcommand{\\rvDgWearWin}{%d}" % win(wear), "\\newcommand{\\rvDgWearN}{%d}" % len(wear),
             "\\newcommand{\\rvBetaWin}{%d}" % win(beta), "\\newcommand{\\rvBetaN}{%d}" % len(beta)]
        open(os.path.join(a.out, f"dg_{lang}.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
