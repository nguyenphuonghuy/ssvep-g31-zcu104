"""Sinh macro cong suat tuyet doi luc nghi va cong suat trung binh o mot quyet dinh moi cua so ( run2) cho bai TBioCAS.
Vao: results/zcu104_power_std_2026_10_29/run2/power_std_segments.json. Dung: python3 make_paper_power.py --out <dir>
Nghi = overlay da nap, xung PL chay, kernel ap_idle, PS ngu; cung duong nen cho moi dieu kien PS va PL (power_std.py).
"""
import argparse
import json
import os
import statistics as st
from collections import defaultdict

from make_paper_stats import num

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "zcu104_power_std_2026_10_29", "run2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    segs = json.load(open(os.path.join(RES, "power_std_segments.json")))["segments"]
    idle, E = defaultdict(list), defaultdict(list)
    for i, s in enumerate(segs):
        if s["label"] == "idle":
            nb = [x["label"] for x in segs[max(0, i - 1):i + 2] if x["label"] != "idle"]
            idle[nb[0][-3:]].append(s["mean"]["12V"])
        else:
            E[s["label"]].append(s["energy_per_decision_mJ"]["12V"])
    assert all(len(v) == 30 for v in idle.values()) and all(len(v) == 5 for v in E.values())
    rsd = max(st.stdev(v) / st.mean(v) for v in E.values())
    for lang in ("en", "vi"):
        L = ["% Sinh boi scripts/make_paper_power.py ( run2); KHONG sua tay.",
             "\\newcommand{\\rvIdleA}{%s}" % num(st.mean(idle["100"]), 2, lang),
             "\\newcommand{\\rvIdleB}{%s}" % num(st.mean(idle["250"]), 2, lang),
             "\\newcommand{\\rvIdleSd}{%s}" % num(max(st.stdev(v) for v in idle.values()), 3, lang),
             "\\newcommand{\\rvErsd}{%s}" % num(100 * rsd, 1, lang)]
        for w, T, sec in (("A", "100", 0.4), ("B", "250", 1.0)):
            L.append("\\newcommand{\\rvAvgPl%s}{%s}" % (w, num(st.mean(E["pl" + T]) / sec, 2, lang)))   # mJ/s = mW
            L.append("\\newcommand{\\rvAvgPs%s}{%s}" % (w, num(st.mean(E["g31ps" + T]) / sec, 1, lang)))
        open(os.path.join(a.out, f"power_{lang}.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
