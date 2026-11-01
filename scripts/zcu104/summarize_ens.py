"""Tong hop do Ensemble-DNN tren PS ZCU104 va ti so so voi G3.1. Chi thu vien chuan.
Dung: /opt/anaconda3/bin/python3 summarize_ens.py   -> results/zcu104_ensdnn_ps_2026_10_21/summary.{md,json}
"""
import json
import os
import re
import statistics as st

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(PROJ, "results", "zcu104_ensdnn_ps_2026_10_21")
G = os.path.join(PROJ, "results", "zcu104_g31_pl_2026_10_18")
log = open(os.path.join(R, "ens_board.log")).read()
assert "CONG A: DAT" in log and "ENS_DONE" in log and log.count("KHONG DAT") == 0
out = {"latency": {}, "power": {}, "g31": {}, "ratio": {}}
for T in ("100", "250"):
    for v in ("A", "B", "NP"):
        rr = re.findall(rf"run=(\d) v={v} BENCH\w* T={T} N=34 R=200 median_us=([\d.]+) p95_us=([\d.]+).*?mean_filt_us=([\d.]+) "
                        rf"mean_dnn_us=([\d.]+) mean_sim_us=([\d.]+)", log)
        assert len(rr) == 7, (T, v, len(rr))
        f = lambda i: st.mean(float(x[i]) for x in rr)  # noqa: E731
        out["latency"][f"{v}{T}"] = dict(median_ms=f(1) / 1e3, p95_ms=f(2) / 1e3, filt_ms=f(3) / 1e3, dnn_ms=f(4) / 1e3,
                                         sim_ms=f(5) / 1e3, runs=7, R=200)
S = json.load(open(os.path.join(R, "ens_power_segments.json")))   # lan do lai 2026-10-24 (lan 1: *_lan1)
idl = [s["mean_W"]["12V"] for s in S["segments"] if s["label"] == "idle"]
assert max(idl) - min(idl) <= 0.05, "duong nen khong on dinh"
for T in ("100", "250"):
    seg = [s for s in S["segments"] if s["label"] == f"ens{T}"]
    out["power"][T] = dict(dP_W=st.mean(s["delta_W"]["12V"] for s in seg), dP_sd=st.stdev(s["delta_W"]["12V"] for s in seg),
                           t_ms=st.mean(s["t_per_decision_ms"] for s in seg), n=[s["n"] for s in seg],
                           E_mJ=st.mean(s["energy_per_decision_mJ"]["12V"] for s in seg),
                           E_sd=st.stdev(s["energy_per_decision_mJ"]["12V"] for s in seg))
idle = [s["mean_W"]["12V"] for s in S["segments"] if s["label"] == "idle"]
out["power"]["idle_W"] = st.mean(idle)
# G3.1 (muc 20): PS golden log, PL ket qua board, cong suat
ps = open(os.path.join(G, "ps", "golden_zcu104.log")).read()
for T in ("100", "250"):
    med = st.mean(float(x) for x in re.findall(rf"BENCH T={T} .*?median_us=([\d.]+)", ps))
    pl = json.load(open(os.path.join(G, f"g3_pl_rt_result_g31rt_T{T}.json")))["runs"][T]
    seg = json.load(open(os.path.join(G, f"g3_power_segments_g31rt_T{T}.json")))["segments"]
    Epl = st.mean(s["energy_per_decision_mJ"]["12V"] for s in seg if s["label"] == f"pl{T}")
    Eps = st.mean(s["energy_per_decision_mJ"]["12V"] for s in seg if s["label"] == f"ps{T}")
    out["g31"][T] = dict(ps_ms=med / 1e3, pl_us=st.mean(r["c_median_us"] for r in pl), E_ps_mJ=Eps, E_pl_mJ=Epl)
    a, g = out["latency"][f"A{T}"], out["g31"][T]
    out["ratio"][T] = dict(lat_A_vs_g31ps=a["median_ms"] / g["ps_ms"], lat_A_vs_g31pl=a["median_ms"] * 1e3 / g["pl_us"],
                           lat_B_vs_g31ps=out["latency"][f"B{T}"]["median_ms"] / g["ps_ms"],
                           E_A_vs_g31ps=out["power"][T]["E_mJ"] / g["E_ps_mJ"], E_A_vs_g31pl=out["power"][T]["E_mJ"] / g["E_pl_mJ"])
json.dump(out, open(os.path.join(R, "summary.json"), "w"), indent=1)
L = ["# Ensemble-DNN trên PS ZCU104", "", "Cổng 1–3 ĐẠT trên board cho A, B, NumPy (240 lượt thử, 2 cửa sổ).", "",
     "| Cửa sổ | Bản | Trung vị (ms) | p95 (ms) | Lọc (ms) | 34 DNN (ms) | Độ tương đồng (ms) |", "|---|---|---|---|---|---|---|"]
for T, w in (("100", "0,4 s"), ("250", "1,0 s")):
    for v in ("A", "B", "NP"):
        x = out["latency"][f"{v}{T}"]
        L.append(f"| {w} | {v} | {x['median_ms']:.1f} | {x['p95_ms']:.1f} | {x['filt_ms']:.2f} | {x['dnn_ms']:.1f} | {x['sim_ms']:.1f} |")
L += ["", "| Cửa sổ | ΔP (W) | t (ms) | E Ensemble-DNN A (mJ) | E G3.1 PS (mJ) | E G3.1 PL (mJ) | Độ trễ A / G3.1 PS | A / G3.1 PL | "
      "E A / G3.1 PS | E A / G3.1 PL |", "|---|---|---|---|---|---|---|---|---|---|"]
for T, w in (("100", "0,4 s"), ("250", "1,0 s")):
    p, g, r = out["power"][T], out["g31"][T], out["ratio"][T]
    L.append(f"| {w} | {p['dP_W']:.3f} ± {p['dP_sd']:.3f} | {p['t_ms']:.0f} | {p['E_mJ']:.0f} ± {p['E_sd']:.0f} | {g['E_ps_mJ']:.3f} | "
             f"{g['E_pl_mJ']:.4f} | {r['lat_A_vs_g31ps']:.0f}× | {r['lat_A_vs_g31pl']:.0f}× | {r['E_A_vs_g31ps']:.0f}× | "
             f"{r['E_A_vs_g31pl']:.0f}× |")
L.append(f"\nCông suất rảnh TB {out['power']['idle_W']:.2f} W; số quyết định mỗi đoạn 60 s: "
         f"{out['power']['100']['n']} (0,4 s), {out['power']['250']['n']} (1,0 s).")
open(os.path.join(R, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
