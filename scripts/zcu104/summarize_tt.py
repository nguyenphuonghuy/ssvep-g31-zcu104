"""Tong hop do tt-CCA va FBCCA (fb5cd) tren PS ZCU104 va ti so so voi G3.1. Chi thu vien chuan.
Dung: python3 summarize_tt.py   -> results/zcu104_ttcca_ps_2026_10_29/summary.{md,json}
"""
import json
import os
import re
import statistics as st

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(PROJ, "results", "zcu104_ttcca_ps_2026_10_29")
G = os.path.join(PROJ, "results", "zcu104_g31_pl_2026_10_18")
log = open(os.path.join(R, "tt_lat.log")).read()
assert "CONG A: DAT" in log and "TT_DONE" in log and log.count("KHONG DAT") == 0
out = {"latency": {}, "power": {}, "g31": {}, "ratio": {}, "macs": {}}
for T in ("100", "250"):
    m = re.findall(rf"T={T} M=240.*?MAC dem duoc mot quyet dinh: tt (\d+) fb (\d+)", log, re.S)
    out["macs"][T] = dict(tt=int(m[0][0]), fb=int(m[0][1]))
    for v in ("A", "B"):
        for meth in ("tt", "fb"):
            rr = re.findall(rf"run=(\d) v={v} BENCH m={meth} T={T} R=200 median_us=([\d.]+) p95_us=([\d.]+).*?mean_filt_us=([\d.]+) "
                            rf"mean_corr_us=([\d.]+)", log)
            assert len(rr) == 7, (T, v, meth, len(rr))
            f = lambda i: st.mean(float(x[i]) for x in rr)  # noqa: E731
            out["latency"][f"{meth}{v}{T}"] = dict(median_ms=f(1) / 1e3, p95_ms=f(2) / 1e3, filt_ms=f(3) / 1e3, corr_ms=f(4) / 1e3,
                                                   runs=7, R=200)
S = json.load(open(os.path.join(R, "tt_power_segments.json")))
idl = [s["mean_W"]["12V"] for s in S["segments"] if s["label"] == "idle"]
out["power"]["idle_W"], out["power"]["idle_range_W"] = st.mean(idl), max(idl) - min(idl)
out["power"]["baseline_ok"] = out["power"]["idle_range_W"] <= 0.05
for meth in ("tt", "fb"):
    for T in ("100", "250"):
        seg = [s for s in S["segments"] if s["label"] == f"{meth}{T}"]
        out["power"][f"{meth}{T}"] = dict(dP_W=st.mean(s["delta_W"]["12V"] for s in seg), dP_sd=st.stdev(s["delta_W"]["12V"] for s in seg),
                                          t_ms=st.mean(s["t_per_decision_ms"] for s in seg), n=[s["n"] for s in seg],
                                          E_mJ=st.mean(s["energy_per_decision_mJ"]["12V"] for s in seg),
                                          E_sd=st.stdev(s["energy_per_decision_mJ"]["12V"] for s in seg))
ps = open(os.path.join(G, "ps", "golden_zcu104.log")).read()
for T in ("100", "250"):
    med = st.mean(float(x) for x in re.findall(rf"BENCH T={T} .*?median_us=([\d.]+)", ps))
    pl = json.load(open(os.path.join(G, f"g3_pl_rt_result_g31rt_T{T}.json")))["runs"][T]
    seg = json.load(open(os.path.join(G, f"g3_power_segments_g31rt_T{T}.json")))["segments"]
    out["g31"][T] = dict(ps_ms=med / 1e3, pl_us=st.mean(r["c_median_us"] for r in pl),
                         E_ps_mJ=st.mean(s["energy_per_decision_mJ"]["12V"] for s in seg if s["label"] == f"ps{T}"),
                         E_pl_mJ=st.mean(s["energy_per_decision_mJ"]["12V"] for s in seg if s["label"] == f"pl{T}"))
    g = out["g31"][T]
    for meth in ("tt", "fb"):
        a, p = out["latency"][f"{meth}A{T}"], out["power"][f"{meth}{T}"]
        out["ratio"][f"{meth}{T}"] = dict(lat_vs_g31ps=a["median_ms"] / g["ps_ms"], lat_vs_g31pl=a["median_ms"] * 1e3 / g["pl_us"],
                                          E_vs_g31ps=p["E_mJ"] / g["E_ps_mJ"], E_vs_g31pl=p["E_mJ"] / g["E_pl_mJ"])
json.dump(out, open(os.path.join(R, "summary.json"), "w"), indent=1)
NM = {"tt": "tt-CCA + bank lọc, trừ TB", "fb": "FBCCA, trừ TB"}
L = ["# tt-CCA và FBCCA (trừ TB) trên PS ZCU104", "",
     "Cổng 1–3 ĐẠT trên board cho A, B (240 lượt thử, 2 cửa sổ). Mô tả, không kiểm định.", "",
     "| Cửa sổ | Phương pháp | Bản | Trung vị (ms) | p95 (ms) | Lọc (ms) | Tương quan (ms) | MAC đếm được (triệu) |",
     "|---|---|---|---|---|---|---|---|"]
for T, w in (("100", "0,4 s"), ("250", "1,0 s")):
    for meth in ("tt", "fb"):
        for v in ("A", "B"):
            x = out["latency"][f"{meth}{v}{T}"]
            L.append(f"| {w} | {NM[meth]} | {v} | {x['median_ms']:.2f} | {x['p95_ms']:.2f} | {x['filt_ms']:.3f} | {x['corr_ms']:.2f} | "
                     f"{out['macs'][T][meth] / 1e6:.2f} |")
L += ["", f"Đường nền: các đoạn rảnh lệch nhau {out['power']['idle_range_W']:.3f} W "
      f"({'ổn định' if out['power']['baseline_ok'] else 'KHÔNG ổn định > 0,05 W: theo đặc tả phải đo lại'}).", "",
      "| Cửa sổ | Phương pháp | ΔP (W) | t (ms) | E (mJ) | E G3.1 PS (mJ) | E G3.1 PL (mJ) | Độ trễ / G3.1 PS | / G3.1 PL | "
      "E / G3.1 PS | E / G3.1 PL |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for T, w in (("100", "0,4 s"), ("250", "1,0 s")):
    g = out["g31"][T]
    for meth in ("tt", "fb"):
        p, r = out["power"][f"{meth}{T}"], out["ratio"][f"{meth}{T}"]
        L.append(f"| {w} | {NM[meth]} | {p['dP_W']:.3f} ± {p['dP_sd']:.3f} | {p['t_ms']:.2f} | {p['E_mJ']:.3f} ± {p['E_sd']:.3f} | "
                 f"{g['E_ps_mJ']:.3f} | {g['E_pl_mJ']:.4f} | {r['lat_vs_g31ps']:.1f}× | {r['lat_vs_g31pl']:.0f}× | {r['E_vs_g31ps']:.1f}× | "
                 f"{r['E_vs_g31pl']:.0f}× |")
open(os.path.join(R, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
