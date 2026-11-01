"""Tong hop ket qua G3 tren PL ZCU104. Chi thu vien chuan.
Dung: /opt/anaconda3/bin/python3 summarize_pl.py <results/zcu104_g3_pl_2026_10_14>
Doc: g3_pl_result.json, g3_power_segments.json, vivado_reports/{util_flat,util_hier,power,timing_summary}.rpt,
     ../zcu104_golden_ps_2026_10_14/golden_zcu104.log (moc PS). Ghi summary.md.
"""
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

R = Path(sys.argv[1])
L = ["# Bộ giải mã G3 trên PL ZCU104 — tổng hợp", ""]
pl = json.load(open(R / "g3_pl_result.json"))
L += [f"Overlay `{pl['overlay']}`, đồng hồ PL {pl['fclk0_mhz']:.2f} MHz; cửa sổ trên PL: {pl['windows']}.", ""]

# --- do tre ---
ps_log = (R.parent / "zcu104_golden_ps_2026_10_14" / "golden_zcu104.log").read_text()
L += ["## Độ trễ một trial (7 lượt × 2000 lần; trung vị và p95: TB qua 7 lượt, max: lớn nhất)", "",
      "| Cửa sổ | Đại lượng | Trung vị (µs) | p95 (µs) | Max (µs) | Lệch pred |", "|---|---|---|---|---|---|"]
for T, rows in pl["runs"].items():
    for key, name in (("kern", "PL: start → done (kernel + polling MMIO)"), ("e2e", "PL: trọn chu trình từ PS (Python PYNQ)")):
        L.append(f"| T{T} | {name} | {st.mean(r[key + '_median_us'] for r in rows):.1f} | "
                 f"{st.mean(r[key + '_p95_us'] for r in rows):.1f} | {max(r[key + '_max_us'] for r in rows):.1f} | "
                 f"{sum(r['pred_lech'] for r in rows)} |")
for T in (100, 250):
    m = re.findall(rf"BENCH T={T} .*?median_us=([\d.]+) p95_us=([\d.]+) min_us=[\d.]+ max_us=([\d.]+) pred_lech=(\d+)", ps_log)
    if m:
        L.append(f"| T{T} | PS: golden C++ 1 luồng (mục 12) | {st.mean(float(a[0]) for a in m):.1f} | "
                 f"{st.mean(float(a[1]) for a in m):.1f} | {max(float(a[2]) for a in m):.1f} | {sum(int(a[3]) for a in m)} |")

# --- tai nguyen ---
vr = R / "vivado_reports"
uf = (vr / "util_flat.rpt").read_text()
def util(name):
    m = re.search(rf"\|\s*{re.escape(name)}\s*\|\s*([\d.]+)\s*\|.*?\|\s*([\d.]+)\s*\|\s*([\d.<]+)\s*\|\n", uf)
    return m.groups() if m else ("?", "?", "?")
L += ["", "## Tài nguyên sau route (xczu7ev)", "", "| Tài nguyên | Dùng | Có | % |", "|---|---|---|---|"]
for n in ("CLB LUTs", "CLB Registers", "Block RAM Tile", "URAM", "DSPs"):
    u, a, p = util(n)
    L.append(f"| {n} | {u} | {a} | {p} |")
uh = (vr / "util_hier.rpt").read_text()
L += ["", "Theo khối (LUT tổng | FF | RAMB36 | RAMB18 | DSP):", ""]
for line in uh.splitlines():
    if re.match(r"\|\s+(g3_T\d+|sc_mem|ps_axi_periph)\s+\|", line):
        c = [x.strip() for x in line.strip("|").split("|")]
        L.append(f"- {c[0]}: {c[2]} | {c[6]} | {c[7]} | {c[8]} | {c[10]}")
tm = (vr / "timing_summary.rpt").read_text()
m = re.search(r"WNS\(ns\).*?\n.*?\n\s+([-\d.]+)\s+[-\d.]+\s+\d+\s+\d+\s+([-\d.]+)", tm)
if m:
    L += ["", f"Timing sau route: WNS = {m.group(1)} ns, WHS = {m.group(2)} ns."]

# --- cong suat uoc tinh ---
pw = (vr / "power.rpt").read_text()
def pv(label):
    m = re.search(rf"\|\s*{re.escape(label)}\s*\|\s*([\d.]+)", pw)
    return m.group(1) if m else "?"
L += ["", "## Công suất ước tính Vivado (vectorless, không SAIF; độ tin cậy " + pv("Confidence Level").replace("?", "Medium") + ")", "",
      f"Tổng trên chip {pv('Total On-Chip Power (W)')} W = động {pv('Dynamic (W)')} W + tĩnh {pv('Device Static (W)')} W.", ""]
for lab in ("Clocks", "CLB Logic", "Signals", "Block RAM", "DSPs", "PS8"):
    L.append(f"- {lab}: {pv(lab)} W")
for blk in ("g3_T100", "g3_T250", "ps", "sc_mem"):
    m = re.search(rf"\|\s+{blk}\s+\|\s*([\d.]+)\s*\|", pw)
    if m:
        L.append(f"- khối {blk}: {m.group(1)} W")

# --- cong suat do ---
segf = R / "g3_power_segments.json"
if segf.exists():
    S = json.load(open(segf))
    agg = defaultdict(lambda: defaultdict(list))
    for s in S["segments"]:
        if s["label"] == "idle":
            for k in ("12V", "INT"):
                agg["idle"][k + "_W"].append(s["mean_W"][k])
            continue
        agg[s["label"]]["t_ms"].append(s["t_per_decision_ms"])
        for k in ("12V", "INT"):
            agg[s["label"]][k + "_dW"].append(s["delta_W"][k])
            agg[s["label"]][k + "_mJ"].append(s["energy_per_decision_mJ"][k])
    f = lambda v: f"{st.mean(v):.3f} ± {st.stdev(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"  # noqa: E731
    L += ["", f"## Công suất đo trên board ({S['reps']} lặp × {S['dur']:.0f} s mỗi đoạn, lấy mẫu {S['dt']} s; TB ± SD qua lặp)", "",
          f"Idle (overlay đã nạp): 12V = {f(agg['idle']['12V_W'])} W; INT = {f(agg['idle']['INT_W'])} W.", "",
          "| Điều kiện | Thời gian/quyết định (ms) | ΔP 12V (W) | ΔP INT (W) | E/quyết định 12V (mJ) | E/quyết định INT (mJ) |",
          "|---|---|---|---|---|---|"]
    for lab in [x for x in agg if x != "idle"]:
        a = agg[lab]
        L.append(f"| {lab} | {f(a['t_ms'])} | {f(a['12V_dW'])} | {f(a['INT_dW'])} | {f(a['12V_mJ'])} | {f(a['INT_mJ'])} |")
    L += ["", "Ghi chú: plXXX gồm 1 lõi A53 bận vòng Python (chép trial, flush, polling); psXXX là golden C++ 1 luồng.",
          "ΔP so với TB hai đoạn idle kề. INT: nhãn PYNQ của rail lõi ~0,85 V [ánh xạ CẦN TÁC GIẢ KIỂM TRA]; độ phân giải ~1/16 W."]
(R / "summary.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
