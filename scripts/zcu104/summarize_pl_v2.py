"""Tong hop kernel G3 v2.2 tren PL ZCU104. Chi thu vien chuan.
Dung: /opt/anaconda3/bin/python3 summarize_pl_v2.py <results/zcu104_g3_pl_v2_2026_10_15>
Doc: g3_pl_rt_result_g3rt_T<T>.json, g3_power_segments_g3rt_T<T>.json, vivado_reports/{T<T>_*, hls_T<T>_*}.rpt,
     ../zcu104_golden_ps_2026_10_14/golden_zcu104.log. Ghi summary.md.
"""
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

R = Path(sys.argv[1])
VR = R / "vivado_reports"
WIN = {"100": "0,4 s", "250": "1,0 s"}
L = ["# Kernel G3 v2.2 trên PL ZCU104 — tổng hợp", "",
     "Tham số nạp khi chạy (một luồng đọc tuần tự + 5 checksum đọc ngược); mỗi cửa sổ một overlay; đồng hồ PL 187,5 MHz.", ""]

# ---- khop tung bit tren moi fold ----
L += ["## Khớp từng bit trên board (35 fold × 240 lượt thử mỗi cửa sổ)", "",
      "| Cửa sổ | Lỗi nạp (status/checksum) | Sai score | Sai pred | Accuracy TB theo fold (%) |", "|---|---|---|---|---|"]
res = {}
for T in ("100", "250"):
    p = R / f"g3_pl_rt_result_g3rt_T{T}.json"
    if not p.exists():
        continue
    d = json.load(open(p))
    res[T] = d
    rows = d["folds"][T]
    assert len(rows) == 35
    L.append(f"| {WIN[T]} | {sum(r['status'] != 1 for r in rows)} | {sum(r['sai_score'] for r in rows)} | "
             f"{sum(r['sai_pred'] for r in rows)} | {100 * st.mean(r['acc'] for r in rows):.2f} |")

# ---- do tre ----
ps = (R.parent / "zcu104_golden_ps_2026_10_14" / "golden_zcu104.log").read_text()
L += ["", "## Độ trễ một trial (7 lượt × 2 000 lần; trung vị, p95: TB qua 7 lượt; max lớn nhất)", "",
      "| Cửa sổ | Đại lượng | Trung vị (µs) | p95 (µs) | Max (µs) |", "|---|---|---|---|---|"]
for T, d in res.items():
    rr = d["runs"][T]
    for key, name in (("c", "PL start→done, polling bằng C"), ("e2e", "PL trọn chu trình, Python PYNQ")):
        L.append(f"| {WIN[T]} | {name} | {st.mean(r[key + '_median_us'] for r in rr):.1f} | "
                 f"{st.mean(r[key + '_p95_us'] for r in rr):.1f} | {max(r[key + '_max_us'] for r in rr):.1f} |")
    cs = (VR / f"hls_T{T}_cosim.rpt").read_text()
    cyc = int(re.search(r"\|\s*Verilog\|\s*Pass\|\s*(\d+)\|", cs).group(1))       # latency nho nhat = giai ma
    L.append(f"| {WIN[T]} | Cosim (giải mã, {cyc} chu kỳ @ {d['fclk0_mhz']:.1f} MHz) | {cyc / d['fclk0_mhz']:.1f} | | |")
    m = re.findall(rf"BENCH T={T} .*?median_us=([\d.]+) p95_us=([\d.]+)", ps)
    L.append(f"| {WIN[T]} | PS golden C++ 1 luồng (mục 12) | {st.mean(float(a[0]) for a in m):.1f} | "
             f"{st.mean(float(a[1]) for a in m):.1f} | |")

# ---- tai nguyen + cong suat uoc tinh ----
L += ["", "## Tài nguyên sau route và công suất ước tính Vivado (vectorless)", "",
      "| Overlay | LUT | FF | BRAM36 | DSP | Kernel ước tính (W) | Toàn chip ước tính (W) | WNS (ns) |", "|---|---|---|---|---|---|---|---|"]
for T in ("100", "250"):
    uf = (VR / f"T{T}_util_flat.rpt").read_text()
    def u(n):
        m = re.search(rf"\|\s*{re.escape(n)}\s*\|\s*([\d.]+)\s*\|.*?\|\s*([\d.<]+)\s*\|\n", uf)
        return f"{m.group(1)} ({m.group(2)}%)"
    pw = (VR / f"T{T}_power.rpt").read_text()
    ker = re.search(rf"\|\s+g3_T{T}\s+\|\s*([\d.]+)\s*\|", pw).group(1)
    tot = re.search(r"\|\s*Total On-Chip Power \(W\)\s*\|\s*([\d.]+)", pw).group(1)
    tm = (VR / f"T{T}_timing_summary.rpt").read_text()
    wns = re.search(r"WNS\(ns\).*?\n.*?\n\s+([-\d.]+)", tm).group(1)
    L.append(f"| g3rt_T{T} | {u('CLB LUTs')} | {u('CLB Registers')} | {u('Block RAM Tile')} | {u('DSPs')} | {ker} | {tot} | {wns} |")

# ---- cong suat do ----
L += ["", "## Công suất đo ở nguồn 12 V (3 lặp × 60 s; TB ± SD qua lặp; PL = vòng giải mã liên tục bằng C)", "",
      "| Overlay | Điều kiện | Thời gian/quyết định (ms) | ΔP (W) | Năng lượng/quyết định (mJ) |", "|---|---|---|---|---|"]
for T in ("100", "250"):
    p = R / f"g3_power_segments_g3rt_T{T}.json"
    if not p.exists():
        continue
    S = json.load(open(p))
    a = defaultdict(lambda: defaultdict(list))
    for s in S["segments"]:
        if s["label"] == "idle":
            a["idle"]["P"].append(s["mean_W"]["12V"])
            continue
        a[s["label"]]["t"].append(s["t_per_decision_ms"])
        a[s["label"]]["dP"].append(s["delta_W"]["12V"])
        a[s["label"]]["E"].append(s["energy_per_decision_mJ"]["12V"])
    f = lambda v, nd: f"{st.mean(v):.{nd}f} ± {st.stdev(v):.{nd}f}"  # noqa: E731
    L.append(f"| g3rt_T{T} | idle (overlay đã nạp) | | {f(a['idle']['P'], 3)} (tuyệt đối) | |")
    for k in [x for x in a if x != "idle"]:
        L.append(f"| g3rt_T{T} | {k} | {f(a[k]['t'], 3)} | {f(a[k]['dP'], 3)} | {f(a[k]['E'], 4)} |")
L += ["", "Ghi chú: ΔP gồm cả 1 lõi A53 bận polling (PL) hoặc tính toán (PS) và DDR; rail INT không dùng.",
      "Kết quả v2/v2.1 (cổng 5 không đạt) không dùng; log ở board ~/ssvep_lite_pkg/pl_rt/v21_fail."]
(R / "summary.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
