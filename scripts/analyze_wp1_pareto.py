"""WP1 - do thi Pareto accuracy - chi phi cua 4 doi chuan hoc sau (chi thu vien chuan, login node).

Diem = (mo hinh, cua so). Accuracy = TB theo nguoi cua accuracy trung binh 5 seed (diem %).
Chi phi = MACs moi quyet dinh va so tham so (tu *_timing.csv, khong doi giua seed/fold).
Diem thuoc mat Pareto khi khong co diem nao khac co accuracy >= va MACs <= (it nhat mot bat dang
thuc chat). Xet ca hai cach: theo tung cua so (cung do tre quyet dinh) va gop moi cua so.
FBCCA chi la duong tham chieu accuracy (chi phi tinh theo MACs khong so sanh truc tiep duoc vi FBCCA
can phan ra QR/SVD).

Dung: /opt/anaconda3/bin/python3 analyze_wp1_pareto.py --out_dir results/wp1_pareto_2026_10_07
"""
import argparse
import csv
import math
import os
import statistics
import sys
from collections import defaultdict

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJ, "results")
DL_DIR = {"benchmark": "wp1_baselines_benchmark_2026_10_02", "beta": "wp1_baselines_beta_2026_10_04"}
G0 = os.path.join(RES, "g0_trainfree_2026_10_02")
N_SUBJ = {"benchmark": 35, "beta": 70}
MODELS = ["eegnet", "compactcnn", "fbtcnn", "guney"]
WINDOWS = ["0.20", "0.40", "0.60", "0.80", "1.00"]


def rows(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def mean_sem(v):
    return statistics.mean(v), statistics.stdev(v) / math.sqrt(len(v))


def load(ds):
    d = os.path.join(RES, DL_DIR[ds])
    acc = defaultdict(lambda: defaultdict(list))          # (m, w) -> subject -> [acc seed]
    cost = defaultdict(set)                                # (m, w) -> {(params, macs)}
    for fn in os.listdir(d):
        p = os.path.join(d, fn)
        if fn.endswith("_per_subject.csv"):
            for r in rows(p):
                if r["tag"] == "base":
                    acc[(r["method"], r["window"])][int(r["subject"])].append(float(r["acc"]))
        elif fn.endswith("_timing.csv"):
            for r in rows(p):
                cost[(r["method"], r["window"])].add((int(r["params"]), int(r["macs"])))
    pts = []
    for m in MODELS:
        for w in WINDOWS:
            a = acc[(m, w)]
            if sorted(a) != list(range(1, N_SUBJ[ds] + 1)) or any(len(v) != 5 for v in a.values()):
                sys.exit(f"LOI {ds} {m} w={w}: thieu nguoi/seed")
            if len(cost[(m, w)]) != 1:
                sys.exit(f"LOI {ds} {m} w={w}: chi phi khong duy nhat {cost[(m, w)]}")
            params, macs = next(iter(cost[(m, w)]))
            mu, se = mean_sem([100 * statistics.mean(v) for v in a.values()])
            pts.append(dict(dataset=ds, model=m, window=w, acc=mu, sem=se, params=params, macs=macs))
    fb = defaultdict(list)
    for r in rows(os.path.join(G0, f"fbcca_{ds}_per_subject.csv")):
        fb[f"{float(r['window']):.2f}"].append(100 * float(r["acc"]))
    return pts, {w: mean_sem(fb[w])[0] for w in WINDOWS}


def front(pts, key):
    out = set()
    for p in pts:
        dom = any(q is not p and q["acc"] >= p["acc"] and q[key] <= p[key] and
                  (q["acc"] > p["acc"] or q[key] < p[key]) for q in pts)
        if not dom:
            out.add(id(p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    out_csv = os.path.join(a.out_dir, "pareto_points.csv")
    if os.path.exists(out_csv) and os.environ.get("OVERWRITE") != "1":
        sys.exit(f"{out_csv} da ton tai; OVERWRITE=1 de ghi de")
    os.makedirs(a.out_dir, exist_ok=True)
    allp, L = [], ["# WP1 — Pareto accuracy – chi phí (đối chuẩn học sâu, LOSO không hiệu chuẩn)", "",
                   "Script: `scripts/analyze_wp1_pareto.py`. Accuracy: TB theo người của TB 5 seed (điểm %).",
                   "MACs: Conv2d + Linear mỗi quyết định (BN, kích hoạt, pooling không tính). "
                   "FBCCA chỉ là tham chiếu accuracy.",
                   "`P-w`: thuộc mặt Pareto trong cùng cửa sổ; `P-all`: mặt Pareto gộp mọi cửa sổ.", ""]
    for ds in DL_DIR:
        pts, fb = load(ds)
        fa = front(pts, "macs")
        fpa = front(pts, "params")
        for w in WINDOWS:
            sub = [p for p in pts if p["window"] == w]
            fw = front(sub, "macs")
            for p in sub:
                p["pareto_window_macs"] = id(p) in fw
        for p in pts:
            p["pareto_all_macs"] = id(p) in fa
            p["pareto_all_params"] = id(p) in fpa
            p["fbcca_acc"] = fb[p["window"]]
        allp += pts
        L += [f"## {ds}", "", "| Cửa sổ | Mô hình | Acc ± SEM | FBCCA | Tham số | MACs (M) | P-w | P-all |",
              "|---|---|---|---|---|---|---|---|"]
        for p in sorted(pts, key=lambda p: (p["window"], p["macs"])):
            L.append(f"| {p['window']} | {p['model']} | {p['acc']:.2f} ± {p['sem']:.2f} | {p['fbcca_acc']:.2f} | "
                     f"{p['params']} | {p['macs'] / 1e6:.2f} | {'x' if p['pareto_window_macs'] else ''} | "
                     f"{'x' if p['pareto_all_macs'] else ''} |")
        L.append("")
    keys = ["dataset", "model", "window", "acc", "sem", "fbcca_acc", "params", "macs",
            "pareto_window_macs", "pareto_all_macs", "pareto_all_params"]
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for p in allp:
            wr.writerow({k: (f"{p[k]:.4f}" if isinstance(p[k], float) else p[k]) for k in keys})
    open(os.path.join(a.out_dir, "pareto_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"ghi {out_csv} ({len(allp)} diem, dich 40)")


if __name__ == "__main__":
    main()
