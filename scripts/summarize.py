"""Tong hop *_per_subject.csv: accuracy va ITR trung binh +- SEM theo cua so.

Chi dung thu vien chuan (chay tren login node bang /opt/anaconda3/bin/python3).
ITR tinh cho TUNG doi tuong tu accuracy cua doi tuong do, roi lay trung binh.
Dung: python3 summarize.py <file.csv> [--gaze 0.55] [--targets 40]
"""
import argparse
import csv
import math
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ssvep.metrics import itr_bpm  # noqa: E402


def mean_sem(v):
    m = statistics.mean(v)
    s = statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0
    return m, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--gaze", type=float, default=0.55)
    ap.add_argument("--targets", type=int, default=40)
    a = ap.parse_args()
    by = defaultdict(dict)  # (method, dataset, window) -> {subject: [acc]}
    with open(a.csv) as f:
        for r in csv.DictReader(f):
            key = (r["method"], r["dataset"], float(r["window"]))
            seed = r.get("seed", "0")
            by[key].setdefault(r["subject"], []).append(float(r["acc"]))
            _ = seed
    print(f"{'method':8s} {'dataset':10s} {'win':>5s} {'n_sub':>5s} "
          f"{'acc% (SEM)':>14s} {'ITR bpm (SEM)':>16s}  gaze={a.gaze}s M={a.targets}")
    best = {}
    for (m, d, w) in sorted(by):
        subj = by[(m, d, w)]
        accs = [statistics.mean(v) for v in subj.values()]  # trung binh qua seed
        itrs = [itr_bpm(p, a.targets, w, a.gaze) for p in accs]
        am, asem = mean_sem(accs)
        im, isem = mean_sem(itrs)
        print(f"{m:8s} {d:10s} {w:5.2f} {len(accs):5d} {100 * am:7.2f} ({100 * asem:4.2f}) "
              f"{im:8.2f} ({isem:5.2f})")
        if im > best.get((m, d), (0, 0, 0))[0]:
            best[(m, d)] = (im, isem, w)
    for (m, d), (im, isem, w) in best.items():
        print(f"ITR cao nhat {m} {d}: {im:.2f} +- {isem:.2f} bpm tai {w:.2f} s")


if __name__ == "__main__":
    main()
