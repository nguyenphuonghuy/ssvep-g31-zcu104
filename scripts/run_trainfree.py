"""Danh gia CCA/FBCCA khong huan luyen tren tung doi tuong (khong can chia fold).

Dung: python run_trainfree.py --dataset benchmark --method fbcca \
          --windows 0.2,0.4,...,2.0 --out_dir results/<ten>
Ghi <out_dir>/<method>_<dataset>_per_subject.csv voi cac cot
method,dataset,window,subject,acc,n. Ghi tang dan theo doi tuong, bo qua doi
tuong da co; tu choi chay neu file da day du tru khi OVERWRITE=1.
"""
import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from ssvep import cca  # noqa: E402
from ssvep.data import load_cfg, load_subject, window  # noqa: E402

FIELDS = ["method", "dataset", "window", "subject", "acc", "n"]


def eval_subject(args):
    ds, subj, method, windows, n_fb, n_h = args
    X, meta = load_subject(ds, subj)
    fs = meta["srate"]
    nb, nk = X.shape[:2]
    fbs = cca.subband_filters(fs, n_fb) if method == "fbcca" else None
    bp = cca.bandpass_filter(fs) if method == "cca" else None
    rows = []
    for w in windows:
        Xw = window(X, meta, w).astype(np.float64)
        qrefs = cca.make_ref_bases(meta["freqs"], fs, Xw.shape[-1], n_h)
        correct = 0
        for b in range(nb):
            for k in range(nk):
                if method == "fbcca":
                    p, _ = cca.fbcca_predict(Xw[b, k], qrefs, fbs)
                else:
                    p, _ = cca.cca_predict(Xw[b, k], qrefs, bp)
                correct += int(p == k)
        rows.append({"method": method, "dataset": ds, "window": f"{w:.2f}",
                     "subject": subj, "acc": f"{correct / (nb * nk):.6f}",
                     "n": nb * nk})
    return subj, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", choices=["cca", "fbcca"], required=True)
    ap.add_argument("--windows", default="0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--n_fb", type=int, default=5)
    ap.add_argument("--n_h", type=int, default=5)
    ap.add_argument("--subjects", default="", help="vd 1,2,3; rong = tat ca")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    windows = [float(x) for x in a.windows.split(",")]
    n_sub = load_cfg(a.dataset)["n_subjects"]
    subjects = [int(s) for s in a.subjects.split(",")] if a.subjects else \
        list(range(1, n_sub + 1))
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"{a.method}_{a.dataset}_per_subject.csv")

    done = set()
    if os.path.exists(out):
        with open(out) as f:
            done = {int(r["subject"]) for r in csv.DictReader(f)}
        if set(subjects) <= done and os.environ.get("OVERWRITE") != "1":
            print(f"{out} da du {len(done)} doi tuong; dat OVERWRITE=1 de chay lai")
            return
        if os.environ.get("OVERWRITE") == "1":
            os.remove(out)
            done = set()
    todo = [s for s in subjects if s not in done]
    print(f"cau hinh: dataset={a.dataset} method={a.method} n_fb={a.n_fb} "
          f"n_h={a.n_h} windows={windows} subjects={len(todo)} (bo qua {len(done)})")

    new = not os.path.exists(out)
    ncpu = int(os.environ.get("NCPUS", 4))
    t0 = time.time()
    with open(out, "a", newline="") as f, Pool(ncpu) as pool:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            wr.writeheader()
        jobs = [(a.dataset, s, a.method, windows, a.n_fb, a.n_h) for s in todo]
        for subj, rows in pool.imap_unordered(eval_subject, jobs):
            wr.writerows(rows)
            f.flush()
            accs = " ".join(f"{r['window']}:{float(r['acc']):.3f}" for r in rows)
            print(f"S{subj} {accs} ({time.time() - t0:.0f}s)", flush=True)
    print("DONE", out)


if __name__ == "__main__":
    main()
