"""WP3: FBCCA voi loc nhan qua (fb5c) va loc thuan-nghich (fb5) tren Benchmark, cua so 0,2-1,0 s.

Tinh rho^2 bang wp2_models.cca_features (GPU), diem = sum_n w_n rho_n^2 voi w_n = n^-1.25 + 0.25, Nh = 5.
Tu kiem (exit 3): ban fb5 phai tai tao FBCCA cua G0 (results/g0_trainfree_2026_10_02) o MOI cua so,
lech TB theo nguoi < 0,15 diem %.
Dau ra: <out_dir>/fbcca_<prep>_per_subject.csv (method,tag,dataset,window,seed,fold,subject,acc,n).
Dung: python wp3_fbcca.py --out_dir <dir>
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G0 = os.path.join(PROJ, "results", "g0_trainfree_2026_10_02", "fbcca_{ds}_per_subject.csv")
WINDOWS = [0.2, 0.4, 0.6, 0.8, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dataset", default="benchmark")
    ap.add_argument("--tag_ds", action="store_true", help="ten file co ten dataset (G3)")
    ap.add_argument("--preps", default="fb5,fb5c")
    a = ap.parse_args()
    import torch
    from ssvep.wp2_models import cca_features
    from train_loso import prepare
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g0 = {}
    if os.path.exists(G0.format(ds=a.dataset)):
        with open(G0.format(ds=a.dataset)) as f:
            for r in csv.DictReader(f):
                g0[(round(float(r["window"]), 2), int(r["subject"]))] = 100 * float(r["acc"])
    else:
        print(f"  [tu kiem] khong co G0 cho {a.dataset}: bo qua so voi G0", flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    w_n = torch.tensor([(n + 1) ** -1.25 + 0.25 for n in range(5)], device=dev)
    for prep in a.preps.split(","):
        out = os.path.join(a.out_dir, f"fbcca_{a.dataset}_{prep}_per_subject.csv" if a.tag_ds
                           else f"fbcca_{prep}_per_subject.csv")
        rows = []
        for win in WINDOWS:
            X, y, meta = prepare(a.dataset, win, prep)
            S, N = X.shape[:2]
            ft = cca_features(torch.from_numpy(X).to(dev).reshape(S * N, *X.shape[2:]),
                              meta["freqs"], meta["srate"])
            pred = (ft["band_nh5"] * w_n).sum(-1).argmax(-1).reshape(S, N).cpu().numpy()
            acc = [100 * float((pred[s] == y).mean()) for s in range(S)]
            print(f"{prep} w={win}: FBCCA {np.mean(acc):.3f}%", flush=True)
            if prep == "fb5" and g0:
                dev_ = np.mean([abs(acc[s] - g0[(win, s + 1)]) for s in range(S)])
                ok = abs(np.mean(acc) - np.mean([g0[(win, s + 1)] for s in range(S)])) < 0.15 and dev_ < 0.5
                print(f"  [tu kiem] so voi G0: TB G0 {np.mean([g0[(win, s + 1)] for s in range(S)]):.3f}%, "
                      f"lech TB theo nguoi {dev_:.3f} {'OK' if ok else 'LOI'}", flush=True)
                if not ok:
                    print("TU KIEM KHONG DAT")
                    sys.exit(3)
            for s in range(S):
                rows.append({"method": "fbcca", "tag": prep, "dataset": a.dataset, "window": f"{win:.2f}",
                             "seed": 0, "fold": s + 1, "subject": s + 1, "acc": f"{acc[s] / 100:.6f}", "n": N})
        with open(out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print("ghi", out, len(rows), f"dong (dich {S * len(WINDOWS)})")
    print("DONE")


if __name__ == "__main__":
    main()
