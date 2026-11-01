""": chan doan DG-transfer tren Zhu uot. Khong sua baselines_tf.py; dung lai ham cua no.
Dung: python diag_dg_zhu.py --test S|V0|V1|V3 --dataset zhu_wet --out_dir results/dg_diag_2026_11_01 [--workers 10]
"""
import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baselines_tf as B  # noqa: E402

G = B.G
ZHU_RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "zhu_baselines_2026_10_29")


def parts(Y, P):
    """Bon thanh phan pt (23)-(24), moi thanh phan sign(x) x^2, shape (4, K)."""
    l1 = B._corr_rows(P["vbar"] @ Y, P["Zbar"])
    pw = np.einsum("mkc,cn->mkn", P["wbar"], Y)
    pu = np.einsum("mkc,cn->mkn", P["ubar"], Y)
    l2 = B._corr_rows(pw, P["Tbar"]).mean(0)
    l3 = B._corr_rows(pu, np.broadcast_to(P["Zbar"], pu.shape)).mean(0)
    l4 = B._corr_rows(pu, P["Tbar"]).mean(0)
    return np.stack([np.sign(x) * x * x for x in (l1, l2, l3, l4)])


def run_fold(t):
    X, nbrs = G["X"], G["nbrs"]
    S, Nb, K, NBd = X.shape[:4]
    src = G["src_fn"](t)
    Gam = np.zeros((5, Nb, K, K))                                       # 0: tong nhu pt (25); 1-4: tung thanh phan
    viol = 0
    for j in range(NBd):
        cache1 = {m: G["step1"][j][m] + (X[m, :, :, j],) for m in src}
        P = B.huang_fold(src, nbrs, cache1)
        viol += P["viol"]
        for b in range(Nb):
            for k in range(K):
                p = parts(X[t, b, k, j], P)
                Gam[0, b, k] += B.BETA_W(j) * p.sum(0) ** 2
                Gam[1:, b, k] += B.BETA_W(j) * p ** 2
    acc = (Gam.argmax(-1) == np.arange(K)[None, None, :]).mean(axis=(1, 2))
    return t, acc, viol


def step1_one(args):
    m, j = args
    return m, j, B.step1_subject(G["X"][m, :, :, j], G["nbrs"])


def noise_index(X):
    """Trung vi theo luot thu cua |x| lon nhat o dai 1, moi nguoi."""
    return np.median(np.abs(X[:, :, :, 0]).max(axis=(-1, -2)), axis=(1, 2))


def synthetic():
    rng = np.random.default_rng(20261101)
    from ssvep.data import load_cfg
    freqs = load_cfg("zhu_wet")["freqs"]
    fs, N, K, C, S, Nb = 250, 250, 12, 8, 20, 10
    t_ = np.arange(1, N + 1) / fs
    mix = rng.standard_normal((C, 2))
    X = np.zeros((S, Nb, K, 1, C, N))
    for s in range(S):
        for b in range(Nb):
            for k in range(K):
                sig = np.stack([np.sin(2 * np.pi * freqs[k] * t_ + k * 0.5 * np.pi + 0.2 * s),
                                np.sin(4 * np.pi * freqs[k] * t_ + k * np.pi)])
                X[s, b, k, 0] = mix @ sig + rng.standard_normal((C, N))
    G.update(X=X, nbrs=B.neighbors("zhu_wet", freqs), src_fn=lambda t: [s for s in range(S) if s != t])
    G["step1"] = [{m: B.step1_subject(X[m, :, :, 0], G["nbrs"]) for m in range(S)}]
    acc = np.mean([run_fold(t)[1][0] for t in range(4)])
    print(f"S: tong hop kich thuoc Zhu, 4 fold: {100 * acc:.1f}% (dat neu >= 90)", flush=True)
    G.clear()
    return acc >= 0.9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, choices=["S", "V0", "V1", "V3"])
    ap.add_argument("--dataset", default="zhu_wet")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--folds", default="")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.test == "S":
        sys.exit(0 if synthetic() else 3)
    t0 = time.time()
    X, meta = B.load_fb(a.dataset, 1.0, "fb5cd")
    idx = noise_index(X)
    S = X.shape[0]
    with open(os.path.join(a.out_dir, f"noise_index_{a.dataset}.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["subject", "noise_index_uV"]); wr.writerows((s + 1, f"{idx[s]:.3f}") for s in range(S))
    print(f"{a.dataset}: chi so nhieu trung vi {np.median(idx):.1f} uV, min {idx.min():.1f}, max {idx.max():.1f}", flush=True)
    if a.test == "V1":
        X = (X - X.mean(-1, keepdims=True)) / (X.std(-1, keepdims=True) + 1e-12)
    src_fn = (lambda t: [s for s in range(S) if s != t])
    if a.test == "V3":
        order = np.argsort(idx)
        src_fn = (lambda t: [int(s) for s in order if s != t][:20])
    G.update(X=X, nbrs=B.neighbors(a.dataset, meta["freqs"]), src_fn=src_fn)
    with Pool(a.workers) as pool:
        st1 = pool.map(step1_one, [(m, j) for m in range(S) for j in range(X.shape[3])], chunksize=1)
    G["step1"] = [{m: r for (m, jj, r) in st1 if jj == j} for j in range(X.shape[3])]
    del st1
    folds = [int(x) - 1 for x in a.folds.split(",")] if a.folds else list(range(S))
    with Pool(a.workers) as pool:
        res = sorted(pool.map(run_fold, folds, chunksize=1), key=lambda r: r[0])
    out = os.path.join(a.out_dir, f"{a.test}_{a.dataset}_w1.00.csv")
    with open(out, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["subject", "acc", "l1", "l2", "l3", "l4", "viol"])
        for t, acc, v in res:
            wr.writerow([t + 1] + [f"{x:.6f}" for x in acc] + [v])
    M = np.array([r[1] for r in res])
    print(f"{a.test} {a.dataset} w=1.0: acc TB {100 * M[:, 0].mean():.2f}%, trung vi {100 * np.median(M[:, 0]):.2f}%; "
          f"l1-l4 TB {np.round(100 * M[:, 1:].mean(0), 2)}; vi pham {sum(r[2] for r in res)} ({time.time() - t0:.0f}s)", flush=True)
    if a.test == "V0" and a.dataset == "zhu_wet":
        ref = {int(r["subject"]): float(r["acc"]) for r in csv.DictReader(open(os.path.join(ZHU_RES, "huang_fb5cd_zhu_wet_per_subject.csv")))
               if float(r["window"]) == 1.0}
        d = max(abs(ref[t + 1] - acc[0]) for t, acc, _ in res)
        print(f"V0 lech so voi muc 27: {d:.2e}", flush=True)
        if d > 1e-6:
            sys.exit(3)


if __name__ == "__main__":
    main()
