"""Doi chung khong hieu chuan.

- ttcca      : tt-CCA goc (Yuan 2015, J Neural Eng 12:046006, Hinh 2): rho = rho1 + rho2 + rho3, Nh = 2, khong loc.
- ttcca_fb5c : bien the phu: bank loc nhan qua 5 dai (nhu G3), Nh = 5, Gamma = sum_j beta_j * rho^(j).
- huang_fb5c / huang_fb5 : DG-transfer (Huang 2023, IEEE TNSRE 31:3307), pt (1)-(26); buoc 3 = SLSQP.
- huang_fb5cd / ttcca_fb5cd : nhu tren voi prep fb5cd (tru TB doan truoc kich thich truoc bank loc nhan qua).
LOSO: nguoi dich = tung nguoi; nguoi nguon = moi nguoi con lai. Tat dinh.
Dau ra <out_dir>/{method}_{ds}_per_subject.csv (method,tag,dataset,window,seed,fold,subject,acc,n).
Dung: python baselines_tf.py --out_dir D --datasets benchmark,beta,nakanishi --methods ttcca,ttcca_fb5c,huang_fb5c,huang_fb5
      [--windows 0.2,...] [--folds 1,2] [--workers 10] [--selftest_only]
"""
import argparse
import multiprocessing as mp
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FIELDS = ["method", "tag", "dataset", "window", "seed", "fold", "subject", "acc", "n"]
BETA_W = lambda j: (j + 1) ** -1.25 + 0.25  # noqa: E731  (j = 0..4)


def fail(msg):
    print(f"TU KIEM KHONG DAT: {msg}", flush=True)
    sys.exit(3)


# ===================== CCA co vector trong so =====================
def _center(a):
    return a - a.mean(axis=-1, keepdims=True)


def cca(X, Y):
    """X (C, N), Y (P, N) -> (rho, wx (C,), wy (P,)) lon nhat (QR + SVD)."""
    Xc, Yc = _center(X).T, _center(Y).T
    Qx, Rx = np.linalg.qr(Xc)
    Qy, Ry = np.linalg.qr(Yc)
    U, s, Vt = np.linalg.svd(Qx.T @ Qy)
    wx = np.linalg.lstsq(Rx, U[:, 0], rcond=None)[0]
    wy = np.linalg.lstsq(Ry, Vt[0], rcond=None)[0]
    return min(s[0], 1.0), wx, wy


def pearson(a, b):
    a, b = _center(a), _center(b)
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def sincos(freqs, fs, N, nh):
    from ssvep.cca import sine_refs
    return [np.asarray(r) for r in sine_refs(freqs, fs, N, nh)]


# ===================== tt-CCA (Yuan 2015) =====================
def ttcca_scores(X, Xbar, refs, wxbar):
    """X (C,N) luot thu; Xbar (K,C,N) template chuyen; refs list K x (2Nh,N); wxbar (K,C) tu CCA(Xbar_k, Y_k)."""
    K = len(refs)
    sc = np.zeros(K)
    for k in range(K):
        rho2, wx, _ = cca(X, refs[k])
        rho1 = pearson(wxbar[k] @ Xbar[k], wxbar[k] @ X)
        rho3 = pearson(wx @ Xbar[k], wx @ X)
        sc[k] = rho1 + rho2 + rho3
    return sc


# ===================== DG-transfer (Huang 2023) =====================
def gen_eig_top(A, B):
    """Vector rieng ung tri rieng lon nhat cua B^-1 A (A, B doi xung, B xac dinh duong)."""
    from scipy.linalg import eigh
    vals, vecs = eigh(A, B)
    return vecs[:, -1], vals[-1]


def cov(a, b=None):
    a = _center(a)
    b = a if b is None else _center(b)
    return a @ b.T


def align(vs):
    """Can dau cac vector theo vector dau (bai khong noi)."""
    out, sg = [vs[0]], [1.0]
    for v in vs[1:]:
        s = 1.0 if np.dot(v, vs[0]) >= 0 else -1.0
        out.append(s * v)
        sg.append(s)
    return out, sg


def step1_subject(Xs, nbrs):
    """Buoc 1 (pt 6-11) cho MOT nguoi nguon. Xs (Nb, K, C, N). Tra w (K, C) TB theo b (da can dau),
    T (K, N) TB theo b, va w_b (Nb, K, C), T_b (Nb, K, N), Xbar_b (Nb, K, C, N) de dung o buoc 2-3."""
    Nb, K, C, N = Xs.shape
    w_b = np.zeros((Nb, K, C))
    T_b = np.zeros((Nb, K, N))
    Xbar_b = np.zeros((Nb, K, C, N))
    for b in range(Nb):
        other = [i for i in range(Nb) if i != b]
        Xbar_b[b] = Xs[other].mean(axis=0)                                    # pt (8)-(9) (Sigma toi Nt)
    for k in range(K):
        group = [k] + list(nbrs[k])
        for b in range(Nb):
            other = [i for i in range(Nb) if i != b]
            S = sum(cov(Xbar_b[b, g]) for g in group)                           # pt (7)
            Q = sum(cov(Xs[i, g]) for g in group for i in other)                # pt (10)
            w, _ = gen_eig_top(S, Q + 1e-9 * np.trace(Q) / C * np.eye(C))
            w_b[b, k] = w
        al, sg = align(list(w_b[:, k]))
        for b in range(Nb):
            w_b[b, k] = al[b]
            T_b[b, k] = w_b[b, k] @ Xbar_b[b, k]                                # pt (11)
    return w_b, T_b, Xbar_b


def step2(xbars, nbrs):
    """Buoc 2 (pt 12-15). xbars: list M mang Xbar_b (Nb, K, C, N) cua tung nguoi nguon (khong xep chong de tiet kiem
    bo nho) -> v_b (Nb, K, C), Z_b (Nb, K, N) (da can dau)."""
    M = len(xbars)
    Nb, K, C, N = xbars[0].shape
    v_b = np.zeros((Nb, K, C))
    Z_b = np.zeros((Nb, K, N))
    for k in range(K):
        group = [k] + list(nbrs[k])
        for b in range(Nb):
            P = np.zeros((C, C))
            R = np.zeros((C, C))
            for g in group:
                ssum = np.zeros((C, N))
                auto = np.zeros((C, C))
                for xb in xbars:
                    A = _center(xb[b, g])
                    ssum += A
                    auto += A @ A.T
                P += ssum @ ssum.T - auto                                         # sum_{m1 != m2} A_m1 A_m2^T
                R += auto
            P /= M * (M - 1)                                                      # pt (13)
            R /= M                                                                # pt (14)
            v, _ = gen_eig_top((P + P.T) / 2, R + 1e-9 * np.trace(R) / C * np.eye(C))
            v_b[b, k] = v
        al, _ = align(list(v_b[:, k]))
        for b in range(Nb):
            v_b[b, k] = al[b]
            Z_b[b, k] = v_b[b, k] @ (sum(xb[b, k] for xb in xbars) / M)          # pt (15)
    return v_b, Z_b


def _obj_grad(u, X, t, z):
    """-(rho(u^T X, t) + rho(u^T X, z)) va gradient; X, t, z da tru trung binh."""
    a = u @ X
    na = np.sqrt(a @ a) + 1e-12
    f, g = 0.0, np.zeros_like(u)
    for r in (t, z):
        nr = np.sqrt(r @ r) + 1e-12
        c = (a @ r) / (na * nr)
        f += c
        g += (X @ r) / (na * nr) - c * (X @ a) / (na * na)
    return -f, -g


def step3(X1, T, Z, u0):
    """Buoc 3 (pt 16-17): max rho(u^T X1, T) + rho(u^T X1, Z), sum(u) = 0, ||u|| = 1, SLSQP."""
    from scipy.optimize import minimize
    Xc, t, z = _center(X1), T - T.mean(), Z - Z.mean()
    C = len(u0)
    u0 = u0 - u0.mean()
    u0 = u0 / (np.linalg.norm(u0) + 1e-12)
    cons = [{"type": "eq", "fun": lambda u: np.array([u.sum()]), "jac": lambda u: np.ones((1, C))},
            {"type": "eq", "fun": lambda u: np.array([u @ u - 1.0]), "jac": lambda u: 2 * u[None, :]}]
    r = minimize(_obj_grad, u0, args=(Xc, t, z), jac=True, method="SLSQP", constraints=cons,
                 options={"maxiter": 100, "ftol": 1e-9})
    u = r.x
    f0 = -_obj_grad(u0, Xc, t, z)[0]
    f1 = -_obj_grad(u, Xc, t, z)[0]
    return (u, f0, f1) if f1 >= f0 - 1e-9 else (u0, f0, f0)


def huang_fold(src, nbrs, cache1):
    """Huan luyen cho mot fold. src: list chi so nguoi nguon; cache1[m] = buoc 1 cua nguoi m (khong phu thuoc fold)."""
    v_b, Z_b = step2([cache1[m][2] for m in src], nbrs)
    M, Nb, K = len(src), v_b.shape[0], v_b.shape[1]
    wbar = np.stack([cache1[m][0].mean(axis=0) for m in src])                        # (M, K, C) pt (18)
    Tbar = np.stack([cache1[m][1].mean(axis=0) for m in src])                        # (M, K, N) pt (21)
    ubar = np.zeros_like(wbar)
    viol = 0
    for i, m in enumerate(src):
        w_b, T_b, _, Xs = cache1[m]
        for k in range(K):
            us = []
            for b in range(Nb):
                u, f0, f1 = step3(Xs[b, k], T_b[b, k], Z_b[b, k], w_b[b, k])
                if abs(u.sum()) > 1e-6 or abs(np.linalg.norm(u) - 1) > 1e-6 or f1 < f0 - 1e-9:
                    viol += 1
                us.append(u)
            ubar[i, k] = np.mean(align(us)[0], axis=0)                                # pt (20)
    return dict(vbar=v_b.mean(axis=0), Zbar=Z_b.mean(axis=0), wbar=wbar, Tbar=Tbar, ubar=ubar, viol=viol)


def _corr_rows(A, B):
    """Pearson theo truc cuoi giua cac hang tuong ung cua A, B (cung shape (..., N))."""
    A = A - A.mean(-1, keepdims=True)
    B = B - B.mean(-1, keepdims=True)
    d = np.sqrt((A * A).sum(-1) * (B * B).sum(-1))
    return np.where(d > 0, (A * B).sum(-1) / np.where(d > 0, d, 1), 0.0)


def huang_gamma(Y, P):
    """pt (23)-(24) cho mot dai con: Y (C, N) -> gamma (K,)."""
    l1 = _corr_rows(P["vbar"] @ Y, P["Zbar"])                                     # (K,)
    pw = np.einsum("mkc,cn->mkn", P["wbar"], Y)
    pu = np.einsum("mkc,cn->mkn", P["ubar"], Y)
    l2 = _corr_rows(pw, P["Tbar"]).mean(0)
    l3 = _corr_rows(pu, np.broadcast_to(P["Zbar"], pu.shape)).mean(0)
    l4 = _corr_rows(pu, P["Tbar"]).mean(0)
    return sum(np.sign(x) * x * x for x in (l1, l2, l3, l4))


# ===================== lang gieng theo bo cuc hien thi =====================
def neighbors(ds, freqs):
    f = np.round(np.asarray(freqs, dtype=float), 2)
    idx = {round(v, 2): i for i, v in enumerate(f)}
    if ds == "benchmark":                                   # luoi 5 x 8: hang r, cot c, f = 8 + c + 0,2 r (Huang Hinh 1b)
        pos = {idx[round(8 + c + 0.2 * r, 2)]: (r, c) for r in range(5) for c in range(8)}
        return [[j for j, (r2, c2) in pos.items() if abs(r2 - pos[i][0]) + abs(c2 - pos[i][1]) == 1] for i in range(len(f))]
    if ds in ("nakanishi", "zhu_dry", "zhu_wet"):          # Zhu 2021 cung bang tan so/ban phim; luoi 4 x 3 (Huang Hinh 1a): cot c, hang r, f = 9,25 + 2c + 0,5r
        pos = {idx[round(9.25 + 2 * c + 0.5 * r, 2)]: (r, c) for r in range(4) for c in range(3)}
        return [[j for j, (r2, c2) in pos.items() if abs(r2 - pos[i][0]) + abs(c2 - pos[i][1]) == 1] for i in range(len(f))]
    if ds == "beta":                                        # QWERTY so le (BETA Hinh 1A; f theo pt (2) cua bai BETA)
        order = [".", ",", "<"] + [chr(c) for c in range(ord("a"), ord("z") + 1)] + [str(d) for d in range(10)] + ["_"]
        key_f = {k: round(8 + 0.2 * i, 2) for i, k in enumerate(order)}
        rows = [(list("1234567890<"), 0.0), (list("qwertyuiop"), 0.5), (list("asdfghjkl"), 1.0), (list("zxcvbnm."), 1.5)]
        geo = {}                                            # phim -> (hang, x_trai, x_phai) theo don vi do rong phim
        for r, (keys, off) in enumerate(rows):
            for i, k in enumerate(keys):
                geo[k] = (r, off + i, off + i + 1)
        geo["_"] = (4, 1.0, 7.0)                            # phim cach (uoc luong tu hinh)
        geo[","] = (4, 8.0, 9.0)                            # dau phay (uoc luong tu hinh)
        out = []
        for i in range(len(f)):
            k = [kk for kk, v in key_f.items() if v == f[i]][0]
            r, a, b = geo[k]
            nb = []
            for k2, (r2, a2, b2) in geo.items():
                if k2 == k:
                    continue
                horiz = r2 == r and (abs(a2 - b) < 1e-9 or abs(b2 - a) < 1e-9)
                vert = abs(r2 - r) == 1 and min(b, b2) - max(a, a2) > 1e-9
                if horiz or vert:
                    nb.append(idx[key_f[k2]])
            out.append(sorted(nb))
        return out
    raise ValueError(ds)


# ===================== du lieu =====================
def load_raw(ds, win):
    """Cua so sau do tre, KHONG loc: (S, Nb, K, 1, C, N)."""
    from ssvep.data import load_cfg, load_subject, window
    Xs = []
    for s in range(1, load_cfg(ds)["n_subjects"] + 1):
        X, meta = load_subject(ds, s)
        Xs.append(window(X, meta, win).astype(np.float64)[:, :, None])                # Nb, K, 1, C, N
    return np.stack(Xs), meta


def load_fb(ds, win, prep):
    """Bank loc 5 dai (fb5c nhan qua / fb5cd nhan qua co tru TB / fb5 pha khong) nhu G3: (S, Nb, K, 5, C, N)."""
    assert prep in ("fb5", "fb5c", "fb5cd"), prep
    from train_loso import prepare
    X, y, meta = prepare(ds, win, prep)                                               # (S, Nb*K, 5, C, N), y khoi-chinh
    K = len(meta["freqs"])
    S, NT = X.shape[:2]
    assert (y == np.tile(np.arange(K), NT // K)).all()
    return X.reshape(S, NT // K, K, *X.shape[2:]).astype(np.float64), meta


# ===================== chay mot fold (tien trinh con) =====================
G = {}   # du lieu dung chung: gan TRUOC khi tao Pool (fork, copy-on-write), khong truyen qua pickle


def run_ttcca(t):
    X, refs, nh_w = G["X"], G["refs"], G["fbw"]
    S, Nb, K, NBd = X.shape[:4]
    src = [s for s in range(S) if s != t]
    tot = np.zeros((Nb, K, K))
    for j in range(NBd):
        Xbar = X[src, :, :, j].mean(axis=(0, 1))                                    # (K, C, N) TB moi nguoi nguon, moi khoi
        wxbar = np.stack([cca(Xbar[k], refs[k])[1] for k in range(K)])
        for b in range(Nb):
            for k in range(K):
                tot[b, k] += nh_w[j] * ttcca_scores(X[t, b, k, j], Xbar, refs, wxbar)
    pred = tot.argmax(-1)
    return t, float((pred == np.arange(K)[None, :]).mean()), Nb * K


def run_huang(t):
    X, nbrs = G["X"], G["nbrs"]
    S, Nb, K, NBd = X.shape[:4]
    src = [s for s in range(S) if s != t]
    Gam = np.zeros((Nb, K, K))
    viol = 0
    for j in range(NBd):
        cache1 = {m: G["step1"][j][m] + (X[m, :, :, j],) for m in src}
        P = huang_fold(src, nbrs, cache1)
        viol += P["viol"]
        for b in range(Nb):
            for k in range(K):
                Gam[b, k] += BETA_W(j) * huang_gamma(X[t, b, k, j], P) ** 2         # pt (25)
    pred = Gam.argmax(-1)
    return t, float((pred == np.arange(K)[None, :]).mean()), Nb * K, viol


def _step1_one(args):
    m, j = args
    w_b, T_b, Xbar_b = step1_subject(G["X"][m, :, :, j], G["nbrs"])
    return m, j, (w_b, T_b, Xbar_b)


# ===================== tu kiem =====================
def selftest():
    from ssvep.cca import _orth, cca_corr
    rng = np.random.default_rng(0)
    for _ in range(5):
        X, Y = rng.standard_normal((9, 100)), rng.standard_normal((4, 100))
        r, wx, wy = cca(X, Y)
        r0 = cca_corr(_orth(X), _orth(Y))
        if abs(r - r0) > 1e-8 or abs(pearson(wx @ X, wy @ Y) - r) > 1e-8:
            fail(f"(i) CCA tu cai {r} vs {r0}")
    # (ii) rho2 = CCA chuan
    X, refs = rng.standard_normal((9, 100)), sincos([8.0, 9.0], 250, 100, 2)
    r2 = cca(X, refs[0])[0]
    if abs(r2 - cca_corr(_orth(X), _orth(refs[0]))) > 1e-8:
        fail("(ii) rho2")
    # (iii) vector rieng
    A = rng.standard_normal((9, 9)); A = A @ A.T
    B = rng.standard_normal((9, 9)); B = B @ B.T + 9 * np.eye(9)
    w, lam = gen_eig_top(A, B)
    if np.abs(np.linalg.solve(B, A @ w) - lam * w).max() > 1e-6 * max(1, lam):
        fail("(iii) Q^-1 S w = lam w")
    for _ in range(200):
        z = rng.standard_normal(9)
        if (z @ A @ z) / (z @ B @ z) > lam + 1e-9:
            fail("(iii) khong phai tri rieng lon nhat")
    # (iv) buoc 3
    Xs, T, Z = rng.standard_normal((9, 100)), rng.standard_normal(100), rng.standard_normal(100)
    u, f0, f1 = step3(Xs, T, Z, rng.standard_normal(9))
    if abs(u.sum()) > 1e-6 or abs(np.linalg.norm(u) - 1) > 1e-6 or f1 < f0 - 1e-9:
        fail("(iv) rang buoc / muc tieu buoc 3")
    # (v) du lieu tong hop: 6 nguoi, 4 khoi, 6 muc tieu (luoi 2 x 3), 9 kenh, 0,5 s
    fs, N, K, C, S, Nb = 250, 125, 6, 9, 6, 4
    fr = [8.0, 9.0, 10.0, 8.2, 9.2, 10.2]
    t_ = np.arange(1, N + 1) / fs
    mix = rng.standard_normal((C, 2))
    Xsyn = np.zeros((S, Nb, K, 1, C, N))
    for s in range(S):
        ph = rng.uniform(0, 2 * np.pi)
        for b in range(Nb):
            for k in range(K):
                sig = np.stack([np.sin(2 * np.pi * fr[k] * t_ + k * 0.5 * np.pi + 0.2 * s),
                                np.sin(4 * np.pi * fr[k] * t_ + k * np.pi)])
                Xsyn[s, b, k, 0] = mix @ sig + 1.0 * rng.standard_normal((C, N))
    nb_syn = [[j for j in range(K) if (j // 3 == i // 3 and abs(j - i) == 1) or abs(j - i) == 3] for i in range(K)]
    G.update(X=Xsyn, refs=sincos(fr, fs, N, 2), fbw=[1.0], nbrs=nb_syn)
    G["step1"] = [{m: step1_subject(Xsyn[m, :, :, 0], nb_syn) for m in range(S)}]
    acc_tt = np.mean([run_ttcca(t)[1] for t in range(S)])
    res = [run_huang(t) for t in range(S)]
    acc_h, viol = np.mean([r[1] for r in res]), sum(r[3] for r in res)
    print(f"[tu kiem] (i)-(iv) DAT; (v) tong hop: tt-CCA {100 * acc_tt:.1f}%, Huang {100 * acc_h:.1f}% (vi pham buoc 3: {viol})",
          flush=True)
    if acc_tt < 0.9 or acc_h < 0.9 or viol:
        fail("(v) du lieu tong hop")
    G.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--datasets", default="benchmark,beta,nakanishi")
    ap.add_argument("--methods", default="ttcca,ttcca_fb5c,huang_fb5c,huang_fb5")
    ap.add_argument("--windows", default="0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--folds", default="")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("NCPUS", 4)))
    ap.add_argument("--selftest_only", action="store_true")
    a = ap.parse_args()
    mp.set_start_method("fork", force=True)
    selftest()
    if a.selftest_only:
        return
    os.makedirs(a.out_dir, exist_ok=True)
    for ds in a.datasets.split(","):
        for meth in a.methods.split(","):
            out = os.path.join(a.out_dir, f"{meth}_{ds}_per_subject.csv")
            done = set()
            if os.path.exists(out):
                with open(out) as f:
                    done = {(r["window"], int(r["subject"])) for r in csv.DictReader(f)}
            for win in [float(w) for w in a.windows.split(",")]:
                t0 = time.time()
                if meth == "ttcca":
                    X, meta = load_raw(ds, win)
                    nh, fbw = 2, [1.0]
                else:
                    X, meta = load_fb(ds, win, meth.rsplit("_", 1)[1])                    # fb5 / fb5c / fb5cd
                    nh, fbw = 5, [BETA_W(j) for j in range(5)]
                S, N = X.shape[0], X.shape[-1]
                folds = [int(x) - 1 for x in a.folds.split(",")] if a.folds else list(range(S))
                folds = [t for t in folds if (f"{win:.2f}", t + 1) not in done]
                if not folds:
                    print(f"bo qua {meth} {ds} w={win} (da co)", flush=True)
                    continue
                shared = dict(X=X, refs=sincos(meta["freqs"], meta["srate"], N, nh), fbw=fbw,
                              nbrs=neighbors(ds, meta["freqs"]))
                G.clear()
                G.update(shared)
                if meth.startswith("huang"):
                    with Pool(a.workers) as pool:
                        st1 = pool.map(_step1_one, [(m, j) for m in range(S) for j in range(X.shape[3])], chunksize=1)
                    G["step1"] = [{m: r for (m, jj, r) in st1 if jj == j} for j in range(X.shape[3])]
                    del st1
                fn = run_ttcca if meth.startswith("ttcca") else run_huang
                with Pool(a.workers) as pool:
                    res = pool.map(fn, folds, chunksize=1)
                G.clear()
                viol = sum(r[3] for r in res) if meth.startswith("huang") else 0
                new = not os.path.exists(out)
                with open(out, "a", newline="") as f:
                    wr = csv.DictWriter(f, fieldnames=FIELDS)
                    if new:
                        wr.writeheader()
                    for r in sorted(res):
                        wr.writerow({"method": meth, "tag": meth.split("_")[-1] if "_" in meth else "raw", "dataset": ds,
                                     "window": f"{win:.2f}", "seed": 0, "fold": r[0] + 1, "subject": r[0] + 1,
                                     "acc": f"{r[1]:.6f}", "n": r[2]})
                print(f"KQ {meth} {ds} w={win}: {len(res)} fold, acc TB {100 * np.mean([r[1] for r in res]):.2f}%"
                      f"{f', vi pham buoc 3: {viol}' if meth.startswith('huang') else ''} ({time.time() - t0:.0f}s)", flush=True)
    print("DONE baselines_tf", flush=True)


if __name__ == "__main__":
    main()
