"""Doi chung Ensemble-DNN (Guney & Ozkan 2023, J. Neural Eng. 20:016013).

Theo ma MATLAB chinh thuc (osmanberke/Ensemble-of-DNNs: training.m, testing.m, PreProcess.m):
  giai doan 1: DNN toan cuc (ssvep.models.GuneyDNN) tren moi nguoi nguon, so epoch co dinh, khong validation;
  giai doan 2: tinh chinh tung nguoi nguon tu DNN toan cuc (huan luyen song song bang torch.func.vmap);
  nhan dang: do tuong dong rho_n (max tren 120 cot lop 2 cua corr^2 voi template + CCA^2 voi sin/cos 5 hai), chon dong k.
Moi to hop (dataset, cua so, seed) -> mot file <out_dir>/ensdnn_{ds}_w{win}_s{seed}_per_subject.csv (ghi tang dan theo fold;
chay lai: bo qua fold da co) + _timing.csv.
Dung: python ensemble_dnn.py --out_dir <dir> --combos benchmark:0.4:0,beta:1.0:3 [--folds 1,2] [--ep1 N --ep2 N]
      python ensemble_dnn.py --selftest
      python ensemble_dnn.py --selftest_causal          (: tu kiem tien xu ly guney3cd tren du lieu that)
--prep guney3cd: loc NHAN QUA nhu G3.1 (tru TB doan truoc kich thich, sosfilt tu dau doan truoc kich thich);
  phuong phap ensemble_dnn_c, tep ensdnnc_{ds}_w{win}_s{seed}_per_subject.csv.
--save_dir <dir>: moi (ds, cua so, seed, fold) chay xong ghi <dir>/ens_{ds}_w{win}_s{seed}_f{fold}.npz gom
  trong so 34 mang (float32), template (TB moi luot thu, 3 dai), co so CCA, luot thu tho + da loc, va chuan PyTorch (logit tung
  mang, du doan tung mang, do tuong dong, k*, du doan cuoi) + hang bo loc (sos, zi, padlen) de cai lai ngoai PyTorch.
"""
import argparse
import copy
import csv
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

FIELDS = ["method", "tag", "dataset", "window", "seed", "fold", "subject", "acc", "n"]
TFIELDS = ["dataset", "window", "seed", "fold", "ep1", "ep2", "sec_stage1", "sec_stage2", "sec_test", "mean_k", "acc"]
# training.m: so epoch giai doan 1, dropout giai doan 2; Nakanishi (khong co trong bai): nhu Benchmark
HP = {"benchmark": (1000, 0.6), "beta": (800, 0.7), "nakanishi": (1000, 0.6)}
LR, B1, EP2, L2, NH = 1e-4, 100, 1000, 0.001, 5


def fail(msg):
    print(f"TU KIEM KHONG DAT: {msg}", flush=True)
    sys.exit(3)


# ---------------- nhan dang (testing.m) ----------------
def ref_basis(freqs, fs, T, torch, dev):
    """Co so truc chuan Q_k (T x 2NH) cua tin hieu tham chieu Y_k da tru TB (canoncorr cua MATLAB tru TB)."""
    t = np.arange(T) / fs
    Q = []
    for f in freqs:
        Y = np.stack([fn(2 * np.pi * h * f * t) for h in range(1, NH + 1) for fn in (np.sin, np.cos)], axis=1)
        Q.append(np.linalg.qr(Y - Y.mean(0))[0])
    return torch.tensor(np.stack(Q), dtype=torch.float64, device=dev)          # K, T, 2NH


def similarity(torch, xs, tmpl_pred, Wc, Qp):
    """xs (M,C,T): luot thu da ket hop dai con; tmpl_pred (M,C,T): template lop du doan; Wc (J,C); Qp (M,T,R).
    rho = max_j [corr(w_j' tmpl, w_j' x)^2 + CCA(w_j' x, Y)^2]  -> (M,)."""
    a = torch.einsum("jc,mct->mjt", Wc, xs)
    b = torch.einsum("jc,mct->mjt", Wc, tmpl_pred)
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    saa, sbb = (a * a).sum(-1), (b * b).sum(-1)
    corr2 = (a * b).sum(-1) ** 2 / (saa * sbb)
    cca2 = (torch.einsum("mjt,mtr->mjr", a, Qp) ** 2).sum(-1) / saa
    s = corr2 + cca2
    s = torch.where(torch.isnan(s), torch.full_like(s, -math.inf), s)          # MATLAB max bo qua NaN
    return s.max(-1).values


def dynamic_select(sims, preds, K):
    """sims, preds (N,) -> (du doan, k*). Vector hoa dung testing.m (sort giam dan on dinh; chi so dau tien khi bang)."""
    order = np.argsort(-sims, kind="stable")
    oh = np.zeros((len(sims), K))
    oh[np.arange(len(sims)), preds[order]] = sims[order]
    V = np.cumsum(oh, axis=0)                                                  # (k, K): phieu voi k nguoi dau
    top = -np.sort(-V, axis=1)
    conf = top[:, 0] - top[:, 1]
    k = int(np.argmax(conf))
    return int(np.argmax(V[k])), k + 1


def dynamic_select_loop(sims, preds, K):
    """Chep nguyen vong lap testing.m (doi chieu cho tu kiem)."""
    N = len(sims)
    idx = sorted(range(N), key=lambda i: -sims[i])                             # Python sort on dinh
    srt = [sims[i] for i in idx]
    votes = np.zeros((K, N))
    for k in range(1, N + 1):
        for k2 in range(k):
            votes[preds[idx[k2]], k - 1] += srt[k2]
    conf = np.zeros(N)
    for k in range(N):
        tv = sorted(votes[:, k], reverse=True)
        conf[k] = tv[0] - tv[1]
    best = 0
    for k in range(N):
        if conf[k] > conf[best]:
            best = k
    col = votes[:, best]
    pred = 0
    for i in range(K):
        if col[i] > col[pred]:
            pred = i
    return pred, best + 1


# ---------------- huan luyen ----------------
def l2_names(model):
    return [f"{m}.weight" for m in ("sub", "ch", "down", "conv", "fc")]


def train_global(torch, Model, X, y, meta, ep1, seed, dev):
    """X (S, n, 3, C, T) nguoi nguon. Adam lr 1e-4, batch 100 (bo lo cuoi thieu), L2 (lambda/2) sum w^2."""
    torch.manual_seed(seed)
    model = Model(X.shape[3], X.shape[4], len(meta["freqs"]), meta["srate"]).to(dev)
    xt = torch.from_numpy(X.reshape(-1, *X.shape[2:])).to(dev)
    yt = torch.from_numpy(np.tile(y, X.shape[0])).long().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    g = torch.Generator(device="cpu").manual_seed(seed)
    nb = len(xt) // B1
    model.train()
    for _ in range(ep1):
        perm = torch.randperm(len(xt), generator=g).to(dev)
        for i in range(nb):
            idx = perm[i * B1:(i + 1) * B1]
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(xt[idx]), yt[idx]) + model.l2_penalty()
            loss.backward()
            opt.step()
    return model


def finetune_stack(torch, gmodel, X, y, p2, ep2, batch, seed, dev, dropout=True):
    """Tinh chinh N ban sao cua gmodel, ban n tren du lieu X[n] (N, n, 3, C, T) — song song bang vmap.
    Tong loss cua N mo hinh -> gradient tung mo hinh doc lap; Adam theo phan tu -> tuong duong N bo toi uu rieng."""
    from torch.func import functional_call, stack_module_state, vmap
    N = X.shape[0]
    base = copy.deepcopy(gmodel)
    base.d1.p = base.d2.p = p2 if dropout else 0.0
    base.d3.p = 0.95 if dropout else 0.0
    params, buffers = stack_module_state([copy.deepcopy(base) for _ in range(N)])
    params = {k: v.detach().clone().requires_grad_(True) for k, v in params.items()}
    meta_m = copy.deepcopy(base).to("meta")
    names = l2_names(base)

    def loss_one(p, b, xb, yb):
        out = functional_call(meta_m, (p, b), (xb,))
        return torch.nn.functional.cross_entropy(out, yb) + 0.5 * L2 * sum((p[k] ** 2).sum() for k in names)

    vloss = vmap(loss_one, randomness="different")
    opt = torch.optim.Adam(params.values(), lr=LR)
    xt = torch.from_numpy(X).to(dev)
    yt = torch.from_numpy(y).long().to(dev)
    n = X.shape[1]
    nb = n // batch
    g = torch.Generator(device="cpu").manual_seed(seed + 7)
    ar = torch.arange(N, device=dev)[:, None]
    meta_m.train()
    for _ in range(ep2):
        perm = torch.stack([torch.randperm(n, generator=g) for _ in range(N)]).to(dev)
        for i in range(nb):
            idx = perm[:, i * batch:(i + 1) * batch]
            opt.zero_grad(set_to_none=True)
            vloss(params, buffers, xt[ar, idx], yt[idx]).sum().backward()
            opt.step()
    return {k: v.detach() for k, v in params.items()}, buffers, meta_m


def predict_stack(torch, meta_m, params, buffers, x):
    from torch.func import functional_call, vmap
    meta_m.eval()
    with torch.no_grad():
        f = vmap(lambda p, b: functional_call(meta_m, (p, b), (x,)), in_dims=(0, 0))
        return f(params, buffers)                                                  # N, M, K (logit)


def run_fold(torch, Model, X, y, meta, ds, fold, seed, ep1, ep2, dev, save=None):
    S = X.shape[0]
    src = [s for s in range(1, S + 1) if s != fold]
    K = len(meta["freqs"])
    nb = X.shape[1] // K
    p2 = HP[ds][1]
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    gm = train_global(torch, Model, X[[s - 1 for s in src]], y, meta, ep1, seed, dev)
    t1 = time.time()
    Xs = X[[s - 1 for s in src]]
    params, buffers, meta_m = finetune_stack(torch, gm, Xs, y, p2, ep2, K * (nb - 1), seed, dev)
    t2 = time.time()
    # ---- nhan dang ----
    xt = torch.from_numpy(X[fold - 1]).to(dev)                                      # M, 3, C, T
    logits = predict_stack(torch, meta_m, params, buffers, xt)
    preds = logits.argmax(-1).cpu().numpy()                                          # N, M
    tm = Xs.reshape(len(src), nb, K, *Xs.shape[2:]).mean(axis=1)                     # N, K, 3, C, T (TB moi luot thu)
    Q = ref_basis(meta["freqs"], meta["srate"], X.shape[4], torch, dev)
    ws = params["sub.weight"].reshape(len(src), -1).double()                          # N, 3
    Wc = params["ch.weight"][:, :, 0, :, 0].double()                                  # N, 120, C
    xd = xt.double()
    tmd = torch.from_numpy(tm).to(dev).double()
    sims = np.zeros((len(src), xt.shape[0]))
    for n in range(len(src)):
        xs = torch.einsum("s,msct->mct", ws[n], xd)
        tk = torch.einsum("s,ksct->kct", ws[n], tmd[n])
        pn = torch.from_numpy(preds[n]).to(dev)
        sims[n] = similarity(torch, xs, tk[pn], Wc[n], Q[pn]).cpu().numpy()
    out, ks = [], []
    for m in range(xt.shape[0]):
        p, k = dynamic_select(sims[:, m], preds[:, m], K)
        out.append(p)
        ks.append(k)
    t3 = time.time()
    acc = float((np.array(out) == y).mean())
    if save:
        np.savez(save, **{f"w_{k}": v.cpu().numpy().astype(np.float32) for k, v in params.items()},
                 tmpl=tm.astype(np.float32), Q=Q.cpu().numpy(), x_filt=X[fold - 1], y=y, freqs=np.asarray(meta["freqs"]),
                 fs=meta["srate"], logits=logits.cpu().numpy().astype(np.float32), preds=preds, sims=sims,
                 kstar=np.array(ks), pred_final=np.array(out), acc=acc, seed=seed, fold=fold)
    return acc, len(y), dict(sec_stage1=t1 - t0, sec_stage2=t2 - t1, sec_test=t3 - t2, mean_k=float(np.mean(ks)))


def export_filters(ds, win, fold, X, path):
    """Hang bo loc guney3 (sos, sosfilt_zi, padlen) + luot thu tho cua nguoi kiem tra; tu kiem: loc lai tu du lieu tho
    bang hang da xuat trung X[fold-1] (du lieu mo hinh da dung)."""
    from scipy.signal import cheby1, sosfilt_zi, sosfiltfilt
    from ssvep.data import load_subject, window
    from ssvep.prep import guney3
    X0, meta = load_subject(ds, fold)
    fs = meta["srate"]
    raw = window(X0, meta, win).astype(np.float64)
    nb, nk, C, T = raw.shape
    raw = raw.reshape(nb * nk, C, T)
    sos = np.stack([cheby1(1, 1, [8 * i / (fs / 2), 90 / (fs / 2)], "bandpass", output="sos") for i in range(1, 4)])
    pad = min(3 * (2 * sos.shape[1]), T - 1)
    mine = np.stack([sosfiltfilt(sos[b], raw, axis=-1, padlen=pad) for b in range(3)], axis=1).astype(np.float32)
    if not (np.array_equal(mine, X[fold - 1]) and np.array_equal(mine, guney3(raw, fs).astype(np.float32))):
        fail(f"loc lai tu du lieu tho khac du lieu mo hinh ({ds} w={win} fold={fold}); max |d| = "
             f"{np.abs(mine - X[fold - 1]).max()}")
    np.savez(path, raw=raw, sos=sos, zi=np.stack([sosfilt_zi(sos[b]) for b in range(3)]), padlen=pad, fs=fs)
    print(f"  xuat bo loc {path}: raw {raw.shape}, sos {sos.shape}, padlen {pad}; loc lai == du lieu mo hinh", flush=True)


def selftest_causal():
    """, tu kiem (ii)-(iv) cua tien xu ly guney3cd tren du lieu that (nguoi 1; Benchmark, BETA, Nakanishi)."""
    from scipy.signal import cheby1, lfilter, sosfilt
    from ssvep.data import load_subject_pre
    from ssvep.prep import guney3_sos, prepare_causal

    def one(X, meta, win, dc):                       # cai lai doc lap cho 1 nguoi (nhu prepare_causal)
        fs, on = meta["srate"], meta["onset_in_array"]
        s0, n = on + int(round(meta["latency_s"] * fs)), int(round(win * fs))
        seg = X[..., :s0 + n].astype(np.float64)
        if dc:
            seg = seg - seg[..., :on].mean(axis=-1, keepdims=True)
        return np.stack([sosfilt(sos, seg, axis=-1)[..., s0:s0 + n] for sos in guney3_sos(fs)], axis=-3), s0 + n
    for ds in ("benchmark", "beta", "nakanishi"):
        X, meta = load_subject_pre(ds, 1)
        for win in (0.4, 1.0):
            Y, end = one(X, meta, win, True)
            if not np.all(np.isfinite(Y)):
                fail(f"(iv) {ds} w={win}: khong huu han")
            X2 = X.astype(np.float64).copy()
            if end < X2.shape[-1]:
                X2[..., end:] = 1e6 * np.random.default_rng(0).standard_normal(X2[..., end:].shape)
                if not np.array_equal(one(X2, meta, win, True)[0], Y):
                    fail(f"(ii) {ds} w={win}: doi mau sau cua so lam doi dau ra")
            fs, on = meta["srate"], meta["onset_in_array"]
            seg = X[..., :end].astype(np.float64)
            for i, sos in enumerate(guney3_sos(fs)):
                b, aa = cheby1(1, 1, [8 * (i + 1) / (fs / 2), 90 / (fs / 2)], "bandpass")
                r1, r2 = sosfilt(sos, seg, axis=-1), lfilter(b, aa, seg, axis=-1)
                if np.abs(r1 - r2).max() > 1e-9 * max(np.abs(r2).max(), 1.0):
                    fail(f"(iii) {ds} w={win} dai {i + 1}: sosfilt khac lfilter")
            print(f"[tu kiem causal] {ds} w={win}: (ii) nhan qua, (iii) sosfilt == lfilter, (iv) huu han: DAT", flush=True)
    # doi chieu cai lai doc lap voi prepare_causal (Nakanishi, it nguoi -> nhanh)
    Xp, _, meta = prepare_causal("nakanishi", 0.4, "guney3cd")
    X, meta1 = load_subject_pre("nakanishi", 1)
    Y, _ = one(X, meta1, 0.4, True)
    if not np.array_equal(Xp[0], Y.reshape(-1, *Y.shape[2:]).astype(np.float32)):
        fail("prepare_causal(guney3cd) khac cai lai doc lap (Nakanishi S1)")
    print("[tu kiem causal] prepare_causal(guney3cd) == cai lai doc lap (Nakanishi S1): DAT", flush=True)


# ---------------- tu kiem ----------------
def selftest(torch, dev):
    global LR
    from ssvep.models import GuneyDNN
    rng = np.random.default_rng(0)
    # (i) CCA 1 chieu - nhieu chieu (QR) vs hoi quy (SVD/lstsq)
    T, fs = 100, 250
    Q = ref_basis([9.0, 11.4], fs, T, torch, dev)
    for _ in range(5):
        a = rng.standard_normal(T) + np.sin(2 * np.pi * 9 * np.arange(T) / fs)
        t = np.arange(T) / fs
        Y = np.stack([fn(2 * np.pi * h * 9.0 * t) for h in range(1, NH + 1) for fn in (np.sin, np.cos)], 1)
        ac, Yc = a - a.mean(), Y - Y.mean(0)
        beta = np.linalg.lstsq(Yc, ac, rcond=None)[0]
        r2 = 1 - ((ac - Yc @ beta) ** 2).sum() / (ac ** 2).sum()
        q = Q[0].cpu().numpy()
        r2q = ((q.T @ ac) ** 2).sum() / (ac ** 2).sum()
        if abs(r2 - r2q) > 1e-6:
            fail(f"(i) CCA QR {r2q} != lstsq {r2}")
    print("[tu kiem i] CCA QR == hoi quy: DAT", flush=True)
    # (ii) mot buoc toi uu: xep chong (vmap, tat dropout) == tung mo hinh
    C, T, K, N = 3, 40, 4, 3
    torch.manual_seed(0)
    gm = GuneyDNN(C, T, K, fs).to(dev)
    X = rng.standard_normal((N, 8, 3, C, T)).astype(np.float32)
    y = np.tile(np.arange(K), 2)
    params, _, _ = finetune_stack(torch, gm, X, y, 0.6, 1, 8, 0, dev, dropout=False)
    for n in range(N):
        m = copy.deepcopy(gm)
        m.d1.p = m.d2.p = m.d3.p = 0.0
        opt = torch.optim.Adam(m.parameters(), lr=LR)
        opt.zero_grad()
        xb = torch.from_numpy(X[n]).to(dev)
        (torch.nn.functional.cross_entropy(m(xb), torch.from_numpy(y).long().to(dev)) + m.l2_penalty()).backward()
        opt.step()
        for k, v in m.state_dict().items():
            if k in params and (v - params[k][n]).abs().max().item() > 1e-5:
                fail(f"(ii) tham so {k} mo hinh {n} lech")
    print("[tu kiem ii] vmap == huan luyen rieng (1 buoc): DAT", flush=True)
    # (iii) chon dong vector hoa == vong lap testing.m
    for trial in range(300):
        Nn, Kk = int(rng.integers(2, 12)), int(rng.integers(2, 6))
        sims = np.round(rng.random(Nn), 1 if trial % 2 else 6)                    # co truong hop bang nhau
        preds = rng.integers(0, Kk, Nn)
        if dynamic_select(sims, preds, Kk) != dynamic_select_loop(sims, preds, Kk):
            fail(f"(iii) chon dong lech: sims={sims} preds={preds}")
    print("[tu kiem iii] chon dong == vong lap testing.m (300 truong hop): DAT", flush=True)
    # (iv) du lieu tong hop
    freqs = [8.0, 10.0, 12.0, 14.0]
    C, T, S, nbk = 4, 100, 6, 10
    t = np.arange(T) / fs
    X = np.zeros((S, nbk * len(freqs), 3, C, T), dtype=np.float32)
    yy = np.tile(np.arange(len(freqs)), nbk)
    for s in range(S):
        for i, k in enumerate(yy):
            sig = np.sin(2 * np.pi * freqs[k] * t + rng.random() * 0.3)
            X[s, i] = sig[None, None] * rng.uniform(0.5, 1.5, (1, C, 1)) + 0.5 * rng.standard_normal((3, C, T))
    meta = {"freqs": freqs, "srate": fs}
    lr0, LR = LR, 3e-3
    acc, _, info = run_fold(torch, GuneyDNN, X, yy, meta, "benchmark", 1, 0, 100, 30, dev)
    LR = lr0
    if acc <= 0.9:
        fail(f"(iv) du lieu tong hop acc {acc:.3f}")
    print(f"[tu kiem iv] du lieu tong hop: acc {acc:.3f} (k TB {info['mean_k']:.1f}): DAT", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--combos", default="", help="ds:win:seed,... vd benchmark:0.4:0,beta:1.0:3")
    ap.add_argument("--folds", default="")
    ap.add_argument("--ep1", type=int, default=0, help="0 = theo training.m")
    ap.add_argument("--ep2", type=int, default=0, help="0 = 1000")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--save_dir", default="", help=": xuat trong so + chuan PyTorch moi fold")
    ap.add_argument("--prep", default="guney3", choices=["guney3", "guney3cd"], help="guney3cd: loc nhan qua")
    ap.add_argument("--selftest_causal", action="store_true")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 1)))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} device {dev}", flush=True)
    if a.selftest:
        selftest(torch, dev)
        print("SELFTEST DONE")
        return
    if a.selftest_causal:
        selftest_causal()
        print("SELFTEST CAUSAL DONE")
        return
    from ssvep.models import GuneyDNN
    from train_loso import prepare
    os.makedirs(a.out_dir, exist_ok=True)
    cache = {}
    for c in a.combos.split(","):
        ds, win, seed = c.split(":")
        win, seed = float(win), int(seed)
        if (ds, win) not in cache:
            cache.clear()
            cache[(ds, win)] = prepare(ds, win, a.prep)
        X, y, meta = cache[(ds, win)]
        ep1, ep2 = a.ep1 or HP[ds][0], a.ep2 or EP2
        folds = [int(f) for f in a.folds.split(",")] if a.folds else list(range(1, X.shape[0] + 1))
        causal = a.prep == "guney3cd"
        stem = os.path.join(a.out_dir, f"{'ensdnnc' if causal else 'ensdnn'}_{ds}_w{win:.2f}_s{seed}")
        out, tout = stem + "_per_subject.csv", stem + "_timing.csv"
        done = set()
        if os.path.exists(out):
            with open(out) as f:
                done = {int(r["fold"]) for r in csv.DictReader(f)}
        for p, fl in ((out, FIELDS), (tout, TFIELDS)):
            if not os.path.exists(p):
                with open(p, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=fl).writeheader()
        print(f"[{ds} w={win} s={seed}] X={X.shape} K={len(meta['freqs'])} ep1={ep1} ep2={ep2} p2={HP[ds][1]} "
              f"folds={len(folds)} (da co {len(done & set(folds))})", flush=True)
        for fold in folds:
            if fold in done:
                continue
            save = None
            if a.save_dir:
                save = os.path.join(a.save_dir, f"ens_{ds}_w{win:.2f}_s{seed}_f{fold}.npz")
                export_filters(ds, win, fold, X, os.path.join(a.save_dir, f"filt_{ds}_w{win:.2f}_f{fold}.npz"))
            acc, n, info = run_fold(torch, GuneyDNN, X, y, meta, ds, fold, seed, ep1, ep2, dev, save)
            with open(out, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writerow(dict(method="ensemble_dnn_c" if causal else "ensemble_dnn",
                                                                   tag=a.prep, dataset=ds,
                                                                   window=f"{win:.2f}", seed=seed, fold=fold, subject=fold,
                                                                   acc=f"{acc:.6f}", n=n))
            with open(tout, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=TFIELDS).writerow(dict(dataset=ds, window=f"{win:.2f}", seed=seed, fold=fold,
                                                                    ep1=ep1, ep2=ep2, acc=f"{acc:.6f}",
                                                                    **{k: f"{v:.2f}" for k, v in info.items()}))
            print(f"KQ {ds} w={win} s={seed} fold={fold}: acc {100 * acc:.2f}% | gd1 {info['sec_stage1']:.0f}s "
                  f"gd2 {info['sec_stage2']:.0f}s test {info['sec_test']:.0f}s k TB {info['mean_k']:.1f}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
