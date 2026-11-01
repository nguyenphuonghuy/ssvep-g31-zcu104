"""Chan doan va hieu chinh lech thoi gian/pha BETA so voi Benchmark cho G3-cross.

1. Uoc luong tren TAP HIEU CHINH BETA S1-S10 (khong dung cho danh gia): tau (luoi 1 ms) -> dich cua so BETA o mau;
   phan du pha tung muc tieu sau dich -> dich template Benchmark d_k mau.
2. Tu kiem (exit 3): cat cua so co dich trung prepare_causal khi dich = 0; |tau'| <= 2 ms sau dich; TB vong |du| <= 0,3 rad
   va nho hon truoc khi dich template.
3. Moi (cua so, seed): huan luyen G3-cross DUNG nhu g3_cross.py; kiem tra tren BETA S11-S70 voi V0 (khong hieu chinh),
   V1 (dich cua so BETA), V2 (V1 + template dich d_k). FBCCA nhan qua voi cua so V0 va V1.
Dau ra <out_dir>: align_params.json, align_beta_w<win>_s<seed>_per_subject.csv (method=V0|V1|V2),
   fbcca_align_beta_per_subject.csv (method=fbcca_V0|fbcca_V1).
Dung: python g3_align.py --out_dir <dir> [--seeds 0,1,2,3,4] [--windows 0.2,...] [--max_epochs N] [--model wp3_g3_dc]
--model wp3_g3_dc (G3.1): prep fb5cd (tru TB [0, onset) roi loc nhan qua) cho moi cua so cat.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from g3_cross import FIELDS, VAL_SEED, label_map  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD_CROSS = os.path.join(PROJ, "results", "g3_independent_2026_10_13")
CAL, EVAL = list(range(1, 11)), list(range(11, 71))
LAT_DIFF = 0.13 - 0.14                       # khac tre cua so BETA - Benchmark (nhu check_phase_beta.py)


def rnd(x):
    """round nua len."""
    return int(math.floor(x + 0.5))


def fail(msg):
    print(f"TU KIEM KHONG DAT: {msg}", flush=True)
    sys.exit(3)


# ---------------- pha (giong check_phase_beta.py, tren cache _pre de dich duoc cua so) ----------------
def grand_oz(ds, subs, off):
    """TB theo nguoi + block, kenh Oz, cua so 1 s tu [kich thich + tre + off_k]; off: so nguyen hoac mang (K,)."""
    from ssvep.data import load_subject_pre
    acc = None
    for s in subs:
        X, meta = load_subject_pre(ds, s)                 # block, target, C, t
        fs, on = meta["srate"], meta["onset_in_array"]
        K = X.shape[1]
        o = np.broadcast_to(np.asarray(off), (K,))
        s0 = on + int(round(meta["latency_s"] * fs))
        n = int(round(1.0 * fs))
        oz = [c.upper() for c in meta["channels"]].index("OZ")
        m = np.stack([X[:, k, oz, s0 + o[k]:s0 + o[k] + n].mean(axis=0) for k in range(K)])
        acc = m if acc is None else acc + m
    return acc / len(subs), meta


def phases(acc, meta):
    fs = meta["srate"]
    t = np.arange(acc.shape[-1]) / fs
    return np.array([np.angle(np.sum((acc[k] - acc[k].mean()) * np.exp(-2j * np.pi * f * t)))
                     for k, f in enumerate(meta["freqs"])])


def residual(beta_off, bench_off):
    """Delta phi_k (BETA - Benchmark - anh huong khac tre), theo thu tu muc tieu BETA, boc ve (-pi, pi]."""
    ab, mb = grand_oz("benchmark", list(range(1, 36)), bench_off)
    at, mt = grand_oz("beta", CAL, beta_off)
    pb, pt = phases(ab, mb), phases(at, mt)
    mp = label_map(mb, mt)                                # chi so BETA -> chi so Benchmark cung tan so
    f = np.array(mt["freqs"])
    d = pt - pb[mp] - 2 * np.pi * f * LAT_DIFF
    return (d + np.pi) % (2 * np.pi) - np.pi, f, mp


def fit_tau(r, f):
    best = None
    for tau_ms in range(-150, 151):
        z = np.mean(np.exp(1j * (r - 2 * np.pi * f * tau_ms / 1000)))
        if best is None or abs(z) > best[0]:
            best = (abs(z), tau_ms, float(np.angle(z)))
    return best                                           # (R, tau_ms, phi0)


# ---------------- cat cua so loc nhan qua co dich ----------------
def causal_windows(ds, subs, win, off, prep="fb5c"):
    """Nhu ssvep.prep.prepare_causal(ds, win, prep) nhung cua so muc tieu k bat dau o s0 + off_k.
    prep fb5cd: tru TB doan truoc kich thich [0, onset) (khong dich) truoc khi loc."""
    from scipy.signal import lfilter
    from ssvep import cca
    from ssvep.data import load_subject_pre
    Xs = []
    for s in subs:
        X, meta = load_subject_pre(ds, s)
        fs, on = meta["srate"], meta["onset_in_array"]
        K = X.shape[1]
        o = np.broadcast_to(np.asarray(off), (K,)).astype(int)
        s0 = on + int(round(meta["latency_s"] * fs))
        n = int(round(win * fs))
        assert s0 + o.min() >= 0
        seg = X[..., :s0 + o.max() + n].astype(np.float64)
        if prep == "fb5cd":
            seg = seg - seg[..., :on].mean(axis=-1, keepdims=True)
        bands = np.stack([lfilter(b, a, seg, axis=-1) for b, a in cca.subband_filters(fs, 5)], axis=-3)
        Xp = np.stack([bands[:, k, ..., s0 + o[k]:s0 + o[k] + n] for k in range(K)], axis=1)   # block, K, 5, C, n
        nb = Xp.shape[0]
        Xs.append(Xp.reshape(nb * K, *Xp.shape[2:]).astype(np.float32))
    return np.stack(Xs), np.tile(np.arange(K), nb), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--windows", default="0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--max_epochs", type=int, default=0)
    ap.add_argument("--model", default="wp3_g3", help="wp3_g3 (G3, muc 14) hoac wp3_g3_dc (G3.1, muc 21a)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    # ---------- 1. uoc luong tren tap hieu chinh ----------
    r0, f, mp = residual(0, 0)
    R, tau_ms, phi0 = fit_tau(r0, f)
    fs = 250
    o = -rnd(tau_ms / 1000 * fs)
    r1, _, _ = residual(o, 0)
    R1, tau1, _ = fit_tau(r1, f)
    print(f"[hieu chinh S1-S10] tau={tau_ms} ms phi0={math.degrees(phi0):+.1f} do R={R:.3f} -> dich cua so BETA o={o} mau; "
          f"sau dich tau'={tau1} ms (R={R1:.3f})", flush=True)
    if abs(tau1) > 2:
        fail(f"|tau'| = {tau1} ms > 2")
    # d_k theo chi so Benchmark (template), tu phan du r1 theo chi so BETA
    K = len(f)
    cand = {}
    for sign in (+1, -1):
        d_bench = np.zeros(K, dtype=int)
        for kb in range(K):
            d_bench[mp[kb]] = sign * rnd(r1[kb] * fs / (2 * np.pi * f[kb]))
        r2, _, _ = residual(o, d_bench)
        cand[sign] = (float(np.mean(np.abs(r2))), d_bench, r2)
        print(f"  dau {sign:+d}: TB |du| sau dich template = {cand[sign][0]:.3f} rad (truoc {np.mean(np.abs(r1)):.3f})",
              flush=True)
    sign = min(cand, key=lambda s: cand[s][0])
    m2, d_bench, r2 = cand[sign]
    if not (m2 <= 0.3 and m2 < np.mean(np.abs(r1))):
        fail(f"TB |du| sau dich template = {m2:.3f} rad")
    print(f"  chon dau {sign:+d}; d_k (mau, theo chi so Benchmark): {d_bench.tolist()}", flush=True)
    json.dump(dict(cal_subjects=CAL, eval_subjects=EVAL, tau_ms=tau_ms, phi0_deg=math.degrees(phi0), R=R,
                   o_samples=o, tau_after_ms=tau1, sign=sign, d_bench=d_bench.tolist(),
                   resid_before=np.abs(r1).mean(), resid_after=m2),
              open(os.path.join(a.out_dir, "align_params.json"), "w"), indent=1, default=float)

    import torch
    from ssvep.models import count_params
    from ssvep.wp2_models import cca_features
    from ssvep.wp3_models import make_variant
    from train_loso import prepare
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 1)))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Model, hp = make_variant(a.model)
    PREP = Model.prep
    assert PREP in ("fb5c", "fb5cd"), PREP
    print(f"mo hinh {a.model}, prep {PREP}", flush=True)
    if a.max_epochs:
        hp["max_epochs"] = a.max_epochs
    w_n = torch.tensor([(n + 1) ** -1.25 + 0.25 for n in range(5)], device=dev)
    fb_rows = []
    for win in [float(w) for w in a.windows.split(",")]:
        Xs, ys, ms = prepare("benchmark", win, Model.prep)
        # tu kiem: cat cua so co dich = 0 trung prepare_causal
        chk, _, _ = causal_windows("benchmark", [1, 2], win, 0, PREP)
        if not np.array_equal(chk, Xs[:2]):
            fail(f"causal_windows(off=0) khac prepare_causal (w={win})")
        Xt0, yt, mt = causal_windows("beta", EVAL, win, 0, PREP)
        Xt1, _, _ = causal_windows("beta", EVAL, win, o, PREP)
        ref, _, _ = prepare("beta", win, Model.prep)
        if not np.array_equal(Xt0, ref[[s - 1 for s in EVAL]]):
            fail(f"BETA V0 khac prepare_causal (w={win})")
        yt_src = label_map(ms, mt)[yt]
        subs = list(range(1, Xs.shape[0] + 1))
        val = sorted(np.random.RandomState(VAL_SEED).choice(subs, 4, replace=False).tolist())
        train = [s for s in subs if s not in val]
        Xs2, _, _ = causal_windows("benchmark", train, win, d_bench, PREP)       # template V2 (31 nguoi train)
        nbk = Xs2.shape[1] // K
        tm2 = Xs2.reshape(len(train), nbk, K, *Xs2.shape[2:]).mean(axis=(0, 1)).transpose(1, 0, 2, 3)
        print(f"w={win}: Benchmark X={Xs.shape}; BETA danh gia X={Xt0.shape}; val={val}", flush=True)
        # FBCCA nhan qua V0 / V1
        for tag, Xt in (("fbcca_V0", Xt0), ("fbcca_V1", Xt1)):
            S, N = Xt.shape[:2]
            ft = cca_features(torch.from_numpy(Xt).to(dev).reshape(S * N, *Xt.shape[2:]), mt["freqs"], mt["srate"])
            pred = (ft["band_nh5"] * w_n).sum(-1).argmax(-1).reshape(S, N).cpu().numpy()
            for i, s in enumerate(EVAL):
                fb_rows.append({"method": tag, "tag": PREP, "dataset": "beta", "window": f"{win:.2f}", "seed": 0,
                                "fold": 0, "subject": s, "acc": f"{float((pred[i] == yt).mean()):.6f}", "n": N})
            print(f"  {tag}: {100 * np.mean([float(r['acc']) for r in fb_rows[-len(EVAL):]]):.2f}%", flush=True)
        for seed in [int(s) for s in a.seeds.split(",")]:
            out = os.path.join(a.out_dir, f"align_beta_w{win:.2f}_s{seed}_per_subject.csv")
            if os.path.exists(out):
                print(f"bo qua {out} (da co)")
                continue

            def stack(X, ss, y):
                xx = np.concatenate([X[s - 1] for s in ss])
                return torch.from_numpy(xx).to(dev), torch.from_numpy(np.tile(y, len(ss))).long().to(dev)

            # ----- huan luyen: chep NGUYEN quy trinh g3_cross.py -----
            xtr, ytr = stack(Xs, train, ys)
            xva, yva = stack(Xs, val, ys)
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = Model(Xs.shape[3], Xs.shape[4], len(ms["freqs"]), ms["srate"], freqs=ms["freqs"]).to(dev)
            xt_ = Xs[[s - 1 for s in train]]
            tm = xt_.reshape(len(train), xt_.shape[1] // K, K, *xt_.shape[2:]).mean(axis=(0, 1))
            model.set_templates(torch.from_numpy(np.ascontiguousarray(tm.transpose(1, 0, 2, 3))).to(dev))
            opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
            lossf = torch.nn.CrossEntropyLoss()
            g = torch.Generator(device="cpu").manual_seed(seed)

            def evaluate(xx, yy):
                model.eval()
                with torch.no_grad():
                    pred = torch.cat([model(xx[i:i + 2048]).argmax(1) for i in range(0, len(xx), 2048)])
                return (pred == yy).float().mean().item()

            best, best_ep, best_state, t0 = -1.0, 0, None, time.time()
            for ep in range(1, hp["max_epochs"] + 1):
                model.train()
                perm = torch.randperm(len(xtr), generator=g).to(dev)
                for i in range(0, len(perm), hp["batch"]):
                    idx = perm[i:i + hp["batch"]]
                    opt.zero_grad()
                    lossf(model(xtr[idx]), ytr[idx]).backward()
                    opt.step()
                va = evaluate(xva, yva)
                if va > best:
                    best, best_ep = va, ep
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                if ep - best_ep >= hp["patience"]:
                    break
            model.load_state_dict(best_state)
            # ----- kiem tra 3 bien the tren S11-S70 -----
            yy = torch.from_numpy(yt_src).long().to(dev)
            rows = []
            for var, Xt, tmv in (("V0", Xt0, None), ("V1", Xt1, None), ("V2", Xt1, tm2)):
                if tmv is not None:
                    model.set_templates(torch.from_numpy(np.ascontiguousarray(tmv)).to(dev))
                for i, s in enumerate(EVAL):
                    acc = evaluate(torch.from_numpy(Xt[i]).to(dev), yy)
                    rows.append({"method": var, "tag": PREP, "dataset": "beta", "window": f"{win:.2f}", "seed": seed,
                                 "fold": 0, "subject": s, "acc": f"{acc:.6f}", "n": len(yt_src)})
            # doi chieu V0 voi job 17026
            old = {}
            p_old = os.path.join(OLD_CROSS, f"g3_cross_beta_w{win:.2f}_s{seed}_per_subject.csv")
            if a.model == "wp3_g3" and os.path.exists(p_old):
                with open(p_old) as fo:
                    old = {int(r["subject"]): float(r["acc"]) for r in csv.DictReader(fo)}
            v0 = {r["subject"]: float(r["acc"]) for r in rows if r["method"] == "V0"}
            dlt = np.mean([abs(v0[s] - old[s]) for s in EVAL]) if old else float("nan")
            with open(out, "w", newline="") as fo:
                wr = csv.DictWriter(fo, fieldnames=FIELDS)
                wr.writeheader()
                wr.writerows(rows)
            msg = " ".join(f"{v}={100 * np.mean([float(r['acc']) for r in rows if r['method'] == v]):.2f}%"
                           for v in ("V0", "V1", "V2"))
            print(f"KQ align w={win} seed={seed} ep={ep}(best {best_ep}) {msg} | V0 lech 17026 TB {100 * dlt:.3f} diem "
                  f"params={count_params(model)} ({time.time() - t0:.0f}s)", flush=True)
    p = os.path.join(a.out_dir, "fbcca_align_beta_per_subject.csv")
    with open(p, "w", newline="") as fo:
        wr = csv.DictWriter(fo, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(fb_rows)
    print("DONE g3_align", flush=True)


if __name__ == "__main__":
    main()
