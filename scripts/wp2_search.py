"""WP2 - tim kien truc bang Optuna tren 5-fold Benchmark.

Moi tien trinh = 1 worker cua 1 ho; nhieu worker cung ho dung chung mot JournalStorage (file) nen
chay song song duoc va chay lai thi tiep tuc (dem trial COMPLETE qua MaxTrialsCallback).
- Muc tieu: (max) accuracy TB tren search_val qua cac fold; (min) MACs moi quyet dinh.
- Trial vuot ngan sach (tham so > 50 000 hoac MACs > 5e6) hoac trung cau hinh da xong: PRUNED,
  khong huan luyen, khong tinh vao so trial.
- Moi trial ghi 1 dong vao <out_dir>/trials_<ho>_w<worker>.csv (ke ca infeasible/duplicate).
--selftest: tai tao FBCCA 0,4 s tu dac trung H-A (G0: 12,54%), kiem shape/gradient/MACs ca 4 ho,
  kiem file chia fold; exit 3 neu khong dat.

Dung: python wp2_search.py --family HB --worker 0 --out_dir <dir> [--n_trials 40] [--folds 0,1,2,3,4]
          [--max_epochs 200] [--patience 30]
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(PROJ, "configs", "splits_benchmark_search.json")
SPLITS_SHA = "b7d7d902f8727054a87dbaf0f5fb45a09e91599a451919bebde4f0b6ef481543"
CACHE = os.path.join(PROJ, "data", "cache", "wp2_fb5_benchmark_w{w:.2f}.npy")
MAX_PARAMS, MAX_MACS = 50_000, 5_000_000
SAMPLER_SEED = 20261008
G0_FBCCA_04 = 12.536          # results/g0_trainfree_2026_10_02, Benchmark 0,4 s (diem %)
FIELDS = ["family", "trial", "worker", "status", "params_json", "n_params", "macs", "acc_mean",
          "acc_folds", "epochs", "best_epochs", "sec", "time"]


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def load_splits():
    sp = json.load(open(SPLITS))
    sha = hashlib.sha256(json.dumps(sp["folds"], sort_keys=True).encode()).hexdigest()
    if sha != SPLITS_SHA:
        sys.exit(f"LOI: file chia fold bi thay doi (sha {sha})")
    return sp["folds"]


def load_data(win):
    """X (35, 240, 5, 9, T) float32 sau bank loc fb5; cache .npy (ghi tmp roi doi ten)."""
    from ssvep.data import load_cfg, load_subject
    from train_loso import prepare
    path = CACHE.format(w=win)
    _, meta = load_subject("benchmark", 1)
    if os.path.exists(path):
        X = np.load(path)
    else:
        X, _, meta = prepare("benchmark", win, "fb5")
        tmp = f"{path}.tmp{os.getpid()}"
        np.save(tmp, X)
        os.replace(tmp + ".npy" if not tmp.endswith(".npy") else tmp, path)
    nk = len(meta["freqs"])
    y = np.tile(np.arange(nk), X.shape[1] // nk)
    assert X.shape[0] == load_cfg("benchmark")["n_subjects"] and X.shape[2] == 5, X.shape
    return X, y, meta


def templates(torch, X, train):
    """Template nhom (5, K, C, T): TB qua nguoi train va block, theo muc tieu."""
    xt = X[[s - 1 for s in train]]                                   # (S, 240, 5, C, T)
    S, N = xt.shape[:2]
    K = 40
    return xt.reshape(S, N // K, K, *xt.shape[2:]).mean(dim=(0, 1)).permute(1, 0, 2, 3).contiguous()


def run_fold(torch, family, cfg, data, y, X, split, hp, seed, meta, device):
    from ssvep.wp2_models import build
    C, T, K, fs = X.shape[3], X.shape[4], len(meta["freqs"]), meta["srate"]

    def stack(ss):
        idx = torch.tensor([s - 1 for s in ss], device=device)
        xx = data.index_select(0, idx)
        return xx.reshape(-1, *xx.shape[2:]), y.repeat(len(ss))

    xtr, ytr = stack(split["train"])
    xes, yes = stack(split["early_stop"])
    xva, yva = stack(split["search_val"])
    torch.manual_seed(seed)
    model, _, _ = build(family, cfg, C, T, K, fs, meta["freqs"])
    model = model.to(device)
    if family == "HC":
        model.set_templates(templates(torch, X, split["train"]))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    lossf = torch.nn.CrossEntropyLoss()
    g = torch.Generator(device="cpu").manual_seed(seed)

    def evaluate(xx, yy):
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(xx[i:i + 1024]).argmax(1) for i in range(0, len(xx), 1024)])
        return (pred == yy).float().mean().item()

    best, best_ep, best_state, ep = -1.0, 0, None, 0
    t0 = time.time()
    for ep in range(1, hp["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(len(xtr), generator=g).to(device)
        for i in range(0, len(perm), hp["batch"]):
            idx = perm[i:i + hp["batch"]]
            opt.zero_grad()
            lossf(model(xtr[idx]), ytr[idx]).backward()
            opt.step()
        a = evaluate(xes, yes)
        if a > best:
            best, best_ep = a, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep - best_ep >= hp["patience"]:
            break
    model.load_state_dict(best_state)
    return evaluate(xva, yva), ep, best_ep, (time.time() - t0) / ep


def selftest(torch, X, y, meta, device):
    from ssvep.cca import subband_filters  # noqa: F401  (kiem import)
    from ssvep.wp2_models import FAMILIES, build, cca_features, ha_input
    ok = True
    folds = load_splits()
    allv = sorted(s for f in folds for s in f["search_val"])
    ok &= allv == list(range(1, 36))
    log(f"[tu kiem] chia fold: sha khop, search_val phu du 35 nguoi: {allv == list(range(1, 36))}")
    # 1. FBCCA tu dac trung H-A
    Xt = torch.from_numpy(X).to(device)
    N = X.shape[0] * X.shape[1]
    feats = cca_features(Xt.reshape(N, *X.shape[2:]), meta["freqs"], meta["srate"])
    w = torch.tensor([(n + 1) ** -1.25 + 0.25 for n in range(5)], device=device)
    pred = (feats["band_nh5"] * w).sum(-1).argmax(-1).reshape(X.shape[0], X.shape[1]).cpu().numpy()
    acc = 100 * float(np.mean([(pred[s] == y).mean() for s in range(X.shape[0])]))
    good = abs(acc - G0_FBCCA_04) < 0.15
    log(f"[tu kiem] FBCCA tu dac trung H-A @0,4 s = {acc:.3f}% (G0 {G0_FBCCA_04}) {'OK' if good else 'LOI'}")
    ok &= good
    log(f"[tu kiem] shape dac trung: " + ", ".join(f"{k} {tuple(v.shape)}" for k, v in feats.items()))
    fs_ = {k: v.reshape(X.shape[0], X.shape[1], *v.shape[1:]) for k, v in feats.items()}
    for feat in ("band", "harm"):
        for nh in (3, 5):
            sh = tuple(ha_input(fs_, {"feat": feat, "nh": nh}).shape)
            want = (X.shape[0], X.shape[1], 40, 5 if feat == "band" else 5 * nh)
            log(f"[tu kiem] ha_input theo nguoi {feat} nh={nh}: {sh} {'OK' if sh == want else 'LOI'}")
            ok &= sh == want
    # 2. moi ho: forward, gradient, MACs
    import optuna
    from ssvep.wp2_models import suggest
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    C, T, K = X.shape[3], X.shape[4], 40
    xb = Xt[0, :8]
    for fam in FAMILIES:
        st = optuna.create_study(directions=["maximize", "minimize"],
                                 sampler=optuna.samplers.RandomSampler(seed=1))
        for _ in range(4):
            tr = st.ask()
            cfg = suggest(tr, fam)
            m, npar, macs = build(fam, cfg, C, T, K, meta["srate"], meta["freqs"])
            m = m.to(device).train()
            if fam == "HC":
                m.set_templates(templates(torch, Xt, list(range(1, 25))))
            inp = ha_input({k: v[:8] for k, v in feats.items()}, cfg) if fam == "HA" else xb
            out = m(inp)
            out.sum().backward()
            nog = [n for n, p in m.named_parameters() if p.grad is None]
            gsum = sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)
            good = tuple(out.shape) == (8, K) and not nog and gsum > 0 and macs > 0 and npar > 0
            log(f"[tu kiem] {fam} {cfg} params={npar} macs={macs} out={tuple(out.shape)} "
                f"grad0={nog} {'OK' if good else 'LOI'}")
            ok &= good
            st.tell(tr, [0.0, float(macs)])
    log("[tu kiem]", "DAT" if ok else "KHONG DAT")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["HA", "HB", "HC", "HD"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_trials", type=int, default=40)
    ap.add_argument("--max_attempts", type=int, default=400)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--window", type=float, default=0.4)
    ap.add_argument("--max_epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 1)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    X, y, meta = load_data(a.window)
    log(f"torch {torch.__version__} device {device}; du lieu X={X.shape} ({time.time() - t0:.0f}s)")
    if a.selftest:
        sys.exit(0 if selftest(torch, X, y, meta, device) else 3)

    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
    from ssvep.wp2_models import GRID_HA, build, cca_features, ha_input, suggest
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    folds = load_splits()
    fold_ids = [int(f) for f in a.folds.split(",")]
    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).long().to(device)
    feats = None
    if a.family == "HA":
        N = X.shape[0] * X.shape[1]
        feats = cca_features(Xt.reshape(N, *X.shape[2:]), meta["freqs"], meta["srate"])
        feats = {k: v.reshape(X.shape[0], X.shape[1], *v.shape[1:]) for k, v in feats.items()}
    C, T, K, fs = X.shape[3], X.shape[4], len(meta["freqs"]), meta["srate"]
    hp = {"max_epochs": a.max_epochs, "patience": a.patience, "batch": a.batch}

    os.makedirs(a.out_dir, exist_ok=True)
    jpath = os.path.join(a.out_dir, f"study_{a.family}.journal")
    storage = JournalStorage(JournalFileBackend(jpath, lock_obj=JournalFileOpenLock(jpath + ".lock")))
    if a.family == "HA":
        sampler = optuna.samplers.GridSampler(GRID_HA, seed=SAMPLER_SEED)
    else:
        sampler = optuna.samplers.NSGAIISampler(population_size=10, seed=SAMPLER_SEED + a.worker)
    study = optuna.create_study(study_name=f"wp2_{a.family}", storage=storage, sampler=sampler,
                                directions=["maximize", "minimize"], load_if_exists=True)
    csv_path = os.path.join(a.out_dir, f"trials_{a.family}_w{a.worker}.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def record(trial, status, cfg, npar, macs, accs=(), eps=(), beps=(), sec=0.0):
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow({
                "family": a.family, "trial": trial.number, "worker": a.worker, "status": status,
                "params_json": json.dumps(cfg, sort_keys=True), "n_params": npar, "macs": macs,
                "acc_mean": f"{100 * np.mean(accs):.4f}" if accs else "",
                "acc_folds": ";".join(f"{100 * x:.4f}" for x in accs),
                "epochs": ";".join(map(str, eps)), "best_epochs": ";".join(map(str, beps)),
                "sec": f"{sec:.1f}", "time": time.strftime("%F %T")})

    def objective(trial):
        cfg = suggest(trial, a.family)
        _, npar, macs = build(a.family, cfg, C, T, K, fs, meta["freqs"])
        trial.set_user_attr("n_params", npar)
        trial.set_user_attr("macs", macs)
        trial.set_user_attr("cfg", cfg)
        if npar > MAX_PARAMS or macs > MAX_MACS:
            trial.set_user_attr("status", "infeasible")
            record(trial, "infeasible", cfg, npar, macs)
            log(f"{a.family} trial {trial.number} INFEASIBLE params={npar} macs={macs} {cfg}")
            raise optuna.TrialPruned()
        done = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        if any(t.params == trial.params for t in done):
            trial.set_user_attr("status", "duplicate")
            record(trial, "duplicate", cfg, npar, macs)
            raise optuna.TrialPruned()
        t1 = time.time()
        accs, eps, beps = [], [], []
        data = Xt if feats is None else ha_input(feats, cfg)
        for fi in fold_ids:
            acc, ep, bep, spe = run_fold(torch, a.family, cfg, data, yt, Xt, folds[fi], hp, a.seed, meta,
                                         device)
            log(f"  {a.family} trial {trial.number} fold {fi}: acc={100 * acc:.2f} ep={ep} best={bep} "
                f"s/ep={spe:.2f}")
            accs.append(acc)
            eps.append(ep)
            beps.append(bep)
        trial.set_user_attr("acc_folds", accs)
        trial.set_user_attr("status", "complete")
        record(trial, "complete", cfg, npar, macs, accs, eps, beps, time.time() - t1)
        log(f"KQ {a.family} trial {trial.number} acc={100 * np.mean(accs):.2f} "
            f"folds={[round(100 * x, 1) for x in accs]} params={npar} macs={macs} ep={eps} "
            f"({time.time() - t1:.0f}s) {cfg}")
        return 100 * float(np.mean(accs)), float(macs)

    n_done = len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))
    log(f"study wp2_{a.family}: da co {n_done} trial COMPLETE, dich {a.n_trials}; worker {a.worker}")
    study.optimize(objective, n_trials=a.max_attempts,
                   callbacks=[optuna.study.MaxTrialsCallback(
                       a.n_trials, states=(optuna.trial.TrialState.COMPLETE,))])
    n_done = len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))
    log(f"DONE {a.family} worker {a.worker}: {n_done} trial COMPLETE")


if __name__ == "__main__":
    main()
