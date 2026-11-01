"""Huan luyen hoc sau theo LOSO (khong hieu chuan) va ghi ket qua theo doi tuong.

Moi fold: 1 doi tuong test; trong cac doi tuong con lai, tach tat dinh --n_val doi tuong lam
validation (phu thuoc chi so fold, KHONG phu thuoc seed hay mo hinh) de chon checkpoint va dung
som. Doi tuong test khong tham gia quyet dinh nao.

Dung: python train_loso.py --dataset benchmark --model eegnet --windows 0.2,1.0 \
          --seeds 0 --out_dir results/<ten> [--folds 1,2] [--max_epochs N] [--patience N]
Moi (mo hinh, cua so, seed) ghi mot file rieng de nhieu tien trinh chay song song an toan:
  <out_dir>/<model>_<dataset>_w<window>_s<seed>_per_subject.csv
    (method,tag,dataset,window,seed,fold,subject,acc,n)
  <out_dir>/<model>_<dataset>_w<window>_s<seed>_timing.csv
--ptq: them dong tag <tag>_ptq_int8 va <tag>_ptq_int8_pot (PTQ INT8) cung
  file per_subject, tinh tren chinh mo hinh FP32 cua fold do. --ckpt_dir: luu state_dict tot nhat.
Ghi tang dan, bo qua fold da xong; OVERWRITE=1 de xoa va chay lai.
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from ssvep.data import load_cfg, load_subject, window  # noqa: E402
from ssvep.prep import PREP  # noqa: E402

PTQ_NCAL = 1024  # so luot thu hieu chinh PTQ, lay ngau nhien tu nguoi train
FIELDS = ["method", "tag", "dataset", "window", "seed", "fold", "subject", "acc", "n"]
TFIELDS = ["method", "tag", "dataset", "window", "seed", "fold", "epochs_run", "best_epoch",
           "sec_per_epoch", "sec_total", "params", "macs", "val_acc", "train_acc"]


def prepare(ds, win, prep):
    """X (n_sub, n_trial, n_band, C, T) float32, y (n_trial,)."""
    if prep in ("fb5c", "fb5cd", "guney3c", "guney3cd", "fb5cz", "fb5cs"):                       # loc nhan qua: can du lieu truoc kich thich
        from ssvep.prep import prepare_causal
        return prepare_causal(ds, win, prep)
    n_sub = load_cfg(ds)["n_subjects"]
    Xs, meta = [], None
    for s in range(1, n_sub + 1):
        X, meta = load_subject(ds, s)
        Xw = window(X, meta, win).astype(np.float64)
        nb, nk, C, T = Xw.shape
        Xp = PREP[prep](Xw, meta["srate"])                       # nb, nk, band, C, T
        Xs.append(Xp.reshape(nb * nk, *Xp.shape[2:]).astype(np.float32))
    y = np.tile(np.arange(nk), nb)
    return np.stack(Xs), y, meta


def val_subjects(fold, train_subj, n_val):
    rng = np.random.RandomState(1000 + fold)
    return sorted(rng.choice(train_subj, n_val, replace=False).tolist())


def run_fold(torch, Model, hp, X, y, meta, fold, seed, n_val, device, ptq=False, ckpt=None):
    from ssvep.models import count_macs, count_params
    subj = list(range(1, X.shape[0] + 1))
    train = [s for s in subj if s != fold]
    val = val_subjects(fold, train, n_val)
    train = [s for s in train if s not in val]

    def stack(ss):
        xx = np.concatenate([X[s - 1] for s in ss])
        yy = np.tile(y, len(ss))
        return torch.from_numpy(xx).to(device), torch.from_numpy(yy).long().to(device)

    xtr, ytr = stack(train)
    xva, yva = stack(val)
    xte, yte = stack([fold])

    torch.manual_seed(seed)
    np.random.seed(seed)
    if getattr(Model, "needs_freqs", False):            # ung vien WP2 (ssvep/wp2_models.py)
        model = Model(X.shape[3], X.shape[4], len(meta["freqs"]), meta["srate"],
                      freqs=meta["freqs"]).to(device)
    else:
        model = Model(X.shape[3], X.shape[4], len(meta["freqs"]), meta["srate"]).to(device)
    if hasattr(model, "set_templates") and getattr(model, "use_tmpl", True):   # template nhom tu nguoi TRAIN
        K = len(meta["freqs"])
        xt = X[[s - 1 for s in train]]                  # (S, n_trial, band, C, T)
        xt = xt.reshape(len(train), xt.shape[1] // K, K, *xt.shape[2:])
        if getattr(model, "tmpl_stat", "mean") == "median":       #  (hau kiem): trung vi theo tung mau
            tm = np.median(xt.reshape(-1, *xt.shape[2:]), axis=0)
        else:
            tm = xt.mean(axis=(0, 1))
        model.set_templates(torch.from_numpy(np.ascontiguousarray(tm.transpose(1, 0, 2, 3))).to(device))
    params = count_params(model)
    macs = model.macs() if hasattr(model, "macs") else count_macs(model, xte)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    lossf = torch.nn.CrossEntropyLoss()
    g = torch.Generator(device="cpu").manual_seed(seed)

    def evaluate(xx, yy, net=None):
        net = model if net is None else net
        net.eval()
        with torch.no_grad():
            pred = torch.cat([net(xx[i:i + 2048]).argmax(1)
                              for i in range(0, len(xx), 2048)])
        return (pred == yy).float().mean().item()

    best, best_ep, best_state, ep = -1.0, 0, None, 0
    t0 = time.time()
    for ep in range(1, hp["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(len(xtr), generator=g).to(device)
        for i in range(0, len(perm), hp["batch"]):
            idx = perm[i:i + hp["batch"]]
            opt.zero_grad()
            loss = lossf(model(xtr[idx]), ytr[idx]) + model.l2_penalty()
            loss.backward()
            opt.step()
            model.apply_constraints()
        va = evaluate(xva, yva)
        if va > best:
            best, best_ep = va, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep == 1 or ep % 50 == 0:
            print(f"  fold={fold} seed={seed} ep={ep} loss={loss.item():.3f} val={va:.3f} "
                  f"best={best:.3f}@{best_ep} ({time.time() - t0:.0f}s)", flush=True)
        if ep - best_ep >= hp["patience"]:
            break
    total = time.time() - t0
    model.load_state_dict(best_state)
    te, tr = evaluate(xte, yte), evaluate(xtr, ytr)
    if ckpt:
        torch.save(best_state, ckpt)
    extra = {}
    if ptq:
        from ssvep.quant import quantize_ptq
        gc = torch.Generator(device="cpu").manual_seed(10_000 + seed)
        xcal = xtr[torch.randperm(len(xtr), generator=gc)[:PTQ_NCAL].to(device)]
        for suf, pot in (("_ptq_int8", False), ("_ptq_int8_pot", True)):
            extra[suf] = evaluate(xte, yte, quantize_ptq(model, xcal, pot=pot))
    timing = {"epochs_run": ep, "best_epoch": best_ep, "sec_per_epoch": f"{total / ep:.3f}",
              "sec_total": f"{total:.1f}", "params": params, "macs": macs,
              "val_acc": f"{best:.4f}", "train_acc": f"{tr:.4f}"}
    return te, len(yte), timing, val, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--windows", default="0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--folds", default="", help="doi tuong test, vd 1,2; rong = tat ca")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--max_epochs", type=int, default=0, help="0 = theo models.TRAIN")
    ap.add_argument("--patience", type=int, default=0, help="0 = theo models.TRAIN")
    ap.add_argument("--n_val", type=int, default=4)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ptq", action="store_true", help="danh gia them PTQ INT8")
    ap.add_argument("--ckpt_dir", default="", help="luu state_dict tot nhat moi fold")
    a = ap.parse_args()

    import torch
    from ssvep.models import MODELS, TRAIN
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 2)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.model.startswith("wp2_"):                      # ung vien WP2, doc tu selection.json
        from ssvep.wp2_models import make_candidate, train_hp
        Model, hp = make_candidate(a.model), train_hp(a.model)
        print(f"ung vien {a.model}: {Model.family} trial {Model.search_trial} cfg={Model.config}", flush=True)
    elif a.model.startswith("wp3_"):                    # bien the WP3
        from ssvep.wp3_models import make_variant
        Model, hp = make_variant(a.model)
        print(f"bien the {a.model}: cfg={Model.config} prep={Model.prep}", flush=True)
    else:
        Model = MODELS[a.model]
        hp = dict(TRAIN[a.model])
    if a.max_epochs:
        hp["max_epochs"] = a.max_epochs
    if a.patience:
        hp["patience"] = a.patience
    print(f"torch {torch.__version__} device {device} model={a.model} hp={hp}", flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    if a.ckpt_dir:
        os.makedirs(a.ckpt_dir, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")]

    for win in [float(w) for w in a.windows.split(",")]:
        t = time.time()
        X, y, meta = prepare(a.dataset, win, Model.prep)
        print(f"du lieu {a.dataset} w={win} X={X.shape} prep={Model.prep} "
              f"kenh={meta['channels']} ({time.time() - t:.0f}s)", flush=True)
        folds = [int(f) for f in a.folds.split(",")] if a.folds else \
            list(range(1, X.shape[0] + 1))
        common = {"method": a.model, "tag": a.tag, "dataset": a.dataset, "window": f"{win:.2f}"}
        for seed in seeds:
            base = os.path.join(a.out_dir, f"{a.model}_{a.dataset}_w{win:.2f}_s{seed}")
            out, tout = base + "_per_subject.csv", base + "_timing.csv"
            if os.environ.get("OVERWRITE") == "1":
                for p in (out, tout):
                    if os.path.exists(p):
                        os.remove(p)
            done = set()
            if os.path.exists(out):
                with open(out) as f:
                    done = {int(r["fold"]) for r in csv.DictReader(f) if r["tag"] == a.tag}
            for path, fields in ((out, FIELDS), (tout, TFIELDS)):
                if not os.path.exists(path):
                    with open(path, "w", newline="") as f:
                        csv.DictWriter(f, fieldnames=fields).writeheader()
            for fold in folds:
                if fold in done:
                    continue
                ck = os.path.join(a.ckpt_dir, os.path.basename(base) + f"_f{fold}.pt") \
                    if a.ckpt_dir else None
                te, n, timing, val, extra = run_fold(torch, Model, hp, X, y, meta, fold, seed,
                                                     a.n_val, device, a.ptq, ck)
                with open(out, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=FIELDS)
                    # dong PTQ ghi truoc, dong <tag> ghi sau cung: co dong <tag> = fold da du
                    for suf, acc in extra.items():
                        w.writerow({**common, "tag": a.tag + suf, "seed": seed, "fold": fold,
                                    "subject": fold, "acc": f"{acc:.6f}", "n": n})
                    w.writerow({**common, "seed": seed, "fold": fold, "subject": fold,
                                "acc": f"{te:.6f}", "n": n})
                with open(tout, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=TFIELDS).writerow(
                        {**common, "seed": seed, "fold": fold, **timing})
                print(f"KQ {a.model} w={win} fold={fold} seed={seed} test={te:.4f} "
                      f"val={timing['val_acc']} train={timing['train_acc']} "
                      f"ep={timing['epochs_run']}(best {timing['best_epoch']}) "
                      f"s/ep={timing['sec_per_epoch']} params={timing['params']} "
                      f"macs={timing['macs']}" +
                      "".join(f" {k[1:]}={v:.4f}" for k, v in extra.items()), flush=True)
    print("DONE", a.model, a.dataset, flush=True)


if __name__ == "__main__":
    main()
