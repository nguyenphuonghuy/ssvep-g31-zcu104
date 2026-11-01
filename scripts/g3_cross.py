"""G3 kiem tra doc lap cheo bo: huan luyen tren Benchmark, kiem tra tren BETA.

- Mo hinh: cau hinh dong bang wp3_g3 (lai H-C #21, QAT 8 bit, loc nhan qua fb5c).
- Huan luyen: 35 nguoi Benchmark; 4 nguoi validation chon tat dinh RandomState(2026) (dung som, chon
  checkpoint); template nhom tu 31 nguoi train. Sieu tham so nhu vong tim kiem.
- Kiem tra: tung nguoi BETA (70), nhan BETA anh xa sang chi so Benchmark theo TAN SO (pha khop sau dinh
  chinh 2022 cua BETA, xem ). Khong nguoi BETA nao tham gia huan luyen/chon checkpoint.
Dau ra: <out_dir>/g3_cross_beta_w<win>_s<seed>_per_subject.csv (method,tag,dataset,window,seed,fold,subject,acc,n)
Dung: python g3_cross.py --out_dir <dir> --seeds 0 --windows 0.4,1.0 [--max_epochs N]
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

FIELDS = ["method", "tag", "dataset", "window", "seed", "fold", "subject", "acc", "n"]
VAL_SEED = 2026


def label_map(meta_src, meta_tgt):
    """chi so muc tieu tgt -> chi so src cung tan so."""
    src = {round(f, 2): i for i, f in enumerate(meta_src["freqs"])}
    m = [src[round(f, 2)] for f in meta_tgt["freqs"]]
    assert sorted(m) == list(range(len(m))), "anh xa tan so khong phai hoan vi"
    return np.array(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--windows", default="0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--max_epochs", type=int, default=0)
    a = ap.parse_args()
    import torch
    from ssvep.models import count_params
    from ssvep.wp3_models import make_variant
    from train_loso import prepare
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 1)))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Model, hp = make_variant("wp3_g3")
    if a.max_epochs:
        hp["max_epochs"] = a.max_epochs
    print(f"torch {torch.__version__} {dev}; wp3_g3 cfg={Model.config} prep={Model.prep} hp={hp}", flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    for win in [float(w) for w in a.windows.split(",")]:
        Xs, ys, ms = prepare("benchmark", win, Model.prep)
        Xt, yt, mt = prepare("beta", win, Model.prep)
        mp = label_map(ms, mt)
        yt_src = mp[yt]
        subs = list(range(1, Xs.shape[0] + 1))
        val = sorted(np.random.RandomState(VAL_SEED).choice(subs, 4, replace=False).tolist())
        train = [s for s in subs if s not in val]
        print(f"w={win}: Benchmark X={Xs.shape}, BETA X={Xt.shape}; val={val}", flush=True)
        for seed in [int(s) for s in a.seeds.split(",")]:
            out = os.path.join(a.out_dir, f"g3_cross_beta_w{win:.2f}_s{seed}_per_subject.csv")
            if os.path.exists(out):
                print(f"bo qua {out} (da co)")
                continue

            def stack(X, ss, y):
                xx = np.concatenate([X[s - 1] for s in ss])
                return torch.from_numpy(xx).to(dev), torch.from_numpy(np.tile(y, len(ss))).long().to(dev)

            xtr, ytr = stack(Xs, train, ys)
            xva, yva = stack(Xs, val, ys)
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = Model(Xs.shape[3], Xs.shape[4], len(ms["freqs"]), ms["srate"], freqs=ms["freqs"]).to(dev)
            K = len(ms["freqs"])
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
            rows = []
            for s in range(1, Xt.shape[0] + 1):
                xx = torch.from_numpy(Xt[s - 1]).to(dev)
                acc = evaluate(xx, torch.from_numpy(yt_src).long().to(dev))
                rows.append({"method": "g3_cross", "tag": "base", "dataset": "beta", "window": f"{win:.2f}",
                             "seed": seed, "fold": 0, "subject": s, "acc": f"{acc:.6f}", "n": len(yt_src)})
            with open(out, "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=FIELDS)
                wr.writeheader()
                wr.writerows(rows)
            print(f"KQ g3_cross w={win} seed={seed} val={best:.4f} ep={ep}(best {best_ep}) "
                  f"BETA TB={100 * np.mean([float(r['acc']) for r in rows]):.2f}% params={count_params(model)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    print("DONE g3_cross", flush=True)


if __name__ == "__main__":
    main()
