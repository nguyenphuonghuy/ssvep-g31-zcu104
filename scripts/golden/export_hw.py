"""Xuat du lieu cho kernel HLS nap tham so khi chay. Chay trong job (can torch, scipy).

Voi moi cua so (0,4; 1,0) va moi checkpoint wp3_g3_benchmark_w<win>_s0_f<fold>.pt (fold = nguoi kiem thu):
  fold<f>_params.bin  : khoi tham so int32 LE theo bo cuc PRM (ben duoi), nap vao kernel o che do 0
  fold<f>_input.bin   : 240 luot thu int16 (C x (S0+T)) cua nguoi kiem thu (giong golden_eval.py)
  fold<f>_score.bin, fold<f>_pred.bin : golden (golden_model.decide), int64
  fold<f>_label.bin   : nhan (int64)
  dims.h              : kich thuoc + RSQRT_LUT + PRM_LEN (dung chung moi fold)
Tu kiem (exit 3): accuracy golden moi fold trung golden_eval.csv (job 17063).
Bo cuc PRM (int32): [MAGIC, T, MX, SHX, KTM, SHTM, KE, SHE, BINT, NSEC[NB], SOS_B[NB][MAXSEC][3], SOS_A[NB][MAXSEC][2],
  SOS_FB[NB][MAXSEC], W8[F][C], A8[D], P8[NB][K][F][T], R8[K][J][T]]
Dung: python export_hw.py --ckpt_dir <dir> --golden_csv <csv> --out_dir <dir>
: --dataset beta --seed 0 --s0_hw 160: neu s0 cua bo du lieu < s0_hw (S0 co dinh cua bitstream), chen s0_hw - s0 ban sao
  mau dau tien vao dau moi luot thu; score/pred ky vong = golden tren DUNG dau vao da chen (S0 = s0_hw, DC_ON giu nguyen).
  Tu kiem accuracy so voi golden_csv dung khung CHUAN (khong chen); accuracy khung chen ghi vao frame_acc.csv.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import golden_model as G  # noqa: E402
from golden_eval import design_sos, export_params, to_int16  # noqa: E402

MAGIC = 0x47335032          # "G3P2" (G3)
MAGIC_DC = 0x47335033       # "G3P3" (G3.1, : cung bo cuc, kernel co tru DC)
MAXSEC = 7


def pack(P, NB, K, F, C, J, D, T, magic=MAGIC):
    def pad(lst, w):
        return [list(s) + [[0] * w] * (MAXSEC - len(s)) for s in lst]
    parts = [[magic, T, P["mx"], P["shx"], P["ktm"], P["shtm"], P["ke"], P["she"], P["bint"]],
             [len(s) for s in P["sos_b"]],
             np.array(pad(P["sos_b"], 3)).reshape(-1), np.array(pad(P["sos_a"], 2)).reshape(-1),
             np.array([list(s) + [0] * (MAXSEC - len(s)) for s in P["sos_fb"]]).reshape(-1),
             P["w8"].reshape(-1), P["a8"].reshape(-1), P["p8"].reshape(-1), P["r8"].reshape(-1)]
    v = np.concatenate([np.asarray(p, dtype=np.int64).reshape(-1) for p in parts])
    assert np.all(np.abs(v) < 2 ** 31)
    assert P["p8"].shape == (NB, K, F, T) and P["r8"].shape == (K, J, T) and len(P["a8"]) == D
    return v.astype("<i4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--golden_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="wp3_g3", help="wp3_g3 (G3) hoac wp3_g3_dc (G3.1)")
    ap.add_argument("--dataset", default="benchmark")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--s0_hw", type=int, default=0, help="S0 co dinh cua bitstream; 0 = dung s0 cua bo du lieu")
    a = ap.parse_args()
    from ssvep.wp3_models import VARIANTS
    prep = VARIANTS[a.model][1]
    assert prep in ("fb5c", "fb5cd"), prep
    magic = MAGIC_DC if prep == "fb5cd" else MAGIC
    print(f"mo hinh {a.model}, prep {prep}, MAGIC {magic:#x}", flush=True)
    import torch
    from ssvep.data import load_subject_pre
    from ssvep.prep import prepare_causal
    ref = {}
    with open(a.golden_csv) as f:
        for r in csv.DictReader(f):
            if int(r.get("seed", 0)) == a.seed:
                ref[(r["window"], int(r["fold"]))] = float(r["acc_golden"])
    ds = a.dataset
    from ssvep.data import load_cfg
    nsub = load_cfg(ds)["n_subjects"]
    frame_rows = []
    for win in (0.4, 1.0):
        _, y, meta = prepare_causal(ds, win, prep)
        fs = meta["srate"]
        T = int(round(win * fs))
        s0 = meta["onset_in_array"] + int(round(meta["latency_s"] * fs))
        dc_on = int(meta["onset_in_array"]) if prep == "fb5cd" else 0
        pad = max(a.s0_hw - s0, 0)
        assert a.s0_hw == 0 or a.s0_hw >= s0, (a.s0_hw, s0)
        s0h = s0 + pad
        filt = design_sos(fs)
        d = os.path.join(a.out_dir, f"T{T}")
        os.makedirs(d, exist_ok=True)
        cks = sorted(glob.glob(os.path.join(a.ckpt_dir, f"{a.model}_{ds}_w{win:.2f}_s{a.seed}_f*.pt")),
                     key=lambda p: int(re.search(r"_f(\d+)\.pt$", p).group(1)))
        assert len(cks) == nsub, len(cks)
        man = None
        for ck in cks:
            fold = int(re.search(r"_f(\d+)\.pt$", ck).group(1))
            sd = {k: v.numpy() for k, v in torch.load(ck, map_location="cpu").items()}
            P = export_params(sd, T, fs, s0h, filt, dc_on=dc_on)
            NB, K, F = P["p8"].shape[:3]
            C, J, D = P["w8"].shape[1], P["r8"].shape[1], len(P["a8"])
            prm = pack(P, NB, K, F, C, J, D, T, magic)
            Xp, _ = load_subject_pre(ds, fold)
            x16 = to_int16(Xp.reshape(-1, *Xp.shape[2:])[..., :s0 + T])            # (n luot, C, s0+T)
            pg, score = G.decide(x16, P)
            acc = float((pg == y).mean())                                         # khung chuan (tu kiem)
            if pad:                                                               # chen pad ban sao mau dau
                x16 = np.concatenate([np.repeat(x16[..., :1], pad, axis=-1), x16], axis=-1)
                pg, score = G.decide(x16, P)
            frame_rows.append(dict(window=f"{win:.1f}", fold=fold, acc_std=acc, acc_hw_frame=float((pg == y).mean()),
                                   pad=pad))
            want = ref[(f"{win:.1f}", fold)]
            if abs(acc - want) > 1e-9:
                print(f"TU KIEM KHONG DAT: w={win} fold={fold} acc golden {acc} != golden_eval.csv {want}")
                sys.exit(3)
            prm.tofile(os.path.join(d, f"fold{fold}_params.bin"))
            x16.astype("<i2").tofile(os.path.join(d, f"fold{fold}_input.bin"))
            score.astype("<i8").tofile(os.path.join(d, f"fold{fold}_score.bin"))
            pg.astype("<i8").tofile(os.path.join(d, f"fold{fold}_pred.bin"))
            y.astype("<i8").tofile(os.path.join(d, f"fold{fold}_label.bin"))
            man = dict(model=a.model, dataset=ds, seed=a.seed, PAD=pad, NFOLD=nsub, MAGIC=magic, DC_ON=dc_on, MDC=P["mdc"],
                       SHDC=P["shdc"], T=T, S0=s0h, NB=NB, K=K, F=F, C=C, J=J, D=D, B=len(y), PRM_LEN=len(prm))
            print(f"w={win} fold={fold}: acc golden {100 * acc:.2f}% (khop golden_eval.csv) PRM_LEN={len(prm)}", flush=True)
        json.dump(man, open(os.path.join(d, "manifest.json"), "w"), indent=1)
        L = ["// Tu dong sinh boi export_hw.py. Kich thuoc co dinh theo cua so; tham so nap khi chay.",
             "#pragma once", "#include <cstdint>",
             f"constexpr int NB={man['NB']}, K={man['K']}, F={man['F']}, C={man['C']}, T={T}, J={man['J']}, "
             f"NH={man['J'] // 2}, S0={s0h}, D={man['D']}, MAXSEC={MAXSEC};",
             f"constexpr int A_FRAC={G.A_FRAC};", f"constexpr int PRM_LEN={man['PRM_LEN']};",
             f"constexpr int32_t PRM_MAGIC={magic};",
             *(["#define G3_DC 1", f"constexpr int DC_ON={dc_on};", f"constexpr int64_t MDC={P['mdc']}, SHDC={P['shdc']};"]
               if dc_on > 0 else []),
             "constexpr int32_t RSQRT_LUT[768]={" + ",".join(str(int(v)) for v in G.LUT) + "};"]
        open(os.path.join(d, "dims.h"), "w").write("\n".join(L) + "\n")
        print(f"w={win}: {nsub} fold -> {d}; tu kiem accuracy golden: DAT", flush=True)
    if any(r["pad"] for r in frame_rows):
        with open(os.path.join(a.out_dir, "frame_acc.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(frame_rows[0]))
            wr.writeheader()
            wr.writerows(frame_rows)
        for w in sorted({r["window"] for r in frame_rows}):
            rr = [r for r in frame_rows if r["window"] == w]
            print(f"w={w}: acc golden khung chuan {100 * np.mean([r['acc_std'] for r in rr]):.2f}%, "
                  f"khung bitstream (chen {rr[0]['pad']}) {100 * np.mean([r['acc_hw_frame'] for r in rr]):.2f}%", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
