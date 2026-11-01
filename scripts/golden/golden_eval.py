"""Xuat tham so nguyen tu checkpoint QAT G3, danh gia golden model fixed-point, sinh header C + vector kiem thu.

. Chay trong job (can torch, scipy).
Voi moi checkpoint wp3_g3_benchmark_w<win>_s0_f<fold>.pt (fold = nguoi kiem thu):
  - export_params -> tham so nguyen (golden_model.py);
  - golden: 240 luot thu cua nguoi kiem thu (du lieu tho int16 tu cache benchmark_pre) -> du doan;
  - QAT: mo hinh fake-quant cung checkpoint tren du lieu loc nhan qua dau phay dong -> du doan;
  - SNR bank IIR fixed-point so voi lfilter dau phay dong tren cung dau vao int16.
Dau ra <out_dir>/golden_eval.csv (window, fold, acc_qat, acc_golden, agree, snr_min_db, snr_mean_db),
<out_dir>/golden_summary.md; voi --vectors_fold: <vec_dir>/T<T>/params.h, input.bin, expected_*.bin, manifest.json.
Dung: python golden_eval.py --ckpt_dir <dir> --out_dir <dir> [--vec_dir <dir> --vectors_fold 1 --n_vec 8]
: --dataset benchmark|beta|nakanishi, --seeds 0,1,..., --ckpt_dir nhieu thu muc cach nhau dau phay;
  them cot dataset, seed; <out_dir>/saturation.csv: mau |x| vuot int16 (buoc 2^-6 uV) tren doan [0, s0 + T).
  Mac dinh (benchmark, seed 0) cho cung so lieu nhu ban truoc.
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import golden_model as G  # noqa: E402


def linf_scale(sos, n=8192):
    """Co gian L-inf: chia lai he so b tung section de dap ung tich luy toi moi section co dinh = 1.
    Ham truyen tong khong doi (dinh cua ca bo loc Chebyshev I = 1); tranh tin hieu trung gian nho hon LSB."""
    from scipy.signal import sosfreqz
    s = np.array(sos, dtype=np.float64)
    peak0 = np.abs(sosfreqz(s, worN=n)[1]).max()
    for i in range(len(s)):
        m = np.abs(sosfreqz(s[:i + 1], worN=n)[1]).max()
        s[i, :3] /= m
    s[-1, :3] *= peak0                       # giu nguyen do loi tong cua bo loc goc
    return s


def design_sos(fs, n_fb=5):
    """Cung bo loc voi ssvep/cca.subband_filters nhung dang SOS da co gian L-inf (va ba de doi chieu)."""
    from scipy.signal import cheb1ord, cheby1
    from ssvep.cca import PASS, STOP
    nq = fs / 2.0
    out = []
    for i in range(n_fb):
        wp, ws = [PASS[i] / nq, 90 / nq], [STOP[i] / nq, 100 / nq]
        N, wn = cheb1ord(wp, ws, 3, 40)
        out.append((linf_scale(cheby1(N, 0.5, wn, btype="bandpass", output="sos")),
                    cheby1(N, 0.5, wn, btype="bandpass")))
    return out


def quant_sos(sos, a_frac=G.A_FRAC):
    b_bits = a_frac + 3                       # he so a: Q2.<a_frac>; b cung so bit, so mu rieng
    bq, aq, fbs = [], [], []
    for b0, b1, b2, _, a1, a2 in sos:
        bmax = max(abs(b0), abs(b1), abs(b2))
        fb = int(math.floor(math.log2(((1 << (b_bits - 1)) - 1) / bmax)))
        bq.append([int(round(v * 2.0 ** fb)) for v in (b0, b1, b2)])
        aq.append([int(round(v * 2.0 ** a_frac)) for v in (a1, a2)])
        assert all(abs(v) < (1 << (b_bits - 1)) for v in bq[-1] + aq[-1]), (bq[-1], aq[-1])
        fbs.append(fb)
    return bq, aq, fbs


def export_params(sd, T, fs, s0, filt, a_frac=G.A_FRAC, dc_on=0):
    """dc_on > 0: G3.1, tru TB [0, dc_on) bang hang nhan (MDC, SHDC) = make_mult(1/dc_on, 24 bit)."""
    W = sd["W"].reshape(sd["W"].shape[-2], sd["W"].shape[-1]).astype(np.float64)       # (F, C)
    s_w = np.abs(W).max(axis=1, keepdims=True) / 127.0
    w8 = np.clip(np.round(W / s_w), -127, 127).astype(np.int64)
    Wq = w8 * s_w
    tm = sd["tmpl"].astype(np.float64)                                                  # (NB, K, C, T)
    p = np.einsum("fc,nkct->nkft", Wq, tm)
    p /= np.linalg.norm(p, axis=-1, keepdims=True) + 1e-12
    p8 = np.clip(np.round(127 * p), -127, 127).astype(np.int64)
    r = sd["refs"].astype(np.float64)
    r /= np.linalg.norm(r, axis=-1, keepdims=True)
    r8 = np.clip(np.round(127 * r), -127, 127).astype(np.int64)
    s_x, s_f = float(sd["q_x.amax"]) / 127.0, float(sd["q_f.amax"]) / 127.0
    mx, shx = G.make_mult(2.0 ** -G.D_SHIFT / s_x)
    ktm, shtm = G.make_mult(1.0 / (127 * 2 ** G.U_FRAC * s_f))
    ke, she = G.make_mult(1.0 / ((127 * 2 ** G.U_FRAC) ** 2 * s_f))
    a = sd["head.weight"].reshape(-1).astype(np.float64)
    s_a = np.abs(a).max() / 127.0
    a8 = np.clip(np.round(a / s_a), -127, 127).astype(np.int64)
    bint = int(round(float(sd["head.bias"].reshape(-1)[0]) / (s_a * s_f)))
    sb, sa, sf = zip(*[quant_sos(sos, a_frac) for sos, _ in filt])
    mdc, shdc = G.make_mult(1.0 / dc_on, 24) if dc_on > 0 else (0, 0)
    return dict(T=T, s0=s0, a_frac=a_frac, dc_on=int(dc_on), mdc=mdc, shdc=shdc, sos_b=list(sb), sos_a=list(sa), sos_fb=list(sf), mx=mx, shx=shx, w8=w8, p8=p8, r8=r8,
                ktm=ktm, shtm=shtm, ke=ke, she=she, a8=a8, bint=bint)


def to_int16(x, bits=16):
    """Luong tu dau vao buoc 2^-IN_SHIFT uV, bao hoa ve `bits` bit (: bits = 24 cho bien the W24)."""
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return np.clip(np.round(x * 2.0 ** G.IN_SHIFT), lo, hi).astype(np.int64)


def write_header(P, path, C):
    """Header C++ voi moi tham so nguyen (dung lai cho HLS)."""
    NB, K, F, T = P["p8"].shape[0], P["p8"].shape[1], P["w8"].shape[0], P["T"]
    J = P["r8"].shape[1]
    L = [f"// Tu dong sinh boi golden_eval.py. T={T}", "#pragma once", "#include <cstdint>",
         f"constexpr int NB={NB}, K={K}, F={F}, C={C}, T={T}, J={J}, NH={J // 2}, S0={P['s0']}, D={len(P['a8'])};",
         f"constexpr int A_FRAC={P['a_frac']};",
         *([f"#define G3_DC 1", f"constexpr int DC_ON={P['dc_on']};",
            f"constexpr int64_t MDC={P['mdc']}, SHDC={P['shdc']};"] if P.get("dc_on", 0) > 0 else []),
         f"constexpr int64_t MX={P['mx']}, SHX={P['shx']}, KTM={P['ktm']}, SHTM={P['shtm']}, KE={P['ke']}, SHE={P['she']}, BINT={P['bint']};"]
    nsec = [len(s) for s in P["sos_b"]]
    L.append(f"constexpr int NSEC[NB]={{{','.join(map(str, nsec))}}};")
    L.append(f"constexpr int MAXSEC={max(nsec)};")

    def arr(name, a, typ="int64_t"):
        a = np.asarray(a)
        dims = "".join(f"[{d}]" for d in a.shape)
        return f"constexpr {typ} {name}{dims}={{{','.join(str(int(v)) for v in a.reshape(-1))}}};"
    pad = lambda lst, w: [list(s) + [[0] * w] * (max(nsec) - len(s)) for s in lst]  # noqa: E731
    L.append(arr("SOS_B", pad(P["sos_b"], 3)))
    L.append(arr("SOS_A", pad(P["sos_a"], 2)))
    L.append(arr("SOS_FB", [list(s) + [0] * (max(nsec) - len(s)) for s in P["sos_fb"]]))
    L.append(arr("W8", P["w8"], "int8_t"))
    L.append(arr("P8", P["p8"], "int8_t"))
    L.append(arr("R8", P["r8"], "int8_t"))
    L.append(arr("A8", P["a8"], "int8_t"))
    L.append(arr("RSQRT_LUT", G.LUT, "int32_t"))
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--vec_dir", default="")
    ap.add_argument("--vectors_fold", type=int, default=1)
    ap.add_argument("--n_vec", type=int, default=8)
    ap.add_argument("--a_frac", type=int, default=G.A_FRAC, help="bit phan le he so a (he so = a_frac + 3 bit)")
    ap.add_argument("--model", default="wp3_g3", help="wp3_g3 (G3) hoac wp3_g3_dc (G3.1)")
    ap.add_argument("--coef_sweep", default="", help="vd 15,18,21,24: chi do SNR bank IIR theo do rong he so")
    ap.add_argument("--dataset", default="benchmark")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--windows", default="0.4,1.0")
    ap.add_argument("--in_bits", type=int, default=16, help="so bit dau vao (: 24); tru TB xong bao hoa int16")
    a = ap.parse_args()
    import torch
    from scipy.signal import lfilter, sosfilt
    from ssvep.data import load_subject_pre
    from ssvep.prep import prepare_causal
    from ssvep.wp3_models import VARIANTS, make_variant
    os.makedirs(a.out_dir, exist_ok=True)
    Model, _ = make_variant(a.model)
    prep = VARIANTS[a.model][1]
    assert prep in ("fb5c", "fb5cd"), prep
    print(f"mo hinh {a.model}, prep {prep}", flush=True)
    ds, seeds = a.dataset, [int(v) for v in a.seeds.split(",")]
    rows, sat_rows = [], []
    for win in [float(v) for v in a.windows.split(",")]:
        cks = []
        for seed in seeds:
            found = {}
            for d in a.ckpt_dir.split(","):
                for p in glob.glob(os.path.join(d, f"{a.model}_{ds}_w{win:.2f}_s{seed}_f*.pt")):
                    found.setdefault(int(re.search(r"_f(\d+)\.pt$", p).group(1)), p)
            cks += [(seed, f, found[f]) for f in sorted(found)]
        if not cks:
            continue
        Xc, y, meta = prepare_causal(ds, win, prep)
        fs, T = meta["srate"], Xc.shape[-1]
        s0 = meta["onset_in_array"] + int(round(meta["latency_s"] * fs))
        dc_on = int(meta["onset_in_array"]) if prep == "fb5cd" else 0
        filt = design_sos(fs)
        print(f"w={win}: T={T} s0={s0}; so section moi dai: {[len(s) for s, _ in filt]}", flush=True)
        if a.coef_sweep:
            from scipy.signal import sosfilt
            Xp, _ = load_subject_pre(ds, 1)
            xs = to_int16(Xp.reshape(-1, *Xp.shape[2:])[:48, :, :s0 + T])
            xin = xs.astype(np.float64) * 2.0 ** -G.IN_SHIFT
            for af in [int(v) for v in a.coef_sweep.split(",")]:
                sb, sa, sf = zip(*[quant_sos(sos, af) for sos, _ in filt])
                Pf = dict(a_frac=af, sos_b=list(sb), sos_a=list(sa), sos_fb=list(sf))
                yq = G.iir_bank(xs, Pf)[..., -T:]
                snr = []
                for n_, (sos, _) in enumerate(filt):
                    ref = sosfilt(sos, xin, axis=-1)[..., -T:]
                    got = yq[n_] * 2.0 ** -G.D_SHIFT
                    snr.append(10 * np.log10((ref ** 2).sum() / ((ref - got) ** 2).sum()))
                print(f"  [quet he so] w={win} he so {af + 3} bit (Q2.{af}): SNR theo dai "
                      f"{[round(float(v), 1) for v in snr]} dB; min {min(snr):.1f}", flush=True)
            continue
        x16_cache = {}
        for seed, fold, ck in cks:
            sdt = torch.load(ck, map_location="cpu")
            sd = {k: v.numpy() for k, v in sdt.items()}
            P = export_params(sd, T, fs, s0, filt, a.a_frac, dc_on)
            # QAT (fake-quant) tren du lieu dau phay dong
            m = Model(Xc.shape[3], T, len(meta["freqs"]), fs, freqs=meta["freqs"])
            m.load_state_dict(sdt)
            m.eval()
            with torch.no_grad():
                pq = m(torch.from_numpy(Xc[fold - 1])).argmax(1).numpy()
            # golden tren du lieu tho int16
            if fold not in x16_cache:
                Xp, _ = load_subject_pre(ds, fold)
                seg = Xp.reshape(-1, *Xp.shape[2:])[..., :s0 + T]                 # (n luot, C, s0+T)
                raw = np.round(seg * 2.0 ** G.IN_SHIFT)
                lim = 1 << (a.in_bits - 1)
                over = (raw < -lim) | (raw > lim - 1)                             # bao hoa dau vao in_bits
                xi = to_int16(seg, a.in_bits)
                if dc_on > 0:                                                     # vuot int16 SAU tru TB
                    dcv = G.rsr(xi[..., :dc_on].sum(axis=-1, keepdims=True) * G.make_mult(1.0 / dc_on, 24)[0],
                                G.make_mult(1.0 / dc_on, 24)[1])
                    post = np.abs(xi - dcv) > 32767
                else:
                    post = np.abs(xi) > 32767
                sat_rows.append(dict(dataset=ds, window=f"{win:.1f}", fold=fold, n_samples=int(over.size),
                                     n_sat=int(over.sum()), n_trials=int(over.shape[0]),
                                     n_trials_sat=int(over.reshape(over.shape[0], -1).any(axis=1).sum()),
                                     absmax_uv=float(np.abs(seg).max()), in_bits=a.in_bits,
                                     n_sat_post_dc=int(post.sum())))
                x16_cache[fold] = xi
                del Xp, seg, raw, over, post
            x16 = x16_cache[fold]
            pg, score, inter = G.decide(x16, P, return_all=True)
            # SNR bank IIR: fixed-point vs lfilter(ba) dau phay dong tren cung dau vao int16
            snr, snr_sos, snr_basos = [], [], []
            db = lambda r_, g_: 10 * np.log10((r_ ** 2).sum() / max(((r_ - g_) ** 2).sum(), 1e-30))  # noqa: E731
            xin = G.dc_remove(x16, P).astype(np.float64) * 2.0 ** -G.IN_SHIFT      # G3.1: sau tru DC so nguyen
            for n, (sos, (b, aa)) in enumerate(filt):
                ref = lfilter(b, aa, xin, axis=-1)[..., -T:]
                ref_sos = sosfilt(sos, xin, axis=-1)[..., -T:]
                got = inter["y"][n] * 2.0 ** -G.D_SHIFT
                snr.append(db(ref, got))
                snr_sos.append(db(ref_sos, got))
                snr_basos.append(db(ref_sos, ref))
            print(f"    SNR theo dai: golden-vs-ba {[round(v, 1) for v in snr]}, golden-vs-sos {[round(v, 1) for v in snr_sos]}, "
                  f"ba-vs-sos {[round(v, 1) for v in snr_basos]}", flush=True)
            r = dict(dataset=ds, seed=seed, window=f"{win:.1f}", fold=fold, acc_qat=float((pq == y).mean()), acc_golden=float((pg == y).mean()),
                     agree=float((pq == pg).mean()), snr_min_db=float(min(snr)), snr_mean_db=float(np.mean(snr)),
                     snr_sos_min_db=float(min(snr_sos)), snr_ba_vs_sos_min_db=float(min(snr_basos)))
            rows.append(r)
            print(f"  {ds} s{seed} fold {fold}: QAT {100 * r['acc_qat']:.2f}% golden {100 * r['acc_golden']:.2f}% "
                  f"khop {100 * r['agree']:.1f}% SNR IIR min {r['snr_min_db']:.1f} dB", flush=True)
            if a.vec_dir and fold == a.vectors_fold and seed == seeds[0]:
                d = os.path.join(a.vec_dir, f"T{T}")
                os.makedirs(d, exist_ok=True)
                write_header(P, os.path.join(d, "params.h"), x16.shape[1])
                B = a.n_vec
                x16[:B].astype("<i2").tofile(os.path.join(d, "input.bin"))
                pb, sb, ib = G.decide(x16[:B], P, return_all=True)
                for k in ("x8", "z", "N", "phi"):
                    ib[k].astype("<i8").tofile(os.path.join(d, f"expected_{k}.bin"))
                sb.astype("<i8").tofile(os.path.join(d, "expected_score.bin"))
                pb.astype("<i8").tofile(os.path.join(d, "expected_pred.bin"))
                json.dump({"B": B, "C": int(x16.shape[1]), "Ttot": int(x16.shape[2]), "T": T, "fold": fold,
                           "labels": y[:B].tolist(), "pred": pb.tolist(),
                           "shapes": {k: list(ib[k].shape) for k in ("x8", "z", "N", "phi")}},
                          open(os.path.join(d, "manifest.json"), "w"), indent=1)
                print(f"  vector kiem thu -> {d}", flush=True)
    if not rows:
        print("DONE (chi quet he so)")
        return
    with open(os.path.join(a.out_dir, "golden_eval.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(a.out_dir, "saturation.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(sat_rows[0]))
        wr.writeheader()
        wr.writerows(sat_rows)
    L = [f"# Golden model fixed-point {a.model} ({prep}) so với QAT ({ds} LOSO, seed {a.seeds}, đầu vào {a.in_bits} bit)", "",
         "TB theo người của TB các seed.", "",
         "| Cửa sổ | Số fold × seed | Acc QAT TB (%) | Acc golden TB (%) | Chênh golden − QAT | Khớp dự đoán TB (%) | SNR IIR min (dB) | Mẫu bão hòa đầu vào (%) | Lượt thử có bão hòa (%) | Vượt int16 sau trừ TB (%) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for w in sorted({r["window"] for r in rows}):
        rr = [r for r in rows if r["window"] == w]
        fl = sorted({r["fold"] for r in rr})
        subj = lambda k: [np.mean([r[k] for r in rr if r["fold"] == f_]) for f_ in fl]  # noqa: E731
        aq, ag, agr = 100 * np.mean(subj("acc_qat")), 100 * np.mean(subj("acc_golden")), 100 * np.mean(subj("agree"))
        sr = [r for r in sat_rows if r["window"] == w]
        ps = 100 * sum(r["n_sat"] for r in sr) / sum(r["n_samples"] for r in sr)
        pt = 100 * sum(r["n_trials_sat"] for r in sr) / sum(r["n_trials"] for r in sr)
        pp = 100 * sum(r["n_sat_post_dc"] for r in sr) / sum(r["n_samples"] for r in sr)
        L.append(f"| {w} s | {len(rr)} | {aq:.2f} | {ag:.2f} | {ag - aq:+.2f} | {agr:.1f} | "
                 f"{min(r['snr_min_db'] for r in rr):.1f} | {ps:.4f} | {pt:.2f} | {pp:.4f} |")
    open(os.path.join(a.out_dir, "golden_summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("DONE")


if __name__ == "__main__":
    main()
