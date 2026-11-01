"""Golden model fixed-point cua bo giai ma G3. CHI DUNG numpy (int64).

Day la DINH NGHIA phep tinh nguyen cho phan cung; golden.cpp phai khop tung bit voi file nay.
Quy uoc: lam tron half-up bang (v + 2^(s-1)) >> s (dich so hoc), bao hoa ve so bit chi dinh.

Tham so (tao boi export_params trong golden_eval.py, tu checkpoint QAT):
  sos_b[nb][L][3], sos_a[nb][L][2], sos_fb[nb][L]  : he so biquad (b 18 bit voi so mu fb, a Q2.15)
  mx, shx       : hang nhan luong tu y (27 bit, LSB 2^-10 uV) -> x8
  w8[F][C]      : loc khong gian int8
  p8[NB][K][F][T], r8[K][2NH][T] : template chieu va sin/cos (int8, thang 1/127)
  ktm, shtm, ke, she : hang nhan dac trung
  a8[D], bint   : ham diem
  dc_on, mdc, shdc : G3.1 — tru TB doan truoc kich thich [0, dc_on) truoc bank IIR; dc_on = 0: G3
"""
import numpy as np

IN_SHIFT, D_SHIFT, D_BITS, A_FRAC, B_BITS = 6, 10, 27, 15, 18
U_FRAC = 8                       # u = cosine * 127 * 2^U_FRAC
LUT_BITS = 10                    # chi so LUT 1/sqrt: 10 bit dau cua phan dinh tri


def sat(v, bits):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return np.clip(v, lo, hi).astype(np.int64)


def rsr(v, s):
    """Dich phai co lam tron half-up; s co the la mang, s <= 0 nghia la dich trai."""
    v = np.asarray(v, dtype=np.int64)
    s = np.asarray(s, dtype=np.int64)
    sp = np.maximum(s, 0)
    rnd = np.where(sp > 0, np.left_shift(np.int64(1), np.maximum(sp - 1, 0)), 0)
    right = np.right_shift(v + rnd, sp)
    left = np.left_shift(v, np.maximum(-s, 0))
    return np.where(s > 0, right, left).astype(np.int64)


def make_mult(k, mbits=16):
    """Hang so thuc k > 0 -> (M, sh) voi M < 2^(mbits-1), k ~ M * 2^-sh."""
    e = int(np.floor(np.log2(k)))
    sh = (mbits - 2) - e
    m = int(round(k * 2.0 ** sh))
    if m >= (1 << (mbits - 1)):
        m, sh = int(round(k * 2.0 ** (sh - 1))), sh - 1
    return m, sh


def rsqrt_lut():
    """LUT[i] = round(2^30 / sqrt((i + 256 + 0,5) * 2^20)), i = 0..767 (17 bit khong dau)."""
    i = np.arange(256, 1 << LUT_BITS, dtype=np.float64)
    return np.round(2.0 ** 30 / np.sqrt((i + 0.5) * 2.0 ** 20)).astype(np.int64)


LUT = rsqrt_lut()


def bit_length(n):
    """So bit cua n >= 1 (int64 < 2^53): bang frexp, khop 64 - clzll trong C++."""
    return np.frexp(n.astype(np.float64))[1].astype(np.int64)


def inv_sqrt(n):
    """1/sqrt(n) ~ lut * 2^-(30 + q). Tra ve (lut, q); n = 0 -> lut = 0."""
    n = np.asarray(n, dtype=np.int64)
    nz = np.maximum(n, 1)
    b = bit_length(nz)
    q = np.floor_divide(b - 29, 2)                     # n / 4^q co 29 hoac 30 bit
    m = np.where(q >= 0, np.right_shift(nz, np.maximum(2 * q, 0)), np.left_shift(nz, np.maximum(-2 * q, 0)))
    idx = np.right_shift(m, 20)                        # [256, 1024)
    lut = np.where(n > 0, LUT[idx - 256], 0)
    return lut.astype(np.int64), q.astype(np.int64)


def dc_remove(x16, P):
    """G3.1: dc = rsr(sum_{t<ON} x * MDC, SHDC); x' = sat16(x - dc) cho moi t. dc_on = 0 -> giu nguyen."""
    on = int(P.get("dc_on", 0))
    x = np.asarray(x16, dtype=np.int64)
    if on <= 0:
        return x
    S = x[..., :on].sum(axis=-1, keepdims=True)
    dc = rsr(S * P["mdc"], P["shdc"])
    return sat(x - dc, 16)


def iir_bank(x16, P):
    """x16: (..., C, Ttot) int16 -> y: (NB, ..., C, Ttot) int64 (27 bit, LSB 2^-10 uV).
    P['a_frac']: so bit phan le cua he so a (mac dinh A_FRAC = 15, tuc he so 18 bit)."""
    af = P.get("a_frac", A_FRAC)
    x0 = np.left_shift(dc_remove(x16, P), D_SHIFT - IN_SHIFT)
    outs = []
    for n in range(len(P["sos_b"])):
        x = x0
        for (b0, b1, b2), (a1, a2), fb in zip(P["sos_b"][n], P["sos_a"][n], P["sos_fb"][n]):
            Tt = x.shape[-1]
            y = np.zeros_like(x)
            x1 = x2 = y1 = y2 = np.zeros(x.shape[:-1], dtype=np.int64)
            for t in range(Tt):
                xt = x[..., t]
                accb = b0 * xt + b1 * x1 + b2 * x2
                acc = rsr(accb, fb - af) - a1 * y1 - a2 * y2
                yt = sat(rsr(acc, af), D_BITS)
                y[..., t] = yt
                x2, x1, y2, y1 = x1, xt, y1, yt
            x = y
        outs.append(x)
    return np.stack(outs)


def decide(x16, P, return_all=False):
    """x16: (B, C, Ttot) int16, Ttot = s0 + T -> (pred (B,), score (B, K)) [+ trung gian]."""
    T = P["T"]
    y = iir_bank(x16, P)[..., -T:]                                  # (NB, B, C, T)
    x8 = sat(rsr(y * P["mx"], P["shx"]), 8)
    z = np.einsum("fc,nbct->bnft", P["w8"], x8)                     # (B, NB, F, T)
    N = np.einsum("bnft,bnft->bnf", z, z)
    lut, q = inv_sqrt(N)
    sh = 30 + q - U_FRAC                                            # (B, NB, F)
    ctm = np.einsum("bnft,nkft->bknf", z, P["p8"])                  # (B, K, NB, F)
    csc = np.einsum("bnft,kjt->bknfj", z, P["r8"])                  # (B, K, NB, F, 2NH)
    utm = rsr(ctm * lut[:, None], sh[:, None])
    usc = rsr(csc * lut[:, None, :, :, None], sh[:, None, :, :, None])
    ftm = sat(rsr(utm * P["ktm"], P["shtm"]), 8)
    e = usc[..., 0::2] ** 2 + usc[..., 1::2] ** 2                   # (B, K, NB, F, NH)
    fe = sat(rsr(e * P["ke"], P["she"]), 8)
    B, K = ftm.shape[:2]
    phi = np.concatenate([ftm.reshape(B, K, -1), fe.reshape(B, K, -1)], axis=-1)
    score = phi @ P["a8"] + P["bint"]
    pred = np.argmax(score, axis=1)
    if return_all:
        return pred, score, dict(y=y, x8=x8, z=z, N=N, lut=lut, q=q, ftm=ftm, fe=fe, phi=phi)
    return pred, score
