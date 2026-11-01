"""CCA va FBCCA khong huan luyen (tham chieu hinh sin).

Cau hinh bank loc theo ma nguon TRCA/FBCCA pho bien cua nhom Tsinghua/Nakanishi:
dai con n co passband [8n-2, 90] Hz, stopband [8n-4 (bang tra), 100] Hz,
Chebyshev I (cheb1ord Rp=3, Rs=40; cheby1 ripple 0.5), loc thuan-nghich
tren chinh cua so du lieu. Trong so dai con w_n = n^-1.25 + 0.25 (Chen 2015b).
"""
import numpy as np
from scipy.signal import cheb1ord, cheby1, filtfilt

PASS = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
STOP = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]


def subband_filters(fs, n_fb):
    nq = fs / 2.0
    out = []
    for i in range(n_fb):
        wp = [PASS[i] / nq, 90 / nq]
        ws = [STOP[i] / nq, 100 / nq]
        N, wn = cheb1ord(wp, ws, 3, 40)
        out.append(cheby1(N, 0.5, wn, btype="bandpass"))
    return out


def bandpass_filter(fs, lo=6.0, hi=80.0):
    """Loc thong dai cho CCA khong dung bank loc (BETA paper: 6-80 Hz)."""
    nq = fs / 2.0
    N, wn = cheb1ord([lo / nq, hi / nq], [(lo - 2) / nq, (hi + 10) / nq], 3, 40)
    return cheby1(N, 0.5, wn, btype="bandpass")


def apply_filter(ba, x):
    """filtfilt theo truc thoi gian; padlen nhu MATLAB, cat neu cua so qua ngan."""
    b, a = ba
    pad = min(3 * (max(len(a), len(b)) - 1), x.shape[-1] - 1)
    return filtfilt(b, a, x, axis=-1, padlen=pad)


def sine_refs(freqs, fs, n_t, n_h):
    t = np.arange(1, n_t + 1) / fs
    refs = []
    for f in freqs:
        r = []
        for h in range(1, n_h + 1):
            r += [np.sin(2 * np.pi * h * f * t), np.cos(2 * np.pi * h * f * t)]
        refs.append(np.array(r))
    return refs  # list of (2*n_h, n_t)


def _orth(m):
    m = m - m.mean(axis=1, keepdims=True)
    q, _ = np.linalg.qr(m.T)
    return q


def cca_corr(qx, qy):
    """Tuong quan chinh tac lon nhat giua hai co so truc giao qx (t, c), qy (t, k)."""
    s = np.linalg.svd(qx.T @ qy, compute_uv=False)
    return min(s[0], 1.0)


def make_ref_bases(freqs, fs, n_t, n_h):
    return [_orth(r) for r in sine_refs(freqs, fs, n_t, n_h)]


def cca_predict(x, qrefs, ba):
    qx = _orth(apply_filter(ba, x))
    rho = np.array([cca_corr(qx, q) for q in qrefs])
    return int(np.argmax(rho)), rho


def fbcca_predict(x, qrefs, fbs, a=1.25, b=0.25):
    w = np.array([(n + 1) ** (-a) + b for n in range(len(fbs))])
    score = np.zeros(len(qrefs))
    for n, ba in enumerate(fbs):
        qx = _orth(apply_filter(ba, x))
        rho = np.array([cca_corr(qx, q) for q in qrefs])
        score += w[n] * rho ** 2
    return int(np.argmax(score)), score
