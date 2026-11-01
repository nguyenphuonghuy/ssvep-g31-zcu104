"""Tien xu ly theo tung mo hinh. Moi ham nhan Xw (..., C, T) float64 (cua so da cat),
tra ve (..., n_band, C, T). Loc thuan-nghich tren chinh cua so (giong FBCCA).
"""
import numpy as np
from scipy.signal import butter, cheby1, sosfiltfilt

from . import cca


def _sos_filt(sos, x):
    n_sec = sos.shape[0]
    pad = min(3 * (2 * n_sec), x.shape[-1] - 1)
    return sosfiltfilt(sos, x, axis=-1, padlen=pad)


def bp6_80(Xw, fs):
    """Mot dai 6–80 Hz (cung bo loc voi CCA), cho EEGNet / Compact-CNN."""
    return cca.apply_filter(cca.bandpass_filter(fs), Xw)[..., None, :, :]


def fbtcnn4(Xw, fs):
    """FB-tCNN (Ding et al. 2021, FB-tCNN_train.py): Butterworth bac 6, 4 dai
    [3,14], [9,26], [14,38], [19,50] Hz."""
    bands = [(3, 14), (9, 26), (14, 38), (19, 50)]
    out = [_sos_filt(butter(6, [lo / (fs / 2), hi / (fs / 2)], "bandpass", output="sos"), Xw)
           for lo, hi in bands]
    return np.stack(out, axis=-3)


def guney3_sos(fs):
    """3 bo loc cua guney3 (dang SOS) — dung chung cho loc pha khong (guney3) va nhan qua (guney3c/cd)."""
    return [cheby1(1, 1, [8 * i / (fs / 2), 90 / (fs / 2)], "bandpass", output="sos") for i in range(1, 4)]


def guney3(Xw, fs):
    """DNN (Guney et al. 2022, PreProcess.m): Chebyshev I, FilterOrder 2 (MATLAB designfilt
    'bandpassiir' -> cheby1 bac 1 cho bandpass trong scipy), ripple 1 dB, dai [8i, 90] Hz, i=1..3."""
    out = []
    for i in range(1, 4):
        sos = cheby1(1, 1, [8 * i / (fs / 2), 90 / (fs / 2)], "bandpass", output="sos")
        out.append(_sos_filt(sos, Xw))
    return np.stack(out, axis=-3)


def fb5(Xw, fs):
    """WP2: bank loc 5 dai con giong het FBCCA (cca.subband_filters, filtfilt tren cua so)."""
    return np.stack([cca.apply_filter(ba, Xw) for ba in cca.subband_filters(fs, 5)], axis=-3)


PREP = {"bp6_80": bp6_80, "fbtcnn4": fbtcnn4, "guney3": guney3, "fb5": fb5}


def fb1(Xw, fs):
    """WP3 ablation: bo bank loc, chi dai con 1 cua FBCCA (6-90 Hz), filtfilt tren cua so."""
    return cca.apply_filter(cca.subband_filters(fs, 1)[0], Xw)[..., None, :, :]


PREP["fb1"] = fb1


def _lfilter_ss(b, a, x):
    """ (fb5cz): loc mot chieu, trang thai dau = trang thai xac lap cho dau vao hang bang mau dau x[..., 0]."""
    from scipy.signal import lfilter, lfilter_zi
    zi = lfilter_zi(b, a) * x[..., :1]
    return lfilter(b, a, x, axis=-1, zi=zi)[0]


def _stream_bands(X, fs, s0, n):
    """ (fb5cs): noi moi luot thu (khoi, roi muc tieu; toan bo epoch da luu) thanh mot luong moi kenh, loc mot lan
    voi trang thai mang qua cac luot (khoi tao xac lap o mau dau luong), roi cat cua so [s0, s0 + n) cua tung luot.
    X: (block, target, C, T) -> (block, target, 5, C, n)."""
    nb, nk, C, T = X.shape
    st = X.transpose(2, 0, 1, 3).reshape(C, nb * nk * T)
    out = []
    for b, a in cca.subband_filters(fs, 5):
        y = _lfilter_ss(b, a, st).reshape(C, nb, nk, T).transpose(1, 2, 0, 3)
        out.append(y[..., s0:s0 + n])
    return np.stack(out, axis=-3)


def prepare_causal(ds, win, prep):
    """WP3 loc nhan qua: bank loc 5 dai con giong FBCCA nhung loc MOT CHIEU (lfilter, trang thai
    dau = 0) tren doan lien tuc tu dau du lieu truoc kich thich (0,5 s o Benchmark) toi het cua so, roi cat
    cua so [tre, tre + win). Mo phong loc lien tuc tren luong du lieu (khong dung mau tuong lai)."""
    from scipy.signal import lfilter, sosfilt
    from .data import load_cfg, load_subject_pre
    assert prep in ("fb5c", "fb5cd", "guney3c", "guney3cd", "fb5cz", "fb5cs"), prep
    n_sub = load_cfg(ds)["n_subjects"]
    Xs, meta = [], None
    for s in range(1, n_sub + 1):
        X, meta = load_subject_pre(ds, s)                  # (block, target, C, t), mau on = kich thich
        fs, on = meta["srate"], meta["onset_in_array"]
        s0 = on + int(round(meta["latency_s"] * fs))
        n = int(round(win * fs))
        if prep == "fb5cs":                                # luong noi, trang thai mang qua cac luot thu
            Xp = _stream_bands(X.astype(np.float64), fs, s0, n)
            nb, nk = Xp.shape[:2]
            Xs.append(Xp.reshape(nb * nk, *Xp.shape[2:]).astype(np.float32))
            continue
        seg = X[..., :s0 + n].astype(np.float64)
        if prep in ("fb5cd", "guney3cd"):                  # tru TB doan truoc kich thich (nhan qua)
            seg = seg - seg[..., :on].mean(axis=-1, keepdims=True)
        if prep.startswith("guney3"):                      # 3 dai cua guney3 (Ensemble-DNN), loc MOT CHIEU
            bands = [sosfilt(sos, seg, axis=-1)[..., s0:s0 + n] for sos in guney3_sos(fs)]
        elif prep == "fb5cz":                              # khoi tao xac lap theo mau dau
            bands = [_lfilter_ss(b, a, seg)[..., s0:s0 + n] for b, a in cca.subband_filters(fs, 5)]
        else:
            bands = [lfilter(b, a, seg, axis=-1)[..., s0:s0 + n] for b, a in cca.subband_filters(fs, 5)]
        Xp = np.stack(bands, axis=-3)                      # block, target, 5, C, n
        nb, nk = Xp.shape[:2]
        Xs.append(Xp.reshape(nb * nk, *Xp.shape[2:]).astype(np.float32))
    y = np.tile(np.arange(nk), nb)
    return np.stack(Xs), y, meta
