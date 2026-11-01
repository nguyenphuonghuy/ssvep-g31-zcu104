"""Doc .mat goc -> cache float32 (data/cache/<ds>/S<n>.npy + meta.json) va tu kiem.

Tu kiem cho moi doi tuong:
  - shape, ten kenh, bang tan so giong nhau giua cac doi tuong;
  - kiem nhan: SNR pho tai cac tan so ung vien (xem label_check); ty le muc tieu
    khop nhan thap gan muc ngau nhien -> thu tu muc tieu/kenh co the sai.
Dung: python build_cache.py <dataset> [--force]
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ssvep.data import CACHE_ROOT, cache_path, load_cfg, read_subject_raw  # noqa: E402


def label_check(X, meta, max_s=3.0):
    """Ty le muc tieu co SNR pho lon nhat dung tan so cua chinh no.

    Doan dai toi da max_s (gioi han boi do dai trial) sau do tre, khu xu huong tuyen tinh,
    pho trung binh qua block va kenh. SNR(f) = bien do tai f / trung binh bien do trong
    [f-1, f+1] Hz tru [f-0.25, f+0.25] Hz (khu anh huong 1/f). Muc ngau nhien = 1/M.
    """
    from scipy.signal import detrend
    fs = meta["srate"]
    s0 = int(round(meta["latency_s"] * fs))
    n = min(int(max_s * fs), X.shape[-1] - s0)
    seg = detrend(X[..., s0:s0 + n].astype(np.float64), axis=-1)
    nfft = 20 * fs  # luoi 0.05 Hz
    amp = np.abs(np.fft.rfft(seg, n=nfft, axis=-1)).mean(axis=(0, 2))  # target, f
    fgrid = np.fft.rfftfreq(nfft, 1 / fs)
    snr = np.zeros((amp.shape[0], len(meta["freqs"])))
    for j, f in enumerate(meta["freqs"]):
        c = int(np.argmin(np.abs(fgrid - f)))
        nb = (np.abs(fgrid - f) <= 1.0) & (np.abs(fgrid - f) > 0.25)
        snr[:, j] = amp[:, c] / amp[:, nb].mean(axis=1)
    pred = np.argmax(snr, axis=1)
    return float(np.mean(pred == np.arange(len(meta["freqs"]))))


def main():
    name = sys.argv[1]
    force = "--force" in sys.argv
    cfg = load_cfg(name)
    os.makedirs(os.path.join(CACHE_ROOT, name), exist_ok=True)
    ref = None
    report = []
    for s in range(1, cfg["n_subjects"] + 1):
        t = time.time()
        X, meta = read_subject_raw(name, s)
        if ref is None:
            ref = meta
            print("channels:", meta["channels"])
            print("freqs:", meta["freqs"])
            print("phases:", meta["phases"])
        for k in ("channels", "freqs", "srate"):
            assert meta[k] == ref[k], (s, k, meta[k], ref[k])
        if not np.isfinite(X).all():
            raise ValueError(f"S{s}: co NaN/Inf")
        out = cache_path(name, s)
        if force or not os.path.exists(out):
            np.save(out, X)
        lc = label_check(X, meta)
        report.append({"subject": s, "shape": list(X.shape), "label_check": lc})
        print(f"S{s:3d} shape={X.shape} label_check={lc:.3f} {time.time() - t:.1f}s",
              flush=True)
    dmeta = {k: ref[k] for k in ("dataset", "srate", "latency_s", "channels", "freqs",
                                 "phases")}
    dmeta["n_subjects"] = cfg["n_subjects"]
    dmeta["subjects"] = report
    with open(os.path.join(CACHE_ROOT, name, "meta.json"), "w") as f:
        json.dump(dmeta, f, indent=1)
    lcs = [r["label_check"] for r in report]
    print(f"label_check trung binh={np.mean(lcs):.3f} min={np.min(lcs):.3f}")
    print("CACHE_OK", name)


if __name__ == "__main__":
    main()
