"""Doc du lieu SSVEP chuan va cache thanh mang float32 thong nhat.

Dinh dang cache cho moi doi tuong: X float32 (n_blocks, n_targets, n_ch, n_t),
mau 0 = thoi diem bat dau kich thich (CHUA cong do tre thi giac).
Nhan cua X[b, k] la muc tieu k.
"""
import json
import os

import numpy as np

PROJ = os.environ.get("SSVEP_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CFG_PATH = os.path.join(PROJ, "configs", "datasets.json")
CACHE_ROOT = os.path.join(PROJ, "data", "cache")


def load_cfg(name):
    with open(CFG_PATH) as f:
        cfg = json.load(f)[name]
    cfg["root"] = os.path.expandvars(cfg["root"].replace("${SSVEP_DATA}", os.environ.get("SSVEP_DATA", os.path.join(PROJ, "datasets"))))
    return cfg


def _norm(s):
    return str(s).strip().upper()


def _pick(names, wanted):
    """Chi so cac kenh `wanted` trong danh sach `names`, doi chieu theo TEN."""
    up = [_norm(n) for n in names]
    idx = []
    for w in wanted:
        if _norm(w) not in up:
            raise KeyError(f"khong tim thay kenh {w} trong {names}")
        idx.append(up.index(_norm(w)))
    return idx


def benchmark_channel_names(root):
    names = []
    with open(os.path.join(root, "64-channels.loc")) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                names.append(parts[3])
    assert len(names) == 64, len(names)
    return names


def read_subject_raw(name, subj, keep_pre=False):
    """Doc 1 doi tuong tu file .mat goc. Tra ve (X, meta) theo dinh dang cache.
    keep_pre=True (WP3): giu ca doan truoc kich thich, meta['onset_in_array'] = chi so mau kich thich."""
    from scipy.io import loadmat
    cfg = load_cfg(name)
    root, on = cfg["root"], (0 if keep_pre else cfg["onset_sample"])
    if name == "benchmark":
        d = loadmat(os.path.join(root, f"S{subj}.mat"))["data"]          # ch,t,target,block
        fp = loadmat(os.path.join(root, "Freq_Phase.mat"))
        freqs = np.asarray(fp["freqs"]).ravel()
        phases = np.asarray(fp["phases"]).ravel()
        chn = benchmark_channel_names(root)
        idx = _pick(chn, cfg["channels"])
        X = d[idx, on:, :, :].transpose(3, 2, 0, 1)                       # block,target,ch,t
        chans = [chn[i] for i in idx]
    elif name == "beta":
        m = loadmat(os.path.join(root, f"S{subj}.mat"), squeeze_me=True,
                    struct_as_record=False)["data"]
        si = m.suppl_info
        chn = [str(c) for c in np.asarray(si.chan)[:, 3]]
        idx = _pick(chn, cfg["channels"])
        X = m.EEG[idx, on:, :, :].transpose(2, 3, 0, 1)                   # ch,t,block,target
        freqs = np.asarray(si.freqs).ravel()
        phases = np.asarray(si.phases).ravel()
        assert int(si.srate) == cfg["srate"], si.srate
        chans = [chn[i] for i in idx]
    elif name == "nakanishi":
        d = loadmat(os.path.join(root, f"s{subj}.mat"))["eeg"]            # target,ch,t,block
        X = d[:, :, on:, :].transpose(3, 0, 1, 2)
        freqs = np.asarray(cfg["freqs"])
        phases = np.full(len(freqs), np.nan)
        chans = list(cfg["channels_in_file"])
    elif name in ("zhu_dry", "zhu_wet"):                               # Zhu 2021, ch,t,electrode,block,target
        d = loadmat(os.path.join(root, f"S{subj:03d}.mat"))["data"]
        assert d.shape == (8, 710, 2, 10, 12), d.shape
        X = d[:, on:, cfg["electrode_index"], :, :].transpose(2, 3, 0, 1)  # block,target,ch,t
        freqs = np.asarray(cfg["freqs"])
        phases = np.asarray(cfg["phases_pi"]) * np.pi
        chans = list(cfg["channels_in_file"])
    else:
        raise ValueError(name)
    meta = {"dataset": name, "subject": subj, "srate": cfg["srate"],
            "latency_s": cfg["latency_s"], "channels": chans,
            "freqs": [float(x) for x in freqs], "phases": [float(x) for x in phases],
            "shape": list(X.shape)}
    if keep_pre:
        meta["onset_in_array"] = cfg["onset_sample"]
    return np.ascontiguousarray(X, dtype=np.float32), meta


def cache_path(name, subj):
    return os.path.join(CACHE_ROOT, name, f"S{subj}.npy")


def load_subject(name, subj):
    X = np.load(cache_path(name, subj))
    with open(os.path.join(CACHE_ROOT, name, "meta.json")) as f:
        meta = json.load(f)
    return X, meta


def window(X, meta, win_s):
    """Cat cua so [latency, latency + win_s) tinh tu luc bat dau kich thich."""
    fs = meta["srate"]
    s0 = int(round(meta["latency_s"] * fs))
    n = int(round(win_s * fs))
    if s0 + n > X.shape[-1]:
        raise ValueError(f"cua so {win_s}s vuot do dai trial {X.shape[-1]} mau")
    return X[..., s0:s0 + n]


# ---------------------------------------------------------------- WP3: du lieu co doan truoc kich thich
def cache_path_pre(name, subj):
    return os.path.join(CACHE_ROOT, f"{name}_pre", f"S{subj}.npy")


def load_subject_pre(name, subj):
    X = np.load(cache_path_pre(name, subj))
    with open(os.path.join(CACHE_ROOT, f"{name}_pre", "meta.json")) as f:
        meta = json.load(f)
    return X, meta
