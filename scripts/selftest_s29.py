""" tu kiem (i)-(iii): khoi tao xac lap, luong mot luot, hinh dang. Chi numpy/scipy. exit 3 neu truot."""
import sys
import numpy as np
from ssvep import cca
from ssvep.prep import _lfilter_ss, _stream_bands, prepare_causal

ok = True
for fs in (250, 256):
    x = np.full((2, 3, 400), 100.0)
    m = max(np.abs(_lfilter_ss(b, a, x)).max() for b, a in cca.subband_filters(fs, 5))
    print(f"(i) fs={fs} dau vao hang 100 uV: max|y| = {m:.2e}"); ok &= m < 1e-6
rng = np.random.default_rng(0)
X = rng.normal(size=(1, 1, 3, 600)) * 10 + 5000
a1 = _stream_bands(X, 250, 160, 100)
a2 = np.stack([_lfilter_ss(b, a, X)[..., 160:260] for b, a in cca.subband_filters(250, 5)], axis=-3)
d = np.abs(a1 - a2).max(); print(f"(ii) luong 1 luot vs fb5cz: lech max {d:.2e}"); ok &= d < 1e-9
for ds in sys.argv[1:] or ["nakanishi"]:
    for w in (0.4, 1.0):
        sh = {p: prepare_causal(ds, w, p)[0].shape for p in ("fb5c", "fb5cz", "fb5cs")}
        print(f"(iii) {ds} w={w}: {sh}"); ok &= len(set(sh.values())) == 1
print("DAT" if ok else "KHONG DAT"); sys.exit(0 if ok else 3)
