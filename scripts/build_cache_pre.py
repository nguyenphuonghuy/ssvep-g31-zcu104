"""WP3: tao cache co doan truoc kich thich (data/cache/<ds>_pre/S<n>.npy + meta.json) cho loc nhan qua.
Tu kiem: phan sau kich thich trung khop tung mau voi cache chuan (data/cache/<ds>/S<n>.npy).
Dung: python build_cache_pre.py benchmark   (bo qua doi tuong da co; --force de ghi lai)"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ssvep.data import CACHE_ROOT, cache_path_pre, load_cfg, load_subject, read_subject_raw  # noqa: E402


def main():
    name = sys.argv[1]
    force = "--force" in sys.argv
    d = os.path.join(CACHE_ROOT, f"{name}_pre")
    os.makedirs(d, exist_ok=True)
    meta0 = None
    for s in range(1, load_cfg(name)["n_subjects"] + 1):
        p = cache_path_pre(name, s)
        if os.path.exists(p) and not force:
            continue
        X, meta = read_subject_raw(name, s, keep_pre=True)
        ref, _ = load_subject(name, s)
        on = meta["onset_in_array"]
        if not np.array_equal(X[..., on:on + ref.shape[-1]], ref):
            sys.exit(f"LOI S{s}: doan sau kich thich khong khop cache chuan")
        np.save(p, X)
        meta0 = meta
        print(f"S{s}: {X.shape} onset_in_array={on} khop cache chuan", flush=True)
    if meta0 is not None:
        meta0.pop("subject", None)
        json.dump(meta0, open(os.path.join(d, "meta.json"), "w"), indent=1)
    print("DONE", name)


if __name__ == "__main__":
    main()
