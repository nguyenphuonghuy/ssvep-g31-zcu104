"""Chan doan cong 2 muc 23 (khong doi nguong/ket qua): logit PyTorch float32 tren CPU (khong TF32) va GPU tat/bat TF32,
dung CHINH trong so + du lieu da loc da xuat (job 17473), so voi chuan GPU luc xuat va voi ban NumPy (ensdnn_np.py).
Dung: python ensdnn_diag_tf32.py <pkg dir>
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ssvep.models import GuneyDNN  # noqa: E402
import ensdnn_np  # noqa: E402

NAMES = ["sub.weight", "ch.weight", "ch.bias", "down.weight", "down.bias", "conv.weight", "conv.bias", "fc.weight", "fc.bias"]
print("cudnn.allow_tf32 mac dinh =", torch.backends.cudnn.allow_tf32, "| matmul.allow_tf32 =", torch.backends.cuda.matmul.allow_tf32)
for T in (100, 250):
    d = os.path.join(sys.argv[1], f"T{T}")
    m = json.load(open(os.path.join(d, "manifest.json")))
    N, C, K, M = m["N"], m["C"], m["K"], m["M"]
    W = np.fromfile(os.path.join(d, "weights.f32"), "<f4").reshape(N, -1)
    x = torch.from_numpy(np.fromfile(os.path.join(d, "ref_xfilt.f32"), "<f4").reshape(M, 3, C, T))
    ref = np.fromfile(os.path.join(d, "ref_logits.f32"), "<f4").reshape(N, M, K)
    net = GuneyDNN(C, T, K, 250.0)
    shapes = [p.shape for p in (net.sub.weight, net.ch.weight, net.ch.bias, net.down.weight, net.down.bias, net.conv.weight,
                                net.conv.bias, net.fc.weight, net.fc.bias)]
    res = {}
    for dev, tf32 in (("cpu", False), ("cuda", False), ("cuda", True)):
        if dev == "cuda" and not torch.cuda.is_available():
            continue
        torch.backends.cudnn.allow_tf32 = tf32
        torch.backends.cuda.matmul.allow_tf32 = tf32
        out = np.zeros((N, M, K), np.float32)
        for n in range(N):
            parts = np.split(W[n], np.cumsum([int(np.prod(s)) for s in shapes])[:-1])
            sd = {k: torch.from_numpy(p.reshape(s).copy()) for k, p, s in zip(NAMES, parts, shapes)}
            net.load_state_dict(sd)
            net.to(dev).eval()
            with torch.no_grad():
                out[n] = net(x.to(dev)).cpu().numpy()
        res[f"{dev}_tf32={tf32}"] = out
    P = ensdnn_np.load(d)
    npl = np.stack([ensdnn_np.decide(P, np.fromfile(os.path.join(d, "raw.f64"), "<f8").reshape(M, C, T)[i])[3] for i in range(M)], 1)
    for k, v in res.items():
        print(f"T{T} {k}: |d| voi chuan xuat {np.abs(v - ref).max():.3g}; voi NumPy {np.abs(v - npl).max():.3g}")
    print(f"T{T} NumPy vs chuan xuat {np.abs(npl - ref).max():.3g}; |logit| lon nhat {np.abs(ref).max():.3g}")
print("DONE")
