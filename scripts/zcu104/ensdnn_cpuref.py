"""Chuan cong 2 (sua doi 2026-10-22): logit PyTorch float32 tren CPU, tung mang rieng, cung trong so va cung
dau vao da loc trong goi. Ghi THEM ref_logits_cpu.f32 (N,M,K) va ref_preds_cpu.i32 (N,M) vao <pkg>/T<T>/ (khong ghi de).
Dung: python ensdnn_cpuref.py <pkg dir>
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ssvep.models import GuneyDNN  # noqa: E402

NAMES = ["sub.weight", "ch.weight", "ch.bias", "down.weight", "down.bias", "conv.weight", "conv.bias", "fc.weight", "fc.bias"]
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 1)))
for T in (100, 250):
    d = os.path.join(sys.argv[1], f"T{T}")
    for f in ("ref_logits_cpu.f32", "ref_preds_cpu.i32"):
        if os.path.exists(os.path.join(d, f)):
            sys.exit(f"LOI: {d}/{f} da ton tai")
    m = json.load(open(os.path.join(d, "manifest.json")))
    N, C, K, M = m["N"], m["C"], m["K"], m["M"]
    W = np.fromfile(os.path.join(d, "weights.f32"), "<f4").reshape(N, -1)
    x = torch.from_numpy(np.fromfile(os.path.join(d, "ref_xfilt.f32"), "<f4").reshape(M, 3, C, T))
    net = GuneyDNN(C, T, K, 250.0).eval()
    shapes = [p.shape for p in (net.sub.weight, net.ch.weight, net.ch.bias, net.down.weight, net.down.bias, net.conv.weight,
                                net.conv.bias, net.fc.weight, net.fc.bias)]
    out = np.zeros((N, M, K), np.float32)
    for n in range(N):
        parts = np.split(W[n], np.cumsum([int(np.prod(s)) for s in shapes])[:-1])
        net.load_state_dict({k: torch.from_numpy(p.reshape(s).copy()) for k, p, s in zip(NAMES, parts, shapes)})
        with torch.no_grad():
            out[n] = net(x).numpy()
    out.astype("<f4").tofile(os.path.join(d, "ref_logits_cpu.f32"))
    out.argmax(-1).astype("<i4").tofile(os.path.join(d, "ref_preds_cpu.i32"))
    ref = np.fromfile(os.path.join(d, "ref_logits.f32"), "<f4").reshape(N, M, K)
    print(f"T{T}: chuan CPU ghi xong; |d| voi chuan luc xuat {np.abs(out - ref).max():.3g}", flush=True)
print("DONE")
