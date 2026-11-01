"""Luong tu hoa sau huan luyen (PTQ) mo phong bang fake-quant, dac ta:

- Trong so moi Conv2d/Linear: INT8 doi xung theo tung kenh ra, thang = max|w| / 127.
- Dau vao moi Conv2d/Linear (ke ca dau vao mang): INT8 doi xung theo ca tensor,
  thang = max|x| / 127 do tren tap hieu chinh (lay tu nguoi TRAIN, khong dung nguoi test).
- Bias, BN, ham kich hoat, pooling giu float32 (chua gop BN vao conv).
- pot=True: lam tron thang len luy thua cua 2 (gan voi so diem co dinh ap_fixed cua hls4ml).
"""
import copy

import torch
import torch.nn as nn

QMAX = 127


def _scale(amax, pot):
    s = amax.clamp(min=1e-12) / QMAX
    return torch.exp2(torch.ceil(torch.log2(s))) if pot else s


def _fq(x, s):
    return torch.clamp(torch.round(x / s), -QMAX, QMAX) * s


@torch.no_grad()
def quantize_ptq(model, xcal, pot=False, batch=2048):
    """Tra ve ban sao da luong tu hoa cua model (eval mode); model goc khong doi."""
    q = copy.deepcopy(model).eval()
    layers = [m for m in q.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    for m in layers:
        w = m.weight
        amax = w.abs().flatten(1).max(1).values.view(-1, *([1] * (w.dim() - 1)))
        m.weight.copy_(_fq(w, _scale(amax, pot)))
    amax = {}

    def cal(m, inp):
        v = inp[0].abs().max()
        amax[m] = torch.maximum(amax[m], v) if m in amax else v

    hs = [m.register_forward_pre_hook(cal) for m in layers]
    for i in range(0, len(xcal), batch):
        q(xcal[i:i + batch])
    for h in hs:
        h.remove()
    for m in layers:
        s = _scale(amax[m], pot)
        m.register_forward_pre_hook(lambda mod, inp, s=s: (_fq(inp[0], s),))
    return q
