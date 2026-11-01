"""WP3: mo hinh lai tong quat + QAT, cho ablation va luong tu hoa.

HybridQ tai hien H-C #21 (wp2_models.HC: loc khong gian dung chung -> chuan hoa cosine -> tuong quan voi
template nhom va nang luong sin/cos -> ham tinh diem tuyen tinh dung chung cho moi muc tieu), co the:
- bat/tat template (use_tmpl), sin/cos (use_sc), doi so dai con nb (5 hoac 1);
- luong tu hoa gia lap khi huan luyen (QAT, bits = 8/6/4), STE:
  * trong so: doi xung, thang = max|w| / qmax (W theo tung bo loc ra; trong so ham diem theo ca tensor);
  * gia tri trung gian: doi xung theo ca tensor, thang = EMA cua max|x| khi huan luyen (dong bang khi eval);
  * diem luong tu: dau vao x, z sau loc khong gian, z va template sau chuan hoa, tham chieu sin/cos,
    vector dac trung truoc ham diem; bias ham diem giu do chinh xac cao (int32 tren phan cung).
  Chuan hoa cosine (can bac hai, phep chia) giu FP32 trong mo phong; danh gia phan cung o G3.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wp2_models import _unit, sincos_refs


class FQAct(nn.Module):
    """Fake-quant gia tri trung gian: doi xung, theo ca tensor, thang tu EMA cua max|x|.
    pct (hau kiem): thang tu EMA cua phan vi pct (%) cua |x| theo batch thay cho max; None = hanh vi cu."""

    def __init__(self, bits, momentum=0.01, pct=None):
        super().__init__()
        self.qmax = 2 ** (bits - 1) - 1
        self.momentum = momentum
        self.pct = pct
        self.register_buffer("amax", torch.zeros(()))
        self.register_buffer("inited", torch.zeros((), dtype=torch.bool))

    def forward(self, x):
        if self.training:
            ax = x.detach().abs()
            if self.pct is None:
                m = ax.max()
            else:
                v = ax.flatten()
                k = max(1, int(round(v.numel() * self.pct / 100.0)))
                m = v.kthvalue(k).values                      # torch.quantile gioi han so phan tu
            if not bool(self.inited):
                self.amax.copy_(m)
                self.inited.fill_(True)
            else:
                self.amax.mul_(1 - self.momentum).add_(self.momentum * m)
        s = self.amax.clamp(min=1e-8) / self.qmax
        q = torch.clamp(torch.round(x / s), -self.qmax, self.qmax) * s
        return x + (q - x).detach()


def fq_weight(w, bits, dim=None):
    """Fake-quant trong so (STE). dim: truc kenh ra (theo tung kenh); None: ca tensor."""
    qmax = 2 ** (bits - 1) - 1
    if dim is None:
        amax = w.detach().abs().max()
    else:
        red = [d for d in range(w.dim()) if d != dim]
        amax = w.detach().abs().amax(dim=red, keepdim=True)
    s = amax.clamp(min=1e-8) / qmax
    q = torch.clamp(torch.round(w / s), -qmax, qmax) * s
    return w + (q - w).detach()


class HybridQ(nn.Module):
    input_kind = "x"

    def __init__(self, cfg, C, T, K, fs, freqs):
        super().__init__()
        assert cfg["norm"] == "cosine" and cfg["hidden"] == 0, "WP3 chi ho tro cau hinh cua H-C #21"
        self.cfg, self.C, self.T, self.K = cfg, C, T, K
        self.F, self.nh, self.nb = cfg["F"], cfg["nh"], cfg.get("nb", 5)
        self.use_tmpl, self.use_sc = cfg.get("use_tmpl", True), cfg.get("use_sc", True)
        assert self.use_tmpl or self.use_sc
        self.bits = cfg.get("bits")
        shape = (1, self.F, C) if cfg["shared"] else (self.nb, self.F, C)
        self.W = nn.Parameter(torch.randn(*shape) / math.sqrt(C))
        self.register_buffer("refs", _unit(sincos_refs(freqs, fs, T, self.nh)))      # (K, 2nh, T)
        if self.use_tmpl:
            self.register_buffer("tmpl", torch.zeros(self.nb, K, C, T))
        d = self.nb * self.F * ((1 if self.use_tmpl else 0) + (self.nh if self.use_sc else 0))
        self.head = nn.Linear(d, 1)
        if self.bits:
            for n in ("x", "z", "zn", "p", "r", "f"):
                setattr(self, f"q_{n}", FQAct(self.bits, pct=cfg.get("act_pct")))

    def set_templates(self, tm):
        self.tmpl.copy_(tm)

    def _q(self, name, x):
        return getattr(self, f"q_{name}")(x) if self.bits else x

    def _Wfull(self):
        W = fq_weight(self.W, self.bits, dim=1) if self.bits else self.W
        return W.expand(self.nb, -1, -1)

    def forward(self, x):
        W = self._Wfull()
        z = torch.einsum("nfc,bnct->bnft", W, self._q("x", x))
        z = self._q("zn", _unit(self._q("z", z)))
        feats = []
        if self.use_tmpl:
            p = self._q("p", _unit(torch.einsum("nfc,nkct->nkft", W, self.tmpl)))
            feats.append(torch.einsum("bnft,nkft->bknf", z, p).flatten(2))
        if self.use_sc:
            c = torch.einsum("bnft,kjt->bknfj", z, self._q("r", self.refs))
            feats.append((c[..., 0::2] ** 2 + c[..., 1::2] ** 2).flatten(2))
        f = self._q("f", torch.cat(feats, dim=-1))
        w = fq_weight(self.head.weight, self.bits) if self.bits else self.head.weight
        return F.linear(f, w, self.head.bias).squeeze(-1)

    def macs(self):
        nbf, T, K = self.nb * self.F, self.T, self.K
        m = nbf * self.C * T + nbf * T                          # loc khong gian + chuan hoa
        d = 0
        if self.use_tmpl:
            m += K * nbf * T                                      # tuong quan template (da chieu san)
            d += nbf
        if self.use_sc:
            m += K * nbf * 2 * self.nh * T + K * nbf * self.nh * 2
            d += nbf * self.nh
        return m + K * d


# ten bien the -> (thay doi so voi cau hinh H-C #21, prep)
VARIANTS = {
    "wp3_qat8": ({"bits": 8}, "fb5"),
    "wp3_qat6": ({"bits": 6}, "fb5"),
    "wp3_qat4": ({"bits": 4}, "fb5"),
    "wp3_causal": ({}, "fb5c"),
    "wp3_abl_tmpl": ({"use_sc": False}, "fb5"),          # chi template
    "wp3_abl_sc": ({"use_tmpl": False}, "fb5"),          # chi sin/cos
    "wp3_abl_nofb": ({"nb": 1}, "fb1"),                  # bo bank loc: 1 dai 6-90 Hz (dai con 1)
    "wp3_g3": ({"bits": 8}, "fb5c"),                     # cau hinh dong bang G3
    "wp3_g3_ss": ({"bits": 8}, "fb5cz"),                 # G3, khoi tao trang thai loc xac lap
    "wp3_g3_st": ({"bits": 8}, "fb5cs"),                 # G3, luong noi mang trang thai
    "wp3_g3_dc": ({"bits": 8}, "fb5cd"),                 # G3 + tru TB doan truoc kich thich truoc loc
    "wp3_g3_dc_fp": ({"bits": None}, "fb5cd"),           #  D2 (hau kiem): G3.1 khong luong tu
    "wp3_g3_dc_p999": ({"bits": 8, "act_pct": 99.9}, "fb5cd"),   #  D3 (hau kiem): thang = phan vi 99,9%
    "wp3_g3_dc_fp_med": ({"bits": None, "tmpl_stat": "median"}, "fb5cd"),   #  D4b (hau kiem): template trung vi
}


def make_variant(name):
    from .wp2_models import load_candidate, train_hp
    fam, base_cfg, trial = load_candidate("wp2_hybrid")
    assert fam == "HC" and trial == 21, (fam, trial)
    delta, prep = VARIANTS[name]
    cfg = dict(base_cfg, **delta)

    class Variant(HybridQ):
        needs_freqs = True
        n_band = cfg.get("nb", 5)
        config, family, search_trial = cfg, "HC", trial

        def __init__(self, n_ch, n_t, n_cls, fs, freqs=None):
            super().__init__(cfg, n_ch, n_t, n_cls, fs, freqs)

        def l2_penalty(self):
            return 0.0

        def apply_constraints(self):
            pass

    Variant.prep = prep
    Variant.tmpl_stat = cfg.get("tmpl_stat", "mean")
    Variant.__name__ = name
    return Variant, train_hp("wp2_hybrid")
