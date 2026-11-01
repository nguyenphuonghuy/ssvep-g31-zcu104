"""Bon ho kien truc WP2. Dau vao chung: bank loc 5 dai con, (B, 5, C, T).

H-A: dac trung tuong quan CCA (tinh truoc, cca_features) -> ham g dung chung cho moi muc tieu.
H-B: loc khong gian hoc duoc -> tich vo huong voi tham chieu sin/cos -> nang luong theo hai -> g.
H-C: nhu H-B, tham chieu = template nhom (TB nguoi train), tuy chon them sin/cos.
H-D: CNN thuan nhe (ReLU, BN, avg-pool), dense 40 lop.
H-A..H-C: g dung chung cho moi muc tieu k (khong co trong so rieng theo lop).
macs(): MACs moi quyet dinh, dem giai tich cho phan hoc duoc va phep tuong quan (BN gop duoc: 0;
phep chieu template co the tinh truoc nen khong tinh). H-D dung count_macs (Conv2d + Linear).
"""
import math
import os

import torch
import torch.nn as nn

from .models import count_macs, count_params

NB = 5          # so dai con cua bank loc (giong FBCCA)


def sincos_refs(freqs, fs, n_t, n_h):
    """(K, 2*n_h, n_t), thu tu [sin h1, cos h1, sin h2, ...], t = (1..n_t)/fs nhu cca.sine_refs."""
    t = torch.arange(1, n_t + 1, dtype=torch.float64) / fs
    out = []
    for f in freqs:
        r = []
        for h in range(1, n_h + 1):
            r += [torch.sin(2 * math.pi * h * f * t), torch.cos(2 * math.pi * h * f * t)]
        out.append(torch.stack(r))
    return torch.stack(out).float()


# ---------------------------------------------------------------- H-A: dac trung CCA
def _orth(x):
    """x (..., T, c) -> co so truc giao Q (..., T, c) sau khi tru trung binh theo T."""
    q, _ = torch.linalg.qr(x - x.mean(dim=-2, keepdim=True))
    return q


@torch.no_grad()
def cca_features(X, freqs, fs, chunk=512):
    """X (N, 5, C, T) -> dict:
    'band_nh3', 'band_nh5': rho^2 (N, K, 5) voi tham chieu Nh = 3 / 5 hai;
    'harm': rho^2 (N, K, 5, 5) theo dai con x hai (tham chieu chi gom sin/cos cua hai h)."""
    T = X.shape[-1]
    dev = X.device
    sets = {"band_nh3": sincos_refs(freqs, fs, T, 3), "band_nh5": sincos_refs(freqs, fs, T, 5)}
    r5 = sets["band_nh5"]
    for h in range(5):
        sets[f"h{h + 1}"] = r5[:, 2 * h:2 * h + 2]
    qy = {k: _orth(v.transpose(1, 2).double()).to(dev) for k, v in sets.items()}   # (K, T, r)
    out = {k: [] for k in sets}
    for i in range(0, len(X), chunk):
        qx = _orth(X[i:i + chunk].transpose(-1, -2).double())                       # (n, 5, T, C)
        for k, q in qy.items():
            m = torch.einsum("nbtc,ktr->nkbcr", qx, q)
            out[k].append(torch.linalg.svdvals(m)[..., 0].clamp(max=1.0).pow(2).float())
    res = {k: torch.cat(v) for k, v in out.items()}                                  # (N, K, 5)
    return {"band_nh3": res["band_nh3"], "band_nh5": res["band_nh5"],
            "harm": torch.stack([res[f"h{h + 1}"] for h in range(5)], dim=-1)}       # (N, K, 5, 5)


def ha_input(feats, cfg):
    """Chon dac trung theo cfg -> (N, K, d)."""
    if cfg["feat"] == "band":
        return feats[f"band_nh{cfg['nh']}"]
    return feats["harm"][..., :cfg["nh"]].flatten(-2)     # chi gop (dai con, hai), giu cac truc truoc


def cca_macs(T, C, K, nh, per_harm):
    """Uoc luong giai tich MACs cua phan CCA (chi de bao cao/rang buoc):
    moi dai con: QR cua X ~ 2*T*C^2; moi muc tieu, moi bo tham chieu r cot: chieu T*C*r + SVD ~ 4*C*r^2."""
    sets = [2 * nh] + ([2] * nh if per_harm else [])
    per_band = 2 * T * C * C + K * sum(T * C * r + 4 * C * r * r for r in sets)
    return NB * per_band


class Head(nn.Module):
    """g: (B, K, d) -> (B, K), dung chung cho moi muc tieu."""

    def __init__(self, d, hidden):
        super().__init__()
        self.d, self.hidden = d, hidden
        self.net = nn.Linear(d, 1) if hidden == 0 else \
            nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, f):
        return self.net(f).squeeze(-1)

    def macs(self, K):
        return K * (self.d if self.hidden == 0 else self.d * self.hidden + self.hidden)


class HA(nn.Module):
    input_kind = "feat"

    def __init__(self, cfg, C, T, K, fs, freqs):
        super().__init__()
        self.cfg, self.C, self.T, self.K = cfg, C, T, K
        d = NB if cfg["feat"] == "band" else NB * cfg["nh"]
        self.head = Head(d, cfg["hidden"])

    def forward(self, f):
        return self.head(f)

    def macs(self):
        return self.head.macs(self.K) + cca_macs(self.T, self.C, self.K, self.cfg["nh"],
                                                 self.cfg["feat"] == "harm")


# ---------------------------------------------------------------- H-B / H-C
class _Spatial(nn.Module):
    def __init__(self, C, F, shared):
        super().__init__()
        self.shared = shared
        shape = (1, F, C) if shared else (NB, F, C)
        self.W = nn.Parameter(torch.randn(*shape) / math.sqrt(C))

    def forward(self, x):                       # (B, 5, C, T) -> (B, 5, F, T)
        W = self.W.expand(NB, -1, -1)
        return torch.einsum("nfc,bnct->bnft", W, x)

    def project(self, tm):                      # template (5, K, C, T) -> (5, K, F, T)
        W = self.W.expand(NB, -1, -1)
        return torch.einsum("nfc,nkct->nkft", W, tm)


def _unit(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class HB(nn.Module):
    input_kind = "x"

    def __init__(self, cfg, C, T, K, fs, freqs):
        super().__init__()
        self.cfg, self.C, self.T, self.K = cfg, C, T, K
        self.F, self.nh, self.cos = cfg["F"], cfg["nh"], cfg["norm"] == "cosine"
        self.sp = _Spatial(C, self.F, cfg["shared"])
        self.register_buffer("refs", _unit(sincos_refs(freqs, fs, T, self.nh)))   # (K, 2nh, T)
        d = NB * self.F * self.nh
        self.bn = None if self.cos else nn.BatchNorm1d(d)
        self.head = Head(d, cfg["hidden"])

    def sincos_energy(self, z):                 # z (B, 5, F, T) -> (B, K, 5*F*nh)
        c = torch.einsum("bnft,kjt->bknfj", z, self.refs)
        e = c[..., 0::2] ** 2 + c[..., 1::2] ** 2
        return e.flatten(2)

    def _bn(self, f):
        if self.bn is None:
            return f
        B, K, d = f.shape
        return self.bn(f.reshape(B * K, d)).reshape(B, K, d)

    def forward(self, x):
        z = self.sp(x)
        if self.cos:
            z = _unit(z)
        return self.head(self._bn(self.sincos_energy(z)))

    def macs(self):
        nbf, T, K = NB * self.F, self.T, self.K
        m = nbf * self.C * T                                  # loc khong gian
        m += nbf * T if self.cos else 0                       # chuan hoa
        m += K * nbf * 2 * self.nh * T + K * nbf * self.nh * 2   # tuong quan + nang luong
        return m + self.head.macs(K)


class HC(HB):
    """Tham chieu = template nhom; set_templates() phai goi truoc khi huan luyen moi fold."""

    def __init__(self, cfg, C, T, K, fs, freqs):
        nn.Module.__init__(self)
        self.cfg, self.C, self.T, self.K = cfg, C, T, K
        self.F, self.nh, self.cos = cfg["F"], cfg["nh"], cfg["norm"] == "cosine"
        self.use_sc = cfg["refs"] == "template+sincos"
        self.sp = _Spatial(C, self.F, cfg["shared"])
        self.register_buffer("refs", _unit(sincos_refs(freqs, fs, T, self.nh)))
        self.register_buffer("tmpl", torch.zeros(NB, K, C, T))
        d = NB * self.F * (1 + (self.nh if self.use_sc else 0))
        self.bn = None if self.cos else nn.BatchNorm1d(d)
        self.head = Head(d, cfg["hidden"])

    def set_templates(self, tm):
        self.tmpl.copy_(tm)

    def forward(self, x):
        z = self.sp(x)                                            # (B, 5, F, T)
        p = self.sp.project(self.tmpl)                            # (5, K, F, T)
        if self.cos:
            z, p = _unit(z), _unit(p)
        f = torch.einsum("bnft,nkft->bknf", z, p).flatten(2)      # (B, K, 5F)
        if self.use_sc:
            f = torch.cat([f, self.sincos_energy(z)], dim=-1)
        return self.head(self._bn(f))

    def macs(self):
        nbf, T, K = NB * self.F, self.T, self.K
        m = nbf * self.C * T + (nbf * T if self.cos else 0)
        m += K * nbf * T                                          # tuong quan voi template (da chieu san)
        if self.use_sc:
            m += K * nbf * 2 * self.nh * T + K * nbf * self.nh * 2
        return m + self.head.macs(K)


# ---------------------------------------------------------------- H-D
class HD(nn.Module):
    input_kind = "x"

    def __init__(self, cfg, C, T, K, fs, freqs):
        super().__init__()
        self.cfg = cfg
        self.sub = nn.Conv2d(NB, 1, 1, bias=False)               # ket hop dai con (khoi tao bang 1)
        nn.init.ones_(self.sub.weight)
        S, Ct, k, p = cfg["S"], cfg["Ct"], cfg["k"], cfg["pool"]
        bn = lambda c: nn.BatchNorm2d(c, momentum=0.01, eps=1e-3)  # noqa: E731
        self.sp = nn.Sequential(nn.Conv2d(1, S, (C, 1), bias=False), bn(S))
        layers, ch, t = [], S, T
        for _ in range(cfg["nblk"]):
            if cfg["sep"]:
                layers += [nn.Conv2d(ch, ch, (1, k), padding=(0, k // 2), groups=ch, bias=False),
                           nn.Conv2d(ch, Ct, 1, bias=False)]
            else:
                layers += [nn.Conv2d(ch, Ct, (1, k), padding=(0, k // 2), bias=False)]
            layers += [bn(Ct), nn.ReLU(), nn.AvgPool2d((1, p)), nn.Dropout(cfg["drop"])]
            ch, t = Ct, t // p
        self.blocks = nn.Sequential(*layers)
        self.n_flat = ch * t
        self.fc = nn.Linear(max(self.n_flat, 1), K)
        self.T, self.C, self.NB = T, C, NB

    def forward(self, x):
        z = self.blocks(self.sp(self.sub(x).squeeze(1).unsqueeze(1)))
        return self.fc(z.flatten(1))

    def macs(self):
        return count_macs(self, torch.zeros(1, NB, self.C, self.T, device=self.fc.weight.device))


FAMILIES = {"HA": HA, "HB": HB, "HC": HC, "HD": HD}


def suggest(trial, family):
    """Khong gian thiet ke (–8.5). Moi bien deu duoc lay mau (khong co dieu kien) de
    NSGA-II/Grid xu ly on dinh; bien khong dung cho cau hinh do bi bo qua."""
    c = {"lr": trial.suggest_categorical("lr", [1e-3, 3e-3])}
    if family == "HA":
        c.update(feat=trial.suggest_categorical("feat", ["band", "harm"]),
                 nh=trial.suggest_categorical("nh", [3, 5]),
                 hidden=trial.suggest_categorical("hidden", [0, 4, 8, 16]))
    elif family in ("HB", "HC"):
        c.update(F=trial.suggest_categorical("F", [1, 2, 4]),
                 shared=trial.suggest_categorical("shared", [True, False]),
                 nh=trial.suggest_categorical("nh", [3, 5]),
                 norm=trial.suggest_categorical("norm", ["cosine", "bn"]),
                 hidden=trial.suggest_categorical("hidden", [0, 8, 16]))
        if family == "HC":
            c["refs"] = trial.suggest_categorical("refs", ["template", "template+sincos"])
    elif family == "HD":
        c.update(S=trial.suggest_categorical("S", [4, 8, 16]),
                 nblk=trial.suggest_categorical("nblk", [1, 2]),
                 Ct=trial.suggest_categorical("Ct", [8, 16, 32]),
                 k=trial.suggest_categorical("k", [9, 17, 33]),
                 sep=trial.suggest_categorical("sep", [True, False]),
                 pool=trial.suggest_categorical("pool", [2, 4]),
                 drop=trial.suggest_categorical("drop", [0.25, 0.5]))
    return c


GRID_HA = {"lr": [1e-3, 3e-3], "feat": ["band", "harm"], "nh": [3, 5], "hidden": [0, 4, 8, 16]}


def build(family, cfg, C, T, K, fs, freqs):
    m = FAMILIES[family](cfg, C, T, K, fs, freqs)
    return m, count_params(m), m.macs()


# ---------------------------------------------------------------- ung vien WP2 cho LOSO
SELECTION = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "wp2_search_2026_10_08", "selection.json")
CANDIDATES = {"wp2_hybrid": "hybrid", "wp2_pure": "pure_dl"}


def load_candidate(name):
    import json
    sel = json.load(open(SELECTION))[CANDIDATES[name]]
    return sel["family"], sel["cfg"], sel["trial"]


def make_candidate(name):
    """Lop mo hinh co giao dien cua train_loso (Model(n_ch, n_t, n_cls, fs, freqs=...))."""
    fam, cfg, trial = load_candidate(name)
    if fam == "HA":
        raise ValueError("H-A chi lam tham chieu, khong chay LOSO")
    base = FAMILIES[fam]

    class Candidate(base):
        prep, n_band, needs_freqs = "fb5", NB, True
        family, config, search_trial = fam, cfg, trial

        def __init__(self, n_ch, n_t, n_cls, fs, freqs=None):
            super().__init__(cfg, n_ch, n_t, n_cls, fs, freqs)

        def l2_penalty(self):
            return 0.0

        def apply_constraints(self):
            pass

    Candidate.__name__ = f"{name}_{fam}{trial}"
    return Candidate


def train_hp(name):
    """Sieu tham so huan luyen giong vong tim kiem; lr theo cau hinh duoc chon."""
    _, cfg, _ = load_candidate(name)
    return {"lr": cfg["lr"], "batch": 256, "max_epochs": 200, "patience": 30}
