"""Mo hinh doi chuan hoc sau (cai lai bang PyTorch theo ma goc) va dem chi phi.

Nguon ma goc (clone trong external/, commit ghi trong AI_CONTEXT.md muc 6):
  EEGNet, Compact-CNN (EEGNet_SSVEP): vlawhern/arl-eegmodels, EEGModels.py
  FB-tCNN: DingWenl/FB-tCNN, FB-tCNN/net.py
  DNN (Guney et al. 2022): osmanberke/Deep-SSVEP-BCI, main.m (chi giai doan 1 — mo hinh chung)
Dau vao moi mo hinh: (B, n_band, C, T). Tien xu ly (bank loc) xem ssvep/prep.py.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNetBase(nn.Module):
    """Khung EEGNet cua arl-eegmodels; EEGNet va EEGNet_SSVEP chi khac sieu tham so."""

    def __init__(self, n_ch, n_t, n_cls, F1, D, F2, kern, drop, dense_norm):
        super().__init__()
        self.conv1 = nn.Conv2d(1, F1, (1, kern), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(F1, momentum=0.01, eps=1e-3)
        self.dw = nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D, momentum=0.01, eps=1e-3)
        self.pool1, self.drop1 = nn.AvgPool2d((1, 4)), nn.Dropout(drop)
        self.sep_dw = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding="same", groups=F1 * D,
                                bias=False)
        self.sep_pw = nn.Conv2d(F1 * D, F2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(F2, momentum=0.01, eps=1e-3)
        self.pool2, self.drop2 = nn.AvgPool2d((1, 8)), nn.Dropout(drop)
        n_flat = F2 * ((n_t // 4) // 8)
        self.fc = nn.Linear(n_flat, n_cls)
        self.dense_norm = dense_norm

    def forward(self, x):
        x = self.bn2(self.dw(self.bn1(self.conv1(x))))
        x = self.drop1(self.pool1(F.elu(x)))
        x = self.bn3(self.sep_pw(self.sep_dw(x)))
        x = self.drop2(self.pool2(F.elu(x)))
        return self.fc(x.flatten(1))

    @torch.no_grad()
    def apply_constraints(self):
        # Keras max_norm(1.) tren depthwise kernel: chuan theo truc kenh cho tung bo loc
        w = self.dw.weight  # (F1*D, 1, C, 1)
        self.dw.weight.copy_(torch.renorm(w, 2, 0, 1.0))
        if self.dense_norm is not None:
            # Keras max_norm tren Dense kernel (in, out), axis=0 -> tung don vi dau ra
            self.fc.weight.copy_(torch.renorm(self.fc.weight, 2, 0, self.dense_norm))

    def l2_penalty(self):
        return 0.0


class EEGNet(EEGNetBase):
    """Lawhern et al. 2018. kernLength = fs/2 (64 o 128 Hz), dropout 0.25 (xuyen doi tuong)."""
    prep, n_band = "bp6_80", 1

    def __init__(self, n_ch, n_t, n_cls, fs):
        super().__init__(n_ch, n_t, n_cls, F1=8, D=2, F2=16, kern=fs // 2, drop=0.25,
                         dense_norm=0.25)


class CompactCNN(EEGNetBase):
    """Waytowich et al. 2018 (EEGNet_SSVEP): F1=96, D=1, F2=96, kernLength = fs (256 o 256 Hz)."""
    prep, n_band = "bp6_80", 1

    def __init__(self, n_ch, n_t, n_cls, fs):
        super().__init__(n_ch, n_t, n_cls, F1=96, D=1, F2=96, kern=fs, drop=0.5,
                         dense_norm=None)


class FBtCNN(nn.Module):
    """Ding et al. 2021. 4 dai con; conv1–3 dung chung trong so giua cac nhanh, BN rieng tung nhanh."""
    prep, n_band = "fbtcnn4", 4

    def __init__(self, n_ch, n_t, n_cls, fs, oc=16, drop=0.4, l2=0.01):
        super().__init__()
        self.n_t, self.l2 = n_t, l2
        self.conv1 = nn.Conv2d(1, oc, (n_ch, 1))
        self.conv2 = nn.Conv2d(oc, oc, (1, n_t), stride=(1, 5))   # padding 'same' kieu Keras
        self.conv3 = nn.Conv2d(oc, oc, (1, 5))
        bn = lambda c: nn.BatchNorm2d(c, momentum=0.01, eps=1e-3)  # noqa: E731
        self.bns = nn.ModuleList([nn.ModuleList([bn(oc), bn(oc), bn(oc)]) for _ in range(4)])
        self.drop = nn.Dropout(drop)
        t2 = math.ceil(n_t / 5)
        self.pad2 = max((t2 - 1) * 5 + n_t - n_t, 0)
        t3 = t2 - 4
        self.conv4 = nn.Conv2d(oc, 32, (1, t3))
        self.bn4 = bn(32)
        self.fc = nn.Linear(32, n_cls)

    def branch(self, x, bns):
        x = F.elu(bns[0](self.conv1(x)))
        x = self.drop(x)
        p = self.pad2
        x = F.pad(x, (p // 2, p - p // 2))  # TF 'same': phan du dat ben phai
        x = F.elu(bns[1](self.conv2(x)))
        x = self.drop(x)
        return F.elu(bns[2](self.conv3(x)))

    def forward(self, x):
        z = sum(self.branch(x[:, b:b + 1], self.bns[b]) for b in range(4))
        z = F.elu(self.bn4(self.conv4(self.drop(z))))
        return self.fc(self.drop(z.flatten(1)))

    def apply_constraints(self):
        pass

    def l2_penalty(self):
        # Keras regularizers.l2(0.01) tren kernel cua moi Conv2D (khong ap cho Dense)
        return self.l2 * sum((m.weight ** 2).sum() for m in
                             (self.conv1, self.conv2, self.conv3, self.conv4))


class GuneyDNN(nn.Module):
    """Guney et al. 2022, giai doan 1 (mo hinh chung). 3 dai con, ReLU duy nhat sau lop 3."""
    prep, n_band = "guney3", 3

    def __init__(self, n_ch, n_t, n_cls, fs, l2=0.001):
        super().__init__()
        self.l2 = l2
        self.sub = nn.Conv2d(3, 1, 1, bias=False)          # ket hop dai con, khoi tao bang 1
        self.ch = nn.Conv2d(1, 120, (n_ch, 1))
        self.down = nn.Conv2d(120, 120, (1, 2), stride=(1, 2))
        self.conv = nn.Conv2d(120, 120, (1, 10), padding="same")
        self.d1, self.d2, self.d3 = nn.Dropout(0.1), nn.Dropout(0.1), nn.Dropout(0.95)
        self.fc = nn.Linear(120 * (n_t // 2), n_cls)
        nn.init.ones_(self.sub.weight)
        for m in (self.ch, self.down, self.conv, self.fc):     # MATLAB 'narrow-normal'
            nn.init.normal_(m.weight, 0.0, 0.01)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.d1(self.ch(self.sub(x)))
        x = F.relu(self.d2(self.down(x)))
        x = self.d3(self.conv(x))
        return self.fc(x.flatten(1))

    def apply_constraints(self):
        pass

    def l2_penalty(self):
        # MATLAB L2Regularization: (lambda/2) * tong w^2 tren trong so (bias khong tinh)
        return 0.5 * self.l2 * sum((m.weight ** 2).sum() for m in
                                   (self.sub, self.ch, self.down, self.conv, self.fc))


MODELS = {"eegnet": EEGNet, "compactcnn": CompactCNN, "fbtcnn": FBtCNN, "guney": GuneyDNN}

# Sieu tham so huan luyen. lr/batch/l2 theo ma goc; max_epochs/patience la quy uoc chung cua du an
# (xem ), KHONG phai cua bai goc.
TRAIN = {
    "eegnet":     {"lr": 1e-3, "batch": 64,  "max_epochs": 500,  "patience": 50},
    "compactcnn": {"lr": 1e-3, "batch": 64,  "max_epochs": 500,  "patience": 50},
    "fbtcnn":     {"lr": 1e-3, "batch": 256, "max_epochs": 150,  "patience": 30},
    "guney":      {"lr": 1e-4, "batch": 100, "max_epochs": 1000, "patience": 100},
}


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_macs(model, x):
    """MACs cua Conv2d va Linear cho mot mau (BN/activation/pool khong tinh)."""
    macs = [0]

    def conv_hook(m, i, o):
        k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        macs[0] += o[0].numel() * k

    def lin_hook(m, i, o):
        macs[0] += m.in_features * m.out_features

    hs = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hs.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hs.append(m.register_forward_hook(lin_hook))
    model.eval()
    with torch.no_grad():
        model(x[:1])
    for h in hs:
        h.remove()
    return macs[0]
