"""
iq_utils.py
Dataset classes, augmentations, and model definitions for the SimCLR + open-set
pipeline. Pulled into a real module (not left in notebook cells) specifically so
DataLoader(num_workers>0) works on Windows -- Windows spawns worker processes and
each one needs to `import` this stuff cleanly, which it can't do from notebook
cell state.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ---------------- augmentations ----------------

def time_shift(x, max_shift=64):
    shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
    return torch.roll(x, shifts=shift, dims=-1)

def phase_rotation(x):
    theta = torch.empty(1).uniform_(0, 2 * np.pi).item()
    i, q = x[0], x[1]
    i_rot = i * np.cos(theta) - q * np.sin(theta)
    q_rot = i * np.sin(theta) + q * np.cos(theta)
    return torch.stack([i_rot, q_rot], dim=0)

def noise_injection(x, snr_db_range=(5, 25)):
    snr_db = torch.empty(1).uniform_(*snr_db_range).item()
    sig_power = (x ** 2).mean()
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(x) * torch.sqrt(noise_power)
    return x + noise

def channel_distortion(x, gain_range=(0.7, 1.3)):
    gain = torch.empty(1).uniform_(*gain_range).item()
    x = x * gain
    alpha = torch.empty(1).uniform_(0.0, 0.3).item()
    x_delayed = torch.roll(x, shifts=1, dims=-1)
    return x + alpha * x_delayed

def augment(x):
    ops = [time_shift, phase_rotation, noise_injection, channel_distortion]
    np.random.shuffle(ops)
    n_ops = np.random.randint(1, len(ops) + 1)
    for op in ops[:n_ops]:
        x = op(x)
    return x


# ---------------- datasets ----------------

class ContrastiveIQDataset(Dataset):
    """Returns two augmented views of the same signal. No labels used."""
    def __init__(self, X):
        self.X = torch.from_numpy(X).float().permute(0, 2, 1)  # [N, 2, 1024]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        return augment(x.clone()), augment(x.clone())


class LabeledIQDataset(Dataset):
    """Plain (signal, label, snr) dataset for probing / eval -- no augmentation."""
    def __init__(self, X, y, snr):
        self.X = torch.from_numpy(X).float().permute(0, 2, 1)
        self.y = torch.from_numpy(y).long()
        self.snr = torch.from_numpy(snr).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.snr[idx]


# ---------------- model ----------------

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x).unsqueeze(-1)
        return x * w


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.se    = SEBlock(out_ch) if use_se else nn.Identity()

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + residual
        return F.relu(out)


class IQEncoder(nn.Module):
    """
    Residual + SE encoder. embed_dim kept at 128 by default to stay
    compatible with the rest of the notebook pipeline (probe, Mahalanobis,
    MSP cells all assume 128-dim embeddings unless you update them too).
    """
    def __init__(self, embed_dim=128):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = nn.Sequential(                 # 64 -> 64
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
        )
        self.layer2 = nn.Sequential(                 # 64 -> 128, downsample
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
        )
        self.layer3 = nn.Sequential(                 # 128 -> 256, downsample
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
        )
        self.layer4 = nn.Sequential(                 # 256 -> 512, downsample
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(512, embed_dim)
        )

        self.embed_dim = embed_dim

    def forward(self, x, return_features=False):
        x = self.stem(x)
        x = self.layer1(x)
        f1 = x
        x = self.layer2(x)
        f2 = x
        x = self.layer3(x)
        f3 = x
        x = self.layer4(x)
        f4 = x

        h = self.pool(x).squeeze(-1)
        emb = self.fc(h)
        emb = F.normalize(emb, dim=-1)          # important for open-set

        if return_features:
            p1 = F.adaptive_avg_pool1d(f1, 1).squeeze(-1)
            p2 = F.adaptive_avg_pool1d(f2, 1).squeeze(-1)
            p3 = F.adaptive_avg_pool1d(f3, 1).squeeze(-1)
            p4 = F.adaptive_avg_pool1d(f4, 1).squeeze(-1)
            return emb, [p1, p2, p3, p4]
        return emb


class ProjectionHead(nn.Module):
    def __init__(self, embed_dim=128, proj_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, proj_dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def nt_xent_loss(z1, z2, temperature=0.5):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -1e9)
    targets = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, targets)
