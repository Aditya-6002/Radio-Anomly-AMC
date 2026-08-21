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

class IQEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(256, embed_dim)

    def forward(self, x):
        h = self.net(x).squeeze(-1)
        return self.fc(h)


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
