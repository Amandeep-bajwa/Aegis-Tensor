"""Small architectures importable by `aegis fuzz --arch tests.fuzz_fixtures:Name`."""

import os

import torch.nn as nn


class ThresholdBackdoor(nn.Module):
    """Dormant trigger: a large input on feature 0 amplifies the hidden layer by 50x."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        h = self.fc1(x)
        if (x[:, 0] > 8.0).any():
            h = h * 50.0
        return self.fc2(h)


class CleanMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        return self.net(x)


class SystemCallOnLoad(nn.Module):
    """A real model whose pickle also runs `touch <marker>` when unpickled."""

    def __init__(self, marker: str = ""):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        self.marker = marker

    def forward(self, x):
        return self.fc(x)

    def __reduce__(self):
        return (_rebuild_with_side_effect, (self.marker,))


def _rebuild_with_side_effect(marker: str):
    os.system(f"touch {marker}")  # noqa: S605 - the benign payload: it only creates a marker file in tmp_path
    return SystemCallOnLoad(marker)
