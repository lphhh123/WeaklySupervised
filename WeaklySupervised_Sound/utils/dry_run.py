from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class DummySpec:
    num_samples: int
    in_channels: int
    clip_frames: int
    num_classes: int
    seed: int = 2024


class DummyDataset(Dataset):
    def __init__(self, spec: DummySpec):
        self.spec = spec
        self.rng = np.random.RandomState(spec.seed)

    def __len__(self) -> int:
        return self.spec.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        _ = idx
        x = self.rng.randn(self.spec.in_channels, self.spec.clip_frames).astype(np.float32)
        y = self.rng.rand(self.spec.num_classes).astype(np.float32)
        y = (y > 0.8).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def make_dummy_loader(spec: DummySpec, batch_size: int = 2) -> DataLoader:
    dataset = DummyDataset(spec)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
