from collections.abc import Callable

import numpy as np
import torch
from torch.utils.data import Dataset


class TFTBoardDataset(Dataset):
    """Dataset for TFT boards."""

    def __init__(self, npz_path: str, transform: Callable | None = None):
        data = np.load(npz_path, mmap_mode="r")
        self.X = data["x"]  # mem-mapped, no RAM load
        self.y = data["y"]
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx]  # shape (C, 8, 7)
        y = self.y[idx]

        x = torch.as_tensor(x, dtype=torch.int32)
        y = torch.as_tensor(y, dtype=torch.float32)

        if self.transform is not None:
            x = self.transform(x)

        return x, y
