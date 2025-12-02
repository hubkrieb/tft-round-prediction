from random import random

import lightning as L
import numpy as np
import torch
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


def rotate_board(
    x_units: torch.Tensor, x_traits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate the board representation by 180 degrees.

    Is equivalent to switch the opponent and the player side.

    Args:
        x_units (torch.Tensor): Tensor of shape (C, 8, 7) representing unit IDs on the board.
        x_traits (torch.Tensor): Tensor of shape (T,) representing trait features.

    Returns:
        torch.Tensor: Rotated unit tensor of shape (C, 8, 7).
        torch.Tensor: Unchanged trait tensor of shape (T,).
    """
    x_units_rotated = F.rotate(x_units, angle=180)

    n = len(x_traits)
    swapped_idx = (torch.arange(n) + n // 2) % n

    x_traits_swapped = x_traits[swapped_idx]

    return x_units_rotated, x_traits_swapped


class TFTBoardDataset(Dataset):
    """Dataset for TFT boards.

    Args:
        npz_path (str): Path to the features .npz file.
    """

    def __init__(self, npz_path: str, transform_prob: float = 0.5):
        data = np.load(npz_path, mmap_mode="r")
        self.X_units = data["x_units"]  # mem-mapped, no RAM load
        self.X_traits = data["x_traits"]
        self.y = data["y"]
        self.transform_prob = transform_prob

    def __len__(self):
        return self.X_units.shape[0]

    def __getitem__(self, idx: int):
        x_units = self.X_units[idx]  # shape (C, 8, 7)
        x_traits = self.X_traits[idx]
        y = self.y[idx]

        x_units = torch.as_tensor(x_units, dtype=torch.int32)
        x_traits = torch.as_tensor(x_traits, dtype=torch.int8)
        y = torch.as_tensor(y, dtype=torch.float32)

        if random() < self.transform_prob:
            x_units, x_traits = rotate_board(x_units, x_traits)

        return x_units, x_traits, y


class TFTBoardDataModule(L.LightningDataModule):
    """
    DataModule for TFT boards.

    Args:
        data_path (str): Path to the features .npz file.
        batch_size (int): Number of samples per batch during training and evaluation.
        num_workers (int): Number of worker processes for DataLoaders.
        pin_memory (bool): Whether to pin memory (improves GPU transfer speed).
        train_split (float): Percentage of data to use for training.
        val_split (float): Percentage of data to use for validation.
        seed (int): Random seed for deterministic splitting.
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        train_split: float = 0.8,
        val_split: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_split = train_split
        self.val_split = val_split
        self.seed = seed

    def setup(self, stage: str | None = None) -> None:
        """Create dataset splits."""
        full_dataset = TFTBoardDataset(self.data_path)
        train_size = int(self.train_split * len(full_dataset))
        val_size = int(self.val_split * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        self.train_ds, self.val_ds, self.test_ds = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.seed),
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.num_workers > 0),
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.num_workers > 0),
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.num_workers > 0),
        )
