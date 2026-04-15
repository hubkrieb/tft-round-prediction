from random import random

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.cnn.data import rotate_board


class TFTBoardDataset(Dataset):
    """Dataset for TFT boards.

    Args:
        npz_path (str): Path to the features .npz file.
    """

    def __init__(
        self, npz_path: str, transform_prob: float = 0.5, lambda_: float = 0.1
    ):
        data = np.load(npz_path, mmap_mode="r")
        self.X_units = data["x_units"]  # mem-mapped, no RAM load
        self.X_traits = data["x_traits"]
        self.X_patch = data["x_patch"]
        self.y = data["y"]
        self.transform_prob = transform_prob

        self.latest_patch_id = int(self.X_patch.max())
        self.lambda_ = lambda_

        patch_age = self.latest_patch_id - self.X_patch
        self.patch_weights = np.exp(-self.lambda_ * patch_age)

    def __len__(self):
        return self.X_units.shape[0]

    def __getitem__(self, idx: int):
        x_units = self.X_units[idx]  # shape (C, 8, 7)
        x_traits = self.X_traits[idx]
        x_patch = self.X_patch[idx]
        y = self.y[idx]
        w = self.patch_weights[idx]

        x_units = torch.as_tensor(x_units, dtype=torch.int32)
        x_traits = torch.as_tensor(x_traits, dtype=torch.int8)
        x_patch = torch.as_tensor(x_patch, dtype=torch.int8)
        y = torch.as_tensor(y, dtype=torch.float32)
        w = torch.as_tensor(w, dtype=torch.float32)

        if self.transform_prob > 0 and random() < self.transform_prob:
            x_units, x_traits, y = rotate_board(x_units, x_traits, y)

        return x_units, x_traits, x_patch, w, y


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
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        train_split: float = 0.8,
        val_split: float = 0.1,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_split = train_split
        self.val_split = val_split

    def setup(self, stage: str | None = None) -> None:
        """Create dataset splits."""
        train_dataset = TFTBoardDataset(self.data_path, transform_prob=0.5)
        eval_dataset = TFTBoardDataset(self.data_path, transform_prob=0.0)

        train_size = int(self.train_split * len(train_dataset))
        val_size = int(self.val_split * len(train_dataset))

        n = len(train_dataset)
        indices = list(range(n))[::-1]

        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size :]

        self.train_ds = Subset(train_dataset, train_indices)
        self.val_ds = Subset(eval_dataset, val_indices)
        self.test_ds = Subset(eval_dataset, test_indices)

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
