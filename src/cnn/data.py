from collections.abc import Callable

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class TFTBoardDataset(Dataset):
    """Dataset for TFT boards.

    Args:
        npz_path (str): Path to the features .npz file.
    """

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
        """
        Create dataset splits.

        Lightning calls this method at appropriate times:
        - Once before `fit`
        - Once before `test`
        - Once before `predict`
        """
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
