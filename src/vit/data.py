from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


def _identity_collate(batch: tuple) -> tuple:
    """Pass a pre-collated batch straight through.

    ``TFTBoardDataset.__getitems__`` already returns stacked batch tensors, so
    there is nothing left to collate. Defined at module level (not a lambda) so
    it stays picklable for spawned DataLoader workers on Windows.
    """
    return batch


class TFTBoardDataset(Dataset):
    """Dataset for TFT boards.

    Args:
        data_path (str): Directory containing the per-array .npy files
            (x_units.npy, x_traits.npy, x_patch.npy, y.npy).

    The memmaps are opened lazily on first access rather than in ``__init__``.
    This matters on Windows, where ``DataLoader(num_workers>0)`` spawns workers
    and pickles the dataset to each one: a memmap stored as an attribute would
    be materialized into RAM by pickle and copied into every worker, defeating
    mmap and blowing up memory. By keeping only the directory path in the
    pickled state and opening the memmaps inside each worker, every worker
    shares the OS page cache and nothing is loaded eagerly.
    """

    def __init__(
        self, data_path: str, transform_prob: float = 0.5, lambda_: float = 0.1
    ):
        self.data_dir = Path(data_path)
        self.transform_prob = transform_prob
        self.lambda_ = lambda_

        # Read only x_patch (small, one int per sample) to derive length and the
        # latest patch id, then drop it so it is not part of the pickled state.
        x_patch = np.load(self.data_dir / "x_patch.npy", mmap_mode="r")
        self._len = x_patch.shape[0]
        self.latest_patch_id = int(x_patch.max())
        del x_patch

        # Opened lazily per process in _ensure_open(); never pickled.
        self._X_units: np.memmap | None = None
        self._X_traits: np.memmap | None = None
        self._X_patch: np.memmap | None = None
        self._y: np.memmap | None = None

    def _ensure_open(self) -> None:
        """Open the memmaps if they aren't already (called inside each worker)."""
        if self._X_units is None:
            self._X_units = np.load(self.data_dir / "x_units.npy", mmap_mode="r")
            self._X_traits = np.load(self.data_dir / "x_traits.npy", mmap_mode="r")
            self._X_patch = np.load(self.data_dir / "x_patch.npy", mmap_mode="r")
            self._y = np.load(self.data_dir / "y.npy", mmap_mode="r")

    def __getstate__(self) -> dict:
        # Exclude memmap handles so pickling (Windows spawn) stays tiny.
        state = self.__dict__.copy()
        state["_X_units"] = None
        state["_X_traits"] = None
        state["_X_patch"] = None
        state["_y"] = None
        return state

    def __len__(self):
        return self._len

    def __getitems__(self, indices: list[int]) -> tuple[torch.Tensor, ...]:
        """Fetch and collate a whole batch at once (vectorized).

        PyTorch's map-style fetcher calls ``__getitems__`` with the full list of
        indices for the batch when it is defined (``Subset`` forwards it too), so
        all per-sample Python overhead — one memmap read, ``np.array`` copy,
        scalar ``exp`` and tensor build per item, then a stacking collate —
        collapses into a single fancy-index gather, one ``exp`` and one
        augmentation pass over the batch. Returned tensors are already stacked,
        so the DataLoader uses ``_identity_collate``.
        """
        self._ensure_open()
        idx = np.asarray(indices)

        # One fancy-index gather per array: each yields a fresh, writable,
        # contiguous copy out of the (read-only) memmap.
        x_units = torch.from_numpy(self._X_units[idx].astype(np.int32, copy=False))
        x_traits = torch.from_numpy(self._X_traits[idx].astype(np.int8))
        patch_ids = self._X_patch[idx].astype(np.int64, copy=False)
        y = torch.from_numpy(self._y[idx].astype(np.float32))

        x_patch = torch.from_numpy(patch_ids.astype(np.int8))
        age = self.latest_patch_id - patch_ids
        w = torch.from_numpy(np.exp(-self.lambda_ * age).astype(np.float32))

        if self.transform_prob > 0:
            self._augment_(x_units, x_traits, y)

        return x_units, x_traits, x_patch, w, y

    def _augment_(
        self, x_units: torch.Tensor, x_traits: torch.Tensor, y: torch.Tensor
    ) -> None:
        """Rotate ~``transform_prob`` of the batch by 180°, in place.

        Rotating the board swaps the player and opponent sides: the unit grid
        flips on both spatial axes, the player/opponent trait halves swap, and
        the outcome inverts. Done as masked batch ops rather than per sample.
        """
        sel = torch.nonzero(
            torch.rand(x_units.shape[0]) < self.transform_prob, as_tuple=True
        )[0]
        if sel.numel() == 0:
            return

        x_units[sel] = torch.flip(x_units[sel], dims=(-2, -1))

        n = x_traits.shape[1]
        swapped_idx = (torch.arange(n) + n // 2) % n
        x_traits[sel] = x_traits[sel][:, swapped_idx]

        y[sel] = 1.0 - y[sel]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        # Single source of truth: reuse the batched path for one element.
        x_units, x_traits, x_patch, w, y = self.__getitems__([idx])
        return x_units[0], x_traits[0], x_patch[0], w[0], y[0]


class TFTBoardDataModule(L.LightningDataModule):
    """
    DataModule for TFT boards.

    Args:
        data_path (str): Directory containing the per-array .npy files.
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
            collate_fn=_identity_collate,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.num_workers > 0),
            collate_fn=_identity_collate,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.num_workers > 0),
            collate_fn=_identity_collate,
        )
