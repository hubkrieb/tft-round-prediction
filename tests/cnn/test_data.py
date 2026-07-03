import numpy as np
import torch

from src.training.cnn.data import TFTBoardDataset, _identity_collate


def _make_dataset(
    tmp_path: object, n: int = 4, n_traits: int = 6, transform_prob: float = 0.0
) -> TFTBoardDataset:
    """Write minimal per-array .npy files and return a dataset over them."""
    np.save(tmp_path / "x_units.npy", np.zeros((n, 5, 8, 7), dtype=np.int32))
    np.save(tmp_path / "x_traits.npy", np.zeros((n, n_traits), dtype=np.int8))
    np.save(tmp_path / "y.npy", np.zeros(n, dtype=np.float32))
    return TFTBoardDataset(str(tmp_path), transform_prob=transform_prob)


def test_augment_inverts_y(tmp_path: object) -> None:
    """With transform_prob=1.0 every row is rotated, so y is inverted."""
    ds = _make_dataset(tmp_path, transform_prob=1.0)

    x_units = torch.zeros((2, 5, 8, 7), dtype=torch.int32)
    x_traits = torch.zeros((2, 6), dtype=torch.int8)
    y = torch.tensor([1.0, 0.0], dtype=torch.float32)

    ds._augment(x_units, x_traits, y)

    assert torch.equal(y, torch.tensor([0.0, 1.0]))


def test_augment_swaps_trait_halves_even(tmp_path: object) -> None:
    """The first and second half of the trait vector swap (player <-> opponent)."""
    ds = _make_dataset(tmp_path, transform_prob=1.0)

    x_units = torch.zeros((1, 5, 8, 7), dtype=torch.int32)
    x_traits = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.int8)
    y = torch.tensor([1.0], dtype=torch.float32)

    ds._augment(x_units, x_traits, y)

    assert torch.equal(x_traits, torch.tensor([[4, 5, 6, 1, 2, 3]], dtype=torch.int8))


def test_augment_swaps_trait_halves_odd(tmp_path: object) -> None:
    """Odd-length trait vectors rotate by n // 2 (matches the ViT augment)."""
    ds = _make_dataset(tmp_path, n_traits=5, transform_prob=1.0)

    x_units = torch.zeros((1, 5, 8, 7), dtype=torch.int32)
    x_traits = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int8)
    y = torch.tensor([1.0], dtype=torch.float32)

    ds._augment(x_units, x_traits, y)

    # n=5, n//2=2. swapped_idx = (0,1,2,3,4 + 2) % 5 = [2, 3, 4, 0, 1]
    assert torch.equal(x_traits, torch.tensor([[3, 4, 5, 1, 2]], dtype=torch.int8))


def test_augment_rotates_units(tmp_path: object) -> None:
    """Units are rotated 180° (flip on both spatial axes), per channel."""
    ds = _make_dataset(tmp_path, transform_prob=1.0)

    C, H, W = 5, 8, 7
    x_units = torch.arange(C * H * W, dtype=torch.int32).reshape((1, C, H, W))
    x_traits = torch.zeros((1, 6), dtype=torch.int8)
    y = torch.tensor([1.0], dtype=torch.float32)

    expected = torch.flip(x_units, dims=(-2, -1))
    ds._augment(x_units, x_traits, y)

    assert torch.equal(x_units, expected)


def test_augment_noop_when_unselected(tmp_path: object) -> None:
    """transform_prob=0 selects no rows, leaving the batch untouched."""
    ds = _make_dataset(tmp_path, transform_prob=0.0)

    x_units = torch.arange(5 * 8 * 7, dtype=torch.int32).reshape((1, 5, 8, 7))
    x_traits = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.int8)
    y = torch.tensor([1.0], dtype=torch.float32)

    units_before = x_units.clone()
    ds._augment(x_units, x_traits, y)

    assert torch.equal(x_units, units_before)
    assert torch.equal(x_traits, torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.int8))
    assert y.item() == 1.0


def test_getitem_and_getitems_shapes(tmp_path: object) -> None:
    """__getitem__ returns single samples; __getitems__ returns a stacked batch."""
    ds = _make_dataset(tmp_path, n=4, n_traits=6, transform_prob=0.0)

    x_units, x_traits, y = ds[0]
    assert x_units.shape == (5, 8, 7)
    assert x_traits.shape == (6,)
    assert y.shape == ()

    bu, bt, by = ds.__getitems__([0, 1, 2])
    assert bu.shape == (3, 5, 8, 7)
    assert bt.shape == (3, 6)
    assert by.shape == (3,)


def test_identity_collate_passthrough() -> None:
    """The collate fn returns its already-stacked argument unchanged."""
    batch = (torch.zeros(2), torch.ones(2), torch.zeros(2))
    assert _identity_collate(batch) is batch
