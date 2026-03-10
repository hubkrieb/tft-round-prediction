import torch

from src.cnn.data import rotate_board


def test_rotate_board_shapes() -> None:
    """Verify that rotate_board preserves the shape of units and traits."""
    C, H, W = 3, 8, 7
    T = 10

    x_units = torch.randint(0, 100, (C, H, W), dtype=torch.int32)
    x_traits = torch.randint(0, 5, (T,), dtype=torch.int8)
    y = torch.tensor(1.0, dtype=torch.float32)

    rotated_units, swapped_traits, inverted_y = rotate_board(x_units, x_traits, y)

    assert rotated_units.shape == (C, H, W)
    assert swapped_traits.shape == (T,)
    assert rotated_units.dtype == torch.int32
    assert swapped_traits.dtype == torch.int8
    assert inverted_y.item() == 0.0


def test_rotate_board_traits_swapping_even() -> None:
    """Verify that the first half and second half of traits are swapped for even length."""
    x_traits = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int8)

    # x_units isn't checked here
    x_units = torch.zeros((1, 8, 7), dtype=torch.int32)
    y_dummy = torch.tensor(1.0)
    _, swapped_traits, _ = rotate_board(x_units, x_traits, y_dummy)

    expected_traits = torch.tensor([4, 5, 6, 1, 2, 3], dtype=torch.int8)
    assert torch.equal(swapped_traits, expected_traits)


def test_rotate_board_traits_swapping_odd() -> None:
    """Verify behavior on odd length traits."""
    x_traits = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int8)

    x_units = torch.zeros((1, 8, 7), dtype=torch.int32)
    y_dummy = torch.tensor(1.0)
    _, swapped_traits, _ = rotate_board(x_units, x_traits, y_dummy)

    # n=5, n//2=2. swapped_idx = (0,1,2,3,4 + 2) % 5 = [2, 3, 4, 0, 1]
    expected_traits = torch.tensor([3, 4, 5, 1, 2], dtype=torch.int8)
    assert torch.equal(swapped_traits, expected_traits)


def test_rotate_board_units_rotation() -> None:
    """Verify that units are correctly rotated by 180 degrees."""
    C, H, W = 1, 8, 7

    # Create a tensor where each element is unique
    x_units = torch.arange(H * W, dtype=torch.int32).reshape((C, H, W))
    y_dummy = torch.tensor(1.0)

    rotated_units, _, _ = rotate_board(x_units, torch.zeros(10), y_dummy)

    # 180 degree rotation of a 2D matrix is equivalent to flipping both spatial dimensions
    expected_units = torch.flip(x_units, dims=[-2, -1])

    assert torch.equal(rotated_units, expected_units)


def test_rotate_board_multichannel_units_rotation() -> None:
    """Verify rotation works correctly independently for multiple channels."""
    C, H, W = 3, 8, 7
    x_units = torch.arange(C * H * W, dtype=torch.int32).reshape((C, H, W))
    y_dummy = torch.tensor(1.0)

    rotated_units, _, _ = rotate_board(x_units, torch.zeros(10), y_dummy)

    expected_units = torch.flip(x_units, dims=[-2, -1])

    assert torch.equal(rotated_units, expected_units)


def test_dataset_augmentation_inverts_y(tmp_path: object) -> None:
    """Verify that y is inverted when data augmentation is applied (board rotated)."""
    C, H, W = 3, 8, 7

    x_units = torch.zeros((1, C, H, W), dtype=torch.int32)
    x_traits = torch.zeros((1, 10), dtype=torch.int8)
    y = torch.tensor([1.0], dtype=torch.float32)

    _, _, rotated_y = rotate_board(x_units, x_traits, y)

    assert rotated_y.item() == 0.0
