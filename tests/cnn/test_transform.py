import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import polars as pl

from src.cnn.transform import (
    C_IDX,
    CHANNELS,
    COLS,
    FIELDS,
    IS_PLAYER_IDX,
    ITEM1_IDX,
    R_IDX,
    ROWS,
    UNIT_IDX,
    assemble_tensors_numba,
    build_units_arrays,
    extract_tensors,
    loc_to_rc,
)


def test_loc_to_rc() -> None:
    """Test string coordinate tracking to grid indicies."""
    # Valid inputs
    assert loc_to_rc("A1") == (0, 0)
    assert loc_to_rc("D7") == (3, 6)
    assert loc_to_rc("A_1") == (0, 0)
    assert loc_to_rc("B_4") == (1, 3)

    # Invalid inputs
    assert loc_to_rc(None) == (-1, -1)
    assert loc_to_rc("") == (-1, -1)
    assert loc_to_rc("Z1") == (-1, -1)
    assert loc_to_rc("A9") == (-1, -1)
    assert loc_to_rc("X_5") == (-1, -1)
    assert loc_to_rc("A_") == (-1, -1)


def test_build_units_arrays() -> None:
    """Test extraction of unit features from board structures into a fixed length numpy array."""
    unit_to_id = {"UnitA": 1, "UnitB": 2}
    item_to_id = {"Item1": 1}

    player_board = [
        {"loc": "A1", "unit": "UnitA", "tier": 2, "item_ids": ["Item1"]},
        {
            "loc": "Z9",
            "unit": "UnitB",
            "tier": 1,
            "item_ids": [],
        },  # Invalid loc, dropped
    ]
    opponent_board = [
        {"loc": "D7", "unit": "UnitB", "tier": 3, "item_ids": ["Item1", "Item1"]},
    ]

    df = pd.DataFrame(
        {"player_board": [player_board], "opponent_board": [opponent_board]}
    )

    units_all, counts = build_units_arrays(df, unit_to_id, item_to_id, max_units=10)

    assert counts[0] == 2  # 1 player unit + 1 opponent unit
    assert units_all.shape == (1, 10, FIELDS)

    # Check Player Unit
    assert units_all[0, 0, R_IDX] == 0
    assert units_all[0, 0, C_IDX] == 0
    assert units_all[0, 0, UNIT_IDX] == 1
    assert units_all[0, 0, ITEM1_IDX] == 1
    assert units_all[0, 0, IS_PLAYER_IDX] == 1

    # Check Opponent Unit
    assert units_all[0, 1, R_IDX] == 3
    assert units_all[0, 1, C_IDX] == 6
    assert units_all[0, 1, UNIT_IDX] == 2
    assert units_all[0, 1, ITEM1_IDX] == 1
    assert units_all[0, 1, IS_PLAYER_IDX] == 0


def test_assemble_tensors_numba() -> None:
    """Test arrangement of the unit features to the final CNN tensor structure."""
    units_all = np.zeros((1, 2, FIELDS), dtype=np.int32)
    counts = np.array([2], dtype=np.int32)

    # Player Unit
    units_all[0, 0, R_IDX] = 0
    units_all[0, 0, C_IDX] = 0
    units_all[0, 0, UNIT_IDX] = 1
    units_all[0, 0, IS_PLAYER_IDX] = 1

    # Opponent Unit
    units_all[0, 1, R_IDX] = 0
    units_all[0, 1, C_IDX] = 0
    units_all[0, 1, UNIT_IDX] = 2
    units_all[0, 1, IS_PLAYER_IDX] = 0

    tensors = assemble_tensors_numba(units_all, counts, 1)

    assert tensors.shape == (1, CHANNELS, 2 * ROWS, COLS)

    # Check player placement (shifted by ROWS for y, same for x) -> row 4, col 0
    assert tensors[0, 0, 4, 0] == 1

    # Check opponent placement (mirrored: 3 - r, 6 - c) -> row 3, col 6
    assert tensors[0, 0, 3, 6] == 2


@patch("src.cnn.transform.extract_traits_one_hot")
def test_extract_tensors(mock_extract_traits_one_hot: MagicMock) -> None:
    """Test the full extraction pipeline over structured game data."""
    mock_extract_traits_one_hot.return_value = pl.DataFrame(
        {"round_idx": ["match_1_1_1_0"], "trait_feature": [1]}
    )

    temp_dir = tempfile.mkdtemp()
    raw_data_path = os.path.join(temp_dir, "dummy_raw.parquet")
    feature_path = os.path.join(temp_dir, "dummy_features")

    player_board = [{"unit": "TFT16_Tristana", "item_ids": [], "loc": "A1", "tier": 1}]
    opponent_board = [{"unit": "TFT16_Lulu", "item_ids": [], "loc": "A1", "tier": 1}]

    df = pl.DataFrame(
        {
            "match_uuid": ["match_1", "match_1"],
            "player_uuid": ["player_1", "player_1"],
            "round_name": ["1_1", "1_2"],
            "round_type": ["PvP", "PVE"],
            "round_outcome": ["victory", "defeat"],
            "timestamp": [1770000000000, 1770000000001],
            "board_data": [
                {"player_board": player_board, "opponent_board": opponent_board},
                {"player_board": player_board, "opponent_board": opponent_board},
            ],
        }
    )

    df.write_parquet(raw_data_path)

    try:
        tensors, traits_feat, outcome = extract_tensors(raw_data_path, feature_path)

        # Output shape first dimension should be 1 (one PvP match)
        assert tensors.shape[0] == 1
        assert traits_feat.shape[0] == 1
        assert outcome.shape[0] == 1

        # Ensure the output directory and per-array files were created
        assert os.path.isdir(feature_path)
        for name in ("x_units.npy", "x_traits.npy", "timestamp.npy", "y.npy"):
            assert os.path.exists(os.path.join(feature_path, name))

        # Outcome 1 corresponds to victory
        assert outcome[0] == 1

    finally:
        if os.path.exists(raw_data_path):
            os.remove(raw_data_path)
        if os.path.isdir(feature_path):
            shutil.rmtree(feature_path)
        os.rmdir(temp_dir)
