import os
import tempfile
from unittest.mock import patch

import polars as pl

from src.vit.transform import MAX_TRAITS, extract_tensors, extract_traits_ids


def test_extract_traits_ids() -> None:
    """Test that extract_traits_ids correctly extracts and pads trait IDs."""
    # Mock TRAITS and TRAIT_VOCAB
    mock_traits = {"TraitA": [2, 4], "TraitB": [1, 3]}
    mock_trait_vocab = {"TraitA_2": 1, "TraitA_4": 2, "TraitB_1": 3, "TraitB_3": 4}

    team_data = pl.DataFrame(
        {
            "round_idx": ["R1", "R1", "R1", "R2", "R3"],
            "unit": ["U1", "U2", "U3", "U4", "U5"],
            "traits": [
                ["TraitA"],
                ["TraitA", "TraitB"],
                ["TraitA"],
                ["TraitB"],
                ["TraitB"],
            ],
        }
    )

    with (
        patch("src.vit.transform.TRAITS", mock_traits),
        patch("src.vit.transform.TRAIT_VOCAB", mock_trait_vocab),
    ):
        result = extract_traits_ids(team_data, "player")

    # Check output structure
    assert "round_idx" in result.columns
    for i in range(MAX_TRAITS):
        assert f"player_trait_{i}" in result.columns

    # Validation for R1
    # TraitA -> 3 units -> passes bp 2 -> TraitA_2 -> id 1
    # TraitB -> 1 unit -> passes bp 1 -> TraitB_1 -> id 3
    r1_result = result.filter(pl.col("round_idx") == "R1")
    assert r1_result["player_trait_0"][0] == 1
    assert r1_result["player_trait_1"][0] == 3
    assert r1_result["player_trait_2"][0] == 0

    # Validation for R2
    r2_result = result.filter(pl.col("round_idx") == "R2")
    assert r2_result["player_trait_0"][0] == 3
    assert r2_result["player_trait_1"][0] == 0

    # Validation for R3
    r3_result = result.filter(pl.col("round_idx") == "R3")
    assert r3_result["player_trait_0"][0] == 3
    assert r3_result["player_trait_1"][0] == 0


def test_extract_tensors() -> None:
    """Test that extract_tensors correctly parses raw parquet data and returns valid tensors."""
    temp_dir = tempfile.mkdtemp()
    raw_data_path = os.path.join(temp_dir, "dummy_raw.parquet")
    feature_path = os.path.join(temp_dir, "dummy_features.npz")

    player_board = [{"unit": "TFT16_Tristana", "item_ids": [], "loc": 0, "tier": 1}]
    opponent_board = [{"unit": "TFT16_Lulu", "item_ids": [], "loc": 1, "tier": 1}]

    df = pl.DataFrame(
        {
            "match_uuid": ["match_1", "match_1"],
            "player_uuid": ["player_1", "player_1"],
            "round_name": ["1_1", "1_2"],
            "round_type": ["PvP", "PVE"],
            "round_outcome": ["victory", "defeat"],
            "board_data": [
                {"player_board": player_board, "opponent_board": opponent_board},
                {"player_board": player_board, "opponent_board": opponent_board},
            ],
        }
    )

    df.write_parquet(raw_data_path)

    try:
        tensors, traits_feat, outcome = extract_tensors(raw_data_path, feature_path)

        # Since one was filtered out (PVE), output shape first dimension should be 1
        assert tensors.shape[0] == 1
        assert traits_feat.shape[0] == 1
        assert outcome.shape[0] == 1

        # Ensure the output file was created
        assert os.path.exists(feature_path)

        # Ensure outcome was parsed correctly ('victory' -> 1)
        assert outcome[0] == 1

    finally:
        if os.path.exists(raw_data_path):
            os.remove(raw_data_path)
        if os.path.exists(feature_path):
            os.remove(feature_path)
        os.rmdir(temp_dir)
