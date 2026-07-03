import os
import shutil
import tempfile
from unittest.mock import patch

import polars as pl

from src.training.vit.transform import MAX_TRAITS, extract_tensors, extract_traits_ids


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
            "item_ids": [[], [], [], [], []],
        },
        schema_overrides={"item_ids": pl.List(pl.String)},
    )

    with (
        patch("src.training.vit.transform.TRAITS", mock_traits),
        patch("src.training.vit.transform.TRAIT_VOCAB", mock_trait_vocab),
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


def test_emblem_pushes_trait_to_higher_breakpoint() -> None:
    """Test that an emblem item adds to a trait count, pushing it to a higher breakpoint."""
    mock_traits = {"TraitA": [2, 4]}
    mock_trait_vocab = {"TraitA_2": 1, "TraitA_4": 2}
    mock_emblem_to_trait = {"EmblemA": "TraitA"}

    # 1 innate TraitA unit + 1 EmblemA -> count = 2, reaches bp 2
    team_data = pl.DataFrame(
        {
            "round_idx": ["R1", "R1"],
            "unit": ["U1", "U2"],
            "traits": [["TraitA"], []],
            "item_ids": [[], ["EmblemA"]],
        }
    )

    with (
        patch("src.training.vit.transform.TRAITS", mock_traits),
        patch("src.training.vit.transform.TRAIT_VOCAB", mock_trait_vocab),
        patch("src.training.vit.transform.EMBLEMS", mock_emblem_to_trait),
    ):
        result = extract_traits_ids(team_data, "player")

    r1 = result.filter(pl.col("round_idx") == "R1")
    # TraitA: 1 innate + 1 emblem = 2 -> bp 2 reached -> TraitA_2 -> id 1
    assert r1["player_trait_0"][0] == 1
    assert r1["player_trait_1"][0] == 0  # No other trait


def test_emblem_activates_new_trait() -> None:
    """Test that an emblem alone can activate a trait the unit doesn't innately have."""
    mock_traits = {"TraitA": [2], "TraitB": [1]}
    mock_trait_vocab = {"TraitA_2": 1, "TraitB_1": 2}
    mock_emblem_to_trait = {"EmblemB": "TraitB"}

    # 2 units with TraitA, one carries an EmblemB -> TraitB activated at bp 1
    team_data = pl.DataFrame(
        {
            "round_idx": ["R1", "R1"],
            "unit": ["U1", "U2"],
            "traits": [["TraitA"], ["TraitA"]],
            "item_ids": [["EmblemB"], []],
        }
    )

    with (
        patch("src.training.vit.transform.TRAITS", mock_traits),
        patch("src.training.vit.transform.TRAIT_VOCAB", mock_trait_vocab),
        patch("src.training.vit.transform.EMBLEMS", mock_emblem_to_trait),
    ):
        result = extract_traits_ids(team_data, "player")

    r1 = result.filter(pl.col("round_idx") == "R1")
    # TraitA: 2 innate -> bp 2 -> id 1
    # TraitB: 1 emblem -> bp 1 -> id 2
    assert r1["player_trait_0"][0] == 1
    assert r1["player_trait_1"][0] == 2
    assert r1["player_trait_2"][0] == 0


def test_multiple_emblems_on_single_unit() -> None:
    """Test that multiple emblem items on a single unit each contribute separately."""
    mock_traits = {"TraitA": [1], "TraitB": [1]}
    mock_trait_vocab = {"TraitA_1": 1, "TraitB_1": 2}
    mock_emblem_to_trait = {"EmblemA": "TraitA", "EmblemB": "TraitB"}

    # Single unit with two different emblems
    team_data = pl.DataFrame(
        {
            "round_idx": ["R1"],
            "unit": ["U1"],
            "traits": [[]],
            "item_ids": [["EmblemA", "EmblemB"]],
        },
        schema_overrides={"traits": pl.List(pl.String)},
    )

    with (
        patch("src.training.vit.transform.TRAITS", mock_traits),
        patch("src.training.vit.transform.TRAIT_VOCAB", mock_trait_vocab),
        patch("src.training.vit.transform.EMBLEMS", mock_emblem_to_trait),
    ):
        result = extract_traits_ids(team_data, "player")

    r1 = result.filter(pl.col("round_idx") == "R1")
    # Both traits activated at bp 1
    assert r1["player_trait_0"][0] == 1  # TraitA_1
    assert r1["player_trait_1"][0] == 2  # TraitB_1
    assert r1["player_trait_2"][0] == 0


def test_no_emblem_items() -> None:
    """Test that rounds without emblem items still work correctly (no regression)."""
    mock_traits = {"TraitA": [2]}
    mock_trait_vocab = {"TraitA_2": 1}
    mock_emblem_to_trait = {"EmblemA": "TraitA"}

    team_data = pl.DataFrame(
        {
            "round_idx": ["R1", "R1"],
            "unit": ["U1", "U2"],
            "traits": [["TraitA"], ["TraitA"]],
            "item_ids": [["SomeOtherItem"], []],
        }
    )

    with (
        patch("src.training.vit.transform.TRAITS", mock_traits),
        patch("src.training.vit.transform.TRAIT_VOCAB", mock_trait_vocab),
        patch("src.training.vit.transform.EMBLEMS", mock_emblem_to_trait),
    ):
        result = extract_traits_ids(team_data, "player")

    r1 = result.filter(pl.col("round_idx") == "R1")
    # TraitA: 2 innate -> bp 2 -> id 1, no emblem contribution
    assert r1["player_trait_0"][0] == 1
    assert r1["player_trait_1"][0] == 0


def test_extract_tensors() -> None:
    """Test that extract_tensors correctly parses raw parquet data and returns valid tensors."""
    temp_dir = tempfile.mkdtemp()
    raw_data_path = os.path.join(temp_dir, "dummy_raw.parquet")
    feature_path = os.path.join(temp_dir, "dummy_features")

    player_board = [{"unit": "TFT16_Tristana", "item_ids": [], "loc": 0, "tier": 1}]
    opponent_board = [{"unit": "TFT16_Lulu", "item_ids": [], "loc": 1, "tier": 1}]

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
        tensors, traits_feat, patch_ids, outcome = extract_tensors(
            raw_data_path, feature_path
        )

        # Since one was filtered out (PVE), output shape first dimension should be 1
        assert tensors.shape[0] == 1
        assert traits_feat.shape[0] == 1
        assert outcome.shape[0] == 1

        # Ensure the output directory and per-array files were created
        assert os.path.isdir(feature_path)
        for name in (
            "x_units.npy",
            "x_traits.npy",
            "x_patch.npy",
            "timestamp.npy",
            "y.npy",
            "round_idx.npy",
        ):
            assert os.path.exists(os.path.join(feature_path, name))

        # Ensure outcome was parsed correctly ('victory' -> 1)
        assert outcome[0] == 1

    finally:
        if os.path.exists(raw_data_path):
            os.remove(raw_data_path)
        if os.path.isdir(feature_path):
            shutil.rmtree(feature_path)
        os.rmdir(temp_dir)
