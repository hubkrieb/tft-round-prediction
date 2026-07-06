"""Turn a user-assembled :class:`BoardState` into model-ready features.

Every encoding here delegates to the *exact* functions used during feature
extraction (:mod:`src.training.cnn.transform`, :mod:`src.training.vit.transform`,
:mod:`src.training.baseline.transform`). Building the same intermediate ``board_data``
representation the raw parquet has and running it through those functions
guarantees the board is encoded identically to the training data, so there is a
single source of truth for the feature semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import polars as pl

from src.training.baseline.transform import (
    FEATURE_KEYS,
    UNIT_INFO_DF,
    extract_traits_one_hot,
    process_team_features,
)
from src.training.cnn.transform import (
    ITEM_VOCAB,
    UNIT_VOCAB,
    assemble_tensors_numba,
    build_units_arrays,
)
from src.training.vit.transform import extract_traits_ids

if TYPE_CHECKING:
    import numpy as np

    from src.api.schema import BoardState

# Polars schema for one side's board: a list of unit records. Declared
# explicitly so an *empty* board still gets the right dtype (polars cannot infer
# a struct schema from an empty list otherwise).
_BOARD_DTYPE = pl.List(
    pl.Struct(
        {
            "unit": pl.String,
            "tier": pl.Int64,
            "item_ids": pl.List(pl.String),
            "loc": pl.String,
        }
    )
)

_ROUND_ID = "board"


def _base_df(board: BoardState) -> pl.DataFrame:
    """Build the single-row polars frame the trait/baseline transforms expect."""
    return pl.DataFrame(
        {
            "round_idx": [_ROUND_ID],
            "player_board": [[u.record() for u in board.player]],
            "opponent_board": [[u.record() for u in board.opponent]],
        },
        schema={
            "round_idx": pl.String,
            "player_board": _BOARD_DTYPE,
            "opponent_board": _BOARD_DTYPE,
        },
    )


def _team_data(base_df: pl.DataFrame, team: str) -> pl.DataFrame:
    """Explode one side into the per-unit frame the trait extractors consume."""
    return (
        base_df.select("round_idx", f"{team}_board")
        .explode(f"{team}_board")
        .unnest(f"{team}_board")
        .select(["round_idx", "unit", "item_ids", "loc", "tier"])
        .join(UNIT_INFO_DF, on="unit", how="left")
    )


def board_tensor(board: BoardState) -> np.ndarray:
    """Encode the matchup as the (1, 5, 8, 7) board tensor used by CNN and ViT.

    Channels are (unit, tier, item1, item2, item3); the player occupies the
    bottom 4 rows and the opponent the (point-mirrored) top 4 rows, exactly as
    :func:`src.training.cnn.transform.assemble_tensors_numba` lays them out.
    """
    board_df = pd.DataFrame(
        [
            {
                "player_board": [u.record() for u in board.player],
                "opponent_board": [u.record() for u in board.opponent],
            }
        ]
    )
    units_all, counts = build_units_arrays(board_df, UNIT_VOCAB, ITEM_VOCAB)
    return assemble_tensors_numba(units_all, counts, units_all.shape[0])


def vit_trait_ids(board: BoardState) -> np.ndarray:
    """Encode active traits as the (1, 30) padded trait-ID vector used by the ViT.

    Player trait IDs occupy the first 15 slots, opponent the last 15, matching
    the column order produced during ViT feature extraction.
    """
    base_df = _base_df(board)
    player = extract_traits_ids(_team_data(base_df, "player"), "player")
    opponent = extract_traits_ids(_team_data(base_df, "opponent"), "opponent")
    return (
        player.join(opponent, on="round_idx", how="inner")
        .select(pl.all().exclude("round_idx"))
        .to_numpy()
    )


def cnn_trait_onehot(board: BoardState) -> np.ndarray:
    """Encode active traits as the one-hot trait vector used by the CNN.

    Player columns first, then opponent, in the canonical ``FEATURE_KEYS`` order.
    """
    base_df = _base_df(board)
    player = extract_traits_one_hot(_team_data(base_df, "player"), "player")
    opponent = extract_traits_one_hot(_team_data(base_df, "opponent"), "opponent")
    return (
        player.join(opponent, on="round_idx", how="inner")
        .select(pl.all().exclude("round_idx"))
        .to_numpy()
    )


def baseline_features(board: BoardState, feature_order: list[str]) -> pd.DataFrame:
    """Build the wide one-hot feature row for the XGBoost baseline.

    Args:
        board: The matchup to encode.
        feature_order: The exact ordered column list the model was trained on
            (saved next to the model as ``*_features.json``); the returned frame
            is reindexed to it so column alignment matches training.
    """
    base_df = _base_df(board)
    player = process_team_features(
        base_df.select("round_idx", "player_board"), "player"
    )
    opponent = process_team_features(
        base_df.select("round_idx", "opponent_board"), "opponent"
    )
    wide = (
        player.join(opponent, on="round_idx", how="inner")
        .select(FEATURE_KEYS)
        .fill_null(0)
    )
    return wide.to_pandas().reindex(columns=feature_order, fill_value=0)
