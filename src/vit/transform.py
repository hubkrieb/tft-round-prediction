import numpy as np
import polars as pl
from numba import njit, prange

from src.baseline.transform import UNIT_INFO_DF, extract_traits_one_hot
from src.cnn.transform import build_units_arrays
from src.utils.static_data import ITEMS, UNITS

UNIT_TO_ID = {unit: i + 1 for i, unit in enumerate(UNITS)}
ITEM_TO_ID = {item: i + 1 for i, item in enumerate(ITEMS)}

MAX_UNITS = 24
CHANNELS = 6
ROWS = 4
COLS = 7

R_IDX = 0
C_IDX = 1
UNIT_IDX = 2
TIER_IDX = 3
ITEM1_IDX = 4
ITEM2_IDX = 5
ITEM3_IDX = 6
IS_PLAYER_IDX = 7
FIELDS = 8


@njit(parallel=True)
def assemble_sequence_numba(
    units_all: np.ndarray, counts: np.ndarray, n_samples: int, max_units: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assemble the board data into sequence format for Transformer.

    Returns unit sequences and attention masks.

    Args:
        units_all (np.ndarray): The board data
        counts (np.ndarray): The unit counts for each board
        n_samples (int): The amount of rounds/boards in units_all
        max_units (int): Maximum number of units per sequence

    Returns:
        tuple[np.ndarray, np.ndarray]: Unit sequences and attention masks
    """
    # Shape: (n_samples, max_units, FIELDS)
    sequences = np.zeros((n_samples, max_units, FIELDS), dtype=np.int32)
    masks = np.zeros((n_samples, max_units), dtype=np.bool_)

    for i in prange(n_samples):
        cnt = counts[i]
        for k in range(min(cnt, max_units)):
            sequences[i, k] = units_all[i, k]
            masks[i, k] = True

    return sequences, masks


def extract_sequences(
    raw_data_path: str, feature_path: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts and processes features from raw game data and saves them as a .npz file.

    Args:
        raw_data_path (str): Path to the input Parquet file containing raw game data.
        feature_path (str): Path where the processed feature .npz file will be saved.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            The unit sequences, attention masks, trait features, and round outcomes.
    """
    df = pl.read_parquet(raw_data_path)

    df = df.with_columns(
        pl.arange(0, pl.count())
        .over(["match_uuid", "player_uuid", "round_name"])
        .alias("round_instance")
    ).with_columns(
        round_idx=(
            pl.col("match_uuid").cast(pl.Utf8)
            + pl.lit("_")
            + pl.col("round_name").cast(pl.Utf8)
            + pl.lit("_")
            + pl.col("round_instance").cast(pl.Utf8)
        )
    )

    # Filter out PVE rounds and missing input or target
    mask = (
        (~pl.col("round_type").eq("PVE"))
        & (pl.col("round_outcome").is_not_null())
        & (pl.all_horizontal(pl.col("board_data").struct.unnest().is_not_null()))
        & (pl.all_horizontal(pl.col("board_data").struct.unnest().list.len() > 0))
    )

    base_df = df.filter(mask).select(
        pl.col("match_uuid"),
        pl.col("round_idx"),
        (pl.col("round_outcome") == "victory").cast(pl.Int8).alias("outcome"),
        pl.col("board_data").struct.unnest(),
    )

    player_data = (
        base_df.select("round_idx", "player_board")
        .explode("player_board")
        .unnest("player_board")
        .select(["round_idx", "unit", "item_ids", "loc", "tier"])
        .join(UNIT_INFO_DF, on="unit", how="left")
    )

    opponent_data = (
        base_df.select("round_idx", "opponent_board")
        .explode("opponent_board")
        .unnest("opponent_board")
        .select(["round_idx", "unit", "item_ids", "loc", "tier"])
        .join(UNIT_INFO_DF, on="unit", how="left")
    )

    player_traits = extract_traits_one_hot(team_data=player_data, team_name="player")
    opponent_traits = extract_traits_one_hot(
        team_data=opponent_data, team_name="opponent"
    )
    trait_features = (
        player_traits.join(opponent_traits, on="round_idx", how="inner")
        .select(pl.all().exclude("round_idx"))
        .to_numpy()
    )

    base_df = base_df.to_pandas()

    outcome = np.array((base_df["outcome"]).astype(int))

    units_all, counts = build_units_arrays(base_df, UNIT_TO_ID, ITEM_TO_ID)

    sequences, masks = assemble_sequence_numba(
        units_all, counts, units_all.shape[0], MAX_UNITS
    )

    np.savez_compressed(
        feature_path,
        x_sequences=sequences,
        x_masks=masks,
        x_traits=trait_features,
        y=outcome,
    )

    return sequences, masks, trait_features, outcome
