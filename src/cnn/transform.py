import numpy as np
import pandas as pd
from numba import njit, prange

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


def loc_to_rc(loc_str: str) -> tuple[int, int]:
    """
    Convert loc strings like "A2", "A_2", "D_7" to (row, col) indices.

    Assumes rows A-D -> 0-3, cols 1-7 -> 0-6.

    Args:
        loc_str (str): String representig the location of a unit

    Returns:
        tuple[int, int]: The (x, y) position of the unit
    """
    if loc_str is None:
        return -1, -1
    s = str(loc_str).replace("_", "").strip()
    if len(s) < 2:
        return -1, -1
    row_char = s[0].upper()
    col_part = s[1:]
    row_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    if row_char not in row_map:
        return -1, -1
    try:
        col = int(col_part) - 1
    except Exception:
        return -1, -1
    row = row_map[row_char]
    if not (0 <= row < ROWS and 0 <= col < COLS):
        return -1, -1
    return row, col


def build_units_arrays(
    board_series: pd.Series,
    unit_to_id: dict[str, int],
    item_to_id: dict[str, int],
    max_units: int = MAX_UNITS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts a pandas Series of board_data dicts into two numpy arrays.

    Each unit record has fields: r, c, unit_id, tier, item1, item2, item3, is_player

    Args:
        board_series (pd.Series): Series containing the board information
        unit_to_id (dict[str, int]): Dictionary mapping unit names to their id
        item_to_id (dict[str, int]): Dictionary mapping unit items to their id
        max_units (int): The maximum amount of units contained

    Returns:
        tuple[np.ndarray, np.ndarray]: The units' arrays and the unit counts for each board.
    """
    n = len(board_series)
    units_all = np.zeros((n, max_units, FIELDS), dtype=np.int32)
    counts = np.zeros(n, dtype=np.int32)

    for i, bd in enumerate(board_series):
        if bd is None:
            counts[i] = 0
            continue
        k = 0
        for side_name, is_player in (("player_board", 1), ("opponent_board", 0)):
            arr = bd.get(side_name, None)
            if arr is None:
                continue
            for rec in arr:
                if k >= max_units:
                    # drop overflow units (should be rare if max_units is set high)
                    break
                loc = rec.get("loc", None)

                r, c = loc_to_rc(loc)
                if r < 0:
                    # skip invalid locs
                    continue
                unit_name = rec.get("unit", None)
                unit_id = unit_to_id.get(unit_name, 0)
                tier = rec.get("tier", 0) or 0
                item_ids = rec.get("item_ids", None)
                it1 = it2 = it3 = 0
                if item_ids is not None:
                    it_list = list(item_ids)
                    if len(it_list) > 0 and it_list[0] is not None:
                        it1 = item_to_id.get(it_list[0], 0)
                    if len(it_list) > 1 and it_list[1] is not None:
                        it2 = item_to_id.get(it_list[1], 0)
                    if len(it_list) > 2 and it_list[2] is not None:
                        it3 = item_to_id.get(it_list[2], 0)
                units_all[i, k, R_IDX] = r
                units_all[i, k, C_IDX] = c
                units_all[i, k, UNIT_IDX] = unit_id
                units_all[i, k, TIER_IDX] = tier
                units_all[i, k, ITEM1_IDX] = it1
                units_all[i, k, ITEM2_IDX] = it2
                units_all[i, k, ITEM3_IDX] = it3
                units_all[i, k, IS_PLAYER_IDX] = is_player
                k += 1
        counts[i] = k
    return units_all, counts


@njit(parallel=True)
def assemble_tensors_numba(
    units_all: np.ndarray, counts: np.ndarray, n_samples: int
) -> np.ndarray:
    """
    Assemble the board data into a single array.

    Args:
        units_all (np.ndarray): The board data
        counts (np.ndarray): The unit counts for each board
        n_samples (int): The amount of rounds/boards in units_all

    Returns:
        np.ndarray: The assembled board data
    """
    tensors = np.zeros((n_samples, CHANNELS, 2 * ROWS, COLS), dtype=np.int32)
    for i in prange(n_samples):
        cnt = counts[i]
        for k in range(cnt):
            r = units_all[i, k, R_IDX]
            c = units_all[i, k, C_IDX]
            unit_id = units_all[i, k, UNIT_IDX]
            tier = units_all[i, k, TIER_IDX]
            it1 = units_all[i, k, ITEM1_IDX]
            it2 = units_all[i, k, ITEM2_IDX]
            it3 = units_all[i, k, ITEM3_IDX]
            is_player = units_all[i, k, IS_PLAYER_IDX]
            if is_player == 1:
                tensors[i, 0, 4 + r, c] = unit_id
                tensors[i, 1, 4 + r, c] = tier
                tensors[i, 3 - 1, 4 + r, c] = it1
                tensors[i, 4 - 1, 4 + r, c] = it2
                tensors[i, 5 - 1, 4 + r, c] = it3
            else:
                tensors[i, 0, 3 - r, 6 - c] = unit_id
                tensors[i, 1, 3 - r, 6 - c] = tier
                tensors[i, 3 - 1, 3 - r, 6 - c] = it1
                tensors[i, 4 - 1, 3 - r, 6 - c] = it2
                tensors[i, 5 - 1, 3 - r, 6 - c] = it3
    return tensors


def extract_tensors(
    raw_data_path: str, feature_path: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts and processes features from raw game data and saves them as a .npz file.

    Args:
        raw_data_path (str): Path to the input Parquet file containing raw game data.
        feature_path (str): Path where the processed feature .npz file will be saved.

    Returns:
        tuple[np.ndarray, np.ndarray]: The board tensors and the round outcomes.

    """
    df = pd.read_parquet(raw_data_path)

    mask = (
        (~df["round_name"].str.startswith("1-", na=True))
        & (~df["round_name"].str.endswith("-4", na=True))
        & (~df["round_name"].str.endswith("-7", na=True))
        & (df["round_outcome"].notnull())
        & (pd.json_normalize(df["board_data"]).notnull().all(axis=1))
    )

    df = df[mask]

    outcome = np.array((df["round_outcome"] == "victory").astype(int))

    units_all, counts = build_units_arrays(df["board_data"], UNIT_TO_ID, ITEM_TO_ID)

    tensors = assemble_tensors_numba(units_all, counts, units_all.shape[0])

    np.savez_compressed(feature_path, x=tensors, y=outcome)

    return tensors, outcome


if __name__ == "__main__":
    extract_tensors("data/set15/raw/1758831215577.parquet")
