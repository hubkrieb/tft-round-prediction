from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
from numba import njit, prange
from tqdm import tqdm

from src.training.baseline.transform import UNIT_INFO_DF, extract_traits_one_hot
from src.training.utils.vocab import load_vocab

UNIT_VOCAB = load_vocab("data/set16/static/vocabulary/unit_vocab.json")
ITEM_VOCAB = load_vocab("data/set16/static/vocabulary/item_vocab.json")

MAX_UNITS = 24
CHANNELS = 5
ROWS = 4
COLS = 7

# Number of parquet row groups to read per chunk. Tune this to balance memory usage vs. overhead.
ROW_GROUPS_PER_CHUNK = 10

R_IDX = 0
C_IDX = 1
UNIT_IDX = 2
TIER_IDX = 3
ITEM1_IDX = 4
ITEM2_IDX = 5
ITEM3_IDX = 6
IS_PLAYER_IDX = 7
FIELDS = 8


def loc_to_rc(loc_str: str | None) -> tuple[int, int]:
    """
    Convert loc strings like "A2", "A_2", "D_7" to (row, col) indices.

    Assumes rows A-D -> 0-3, cols 1-7 -> 0-6.

    Args:
        loc_str (str | None): String representig the location of a unit

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
    board_df: pd.DataFrame,
    unit_to_id: dict[str, int],
    item_to_id: dict[str, int],
    max_units: int = MAX_UNITS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts a pandas DataFrame containing board_data columns into two numpy arrays.

    Each unit record has fields: r, c, unit_id, tier, item1, item2, item3, is_player

    Args:
        board_df (pd.DataFrame): DataFrame containing the board information
        unit_to_id (dict[str, int]): Dictionary mapping unit names to their id
        item_to_id (dict[str, int]): Dictionary mapping unit items to their id
        max_units (int): The maximum amount of units contained

    Returns:
        tuple[np.ndarray, np.ndarray]: The units' arrays and the unit counts for each board.
    """
    n = len(board_df)
    units_all = np.zeros((n, max_units, FIELDS), dtype=np.int32)
    counts = np.zeros(n, dtype=np.int32)

    for i, bd in tqdm(
        board_df.iterrows(), total=n, desc="Building unit arrays", leave=False
    ):
        if bd is None:
            counts[i] = 0
            continue
        k = 0
        for side_name, is_player in (("player_board", 1), ("opponent_board", 0)):
            arr = bd[side_name]
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
                unit_id = unit_to_id.get(unit_name, None)
                if unit_id is None:
                    continue
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
    for i in prange(n_samples):  # ty: ignore[not-iterable]
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


def _process_chunk(df: pl.DataFrame) -> dict | None:
    """
    Process a single chunk of raw data through the full feature pipeline.

    Args:
        df (pl.DataFrame): A chunk of raw parquet data.

    Returns:
        dict | None: Dictionary with keys 'tensors', 'trait_features',
            'timestamps', 'outcome', or ``None`` if the chunk has no valid rows.
    """
    df = df.with_columns(
        pl.arange(0, pl.len())
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
        & (~pl.col("round_name").str.starts_with("1-"))
        & (pl.col("round_outcome").is_not_null())
        & (pl.all_horizontal(pl.col("board_data").struct.unnest().is_not_null()))
        & (pl.all_horizontal(pl.col("board_data").struct.unnest().list.len() > 0))
    )

    base_df = df.filter(mask).select(
        pl.col("match_uuid"),
        pl.col("round_idx"),
        pl.col("timestamp"),
        (pl.col("round_outcome") == "victory").cast(pl.Int8).alias("outcome"),
        pl.col("board_data").struct.unnest(),
    )
    del df

    if base_df.height == 0:
        return None

    # --- trait features ---
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
    del player_data, opponent_data

    trait_features = (
        player_traits.join(opponent_traits, on="round_idx", how="inner")
        .select(pl.all().exclude("round_idx"))
        .to_numpy()
    )
    del player_traits, opponent_traits

    # --- board tensors ---
    base_pd = base_df.to_pandas()
    del base_df

    outcome = np.array(base_pd["outcome"].astype(int))
    timestamps = np.array(base_pd["timestamp"].astype("int64"))

    units_all, counts = build_units_arrays(base_pd, UNIT_VOCAB, ITEM_VOCAB)
    del base_pd

    tensors = assemble_tensors_numba(units_all, counts, units_all.shape[0])
    del units_all, counts

    return {
        "tensors": tensors,
        "trait_features": trait_features,
        "timestamps": timestamps,
        "outcome": outcome,
    }


def extract_tensors(
    raw_data_path: str, feature_path: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts and processes features from raw game data and saves them as per-array .npy files.

    Reads the parquet file in chunks of row groups to keep memory usage
    bounded, even for very large datasets.

    Args:
        raw_data_path (str): Path to the input Parquet file containing raw game data.
        feature_path (str): Directory where the processed feature .npy files will be saved
            (x_units.npy, x_traits.npy, timestamp.npy, y.npy). Storing each array in its
            own .npy file lets the dataloader memory-map them at training time; a single
            .npz bundles them in a zip and cannot be mmap'd. timestamp.npy holds the
            per-row event time the datamodule sorts on to build a chronological
            train/val/test split.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The board tensors, trait features and
            round outcomes.

    """
    pf = pq.ParquetFile(raw_data_path)
    n_groups = pf.metadata.num_row_groups

    # Build list of row-group index batches
    group_batches = [
        list(range(start, min(start + ROW_GROUPS_PER_CHUNK, n_groups)))
        for start in range(0, n_groups, ROW_GROUPS_PER_CHUNK)
    ]

    all_tensors: list[np.ndarray] = []
    all_traits: list[np.ndarray] = []
    all_timestamps: list[np.ndarray] = []
    all_outcomes: list[np.ndarray] = []

    for batch in tqdm(group_batches, desc="Processing chunks", unit="chunk"):
        table = pf.read_row_groups(batch)
        df = pl.from_arrow(table)
        assert isinstance(df, pl.DataFrame)  # from_arrow on a Table, never a Series
        del table

        result = _process_chunk(df)
        del df

        if result is None:
            continue

        all_tensors.append(result["tensors"])
        all_traits.append(result["trait_features"])
        all_timestamps.append(result["timestamps"])
        all_outcomes.append(result["outcome"])
        del result

    tensors = np.concatenate(all_tensors)
    trait_features = np.concatenate(all_traits)
    timestamps = np.concatenate(all_timestamps)
    outcome = np.concatenate(all_outcomes)
    del all_tensors, all_traits, all_timestamps, all_outcomes

    out_dir = Path(feature_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "x_units.npy", tensors)
    np.save(out_dir / "x_traits.npy", trait_features)
    np.save(out_dir / "timestamp.npy", timestamps)
    np.save(out_dir / "y.npy", outcome)

    sort_features_by_timestamp(feature_path)

    return tensors, trait_features, outcome


_FEATURE_ARRAY_NAMES = ("x_units", "x_traits", "timestamp", "y")


def sort_features_by_timestamp(feature_path: str) -> None:
    """Reorder every per-row .npy in the feature dir to be globally timestamp-ascending.

    Extraction only sorts rows within each row-group chunk, so the saved arrays
    are not globally chronological. The datamodule's chronological split then
    has to argsort by timestamp at setup, and the resulting train/val/test row
    sets are scattered all over the file. A shuffled batch fans out random
    memmap reads across the whole array, which makes training extremely slow
    once the data is bigger than the OS page cache. Sorting once on disk makes
    each split a contiguous range, restoring cache-friendly access.

    All arrays in :data:`_FEATURE_ARRAY_NAMES` are permuted by the same order,
    which preserves whatever per-row alignment already exists between them.
    No-op if the file is already sorted.
    """
    out = Path(feature_path)
    ts = np.load(out / "timestamp.npy")
    if ts.size > 1 and bool(np.all(ts[:-1] <= ts[1:])):
        return
    order = np.argsort(ts, kind="stable")
    for name in _FEATURE_ARRAY_NAMES:
        path = out / f"{name}.npy"
        if not path.exists():
            continue
        arr = np.load(path)
        np.save(path, arr[order])
