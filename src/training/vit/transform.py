from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from tqdm import tqdm

from src.training.baseline.transform import UNIT_INFO_DF
from src.training.cnn.transform import assemble_tensors_numba, build_units_arrays
from src.training.utils.static_data import EMBLEMS, PATCHES, TRAITS
from src.training.utils.vocab import load_vocab

UNIT_VOCAB = load_vocab("data/set16/static/vocabulary/unit_vocab.json")
ITEM_VOCAB = load_vocab("data/set16/static/vocabulary/item_vocab.json")
TRAIT_VOCAB = load_vocab("data/set16/static/vocabulary/trait_vocab.json")
PATCH_VOCAB = load_vocab("data/set16/static/vocabulary/patch_vocab.json")

MAX_TRAITS = 15  # Max active traits per player to keep

# Number of parquet row groups to read per chunk. Tune this to balance
# memory usage vs. overhead.  With ~30 k rows per row group the default
# of 10 gives chunks of ~300 k rows which keeps peak RAM well under 32 GB.
ROW_GROUPS_PER_CHUNK = 10


def extract_traits_ids(team_data: pl.DataFrame, team_name: str) -> pl.DataFrame:
    """
    Calculates trait features as a list of IDs for a single team.

    Emblem items carried by units are counted towards trait totals,
    allowing them to push traits to higher breakpoints.

    Args:
        team_data (pl.DataFrame): Dataframe containing exploded team data with unit info.
        team_name (str): Name of the team. Either "player" or "opponent".

    Returns:
        pl.DataFrame: DataFrame with columns {team_name}_trait_0 to {team_name}_trait_{MAX_TRAITS-1}
                      containing trait IDs.
    """
    # Create trait breakpoints DataFrame locally
    trait_bps_df = pl.DataFrame(
        [{"trait": name, "breakpoints": list(bps)} for name, bps in TRAITS.items()]
    )

    # 1. Count units per trait for each round (from innate unit traits)
    unit_trait_counts = (
        team_data.unique(subset=["round_idx", "unit"])
        .explode("traits")
        .group_by("round_idx", "traits")
        .len()
    )

    # 2. Count bonus traits from emblem items
    emblem_map_df = pl.DataFrame(
        [{"item": item, "traits": trait} for item, trait in EMBLEMS.items()]
    )

    emblem_trait_counts = (
        team_data.select("round_idx", "item_ids")
        .with_columns(pl.col("item_ids").cast(pl.List(pl.String)))
        .explode("item_ids")
        .join(emblem_map_df, left_on="item_ids", right_on="item", how="inner")
        .group_by("round_idx", "traits")
        .len()
    )

    # 3. Combine innate and emblem trait counts
    trait_counts = (
        pl.concat([unit_trait_counts, emblem_trait_counts])
        .group_by("round_idx", "traits")
        .agg(pl.col("len").sum())
        .join(trait_bps_df, left_on="traits", right_on="trait", how="left")
    )

    # 4. Determine active breakpoints
    active_traits = (
        trait_counts.explode("breakpoints")
        .filter(pl.col("len") >= pl.col("breakpoints"))
        .group_by("round_idx", "traits")
        .agg(pl.col("breakpoints").max().alias("active_bp"))
    )

    # 5. Map to IDs
    # Create mapping DataFrame for join
    trait_map_df = pl.DataFrame(
        [
            {"traits": t.split("_")[0], "active_bp": int(t.split("_")[-1]), "id": i}
            for t, i in TRAIT_VOCAB.items()
        ]
    )

    trait_ids = (
        active_traits.join(
            trait_map_df, on=["traits", "active_bp"], how="left"
        )  # Join to get IDs
        .select("round_idx", "id")
        .filter(pl.col("id").is_not_null())
    )

    # 6. Aggregate into lists and pad
    # We want a fix width output of IDs
    trait_vectors = (
        trait_ids.sort("id")  # Sort for consistent order or any other logic
        .group_by("round_idx")
        .agg(pl.col("id").slice(0, MAX_TRAITS).alias("trait_ids"))
    )

    # Explode the list to columns
    # First ensure we have all rounds
    all_rounds = team_data.select("round_idx").unique()
    trait_vectors = all_rounds.join(trait_vectors, on="round_idx", how="left")

    # Pad with 0s
    # There isn't a direct "pad" in simple polars expressions for lists without being a bit verbose,
    # so we can use struct unpacking or list resizing.
    # A robust way is to convert to struct with fixed fields.

    cols = [
        pl.col("trait_ids")
        .list.get(i, null_on_oob=True)
        .fill_null(0)
        .alias(f"{team_name}_trait_{i}")
        for i in range(MAX_TRAITS)
    ]

    return trait_vectors.with_columns(cols).drop("trait_ids")


def build_patch_df() -> pl.DataFrame:
    """
    Build a DataFrame containing patch information.

    Returns:
        pl.DataFrame: DataFrame with the patch_id column
    """
    rows = []

    for patch_name, patch_ts in PATCHES.items():
        rows.append(
            {
                "patch": patch_name,
                "release_ts": patch_ts,
                "patch_id": PATCH_VOCAB[patch_name],
            }
        )

    patch_df = (
        pl.DataFrame(rows)
        .with_columns(pl.col("release_ts").cast(pl.Int64))
        .sort("release_ts")
    )

    return patch_df


def add_patch_ids(df: pl.DataFrame, patch_df: pl.DataFrame) -> pl.DataFrame:
    """
    Assigns patch_id to each row based on timestamp using asof join.

    Args:
        df (pl.DataFrame): DataFrame containing the game data
        patch_df (pl.DataFrame): DataFrame containing the patch information

    Returns:
        pl.DataFrame: DataFrame with the patch_id column
    """
    df = df.with_columns(pl.col("timestamp").cast(pl.Int64)).sort("timestamp")

    df = df.join_asof(
        patch_df,
        left_on="timestamp",
        right_on="release_ts",
        strategy="backward",
    )

    # Fill null patch_ids with 0 (pre-set16)
    df = df.with_columns(pl.col("patch_id").fill_null(0))

    return df


def _process_chunk(df: pl.DataFrame, patch_df: pl.DataFrame) -> dict | None:
    """
    Process a single chunk of raw data through the full feature pipeline.

    Args:
        df (pl.DataFrame): A chunk of raw parquet data.
        patch_df (pl.DataFrame): Patch lookup table.

    Returns:
        dict | None: Dictionary with keys 'tensors', 'trait_features', 'patch_ids',
            'outcome', 'round_idx', or ``None`` if the chunk has no valid rows.
    """
    df = add_patch_ids(df, patch_df)

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
        pl.col("patch_id"),
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

    player_traits = extract_traits_ids(team_data=player_data, team_name="player")
    opponent_traits = extract_traits_ids(team_data=opponent_data, team_name="opponent")
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
    patch_ids = np.array(base_pd["patch_id"].astype(int))
    timestamps = np.array(base_pd["timestamp"].astype("int64"))
    round_idx = base_pd["round_idx"].to_numpy()

    units_all, counts = build_units_arrays(base_pd, UNIT_VOCAB, ITEM_VOCAB)
    del base_pd

    tensors = assemble_tensors_numba(units_all, counts, units_all.shape[0])
    del units_all, counts

    return {
        "tensors": tensors,
        "trait_features": trait_features,
        "patch_ids": patch_ids,
        "timestamps": timestamps,
        "outcome": outcome,
        "round_idx": round_idx,
    }


def extract_tensors(
    raw_data_path: str, feature_path: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts and processes features from raw game data and saves them as per-array .npy files.

    Reads the parquet file in chunks of row groups to keep memory usage
    bounded, even for very large datasets.

    Args:
        raw_data_path (str): Path to the input Parquet file containing raw game data.
        feature_path (str): Directory where the processed feature .npy files will be saved
            (x_units.npy, x_traits.npy, x_patch.npy, timestamp.npy, y.npy, round_idx.npy).
            Storing each array in its own .npy file lets the dataloader memory-map them at
            training time; a single .npz bundles them in a zip and cannot be mmap'd.
            timestamp.npy holds the per-row event time the datamodule sorts on to build a
            chronological train/val/test split.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The board tensors,
            trait features, patch ids and round outcomes.

    """
    patch_df = build_patch_df()
    pf = pq.ParquetFile(raw_data_path)
    n_groups = pf.metadata.num_row_groups

    # Build list of row-group index batches
    group_batches = [
        list(range(start, min(start + ROW_GROUPS_PER_CHUNK, n_groups)))
        for start in range(0, n_groups, ROW_GROUPS_PER_CHUNK)
    ]

    all_tensors: list[np.ndarray] = []
    all_traits: list[np.ndarray] = []
    all_patches: list[np.ndarray] = []
    all_timestamps: list[np.ndarray] = []
    all_outcomes: list[np.ndarray] = []
    all_round_idx: list[np.ndarray] = []

    for batch in tqdm(group_batches, desc="Processing chunks", unit="chunk"):
        table = pf.read_row_groups(batch)
        df = pl.from_arrow(table)
        assert isinstance(df, pl.DataFrame)  # from_arrow on a Table, never a Series
        del table

        result = _process_chunk(df, patch_df)
        del df

        if result is None:
            continue

        all_tensors.append(result["tensors"])
        all_traits.append(result["trait_features"])
        all_patches.append(result["patch_ids"])
        all_timestamps.append(result["timestamps"])
        all_outcomes.append(result["outcome"])
        all_round_idx.append(result["round_idx"])
        del result

    tensors = np.concatenate(all_tensors)
    trait_features = np.concatenate(all_traits)
    patch_ids = np.concatenate(all_patches)
    timestamps = np.concatenate(all_timestamps)
    outcome = np.concatenate(all_outcomes)
    round_idx = np.concatenate(all_round_idx)
    del (
        all_tensors,
        all_traits,
        all_patches,
        all_timestamps,
        all_outcomes,
        all_round_idx,
    )

    out_dir = Path(feature_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "x_units.npy", tensors)
    np.save(out_dir / "x_traits.npy", trait_features)
    np.save(out_dir / "x_patch.npy", patch_ids)
    np.save(out_dir / "timestamp.npy", timestamps)
    np.save(out_dir / "y.npy", outcome)
    np.save(out_dir / "round_idx.npy", round_idx)

    sort_features_by_timestamp(feature_path)

    return tensors, trait_features, patch_ids, outcome


_FEATURE_ARRAY_NAMES = ("x_units", "x_traits", "x_patch", "timestamp", "y", "round_idx")


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
        allow_pickle = name == "round_idx"  # object-dtype string array
        arr = np.load(path, allow_pickle=allow_pickle)
        np.save(path, arr[order], allow_pickle=allow_pickle)
