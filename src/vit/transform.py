from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from tqdm import tqdm

from src.baseline.transform import UNIT_INFO_DF
from src.cnn.transform import assemble_tensors_numba, build_units_arrays
from src.utils.static_data import EMBLEMS, PATCHES, TRAITS
from src.utils.vocab import load_vocab

UNIT_VOCAB = load_vocab("data/set16/static/vocabulary/unit_vocab.json")
ITEM_VOCAB = load_vocab("data/set16/static/vocabulary/item_vocab.json")
TRAIT_VOCAB = load_vocab("data/set16/static/vocabulary/trait_vocab.json")
PATCH_VOCAB = load_vocab("data/set16/static/vocabulary/patch_vocab.json")

MAX_TRAITS = 15  # Max active traits per player to keep

# Number of parquet row groups to read per chunk in the legacy chunked-read
# helpers (backfill_timestamps and similar tooling that rebuilds round_idx
# against already-saved feature dirs). extract_tensors no longer uses this; it
# loads the whole raw frame, sorts globally, and then chunks by row count.
ROW_GROUPS_PER_CHUNK = 10

# Row-count batch size for the tensor-building loop inside extract_tensors.
# Chosen so the heavy intermediate arrays (units_all, the (B,5,8,7) board grid)
# stay bounded; the input dataframe is already fully in RAM by this point.
CHUNK_ROWS = 300_000


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


def _process_chunk(df: pl.DataFrame) -> dict | None:
    """Build tensors/traits for one chunk of an already-prepared dataframe.

    Expects ``df`` to already carry the columns added by :func:`extract_tensors`
    upstream — ``timestamp`` (sorted), ``patch_id``, and ``round_idx`` — so this
    function only filters out invalid rows and builds the per-chunk tensors.

    Returns ``None`` if the chunk has no surviving rows after the mask.
    """
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
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-row board features from raw game data and save them as .npy files.

    The raw frame is loaded fully, sorted by ``timestamp`` once, and then the
    patch_id (asof join against the patch table) and ``round_idx`` (a unique row
    id built from ``match_uuid``/``round_name``/within-group ``round_instance``)
    are computed *globally* on the sorted frame. Doing the sort up front (the
    raw parquet is much smaller than the resulting feature arrays — ~1 GB vs
    ~7 GB for set16) means:

    - the saved arrays come out globally time-ordered by construction, so the
      datamodule's chronological split is a contiguous slice and no
      post-extraction array permutation is needed; and
    - ``round_instance`` enumerates within each ``(match, player, round_name)``
      group across the whole dataset rather than per-chunk, which removes the
      (previously latent) risk of a group split across chunk boundaries
      producing collisions.

    Args:
        raw_data_path (str): Path to the input Parquet file containing raw game data.
        feature_path (str): Directory where the processed feature .npy files will be saved
            (x_units.npy, x_traits.npy, x_patch.npy, timestamp.npy, y.npy, round_idx.npy).
            Storing each array in its own .npy file lets the dataloader memory-map them at
            training time; a single .npz bundles them in a zip and cannot be mmap'd.
            timestamp.npy holds the per-row event time the datamodule sorts on to build a
            chronological train/val/test split.

    Returns:
        tuple[np.ndarray, np.ndarray]: The board tensors and the round outcomes.

    """
    patch_df = build_patch_df()

    # Global pre-sort + global round_idx assignment. add_patch_ids both sorts by
    # timestamp and adds patch_id; round_instance is then a window enumeration
    # over the whole frame so it (and round_idx) are chunk-boundary free.
    df_all = add_patch_ids(pl.read_parquet(raw_data_path), patch_df)
    df_all = df_all.with_columns(
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

    n_rows = df_all.height
    n_chunks = (n_rows + CHUNK_ROWS - 1) // CHUNK_ROWS

    all_tensors: list[np.ndarray] = []
    all_traits: list[np.ndarray] = []
    all_patches: list[np.ndarray] = []
    all_timestamps: list[np.ndarray] = []
    all_outcomes: list[np.ndarray] = []
    all_round_idx: list[np.ndarray] = []

    for i in tqdm(range(n_chunks), desc="Processing chunks", unit="chunk"):
        result = _process_chunk(df_all.slice(i * CHUNK_ROWS, CHUNK_ROWS))
        if result is None:
            continue

        all_tensors.append(result["tensors"])
        all_traits.append(result["trait_features"])
        all_patches.append(result["patch_ids"])
        all_timestamps.append(result["timestamps"])
        all_outcomes.append(result["outcome"])
        all_round_idx.append(result["round_idx"])
        del result

    del df_all

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

    return tensors, trait_features, patch_ids, outcome


def sort_features_by_timestamp(feature_path: str) -> None:
    """Reorder every per-row .npy in the feature dir to be globally timestamp-ascending.

    Migration helper for feature dirs produced before :func:`extract_tensors`
    started pre-sorting the raw data globally — those dirs were sorted only
    within each extraction chunk, so the datamodule's chronological split came
    out as a scattered set of indices across the array and shuffled batches
    fanned out random memmap reads. Permuting every array uniformly by the
    saved timestamps restores cache-friendly contiguous slices. No-op if the
    file is already sorted, so it's safe to run on freshly extracted data too.
    """
    out = Path(feature_path)
    ts = np.load(out / "timestamp.npy")
    if ts.size > 1 and bool(np.all(ts[:-1] <= ts[1:])):
        return
    order = np.argsort(ts, kind="stable")
    for name in ("x_units", "x_traits", "x_patch", "timestamp", "y", "round_idx"):
        path = out / f"{name}.npy"
        if not path.exists():
            continue
        allow_pickle = name == "round_idx"  # object-dtype string array
        arr = np.load(path, allow_pickle=allow_pickle)
        np.save(path, arr[order], allow_pickle=allow_pickle)


def backfill_timestamps(feature_path: str, raw_data_path: str) -> np.ndarray:
    """Write timestamp.npy for a feature dir extracted before timestamps were saved.

    Reconstructs ``round_idx`` from the raw parquet using the **legacy per-chunk
    procedure** (the procedure that produced the saved ``round_idx.npy``) and
    looks up each saved feature row's timestamp by ``round_idx`` so the result
    is aligned to the existing arrays. Uses :data:`ROW_GROUPS_PER_CHUNK` to match
    the chunk boundaries the original extraction used; this is intentionally a
    separate code path from the current :func:`extract_tensors` (which now does
    a single global sort) — the legacy-compatible chunked rebuild guarantees the
    round_idx strings line up with what was previously saved.

    Args:
        feature_path (str): Feature dir containing round_idx.npy (gets timestamp.npy).
        raw_data_path (str): Raw parquet the features were extracted from.

    Returns:
        np.ndarray: The per-row timestamps written to ``timestamp.npy``.
    """
    out_dir = Path(feature_path)
    ri = np.load(out_dir / "round_idx.npy", allow_pickle=True).astype(str)

    patch_df = build_patch_df()
    pf = pq.ParquetFile(raw_data_path)
    n_groups = pf.metadata.num_row_groups
    cols = ["match_uuid", "player_uuid", "round_name", "timestamp"]

    frames = []
    for start in range(0, n_groups, ROW_GROUPS_PER_CHUNK):
        batch = list(range(start, min(start + ROW_GROUPS_PER_CHUNK, n_groups)))
        df = add_patch_ids(
            pl.from_arrow(pf.read_row_groups(batch, columns=cols)), patch_df
        )
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
        frames.append(df.select("round_idx", "timestamp"))

    lut = pl.concat(frames).unique(subset="round_idx", keep="first")
    aligned = pl.DataFrame({"round_idx": ri}).join(lut, on="round_idx", how="left")

    ts = aligned["timestamp"].to_numpy()
    missing = int(np.count_nonzero(~np.isfinite(ts.astype("float64"))))
    if missing:
        raise ValueError(
            f"{missing} feature rows had no matching round_idx in raw data"
        )

    timestamps = ts.astype("int64")
    np.save(out_dir / "timestamp.npy", timestamps)
    return timestamps
