import polars as pl
import pyarrow.parquet as pq
from tqdm import tqdm

from src.utils.static_data import TRAITS, UNITS

# Number of parquet row groups to read per chunk. Tune this to balance memory
# usage vs. overhead (matches the CNN/ViT extractors).
ROW_GROUPS_PER_CHUNK = 10

# Precomputed constants and schema for faster execution
TEAMS = ("player", "opponent")
UNIT_NAMES = tuple(UNITS.keys())
TRAIT_ITEMS = tuple((name, tuple(sorted(bps))) for name, bps in TRAITS.items())
TRAIT_MAP = {name: bps for name, bps in TRAIT_ITEMS}
TIER_MULT = (0, 1, 3, 9, 27)  # 3 ** (tier - 1) for tiers 1..4

FEATURE_KEYS = (
    [
        f"{team}_{api_name}_{t}"
        for team in TEAMS
        for api_name in UNIT_NAMES
        for t in range(1, 5)
    ]
    + [
        f"{team}_{trait}_{bp}"
        for team in TEAMS
        for trait, bps in TRAIT_ITEMS
        for bp in range(1, len(bps) + 1)
    ]
    + ["player_total_cost", "opponent_total_cost"]
)

# --- Pre-computation for Native Polars ---

# Create a DataFrame mapping units to their info
UNIT_INFO_DF = pl.DataFrame(
    [
        {"unit": api_name, "cost": info["cost"], "traits": info["traits"]}
        for api_name, info in UNITS.items()
    ]
).with_columns(pl.col("traits").cast(pl.List(pl.String)))

# Create a DataFrame mapping traits to their breakpoints
trait_bps_df = pl.DataFrame(
    [{"trait": name, "breakpoints": list(bps)} for name, bps in TRAIT_ITEMS]
)

# Create a Polars Series for TIER_MULT for use with list.gather
TIER_MULT_PL = pl.Series(TIER_MULT)


def extract_traits_one_hot(team_data: pl.DataFrame, team_name: str) -> pl.DataFrame:
    """
    Calculates one-hot encoded trait features for a single team (player or opponent).

    Args:
        team_data (pl.DataFrame): Dataframe containing exploded team data with unit info.
        team_name (str): Name of the team. Either "player" or "opponent".

    Returns:
        pl.DataFrame: One-hot encoded trait features dataframe.
    """
    trait_counts = (
        team_data.explode("traits")
        .group_by("round_idx", "traits")
        .count()
        .join(trait_bps_df, left_on="traits", right_on="trait", how="left")
    )

    trait_features_calc = (
        trait_counts.explode("breakpoints")
        .with_columns((pl.col("breakpoints") <= pl.col("count")).alias("active_bp"))
        .group_by("round_idx", "traits")
        .agg(
            pl.col("active_bp").sum().alias("breakpoint_num"),
        )
        .filter(pl.col("breakpoint_num") > 0)
    )

    trait_features = (
        trait_features_calc.with_columns(
            pl.concat_str(
                [pl.lit(team_name), pl.col("traits"), pl.col("breakpoint_num")],
                separator="_",
            ).alias("feature_name"),
            pl.lit(1).cast(pl.Int8).alias("value"),
        )
        .pivot(
            index="round_idx",
            columns="feature_name",
            values="value",
            aggregate_function="first",  # 'first' or 'max' will work
        )
        .fill_null(0)
    )

    trait_cols = [
        f"{team_name}_{trait}_{bp}"
        for trait, bps in TRAIT_ITEMS
        for bp in range(1, len(bps) + 1)
    ]

    trait_features = trait_features.with_columns(
        [
            pl.lit(0).cast(pl.Int8).alias(c)
            for c in trait_cols
            if c not in trait_features.columns
        ]
    )

    all_rounds = team_data.select("round_idx").unique()

    trait_features = all_rounds.join(
        trait_features, on="round_idx", how="left"
    ).fill_null(0)

    return trait_features


def process_team_features(base_df: pl.DataFrame, team_name: str) -> pl.DataFrame:
    """

    Calculates all features for a single team (player or opponent) using native Polars expressions.

    Args:
        base_df (pl.DataFrame): Input dataframe containing raw data.
        team_name (str): Name of the team. Either "player" or "opponent".

    Returns:
        pl.DataFrame: Features dataframe.

    """
    board_col = f"{team_name}_board"

    # 1. Explode board list and join unit info
    team_data = (
        base_df.select("round_idx", board_col)
        .explode(board_col)
        .unnest(board_col)
        .select(["round_idx", "unit", "item_ids", "loc", "tier"])
        .join(UNIT_INFO_DF, on="unit", how="left")
    )

    # 2. Calculate Total Cost
    team_cost = (
        team_data.with_columns(
            (
                pl.col("cost")
                * pl.when(pl.col("tier") == 0)
                .then(0)
                .when(pl.col("tier") == 1)
                .then(1)
                .when(pl.col("tier") == 2)
                .then(3)
                .when(pl.col("tier") == 3)
                .then(9)
                .when(pl.col("tier") == 4)
                .then(27)
                .otherwise(0)
            ).alias("unit_total_cost")
        )
        .group_by("round_idx")
        .agg(pl.col("unit_total_cost").sum().alias(f"{team_name}_total_cost"))
    )

    # 3. Calculate Unit Count features
    unit_features = (
        team_data.with_columns(
            pl.concat_str(
                [pl.lit(team_name), pl.col("unit"), pl.col("tier")], separator="_"
            ).alias("feature_name")
        )
        .pivot(
            index="round_idx",
            columns="feature_name",
            values="unit",
            aggregate_function="count",
        )
        .fill_null(0)
    )

    unit_cols = [
        f"{team_name}_{api_name}_{t}" for api_name in UNIT_NAMES for t in range(1, 5)
    ]

    unit_features = unit_features.with_columns(
        [
            pl.lit(0).cast(pl.Int8).alias(c)
            for c in unit_cols
            if c not in unit_features.columns
        ]
    )

    if len(unit_features.columns) > 1:
        unit_features = unit_features.with_columns(
            pl.all().exclude("round_idx").cast(pl.Int8)
        )

    # 4. Calculate Trait features
    trait_features = extract_traits_one_hot(team_data=team_data, team_name=team_name)

    # 5. Join all features for this team
    return (
        unit_features.join(trait_features, on="round_idx", how="inner")
        .join(team_cost, on="round_idx", how="inner")
        .fill_null(0)
    )


def _process_chunk(df: pl.DataFrame) -> pl.DataFrame | None:
    """
    Process a single chunk of raw data into the wide one-hot feature frame.

    ``round_instance`` is enumerated within each chunk (mirroring the CNN/ViT
    extractors), so the resulting ``round_idx`` is unique within the chunk, which
    is all the per-chunk player/opponent joins below require.

    Args:
        df (pl.DataFrame): A chunk of raw parquet data.

    Returns:
        pl.DataFrame | None: Frame with columns ``round_idx, outcome, timestamp,
            *FEATURE_KEYS``, or ``None`` if the chunk has no valid rows.
    """
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
        & (~pl.col("round_name").str.starts_with("1-"))
        & (pl.col("round_outcome").is_not_null())
        & (pl.all_horizontal(pl.col("board_data").struct.unnest().is_not_null()))
        & (
            pl.all_horizontal(pl.col("board_data").struct.unnest().list.len() > 0)
        )  # TODO: Double check that if one is empty it's most of the time garbage data
    )

    base_df = df.filter(mask).select(
        pl.col("match_uuid"),
        pl.col("round_idx"),
        pl.col("timestamp"),
        (pl.col("round_outcome") == "victory").cast(pl.Int8).alias("outcome"),
        pl.col("board_data").struct.unnest(),
    )

    if base_df.height == 0:
        return None

    # Process features for each team in parallel
    player_features = process_team_features(
        base_df.select("round_idx", "player_board"), "player"
    )
    opponent_features = process_team_features(
        base_df.select("round_idx", "opponent_board"), "opponent"
    )

    # Join all features together
    final_features = (
        base_df.select("round_idx", "outcome", "timestamp")
        .join(player_features, on="round_idx", how="inner")
        .join(opponent_features, on="round_idx", how="inner")
    )

    # Select in the canonical FEATURE_KEYS order and fill nulls from left joins.
    return final_features.select(
        "round_idx", "outcome", "timestamp", *FEATURE_KEYS
    ).fill_null(0)


def extract_features(raw_data_path: str, feature_path: str) -> pl.DataFrame:
    """
    Extracts and processes features from raw game data and saves them as a Parquet file.

    Reads the parquet file in chunks of row groups to keep memory usage bounded,
    even for very large datasets: the heavy nested ``board_data`` and its
    explode/pivot intermediates only ever exist for one chunk at a time. Only the
    compact (Int8-heavy) wide feature frames are accumulated across chunks.

    Args:
        raw_data_path (str): Path to the input Parquet file containing raw game data.
        feature_path (str): Path where the processed feature Parquet file will be saved.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the extracted features, sorted
            by ``timestamp`` ascending, including:
            - round_idx (str): Unique identifier for the game/round.
            - outcome (int): Binary target (1 if victory, 0 otherwise).
            - timestamp (int): Per-round event time; the chronological split key.
            - *FEATURE_KEYS: Columns representing processed features for both players and opponents.
    """
    pf = pq.ParquetFile(raw_data_path)
    n_groups = pf.metadata.num_row_groups

    group_batches = [
        list(range(start, min(start + ROW_GROUPS_PER_CHUNK, n_groups)))
        for start in range(0, n_groups, ROW_GROUPS_PER_CHUNK)
    ]

    chunk_frames: list[pl.DataFrame] = []
    for batch in tqdm(group_batches, desc="Processing chunks", unit="chunk"):
        df = pl.from_arrow(pf.read_row_groups(batch))
        result = _process_chunk(df)
        del df
        if result is not None:
            chunk_frames.append(result)

    # Concatenate the per-chunk wide frames and sort chronologically (oldest
    # first) so train_baseline can take a temporal train/test split that matches
    # the CNN/ViT datamodules (test = newest rounds).
    final_features = pl.concat(chunk_frames).sort("timestamp")

    final_features.write_parquet(feature_path)

    return final_features
