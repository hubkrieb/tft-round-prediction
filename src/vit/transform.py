import numpy as np
import polars as pl

from src.baseline.transform import UNIT_INFO_DF
from src.cnn.transform import assemble_tensors_numba, build_units_arrays
from src.utils.static_data import EMBLEMS, PATCHES, TRAITS
from src.utils.vocab import load_vocab

UNIT_VOCAB = load_vocab("data/set16/static/vocabulary/unit_vocab.json")
ITEM_VOCAB = load_vocab("data/set16/static/vocabulary/item_vocab.json")
TRAIT_VOCAB = load_vocab("data/set16/static/vocabulary/trait_vocab.json")
PATCH_VOCAB = load_vocab("data/set16/static/vocabulary/patch_vocab.json")

MAX_TRAITS = 15  # Max active traits per player to keep


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
    df = pl.read_parquet(raw_data_path)

    patch_df = build_patch_df()
    df = add_patch_ids(df, patch_df)

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
        & (pl.all_horizontal(pl.col("board_data").struct.unnest().list.len() > 0))
    )

    base_df = df.filter(mask).select(
        pl.col("match_uuid"),
        pl.col("round_idx"),
        pl.col("patch_id"),
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

    player_traits = extract_traits_ids(team_data=player_data, team_name="player")
    opponent_traits = extract_traits_ids(team_data=opponent_data, team_name="opponent")

    trait_features = (
        player_traits.join(opponent_traits, on="round_idx", how="inner")
        .select(pl.all().exclude("round_idx"))
        .to_numpy()
    )

    base_df = base_df.to_pandas()

    outcome = np.array((base_df["outcome"]).astype(int))
    patch_ids = np.array(base_df["patch_id"].astype(int))

    units_all, counts = build_units_arrays(base_df, UNIT_VOCAB, ITEM_VOCAB)

    tensors = assemble_tensors_numba(units_all, counts, units_all.shape[0])

    np.savez_compressed(
        feature_path,
        x_units=tensors,
        x_traits=trait_features,
        x_patch=patch_ids,
        y=outcome,
        round_idx=base_df["round_idx"].to_numpy(),
    )

    return tensors, trait_features, patch_ids, outcome
