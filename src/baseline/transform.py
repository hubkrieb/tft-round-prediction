import polars as pl

from src.utils.static_data import TRAITS, UNITS

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
unit_info_df = pl.DataFrame(
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
        base_df.select("uuid", "round_id", board_col)
        .explode(board_col)
        .unnest(board_col)
        .join(unit_info_df, on="unit", how="left")
        .filter(pl.col("cost").is_not_null())  # Remove units not in our map
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
        .group_by("uuid", "round_id")
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
            index=["uuid", "round_id"],
            columns="feature_name",
            values="unit",
            aggregate_function="count",
        )
        .fill_null(0)
    )

    if len(unit_features.columns) > 1:
        unit_features = unit_features.with_columns(
            pl.all().exclude("uuid", "round_id").cast(pl.Int8)
        )

    # 4. Calculate Trait features
    trait_counts = (
        team_data.explode("traits")
        .group_by("uuid", "round_id", "traits")
        .count()
        .join(trait_bps_df, left_on="traits", right_on="trait", how="left")
    )

    trait_features_calc = (
        trait_counts.explode("breakpoints")
        .with_columns((pl.col("breakpoints") <= pl.col("count")).alias("active_bp"))
        .group_by("uuid", "round_id", "traits")  # or whatever unique ID per row
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
            index=["uuid", "round_id"],
            columns="feature_name",
            values="value",
            aggregate_function="first",  # 'first' or 'max' will work
        )
        .fill_null(0)
    )

    # 5. Join all features for this team
    return (
        unit_features.join(trait_features, on=["uuid", "round_id"], how="inner")
        .join(team_cost, on=["uuid", "round_id"], how="inner")
        .fill_null(0)
    )


def extract_features(raw_data_path: str, feature_path: str) -> pl.DataFrame:
    """
    Extracts and processes features from raw game data and saves them as a Parquet file.

    Args:
        raw_data_path (str): Path to the input Parquet file containing raw game data.
        feature_path (str): Path where the processed feature Parquet file will be saved.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the extracted features, including:
            - uuid (str): Unique identifier for the game/round.
            - round_id (int): Identifier for the round.
            - outcome (int): Binary target (1 if victory, 0 otherwise).
            - *FEATURE_KEYS: Columns representing processed features for both players and opponents.
    """
    df = pl.read_parquet(raw_data_path)

    # Filter out PVE rounds and missing data or target
    mask = (
        (~pl.col("round_name").str.starts_with("1-"))
        & (~pl.col("round_name").str.ends_with("-4"))
        & (~pl.col("round_name").str.ends_with("-7"))
        & (pl.col("round_outcome").is_not_null())
        & (pl.all_horizontal(pl.col("board_data").struct.unnest().is_not_null()))
    )

    # Select base data and unnest the boards
    base_df = df.filter(mask).select(
        pl.col("uuid"),
        pl.col("round_id").cast(pl.Int8),
        (pl.col("round_outcome") == "victory").cast(pl.Int8).alias("outcome"),
        pl.col("board_data").struct.unnest(),
    )

    # Process features for each team in parallel
    player_features = process_team_features(
        base_df.select("uuid", "round_id", "player_board"), "player"
    )
    opponent_features = process_team_features(
        base_df.select("uuid", "round_id", "opponent_board"), "opponent"
    )

    # Join all features together
    final_features = (
        base_df.select("uuid", "round_id", "outcome")
        .join(player_features, on=["uuid", "round_id"], how="inner")
        .join(opponent_features, on=["uuid", "round_id"], how="inner")
    )

    # Fill df to go from sparse to dense
    cols_to_add = set(FEATURE_KEYS) - set(final_features.columns)

    if cols_to_add:
        final_features = final_features.with_columns(
            [pl.lit(0).cast(pl.Int8).alias(c) for c in cols_to_add]
        )

    # Select in correct order and fill any remaining nulls from left joins
    final_features = final_features.select(
        "uuid", "round_id", "outcome", *FEATURE_KEYS
    ).fill_null(0)

    final_features.write_parquet(feature_path)

    return final_features
