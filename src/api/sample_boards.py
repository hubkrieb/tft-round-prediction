"""Extract real matchups from the raw parquet into ``src/web/sample_boards.json``.

The board builder UI offers a "random board" button that loads one of these
pre-saved boards so the models can be tested on realistic positions instead of
hand-assembled ones. Boards are sampled from real PVP rounds, spread across
game stages, and only kept when every unit and item is present in the UI
catalog (so all icons resolve).
"""

from __future__ import annotations

import json
import random
import re

import polars as pl

from src.api.config import CATALOG_PATH, MAX_ITEMS, WEB_DIR, resolve

DEFAULT_RAW_PATH = "data/set16/raw/merged_data.parquet"
SAMPLE_BOARDS_PATH = WEB_DIR / "data" / "sample_boards.json"

_LOC_RE = re.compile(r"^[A-D][1-7]$")


def _convert_side(
    raw_units: list[dict], units_ok: set[str], items_ok: set[str]
) -> list[dict] | None:
    """Convert one side's raw ``board_data`` units to UI records.

    Records use the same own-frame coordinates as :class:`src.api.schema.PlacedUnit`
    (row 0 = that side's frontline). Returns ``None`` when any unit or item is
    missing from the catalog, so the board is skipped rather than silently
    altered.
    """
    out = []
    for u in raw_units:
        # The merged raw data mixes two loc encodings ("A1" and "A_1").
        loc = (u.get("loc") or "").replace("_", "")
        if not _LOC_RE.match(loc):
            return None  # unplaced/bench unit: not a clean board snapshot
        if u["unit"] not in units_ok:
            return None
        items = [i for i in (u.get("item_ids") or []) if i][:MAX_ITEMS]
        if any(i not in items_ok for i in items):
            return None
        out.append(
            {
                "unit": u["unit"],
                "tier": min(max(int(u.get("tier") or 1), 1), 4),
                "items": items,
                "row": ord(loc[0]) - ord("A"),
                "col": int(loc[1]) - 1,
            }
        )
    return out


def extract_sample_boards(
    raw_path: str = DEFAULT_RAW_PATH,
    out_path: str | None = None,
    per_stage: int = 4,
    min_units: int = 6,
    seed: int = 16,
) -> int:
    """Sample real PVP boards from the raw data and write the UI JSON.

    Args:
        raw_path: Raw rounds parquet (``board_data`` struct column).
        out_path: Output JSON path; defaults to ``src/web/sample_boards.json``.
        per_stage: Boards to keep per game stage (stages 2..7), each from a
            distinct match.
        min_units: Minimum units on *each* side for a board to qualify.
        seed: RNG seed so the published sample set is reproducible.

    Returns:
        The number of boards written.
    """
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    units_ok = {u["apiName"] for u in catalog["units"] if u.get("hasIcon")}
    items_ok = {i["apiName"] for i in catalog["items"]}

    df = (
        pl.scan_parquet(resolve(raw_path))
        .filter(
            (pl.col("round_type") == "PVP")
            & pl.col("round_outcome").is_in(["victory", "defeat"])
            & (
                pl.col("board_data").struct.field("player_board").list.len()
                >= min_units
            )
            & (
                pl.col("board_data").struct.field("opponent_board").list.len()
                >= min_units
            )
        )
        .select("match_uuid", "round_name", "round_outcome", "board_data")
        .collect()
    )

    rng = random.Random(seed)
    rows = df.to_dicts()
    rng.shuffle(rows)

    by_stage: dict[str, list[dict]] = {}
    used_matches: set[str] = set()
    for row in rows:
        stage = (row["round_name"] or "").split("-")[0]
        if not stage.isdigit():
            continue
        bucket = by_stage.setdefault(stage, [])
        if len(bucket) >= per_stage or row["match_uuid"] in used_matches:
            continue
        player = _convert_side(row["board_data"]["player_board"], units_ok, items_ok)
        opponent = _convert_side(
            row["board_data"]["opponent_board"], units_ok, items_ok
        )
        if player is None or opponent is None:
            continue
        used_matches.add(row["match_uuid"])
        bucket.append(
            {
                "stage": row["round_name"],
                "outcome": row["round_outcome"],
                "player": player,
                "opponent": opponent,
            }
        )

    boards = [b for stage in sorted(by_stage) for b in by_stage[stage]]
    out = resolve(out_path) if out_path else SAMPLE_BOARDS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"boards": boards}, indent=1), encoding="utf-8")
    return len(boards)
