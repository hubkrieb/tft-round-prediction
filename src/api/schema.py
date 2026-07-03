"""Request/response models for the inference API and the internal board type."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.api.config import BOARD_COLS, BOARD_ROWS, MAX_ITEMS


class PlacedUnit(BaseModel):
    """A single champion placed on a board cell.

    Attributes:
        unit: Champion apiName, e.g. ``"TFT16_Tristana"``.
        tier: Star level (1-3); also the value the models embed as the unit tier.
        items: Up to three item apiNames held by the unit.
        row: Board row index, 0-3 (top-to-bottom of that side).
        col: Board column index, 0-6 (left-to-right).
    """

    unit: str
    tier: int = Field(default=1, ge=1, le=4)
    items: list[str] = Field(default_factory=list)
    row: int = Field(ge=0, lt=BOARD_ROWS)
    col: int = Field(ge=0, lt=BOARD_COLS)

    def loc(self) -> str:
        """Return the raw-data location string for this cell, e.g. ``"A1"``."""
        return f"{chr(ord('A') + self.row)}{self.col + 1}"

    def record(self) -> dict:
        """Return the unit as a raw ``board_data`` record understood by the transforms."""
        return {
            "unit": self.unit,
            "tier": int(self.tier),
            "item_ids": [i for i in self.items[:MAX_ITEMS] if i],
            "loc": self.loc(),
        }


class BoardState(BaseModel):
    """A full matchup: the player's board versus the opponent's board."""

    player: list[PlacedUnit] = Field(default_factory=list)
    opponent: list[PlacedUnit] = Field(default_factory=list)


class PredictRequest(BoardState):
    """A prediction request: a board plus the model to score it with."""

    model: str = Field(default="vit", description="One of: vit, cnn, xgboost")


class PredictResponse(BaseModel):
    """Prediction result for a matchup."""

    model: str
    win_probability: float
    prediction: str
