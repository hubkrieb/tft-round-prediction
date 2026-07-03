"""Unified predictors that turn a :class:`BoardState` into a win probability.

Three backends share one interface (:meth:`Predictor.predict_proba`):

* :class:`VitPredictor`  - loads a ViT ``.ckpt`` (board tensor + trait IDs).
* :class:`CnnPredictor`  - loads a CNN ``.ckpt`` (board tensor + one-hot traits).
* :class:`XgbPredictor`  - loads a saved XGBoost model (wide one-hot features).

Models are loaded lazily and cached by (kind, path) so the API pays the load
cost once. All default paths resolve against the repo root.
"""

from __future__ import annotations

import json
from functools import cache
from typing import TYPE_CHECKING

import torch

from src.api import config
from src.api import featurize as F

if TYPE_CHECKING:
    from src.api.schema import BoardState


class Predictor:
    """Common interface for the three model backends."""

    def predict_proba(self, board: BoardState) -> float:
        """Return the player's win probability in ``[0, 1]``."""
        raise NotImplementedError


class VitPredictor(Predictor):
    """Vision-Transformer backend."""

    def __init__(self, ckpt_path: str, device: str = "cpu") -> None:
        from src.training.vit.model import TFTViT

        self.device = device
        self.model = (
            TFTViT.load_from_checkpoint(
                str(config.resolve(ckpt_path)), map_location=device
            )
            .to(device)
            .eval()
        )

    @torch.no_grad()
    def predict_proba(self, board: BoardState) -> float:
        """Return the ViT win probability for the matchup."""
        x_units = torch.from_numpy(F.board_tensor(board)).long().to(self.device)
        x_traits = torch.from_numpy(F.vit_trait_ids(board)).long().to(self.device)
        logit = self.model(x_units, x_traits)
        return float(torch.sigmoid(logit).item())


class CnnPredictor(Predictor):
    """Convolutional backend."""

    def __init__(self, ckpt_path: str, device: str = "cpu") -> None:
        from src.training.cnn.model import TFTCNN

        self.device = device
        self.model = (
            TFTCNN.load_from_checkpoint(
                str(config.resolve(ckpt_path)), map_location=device
            )
            .to(device)
            .eval()
        )

    @torch.no_grad()
    def predict_proba(self, board: BoardState) -> float:
        """Return the CNN win probability for the matchup."""
        x_units = torch.from_numpy(F.board_tensor(board)).long().to(self.device)
        x_traits = torch.from_numpy(F.cnn_trait_onehot(board)).float().to(self.device)
        # TFTCNN.forward already applies the sigmoid and returns a win probability.
        return float(self.model(x_units, x_traits).item())


class XgbPredictor(Predictor):
    """XGBoost baseline backend."""

    def __init__(self, model_path: str) -> None:
        from xgboost import XGBClassifier

        path = config.resolve(model_path)
        features_path = path.with_name(f"{path.stem}_features.json")
        if not path.exists():
            raise FileNotFoundError(
                f"XGBoost model not found at {path}. Train it first with "
                "`trp train-baseline -f <features.parquet> -m <model.json>`."
            )
        self.feature_order = json.loads(features_path.read_text())
        self.model = XGBClassifier()
        self.model.load_model(str(path))

    def predict_proba(self, board: BoardState) -> float:
        """Return the XGBoost win probability for the matchup."""
        X = F.baseline_features(board, self.feature_order)
        return float(self.model.predict_proba(X)[0, 1])


# ---------------------------------------------------------------------------
# Loading / caching
# ---------------------------------------------------------------------------

_KINDS = ("vit", "cnn", "xgboost")
_DEFAULT_PATHS = {
    "vit": config.DEFAULT_VIT_CKPT,
    "cnn": config.DEFAULT_CNN_CKPT,
    "xgboost": config.DEFAULT_XGB_MODEL,
}


# Which kinds have finished loading at least once — lets the UI tell the user
# that their first prediction includes the (slow) model load.
_LOADED: set[str] = set()


@cache
def _load(kind: str, path: str, device: str) -> Predictor:
    if kind == "vit":
        predictor: Predictor = VitPredictor(path, device=device)
    elif kind == "cnn":
        predictor = CnnPredictor(path, device=device)
    elif kind == "xgboost":
        predictor = XgbPredictor(path)
    else:
        raise ValueError(f"Unknown model kind {kind!r}; expected one of {_KINDS}.")
    _LOADED.add(kind)
    return predictor


def get_predictor(kind: str, path: str | None = None, device: str = "cpu") -> Predictor:
    """Return a cached predictor of the given ``kind`` (``vit``/``cnn``/``xgboost``).

    Args:
        kind: Which backend to use.
        path: Optional model/checkpoint path; falls back to the configured default.
        device: Torch device for the neural backends (ignored by XGBoost).
    """
    kind = kind.lower()
    if kind not in _KINDS:
        raise ValueError(f"Unknown model kind {kind!r}; expected one of {_KINDS}.")
    return _load(kind, path or _DEFAULT_PATHS[kind], device)


def available_models() -> dict[str, bool]:
    """Report which default models are present on disk (for the UI to grey out)."""
    return {kind: config.resolve(_DEFAULT_PATHS[kind]).exists() for kind in _KINDS}


def loaded_models() -> list[str]:
    """Report which model kinds are already loaded in memory."""
    return sorted(_LOADED)
