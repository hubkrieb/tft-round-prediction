"""Unified predictors that turn a :class:`BoardState` into a win probability.

Three backends share one interface (:meth:`Predictor.predict_proba`):

* :class:`VitPredictor`  - loads a ViT ``.onnx`` (board tensor + trait IDs).
* :class:`CnnPredictor`  - loads a CNN ``.onnx`` (board tensor + one-hot traits).
* :class:`XgbPredictor`  - loads a saved XGBoost model (wide one-hot features).

The neural backends run on onnxruntime, so serving does not need torch or
lightning at all. Models are loaded lazily and cached by (kind, path) so the
API pays the load cost once. All default paths resolve against the repo root.
"""

from __future__ import annotations

import json
import math
from functools import cache
from typing import TYPE_CHECKING

import numpy as np

from src.api import config
from src.api import featurize as F

if TYPE_CHECKING:
    from src.api.schema import BoardState


class Predictor:
    """Common interface for the three model backends."""

    def predict_proba(self, board: BoardState) -> float:
        """Return the player's win probability in ``[0, 1]``."""
        raise NotImplementedError


class _OnnxPredictor(Predictor):
    """Shared onnxruntime session handling for the ViT and CNN backends."""

    #: CLI command that produces the missing .onnx file (set by subclasses).
    train_cmd = ""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        import onnxruntime as ort

        path = config.resolve(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {path}. Train it first with "
                f"`{self.train_cmd}` (which also exports ONNX)."
            )
        providers = ["CPUExecutionProvider"]
        if device != "cpu":
            providers.insert(0, "CUDAExecutionProvider")
        self.session = ort.InferenceSession(str(path), providers=providers)

    def _run(self, x_units: np.ndarray, x_traits: np.ndarray) -> float:
        out = self.session.run(None, {"x_units": x_units, "x_traits": x_traits})
        return float(np.asarray(out[0])[0])


class VitPredictor(_OnnxPredictor):
    """Vision-Transformer backend."""

    train_cmd = "trp train-vit -f <features-dir>"

    def predict_proba(self, board: BoardState) -> float:
        """Return the ViT win probability for the matchup."""
        x_units = F.board_tensor(board).astype(np.int64)
        x_traits = F.vit_trait_ids(board).astype(np.int64)
        # The ViT graph outputs a logit; squash it to a probability.
        logit = self._run(x_units, x_traits)
        return 1.0 / (1.0 + math.exp(-logit))


class CnnPredictor(_OnnxPredictor):
    """Convolutional backend."""

    train_cmd = "trp train-cnn -f <features-dir>"

    def predict_proba(self, board: BoardState) -> float:
        """Return the CNN win probability for the matchup."""
        x_units = F.board_tensor(board).astype(np.int64)
        x_traits = F.cnn_trait_onehot(board).astype(np.float32)
        # The CNN graph already applies the sigmoid and outputs a probability.
        return self._run(x_units, x_traits)


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
    "vit": config.DEFAULT_VIT_ONNX,
    "cnn": config.DEFAULT_CNN_ONNX,
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
