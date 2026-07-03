"""Inference API: featurize a user-built board and predict the round outcome.

This package is the backend that sits between the trained models in
:mod:`src.training` and the :mod:`src.web` frontend. It reuses the training
feature transforms verbatim so a board assembled in the UI is encoded
identically to the rounds the models were trained on.

Modules:
    config       - shared paths and board geometry.
    schema       - request/response models and the board representation.
    featurize    - board -> model-ready features (tensors, traits, wide vector).
    predictor    - load a ViT/CNN checkpoint or XGBoost model and score a board.
    app          - the FastAPI application and ``serve`` entrypoint.
    fetch_assets - download set 16 icons and build the frontend catalog.
"""
