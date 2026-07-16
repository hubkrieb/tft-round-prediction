import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


def train_baseline(
    feature_path: str, test_size: float = 0.2, model_path: str | None = None
) -> None:
    """
    Trains a baseline XGBoost model.

    Uses a chronological train/test split (oldest rounds train, newest rounds
    test) to match the CNN/ViT datamodules, so the three models are evaluated on
    comparable held-out periods. ``extract_features`` writes the parquet sorted
    by ``timestamp`` ascending; this re-sorts defensively before splitting.

    Args:
        feature_path (str): Path to the input features.
        test_size (float): Fraction of the most recent rounds held out for test.
        model_path (str | None): If given, the fitted model is saved to this path
            (XGBoost JSON format) alongside a ``<stem>_features.json`` file holding
            the exact ordered feature-column list. Both are needed to score new
            boards at inference time. Defaults to ``models/xgboost/xgboost.json``.

    """
    data = pd.read_parquet(feature_path)

    # Chronological split: oldest -> train, newest -> test (no shuffle).
    data = data.sort_values("timestamp").reset_index(drop=True)
    X = data.drop(["round_idx", "outcome", "timestamp"], axis=1)
    y = data["outcome"]
    train_size = int((1.0 - test_size) * len(data))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # Train XGBoost classifier
    model = XGBClassifier(eval_metric="logloss")
    model.fit(X_train, y_train)

    # Predict on the test set
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"Train set accuracy: {train_accuracy:.4f}")
    print(f"Test set accuracy: {test_accuracy:.4f}")

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_test_pred))

    feature_names = X.columns
    importances = model.feature_importances_
    feat_imp = (
        pd.Series(importances, index=feature_names)
        .sort_values(ascending=False)
        .head(10)
    )

    print("Feature importances (descending):")
    for name, score in feat_imp.items():
        print(f"{name}: {score:.6f}")

    # Persist the fitted model plus its exact feature ordering so the inference
    # API can rebuild the same wide feature vector for a user-supplied board.
    if model_path is None:
        model_path = "models/xgboost/xgboost.json"
    out = Path(model_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out))
    features_path = out.with_name(f"{out.stem}_features.json")
    with open(features_path, "w") as f:
        json.dump(list(feature_names), f)
    print(f"\nSaved model -> {out}")
    print(f"Saved feature order -> {features_path}")
