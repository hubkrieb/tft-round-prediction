import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def train_baseline(feature_path: str) -> None:
    """
    Trains a baseline XGBoost model.

    Args:
        feature_path (str): Path to the input features.

    """
    data = pd.read_parquet(feature_path)

    # Split data into train and test sets
    X = data.drop(["uuid", "round_id", "outcome"], axis=1)
    y = data["outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

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
