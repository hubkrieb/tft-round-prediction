import typer

from src.baseline.train import train_baseline
from src.baseline.transform import extract_features

app = typer.Typer()


@app.command(name="extract-baseline-features")
def extract_feature_command(
    raw_path: str | None = typer.Option(
        None, "--raw-path", "-r", help="Path to the raw data parquet file"
    ),
    feature_path: str | None = typer.Option(
        None, "--feature-path", "-f", help="Path to features parquet file"
    ),
) -> None:
    """Transform raw TFT round data into features."""
    extract_features(raw_data_path=raw_path, feature_path=feature_path)


@app.command(name="train-baseline")
def train_baseline_command(
    feature_path: str | None = typer.Option(
        None, "--feature-path", "-f", help="Path to features parquet file"
    ),
) -> None:
    """Train round prediction XGBoost model."""
    train_baseline(feature_path)


if __name__ == "__main__":
    app()
