import typer

from src.baseline.train import train_baseline
from src.baseline.transform import extract_features
from src.cnn.train import train_cnn
from src.cnn.transform import extract_tensors

app = typer.Typer()


@app.command(name="extract-baseline-features")
def extract_baseline_feature_command(
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
    train_baseline(feature_path=feature_path)


@app.command(name="extract-cnn-features")
def extract_cnn_feature_command(
    raw_path: str | None = typer.Option(
        None, "--raw-path", "-r", help="Path to the raw data parquet file"
    ),
    feature_path: str | None = typer.Option(
        None, "--feature-path", "-f", help="Path to features .npz file"
    ),
) -> None:
    """Transform raw TFT round data into feature tensors."""
    extract_tensors(raw_data_path=raw_path, feature_path=feature_path)


@app.command(name="train-cnn")
def train_cnn_command(
    feature_path: str | None = typer.Option(
        None, "--feature-path", "-f", help="Path to features .npz file"
    ),
    batch_size: int = typer.Option(
        512, "--batch-size", "-b", help="Batch size to use for training"
    ),
    num_workers: int = typer.Option(
        4, "--num-workers", "-w", help="Number of workers to use for dataloader"
    ),
) -> None:
    """Train round prediction CNN model."""
    train_cnn(feature_path=feature_path, batch_size=batch_size, num_workers=num_workers)


if __name__ == "__main__":
    app()
