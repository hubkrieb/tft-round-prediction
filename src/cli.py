import json

import typer

from src.baseline.train import train_baseline
from src.baseline.transform import extract_features
from src.cnn.hpo import run_optuna
from src.cnn.train import train_cnn
from src.cnn.transform import extract_tensors as extract_cnn_tensors
from src.vit.hpo import run_optuna as run_optuna_vit
from src.vit.train import train_vit
from src.vit.transform import extract_tensors as extract_vit_tensors

app = typer.Typer()

DATA_KW = typer.Option(
    None,
    "--data-kw",
    help="Extra KEY=VALUE arguments forwarded to the data module",
)

MODEL_KW = typer.Option(
    None,
    "--model-kw",
    help="Extra KEY=VALUE arguments forwarded to the model",
)

TRAINER_KW = typer.Option(
    None,
    "--trainer-kw",
    help="Extra KEY=VALUE arguments forwarded to the Trainer",
)


def parse_kv_options(values: list[str] | None) -> dict:
    """Parse repeated KEY=VALUE CLI options into a dict.

    Args:
        values (list[str] | None): List of KEY=VALUE strings.

    Returns:
        dict: Parsed key-value pairs.
    """
    if not values:
        return {}

    parsed = {}
    for item in values:
        key, value = item.split("=", 1)
        parsed[key] = json.loads(value)
    return parsed


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
    extract_cnn_tensors(raw_data_path=raw_path, feature_path=feature_path)


@app.command(name="train-cnn")
def train_cnn_command(
    feature_path: str | None = typer.Option(
        None, "--feature-path", "-f", help="Path to features .npz file"
    ),
    batch_size: int = typer.Option(
        512, "--batch-size", "-b", help="Batch size to use for training"
    ),
    learning_rate: float = typer.Option(
        0.004, "--lr", help="Learning rate to use for training"
    ),
    num_workers: int = typer.Option(
        4, "--num-workers", "-w", help="Number of workers to use for dataloader"
    ),
    max_epochs: int = typer.Option(
        100, "--max-epochs", "-e", help="Maximum amount of training epochs"
    ),
    seed: int = typer.Option(
        54, "--seed", "-s", help="Random seed for reproducibility"
    ),
    data_kw: list[str] | None = DATA_KW,
    model_kw: list[str] | None = MODEL_KW,
    trainer_kw: list[str] | None = TRAINER_KW,
) -> None:
    """Train round prediction CNN model."""
    train_cnn(
        feature_path=feature_path,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_workers=num_workers,
        max_epochs=max_epochs,
        seed=seed,
        data_kwargs=parse_kv_options(data_kw),
        model_kwargs=parse_kv_options(model_kw),
        trainer_kwargs=parse_kv_options(trainer_kw),
    )


@app.command(name="hpo-cnn")
def hpo_cnn_command(
    feature_path: str | None = typer.Option(
        None, "--feature-path", "-f", help="Path to features .npz file"
    ),
    n_trials: int = typer.Option(
        30, "--n-trials", "-n", help="Number of HPO trials to run"
    ),
    max_epochs: int = typer.Option(
        100, "--max-epochs", "-e", help="Maximum amount of training epochs"
    ),
    seed: int = typer.Option(
        54, "--seed", "-s", help="Random seed for reproducibility"
    ),
    num_workers: int = typer.Option(
        4, "--num-workers", "-w", help="Number of workers to use for dataloader"
    ),
    pin_memory: bool = typer.Option(
        True,
        "--pin-memory/--no-pin-memory",
        help="Whether to pin memory in data loader",
    ),
    study_dir: str = typer.Option(
        "optuna_runs", "--study-dir", help="Directory to save study results"
    ),
    wandb_project: str = typer.Option(
        "my-project", "--wandb-project", help="Weights & Biases project name"
    ),
) -> None:
    """Run hyperparameter optimization for CNN model."""
    run_optuna(
        feature_path=feature_path,
        n_trials=n_trials,
        max_epochs=max_epochs,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        study_dir=study_dir,
        wandb_project=wandb_project,
    )


@app.command(name="extract-vit-features")
def extract_vit_feature_command(
    raw_path: str | None = typer.Option(
        None, "--raw-path", "-r", help="Path to the raw data parquet file"
    ),
    feature_path: str | None = typer.Option(
        None, "--feature-path", "-f", help="Path to features .npz file"
    ),
) -> None:
    """Transform raw TFT round data into feature tensors."""
    extract_vit_tensors(raw_data_path=raw_path, feature_path=feature_path)


@app.command(name="train-vit")
def train_vit_command(
    feature_path: str | None = typer.Option(
        None, "--feature-path", "-f", help="Path to features .npz file"
    ),
    batch_size: int = typer.Option(
        1024, "--batch-size", "-b", help="Batch size to use for training"
    ),
    learning_rate: float = typer.Option(
        2e-3, "--lr", help="Learning rate to use for training"
    ),
    num_workers: int = typer.Option(
        4, "--num-workers", "-w", help="Number of workers to use for dataloader"
    ),
    max_epochs: int = typer.Option(
        100, "--max-epochs", "-e", help="Maximum amount of training epochs"
    ),
    seed: int = typer.Option(
        54, "--seed", "-s", help="Random seed for reproducibility"
    ),
    ckpt_path: str | None = typer.Option(
        None, "--ckpt-path", "-c", help="Path to checkpoint to resume from"
    ),
    data_kw: list[str] | None = DATA_KW,
    model_kw: list[str] | None = MODEL_KW,
    trainer_kw: list[str] | None = TRAINER_KW,
) -> None:
    """Train round prediction ViT model."""
    train_vit(
        feature_path=feature_path,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_workers=num_workers,
        max_epochs=max_epochs,
        seed=seed,
        ckpt_path=ckpt_path,
        data_kwargs=parse_kv_options(data_kw),
        model_kwargs=parse_kv_options(model_kw),
        trainer_kwargs=parse_kv_options(trainer_kw),
    )


@app.command(name="hpo-vit")
def hpo_vit_command(
    feature_path: str | None = typer.Option(
        None, "--feature-path", "-f", help="Path to features .npz file"
    ),
    n_trials: int = typer.Option(
        30, "--n-trials", "-n", help="Number of HPO trials to run"
    ),
    max_epochs: int = typer.Option(
        50, "--max-epochs", "-e", help="Maximum amount of training epochs"
    ),
    seed: int = typer.Option(
        54, "--seed", "-s", help="Random seed for reproducibility"
    ),
    num_workers: int = typer.Option(
        4, "--num-workers", "-w", help="Number of workers to use for dataloader"
    ),
    pin_memory: bool = typer.Option(
        True,
        "--pin-memory/--no-pin-memory",
        help="Whether to pin memory in data loader",
    ),
    study_dir: str = typer.Option(
        "optuna_vit_runs", "--study-dir", help="Directory to save study results"
    ),
    wandb_project: str = typer.Option(
        "tft-vit", "--wandb-project", help="Weights & Biases project name"
    ),
) -> None:
    """Run hyperparameter optimization for ViT model."""
    run_optuna_vit(
        feature_path=feature_path,
        n_trials=n_trials,
        max_epochs=max_epochs,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        study_dir=study_dir,
        wandb_project=wandb_project,
    )


if __name__ == "__main__":
    app()
