import json

import typer

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
    from src.training.baseline.transform import extract_features

    extract_features(raw_data_path=raw_path, feature_path=feature_path)


@app.command(name="train-baseline")
def train_baseline_command(
    feature_path: str | None = typer.Option(
        None, "--feature-path", "-f", help="Path to features parquet file"
    ),
    model_path: str | None = typer.Option(
        None,
        "--model-path",
        "-m",
        help="Where to save the fitted model (defaults to models/baseline/xgboost.json)",
    ),
) -> None:
    """Train round prediction XGBoost model."""
    from src.training.baseline.train import train_baseline

    train_baseline(feature_path=feature_path, model_path=model_path)


@app.command(name="extract-cnn-features")
def extract_cnn_feature_command(
    raw_path: str | None = typer.Option(
        None, "--raw-path", "-r", help="Path to the raw data parquet file"
    ),
    feature_path: str | None = typer.Option(
        None, "--feature-path", "-f", help="Directory for features .npy files"
    ),
) -> None:
    """Transform raw TFT round data into feature tensors."""
    from src.training.cnn.transform import extract_tensors as extract_cnn_tensors

    extract_cnn_tensors(raw_data_path=raw_path, feature_path=feature_path)


@app.command(name="train-cnn")
def train_cnn_command(
    feature_path: str | None = typer.Option(
        None, "--feature-path", "-f", help="Directory of features .npy files"
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
    from src.training.cnn.train import train_cnn

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
        None, "--feature-path", "-f", help="Directory of features .npy files"
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
    from src.training.cnn.hpo import run_optuna

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
        None, "--feature-path", "-f", help="Directory for features .npy files"
    ),
) -> None:
    """Transform raw TFT round data into feature tensors."""
    from src.training.vit.transform import extract_tensors as extract_vit_tensors

    extract_vit_tensors(raw_data_path=raw_path, feature_path=feature_path)


@app.command(name="train-vit")
def train_vit_command(
    feature_path: str | None = typer.Option(
        None, "--feature-path", "-f", help="Directory of features .npy files"
    ),
    batch_size: int = typer.Option(
        1024, "--batch-size", "-b", help="Batch size to use for training"
    ),
    learning_rate: float = typer.Option(
        2.8e-3, "--lr", help="Learning rate to use for training"
    ),
    num_workers: int = typer.Option(
        4, "--num-workers", "-w", help="Number of workers to use for dataloader"
    ),
    max_epochs: int = typer.Option(
        50, "--max-epochs", "-e", help="Maximum amount of training epochs"
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
    from src.training.vit.train import train_vit

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
        None, "--feature-path", "-f", help="Directory of features .npy files"
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
    from src.training.vit.hpo import run_optuna as run_optuna_vit

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


@app.command(name="fetch-assets")
def fetch_assets_command(
    max_workers: int = typer.Option(
        16, "--workers", "-w", help="Number of parallel download workers"
    ),
) -> None:
    """Download set 16 champion/item/trait icons and build the UI catalog."""
    from src.api.fetch_assets import fetch_assets

    fetch_assets(max_workers=max_workers)


@app.command(name="extract-sample-boards")
def extract_sample_boards_command(
    raw_path: str = typer.Option(
        "data/set16/raw/merged_data.parquet",
        "--raw-path",
        "-r",
        help="Path to the raw data parquet file",
    ),
    per_stage: int = typer.Option(
        10, "--per-stage", help="Boards to keep per game stage"
    ),
    min_units: int = typer.Option(
        3, "--min-units", help="Minimum units per side for a board to qualify"
    ),
) -> None:
    """Sample real PVP boards into src/web/sample_boards.json for the UI."""
    from src.api.sample_boards import extract_sample_boards

    n = extract_sample_boards(
        raw_path=raw_path, per_stage=per_stage, min_units=min_units
    )
    typer.echo(f"Wrote {n} sample boards to src/web/data/sample_boards.json")


@app.command(name="serve")
def serve_command(
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind"),
    reload: bool = typer.Option(
        False, "--reload", help="Enable auto-reload (development)"
    ),
) -> None:
    """Serve the board-builder UI and prediction API."""
    from src.api.app import serve

    serve(host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
