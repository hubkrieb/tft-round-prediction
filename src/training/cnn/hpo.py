import json
import os

import optuna
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from optuna.integration import PyTorchLightningPruningCallback

import wandb
from src.training.cnn.data import TFTBoardDataModule
from src.training.cnn.model import TFTCNN
from src.training.utils.static_data import ITEMS, TRAITS, UNITS


def objective(
    trial: optuna.Trial,
    feature_path: str,
    max_epochs: int,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    study_dir: str,
    wandb_project: str,
) -> float:
    """
    Objective function for CNN hyperparameter optimization.

    Args:
        trial (optuna.Trial): Optuna trial object.
        feature_path (str): Path to the feature data.
        max_epochs (int): Maximum number of training epochs.
        seed (int): Random seed for reproducibility.
        num_workers (int): Number of workers for data loading.
        pin_memory (bool): Whether to pin memory in data loader.
        study_dir (str): Directory to save study results.
        wandb_project (str): Weights & Biases project name.

    Returns:
        float: Validation loss of the trained model.
    """
    emb_size_unit = trial.suggest_categorical("emb_size_unit", [8, 16, 32, 64])
    emb_size_item = trial.suggest_categorical("emb_size_item", [8, 16, 32])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)

    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)

    warmup_ratio = trial.suggest_float("warmup_ratio", 0.01, 0.2, step=0.01)
    plateau_ratio = trial.suggest_float("plateau_ratio", 0.1, 0.9, step=0.1)

    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])

    seed_everything(seed, workers=True)

    n_traits = 2 * sum(len(bp) for bp in TRAITS.values())

    trial_dir = os.path.join(study_dir, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)

    wandb_logger = WandbLogger(
        project=wandb_project,
        name=f"trial_{trial.number}",
        reinit=True,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=trial_dir,
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=10,
    )

    pruning_cb = PyTorchLightningPruningCallback(trial, monitor="val_loss")

    lr_monitor = LearningRateMonitor(logging_interval="step")

    try:
        model = TFTCNN(
            n_units=len(UNITS) + 1,
            n_items=len(ITEMS) + 1,
            n_traits=n_traits,
            emb_size_unit=emb_size_unit,
            emb_size_item=emb_size_item,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            plateau_ratio=plateau_ratio,
        )

        datamodule = TFTBoardDataModule(
            data_path=feature_path,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        trainer = Trainer(
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,
            max_epochs=max_epochs,
            logger=wandb_logger,
            callbacks=[
                checkpoint_cb,
                early_stop_cb,
                pruning_cb,
                lr_monitor,
            ],
            deterministic=True,
            enable_checkpointing=True,
        )

        trainer.fit(model, datamodule=datamodule)

        best_val = checkpoint_cb.best_model_score
        if best_val is None:
            raise RuntimeError("No validation score found. Check val dataloader.")

        best_val = best_val.item()

        trial.set_user_attr("best_ckpt", checkpoint_cb.best_model_path)

        return best_val

    except optuna.TrialPruned:
        wandb.run.summary["state"] = "pruned"
        raise

    finally:
        wandb.finish()


def run_optuna(
    feature_path: str,
    study_name: str = "tft_cnn_hpo",
    storage: str = "sqlite:///optuna.db",
    n_trials: int = 30,
    max_epochs: int = 50,
    seed: int = 54,
    num_workers: int = 4,
    pin_memory: bool = True,
    study_dir: str = "optuna_runs",
    wandb_project: str = "my-project",
) -> None:
    """
    Runs hyperparameter optimization for the CNN model using Optuna.

    Args:
        feature_path (str): Path to the feature data.
        study_name (str): Name of the Optuna study.
        storage (str): Storage URL for Optuna study.
        n_trials (int): Number of trials to run.
        max_epochs (int): Maximum number of training epochs per trial.
        seed (int): Random seed for reproducibility.
        num_workers (int): Number of workers for data loading.
        pin_memory (bool): Whether to pin memory in data loader.
        study_dir (str): Directory to save study results.
        wandb_project (str): Weights & Biases project name.
    """
    os.makedirs(study_dir, exist_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    study.optimize(
        lambda t: objective(
            t,
            feature_path=feature_path,
            max_epochs=max_epochs,
            seed=seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
            study_dir=study_dir,
            wandb_project=wandb_project,
        ),
        n_trials=n_trials,
    )

    best = study.best_trial
    summary = {
        "value": best.value,
        "params": best.params,
        "best_ckpt": best.user_attrs["best_ckpt"],
    }

    with open(os.path.join(study_dir, "best_trial.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("Best trial:")
    print(json.dumps(summary, indent=4))
