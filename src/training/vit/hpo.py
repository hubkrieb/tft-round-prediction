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
from src.training.utils.static_data import ITEMS, TRAITS, UNITS
from src.training.vit.data import TFTBoardDataModule
from src.training.vit.model import TFTViT


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
    Objective function for ViT hyperparameter optimization.

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
    # Embedding dimensions
    unit_embed_dim = trial.suggest_categorical("unit_embed_dim", [8, 16, 32])
    tier_embed_dim = trial.suggest_categorical("tier_embed_dim", [4, 8, 16])
    item_embed_dim = trial.suggest_categorical("item_embed_dim", [4, 8, 16])
    trait_embed_dim = trial.suggest_categorical("trait_embed_dim", [16, 32, 64])

    # Transformer architecture
    d_model = trial.suggest_categorical("d_model", [32, 64, 128])
    # Ensure d_model is divisible by n_heads (timm Block requirement)
    # Using [2, 4, 8] safely divides all of [32, 64, 128].
    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
    n_layers = trial.suggest_int("n_layers", 2, 6)
    dim_feedforward = trial.suggest_categorical(
        "dim_feedforward", [128, 256, 512, 1024]
    )

    # Regularization & Learning Rate
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.4)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    # LR schedule is now expressed in absolute optimizer steps (the model no
    # longer derives them from ratios of total_steps).
    warmup_steps = trial.suggest_int("warmup_steps", 500, 5000, step=500)
    plateau_steps = trial.suggest_int("plateau_steps", 2000, 15000, step=1000)
    decay_steps = trial.suggest_int("decay_steps", 10000, 40000, step=5000)

    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])

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
        model = TFTViT(
            n_units=len(UNITS) + 1,
            n_items=len(ITEMS) + 1,
            n_traits=n_traits,
            unit_embed_dim=unit_embed_dim,
            tier_embed_dim=tier_embed_dim,
            item_embed_dim=item_embed_dim,
            trait_embed_dim=trait_embed_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            plateau_steps=plateau_steps,
            decay_steps=decay_steps,
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
    study_name: str = "tft_vit_hpo",
    storage: str = "sqlite:///optuna_vit.db",
    n_trials: int = 30,
    max_epochs: int = 50,
    seed: int = 54,
    num_workers: int = 4,
    pin_memory: bool = True,
    study_dir: str = "optuna_vit_runs",
    wandb_project: str = "tft-vit",
) -> None:
    """
    Runs hyperparameter optimization for the ViT model using Optuna.

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
