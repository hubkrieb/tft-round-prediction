from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger

from src.api import config
from src.training.utils.export import export_model
from src.training.utils.static_data import ITEMS, TRAITS, UNITS
from src.training.vit.data import TFTBoardDataModule
from src.training.vit.model import TFTViT


def train_vit(
    feature_path: str,
    batch_size: int,
    learning_rate: float,
    num_workers: int,
    pin_memory: bool = True,
    max_epochs: int = 100,
    seed: int = 54,
    ckpt_path: str | None = None,
    model_path: str | None = None,
    *,
    data_kwargs: dict | None = None,
    model_kwargs: dict | None = None,
    trainer_kwargs: dict | None = None,
) -> None:
    """Trains a Vision Transformer model using the provided feature data.

    Args:
        feature_path (str): Directory containing the preprocessed board feature
            .npy files (x_units, x_traits, x_patch, y).
        batch_size (int): The number of samples per batch to load for training.
        learning_rate (float): The learning rate used for training.
        num_workers (int): The number of subprocesses to use for data loading.
            Set to 0 for single-process loading.
        pin_memory (bool): If ``True``, the data loader will copy Tensors into
            device/CUDA pinned memory before returning them.
        max_epochs (int): Maximum amount of training epochs.
        seed (int): Random seed for reproducibility.
        ckpt_path (str | None): Path to a checkpoint to resume training from.
        model_path (str | None): Where to save the ONNX export of the best model
            (a ``.ckpt`` copy is saved alongside with the same stem). Defaults
            to the model the app serves, ``models/vit/vit.onnx``.
        data_kwargs (dict | None): Additional keyword arguments for the data module.
        model_kwargs (dict | None): Additional keyword arguments for the model.
        trainer_kwargs (dict | None): Additional keyword arguments for the trainer.
    """
    data_kwargs = data_kwargs or {}
    model_kwargs = model_kwargs or {}
    trainer_kwargs = trainer_kwargs or {}

    seed_everything(seed=seed, workers=True)

    model = TFTViT(
        n_units=len(UNITS) + 1,
        n_items=len(ITEMS) + 1,
        n_traits=2 * sum(len(bp) for bp in TRAITS.values()),
        learning_rate=learning_rate,
        **model_kwargs,
    )

    datamodule = TFTBoardDataModule(
        data_path=feature_path,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **data_kwargs,
    )

    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best-{epoch}-{val_loss:.4f}",
    )
    callbacks: list[Callback] = [
        EarlyStopping(monitor="val_loss", mode="min", patience=5),
        LearningRateMonitor(logging_interval="step"),
        checkpoint_cb,
    ]

    wandb_logger = WandbLogger(project="tft-vit")

    trainer = Trainer(
        accelerator="auto",
        precision="16-mixed",
        logger=wandb_logger,
        max_epochs=max_epochs,
        callbacks=callbacks,
        gradient_clip_val=5,
        **trainer_kwargs,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    trainer.test(model, ckpt_path="best", datamodule=datamodule)

    if trainer.is_global_zero:
        export_model(
            model_cls=TFTViT,
            ckpt_path=checkpoint_cb.best_model_path,
            model_path=config.resolve(model_path or config.DEFAULT_VIT_ONNX),
            sample_batch=next(iter(datamodule.test_dataloader())),
        )
