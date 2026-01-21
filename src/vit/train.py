from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from src.cnn.data import TFTBoardDataModule
from src.utils.static_data import ITEMS, TRAITS, UNITS
from src.vit.model import TFTViT


def train_vit(
    feature_path: str,
    batch_size: int,
    learning_rate: float,
    num_workers: int,
    pin_memory: bool = True,
    max_epochs: int = 100,
    seed: int = 54,
    *,
    data_kwargs: dict | None = None,
    model_kwargs: dict | None = None,
    trainer_kwargs: dict | None = None,
) -> None:
    """Trains a Vision Transformer model using the provided feature data.

    Args:
        feature_path (str): The file path (e.g., path to an NPZ file) containing
            the preprocessed board feature data.
        batch_size (int): The number of samples per batch to load for training.
        learning_rate (float): The learning rate used for training.
        num_workers (int): The number of subprocesses to use for data loading.
            Set to 0 for single-process loading.
        pin_memory (bool): If ``True``, the data loader will copy Tensors into
            device/CUDA pinned memory before returning them.
        max_epochs (int): Maximum amount of training epochs.
        seed (int): Random seed for reproducibility.
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

    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=20),
        LearningRateMonitor(logging_interval="step"),
    ]

    wandb_logger = WandbLogger(project="tft-vit")

    trainer = Trainer(
        accelerator="auto",
        logger=wandb_logger,
        max_epochs=max_epochs,
        callbacks=callbacks,
        gradient_clip_val=1,
        **trainer_kwargs,
    )

    trainer.fit(model, datamodule=datamodule)

    trainer.test(model, ckpt_path="best", datamodule=datamodule)
