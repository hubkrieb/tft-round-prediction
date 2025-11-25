import lightning as L
from lightning.pytorch.loggers import WandbLogger

from src.cnn.data import TFTBoardDataModule
from src.cnn.model import TFTCNN
from src.utils.static_data import ITEMS, UNITS


def train_cnn(
    feature_path: str, batch_size: int, num_workers: int, pin_memory: bool = True
) -> None:
    """Trains a CNN model using the provided feature data.

    Args:
        feature_path: The file path (e.g., path to an NPZ file) containing
            the preprocessed board feature data.
        batch_size: The number of samples per batch to load for training.
        num_workers: The number of subprocesses to use for data loading.
            Set to 0 for single-process loading.
        pin_memory: If ``True``, the data loader will copy Tensors into
            device/CUDA pinned memory before returning them.
    """
    model = TFTCNN(n_units=len(UNITS) + 1, n_items=len(ITEMS) + 1, n_traits=1)

    datamodule = TFTBoardDataModule(
        data_path=feature_path,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    wandb_logger = WandbLogger(project="my-project")

    trainer = L.Trainer(accelerator="auto", logger=wandb_logger)

    trainer.fit(model, datamodule=datamodule)
