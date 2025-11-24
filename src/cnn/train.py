import lightning as L
from torch.utils.data import DataLoader

from src.cnn.data import TFTBoardDataset
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
    dataset = TFTBoardDataset(npz_path=feature_path)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    trainer = L.Trainer(accelerator="auto")

    trainer.fit(model, train_dataloaders=dataloader)
