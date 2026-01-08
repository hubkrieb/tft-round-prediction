import math

import lightning as L
import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score


class TFTBoardEncoder(nn.Module):
    """Embedding model for TFT board inputs (units, tiers, items)."""

    def __init__(
        self,
        n_units: int,
        n_items: int,
        n_tiers: int = 5,
        emb_size_unit: int = 16,
        emb_size_item: int = 8,
    ) -> None:
        super().__init__()
        self.unit_embedding = nn.Embedding(n_units, emb_size_unit)
        self.tier_embedding = nn.Embedding(n_tiers, emb_size_unit)
        self.item_embedding = nn.Embedding(n_items, emb_size_item)
        self.norm = nn.LayerNorm(emb_size_unit + 3 * emb_size_item)

    def forward(  # noqa: D102
        self, unit_id: torch.Tensor, tier: torch.Tensor, item_ids: torch.Tensor
    ) -> torch.Tensor:
        unit_vec = self.unit_embedding(unit_id)
        tier_vec = self.tier_embedding(tier)

        # TODO: Try without sum
        unit_final = unit_vec + tier_vec

        item_vecs = self.item_embedding(item_ids)

        # Flatten the 3 items into one long vector: (B, H, W, 3*I_Dim)
        item_final = item_vecs.permute(0, 2, 3, 1, 4).flatten(start_dim=3)

        board_vec = torch.cat([unit_final, item_final], dim=-1)
        return self.norm(board_vec)


class TFTCNN(L.LightningModule):
    """CNN model for TFT round prediction."""

    def __init__(
        self,
        n_units: int,
        n_items: int,
        n_traits: int,
        board_height: int = 8,
        board_width: int = 7,
        emb_size_unit: int = 32,
        emb_size_item: int = 32,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.encoder = TFTBoardEncoder(
            n_units, n_items, emb_size_unit=emb_size_unit, emb_size_item=emb_size_item
        )
        in_channels = emb_size_unit + 3 * emb_size_item

        # Convolutional backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(64 * board_height * board_width + n_traits, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # binary classification (logit output)
        )

        self.lr = learning_rate
        self.warmup_steps = 5000
        self.total_steps = 50000

        self.val_accuracy = Accuracy(task="binary")
        self.val_f1 = F1Score(task="binary")

        self.test_accuracy = Accuracy(task="binary")
        self.test_f1 = F1Score(task="binary")

        self.save_hyperparameters()

    def forward(self, X_units: torch.Tensor, X_traits: torch.Tensor) -> torch.Tensor:  # noqa: D102
        units = X_units[:, 0]
        tiers = X_units[:, 1]
        items = X_units[:, 2:5]

        # Encode both boards
        embedding = self.encoder(units, tiers, items)  # (B, R, C, F)

        # Rearrange for CNN: (B, F, R, C)
        embedding = embedding.permute(0, 3, 1, 2)

        # CNN
        feat = self.cnn(embedding)
        feat = feat.flatten(1)

        # Combine CNN features with trait tiers
        feat = torch.cat([feat, X_traits.float()], dim=-1)

        # Prediction
        logit = self.mlp(feat)
        prob = torch.sigmoid(logit)  # win probability
        return prob.squeeze(1)

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> float:  # noqa: D102
        x_units, x_traits, y = batch
        x_hat = self.forward(x_units, x_traits)
        loss = nn.functional.binary_cross_entropy(x_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> float:  # noqa: D102
        x_units, x_traits, y = batch
        x_hat = self.forward(x_units, x_traits)
        loss = nn.functional.binary_cross_entropy(x_hat, y)

        preds = (x_hat > 0.5).int()

        self.val_accuracy.update(preds, y.int())
        self.val_f1.update(preds, y.int())

        self.log("val_accuracy", self.val_accuracy, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> float:  # noqa: D102
        x_units, x_traits, y = batch
        x_hat = self.forward(x_units, x_traits)
        loss = nn.functional.binary_cross_entropy(x_hat, y)

        preds = (x_hat > 0.5).int()

        self.test_accuracy.update(preds, y.int())
        self.test_f1.update(preds, y.int())

        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> float:  # noqa: D102
        x_units, x_traits, _ = batch
        return self.forward(x_units, x_traits)

    def on_test_epoch_end(self) -> None:  # noqa: D102
        acc = self.test_accuracy.compute()
        f1 = self.test_f1.compute()

        self.log("test_accuracy", acc)
        self.log("test_f1", f1)

        self.test_accuracy.reset()
        self.test_f1.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:  # noqa: D102
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        def lr_lambda(step: int):  # noqa: ANN202
            if step < self.warmup_steps:
                return step / self.warmup_steps
            progress = (step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        self.logger.experiment.config["optimizer"] = optimizer.__class__.__name__
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_fit_start(self) -> None:  # noqa: D102
        self.logger.experiment.config["batch_size"] = self.trainer.datamodule.batch_size
