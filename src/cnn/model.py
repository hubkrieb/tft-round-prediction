import lightning as L
import torch
import torch.nn as nn


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
        self.norm = nn.LayerNorm(emb_size_unit + emb_size_item)

    def forward(  # noqa: D102
        self, unit_id: torch.Tensor, tier: torch.Tensor, item_ids: torch.Tensor
    ) -> torch.Tensor:
        unit_vec = self.unit_embedding(unit_id)
        tier_vec = self.tier_embedding(tier)

        # TODO: Try without sum
        unit_final = unit_vec + tier_vec

        item_vecs = self.item_embedding(item_ids)

        # TODO: Try without sum
        item_final = item_vecs.sum(dim=-4)

        board_vec = torch.cat([unit_final, item_final], dim=-1)
        return self.norm(board_vec)


class TFTCNN(L.LightningModule):
    """CNN model for TFT round prediction."""

    def __init__(
        self,
        n_units: int,
        n_items: int,
        n_traits: int,
        emb_size_unit: int = 16,
        emb_size_item: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = TFTBoardEncoder(
            n_units, n_items, emb_size_unit=emb_size_unit, emb_size_item=emb_size_item
        )
        in_channels = emb_size_unit + emb_size_item

        # Convolutional backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Final classification head
        self.fc = nn.Linear(128 * 8 * 7, 1)  # TODO: parametrize size * n_rows * n_cols

    def forward(self, X: torch.Tensor) -> torch.Tensor:  # noqa: D102
        units = X[:, 0]
        tiers = X[:, 1]
        items = X[:, 2:5]

        # Encode both boards
        embedding = self.encoder(units, tiers, items)  # (B, R, C, F)

        # Rearrange for CNN: (B, F, R, C)
        embedding = embedding.permute(0, 3, 1, 2)

        # CNN
        feat = self.cnn(embedding)  # (B, 128, 1, 1)
        feat = feat.flatten(1)  # (B, 128)

        # Combine CNN features with trait tiers
        # combined = torch.cat([feat, traits.float()], dim=-1)

        # Prediction
        logit = self.fc(feat)
        prob = torch.sigmoid(logit)  # win probability
        return prob.squeeze(1)

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> float:  # noqa: D102
        x, y = batch
        x_hat = self.forward(x)
        loss = nn.functional.binary_cross_entropy(x_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> float:  # noqa: D102
        x, y = batch
        x_hat = self.forward(x)
        loss = nn.functional.binary_cross_entropy(x_hat, y)
        self.log("val_loss", loss)
        return loss

    def predict_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> float:  # noqa: D102
        x, _ = batch
        return self.forward(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:  # noqa: D102
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
