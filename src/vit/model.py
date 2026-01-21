import lightning as L
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from torchmetrics import Accuracy, F1Score

from src.vit.positional_encoding import get_2d_sincos_pos_embed


class PatchEmbedding(nn.Module):
    """Converts board grid into patch embeddings for ViT."""

    def __init__(
        self,
        n_units: int,
        n_items: int,
        n_tiers: int = 5,
        img_height: int = 8,
        img_width: int = 7,
        d_model: int = 128,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()

        self.patch_h = 1
        self.patch_w = 1
        self.n_patches_h = img_height
        self.n_patches_w = img_width
        self.n_patches = self.n_patches_h * self.n_patches_w

        # Embeddings for each channel
        self.unit_embed = nn.Embedding(n_units, d_model // 4, padding_idx=0)
        self.tier_embed = nn.Embedding(n_tiers, d_model // 8, padding_idx=0)
        self.item_embed = nn.Embedding(n_items, d_model // 8, padding_idx=0)

        # Patch embedding: project combined features to d_model
        # Each patch has: unit + tier + 3*item features
        patch_features = (d_model // 4) + (d_model // 8) + 3 * (d_model // 8)

        self.patch_projection = nn.Linear(patch_features, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        B = x.shape[0]

        # Extract channels
        units = x[:, 0].long()  # (B, H, W)
        tiers = x[:, 1].long()
        item1 = x[:, 2].long()
        item2 = x[:, 3].long()
        item3 = x[:, 4].long()

        # Embed each channel
        unit_emb = self.unit_embed(units)  # (B, H, W, d//4)
        tier_emb = self.tier_embed(tiers)  # (B, H, W, d//8)
        item1_emb = self.item_embed(item1)  # (B, H, W, d//8)
        item2_emb = self.item_embed(item2)
        item3_emb = self.item_embed(item3)

        # Combine all embeddings
        combined = torch.cat(
            [unit_emb, tier_emb, item1_emb, item2_emb, item3_emb], dim=-1
        )
        # (B, H, W, C)

        # Reshape into patches
        patches = combined.reshape(B, self.n_patches, -1)  # (B, H*W, C)

        # Project to d_model
        patches = self.patch_projection(patches)  # (B, n_patches, d_model)
        return self.norm(patches)


class TFTViT(L.LightningModule):
    """Vision Transformer model for TFT round prediction."""

    def __init__(
        self,
        n_units: int,
        n_items: int,
        n_traits: int,
        board_height: int = 8,
        board_width: int = 7,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 256,
        learning_rate: float = 5e-4,
        warmup_ratio: float = 0.05,
        plateau_ratio: float = 0.7,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            n_units=n_units,
            n_items=n_items,
            img_height=board_height,
            img_width=board_width,
            d_model=d_model,
            dropout_rate=dropout_rate,
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional embedding for patches + cls token
        pos_embed = get_2d_sincos_pos_embed(
            d_model, board_height, board_width, cls_token=True
        )
        self.pos_embed = nn.Parameter(pos_embed.float())
        self.pos_drop = nn.Dropout(dropout_rate)

        # Stochastic depth decay rule
        dpr = torch.linspace(0, dropout_rate, n_layers)

        # Transformer encoder
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=d_model,
                    num_heads=n_heads,
                    mlp_ratio=dim_feedforward / d_model,
                    qkv_bias=True,
                    proj_drop=dropout_rate,
                    attn_drop=dropout_rate,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                )
                for i in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)

        # Classification head
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model + n_traits),
            nn.Linear(d_model + n_traits, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.dropout_rate = dropout_rate
        self.lr = learning_rate
        self.warmup_ratio = warmup_ratio
        self.plateau_ratio = plateau_ratio

        self.val_accuracy = Accuracy(task="binary")
        self.val_f1 = F1Score(task="binary")

        self.test_accuracy = Accuracy(task="binary")
        self.test_f1 = F1Score(task="binary")

        self.save_hyperparameters()

    def forward(self, X_units: torch.Tensor, X_traits: torch.Tensor) -> torch.Tensor:  # noqa: D102
        B = X_units.shape[0]

        # Extract patch embeddings
        patches = self.patch_embed(X_units)  # (B, N, d_model)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls_tokens, patches], dim=1)  # (B, N+1, d_model)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)  # (B, N+1, d_model)

        # Extract CLS token output
        cls_output = x[:, 0]  # (B, d_model)

        # Combine with trait features
        combined = torch.cat([cls_output, X_traits.float()], dim=-1)

        # Classification
        logit = self.mlp(combined)
        return logit.squeeze(1)

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> float:  # noqa: D102
        x_units, x_traits, y = batch
        x_hat = self.forward(x_units, x_traits)
        loss = nn.functional.binary_cross_entropy_with_logits(x_hat, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> float:  # noqa: D102
        x_units, x_traits, y = batch
        x_hat = self.forward(x_units, x_traits)
        loss = nn.functional.binary_cross_entropy_with_logits(x_hat, y)

        preds = (torch.sigmoid(x_hat) > 0.5).int()

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        self.log(
            "val_accuracy",
            self.val_accuracy(preds, y.int()),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_f1",
            self.val_f1(preds, y.int()),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def test_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> float:  # noqa: D102
        x_units, x_traits, y = batch
        x_hat = self.forward(x_units, x_traits)
        loss = nn.functional.binary_cross_entropy_with_logits(x_hat, y)

        preds = (torch.sigmoid(x_hat) > 0.5).int()

        self.log("test_loss", loss, on_step=False, on_epoch=True)

        self.log(
            "test_accuracy",
            self.test_accuracy(preds, y.int()),
            on_step=False,
            on_epoch=True,
        )
        self.log("test_f1", self.test_f1(preds, y.int()), on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:  # noqa: D102
        x_units, x_traits, _ = batch
        return torch.sigmoid(self.forward(x_units, x_traits))

    def on_validation_epoch_end(self) -> None:  # noqa: D102
        self.val_accuracy.reset()
        self.val_f1.reset()

    def on_test_epoch_end(self) -> None:  # noqa: D102
        self.test_accuracy.reset()
        self.test_f1.reset()

    def configure_optimizers(self) -> dict:  # noqa: D102
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        total_steps = self.trainer.estimated_stepping_batches

        warmup_steps = int(self.warmup_ratio * total_steps)
        plateau_steps = int(self.plateau_ratio * total_steps)

        def lr_lambda(step: int) -> float:  # noqa: ANN202
            if step < warmup_steps:
                return step / max(1, warmup_steps)

            if step < warmup_steps + plateau_steps:
                return 1.0

            decay_steps = total_steps - warmup_steps - plateau_steps
            if decay_steps <= 0:
                return 1.0  # safety fallback

            decay_progress = (step - warmup_steps - plateau_steps) / decay_steps

            return max(0.0, 1.0 - decay_progress)

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
