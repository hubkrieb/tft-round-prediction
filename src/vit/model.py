import lightning as L
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from torchmetrics import Accuracy

from src.vit.positional_encoding import get_2d_sincos_pos_embed


class PatchEmbedding(nn.Module):
    """Converts board grid into patch embeddings for ViT."""

    def __init__(
        self,
        n_units: int,
        n_items: int,
        n_tiers: int,
        img_height: int,
        img_width: int,
        unit_embed_dim: int,
        tier_embed_dim: int,
        item_embed_dim: int,
        d_model: int,
    ) -> None:
        super().__init__()

        self.patch_h = 1
        self.patch_w = 1
        self.n_patches_h = img_height
        self.n_patches_w = img_width
        self.n_patches = self.n_patches_h * self.n_patches_w

        # Embeddings for each channel
        self.unit_embed = nn.Embedding(n_units, unit_embed_dim, padding_idx=0)
        self.tier_embed = nn.Embedding(n_tiers, tier_embed_dim, padding_idx=0)
        self.item_embed = nn.Embedding(n_items, item_embed_dim, padding_idx=0)

        # Patch embedding: project combined features to d_model
        # Each patch has: unit + tier + 3*item features
        patch_features = unit_embed_dim + tier_embed_dim + 3 * item_embed_dim

        self.patch_projection = nn.Linear(patch_features, d_model)

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
        return self.patch_projection(patches)  # (B, n_patches, d_model)


class TFTViT(L.LightningModule):
    """Vision Transformer model for TFT round prediction."""

    def __init__(
        self,
        n_units: int,
        n_items: int,
        n_traits: int,
        board_height: int = 8,
        board_width: int = 7,
        unit_embed_dim: int = 8,
        tier_embed_dim: int = 8,
        item_embed_dim: int = 8,
        trait_embed_dim: int = 32,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 512,
        learning_rate: float = 4e-3,
        warmup_steps: int = 2500,
        plateau_steps: int = 10000,
        decay_steps: int = 25000,
        dropout_rate: float = 0.02,
    ) -> None:
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            n_units=n_units,
            n_items=n_items,
            n_tiers=5,
            img_height=board_height,
            img_width=board_width,
            d_model=d_model,
            unit_embed_dim=unit_embed_dim,
            tier_embed_dim=tier_embed_dim,
            item_embed_dim=item_embed_dim,
        )

        self.traits_embed = nn.Embedding(n_traits, trait_embed_dim, padding_idx=0)
        self.traits_proj = nn.Linear(trait_embed_dim, d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional embedding for patches + cls token
        pos_embed = get_2d_sincos_pos_embed(
            d_model, board_height, board_width, cls_token=True
        )
        self.pos_embed = nn.Parameter(pos_embed.float(), requires_grad=False)
        self.pos_drop = nn.Dropout(dropout_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, dropout_rate, n_layers)]

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
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, 1),
        )

        nn.init.trunc_normal_(self.cls_token, std=1e-6)

        self.dropout_rate = dropout_rate
        self.lr = learning_rate
        self.warmup_steps = warmup_steps
        self.plateau_steps = plateau_steps
        self.decay_steps = decay_steps

        self.val_accuracy = Accuracy(task="binary")
        self.test_accuracy = Accuracy(task="binary")

        self.save_hyperparameters()

    def forward(self, X_units: torch.Tensor, X_traits: torch.Tensor) -> torch.Tensor:  # noqa: D102
        B = X_units.shape[0]

        X_traits = X_traits.long()

        # Extract patch embeddings
        patches = self.patch_embed(X_units)  # (B, N, d_model)
        traits_tokens = self.traits_embed(X_traits)  # (B, M, d_model)
        traits_tokens = self.traits_proj(traits_tokens)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls_tokens, patches], dim=1)  # (B, N+1, d_model)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = torch.cat([x, traits_tokens], dim=1)  # (B, N+M+1, d_model)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)  # (B, N+M+1, d_model)

        # Extract CLS token output
        cls_output = x[:, 0]  # (B, d_model)

        # Combine with trait features
        # combined = torch.cat([cls_output, X_traits.float()], dim=-1)

        # Classification
        logit = self.mlp(cls_output)
        return logit.squeeze(1)

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> float:  # noqa: D102
        x_units, x_traits, _, _, y = batch
        x_hat = self.forward(x_units, x_traits)
        loss = nn.functional.binary_cross_entropy_with_logits(x_hat, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> float:  # noqa: D102
        x_units, x_traits, _, _, y = batch
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
        return loss

    def test_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> float:  # noqa: D102
        x_units, x_traits, _, _, y = batch
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
        return loss

    def predict_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:  # noqa: D102
        x_units, x_traits, _, _, _ = batch
        return torch.sigmoid(self.forward(x_units, x_traits))

    def on_validation_epoch_end(self) -> None:  # noqa: D102
        self.val_accuracy.reset()

    def on_test_epoch_end(self) -> None:  # noqa: D102
        self.test_accuracy.reset()

    def configure_optimizers(self) -> dict:  # noqa: D102
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        def lr_lambda(step: int) -> float:  # noqa: ANN202
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)

            if step < self.warmup_steps + self.plateau_steps:
                return 1.0

            decay_progress = (
                step - self.warmup_steps - self.plateau_steps
            ) / self.decay_steps

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
