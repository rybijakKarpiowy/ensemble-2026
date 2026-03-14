import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# ---------------------------------------------------------------------------
# torchmetrics compatibility shim
# torchmetrics < 0.10  → torchmetrics.Dice
# torchmetrics >= 0.10 → torchmetrics.classification.BinaryF1Score (F1 = Dice for binary)
#                        or torchmetrics.segmentation.DiceScore (>= 1.0)
# ---------------------------------------------------------------------------
def _make_dice_metric():
    # torchmetrics >= 1.0
    try:
        from torchmetrics.segmentation import DiceScore
        return DiceScore(num_classes=2, input_format="index")
    except ImportError:
        pass
    # torchmetrics 0.10 – 0.x
    try:
        from torchmetrics.classification import BinaryF1Score
        return BinaryF1Score()
    except ImportError:
        pass
    # torchmetrics < 0.10
    try:
        return __import__("torchmetrics").Dice(threshold=0.5)
    except AttributeError:
        pass
    # ultimate fallback – manual Dice as a Module
    class _ManualDice(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._tp = self._fp = self._fn = None
            self.reset()
        def reset(self):
            self._tp = self._fp = self._fn = 0
        def update(self, preds, target):
            self._tp += ((preds == 1) & (target == 1)).sum().item()
            self._fp += ((preds == 1) & (target == 0)).sum().item()
            self._fn += ((preds == 0) & (target == 1)).sum().item()
        def compute(self):
            denom = 2 * self._tp + self._fp + self._fn
            return torch.tensor(2 * self._tp / denom if denom > 0 else 0.0)
        def forward(self, preds, target):
            self.update(preds, target)
            return self.compute()
    return _ManualDice()


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual Block with SiLU activation."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1   = nn.Conv2d(in_channels,  out_channels, 3, padding=1)
        self.bn1     = nn.BatchNorm2d(out_channels)
        self.conv2   = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2     = nn.BatchNorm2d(out_channels)
        self.silu    = nn.SiLU()
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.silu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.silu(x + residual)


class ResUNet(nn.Module):
    """U-Net with Residual Blocks."""
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        # Encoder
        self.enc1  = ResBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2  = ResBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = ResBlock(128, 256)
        # Decoder
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2    = ResBlock(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64,  2, stride=2)
        self.dec1    = ResBlock(128, 64)
        # Output
        self.final_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b  = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([e2, self.upconv2(b)],  dim=1))
        d1 = self.dec1(torch.cat([e1, self.upconv1(d2)], dim=1))
        return self.final_conv(d1)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs        = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union        = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * intersection + smooth) / (union + smooth)).mean()


def combined_loss(logits: torch.Tensor, targets: torch.Tensor,
                  bce_weight: float = 0.5) -> torch.Tensor:
    bce  = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss(logits, targets)
    return bce_weight * bce + (1.0 - bce_weight) * dice


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class LitECGModel(pl.LightningModule):
    """PyTorch Lightning wrapper — ResUNet with combined BCE+Dice loss."""

    def __init__(self, learning_rate: float = 0.005):
        super().__init__()
        self.save_hyperparameters()
        self.model = ResUNet()

        # torchmetrics – works across all versions via compatibility shim
        self.train_dice = _make_dice_metric()
        self.val_dice   = _make_dice_metric()

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ------------------------------------------------------------------ #
    def _shared_step(self, batch):
        images, masks = batch
        logits = self(images)
        loss   = combined_loss(logits, masks)
        preds  = (torch.sigmoid(logits) > 0.5).long()
        return loss, preds, masks.long()

    # ------------------------------------------------------------------ #
    def training_step(self, batch, batch_idx):
        loss, preds, masks = self._shared_step(batch)
        self.train_dice(preds, masks)
        self.log("train_loss", loss,            on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_dice", self.train_dice, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # ------------------------------------------------------------------ #
    def validation_step(self, batch, batch_idx):
        loss, preds, masks = self._shared_step(batch)
        self.val_dice(preds, masks)
        self.log("val_loss", loss,          on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_dice", self.val_dice, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    # ------------------------------------------------------------------ #
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }