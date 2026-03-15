import os
import pandas as pd
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger

from dataset import ECGDataset
from model import LitECGModel


# ---------------------------------------------------------------------------
# Custom callback: saves per-epoch metrics to a Parquet file
# ---------------------------------------------------------------------------
class ParquetMetricsCallback(pl.Callback):
    """Appends train/val metrics for every epoch to a Parquet file."""

    def __init__(self, save_path: str = "metrics_log.parquet"):
        super().__init__()
        self.save_path = save_path
        self._rows: list[dict] = []

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = {
            k: (v.item() if hasattr(v, "item") else v)
            for k, v in trainer.callback_metrics.items()
        }
        metrics["epoch"] = trainer.current_epoch
        self._rows.append(metrics)
        pd.DataFrame(self._rows).to_parquet(self.save_path, index=False)


# ---------------------------------------------------------------------------
def train_model():
    # -----------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------
    IMAGE_DIR   = "ecg_dataset/train"
    MASK_DIR    = "ecg_dataset/train"
    BATCH_SIZE  = 8
    EPOCHS      = 300
    LR          = 0.005
    VAL_SPLIT   = 0.15
    NUM_WORKERS = 4          # set to 0 on Windows if multiprocessing errors occur
    CKPT_DIR    = "checkpoints"
    LOG_DIR     = "logs"
    PARQUET_OUT = "metrics_log.parquet"

    # -----------------------------------------------------------------------
    # Data – train / val split
    # -----------------------------------------------------------------------
    full_dataset = ECGDataset(image_dir=IMAGE_DIR, mask_dir=MASK_DIR)

    val_size   = max(1, int(len(full_dataset) * VAL_SPLIT))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Train samples: {train_size} | Val samples: {val_size}")

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = LitECGModel(learning_rate=LR)

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------

    # 1. Save best checkpoint by val_dice (higher = better segmentation)
    best_ckpt = ModelCheckpoint(
        dirpath=CKPT_DIR,
        filename="best-{epoch:03d}-{val_dice:.4f}",
        monitor="val_dice",
        mode="max",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )

    # 2. Also keep a secondary checkpoint for best val_loss
    loss_ckpt = ModelCheckpoint(
        dirpath=CKPT_DIR,
        filename="best-loss-{epoch:03d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=False,
    )

    # 3. Early stopping on val_dice
    early_stop = EarlyStopping(
        monitor="val_dice",
        patience=40,
        mode="max",
        verbose=True,
    )

    # 4. Log LR every epoch to track scheduler
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # 5. Rich progress bar
    progress_bar = RichProgressBar()

    # 6. Custom Parquet metrics logger
    parquet_cb = ParquetMetricsCallback(save_path=PARQUET_OUT)

    callbacks = [best_ckpt, loss_ckpt, early_stop, lr_monitor, progress_bar, parquet_cb]

    # -----------------------------------------------------------------------
    # Logger
    # -----------------------------------------------------------------------
    csv_logger = CSVLogger(save_dir=LOG_DIR, name="ecg_training")

    # -----------------------------------------------------------------------
    # Trainer
    # -----------------------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",      # AMP – halves VRAM on modern GPUs
        log_every_n_steps=1,
        callbacks=callbacks,
        logger=csv_logger,
        enable_checkpointing=True,
        deterministic=False,
    )

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    print("Starting training with PyTorch Lightning...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # -----------------------------------------------------------------------
    # Post-training summary
    # -----------------------------------------------------------------------
    print(f"\n✅  Best checkpoint (dice) : {best_ckpt.best_model_path}")
    print(f"   Best val_dice          : {best_ckpt.best_model_score:.6f}")
    print(f"   Metrics saved to       : {PARQUET_OUT}")

    return best_ckpt.best_model_path


def main():
    train_model()


if __name__ == "__main__":
    main()