from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from torchmetrics.classification import MultilabelF1Score
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra
from pandas import read_parquet
from task1.dataloader import ChemicalDataModule
from task1.model import HierarchicalChemicalClassifier
from task1.utils import prepare_hierarchy_and_weights, set_seed

    
@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    set_seed(40)
    
    # Load data
    train_df = read_parquet("task1/data/chebi_dataset_train.parquet")
    # test_df = read_parquet("task1/data/chebi_dataset_test_empty.parquet")
    
    label_cols = [col for col in train_df.columns if col.startswith("class_")]
    adj_matrix, pos_weights = prepare_hierarchy_and_weights(
        "task1/data/chebi_classes.obo",
        train_df, label_cols)
    
    # Initialize DataModule
    dm = ChemicalDataModule(train_df, label_cols=label_cols, radius=cfg.radius)

    # Initialize Model (from previous snippet)
    model = HierarchicalChemicalClassifier(
        input_dim=2048, 
        num_classes=500, 
        adj_matrix=adj_matrix, 
        pos_weights=pos_weights,
        lambda_hierarchy=cfg.train.lambda_hierarchy
    )
    
    # Initialize Trainer    
    run_id = wandb.util.generate_id() # type: ignore
    checkpoint_callback = ModelCheckpoint(
        monitor="val/macro_f1",
        dirpath=f"task1/checkpoints/{run_id}_radius_{cfg.radius}",
        filename="model-{epoch:02d}-{val_macro_f1:.24}",
        mode="max",
        save_top_k=3,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor=cfg.train.early_stopping.monitor,
        patience=cfg.train.early_stopping.patience,
        mode=cfg.train.early_stopping.mode,
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    wandb_logger = WandbLogger(
        project="Ensemble-2026-task1",
        name=f"radius_{cfg.radius}_enforce_hierarchy_run_{run_id}",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    trainer = L.Trainer(
        max_epochs=cfg.train.epochs,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        accelerator="auto",
        log_every_n_steps=10,
        logger=wandb_logger,
        gradient_clip_val=cfg.train.gradient_clip_val,
    )

    # Fit
    trainer.fit(model, datamodule=dm)
    
if __name__ == "__main__":
    main() # type: ignore