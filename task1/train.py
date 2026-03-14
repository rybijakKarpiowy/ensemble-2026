from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch
import pytorch_lightning as L
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra
from pandas import read_parquet
from task1.datasets.dataloader import ChemicalDataModule
from task1.models.FingerprintMLP import FingerprintMLP
from task1.models.chemBERTa import SMILESDescriptionModel
from task1.utils import prepare_hierarchy_and_weights, set_seed


def build_model(cfg, adj_matrix, pos_weights):
    model_type = cfg.model.type

    if model_type == "fingerprint_mlp":
        return FingerprintMLP(
            input_dim=2048,
            num_classes=500,
            adj_matrix=adj_matrix,
            pos_weights=pos_weights,
            lambda_hierarchy=cfg.train.lambda_hierarchy,
        )

    elif model_type == "smiles_description":
        label_embeddings = torch.load(cfg.model.label_embeddings_path)
        return SMILESDescriptionModel(
            num_classes=500,
            adj_matrix=adj_matrix,
            label_embeddings=label_embeddings,
            pos_weights=pos_weights,
            lr=cfg.train.lr,
            proj_dim=cfg.model.proj_dim,
            dropout=cfg.model.dropout,
            smiles_encoder_name=cfg.model.smiles_encoder,
            hierarchy_depth=cfg.model.hierarchy_depth,
        )

    else:
        raise ValueError(f"Unknown model type: '{model_type}'. Choose 'fingerprint_mlp' or 'smiles_description'.")


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    set_seed(40)

    # Load data
    train_df = read_parquet("task1/data/chebi_dataset_train.parquet")
    label_cols = [col for col in train_df.columns if col.startswith("class_")]
    adj_matrix, pos_weights = prepare_hierarchy_and_weights(
        "task1/data/chebi_classes.obo",
        train_df, label_cols,
    )

    
    # DataModule — SMILES column used when model needs it, fingerprints otherwise
    if cfg.model.type == "smiles_description":
        from task1.datasets.BERTdataset import SMILESDataModule
        dm = SMILESDataModule(
            df=train_df,
            label_cols=label_cols,
            val_size=0.2,
            encoder_name=cfg.model.smiles_encoder,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.data.num_workers,
            max_length=cfg.model.max_length,
            seed=cfg.seed,
        )
    else:
        dm = ChemicalDataModule(train_df, label_cols=label_cols, radius=cfg.radius)

    # Model selection
    model = build_model(cfg, adj_matrix, pos_weights)

    # Trainer setup (unchanged)
    run_id = wandb.util.generate_id()  # type: ignore
    checkpoint_callback = ModelCheckpoint(
        monitor="val/macro_f1",
        dirpath=f"task1/checkpoints/{run_id}_{cfg.model.type}",
        filename="model-{epoch:02d}-{val_macro_f1:.4f}",
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
        name=f"{cfg.model.type}_radius_{cfg.radius}_run_{run_id}",
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

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()  # type: ignore