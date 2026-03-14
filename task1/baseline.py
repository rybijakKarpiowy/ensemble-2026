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
from task1.utils import prepare_hierarchy_and_weights

class HierarchicalChemicalClassifier(L.LightningModule):
    def __init__(self, input_dim, num_classes, adj_matrix, pos_weights=None, lr=1e-3):
        """
        Args:
            input_dim: Size of input vector (e.g., 2048 for Morgan bits).
            num_classes: 500 classes.
            adj_matrix: Tensor (num_classes, num_classes), adj[i, j] = 1 if i is parent of j.
            pos_weights: Tensor (num_classes,) to handle sparse leaf imbalance.
            lr: Learning rate.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['adj_matrix', 'pos_weights'])
        
        # 1. Architecture
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        
        # 2. Hierarchy & Weights (Buffers move to GPU automatically)
        self.register_buffer("adj_matrix", adj_matrix.float())
        if pos_weights is not None:
            self.register_buffer("pos_weights", pos_weights.float())
        else:
            self.register_buffer("pos_weights", torch.ones(num_classes))
        
        # 3. Metrics
        self.train_f1 = MultilabelF1Score(num_labels=num_classes, average='macro')
        self.val_f1 = MultilabelF1Score(num_labels=num_classes, average='macro')

    def forward(self, x):
        return self.model(x)

    def hierarchical_loss(self, logits, targets):
        # Weighted BCE to handle sparse leaf nodes
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weights
        )
        
        # Consistency Penalty (P_child <= P_parent)
        probs = torch.sigmoid(logits)
        parents, children = torch.where(self.adj_matrix > 0)
        
        violations = 0
        if len(parents) > 0:
            diff = probs[:, children] - probs[:, parents]
            violations = torch.mean(torch.relu(diff) ** 2)
            
        return bce_loss + 0.5 * violations

    def calculate_consistency_metric(self, preds):
        """Tie-breaker metric: Percentage of samples with zero logical violations."""
        # (N, num_parents) -> sum of active children for each parent
        active_children = torch.matmul(preds, self.adj_matrix.T) 
        # Violation if parent is 0 but has active children
        violations = (preds == 0) & (active_children > 0)
        consistent_samples = (~violations.any(dim=1)).float().mean()
        return consistent_samples

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.hierarchical_loss(logits, y)
        
        probs = torch.sigmoid(logits)
        self.train_f1(probs, y)
        
        # Logging to WandB/Progress Bar
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/macro_f1", self.train_f1, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.hierarchical_loss(logits, y)
        
        # Apply constraint for the tie-breaker/inference
        raw_probs = torch.sigmoid(logits)
        consistent_preds = self.apply_hierarchy_constraint(raw_probs)
        
        # Metrics
        self.val_f1(consistent_preds, y)
        consistency_score = self.calculate_consistency_metric(consistent_preds)
        
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/macro_f1", self.val_f1, prog_bar=True)
        self.log("val/graph_consistency", consistency_score, prog_bar=True)

    def apply_hierarchy_constraint(self, probs, threshold=0.5):
        preds = (probs > threshold).float()
        # Bottom-up propagation: If child=1, parent=1
        for _ in range(8): # Increased depth for complex chemical ontologies
            preds = torch.max(preds, torch.matmul(preds, self.adj_matrix.T).clamp(0, 1))
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/macro_f1"}
        }

def get_dataloader(dataset, cfg, shuffle=False):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )
    
@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Load data
    train_df = read_parquet("task1/data/chebi_dataset_train.parquet")
    # test_df = read_parquet("task1/data/chebi_dataset_test_empty.parquet")
    
    label_cols = [col for col in train_df.columns if col.startswith("class_")]
    adj_matrix, pos_weights = prepare_hierarchy_and_weights(
        "task1/data/chebi_classes.obo",
        train_df, label_cols)
    
    # Initialize DataModule
    dm = ChemicalDataModule(train_df, label_cols=label_cols)

    # Initialize Model (from previous snippet)
    model = HierarchicalChemicalClassifier(
        input_dim=2048, 
        num_classes=500, 
        adj_matrix=adj_matrix, 
        pos_weights=pos_weights
    )
    
    # Initialize Trainer
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.train.early_stopping.monitor,
        dirpath="task1/checkpoints",
        filename="model-{epoch:02d}-{val_loss:.2f}",
        mode=cfg.train.early_stopping.mode,
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
        name=f"baseline_run",
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