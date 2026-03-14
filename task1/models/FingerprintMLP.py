from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from torchmetrics.classification import MultilabelF1Score


class FingerprintMLP(L.LightningModule):
    def __init__(self, input_dim, num_classes, adj_matrix, pos_weights=None, lr=1e-3, lambda_hierarchy=0.05):
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
            nn.Linear(1024, 1024),
            nn.ReLU(),
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

    def bce_loss(self, logits, targets):
        # Weighted BCE to handle sparse leaf nodes
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weights
        )
            
        return bce_loss

    def soft_macro_f1(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # tp, fp, fn calculated per class (dim 0 is batch)
        tp = (targets * probs).sum(dim=0)
        fp = ((1 - targets) * probs).sum(dim=0)
        fn = (targets * (1 - probs)).sum(dim=0)

        # Calculate F1 per class
        soft_f1 = 2 * tp / (2 * tp + fp + fn + 1e-7)
        
        # Return 1 - mean(F1) so we can minimize it
        # This treats a sparse leaf node exactly the same as a root node
        return 1 - soft_f1.mean()

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
        
        epoch = self.current_epoch
        alpha = min(1.0, epoch / 20)  # Linearly increase lambda over first 10 epochs

        loss = (1 - alpha) * self.bce_loss(logits, y) + alpha * self.soft_macro_f1(logits, y)
        
        probs = torch.sigmoid(logits)
        self.train_f1(probs, y)
        
        # Logging to WandB/Progress Bar
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/macro_f1", self.train_f1, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
            
        epoch = self.current_epoch
        alpha = min(1.0, epoch / 20)  # Linearly increase lambda over first 10 epochs
            
        loss = (1 - alpha) * self.bce_loss(logits, y) + alpha * self.soft_macro_f1(logits, y)
        
        # Apply constraint for the tie-breaker/inference
        raw_probs = torch.sigmoid(logits)
        consistent_preds = self.apply_hierarchy_constraint(raw_probs)
        
        # Metrics
        self.val_f1(consistent_preds, y)
        consistency_score = self.calculate_consistency_metric(consistent_preds)
        
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/macro_f1", self.val_f1, prog_bar=True)
        self.log("val_macro_f1", self.val_f1)
        self.log("val/graph_consistency", consistency_score, prog_bar=True)

    # def apply_hierarchy_constraint(self, probs, threshold=0.5):
    #     preds = (probs > threshold).float()
    #     # Bottom-up propagation: If child=1, parent=1
    #     for _ in range(63):
    #         preds = torch.max(preds, torch.matmul(preds, self.adj_matrix.T).clamp(0, 1))
    #     return preds
    
    def apply_hierarchy_constraint(self, probs, threshold=0.5):
        """
        Greedy top-down single-path constraint.
        At each active node, only the highest-probability child is activated.
        """
        N, C = probs.shape
        preds = torch.zeros(N, C, device=probs.device)

        # Activate roots unconditionally (they have no parents)
        has_parents = (self.adj_matrix.sum(dim=0) > 0)  # (C,)
        roots = (~has_parents).nonzero(as_tuple=True)[0]
        preds[:, roots] = (probs[:, roots] > threshold).float()

        # Top-down: for each active node, activate only the best child
        for _ in range(63):
            prev = preds.clone()
            for i in range(C):
                children = self.adj_matrix[i].nonzero(as_tuple=True)[0]  # adj[i,j]=1 → j is child of i
                if len(children) == 0:
                    continue

                active = preds[:, i].bool()  # samples where node i is active
                if not active.any():
                    continue

                child_probs  = probs[active][:, children]          # (n_active, n_children)
                best_vals, best_idx = child_probs.max(dim=1)       # (n_active,)

                # Only activate if best child clears threshold
                sample_indices = active.nonzero(as_tuple=True)[0]
                for k, sample in enumerate(sample_indices):
                    if best_vals[k] > threshold:
                        preds[sample, children[best_idx[k]]] = 1.0

            if (preds == prev).all():  # early stop if nothing changed
                break

        return preds
        

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/macro_f1"}
        }