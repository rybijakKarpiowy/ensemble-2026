from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from torchmetrics.classification import MultilabelF1Score


class FingerprintMLP(L.LightningModule):
    def __init__(self, input_dim, num_classes, adj_matrix, pos_weights=None, lr=1e-3, lambda_hierarchy=0.05):
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
        
        # 2. Hierarchy & Weights
        self.register_buffer("adj_matrix", adj_matrix.float())
        if pos_weights is not None:
            self.register_buffer("pos_weights", pos_weights.float())
        else:
            self.register_buffer("pos_weights", torch.ones(num_classes))

        # 3. Precompute successor tree — list[tensor] where children[i] = tensor of child indices
        #    Built once here; no .nonzero() calls at inference time.
        adj = adj_matrix.float()
        self.children_list = [
            adj[i].nonzero(as_tuple=True)[0]   # 1-D tensor of child indices, may be empty
            for i in range(num_classes)
        ]
        has_parents = (adj.sum(dim=0) > 0)
        self.roots = (~has_parents).nonzero(as_tuple=True)[0]  # 1-D tensor of root indices

        # 4. Metrics
        self.train_f1 = MultilabelF1Score(num_labels=num_classes, average='macro')
        self.val_f1   = MultilabelF1Score(num_labels=num_classes, average='macro')

    def forward(self, x):
        return self.model(x)

    def bce_loss(self, logits, targets):
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weights
        )

    def soft_macro_f1(self, logits, targets):
        probs = torch.sigmoid(logits)
        tp = (targets * probs).sum(dim=0)
        fp = ((1 - targets) * probs).sum(dim=0)
        fn = (targets * (1 - probs)).sum(dim=0)
        soft_f1 = 2 * tp / (2 * tp + fp + fn + 1e-7)
        return 1 - soft_f1.mean()

    def calculate_consistency_metric(self, preds):
        active_children = torch.matmul(preds, self.adj_matrix.T)
        violations = (preds == 0) & (active_children > 0)
        return (~violations.any(dim=1)).float().mean()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        alpha = min(1.0, self.current_epoch / 20)
        loss = (1 - alpha) * self.bce_loss(logits, y) + alpha * self.soft_macro_f1(logits, y)

        self.train_f1(torch.sigmoid(logits), y)
        self.log("train/loss",     loss,          on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/macro_f1", self.train_f1, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        alpha = min(1.0, self.current_epoch / 20)
        loss = (1 - alpha) * self.bce_loss(logits, y) + alpha * self.soft_macro_f1(logits, y)

        raw_probs        = torch.sigmoid(logits)
        consistent_preds = self.apply_hierarchy_constraint(raw_probs)

        self.val_f1(consistent_preds, y)
        self.log("val/loss",              loss,                                     prog_bar=True)
        self.log("val/macro_f1",          self.val_f1,                              prog_bar=True)
        self.log("val_macro_f1",          self.val_f1)
        self.log("val/graph_consistency", self.calculate_consistency_metric(consistent_preds), prog_bar=True)

    def apply_hierarchy_constraint(self, probs, threshold=0.5):
        N, C = probs.shape
        device = probs.device
        preds = torch.zeros(N, C, device=device)

        roots = self.roots.to(device)
        children_list = [c.to(device) for c in self.children_list]
        adj = self.adj_matrix.to(device)

        # Activate roots
        preds[:, roots] = (probs[:, roots] > threshold).float()

        # Top-down greedy: follow single best path
        for _ in range(63):
            prev = preds.clone()
            for i in range(C):
                children = children_list[i]
                if len(children) == 0:
                    continue
                active = preds[:, i].bool()
                if not active.any():
                    continue
                child_probs        = probs[active][:, children]
                best_vals, best_idx = child_probs.max(dim=1)
                above_thresh        = best_vals > threshold
                active_samples      = active.nonzero(as_tuple=True)[0]
                preds[active_samples[above_thresh], children[best_idx[above_thresh]]] = 1.0
            if (preds == prev).all():
                break

        # Bottom-up: if node j is active, all its parents must be active too.
        # Needed because ChEBI is a DAG — a node activated via one parent
        # leaves its other parents inactive, causing consistency violations.
        # (preds @ adj.T)[n, i] = number of active children of i → activates i if > 0
        for _ in range(63):
            preds = torch.max(preds, torch.matmul(preds, adj.T).clamp(0, 1))

        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/macro_f1"}
        }