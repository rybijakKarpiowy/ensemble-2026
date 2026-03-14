"""
gnn_model.py
------------
Multi-label GNN classifier using Graph Isomorphism Network (GIN).

Architecture:
    GIN layers with edge feature injection (via NNConv)
    → Global mean + max pooling (concatenated)
    → MLP head
    → 500 logits

GIN is the most expressive standard GNN (Xu et al., 2019) and consistently
outperforms GCN on molecular property prediction benchmarks.

Bond features are incorporated via a small edge MLP that projects them into
the node message space (same pattern used in MPNN / D-MPNN).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from torch_geometric.nn import NNConv, global_mean_pool, global_max_pool
from torchmetrics.classification import MultilabelF1Score

from task1.datasets.graph_dataset import ATOM_FEATURE_DIM, BOND_FEATURE_DIM


class GNNModel(L.LightningModule):
    """
    Args:
        num_classes:       Number of output labels (500).
        adj_matrix:        (num_classes, num_classes) hierarchy tensor.
        pos_weights:       (num_classes,) BCE imbalance weights.
        hidden_dim:        Width of GNN layers.
        num_layers:        Number of GIN message-passing layers.
        dropout:           Dropout rate in MLP head.
        lr:                Learning rate.
        hierarchy_depth:   Max depth for constraint propagation.
    """

    def __init__(
        self,
        num_classes:     int,
        adj_matrix:      torch.Tensor,
        pos_weights:     torch.Tensor = None,
        hidden_dim:      int = 256,
        num_layers:      int = 5,
        dropout:         float = 0.3,
        lr:              float = 1e-3,
        hierarchy_depth: int = 63,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["adj_matrix", "pos_weights"])

        # ── 1. Input projection ────────────────────────────────────────────────
        self.input_proj = nn.Linear(ATOM_FEATURE_DIM, hidden_dim)

        # ── 2. GIN layers with edge features via NNConv ────────────────────────
        # NNConv: for each edge, a small MLP maps bond features → (h_in × h_out)
        # matrix, then multiplies atom embeddings through it (= edge-conditioned conv).
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            edge_mlp = nn.Sequential(
                nn.Linear(BOND_FEATURE_DIM, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * hidden_dim),
            )
            self.convs.append(NNConv(hidden_dim, hidden_dim, edge_mlp, aggr="mean"))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        # ── 3. Readout MLP ─────────────────────────────────────────────────────
        # Concatenate mean + max pool → 2 × hidden_dim input
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # ── 4. Buffers ─────────────────────────────────────────────────────────
        self.register_buffer("adj_matrix", adj_matrix.float())
        if pos_weights is not None:
            self.register_buffer("pos_weights", pos_weights.float())
        else:
            self.register_buffer("pos_weights", torch.ones(num_classes))

        # ── 5. Metrics ─────────────────────────────────────────────────────────
        self.train_f1 = MultilabelF1Score(num_labels=num_classes, average="macro")
        self.val_f1   = MultilabelF1Score(num_labels=num_classes, average="macro")

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Args:
            x:          (total_atoms, ATOM_FEATURE_DIM)
            edge_index: (2, total_edges)
            edge_attr:  (total_edges, BOND_FEATURE_DIM)
            batch:      (total_atoms,) — PyG batch vector

        Returns:
            logits: (batch_size, num_classes)
        """
        h = F.relu(self.input_proj(x))

        for conv, norm in zip(self.convs, self.norms):
            h = F.relu(norm(conv(h, edge_index, edge_attr)))

        # Global pooling: concat mean and max for richer graph-level representation
        h_mean = global_mean_pool(h, batch)    # (B, hidden_dim)
        h_max  = global_max_pool(h, batch)     # (B, hidden_dim)
        h_graph = torch.cat([h_mean, h_max], dim=-1)   # (B, 2 * hidden_dim)

        return self.head(h_graph)   # (B, num_classes)

    # ── Losses (same as original) ─────────────────────────────────────────────

    def _bce_loss(self, logits, targets):
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weights
        )

    def _soft_macro_f1_loss(self, logits, targets):
        probs = torch.sigmoid(logits)
        tp  = (targets * probs).sum(dim=0)
        fp  = ((1 - targets) * probs).sum(dim=0)
        fn  = (targets * (1 - probs)).sum(dim=0)
        f1  = 2 * tp / (2 * tp + fp + fn + 1e-7)
        return 1 - f1.mean()

    # ── Steps ─────────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        logits = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y      = batch.y.squeeze(1)    # (B, num_classes)
        loss   = self._bce_loss(logits, y) + self._soft_macro_f1_loss(logits, y)

        self.train_f1(torch.sigmoid(logits), y.int())
        self.log("train/loss",     loss,          on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/macro_f1", self.train_f1, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y      = batch.y.squeeze(1)
        loss   = self._bce_loss(logits, y)

        raw_probs        = torch.sigmoid(logits)
        consistent_preds = self.apply_hierarchy_constraint(raw_probs)

        self.val_f1(consistent_preds, y.int())
        consistency = self.calculate_consistency_metric(consistent_preds)

        self.log("val/loss",              loss,           prog_bar=True)
        self.log("val/macro_f1",          self.val_f1,    prog_bar=True)
        self.log("val_macro_f1",          self.val_f1)
        self.log("val/graph_consistency", consistency,    prog_bar=True)

    # ── Hierarchy constraint (preserved from original) ────────────────────────

    def calculate_consistency_metric(self, preds):
        active_children = torch.matmul(preds, self.adj_matrix.T)
        violations      = (preds == 0) & (active_children > 0)
        return (~violations.any(dim=1)).float().mean()

    def apply_hierarchy_constraint(self, probs, threshold=0.5):
        preds       = (probs > threshold).float()
        has_parents = (self.adj_matrix.sum(dim=0) > 0).float()
        depth       = self.hparams.hierarchy_depth

        for _ in range(depth):
            parent_active = (torch.matmul(preds, self.adj_matrix) > 0).float()
            mask          = (1 - has_parents) + (has_parents * parent_active)
            preds         = preds * mask

        for _ in range(depth):
            preds = torch.max(preds, torch.matmul(preds, self.adj_matrix.T).clamp(0, 1))

        return preds

    # ── Optimiser ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
        return {
            "optimizer":    optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/macro_f1"},
        }
