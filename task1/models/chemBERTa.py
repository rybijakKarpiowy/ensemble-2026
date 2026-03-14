"""
model.py
--------
Multi-label SMILES classifier with:
  - Frozen ChemBERTa encoder
  - Trainable MLP projector
  - Class description embeddings as label prototypes (knowledge base)
  - Cosine-similarity logits with learnable temperature
  - Hierarchical consistency constraint (from original model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from torchmetrics.classification import MultilabelF1Score
from transformers import AutoModel


SMILES_ENCODER_NAME = "seyonec/ChemBERTa-zinc-base-v1"


class SMILESDescriptionModel(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        adj_matrix: torch.Tensor,           # (num_classes, num_classes) — adj[i,j]=1 means i is parent of j
        label_embeddings: torch.Tensor,     # (num_classes, text_embed_dim) — precomputed, L2-normalised
        pos_weights: torch.Tensor = None,   # (num_classes,) for BCE imbalance correction
        lr: float = 1e-3,
        proj_dim: int = 512,                # hidden size of projector
        dropout: float = 0.2,
        smiles_encoder_name: str = SMILES_ENCODER_NAME,
        hierarchy_depth: int = 63,          # max tree depth for constraint propagation
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["adj_matrix", "pos_weights", "label_embeddings"])

        # ── 1. Frozen SMILES encoder ──────────────────────────────────────────
        self.smiles_encoder = AutoModel.from_pretrained(smiles_encoder_name)
        for param in self.smiles_encoder.parameters():
            param.requires_grad = False
        mol_dim   = self.smiles_encoder.config.hidden_size   # 768 for ChemBERTa
        label_dim = label_embeddings.shape[1]                # e.g. 768 for all-mpnet

        # ── 2. Trainable projector: mol_space → label_description_space ──────
        self.projector = nn.Sequential(
            nn.LayerNorm(mol_dim),
            nn.Linear(mol_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, label_dim),
        )

        # Learnable temperature: logits = exp(scale) * cosine_sim
        # Initialised at 2.0 → exp(2) ≈ 7.4 which is a good starting point
        self.logit_scale = nn.Parameter(torch.tensor(2.0))

        # ── 3. Buffers (auto-moved to GPU) ────────────────────────────────────
        self.register_buffer("adj_matrix",       adj_matrix.float())
        self.register_buffer("label_embeddings", F.normalize(label_embeddings.float(), dim=-1))

        if pos_weights is not None:
            self.register_buffer("pos_weights", pos_weights.float())
        else:
            self.register_buffer("pos_weights", torch.ones(num_classes))

        # ── 4. Metrics ────────────────────────────────────────────────────────
        self.train_f1 = MultilabelF1Score(num_labels=num_classes, average="macro")
        self.val_f1   = MultilabelF1Score(num_labels=num_classes, average="macro")

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns raw logits of shape (N, num_classes).
        """
        # [CLS] embedding from frozen ChemBERTa
        with torch.no_grad():
            cls_emb = self.smiles_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state[:, 0, :]                          # (N, 768)

        # Project into label space and L2-normalise
        mol_proj = F.normalize(self.projector(cls_emb), dim=-1)   # (N, label_dim)

        # Cosine similarity × temperature → logits
        scale  = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * (mol_proj @ self.label_embeddings.T)     # (N, num_classes)
        return logits

    # ── Losses ────────────────────────────────────────────────────────────────

    def _bce_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weights
        )

    def _soft_macro_f1_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Differentiable 1 - macro_F1, minimised jointly with BCE."""
        probs = torch.sigmoid(logits)
        tp    = (targets * probs).sum(dim=0)
        fp    = ((1 - targets) * probs).sum(dim=0)
        fn    = (targets * (1 - probs)).sum(dim=0)
        f1    = 2 * tp / (2 * tp + fp + fn + 1e-7)
        return 1 - f1.mean()

    # ── Steps ─────────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, y = batch
        logits = self(input_ids, attention_mask)
        loss   = self._bce_loss(logits, y) + self._soft_macro_f1_loss(logits, y)

        self.train_f1(torch.sigmoid(logits), y.int())
        self.log("train/loss",     loss,           on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/macro_f1", self.train_f1,  on_step=False, on_epoch=True)
        self.log("train/logit_scale", self.logit_scale.exp().item(), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, y = batch
        logits = self(input_ids, attention_mask)
        loss   = self._bce_loss(logits, y)

        raw_probs        = torch.sigmoid(logits)
        consistent_preds = self.apply_hierarchy_constraint(raw_probs)

        self.val_f1(consistent_preds, y.int())
        consistency = self.calculate_consistency_metric(consistent_preds)

        self.log("val/loss",              loss,            prog_bar=True)
        self.log("val/macro_f1",          self.val_f1,     prog_bar=True)
        self.log("val_macro_f1",          self.val_f1)     # monitor key for checkpoint
        self.log("val/graph_consistency", consistency,     prog_bar=True)

    # ── Hierarchy constraint (preserved from original) ────────────────────────

    def calculate_consistency_metric(self, preds: torch.Tensor) -> torch.Tensor:
        """Fraction of samples with zero hierarchical violations."""
        active_children = torch.matmul(preds, self.adj_matrix.T)
        violations      = (preds == 0) & (active_children > 0)
        return (~violations.any(dim=1)).float().mean()

    def apply_hierarchy_constraint(self, probs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Two-pass enforcement:
          1. Top-down:  a child can only be 1 if at least one parent is 1.
          2. Bottom-up: if a child is 1, its parent must be 1.
        """
        preds       = (probs > threshold).float()
        has_parents = (self.adj_matrix.sum(dim=0) > 0).float()   # (num_classes,)
        depth       = self.hparams.hierarchy_depth

        # Top-down pass
        for _ in range(depth):
            parent_active = (torch.matmul(preds, self.adj_matrix) > 0).float()
            mask          = (1 - has_parents) + (has_parents * parent_active)
            preds         = preds * mask

        # Bottom-up pass
        for _ in range(depth):
            preds = torch.max(preds, torch.matmul(preds, self.adj_matrix.T).clamp(0, 1))

        return preds

    # ── Optimiser ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        # Only the projector + logit_scale are trainable
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
        return {
            "optimizer":    optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/macro_f1"},
        }
