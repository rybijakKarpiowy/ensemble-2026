"""
dataset.py
----------
Dataset and DataModule for SMILES multi-label classification.

Expected data format (CSV or DataFrame):
    smiles   | label_0 | label_1 | ... | label_499
    "CC(=O)…"  0         1               0
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L
import pandas as pd
from transformers import AutoTokenizer


SMILES_ENCODER_NAME = "seyonec/ChemBERTa-zinc-base-v1"


# ── Dataset ───────────────────────────────────────────────────────────────────

class SMILESDataset(Dataset):
    """
    Args:
        df:         DataFrame with a 'smiles' column and 500 binary label columns.
        tokenizer:  HuggingFace tokenizer for the SMILES encoder.
        max_length: Token sequence length (128 is sufficient for most SMILES).
        label_cols: List of label column names (ordered by class_id).
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128, label_cols: list = None):
        self.smiles    = df["smiles"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

        if label_cols is None:
            label_cols = [c for c in df.columns if c != "smiles"]
        self.labels = torch.tensor(df[label_cols].values, dtype=torch.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.smiles[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return (
            encoding["input_ids"].squeeze(0),        # (max_length,)
            encoding["attention_mask"].squeeze(0),   # (max_length,)
            self.labels[idx],                        # (num_classes,)
        )


# ── DataModule ────────────────────────────────────────────────────────────────

class SMILESDataModule(L.LightningDataModule):
    """
    Args:
        train_df / val_df / test_df: DataFrames (see SMILESDataset).
        encoder_name:  HuggingFace model ID for the tokenizer.
        batch_size:    Training batch size.
        num_workers:   DataLoader workers.
        max_length:    Max token length for SMILES strings.
        label_cols:    Ordered list of label column names.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df:   pd.DataFrame,
        test_df:  pd.DataFrame = None,
        encoder_name: str = SMILES_ENCODER_NAME,
        batch_size:   int = 64,
        num_workers:  int = 4,
        max_length:   int = 128,
        label_cols:   list = None,
    ):
        super().__init__()
        self.train_df     = train_df
        self.val_df       = val_df
        self.test_df      = test_df
        self.encoder_name = encoder_name
        self.batch_size   = batch_size
        self.num_workers  = num_workers
        self.max_length   = max_length
        self.label_cols   = label_cols

    def setup(self, stage=None):
        tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)

        self.train_dataset = SMILESDataset(self.train_df, tokenizer, self.max_length, self.label_cols)
        self.val_dataset   = SMILESDataset(self.val_df,   tokenizer, self.max_length, self.label_cols)
        if self.test_df is not None:
            self.test_dataset = SMILESDataset(self.test_df, tokenizer, self.max_length, self.label_cols)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True,  num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size * 2,
                          shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        if self.test_df is None:
            raise ValueError("No test_df provided.")
        return DataLoader(self.test_dataset, batch_size=self.batch_size * 2,
                          shuffle=False, num_workers=self.num_workers, pin_memory=True)


# ── Utility: compute pos_weights from training labels ─────────────────────────

def compute_pos_weights(train_df: pd.DataFrame, label_cols: list, clip: float = 50.0) -> torch.Tensor:
    """
    Inverse-frequency weighting per class.  Clipped at `clip` to avoid extreme values.
    """
    counts    = train_df[label_cols].sum(axis=0).values          # positives per class
    n         = len(train_df)
    neg       = n - counts
    weights   = torch.tensor(neg / (counts + 1e-6), dtype=torch.float32)
    return weights.clamp(max=clip)
