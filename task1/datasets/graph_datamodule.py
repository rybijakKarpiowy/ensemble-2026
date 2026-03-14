"""
graph_datamodule.py
-------------------
Lightning DataModule that wraps MoleculeGraphDataset.
Handles train/val split internally, identical API to SMILESDataModule.
"""

import pandas as pd
import pytorch_lightning as L
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from task1.datasets.graph_dataset import MoleculeGraphDataset


class MoleculeGraphDataModule(L.LightningDataModule):
    """
    Args:
        df:          Full DataFrame with 'SMILES' + class_* label columns.
        label_cols:  Ordered list of label column names.
        val_size:    Fraction reserved for validation (default 0.1).
        batch_size:  Training batch size.
        num_workers: DataLoader workers.
        seed:        Random seed for reproducible split.
        test_df:     Optional held-out test DataFrame (no split applied).
    """

    def __init__(
        self,
        df:          pd.DataFrame,
        label_cols:  list = None,
        val_size:    float = 0.1,
        batch_size:  int = 64,
        num_workers: int = 4,
        seed:        int = 42,
        test_df:     pd.DataFrame = None,
    ):
        super().__init__()
        self.df          = df
        self.label_cols  = label_cols or [c for c in df.columns if c.startswith("class_")]
        self.val_size    = val_size
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.seed        = seed
        self.test_df     = test_df

    def setup(self, stage=None):
        train_df, val_df = train_test_split(
            self.df,
            test_size=self.val_size,
            random_state=self.seed,
            shuffle=True,
        )

        self.train_dataset = MoleculeGraphDataset(train_df, self.label_cols)
        self.val_dataset   = MoleculeGraphDataset(val_df,   self.label_cols)

        if self.test_df is not None:
            self.test_dataset = MoleculeGraphDataset(self.test_df, self.label_cols)

        print(f"Split — train: {len(self.train_dataset):,}  val: {len(self.val_dataset):,}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        if self.test_df is None:
            raise ValueError("No test_df provided to MoleculeGraphDataModule.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
