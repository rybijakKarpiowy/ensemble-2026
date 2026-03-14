import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L
from skfp.fingerprints import ECFPFingerprint

class FastChemicalDataset(Dataset):
    def __init__(self, features, labels=None):
        """
        features: Pre-calculated tensor (N, 2048)
        labels: Pre-calculated tensor (N, 500)
        """
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        if self.labels is not None:
            y = self.labels[idx]
            return x, y
        return x

class ChemicalDataModule(L.LightningDataModule):
    def __init__(self, df, label_cols, batch_size=1024, num_workers=4):
        super().__init__()
        self.df = df
        self.label_cols = label_cols
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fp_transformer = ECFPFingerprint(fp_size=2048, radius=2, n_jobs=-1)

    def setup(self, stage=None):
        # 1. Vectorize the entire dataset once (Multiprocessed via n_jobs=-1)
        print(f"Vectorizing {len(self.df)} molecules...")
        all_features = self.fp_transformer.transform(self.df['SMILES'].tolist())
        all_features = torch.tensor(all_features, dtype=torch.float32)
        
        all_labels = torch.tensor(self.df[self.label_cols].values, dtype=torch.float32)

        # 2. Split indices
        indices = np.arange(len(self.df))
        np.random.seed(42)
        np.random.shuffle(indices)
        
        split = int(0.8 * len(self.df))
        train_idx, val_idx = indices[:split], indices[split:]

        # 3. Create datasets with pre-sliced tensors
        self.train_ds = FastChemicalDataset(
            features=all_features[train_idx],
            labels=all_labels[train_idx]
        )

        self.val_ds = FastChemicalDataset(
            features=all_features[val_idx],
            labels=all_labels[val_idx]
        )
        print("Setup complete. Data moved to tensors.")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            pin_memory=True
        )