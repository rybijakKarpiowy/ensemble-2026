"""
graph_dataset.py
----------------
Converts SMILES strings to PyTorch Geometric Data objects.

Atom features (9-dim per atom):
    - Atomic number (one-hot, top 44 elements + "other")
    - Degree (one-hot, 0-10)
    - Formal charge
    - Number of Hs (one-hot, 0-4)
    - Hybridization (one-hot: s, sp, sp2, sp3, sp3d, sp3d2)
    - Aromaticity (binary)
    - In ring (binary)

Bond features (4-dim per edge):
    - Bond type (one-hot: single, double, triple, aromatic)
    - Conjugated (binary)
    - In ring (binary)
    - Stereo (one-hot: none, any, Z, E)
"""

import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdchem
from torch_geometric.data import Data, Dataset
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
from typing import Optional


# ── Atom feature constants ─────────────────────────────────────────────────────

ATOM_TYPES = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br",
    "Mg", "Na", "Ca", "Fe", "As", "Al", "I", "B", "V",
    "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd", "Co", "Se",
    "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd",
    "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb", "other",
]  # 44 elements + other = 45

HYBRIDIZATION_TYPES = [
    rdchem.HybridizationType.S,
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]

BOND_TYPES = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]

STEREO_TYPES = [
    rdchem.BondStereo.STEREONONE,
    rdchem.BondStereo.STEREOANY,
    rdchem.BondStereo.STEREOZ,
    rdchem.BondStereo.STEREOE,
]

# Total feature dimensions (for reference in model)
ATOM_FEATURE_DIM = 45 + 11 + 1 + 5 + 6 + 1 + 1   # = 70
BOND_FEATURE_DIM = 4 + 1 + 1 + 4                   # = 10


# ── Feature helpers ────────────────────────────────────────────────────────────

def _one_hot(value, choices: list) -> list:
    """One-hot encode `value` against `choices`. Unknown values map to last bin."""
    encoding = [0] * len(choices)
    try:
        idx = choices.index(value)
    except ValueError:
        idx = len(choices) - 1   # "other" bucket
    encoding[idx] = 1
    return encoding


def atom_features(atom: rdchem.Atom) -> list:
    symbol = atom.GetSymbol()
    return (
        _one_hot(symbol, ATOM_TYPES)                                # 45
        + _one_hot(atom.GetDegree(), list(range(11)))               # 11  (0-10)
        + [atom.GetFormalCharge()]                                   # 1
        + _one_hot(atom.GetTotalNumHs(), list(range(5)))            # 5   (0-4)
        + _one_hot(atom.GetHybridization(), HYBRIDIZATION_TYPES)    # 6
        + [int(atom.GetIsAromatic())]                               # 1
        + [int(atom.IsInRing())]                                    # 1
    )                                                               # = 70


def bond_features(bond: rdchem.Bond) -> list:
    return (
        _one_hot(bond.GetBondType(), BOND_TYPES)    # 4
        + [int(bond.GetIsConjugated())]             # 1
        + [int(bond.IsInRing())]                    # 1
        + _one_hot(bond.GetStereo(), STEREO_TYPES)  # 4
    )                                               # = 10


# ── SMILES → PyG Data ──────────────────────────────────────────────────────────

def smiles_to_graph(smiles: str, y: Optional[torch.Tensor] = None) -> Optional[Data]:
    """
    Convert a SMILES string to a PyG Data object.

    Returns None if the molecule cannot be parsed (caller should filter these out).

    Args:
        smiles: SMILES string.
        y:      Label tensor (num_classes,) — optional.

    Returns:
        PyG Data with:
            x:          (num_atoms, ATOM_FEATURE_DIM) atom features
            edge_index: (2, num_edges * 2)  — undirected, each bond stored twice
            edge_attr:  (num_edges * 2, BOND_FEATURE_DIM) bond features
            y:          (1, num_classes) label tensor
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    x = torch.tensor(
        [atom_features(atom) for atom in mol.GetAtoms()],
        dtype=torch.float,
    )   # (num_atoms, 70)

    # Edges — PyG expects COO format; we add both directions for undirected graphs
    rows, cols, edge_attrs = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feats = bond_features(bond)
        rows  += [i, j]
        cols  += [j, i]
        edge_attrs += [feats, feats]   # same features for both directions

    if len(rows) == 0:
        # Single-atom molecule — no bonds; add a self-loop so GNN doesn't crash
        edge_index = torch.zeros((2, 1), dtype=torch.long)
        edge_attr  = torch.zeros((1, BOND_FEATURE_DIM), dtype=torch.float)
    else:
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr  = torch.tensor(edge_attrs, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if y is not None:
        data.y = y.unsqueeze(0)   # (1, num_classes)
    return data


# ── PyG Dataset ────────────────────────────────────────────────────────────────

class MoleculeGraphDataset(Dataset):
    """
    In-memory PyG Dataset built from a DataFrame.

    Args:
        df:         DataFrame with 'SMILES' column and class_* label columns.
        label_cols: Ordered list of label column names.
    """

    def __init__(self, df: pd.DataFrame, label_cols: list):
        super().__init__()
        self.data_list = []
        skipped = 0

        labels = torch.tensor(df[label_cols].values, dtype=torch.float32)

        for idx, row in enumerate(df.itertuples(index=False)):
            graph = smiles_to_graph(row.SMILES, y=labels[idx])
            if graph is None:
                skipped += 1
                continue
            self.data_list.append(graph)

        if skipped:
            print(f"[MoleculeGraphDataset] Skipped {skipped} unparseable SMILES.")

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
