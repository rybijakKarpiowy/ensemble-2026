"""
xgboost_predict.py
------------------
Inference with saved XGBoost models.

Usage:
    python task1/models/xgboost_predict.py \
        --models   task1/checkpoints/xgboost/models.joblib \
        --test_csv task1/data/chebi_dataset_test_empty.parquet \
        --out      task1/predictions/xgboost_preds.parquet \
        --adj      task1/data/adj_matrix.pt \
        --fp_path  task1/data/fingerprints_test.pt   # optional
"""

import argparse
import os
import numpy as np
import torch
import pandas as pd
import joblib
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from task1.models.xgboost_trainer import load_or_compute, apply_hierarchy_constraint_np


def predict(
    models: list,
    X: np.ndarray,
    adj_np: np.ndarray,
    label_cols: list,
) -> pd.DataFrame:
    """
    Run inference with a list of fitted XGBClassifiers.

    Args:
        models:     List of XGBClassifier | None (None = constant-class, always predicts 0)
        X:          (N, n_bits) fingerprint array
        adj_np:     (num_classes, num_classes) adjacency matrix
        label_cols: Ordered label column names

    Returns:
        DataFrame (N, num_classes) with binary predictions
    """
    preds_raw = np.zeros((len(X), len(models)), dtype=np.float32)

    for idx, clf in enumerate(tqdm(models, desc="Predicting")):
        if clf is None:
            # Was a constant class during training — predict 0
            preds_raw[:, idx] = 0.0
        else:
            preds_raw[:, idx] = clf.predict(X)

    preds = apply_hierarchy_constraint_np(preds_raw, adj_np)
    return pd.DataFrame(preds.astype(np.int8), columns=label_cols)


def test_xgboost(
    cfg,
    test_df: pd.DataFrame,
    label_cols: list,
    adj_matrix,
    models_path: str = "task1/checkpoints/xgboost/models.joblib",
    out_path: str = "task1/predictions/xgboost_preds.parquet",
    fp_path: str = None,
):
    """
    Load saved models, run predictions on test_df, save results.

    Args:
        cfg:         Hydra config (uses cfg.radius)
        test_df:     DataFrame with 'SMILES' column (labels optional — used for eval if present)
        label_cols:  Ordered list of label column names
        adj_matrix:  (num_classes, num_classes) torch.Tensor or numpy array
        models_path: Path to joblib file saved by train_xgboost
        out_path:    Where to save prediction parquet
        fp_path:     Optional .pt fingerprint cache for test set
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────────
    print(f"Loading models from {models_path} …")
    models = joblib.load(models_path)
    print(f"Loaded {len(models)} classifiers ({sum(m is None for m in models)} constant-class skips)")

    # ── Adjacency matrix ──────────────────────────────────────────────────────
    if isinstance(adj_matrix, torch.Tensor):
        adj_np = adj_matrix.cpu().numpy()
    else:
        adj_np = adj_matrix

    # ── Fingerprints ──────────────────────────────────────────────────────────
    X_test = load_or_compute(fp_path, test_df["SMILES"].tolist(), radius=cfg.radius)

    # ── Predict ───────────────────────────────────────────────────────────────
    pred_df = predict(models, X_test, adj_np, label_cols)

    # ── Evaluate if ground truth is available ─────────────────────────────────
    if all(c in test_df.columns for c in label_cols):
        from sklearn.metrics import f1_score
        Y_true = test_df[label_cols].values.astype(np.float32)
        macro_f1 = f1_score(Y_true, pred_df.values, average="macro", zero_division=0)
        micro_f1 = f1_score(Y_true, pred_df.values, average="micro", zero_division=0)
        print(f"Test macro-F1: {macro_f1:.4f}  |  micro-F1: {micro_f1:.4f}")
    else:
        print("No ground truth labels found — skipping evaluation.")

    # ── Save ──────────────────────────────────────────────────────────────────
    # Prepend mol_id if available so predictions can be joined back to the source
    if "mol_id" in test_df.columns:
        pred_df.insert(0, "mol_id", test_df["mol_id"].values)

    pred_df.to_parquet(out_path, index=False)
    print(f"Saved predictions → {out_path}  shape={pred_df.shape}")

    return pred_df


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",    required=True,  help="Path to models.joblib")
    parser.add_argument("--test_data", required=True,  help="Path to test parquet/csv")
    parser.add_argument("--adj",       required=True,  help="Path to adj_matrix.pt")
    parser.add_argument("--out",       default="task1/predictions/xgboost_preds.parquet")
    parser.add_argument("--fp_path",   default=None,   help="Optional .pt fingerprint cache")
    parser.add_argument("--radius",    type=int, default=2)
    args = parser.parse_args()

    # Minimal cfg shim so test_xgboost works outside Hydra
    class _Cfg:
        radius = args.radius
        class model:
            @staticmethod
            def get(k, default=None): return default

    test_df    = pd.read_parquet(args.test_data) if args.test_data.endswith(".parquet") else pd.read_csv(args.test_data)
    label_cols = [c for c in test_df.columns if c.startswith("class_")]
    adj_matrix = torch.load(args.adj)

    test_xgboost(
        cfg         = _Cfg(),
        test_df     = test_df,
        label_cols  = label_cols,
        adj_matrix  = adj_matrix,
        models_path = args.models,
        out_path    = args.out,
        fp_path     = args.fp_path,
    )
