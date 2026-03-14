"""
xgboost_trainer.py
------------------
Multi-label XGBoost classifier using Morgan fingerprints.
Trains one XGBoost classifier per class (OneVsRest style),
with optional hierarchy constraint applied at inference.

Usage (called from train.py when cfg.model.type == "xgboost"):
    from task1.models.xgboost_trainer import train_xgboost
    train_xgboost(cfg, train_df, label_cols, adj_matrix)
"""

import os
import numpy as np
import torch
import wandb
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit



# ── Fingerprint computation ───────────────────────────────────────────────────

def compute_fingerprints(smiles_list: list, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    fps = []
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    for smi in tqdm(smiles_list, desc="Computing fingerprints"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.uint8))
        else:
            fp = gen.GetFingerprintAsNumPy(mol)
            fps.append(np.array(fp, dtype=np.uint8))
    return np.stack(fps)


# ── Hierarchy constraint (CPU / numpy) ───────────────────────────────────────

def apply_hierarchy_constraint_np(preds: np.ndarray, adj_matrix: np.ndarray, depth: int = 63) -> np.ndarray:
    """Bottom-up: if child=1, all ancestors must be 1."""
    preds = preds.copy()
    for _ in range(depth):
        prev = preds.copy()
        preds = np.clip(preds + (preds @ adj_matrix.T > 0).astype(float), 0, 1)
        if np.array_equal(preds, prev):
            break
    return preds


# ── Main training function ────────────────────────────────────────────────────

def train_xgboost(cfg, train_df, label_cols, adj_matrix, out_dir: str = "task1/checkpoints/xgboost"):
    """
    Train one XGBClassifier per label, evaluate on val split, log to WandB.
 
    Args:
        cfg:        Hydra config (uses cfg.radius, cfg.model, cfg.data.val_size, cfg.random_seed)
        train_df:   Full training DataFrame with 'SMILES' + label columns
        label_cols: Ordered list of label column names
        adj_matrix: (num_classes, num_classes) torch.Tensor or numpy array
        out_dir:    Directory to save fitted models
    """
    os.makedirs(out_dir, exist_ok=True)
 
    if isinstance(adj_matrix, torch.Tensor):
        adj_np = adj_matrix.cpu().numpy()
    else:
        adj_np = adj_matrix
 
    # ── Split ──────────────────────────────────────────────────────────────────
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=cfg.random_seed
    )
    Y_all = train_df[label_cols].values
    train_idx, val_idx = next(msss.split(train_df, Y_all))
    train_split = train_df.iloc[train_idx]
    val_split   = train_df.iloc[val_idx]
    print(f"Split — train: {len(train_split):,}  val: {len(val_split):,}")
 
    # ── Fingerprints ──────────────────────────────────────────────────────────
    fp_path = cfg.model.get("fingerprints_path", None)
    X_all   = compute_fingerprints(train_df["SMILES"].tolist(), radius=cfg.radius)
    # Re-index to match the split (train_test_split preserves original index)
    X_train = X_all[train_split.index]
    X_val   = X_all[val_split.index]
 
    Y_train = train_split[label_cols].values.astype(np.float32)  # (N, 500)
    Y_val   = val_split[label_cols].values.astype(np.float32)
 
    # ── XGBoost hyperparams from config ───────────────────────────────────────
    xgb_params = dict(
        n_estimators     = cfg.model.get("n_estimators", 300),
        max_depth        = cfg.model.get("max_depth", 6),
        learning_rate    = cfg.model.get("learning_rate", 0.1),
        subsample        = cfg.model.get("subsample", 0.8),
        colsample_bytree = cfg.model.get("colsample_bytree", 0.8),
        use_label_encoder= False,
        eval_metric      = "logloss",
        tree_method      = "hist",   # fast on CPU; use "gpu_hist" if GPU available
        device           = cfg.model.get("device", "cpu"),
        n_jobs           = cfg.model.get("n_jobs", -1),
    )
 
    # ── Train one classifier per class ────────────────────────────────────────
    val_preds_raw = np.zeros_like(Y_val)
    models = []
 
    run = wandb.init(
        project = "Ensemble-2026-task1",
        name    = f"xgboost_radius_{cfg.radius}",
        config  = dict(xgb_params, radius=cfg.radius, val_size=0.2),
    )
 
    for idx, col in enumerate(tqdm(label_cols, desc="Training classifiers")):
        y_tr = Y_train[:, idx]
        y_va = Y_val[:, idx]
 
        clf = XGBClassifier(**xgb_params)
        clf.fit(X_train, y_tr, eval_set=[(X_val, y_va)], verbose=False)
 
        val_preds_raw[:, idx] = clf.predict(X_val)
        models.append(clf)
 
        # Log per-class F1 every 50 classes to keep WandB traffic low
        if (idx + 1) % 50 == 0:
            f1_so_far = f1_score(Y_val[:, :idx+1], val_preds_raw[:, :idx+1], average="macro", zero_division=0)
            wandb.log({"val/macro_f1_partial": f1_so_far, "classes_trained": idx + 1})
 
    # ── Hierarchy constraint + final metrics ──────────────────────────────────
    val_preds = apply_hierarchy_constraint_np(val_preds_raw, adj_np)
 
    macro_f1 = f1_score(Y_val, val_preds, average="macro",  zero_division=0)
    micro_f1 = f1_score(Y_val, val_preds, average="micro",  zero_division=0)
 
    print(f"\nVal macro-F1: {macro_f1:.4f}  |  micro-F1: {micro_f1:.4f}")
    wandb.log({"val/macro_f1": macro_f1, "val/micro_f1": micro_f1})
    run.finish()
 
    # ── Save ──────────────────────────────────────────────────────────────────
    save_path = os.path.join(out_dir, "models.joblib")
    joblib.dump(models, save_path)
    print(f"Saved {len(models)} models → {save_path}")
 
    return models