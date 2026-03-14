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
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


# ── Fingerprint loading / computation ────────────────────────────────────────

def load_or_compute(pt_path, smiles_list: list, radius: int, n_bits: int = 2048) -> np.ndarray:
    if pt_path and os.path.exists(pt_path):
        print(f"Loading fingerprints from {pt_path}")
        return torch.load(pt_path).numpy()
    print("No .pt file found — computing fingerprints from SMILES…")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = []
    for smi in tqdm(smiles_list, desc="Computing fingerprints"):
        mol = Chem.MolFromSmiles(smi)
        fps.append(
            gen.GetFingerprintAsNumPy(mol).astype(np.uint8)
            if mol is not None else np.zeros(n_bits, dtype=np.uint8)
        )
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
    os.makedirs(out_dir, exist_ok=True)

    if isinstance(adj_matrix, torch.Tensor):
        adj_np = adj_matrix.cpu().numpy()
    else:
        adj_np = adj_matrix

    # ── Split ─────────────────────────────────────────────────────────────────
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
    X_all   = load_or_compute(fp_path, train_df["SMILES"].tolist(), radius=cfg.radius)
    X_train = X_all[train_idx]   # integer indices — safe regardless of DataFrame index
    X_val   = X_all[val_idx]

    Y_train = train_split[label_cols].values.astype(np.float32)
    Y_val   = val_split[label_cols].values.astype(np.float32)

    # ── XGBoost hyperparams from config ───────────────────────────────────────
    xgb_params = dict(
        n_estimators     = cfg.model.get("n_estimators", 300),
        max_depth        = cfg.model.get("max_depth", 6),
        learning_rate    = cfg.model.get("learning_rate", 0.1),
        subsample        = cfg.model.get("subsample", 0.8),
        colsample_bytree = cfg.model.get("colsample_bytree", 0.8),
        eval_metric      = "logloss",
        tree_method      = "hist",
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

    skipped = 0
    for idx, col in enumerate(tqdm(label_cols, desc="Training classifiers")):
        y_tr = Y_train[:, idx]
        y_va = Y_val[:, idx]

        # Skip if either split is constant — can happen for very rare classes
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            val_preds_raw[:, idx] = np.unique(y_tr)[0]
            models.append(None)
            skipped += 1
            continue

        clf = XGBClassifier(**xgb_params)
        clf.fit(X_train, y_tr, eval_set=[(X_val, y_va)], verbose=False)
        val_preds_raw[:, idx] = clf.predict(X_val)
        models.append(clf)

        if (idx + 1) % 50 == 0:
            f1_so_far = f1_score(Y_val[:, :idx+1], val_preds_raw[:, :idx+1], average="macro", zero_division=0)
            wandb.log({"val/macro_f1_partial": f1_so_far, "classes_trained": idx + 1})

    if skipped:
        print(f"Skipped {skipped} constant classes (too rare to stratify).")

    # ── Hierarchy constraint + final metrics ──────────────────────────────────
    val_preds = apply_hierarchy_constraint_np(val_preds_raw, adj_np)

    macro_f1 = f1_score(Y_val, val_preds, average="macro", zero_division=0)
    micro_f1 = f1_score(Y_val, val_preds, average="micro", zero_division=0)

    print(f"\nVal macro-F1: {macro_f1:.4f}  |  micro-F1: {micro_f1:.4f}")
    wandb.log({"val/macro_f1": macro_f1, "val/micro_f1": micro_f1})
    run.finish()

    # ── Save ──────────────────────────────────────────────────────────────────
    save_path = os.path.join(out_dir, "models.joblib")
    joblib.dump(models, save_path)
    print(f"Saved {len(models)} models → {save_path}")

    return models
