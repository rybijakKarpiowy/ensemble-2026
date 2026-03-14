import pronto
import torch
import pandas as pd
import numpy as np

def prepare_hierarchy_and_weights(obo_path, train_df, label_cols):
    """
    Args:
        obo_path: Path to your .obo file.
        train_df: The training DataFrame.
        label_cols: List of the 500 column names (e.g., ['class_0', ...]).
    """
    # 1. Parse Ontology
    print(f"Parsing {obo_path}...")
    ont = pronto.Ontology(obo_path)
    
    # Mapping ID -> Index (Assuming column names match IDs in the .obo file)
    # If your columns are 'class_0', you might need a mapping dictionary.
    id_to_idx = {cid: i for i, cid in enumerate(label_cols)}
    num_classes = len(label_cols)
    
    # 2. Extract Adjacency Matrix
    adj_matrix = torch.zeros((num_classes, num_classes))
    
    for term in ont.terms():
        if term.id in id_to_idx:
            child_idx = id_to_idx[term.id]
            # Get direct parents (distance=1)
            for parent in term.superclasses(distance=1, with_self=False):
                if parent.id in id_to_idx:
                    parent_idx = id_to_idx[parent.id]
                    adj_matrix[parent_idx, child_idx] = 1.0

    # 3. Calculate pos_weights for Sparse Classes
    # Formula: (Total Samples - Positive Samples) / Positive Samples
    print("Calculating class weights...")
    pos_counts = train_df[label_cols].sum(axis=0).values
    total_samples = len(train_df)
    
    # We add a small epsilon to avoid division by zero
    # and clamp to prevent extreme gradients from super-rare leaves.
    pos_weights = (total_samples - pos_counts) / (pos_counts + 1e-6)
    pos_weights = torch.tensor(pos_weights, dtype=torch.float32)
    pos_weights = torch.clamp(pos_weights, min=1.0, max=100.0)

    print(f"Hierarchy extracted: {int(adj_matrix.sum())} active edges.")
    return adj_matrix, pos_weights