"""
precompute_labels.py
--------------------
Run once to encode your 500 class descriptions into fixed embeddings.
Saves a (500, embed_dim) tensor ordered by class_id.

Usage:
    python precompute_labels.py --csv classes.csv --out label_embeddings.pt

CSV format expected:
    class_id, descriptive_name, description
"""

import argparse
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


# ── Recommended text encoders (pick one) ──────────────────────────────────────
# "sentence-transformers/all-mpnet-base-v2"   → fast, general purpose (768-d)
# "allenai/specter2"                          → scientific/chemistry domain (768-d)
# "pritamdeka/S-PubMedBert-MS-MARCO"          → biomedical (768-d)
DEFAULT_TEXT_ENCODER = "sentence-transformers/all-mpnet-base-v2"


def build_text_for_class(row: pd.Series) -> str:
    """
    Combine name + description into a single string for encoding.
    Tweak this template to emphasise what matters most.
    """
    return f"{row['name']}: {row['definition']}"


def precompute(csv_path: str, out_path: str, encoder_name: str, num_classes: int):
    df = pd.read_csv(csv_path)
    df = df.sort_values("chebi_id").reset_index(drop=True)

    assert len(df) == num_classes, (
        f"Expected {num_classes} classes, got {len(df)}. "
        "Make sure every class_id from 0 to N-1 has a row."
    )

    texts = [build_text_for_class(row) for _, row in df.iterrows()]

    print(f"Loading text encoder: {encoder_name}")
    encoder = SentenceTransformer(encoder_name)

    print(f"Encoding {len(texts)} class descriptions …")
    # Returns numpy array (N, D)
    embeddings = encoder.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    label_emb = torch.tensor(embeddings)          # (500, D)
    torch.save(label_emb, out_path)
    print(f"Saved label embeddings → {out_path}  shape={label_emb.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",         default="classes.csv")
    parser.add_argument("--out",         default="label_embeddings.pt")
    parser.add_argument("--encoder",     default=DEFAULT_TEXT_ENCODER)
    parser.add_argument("--num_classes", type=int, default=500)
    args = parser.parse_args()

    precompute(args.csv, args.out, args.encoder, args.num_classes)
