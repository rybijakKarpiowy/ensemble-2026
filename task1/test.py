import hydra
from omegaconf import DictConfig
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as L
from skfp.fingerprints import ECFPFingerprint
from tqdm import tqdm
import os
from task1.model import HierarchicalChemicalClassifier
from task1.utils import prepare_hierarchy_and_weights


class InferenceDataset(Dataset):
    def __init__(self, features):
        self.features = features
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx]

def run_inference(input_parquet, output_parquet, checkpoint_path, radius=2, adj_matrix_path="task1/data/chebi_classes.obo"):
    # 1. Load Test Data
    print(f"Loading test data from {input_parquet}...")
    train_df = pd.read_parquet("task1/data/chebi_dataset_train.parquet") # Needed for hierarchy and weights
    test_df = pd.read_parquet(input_parquet)
    
    # 2. Generate Fingerprints (Match training params: 2048, radius 2)
    print("Generating fingerprints...")
    fp_transformer = ECFPFingerprint(fp_size=2048, radius=radius, n_jobs=-1)
    # Using a cache check here is good, but for the final submission, we run fresh
    fps = fp_transformer.transform(test_df['SMILES'].tolist())
    fps = torch.tensor(fps, dtype=torch.float32)
    
    # 3. Load Model from Checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {checkpoint_path}...")
    adj_matrix, pos_weights = prepare_hierarchy_and_weights(adj_matrix_path, train_df, [f"class_{i}" for i in range(500)])
    
    # We load with map_location to ensure it works on CPU if no GPU is available
    model = HierarchicalChemicalClassifier(2048, 500, adj_matrix, pos_weights) # Dummy init to load state dict
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.freeze() # No gradients needed
    model.to(device)

    # 4. Batch Inference
    loader = DataLoader(InferenceDataset(fps), batch_size=1024, shuffle=False, num_workers=4)
    all_preds = []

    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits)
            
            # Apply the hierarchy constraint (The Tie-Breaker)
            # This ensures that if a child is predicted, all parents are too.
            consistent_preds = model.apply_hierarchy_constraint(probs, threshold=0.5)
            # TODO: check with consistency constraint or not for final submission
            # consistent_preds = (probs > 0.5)
            all_preds.append(consistent_preds.cpu().numpy())

    # 5. Prepare Submission DataFrame
    predictions = np.vstack(all_preds)
    
    # Get the class column names from the model hyperparameters or a known list
    # Assuming the classes are named class_0 ... class_499
    num_classes = predictions.shape[1]
    class_cols = [f"class_{i}" for i in range(num_classes)]
    
    # Create the result dataframe
    result_df = pd.DataFrame(predictions, columns=class_cols, index=test_df.index)
    
    # Join with the original test_df to include SMILES and any other required columns
    result_df = test_df.join(result_df)
    
    print("Inference completed. Sample predictions:")
    # print(result_df.head())

    # 6. Save to Parquet
    print(f"Saving predictions to {output_parquet}...")
    result_df.to_parquet(output_parquet, index=False)
    print("Done!")

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    input_file = "task1/data/chebi_dataset_test_empty.parquet"
    output_file = "task1/data/submission.parquet"
    checkpoint_folder = "task1/checkpoints/"
    
    # Get the last modified checkpoint folder (assuming each run creates a new folder)
    checkpoint_folders = [f for f in os.listdir(checkpoint_folder) if os.path.isdir(os.path.join(checkpoint_folder, f))]
    if not checkpoint_folders:
        raise FileNotFoundError(f"No checkpoint folders found in {checkpoint_folder}")
    latest_folder = max(checkpoint_folders, key=lambda x: os.path.getmtime(os.path.join(checkpoint_folder, x)))
    checkpoint_folder = os.path.join(checkpoint_folder, latest_folder)
    
    # Look for the best checkpoint in the folder (based on filename pattern)
    checkpoint_files = [f for f in os.listdir(checkpoint_folder) if f.endswith(".ckpt") and f.startswith("model")]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_folder}")
    # Assuming the best checkpoint is the one with "model" in the name and highest val_macro_f1
    best_checkpoint = sorted(checkpoint_files, key=lambda x: float(x.split("val_macro_f1=")[-1].split(".ckpt")[0]), reverse=True)[0]
    checkpoint_path = os.path.join(checkpoint_folder, best_checkpoint)
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Extract radius
    radius_str = latest_folder.split("radius_")[-1]
    radius = int(radius_str) if radius_str.isdigit() else 2 # Default to 2 if not found
    
    run_inference(input_file, output_file, checkpoint_path, radius=radius)