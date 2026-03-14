import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Import your custom modules
from dataset import ECGDataset
from model import LitECGModel

def train_model():
    # --- Configuration ---
    IMAGE_DIR = 'ecg_dataset/train'
    MASK_DIR = 'ecg_dataset/train_masks'
    BATCH_SIZE = 8
    EPOCHS = 300 # Trained for 300 epochs 
    LR = 0.005   # Learning rate of 0.005 

    # --- Data Loading ---
    # Using the exact dataset class we defined previously
    dataset = ECGDataset(image_dir=IMAGE_DIR, mask_dir=MASK_DIR)
    
    # num_workers speeds up data loading. Set to 0 if you get Windows multiprocessing errors.
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # --- Initialize Lightning Model ---
    model = LitECGModel(learning_rate=LR)

    # --- Initialize PyTorch Lightning Trainer ---
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='auto', # Automatically finds your GPU (CUDA) or Apple Silicon (MPS)
        devices='auto',
        log_every_n_steps=1,
        enable_checkpointing=True # Automatically saves best models in a lightning_logs folder
    )

    # --- Train! ---
    print("Starting training with PyTorch Lightning...")
    trainer.fit(model, train_dataloaders=dataloader)



def main():
    train_model()
