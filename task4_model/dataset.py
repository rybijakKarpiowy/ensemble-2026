import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class ECGDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(256, 256)):
        """
        Args:
            image_dir: Folder containing the raw .jpg ECG images.
            mask_dir: Folder containing the generated _mask.png files.
            image_size: The target size for the U-Net (256x256 as per the paper).
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Setup file paths
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Mask names typically mirror image names (e.g., ecg_001.png -> ecg_001_mask.png)
        mask_name = img_name
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Load image (RGB) and mask (Grayscale)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize to 256x256 pixels
        image = cv2.resize(image, self.image_size)
        mask = cv2.resize(mask, self.image_size)

        # Normalize image to [0, 1] and mask to binary [0, 1]
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32) # Ensure strict binary

        # Convert to PyTorch Tensors: Shape becomes (Channels, Height, Width)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0) # Add channel dim

        return image_tensor, mask_tensor