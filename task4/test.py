import json
import wfdb
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- Configuration ---
# The base name of your files (without extensions)
file_base_name = 'ecg_dataset/train/ecg_train_0001'

def visualize_json_on_image():
    """
    Loads the JPEG and overlays the coordinates found in the JSON file 
    to verify the spatial ground truth.
    """
    print("Loading image and JSON...")
    
    # 1. Load the image
    png_path = f"{file_base_name}.png"
    jpg_path = f"{file_base_name}.jpg"
    img_path = png_path if Path(png_path).exists() else jpg_path
    # OpenCV loads images in BGR format, we convert to RGB for matplotlib
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(
            f"Could not read image. Tried: {png_path} and {jpg_path}"
        )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. Load the JSON metadata
    json_path = f"{file_base_name}.json"
    with open(json_path, 'r') as f:
        metadata = json.load(f)
        
    # 3. Setup matplotlib to show the image
    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.title("ECG Image with JSON Ground Truth Overlaid")
    
    # 4. Iterate through each lead and plot the pixels
    # We will use a different color for each lead to tell them apart
    colors = plt.cm.rainbow(np.linspace(0, 1, len(metadata['leads'])))
    
    for i, lead in enumerate(metadata['leads']):
        lead_name = lead['lead_name']
        pixels = np.array(lead['plotted_pixels'])
        
        # pixels[:, 0] are X coordinates, pixels[:, 1] are Y coordinates
        # Note: Image coordinates typically have (0,0) at top-left
        plt.scatter(pixels[:, 0], pixels[:, 1], color=colors[i], s=1, label=f"Lead {lead_name}")
        
    # Put a legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def visualize_wfdb_signal():
    """
    Loads the .dat and .hea files using the wfdb library and plots the 
    raw 1D time-series signal (the mathematical ground truth).
    """
    print("Loading WFDB files (.dat and .hea)...")
    
    # Read the record. wfdb automatically looks for both the .hea and .dat files
    record = wfdb.rdrecord(file_base_name)
    
    # The actual signal matrix is stored in record.p_signal
    # Shape is typically (Number of Samples, Number of Leads)
    signals = record.p_signal
    lead_names = record.sig_name
    
    # Let's plot just the first 3 leads to keep the graph readable
    leads_to_plot = 12
    
    plt.figure(figsize=(15, 6))
    for i in range(min(leads_to_plot, signals.shape[1])):
        # Create a time axis based on the sampling frequency
        time_axis = np.arange(signals.shape[0]) / record.fs 
        
        # Offset each lead slightly on the Y-axis so they don't overlap completely
        offset = i * 3.0  
        plt.plot(time_axis, signals[:, i] - offset, label=f"Lead: {lead_names[i]}")
        
    plt.title(f"Raw 1D Time-Series Signal (First {leads_to_plot} Leads)")
    plt.xlabel("Time (Seconds)")
    plt.ylabel("Voltage (mV) - Offset for visualization")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run the visualization functions
    visualize_json_on_image()
    visualize_wfdb_signal()