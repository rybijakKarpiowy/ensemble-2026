"""
generate_masks.py
-----------------
Generates binary segmentation masks from ECG JSON annotations for UNet training.

Pipeline per sample:
  1. Load .png + .json
  2. Un-rotate the image (JSON coords are in pre-rotation space)
  3. Splat each plotted_pixel as a 1-px centerline
  4. Dilate by DILATION_RADIUS to match the rendered line width
  5. Save image + mask pair (full-res and/or 512×512)

Usage:
    python generate_masks.py --input_dir /data/ecg_train \
                             --output_dir /data/ecg_masks \
                             --size 512 \
                             --dilation 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_DILATION = 5   # px radius  →  kernel = (2*r+1)²
DEFAULT_SIZE     = 512 # UNet input edge length (0 = keep full-res)
# ──────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np

def detect_rotation_angle(image_path):
    # 1. Load and grayscale
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Edge detection (Canny)
    # We use a wide range to catch the grid lines
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # 3. Hough Line Transform
    # Adjust threshold and minLineLength based on your image resolution
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                            minLineLength=100, maxLineGap=10)
    
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate angle in degrees
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            # We only care about nearly horizontal lines (e.g., -15 to 15 degrees)
            if -15 < angle < 15:
                angles.append(angle)
    
    # 4. Return the median angle (more robust than mean)
    return np.median(angles) if angles else 0.0

def find_best_shift(img_unrot_arr: np.ndarray, json_mask: np.ndarray) -> tuple[int, int]:
    """
    Finds the (dx, dy) shift that maximizes the overlap between 
    the JSON-generated mask and the actual lines in the image.
    """
    # 1. Prepare the 'search' image (high values where lines are)
    gray = cv2.cvtColor(img_unrot_arr, cv2.COLOR_RGB2GRAY)
    # Invert so dark ECG lines become bright (high signal)
    _, target = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # 2. Prepare the 'template' (the JSON mask we already built)
    # We'll take a central crop to avoid border artifacts
    h, w = json_mask.shape
    template = (json_mask.astype(np.uint8) * 255)
    
    # 3. Match Template
    # This returns a map of correlation scores
    res = cv2.matchTemplate(target, template, cv2.TM_CCORR_NORMED)
    
    # 4. Find the peak location
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # max_loc gives the top-left corner where the match is best
    # Since we matched a template of the same size (or similar), 
    # we calculate the deviation from the origin (0,0)
    # Note: If using a smaller template, you'd subtract the crop offset.
    dx = max_loc[0]
    dy = max_loc[1]
    
    return dx, dy

import cv2
import numpy as np

def build_mask(json_path: Path, dilation_radius: int = DEFAULT_DILATION) -> tuple[np.ndarray, np.ndarray]:
    with open(json_path) as f:
        meta = json.load(f)

    img_path = json_path.with_suffix(".png")
    # Load with OpenCV for processing
    cv_img = cv2.imread(str(img_path))
    H, W = cv_img.shape[:2]

    # --- 1. Detect and Undo Rotation ---
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    detected_skew = 0.0
    if lines is not None:
        angles = [np.degrees(np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0])) for l in lines]
        # Filter for near-horizontal lines to find paper tilt
        horiz_angles = [a for a in angles if -15 < a < 15]
        if horiz_angles:
            detected_skew = np.median(horiz_angles)

    # Calculate total rotation (JSON base + physical skew)
    total_angle = detected_skew
    
    # Rotate image using PIL (to keep your existing pipeline compatible)
    img_pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    img_unrot = img_pil.rotate(total_angle, expand=False, fillcolor=(0, 0, 0))
    img_unrot_arr = np.array(img_unrot)

    # --- 2. Initial "Splat" (Base JSON Mask) ---
    # We create a temporary mask to use as a template for alignment
    base_mask = np.zeros((H, W), dtype=np.uint8)
    for lead in meta.get("leads", []):
        for px in lead["plotted_pixels"]:
            r, c = int(round(px[0])), int(round(px[1]))
            if 0 <= r < H and 0 <= c < W:
                base_mask[r, c] = 255

    # --- 3. Find Best Shift (Translation) ---
    # Prepare the target: Invert grayscaled unrotated image so lines are bright
    target_gray = cv2.cvtColor(img_unrot_arr, cv2.COLOR_RGB2GRAY)
    _, target_bin = cv2.threshold(target_gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Use template matching to find the (dx, dy) offset
    # We use a smaller search area if performance is an issue
    res = cv2.matchTemplate(target_bin, base_mask, cv2.TM_CCORR)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    
    # max_loc returns (x, y) which is (column_shift, row_shift)
    # Since our template was the same size as the image, the 'ideal' 
    # match at 0 shift would be at (0,0) in some CV2 versions, 
    # but practically we calculate the delta from center or origin.
    # For a same-size template, max_loc gives the top-left shift.
    dy, dx = max_loc[1], max_loc[0] 

    # --- 4. Final Mask Generation (with Shift) ---
    final_mask = np.zeros((H, W), dtype=bool)
    for lead in meta.get("leads", []):
        for px in lead["plotted_pixels"]:
            # Apply the detected shift to the JSON coordinates
            r = int(round(px[0] + dy))
            c = int(round(px[1] + dx))
            if 0 <= r < H and 0 <= c < W:
                final_mask[r, c] = True

    # --- 5. Dilate ---
    if dilation_radius > 0:
        k = 2 * dilation_radius + 1
        final_mask = binary_dilation(final_mask, structure=np.ones((k, k), dtype=bool))

    return img_unrot_arr, final_mask

def save_pair(img_arr: np.ndarray, mask: np.ndarray,
              out_img: Path, out_mask: Path,
              size: int = DEFAULT_SIZE) -> None:
    """Resize (if size > 0) and save image + mask pair."""
    img_pil  = Image.fromarray(img_arr)
    mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)

    if size > 0:
        img_pil  = img_pil.resize((size, size), Image.LANCZOS)
        mask_pil = mask_pil.resize((size, size), Image.NEAREST)

    img_pil.save(out_img)
    mask_pil.save(out_mask)


def process_directory(input_dir: str, output_dir: str,
                      size: int = DEFAULT_SIZE,
                      dilation: int = DEFAULT_DILATION,
                      save_overlay: bool = False) -> None:
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)

    img_dir  = output_dir / "images"
    mask_dir = output_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"No .json files found in {input_dir}")
        return

    print(f"Processing {len(json_files)} samples  →  {output_dir}")

    for i, jf in enumerate(json_files, 1):
        stem = jf.stem
        try:
            img_arr, mask = build_mask(jf, dilation_radius=dilation)
        except Exception as e:
            print(f"  [{i}/{len(json_files)}] SKIP {stem}: {e}")
            continue

        save_pair(
            img_arr, mask,
            out_img=img_dir / f"{stem}.png",
            out_mask=mask_dir / f"{stem}.png",
            size=size,
        )

        if save_overlay:
            ov_dir = output_dir / "overlays"
            ov_dir.mkdir(exist_ok=True)
            ov = img_arr.copy().astype(float)
            ov[mask, 0] = ov[mask, 0] * 0.2 + 220 * 0.8
            ov[mask, 1] = ov[mask, 1] * 0.2 + 60  * 0.8
            ov[mask, 2] = ov[mask, 2] * 0.2 + 60  * 0.8
            ov_pil = Image.fromarray(ov.clip(0, 255).astype(np.uint8))
            if size > 0:
                ov_pil = ov_pil.resize((size, size), Image.LANCZOS)
            ov_pil.save(ov_dir / f"{stem}.png")

        if i % 50 == 0 or i == len(json_files):
            print(f"  {i}/{len(json_files)}  {stem}")

    print("Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate UNet masks from ECG JSON annotations")
    parser.add_argument("--input_dir",    default=".",       help="Directory with .json + .png files")
    parser.add_argument("--output_dir",   default="./masks", help="Output root directory")
    parser.add_argument("--size",         type=int, default=DEFAULT_SIZE,
                        help="Resize edge length (0 = full-res)")
    parser.add_argument("--dilation",     type=int, default=DEFAULT_DILATION,
                        help="Mask dilation radius in pixels")
    parser.add_argument("--overlay",      action="store_true",
                        help="Also save image+mask overlay PNGs for visual QC")
    args = parser.parse_args()

    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        size=args.size,
        dilation=args.dilation,
        save_overlay=args.overlay,
    )

def main():
    process_directory(
        input_dir="ecg_dataset/train",
        output_dir="ecg_masks",
        size=512,
        dilation=5,
        save_overlay=True,
    )
