from __future__ import annotations

import argparse
from pathlib import Path
import csv
import torch

from pipeline import process_ecg_image


def auto_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    ap = argparse.ArgumentParser(description="Single-image ECG digitization")
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = ap.parse_args()

    device = auto_device() if args.device == "auto" else args.device
    result = process_ecg_image(args.image, args.weights, args.out_dir, device, save_artifacts=True)

    stem = Path(args.image).stem
    csv_path = Path(args.out_dir) / f"{stem}_signals_500hz.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lead", "sample_idx", "value_mv"])
        for lead, payload in result["signals"].items():
            for idx, y in enumerate(payload["signal_mv"]):
                writer.writerow([lead, idx, y])
    print(f"Gotowe. Wyniki zapisane do: {args.out_dir}")


if __name__ == "__main__":
    main()
