from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from pipeline import process_ecg_image, make_submission_record, SHORT_LEADS, LONG_LEAD_KEY

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def auto_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_images(input_path: str, recursive: bool = False) -> list[Path]:
    p = Path(input_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        iterator = p.rglob("*") if recursive else p.glob("*")
        return sorted(x for x in iterator if x.is_file() and x.suffix.lower() in IMAGE_EXTS)
    matches = sorted(Path().glob(input_path))
    return [m for m in matches if m.is_file() and m.suffix.lower() in IMAGE_EXTS]


def write_csv(csv_path: Path, record_name: str, signals: dict[str, dict]) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["record", "lead", "sample_idx", "value_mv"])
        for lead, payload in signals.items():
            for idx, y in enumerate(payload["signal_mv"]):
                writer.writerow([record_name, lead, idx, y])


def main() -> None:
    ap = argparse.ArgumentParser(description="ECG digitization + submission.npz generator")
    ap.add_argument("--input", required=True, help="Plik obrazu, katalog lub glob z obrazami testowymi")
    ap.add_argument("--weights", default=None, help="Opcjonalne wagi segmentera")
    ap.add_argument("--submission", default="submission.npz", help="Ścieżka wynikowego pliku submission.npz")
    ap.add_argument("--out-dir", default=None, help="Katalog na artefakty debugowe; używany tylko z --save-debug")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--recursive", action="store_true", help="Rekurencyjne przeszukiwanie katalogu wejściowego")
    ap.add_argument("--save-debug", action="store_true", help="Zapisz obrazy pośrednie, overlaye i JSON/CSV")
    ap.add_argument("--include-long-lead", action="store_true", default=True,
                    help=f"Dołącz {LONG_LEAD_KEY} do submission (domyślnie włączone)")
    ap.add_argument("--no-long-lead", dest="include_long_lead", action="store_false",
                    help=f"Nie dołączaj {LONG_LEAD_KEY} do submission")
    args = ap.parse_args()

    device = auto_device() if args.device == "auto" else args.device
    images = resolve_images(args.input, recursive=args.recursive)
    if not images:
        raise SystemExit(f"Nie znaleziono obrazów dla: {args.input}")

    if args.save_debug:
        out_dir = Path(args.out_dir or "outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    submission_dict: dict[str, np.ndarray] = {}
    processed = 0
    for image_path in images:
        record_name = image_path.stem
        record_out_dir = (out_dir / record_name) if out_dir is not None else None
        result = process_ecg_image(
            image_path=image_path,
            weights_path=args.weights,
            out_dir=record_out_dir,
            device=device,
            save_artifacts=record_out_dir is not None,
            include_long_lead=args.include_long_lead,
        )
        signals = result["signals"]
        submission_dict.update(
            make_submission_record(
                record_name=record_name,
                signals=signals,
                include_long_lead=args.include_long_lead,
            )
        )
        if record_out_dir is not None:
            write_csv(record_out_dir / f"{record_name}_signals_500hz.csv", record_name, signals)
        processed += 1
        print(f"[{processed}/{len(images)}] {record_name}")

    submission_path = Path(args.submission)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(submission_path, **submission_dict)

    lead_count = len(SHORT_LEADS) + (1 if args.include_long_lead else 0)
    print(f"Zapisano: {submission_path}")
    print(f"Przetworzono rekordów: {processed}")
    print(f"Leadów na rekord: {lead_count}")
    print(f"Łącznie tablic w .npz: {len(submission_dict)}")
    print(f"Urządzenie: {device}")


if __name__ == "__main__":
    main()
