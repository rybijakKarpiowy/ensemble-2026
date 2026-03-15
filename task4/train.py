from __future__ import annotations

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import ECGPhotoDataset, SyntheticECGDataset
from model import TinyUNet


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2, 3)) + eps
    den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps
    return 1 - (num / den).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--records", nargs="*", default=[], help="Ścieżki do plików .hea używanych do syntetycznego generowania danych treningowych")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="weights_ecg_trace.pt")
    ap.add_argument("--steps-per-epoch", type=int, default=64)
    ap.add_argument("--base", type=int, default=16)
    args = ap.parse_args()

    if not args.images and not args.records:
        raise SystemExit("Podaj przynajmniej --images albo --records")

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    datasets = []
    if args.images:
        datasets.append(ECGPhotoDataset(args.images))
    if args.records:
        datasets.append(SyntheticECGDataset(args.records))

    if len(datasets) == 1:
        ds = datasets[0]
    else:
        ds = torch.utils.data.ConcatDataset(datasets)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = TinyUNet(in_ch=1, out_ch=1, base=args.base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        denom = 0
        for step, (x, y) in enumerate(dl):
            if step >= args.steps_per_epoch:
                break
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            bce = F.binary_cross_entropy_with_logits(logits, y)
            loss = 0.5 * bce + 0.5 * dice_loss(logits, y)
            loss.backward()
            opt.step()
            total += float(loss.item())
            denom += 1
        print(f"epoch {epoch+1}/{args.epochs} loss={total/max(1, denom):.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f"Zapisano wagi: {out}")


if __name__ == "__main__":
    main()
