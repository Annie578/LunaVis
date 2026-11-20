"""Train the two-stage candidate-object detector and crater/not-crater classifier."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.custom_cnn import CraterClassifierCNN, ObjectProposalCNN


class ObjectDataset(Dataset):
    def __init__(self, image_dir, annotations, size):
        self.image_dir, self.size = Path(image_dir), size
        self.images = sorted(p for p in self.image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})
        self.annotations = annotations

    def __len__(self): return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        image = Image.open(path).convert("RGB")
        w, h = image.size
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        rows = self.annotations[self.annotations.filename == path.name]
        for row in rows.itertuples(index=False):
            x, y, r = int(row.x_center), int(row.y_center), int(row.radius)
            draw.ellipse((x-r, y-r, x+r, y+r), fill=255)
        image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)
        x = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        y = torch.from_numpy((np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)[None])
        return x, y


class CropDataset(Dataset):
    def __init__(self, image_dir, annotations, size, padding=1.5):
        self.image_dir, self.annotations, self.size, self.padding = Path(image_dir), annotations, size, padding
        self.rows = []
        for row in annotations.itertuples(index=False):
            path = self.image_dir / row.filename
            if path.is_file() and row.label in {"crater", "not_crater"}:
                self.rows.append((path, int(row.x_center), int(row.y_center), int(row.radius), 1 if row.label == "crater" else 0))
        if not any(r[-1] == 0 for r in self.rows) or not any(r[-1] == 1 for r in self.rows):
            raise ValueError("Classifier training needs both crater and not_crater candidate annotations.")

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        path, x, y, radius, label = self.rows[index]
        image = Image.open(path).convert("RGB")
        half = max(4, int(radius * self.padding))
        crop = image.crop((x-half, y-half, x+half, y+half)).resize((self.size, self.size), Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.asarray(crop, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        return tensor, torch.tensor(label, dtype=torch.long)


def train_epoch(model, loader, optimizer, loss_fn, device, segmentation=False):
    model.train(); total = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        if segmentation:
            probs = torch.sigmoid(logits)
            loss = loss_fn(logits, labels) + 1 - (2 * (probs * labels).sum() + 1) / (probs.sum() + labels.sum() + 1)
        else:
            loss = loss_fn(logits, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default="data/processed/images")
    parser.add_argument("--annotations", default="data/processed/annotations.csv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--output", default="models")
    args = parser.parse_args()
    annotations = pd.read_csv(args.annotations)
    required = {"filename", "x_center", "y_center", "radius", "label"}
    if not required.issubset(annotations.columns):
        raise ValueError(f"Annotations CSV needs columns: {', '.join(sorted(required))}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    detector_loader = DataLoader(ObjectDataset(args.images, annotations, args.size), batch_size=args.batch_size, shuffle=True)
    detector = ObjectProposalCNN().to(device)
    optimizer = torch.optim.AdamW(detector.parameters(), lr=1e-3)
    for epoch in range(args.epochs):
        loss = train_epoch(detector, detector_loader, optimizer, nn.BCEWithLogitsLoss(), device, segmentation=True)
        print(f"proposal epoch {epoch+1}/{args.epochs} loss={loss:.4f}")
    torch.save(detector.cpu().state_dict(), out / "object_proposal.pt")

    crop_loader = DataLoader(CropDataset(args.images, annotations, args.size), batch_size=args.batch_size, shuffle=True)
    classifier = CraterClassifierCNN().to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3)
    for epoch in range(args.epochs):
        loss = train_epoch(classifier, crop_loader, optimizer, nn.CrossEntropyLoss(), device)
        print(f"classifier epoch {epoch+1}/{args.epochs} loss={loss:.4f}")
    torch.save(classifier.cpu().state_dict(), out / "crater_classifier.pt")
    print(f"Saved both model stages in {out}")


if __name__ == "__main__": main()
