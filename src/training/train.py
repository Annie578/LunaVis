"""Train the custom CNN on paired RGB images and binary crater masks."""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.custom_cnn import CraterUNet


class MaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, size):
        self.image_dir, self.mask_dir, self.size = Path(image_dir), Path(mask_dir), size
        self.images = sorted(p for p in self.image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})
        self.images = [p for p in self.images if (self.mask_dir / (p.stem + ".png")).exists()]
        if not self.images:
            raise ValueError(f"No matching images and PNG masks found in {image_dir} and {mask_dir}")

    def __len__(self): return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        image = Image.open(path).convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
        mask = Image.open(self.mask_dir / (path.stem + ".png")).convert("L").resize((self.size, self.size), Image.Resampling.NEAREST)
        x = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        y = torch.from_numpy((np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)[None])
        return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/processed/images")
    parser.add_argument("--masks", default="data/processed/masks")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--output", default="models/crater_unet.pt")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(MaskDataset(args.images, args.masks, args.size), batch_size=args.batch_size, shuffle=True)
    model = CraterUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    for epoch in range(args.epochs):
        model.train(); total = 0.0
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            dice = 1 - (2 * (probs * masks).sum() + 1) / (probs.sum() + masks.sum() + 1)
            loss = bce(logits, masks) + dice
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item()
        print(f"epoch {epoch + 1}/{args.epochs} loss={total / len(loader):.4f}")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), output)
    print(f"Saved model to {output}")


if __name__ == "__main__": main()
