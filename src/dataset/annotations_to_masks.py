"""Turn circle annotations (filename,x_center,y_center,radius) into PNG masks."""
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/processed/images")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", default="data/processed/masks")
    args = parser.parse_args()
    image_dir, output_dir = Path(args.images), Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.annotations)
    required = {"filename", "x_center", "y_center", "radius"}
    if not required.issubset(table.columns):
        raise ValueError(f"CSV must have columns: {', '.join(sorted(required))}")
    for filename, rows in table.groupby("filename"):
        image_path = image_dir / filename
        if not image_path.is_file():
            print(f"Skipping missing image: {image_path}"); continue
        image = Image.open(image_path)
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        for row in rows.itertuples(index=False):
            r, x, y = int(row.radius), int(row.x_center), int(row.y_center)
            draw.ellipse((x-r, y-r, x+r, y+r), fill=255)
        mask.save(output_dir / f"{image_path.stem}.png")
    print(f"Masks written to {output_dir}")


if __name__ == "__main__": main()
