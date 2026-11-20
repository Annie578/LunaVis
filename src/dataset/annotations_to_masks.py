"""LunaVis: custom two-stage CNN for lunar object proposals and crater classification.

Annotation CSV columns: filename,x_center,y_center,radius,label where label is
'crater' or 'not_crater'. This helper creates proposal masks containing all
annotated candidates; classification uses the labels on individual crops.
"""
import argparse
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default="data/processed/images")
    parser.add_argument("--annotations", default="data/processed/annotations.csv")
    parser.add_argument("--output", default="data/processed/proposal_masks")
    args = parser.parse_args()
    image_dir, out = Path(args.images), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.annotations)
    required = {"filename", "x_center", "y_center", "radius", "label"}
    if not required.issubset(table.columns):
        raise ValueError(f"CSV must have columns: {', '.join(sorted(required))}")
    for filename, rows in table.groupby("filename"):
        path = image_dir / filename
        if not path.is_file():
            print(f"Skipping missing image: {path}"); continue
        mask = Image.new("L", Image.open(path).size, 0)
        draw = ImageDraw.Draw(mask)
        for row in rows.itertuples(index=False):
            x, y, r = int(row.x_center), int(row.y_center), int(row.radius)
            draw.ellipse((x-r, y-r, x+r, y+r), fill=255)
        mask.save(out / f"{path.stem}.png")
    print(f"Proposal masks written to {out}")


if __name__ == "__main__": main()
