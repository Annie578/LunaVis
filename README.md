# 🌙 LunaVis

Custom two-stage CNN for lunar surface object detection followed by crater classification.

## Pipeline

1. **Object proposal CNN:** a U-Net-style segmentation model finds annotated candidate objects and generates proposal regions.
2. **Crater classifier CNN:** a second CNN classifies each proposed crop as `crater` or `not_crater`.

This is a research baseline, not a pretrained or scientifically validated detector. The Keio Artificial Lunar Rocky Landscape dataset provides rock labels, not crater labels, so you must create/import crater and non-crater annotations before training.

## Install and run

Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## Prepare labeled examples

Place images under `data/processed/images/`. In the app, upload an image and annotate candidate object centers/radii; assign each `crater` or `not_crater`, then download the CSV and save it as `data/processed/annotations.csv`. The annotation filename must match the corresponding local image filename. Collect examples across many images, and include both classes. Image-level splitting and held-out evaluation should be added before treating metrics as meaningful.

The Keio images can contribute non-crater rock examples after manual verification, but rock annotations are not included in this scaffold's training format. Do not treat rocks as automatically exhaustive negatives.

## Train both stages

```bash
python -m src.training.train \\
  --images data/processed/images \\
  --annotations data/processed/annotations.csv \\
  --epochs 20 --batch-size 4
```

Training creates `models/object_proposal.pt` and `models/crater_classifier.pt`. Stage one learns a binary objectness mask from all annotated circles; stage two learns from labeled circle crops. The current pipeline uses connected components as simple proposal extraction, so nearby touching objects may merge. Improve proposal post-processing and add validation/augmentation for serious use.

## Keio dataset

Accept the terms on the [Kaggle dataset page](https://www.kaggle.com/datasets/romainpessia/artificial-lunar-rocky-landscape-dataset), configure Kaggle CLI credentials, then:

```bash
pip install kaggle
python -m src.dataset.download_keio --output data/raw/keio
```

Dataset is not bundled. Follow its current license and attribution terms; never commit Kaggle credentials or private data.

## Public deployment

Push to GitHub and deploy `app.py` using [Streamlit Community Cloud](https://share.streamlit.io/). Model checkpoints are ignored by Git; place them in a suitable model artifact store for a deployed instance and configure secure access as needed. Before accepting public uploads, publish a privacy notice and ensure uploads are not retained.
