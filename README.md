# 🌙 LunaVis

A computer-vision project for lunar surface analysis, with a custom CNN crater-segmentation baseline and a Streamlit web app.

## What is included

- `app.py`: public-facing Streamlit interface for image upload, simple crater-circle annotation, CSV export, and CNN inference when a trained checkpoint is present.
- `src/models/custom_cnn.py`: custom U-Net-style CNN for binary crater-mask segmentation.
- `src/dataset/download_keio.py`: optional download helper for the Keio Artificial Lunar Rocky Landscape dataset.
- `src/dataset/annotations_to_masks.py`: converts circle annotations into segmentation masks.
- `src/training/train.py`: minimal training loop for paired image/mask data.

**Dataset limitation:** The Keio dataset contains rock/terrain labels, not crater labels. It can be explored as lunar imagery, but it does not by itself provide ground truth to train a crater detector. The CNN requires crater annotations/masks. The included annotation workflow provides a starting point; validate labels and gather a suitable quantity of data before drawing scientific conclusions.

## Run locally

Requires Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Download the Keio dataset

Create a Kaggle account, accept the dataset's terms on its [Kaggle page](https://www.kaggle.com/datasets/romainpessia/artificial-lunar-rocky-landscape-dataset), install/configure the Kaggle CLI, then run:

```bash
pip install kaggle
python -m src.dataset.download_keio --output data/raw/keio
```

Do not commit the downloaded data or Kaggle credentials. Check the dataset card for current licensing and attribution terms; the dataset is not bundled with LunaVis.

## Prepare annotations and train

1. Copy your selected RGB images into `data/processed/images/`.
2. In the Streamlit app, upload an image and enter crater center/radius annotations. Export the CSV as `crater_annotations.csv` and place it in the repository root. The app expects the uploaded image filename to match a file in the images directory.
3. Generate binary masks and train:

```bash
python -m src.dataset.annotations_to_masks --annotations crater_annotations.csv
python -m src.training.train --epochs 20
```

Training expects image files in `data/processed/images/` and PNG masks with matching basenames in `data/processed/masks/`. It saves `models/crater_unet.pt`; the Streamlit app then enables inference. The current training script is a baseline without validation or augmentation; add held-out evaluation before relying on model results.

## Deploy publicly

Push this repository to GitHub, then create an app on [Streamlit Community Cloud](https://share.streamlit.io/) and select `app.py` as the entry point. The repository includes Streamlit server configuration. Add a trained `models/crater_unet.pt` artifact using an appropriate model hosting/storage solution for production; model weights and datasets are intentionally ignored by Git. Public hosting means uploaded imagery may be processed by the hosted app, so publish a privacy notice and avoid retaining user uploads.

## License and attribution

Review the upstream dataset's license and attribution requirements before redistributing data or derived artifacts. This code scaffold does not grant rights to the dataset.
