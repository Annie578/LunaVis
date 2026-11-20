"""LunaVis: upload lunar imagery, annotate craters, and run a trained CNN."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="LunaVis | Crater Lab", page_icon="🌙", layout="wide")
st.title("🌙 LunaVis — Lunar Crater Lab")
st.write("Explore lunar images, create crater annotations, and test a custom CNN model.")
st.info("The Keio Artificial Lunar Rocky Landscape dataset contains rock labels, not crater labels. Add crater annotations before training a crater detector.")

uploaded = st.file_uploader("Upload a lunar surface image", type=["png", "jpg", "jpeg", "tif", "tiff"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    arr = np.asarray(image)
    left, right = st.columns(2)
    with left:
        st.image(image, caption=f"{image.width} × {image.height}", use_container_width=True)
    with right:
        st.subheader("Add a crater annotation")
        st.caption("Enter the crater center and radius in pixels. Repeat for each crater in this image.")
        with st.form("annotation"):
            x = st.number_input("Center X (pixels)", min_value=0, max_value=max(0, image.width - 1), value=image.width // 2)
            y = st.number_input("Center Y (pixels)", min_value=0, max_value=max(0, image.height - 1), value=image.height // 2)
            radius = st.number_input("Radius (pixels)", min_value=1, max_value=max(1, max(image.size)), value=max(1, min(image.size) // 20))
            add = st.form_submit_button("Add annotation")
        if "annotations" not in st.session_state:
            st.session_state.annotations = []
        if add:
            st.session_state.annotations.append({"filename": uploaded.name, "x_center": int(x), "y_center": int(y), "radius": int(radius)})
            st.success("Annotation added.")
        rows = [r for r in st.session_state.annotations if r["filename"] == uploaded.name]
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.download_button("Download annotations CSV", pd.DataFrame(st.session_state.annotations).to_csv(index=False), "crater_annotations.csv", "text/csv")

st.divider()
st.subheader("CNN inference")
model_file = ROOT / "models" / "crater_unet.pt"
if not model_file.exists():
    st.warning("No trained checkpoint found yet. Annotate images, convert annotations to masks, then train using the README instructions.")
elif uploaded:
    try:
        import torch
        from src.models.custom_cnn import CraterUNet
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_file, map_location=device, weights_only=True)
        model = CraterUNet().to(device)
        model.load_state_dict(checkpoint)
        model.eval()
        tensor = torch.from_numpy(arr.astype("float32").transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
        with torch.no_grad():
            probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
        threshold = st.slider("Detection threshold", 0.1, 0.9, 0.5)
        overlay = arr.copy()
        overlay[probability >= threshold] = (255, 70, 50)
        st.image(overlay, caption="CNN crater-mask overlay", use_container_width=True)
    except Exception as exc:
        st.error(f"Could not load or run the checkpoint: {exc}")
