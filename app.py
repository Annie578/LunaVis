"""Streamlit UI: annotate candidate objects and run two-stage CNN inference."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
st.set_page_config(page_title="LunaVis | Crater Lab", page_icon="🌙", layout="wide")
st.title("🌙 LunaVis — Lunar Object & Crater Detector")
st.write("Stage 1 proposes surface objects; stage 2 classifies each proposal as crater or not crater.")
st.info("The Keio dataset labels rocks, not craters. Training requires labeled crater and non-crater objects; Keio imagery alone cannot supervise this task.")

uploaded = st.file_uploader("Upload lunar surface image", type=["png", "jpg", "jpeg", "tif", "tiff"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    w, h = image.size
    left, right = st.columns(2)
    with left:
        st.image(image, caption=f"{w} × {h}", use_container_width=True)
    with right:
        st.subheader("Label a candidate object")
        st.caption("Annotate both craters and non-crater objects (e.g. rocks). Coordinates/radius are pixels.")
        with st.form("annotation"):
            x = st.number_input("Center X", min_value=0, max_value=max(0, w-1), value=w//2)
            y = st.number_input("Center Y", min_value=0, max_value=max(0, h-1), value=h//2)
            radius = st.number_input("Approximate radius", min_value=1, max_value=max(1, max(w, h)), value=max(1, min(w,h)//20))
            label = st.selectbox("Class", ["crater", "not_crater"])
            add = st.form_submit_button("Add labeled object")
        if "annotations" not in st.session_state:
            st.session_state.annotations = []
        if add:
            st.session_state.annotations.append({"filename": uploaded.name, "x_center": int(x), "y_center": int(y), "radius": int(radius), "label": label})
            st.success("Object label added.")
        if st.session_state.annotations:
            table = pd.DataFrame(st.session_state.annotations)
            st.dataframe(table, use_container_width=True)
            st.download_button("Download annotations CSV", table.to_csv(index=False), "annotations.csv", "text/csv")

st.divider()
st.subheader("Two-stage CNN inference")
detector_path, classifier_path = ROOT / "models/object_proposal.pt", ROOT / "models/crater_classifier.pt"
if not detector_path.exists() or not classifier_path.exists():
    st.warning("Train both CNN stages first. See README for annotation and training steps.")
elif uploaded:
    try:
        import cv2
        import torch
        from src.models.custom_cnn import CraterClassifierCNN, ObjectProposalCNN
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        @st.cache_resource
        def load_models(detector_file, classifier_file, device_name):
            dev = torch.device(device_name)
            det, clf = ObjectProposalCNN().to(dev), CraterClassifierCNN().to(dev)
            det.load_state_dict(torch.load(detector_file, map_location=dev, weights_only=True))
            clf.load_state_dict(torch.load(classifier_file, map_location=dev, weights_only=True))
            return det.eval(), clf.eval()
        detector, classifier = load_models(str(detector_path), str(classifier_path), str(device))
        size = 256
        resized = image.resize((size, size), Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.asarray(resized, dtype=np.float32).transpose(2,0,1)/255.).unsqueeze(0).to(device)
        threshold = st.slider("Object proposal threshold", 0.1, 0.9, 0.5)
        with torch.no_grad():
            probability = torch.sigmoid(detector(tensor))[0,0].cpu().numpy()
        binary = (probability >= threshold).astype(np.uint8)
        count, _, stats, centers = cv2.connectedComponentsWithStats(binary, 8)
        min_area = st.slider("Minimum proposal area (pixels at model resolution)", 1, 1000, 12)
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        results = []
        for i in range(1, count):
            x0, y0, bw, bh, area = map(int, stats[i])
            if area < min_area: continue
            pad = max(4, int(max(bw,bh)*0.25))
            x1, y1 = max(0,x0-pad), max(0,y0-pad)
            x2, y2 = min(size,x0+bw+pad), min(size,y0+bh+pad)
            crop = resized.crop((x1,y1,x2,y2)).resize((128,128), Image.Resampling.BILINEAR)
            crop_t = torch.from_numpy(np.asarray(crop,dtype=np.float32).transpose(2,0,1)/255.).unsqueeze(0).to(device)
            with torch.no_grad():
                scores = torch.softmax(classifier(crop_t), dim=1)[0]
            cls = int(scores.argmax().item())
            scale_x, scale_y = w/size, h/size
            box = (int(x0*scale_x), int(y0*scale_y), int((x0+bw)*scale_x), int((y0+bh)*scale_y))
            color = "lime" if cls == 1 else "orange"
            draw.rectangle(box, outline=color, width=3)
            draw.text((box[0], max(0,box[1]-16)), f"{'crater' if cls else 'not crater'} {scores[cls]:.2f}", fill=color)
            results.append({"class":"crater" if cls else "not_crater", "confidence":round(float(scores[cls]),3), "box_xyxy":box})
        st.image(canvas, caption="Green = crater; orange = not crater", use_container_width=True)
        st.dataframe(pd.DataFrame(results), use_container_width=True)
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
