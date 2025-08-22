from __future__ import annotations
import os
import io
from typing import Optional, Dict

import streamlit as st
import numpy as np
import cv2
import torch
from PIL import Image

from ultralytics import YOLO
from huggingface_hub.utils import HfHubHTTPError

# local utils
from src.hf_utils import hf_download
from src.yolo_utils import detect_lungs, crop_with_box, draw_box
from src.model import PneumoniaModel
from src.cam_utils import cam_mask, overlay_and_shapes, CAM_METHODS

# ------------------ App Config ------------------
st.set_page_config(
    page_title="Pediatric CXR Classifier • YOLO→DPN68→XAI",
    page_icon="🫁",
    layout="wide"
)

st.title("🫁 Pediatric Chest X-ray Classifier • Lung detector → Classifier → XAI")
st.markdown(
"""
Upload a chest X-ray. The app will:

1. **Detect the lung region** with a fine-tuned **YOLO12s** (single class).
2. **Crop** to the detected lungs.
3. **Classify** (Normal vs Abnormal) with a fine-tuned **DPN-68**.
4. Show **Grad-CAM** (heatmap, contours, bboxes) for explainability.
"""
)

# ------------------ Constants / HF repos ------------------
# If your repos are public, token is not needed. If private, add in Streamlit secrets.
HF_TOKEN = st.secrets.get("HF_TOKEN", None)

REPO_YOLO = st.secrets.get("HF_MODEL_REPO_YOLO", "sivaramakrishhnan/cxr-yolo12s-lung")
FILE_YOLO = st.secrets.get("HF_FILENAME_YOLO",   "best.pt")

REPO_DPN  = st.secrets.get("HF_MODEL_REPO_DPN",  "sivaramakrishhnan/cxr-dpn68-tb-cls")
FILE_DPN  = st.secrets.get("HF_FILENAME_DPN",    "dpn68_fold2.ckpt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------ Caching model loads ------------------
@st.cache_resource(show_spinner="Downloading YOLO12s weights from Hugging Face…")
def load_yolo() -> YOLO:
    local = hf_download(REPO_YOLO, FILE_YOLO, repo_type="model", token=HF_TOKEN)
    return YOLO(local)

@st.cache_resource(show_spinner="Downloading DPN-68 checkpoint from Hugging Face…")
def load_dpn(hdict: Dict) -> PneumoniaModel:
    local = hf_download(REPO_DPN, FILE_DPN, repo_type="model", token=HF_TOKEN)
    # strict=False to be tolerant to small head differences
    model = PneumoniaModel.load_from_checkpoint(local, h=hdict, strict=False, map_location=DEVICE)
    model.to(DEVICE).eval()
    return model

# ------------------ Sidebar controls ------------------
st.sidebar.header("⚙️ Settings")

# Detector params
conf = st.sidebar.slider("YOLO confidence", 0.05, 0.95, 0.25, 0.01)
iou  = st.sidebar.slider("YOLO IoU",        0.05, 0.95, 0.75, 0.01)

# Classifier params (locked to 224x224)
img_size = 224
h_dict = {
    "model": "dpn68",          # name mapping handled in model.py
    "img_size": img_size,
    "dropout": 0.3,
    "num_classes": 2,
    "lr": 5e-5,
    "max_epochs": 1,           # not used in inference
}

# Grad-CAM params
cam_method = st.sidebar.selectbox(
    "CAM method",
    options=sorted(CAM_METHODS.keys()),
    index=sorted(CAM_METHODS.keys()).index("gradcam")
)
cam_alpha = st.sidebar.slider("CAM heatmap alpha", 0.0, 1.0, 0.5, 0.05)
cam_thr   = st.sidebar.slider("CAM threshold (contours)", 0.0, 1.0, 0.4, 0.05)

# ------------------ Image uploader ------------------
uploaded = st.file_uploader("Upload a pediatric chest X-ray", type=["png","jpg","jpeg","bmp","tif","tiff"])
col_a, col_b = st.columns(2)

if uploaded is not None:
    # Read to BGR for OpenCV routines
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if bgr is None:
        st.error("Could not read the image.")
        st.stop()
    bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)

    col_a.subheader("Original")
    col_a.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

    # 1) Load detector / classifier once (cached)
    try:
        yolo = load_yolo()
    except HfHubHTTPError as e:
        st.error(f"Could not download YOLO weights from HF: {e}")
        st.stop()
    try:
        dpn = load_dpn(h_dict)
    except HfHubHTTPError as e:
        st.error(f"Could not download DPN-68 ckpt from HF: {e}")
        st.stop()

    # 2) Detect lungs
    box = detect_lungs(yolo, bgr, conf=conf, iou=iou)
    if box is None:
        st.warning("No lung bounding box detected. Showing original image for classification.")
        crop = bgr.copy()
        det_vis = bgr
    else:
        crop = crop_with_box(bgr, box)
        det_vis = draw_box(bgr, box, (0,255,0), 3)

    col_b.subheader("Detection")
    col_b.image(cv2.cvtColor(det_vis, cv2.COLOR_BGR2RGB), use_column_width=True)

    # 3) Prepare classifier input (224×224, normalized)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_res = cv2.resize(crop_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    # normalize to ImageNet stats
    x = torch.from_numpy(crop_res).float().permute(2,0,1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    x = (x - mean) / std
    x = x.unsqueeze(0).to(DEVICE)

    # 4) Run classifier
    with torch.inference_mode():
        logits = dpn(x)
        probs  = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
    pred_idx = int(probs.argmax())
    label_map = {0: "Normal", 1: "Abnormal"}
    st.markdown(f"### Prediction: **{label_map[pred_idx]}**  (Normal={probs[0]:.3f}, Abnormal={probs[1]:.3f})")

    # 5) CAM
    mask_small = cam_mask(
        model=dpn,
        image_tensor=x,
        method=cam_method,
        class_idx=1,            # Abnormal class
        aug_smooth=True,
        eigen_smooth=True
    )
    # Resize mask to crop and generate overlays
    crop_bgr = cv2.cvtColor(crop_res, cv2.COLOR_RGB2BGR)
    overlay, cont_img, box_img = overlay_and_shapes(
        orig_bgr=crop_bgr,
        mask=mask_small,   # resized inside
        alpha=cam_alpha,
        threshold=cam_thr,
        contour_color=(0,0,255),
        contour_thickness=2
    )

    c1, c2, c3 = st.columns(3)
    c1.subheader("CAM Heatmap (crop)")
    c1.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_column_width=True)
    c2.subheader("Contours (crop)")
    c2.image(cv2.cvtColor(cont_img, cv2.COLOR_BGR2RGB), use_column_width=True)
    c3.subheader("Bounding Boxes (crop)")
    c3.image(cv2.cvtColor(box_img, cv2.COLOR_BGR2RGB), use_column_width=True)

    st.info("Tip: adjust YOLO conf/IoU and CAM method/alpha/threshold from the sidebar to explore behavior.")
else:
    st.write("⬆️ Upload a PNG/JPG/TIFF pediatric chest X-ray to begin.")
