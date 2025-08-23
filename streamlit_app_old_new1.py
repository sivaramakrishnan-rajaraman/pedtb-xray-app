# streamlit_app.py  — drop-in replacement
from __future__ import annotations

import os
import io
import numpy as np
import streamlit as st
from PIL import Image
import torch
import cv2

from ultralytics import YOLO

from src.hf_utils import hf_download_robust
from src.pneumonia_model import PneumoniaModel
from src.cam_utils import (
    compute_cam_mask,
    overlay_heatmap_on_bgr,
    contours_and_boxes_on_bgr,
)

st.set_page_config(page_title="PedTB X-ray Demo", layout="wide")

# -------------------------
# Hugging Face config (public)
# -------------------------
HF_MODEL_REPO_YOLO = st.secrets.get("HF_MODEL_REPO_YOLO", "sivaramakrishhnan/cxr-yolo12s-lung")
HF_FILENAME_YOLO   = st.secrets.get("HF_FILENAME_YOLO",   "best.pt")   # use .pt as you asked

HF_MODEL_REPO_DPN  = st.secrets.get("HF_MODEL_REPO_DPN",  "sivaramakrishhnan/cxr-dpn68-tb-cls")
HF_FILENAME_DPN    = st.secrets.get("HF_FILENAME_DPN",    "dpn68_fold2.ckpt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Classifier preprocessing
# -------------------------
def preprocess_cxr_rgb_to_tensor(rgb: np.ndarray, size: int = 224) -> torch.Tensor:
    # Resize to classifier input and normalize (ImageNet stats)
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img)[None]  # (1,3,224,224)

# -------------------------
# Model loaders (cached)
# -------------------------
@st.cache_resource(show_spinner="Downloading YOLO (.pt) from Hugging Face…")
def get_yolo() -> YOLO:
    # Try common filenames; you confirmed best.pt exists
    yolo_path = hf_download_robust(
        repo_id=HF_MODEL_REPO_YOLO,
        filename_or_list=[HF_FILENAME_YOLO, "best.pt", "weights.pt", "model.pt"],
        repo_type="model",
        token=None,
    )
    model = YOLO(yolo_path)  # Ultralytics will handle letterbox + scaling internally
    return model

@st.cache_resource(show_spinner="Downloading DPN-68 checkpoint from Hugging Face…")
def get_classifier() -> PneumoniaModel:
    ckpt_path = hf_download_robust(
        repo_id=HF_MODEL_REPO_DPN,
        filename_or_list=[HF_FILENAME_DPN, "dpn68_fold2.ckpt", "best.ckpt"],
        repo_type="model",
        token=None,
    )
    h = {
        "model": "dpn68_new",
        "img_size": 224,
        "batch_size": 64,
        "num_workers": 2,
        "dropout": 0.3,
        "num_classes": 2,
        "pin_memory": True,
        "lr": 5e-5,
        "max_epochs": 64,
        "patience": 10,
        "balance": True,
    }
    model = PneumoniaModel.load_from_checkpoint(ckpt_path, h=h, strict=False, map_location=DEVICE)
    model.to(DEVICE).eval()
    return model

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Detection")
conf = st.sidebar.slider("YOLO confidence", 0.0, 1.0, 0.25, 0.01)
iou  = st.sidebar.slider("YOLO IoU (NMS)", 0.10, 0.95, 0.75, 0.01)
imgsz = st.sidebar.select_slider("YOLO imgsz", options=[512, 640, 768], value=640)

st.sidebar.header("Grad-CAM")
cam_method = st.sidebar.selectbox(
    "CAM method",
    ["gradcam", "gradcam++", "xgradcam", "layercam", "eigencam", "eigengradcam", "hirescam"],
    index=0,
)
cam_alpha = st.sidebar.slider("Heat alpha", 0.0, 1.0, 0.5, 0.05)
cam_thr   = st.sidebar.slider("Contour threshold", 0.0, 1.0, 0.4, 0.05)

st.sidebar.header("Display")
display_size = st.sidebar.select_slider("Display width (px)", options=[224, 384, 512, 768], value=512)

# -------------------------
# Title + uploader
# -------------------------
st.title("🩺 Pediatric TB X-ray – Detection • Classification • Grad-CAM")

up = st.file_uploader("Upload a chest X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])
if not up:
    st.info("Upload an image to begin.")
    st.stop()

orig = Image.open(io.BytesIO(up.read())).convert("RGB")
orig_rgb = np.array(orig)  # HxWx3 RGB
H0, W0 = orig_rgb.shape[:2]
disp_h = int(H0 * (display_size / float(W0)))
st.image(cv2.resize(orig_rgb, (display_size, disp_h)), caption="Original", use_column_width=False)

# -------------------------
# Load models
# -------------------------
yolo = get_yolo()
clf  = get_classifier()

# -------------------------
# 1) YOLO detection on ORIGINAL image size
#     Ultralytics expects BGR np.ndarray; it letterboxes internally and
#     returns boxes in ORIGINAL coordinates (xyxy).
# -------------------------
with st.spinner("Detecting lungs (Ultralytics)…"):
    orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
    res = yolo.predict(
        source=orig_bgr,       # np.ndarray BGR in original resolution
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
        device=0 if DEVICE == "cuda" else "cpu",
    )[0]

if res.boxes is None or len(res.boxes) == 0:
    st.error("No lungs detected.")
    st.stop()

# Boxes in original pixel coords
xyxy = res.boxes.xyxy.cpu().numpy()        # (N,4)
scores = res.boxes.conf.cpu().numpy()      # (N,)
# Pick highest-confidence (single class)
k = int(np.argmax(scores))
x1, y1, x2, y2 = xyxy[k]
score = float(scores[k])

# Visualize raw YOLO box
raw_vis = orig_rgb.copy()
cv2.rectangle(raw_vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
st.image(
    cv2.resize(raw_vis, (display_size, disp_h)),
    caption=f"YOLO detection (conf {score:.2f}) — coords in ORIGINAL image space",
    use_column_width=False,
)

# Clamp & crop from ORIGINAL
x1i, y1i, x2i, y2i = int(max(0, round(x1))), int(max(0, round(y1))), int(min(W0-1, round(x2))), int(min(H0-1, round(y2)))
if x2i <= x1i: x2i = min(W0-1, x1i+1)
if y2i <= y1i: y2i = min(H0-1, y1i+1)

crop_rgb = orig_rgb[y1i:y2i, x1i:x2i].copy()
if crop_rgb.size == 0:
    st.error("Crop is empty. Detector box was degenerate.")
    st.stop()

# -------------------------
# 2) Classify crop (resize to 224×224 for model)
# -------------------------
inp = preprocess_cxr_rgb_to_tensor(crop_rgb, size=224).to(DEVICE)
with torch.no_grad():
    logits = clf(inp)
    probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred   = int(probs.argmax())

classes = ["normal", "normal_not"]
st.markdown(f"**Prediction:** {classes[pred]}  |  **P(normal_not)** = {probs[1]:.4f}")

# -------------------------
# 3) Grad-CAM at 224×224, then resize CAM mask to the CROP’s ORIGINAL size
# -------------------------
cam_mask_224 = compute_cam_mask(
    model=clf,
    input_tensor=inp,           # (1,3,224,224)
    class_index=1,              # abnormal class
    method=cam_method,
    aug_smooth=True,
    eigen_smooth=True,
)  # float32 in [0,1], shape (224,224)

# Resize CAM to the crop’s resolution for *accurate* overlay
crop_h, crop_w = crop_rgb.shape[:2]
cam_mask_crop = cv2.resize((cam_mask_224 * 255).astype(np.uint8), (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
cam_mask_crop = cam_mask_crop.astype(np.float32) / 255.0  # back to [0,1]

crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
cam_on_crop = overlay_heatmap_on_bgr(
    base_bgr=crop_bgr,
    cam_mask=cam_mask_crop,
    alpha=cam_alpha,
    colormap=cv2.COLORMAP_HOT,
)
cont_on_crop, box_on_crop = contours_and_boxes_on_bgr(
    base_bgr=crop_bgr,
    cam_mask=cam_mask_crop,
    threshold=cam_thr,
    color=(0, 0, 255),
    thickness=3,
)

# OPTIONAL: paste overlays back into full image (so you see context on original)
full_heat = orig_bgr.copy()
full_cont = orig_bgr.copy()
full_box  = orig_bgr.copy()
full_heat[y1i:y2i, x1i:x2i] = cam_on_crop
full_cont[y1i:y2i, x1i:x2i] = cont_on_crop
full_box[y1i:y2i, x1i:x2i]  = box_on_crop

def show_resized_bgr(img_bgr: np.ndarray, caption: str):
    h, w = img_bgr.shape[:2]
    disp_h2 = int(h * (display_size / float(w)))
    st.image(cv2.cvtColor(cv2.resize(img_bgr, (display_size, disp_h2)), cv2.COLOR_BGR2RGB),
             caption=caption, use_column_width=False)

st.subheader("Explainability on the CROPPED lungs (correct spatial mapping)")
show_resized_bgr(crop_bgr,     "Cropped lungs")
show_resized_bgr(cam_on_crop,  "Grad-CAM heatmap on crop")
show_resized_bgr(cont_on_crop, "Contours (CAM≥threshold)")
show_resized_bgr(box_on_crop,  "Bounding boxes (CAM≥threshold)")

st.subheader("(Optional) Overlays pasted into the ORIGINAL image")
show_resized_bgr(full_heat, "Heatmap in full image")
show_resized_bgr(full_cont, "Contours in full image")
show_resized_bgr(full_box,  "CAM boxes in full image")

st.success("Done.")
