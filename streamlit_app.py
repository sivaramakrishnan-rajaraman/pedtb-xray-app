# streamlit_app.py
from __future__ import annotations
import io
from typing import Tuple
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import torch

from src.hf_utils import hf_download
from src.yolo_onnx import YOLOOnnx
from src.cam_utils import (
    compute_cam_mask,
    overlay_heatmap_on_bgr,
    contours_and_boxes_on_bgr,
)

# ======= App config =======
st.set_page_config(page_title="PedTB X-ray Demo", layout="wide")

# HF public repos & filenames
# ✅ Your actual public repos
HF_MODEL_REPO_YOLO = st.secrets.get("HF_MODEL_REPO_YOLO", "sivaramakrishhnan/cxr-yolo12s-lung")
HF_FILENAME_YOLO   = st.secrets.get("HF_FILENAME_YOLO",   "best.onnx")  # <-- default to 'best.onnx'

HF_MODEL_REPO_DPN  = st.secrets.get("HF_MODEL_REPO_DPN",  "sivaramakrishhnan/cxr-dpn68-tb-cls")
HF_FILENAME_DPN    = st.secrets.get("HF_FILENAME_DPN",    "dpn68_fold2.ckpt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Your PneumoniaModel (already compatible with the ckpt)
from src.pneumonia_model import PneumoniaModel

# Preprocess for classifier (same normalization used in training)
def preprocess_cxr_rgb_to_tensor(rgb: np.ndarray, size: int = 224) -> torch.Tensor:
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # just to use cv2 resize (either space OK)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485,0.456,0.406], dtype=np.float32)
    std  = np.array([0.229,0.224,0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2,0,1)  # CHW
    return torch.from_numpy(img)[None]  # (1,3,H,W)

@st.cache_resource(show_spinner="Downloading YOLO ONNX from Hugging Face…")
def get_yolo_onnx():
    onnx_path = hf_download(HF_MODEL_REPO_YOLO, HF_FILENAME_YOLO, repo_type="model")
    return YOLOOnnx(onnx_path, input_size=640)

@st.cache_resource(show_spinner="Downloading DPN-68 checkpoint from Hugging Face…")
def get_classifier():
    ckpt_path = hf_download(HF_MODEL_REPO_DPN, HF_FILENAME_DPN, repo_type="model")
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

# ======= Sidebar controls =======
st.sidebar.header("Detection")
conf = st.sidebar.slider("YOLO confidence", 0.0, 1.0, 0.25, 0.01)
iou  = st.sidebar.slider("YOLO IoU (NMS)", 0.1, 0.95, 0.75, 0.01)

st.sidebar.header("Grad-CAM")
cam_method = st.sidebar.selectbox(
    "CAM method",
    ["gradcam", "gradcam++", "xgradcam", "layercam", "eigencam", "eigengradcam", "hirescam"],
    index=0,
)
cam_alpha = st.sidebar.slider("Heat alpha", 0.0, 1.0, 0.5, 0.05)
cam_thr   = st.sidebar.slider("Contour threshold", 0.0, 1.0, 0.4, 0.05)
display_size = st.sidebar.select_slider("Display size", options=[224, 384, 512, 768], value=512)

st.title("🩺 Pediatric TB X-ray – Detection • Classification • Grad-CAM")

# ======= File uploader =======
up = st.file_uploader("Upload a chest X-ray (JPG/PNG)", type=["jpg","jpeg","png"])
if not up:
    st.info("Upload an image to begin.")
    st.stop()

# Read as RGB (np.uint8)
orig = Image.open(io.BytesIO(up.read())).convert("RGB")
orig_rgb = np.array(orig)  # HxWx3 RGB
H0, W0 = orig_rgb.shape[:2]

# Show original
st.image(cv2.resize(orig_rgb, (display_size, int(H0 * display_size / W0))), caption="Original", use_column_width=False)

# ======= Load models =======
yolo = get_yolo_onnx()
clf  = get_classifier()

# ======= 1) Detect lungs (ONNX YOLO) =======
with st.spinner("Detecting lungs…"):
    boxes = yolo.predict(cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR), conf=conf, iou=iou)

if len(boxes) == 0:
    st.error("No lungs detected.")
    st.stop()

# Pick the top-scoring box (single class detector)
x1, y1, x2, y2, score = max(boxes, key=lambda b: b[4])
x1i, y1i, x2i, y2i = map(lambda v: int(round(v)), (x1, y1, x2, y2))

# Draw detection on full-res image (for display)
det_vis = orig_rgb.copy()
cv2.rectangle(det_vis, (x1i,y1i), (x2i,y2i), (0,255,0), 3)
st.image(cv2.resize(det_vis, (display_size, int(H0 * display_size / W0))), caption=f"Lung detection (conf {score:.2f})", use_column_width=False)

# Crop at original resolution, then resize ONLY for classifier input size
crop_rgb = orig_rgb[y1i:y2i, x1i:x2i].copy()
if crop_rgb.size == 0:
    st.error("Crop is empty after detection. Check input.")
    st.stop()

# ======= 2) Classify =======
inp = preprocess_cxr_rgb_to_tensor(crop_rgb, size=224).to(DEVICE)

with torch.no_grad():  # prediction only
    logits = clf(inp)
    probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred   = int(probs.argmax())

classes = ["normal", "normal_not"]
pred_text = f"Prediction: **{classes[pred]}** | P(normal_not) = **{probs[1]:.4f}**"
st.markdown(pred_text)

# ======= 3) Grad-CAM (must NOT be in no_grad) =======
cam_mask = compute_cam_mask(
    model=clf,
    input_tensor=inp,           # (1,3,224,224)
    class_index=1,              # abnormal class
    method=cam_method,
    aug_smooth=True,
    eigen_smooth=True,
)

# cam_mask is at 224×224 (classifier input). Resize to crop’s resolution for overlay.
cam_on_crop = overlay_heatmap_on_bgr(
    base_bgr=cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR),
    cam_mask=cam_mask,
    alpha=cam_alpha,
    colormap=cv2.COLORMAP_HOT,
)
cont_on_crop, box_on_crop = contours_and_boxes_on_bgr(
    base_bgr=cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR),
    cam_mask=cam_mask,
    threshold=cam_thr,
    color=(0,0,255),
    thickness=3,
)

# For display, downscale AFTER overlay (so mapping is correct)
def show_resized_bgr(img_bgr, caption):
    h, w = img_bgr.shape[:2]
    disp = cv2.resize(img_bgr, (display_size, int(h * display_size / w)))
    st.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), caption=caption, use_column_width=False)

st.subheader("Explainability on the CROPPED lungs")
show_resized_bgr(cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR), "Crop (BGR view)")
show_resized_bgr(cam_on_crop, "Grad-CAM heatmap")
show_resized_bgr(cont_on_crop, "Contours")
show_resized_bgr(box_on_crop, "Bounding boxes")

st.success("Done.")
