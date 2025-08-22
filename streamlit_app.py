# streamlit_app.py  (drop-in replacement)
from __future__ import annotations

import os
import io
import numpy as np
import streamlit as st
from PIL import Image
import torch
import cv2

from src.hf_utils import hf_download_robust
from src.pneumonia_model import PneumoniaModel
from src.yolo_onnx import YOLOOnnx
from src.cam_utils import (
    compute_cam_mask,
    overlay_heatmap_on_bgr,
    contours_and_boxes_on_bgr,
)

st.set_page_config(page_title="PedTB X-ray Demo", layout="wide")

# ---- Hugging Face public repos & filenames ----
HF_MODEL_REPO_YOLO = st.secrets.get("HF_MODEL_REPO_YOLO", "sivaramakrishhnan/cxr-yolo12s-lung")
HF_FILENAME_YOLO   = st.secrets.get("HF_FILENAME_YOLO",   "best.onnx")

HF_MODEL_REPO_DPN  = st.secrets.get("HF_MODEL_REPO_DPN",  "sivaramakrishhnan/cxr-dpn68-tb-cls")
HF_FILENAME_DPN    = st.secrets.get("HF_FILENAME_DPN",    "dpn68_fold2.ckpt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# Classifier preprocessing
# --------------------------
def preprocess_cxr_rgb_to_tensor(rgb: np.ndarray, size: int = 224) -> torch.Tensor:
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img)[None]

# --------------------------
# Cached model loaders
# --------------------------
@st.cache_resource(show_spinner="Downloading YOLO (ONNX) from Hugging Face…")
def get_yolo_onnx() -> YOLOOnnx:
    onnx_path = hf_download_robust(
        repo_id=HF_MODEL_REPO_YOLO,
        filename_or_list=[HF_FILENAME_YOLO, "best.onnx", "model.onnx", "yolo12s_lung.onnx"],
        repo_type="model",
        token=None,
    )
    st.info(f"YOLO ONNX loaded from: {onnx_path}")
    return YOLOOnnx(onnx_path, input_size=640)

@st.cache_resource(show_spinner="Downloading DPN-68 checkpoint from Hugging Face…")
def get_classifier() -> PneumoniaModel:
    ckpt_path = hf_download_robust(
        repo_id=HF_MODEL_REPO_DPN,
        filename_or_list=[HF_FILENAME_DPN, "dpn68_fold2.ckpt", "best.ckpt", "model.ckpt"],
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
    st.info(f"DPN-68 loaded from: {ckpt_path}")
    return model

# --------------------------
# YOLO box refinement utils
# --------------------------
def _clamp_box(x1, y1, x2, y2, W, H):
    x1 = max(0, min(int(round(x1)), W - 1))
    y1 = max(0, min(int(round(y1)), H - 1))
    x2 = max(0, min(int(round(x2)), W - 1))
    y2 = max(0, min(int(round(y2)), H - 1))
    if x2 <= x1: x2 = min(W - 1, x1 + 1)
    if y2 <= y1: y2 = min(H - 1, y1 + 1)
    return x1, y1, x2, y2

def _expand_box(x1, y1, x2, y2, W, H, expand_pct: float):
    """Expand box by expand_pct of its size on each side."""
    w = x2 - x1
    h = y2 - y1
    dx = int(round(w * expand_pct))
    dy = int(round(h * expand_pct))
    return _clamp_box(x1 - dx, y1 - dy, x2 + dx, y2 + dy, W, H)

def _central_fallback(W, H, margin_pct: float = 0.05):
    """Center crop fallback if detector fails: remove thin margins."""
    mx = int(round(W * margin_pct))
    my = int(round(H * margin_pct))
    return mx, my, W - mx, H - my

def select_and_refine_box(
    boxes,
    W: int,
    H: int,
    min_area_ratio: float = 0.25,
    min_w_frac: float = 0.50,
    min_h_frac: float = 0.50,
    prefer_center_lambda: float = 0.30,
    expand_pct: float = 0.08,
):
    """
    From raw YOLO boxes [(x1,y1,x2,y2,score)], choose a plausible lung box:
    - filter by area and size thresholds
    - score candidates by (score - λ * normalized_center_distance)
    - expand the chosen box by expand_pct and clamp to image
    - fallback: central crop if nothing passes
    """
    if len(boxes) == 0:
        return _central_fallback(W, H)

    cx_img, cy_img = W / 2.0, H / 2.0
    diag = np.hypot(W, H)

    # filter
    cands = []
    for (x1, y1, x2, y2, s) in boxes:
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        area = w * h
        if (area < min_area_ratio * W * H) or (w < min_w_frac * W) or (h < min_h_frac * H):
            continue
        # distance to center
        cx, cy = x1 + w / 2.0, y1 + h / 2.0
        dist = np.hypot(cx - cx_img, cy - cy_img) / diag  # 0..~0.7
        score_adj = float(s) - prefer_center_lambda * float(dist)
        cands.append((score_adj, x1, y1, x2, y2, s, dist))

    if not cands:
        # No candidates pass guard — take highest-score raw box but expand and clamp.
        x1, y1, x2, y2, s = max(boxes, key=lambda b: b[4])
        x1, y1, x2, y2 = _expand_box(x1, y1, x2, y2, W, H, expand_pct)
        return x1, y1, x2, y2

    # Best adjusted score
    cands.sort(key=lambda t: t[0], reverse=True)
    _, x1, y1, x2, y2, s, dist = cands[0]
    x1, y1, x2, y2 = _expand_box(x1, y1, x2, y2, W, H, expand_pct)
    return x1, y1, x2, y2

# --------------------------
# Sidebar controls
# --------------------------
st.sidebar.header("Detection")
conf = st.sidebar.slider("YOLO confidence", 0.0, 1.0, 0.25, 0.01)
iou  = st.sidebar.slider("YOLO IoU (NMS)", 0.1, 0.95, 0.75, 0.01)

st.sidebar.subheader("Detection guard (post-processing)")
min_area_ratio = st.sidebar.slider("Min area (fraction of image)", 0.05, 0.80, 0.25, 0.01)
min_w_frac     = st.sidebar.slider("Min width fraction", 0.10, 0.95, 0.50, 0.01)
min_h_frac     = st.sidebar.slider("Min height fraction", 0.10, 0.95, 0.50, 0.01)
center_lambda  = st.sidebar.slider("Center bias λ", 0.00, 1.00, 0.30, 0.01)
expand_pct     = st.sidebar.slider("Expand bbox (%)", 0.00, 0.50, 0.08, 0.01)

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

st.title("🩺 Pediatric TB X-ray – Detection • Classification • Grad-CAM")

# --------------------------
# Uploader
# --------------------------
up = st.file_uploader("Upload a chest X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])
if not up:
    st.info("Upload an image to begin.")
    st.stop()

orig = Image.open(io.BytesIO(up.read())).convert("RGB")
orig_rgb = np.array(orig)
H0, W0 = orig_rgb.shape[:2]
disp_h = int(H0 * (display_size / float(W0)))
st.image(cv2.resize(orig_rgb, (display_size, disp_h)), caption="Original", use_column_width=False)

# --------------------------
# Load models
# --------------------------
yolo = get_yolo_onnx()
clf  = get_classifier()

# --------------------------
# 1) Detect lungs
# --------------------------
with st.spinner("Detecting lungs…"):
    boxes = yolo.predict(cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR), conf=conf, iou=iou)

if len(boxes) == 0:
    st.error("No lungs detected.")
    st.stop()

# Show raw TOP-SCORE YOLO box
raw_x1, raw_y1, raw_x2, raw_y2, raw_s = max(boxes, key=lambda b: b[4])
raw_vis = orig_rgb.copy()
cv2.rectangle(raw_vis, (int(raw_x1), int(raw_y1)), (int(raw_x2), int(raw_y2)), (0, 255, 0), 3)
st.image(
    cv2.resize(raw_vis, (display_size, disp_h)),
    caption=f"Raw YOLO box (conf {raw_s:.2f})",
    use_column_width=False,
)

# Refine & expand box with guards
x1i, y1i, x2i, y2i = select_and_refine_box(
    boxes,
    W=W0,
    H=H0,
    min_area_ratio=min_area_ratio,
    min_w_frac=min_w_frac,
    min_h_frac=min_h_frac,
    prefer_center_lambda=center_lambda,
    expand_pct=expand_pct,
)

ref_vis = orig_rgb.copy()
cv2.rectangle(ref_vis, (x1i, y1i), (x2i, y2i), (255, 165, 0), 3)  # orange
st.image(
    cv2.resize(ref_vis, (display_size, disp_h)),
    caption="Refined/expanded box (used for crop)",
    use_column_width=False,
)

# Crop at ORIGINAL resolution from refined box
crop_rgb = orig_rgb[y1i:y2i, x1i:x2i].copy()
if crop_rgb.size == 0:
    st.error("Crop is empty after refinement. Falling back to center crop.")
    x1i, y1i, x2i, y2i = _central_fallback(W0, H0, margin_pct=0.05)
    crop_rgb = orig_rgb[y1i:y2i, x1i:x2i].copy()

# --------------------------
# 2) Classify crop
# --------------------------
inp = preprocess_cxr_rgb_to_tensor(crop_rgb, size=224).to(DEVICE)
with torch.no_grad():
    logits = clf(inp)
    probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred   = int(probs.argmax())

classes = ["normal", "normal_not"]
st.markdown(f"**Prediction:** {classes[pred]}  |  **P(normal_not)** = {probs[1]:.4f}")

# --------------------------
# 3) Grad-CAM on crop
# --------------------------
cam_mask = compute_cam_mask(
    model=clf,
    input_tensor=inp,
    class_index=1,
    method=cam_method,
    aug_smooth=True,
    eigen_smooth=True,
)

crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
cam_on_crop = overlay_heatmap_on_bgr(
    base_bgr=crop_bgr,
    cam_mask=cam_mask,
    alpha=cam_alpha,
    colormap=cv2.COLORMAP_HOT,
)
cont_on_crop, box_on_crop = contours_and_boxes_on_bgr(
    base_bgr=crop_bgr,
    cam_mask=cam_mask,
    threshold=cam_thr,
    color=(0, 0, 255),
    thickness=3,
)

def show_resized_bgr(img_bgr: np.ndarray, caption: str):
    h, w = img_bgr.shape[:2]
    disp_h_lung = int(h * (display_size / float(w)))
    st.image(cv2.cvtColor(cv2.resize(img_bgr, (display_size, disp_h_lung)), cv2.COLOR_BGR2RGB),
             caption=caption, use_column_width=False)

st.subheader("Explainability on the CROPPED lungs (refined bbox)")
show_resized_bgr(crop_bgr, "Cropped lungs")
show_resized_bgr(cam_on_crop, "Grad-CAM heatmap")
show_resized_bgr(cont_on_crop, "Contours (CAM≥threshold)")
show_resized_bgr(box_on_crop, "Bounding boxes (CAM≥threshold)")

st.success("Done.")
