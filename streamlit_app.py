# streamlit_app.py
# --------------------------------------------------------------------------------------
# Pediatric TB X-ray Demo (Streamlit)
# - Loads YOLOv5/YOLOv8-style ONNX lung detector from Hugging Face
# - Crops lungs on the ORIGINAL resolution
# - Classifies the crop with a fine-tuned DPN-68 Lightning checkpoint
# - Computes Grad-CAM (pytorch-grad-cam) on the crop and overlays heatmaps/contours/boxes
# - Displays everything at a user-selected DISPLAY size (render-only; never used for CAM math)
# --------------------------------------------------------------------------------------
from __future__ import annotations

import os
import io
import numpy as np
import streamlit as st
from PIL import Image
import torch
import cv2  # headless build in requirements

from src.hf_utils import hf_download, hf_download_robust
from src.pneumonia_model import PneumoniaModel
from src.yolo_onnx import YOLOOnnx
from src.cam_utils import (
    compute_cam_mask,
    overlay_heatmap_on_bgr,
    contours_and_boxes_on_bgr,
)

# ======= App config =======
st.set_page_config(page_title="PedTB X-ray Demo", layout="wide")

# ---- Hugging Face public repos & filenames (edit via Streamlit secrets if needed) ----
HF_MODEL_REPO_YOLO = st.secrets.get("HF_MODEL_REPO_YOLO", "sivaramakrishhnan/cxr-yolo12s-lung")
HF_FILENAME_YOLO   = st.secrets.get("HF_FILENAME_YOLO",   "best.onnx")  # the file that exists in your repo

HF_MODEL_REPO_DPN  = st.secrets.get("HF_MODEL_REPO_DPN",  "sivaramakrishhnan/cxr-dpn68-tb-cls")
HF_FILENAME_DPN    = st.secrets.get("HF_FILENAME_DPN",    "dpn68_fold2.ckpt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======= Classifier preprocessing (match training stats) =======
def preprocess_cxr_rgb_to_tensor(rgb: np.ndarray, size: int = 224) -> torch.Tensor:
    """
    Convert RGB uint8 image (HxWx3) -> normalized FloatTensor (1,3,size,size)
    Uses ImageNet mean/std; resizing occurs AFTER cropping to the detection box.
    """
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # use cv2 resize; color space is irrelevant for resize
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)  # CHW
    return torch.from_numpy(img)[None]  # (1,3,H,W)


# ======= Cached model loaders (download once per session) =======
@st.cache_resource(show_spinner="Downloading YOLO (ONNX) from Hugging Face…")
def get_yolo_onnx() -> YOLOOnnx:
    """
    Robustly download YOLO ONNX from the Hub.
    Tries the configured filename, then a few safe fallbacks, and raises a helpful error if missing.
    Returns a ready YOLOOnnx wrapper whose .predict() yields boxes in ORIGINAL image coordinates.
    """
    candidates = [
        HF_FILENAME_YOLO,
        "best.onnx",
        "yolo12s_lung.onnx",
        "model.onnx",
    ]
    onnx_path = hf_download_robust(
        repo_id=HF_MODEL_REPO_YOLO,
        filename_or_list=candidates,
        repo_type="model",
        token=None,  # public repo
    )
    st.info(f"YOLO ONNX loaded from: {onnx_path}")
    return YOLOOnnx(onnx_path, input_size=640)


@st.cache_resource(show_spinner="Downloading DPN-68 checkpoint from Hugging Face…")
def get_classifier() -> PneumoniaModel:
    """
    Download and load the Lightning checkpoint. strict=False to accommodate tiny head diffs across versions.
    """
    candidates = [
        HF_FILENAME_DPN,
        "dpn68_fold2.ckpt",
        "best.ckpt",
        "model.ckpt",
    ]
    ckpt_path = hf_download_robust(
        repo_id=HF_MODEL_REPO_DPN,
        filename_or_list=candidates,
        repo_type="model",
        token=None,  # public repo
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

st.sidebar.header("Display")
display_size = st.sidebar.select_slider("Display width (px)", options=[224, 384, 512, 768], value=512)

st.title("🩺 Pediatric TB X-ray – Detection • Classification • Grad-CAM")

# ======= File uploader =======
up = st.file_uploader("Upload a chest X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])
if not up:
    st.info("Upload an image to begin.")
    st.stop()

# Read as RGB uint8
orig = Image.open(io.BytesIO(up.read())).convert("RGB")
orig_rgb = np.array(orig)  # HxWx3 RGB
H0, W0 = orig_rgb.shape[:2]

# Display original at chosen display width (preserve aspect ratio)
disp_h = int(H0 * (display_size / float(W0)))
st.image(cv2.resize(orig_rgb, (display_size, disp_h)), caption="Original", use_column_width=False)

# ======= Load models =======
try:
    yolo = get_yolo_onnx()
except Exception as e:
    st.error(f"Failed to load YOLO ONNX model: {e}")
    st.stop()

try:
    clf = get_classifier()
except Exception as e:
    st.error(f"Failed to load DPN-68 checkpoint: {e}")
    st.stop()

# ======= 1) Detect lungs (on ORIGINAL image) =======
with st.spinner("Detecting lungs…"):
    # YOLOOnnx expects BGR input
    boxes = yolo.predict(cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR), conf=conf, iou=iou)

if len(boxes) == 0:
    st.error("No lungs detected.")
    st.stop()

# Choose the best detection (single-class model). Highest score by default.
x1, y1, x2, y2, score = max(boxes, key=lambda b: b[4])
x1i, y1i, x2i, y2i = map(lambda v: int(round(v)), (x1, y1, x2, y2))

# Visualize detection on the FULL-RES original (for presentation)
det_vis = orig_rgb.copy()
cv2.rectangle(det_vis, (x1i, y1i), (x2i, y2i), (0, 255, 0), 3)
st.image(
    cv2.resize(det_vis, (display_size, disp_h)),
    caption=f"Lung detection (conf {score:.2f})",
    use_column_width=False,
)

# Crop at ORIGINAL resolution (correct!)
crop_rgb = orig_rgb[y1i:y2i, x1i:x2i].copy()
if crop_rgb.size == 0:
    st.error("Crop is empty after detection. Please try another image.")
    st.stop()

# ======= 2) Classify crop (224×224 input for the classifier) =======
inp = preprocess_cxr_rgb_to_tensor(crop_rgb, size=224).to(DEVICE)

with torch.no_grad():
    logits = clf(inp)
    probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred   = int(probs.argmax())

classes = ["normal", "normal_not"]
st.markdown(f"**Prediction:** {classes[pred]}  |  **P(normal_not)** = {probs[1]:.4f}")

# ======= 3) Grad-CAM on the crop =======
# IMPORTANT: Do NOT wrap CAM in no_grad(). Our cam_utils.compute_cam_mask enables grad internally.
cam_mask = compute_cam_mask(
    model=clf,
    input_tensor=inp,           # (1,3,224,224)
    class_index=1,              # abnormal class index
    method=cam_method,
    aug_smooth=True,
    eigen_smooth=True,
)

# cam_mask is defined at the CLASSIFIER INPUT resolution (224×224).
# Overlay onto the ORIGINAL crop, not the original full image.
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

# ======= 4) Display overlays at presentation resolution (resize AFTER overlay) =======
def show_resized_bgr(img_bgr: np.ndarray, caption: str):
    h, w = img_bgr.shape[:2]
    disp_h_lung = int(h * (display_size / float(w)))
    st.image(cv2.cvtColor(cv2.resize(img_bgr, (display_size, disp_h_lung)), cv2.COLOR_BGR2RGB),
             caption=caption, use_column_width=False)

st.subheader("Explainability on the CROPPED lungs")
show_resized_bgr(crop_bgr, "Cropped lungs")
show_resized_bgr(cam_on_crop, "Grad-CAM heatmap")
show_resized_bgr(cont_on_crop, "Contours (CAM≥threshold)")
show_resized_bgr(box_on_crop, "Bounding boxes (CAM≥threshold)")

st.success("Done.")
