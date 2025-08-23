# streamlit_app.py
from __future__ import annotations

import io
import numpy as np
import streamlit as st
from PIL import Image
import torch
import cv2

from ultralytics import YOLO

# Local helpers
from src.hf_utils import hf_download
from src.pneumonia_model import PneumoniaModel
from src.cam_utils import compute_cam_mask, overlay_heatmap_on_bgr

# =====================
# Page & global config
# =====================
st.set_page_config(page_title="PedTB X-ray Demo", layout="wide")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Internal YOLO defaults (sidebar detection controls removed per request)
DEFAULT_CONF = 0.25
DEFAULT_IOU  = 0.75
DEFAULT_IMGSZ = 640

# -------------------------
# Hugging Face config (public)
# -------------------------
HF_MODEL_REPO_YOLO = st.secrets.get("HF_MODEL_REPO_YOLO", "sivaramakrishhnan/cxr-yolo12s-lung")
HF_FILENAME_YOLO   = st.secrets.get("HF_FILENAME_YOLO",   "best.pt")

HF_MODEL_REPO_DPN  = st.secrets.get("HF_MODEL_REPO_DPN",  "sivaramakrishhnan/cxr-dpn68-tb-cls")
HF_FILENAME_DPN    = st.secrets.get("HF_FILENAME_DPN",    "dpn68_fold2.ckpt")

# -------------------------
# Classifier preprocessing (match Biowulf test/val)
# -------------------------
def preprocess_cxr_rgb_to_tensor(rgb: np.ndarray, size: int = 224) -> torch.Tensor:
    """
    EXACT Biowulf test/val transform (no CLAHE):
      - Resize to 224x224
      - Normalize with ImageNet mean/std
      - CHW tensor, batched
    """
    img = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)  # CHW
    return torch.from_numpy(img)[None]  # (1,3,224,224)

# -------------------------
# Small HF fallback helper
# -------------------------
def _hf_download_with_fallbacks(repo_id: str, candidates, repo_type: str = "model"):
    """
    Try a list of filenames on the Hugging Face Hub and return the first that exists.
    """
    last_err = None
    for fn in candidates:
        try:
            return hf_download(
                repo_id=repo_id,
                filename=fn,
                repo_type=repo_type,
                token=None,
                force_download=False,
            )
        except Exception as e:
            last_err = e
    raise last_err

# -------------------------
# Cached model loaders
# -------------------------
@st.cache_resource(show_spinner="Downloading YOLO (.pt) from Hugging Face…")
def get_yolo() -> YOLO:
    yolo_path = _hf_download_with_fallbacks(
        repo_id=HF_MODEL_REPO_YOLO,
        candidates=[HF_FILENAME_YOLO, "best.pt", "weights.pt", "model.pt"],
        repo_type="model",
    )
    return YOLO(yolo_path)

@st.cache_resource(show_spinner="Downloading DPN-68 checkpoint from Hugging Face…")
def get_classifier() -> PneumoniaModel:
    ckpt_path = _hf_download_with_fallbacks(
        repo_id=HF_MODEL_REPO_DPN,
        candidates=[HF_FILENAME_DPN, "dpn68_fold2.ckpt", "best.ckpt"],
        repo_type="model",
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
# Sidebar (Explainability only)
# -------------------------
st.sidebar.header("Explainability")
# Full Jacobgil CAM menu
CAM_METHODS = [
    "gradcam",
    "gradcam++",
    "scorecam",
    "ablationcam",
    "xgradcam",
    "layercam",
    "fullgrad",
    "eigencam",
    "eigengradcam",
    "hirescam",
]
cam_method = st.sidebar.selectbox("CAM method", CAM_METHODS, index=0)
cam_alpha  = st.sidebar.slider("Heatmap alpha", 0.0, 1.0, 0.5, 0.05)

# (Commented out as requested — keep defaults internally)
# st.sidebar.header("Detection")
# conf = st.sidebar.slider("YOLO confidence", 0.0, 1.0, 0.25, 0.01)
# iou  = st.sidebar.slider("YOLO IoU (NMS)", 0.10, 0.95, 0.75, 0.01)
# imgsz = st.sidebar.select_slider("YOLO imgsz", options=[512, 640, 768], value=640)
# st.sidebar.header("Display")
# pipe_width = st.sidebar.select_slider("Column width (px)", options=[300, 400, 500, 600], value=400)

# -------------------------
# Title + uploader
# -------------------------
st.title("🩺 Pediatric TB X-ray — Detection → Cropped Classification → (Conditional) Grad-CAM")

up = st.file_uploader("Upload a chest X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])
if not up:
    st.info("Upload an image to begin.")
    st.stop()

# Read original image (RGB)
orig = Image.open(io.BytesIO(up.read())).convert("RGB")
orig_rgb = np.array(orig)  # HxWx3 RGB
H0, W0 = orig_rgb.shape[:2]

# Prepare progressive pipeline layout
cols = st.columns(5)
c1, c2, c3, c4, c5 = cols

ph_orig   = c1.empty()
ph_yolo   = c2.empty()
ph_crop   = c3.empty()
ph_verdict= c4.empty()
ph_cam    = c5.empty()

# Show ORIGINAL immediately
ph_orig.image(orig_rgb, caption="Original", use_container_width=True)

# -------------------------
# Load models
# -------------------------
yolo = get_yolo()
clf  = get_classifier()

# -------------------------
# 1) YOLO detection on ORIGINAL-resolution image
#    Ultralytics returns xyxy in ORIGINAL coords.
# -------------------------
orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
with st.spinner("Detecting lungs…"):
    res = yolo.predict(
        source=orig_bgr,
        conf=DEFAULT_CONF,
        iou=DEFAULT_IOU,
        imgsz=DEFAULT_IMGSZ,
        verbose=False,
        device=(0 if DEVICE == "cuda" else "cpu"),
    )[0]

if res.boxes is None or len(res.boxes) == 0:
    st.error("No lungs detected.")
    st.stop()

xyxy   = res.boxes.xyxy.cpu().numpy()   # (N,4)
scores = res.boxes.conf.cpu().numpy()   # (N,)
k      = int(np.argmax(scores))         # highest confidence
x1, y1, x2, y2 = xyxy[k]
score = float(scores[k])

# Clamp to image bounds and ensure valid crop
x1i, y1i = int(max(0, round(x1))), int(max(0, round(y1)))
x2i, y2i = int(min(W0 - 1, round(x2))), int(min(H0 - 1, round(y2)))
if x2i <= x1i: x2i = min(W0 - 1, x1i + 1)
if y2i <= y1i: y2i = min(H0 - 1, y1i + 1)

# Draw YOLO box (ORIGINAL coords) and show immediately
orig_with_box = orig_rgb.copy()
cv2.rectangle(orig_with_box, (x1i, y1i), (x2i, y2i), (0, 255, 0), 3)
ph_yolo.image(orig_with_box, caption=f"YOLO box (conf {score:.2f})", use_container_width=True)

# Crop from ORIGINAL and show immediately
crop_rgb = orig_rgb[y1i:y2i, x1i:x2i].copy()
if crop_rgb.size == 0:
    st.error("Crop is empty. Detector box was degenerate.")
    st.stop()
ph_crop.image(crop_rgb, caption="Cropped lungs (original resolution)", use_container_width=True)

# -------------------------
# 2) Classify the crop (exact test preprocessing)
# -------------------------
with st.spinner("Classifying cropped lungs…"):
    inp = preprocess_cxr_rgb_to_tensor(crop_rgb, size=224).to(DEVICE)
    with torch.no_grad():
        logits = clf(inp)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred   = int(probs.argmax())

classes = ["normal", "normal_not"]
pred_prob = float(probs[pred])  # only the predicted class prob

if pred == 1:
    verdict_text = f"This chest X-ray **manifests TB-related manifestations** (probability **{pred_prob:.4f}**)."
else:
    verdict_text = f"This chest X-ray **shows normal lungs** (probability **{pred_prob:.4f}**)."

ph_verdict.markdown("#### Model verdict")
ph_verdict.markdown(verdict_text)

# -------------------------
# 3) Grad-CAM (ONLY if abnormal). Compute at 224×224, resize to crop size, overlay HOT.
#    Ensure CAM runs outside no_grad().
# -------------------------
if pred == 1:
    with st.spinner("Computing Grad-CAM…"):
        # Compute CAM on the 224×224 classifier input
        cam_mask_224 = compute_cam_mask(
            model=clf,
            input_tensor=inp,            # (1,3,224,224)
            class_index=1,               # abnormal class
            method=cam_method,
            aug_smooth=True,
            eigen_smooth=True,
        )  # float32 [0,1], (224,224)

        # Resize CAM to the crop’s ORIGINAL size for accurate overlay
        crop_h, crop_w = crop_rgb.shape[:2]
        cam_mask_crop = cv2.resize(cam_mask_224, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

        # Overlay HOT colormap on the crop
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        cam_on_crop_bgr = overlay_heatmap_on_bgr(
            base_bgr=crop_bgr,
            cam_mask=cam_mask_crop,   # [0,1]
            alpha=cam_alpha,
            colormap=cv2.COLORMAP_HOT,
        )
        cam_on_crop_rgb = cv2.cvtColor(cam_on_crop_bgr, cv2.COLOR_BGR2RGB)

        ph_cam.image(cam_on_crop_rgb, caption="Grad-CAM (HOT) on crop", use_container_width=True)
else:
    # No explanation for normal predictions (per requirement)
    ph_cam.markdown(" ")

