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
from src.hf_utils import hf_download_robust
from src.pneumonia_model import PneumoniaModel
from src.cam_utils import compute_cam_mask, overlay_heatmap_on_bgr

# =====================
# Page & global config
# =====================
st.set_page_config(page_title="PedTB X-ray Demo", layout="wide")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Hugging Face config (public)
# -------------------------
HF_MODEL_REPO_YOLO = st.secrets.get("HF_MODEL_REPO_YOLO", "sivaramakrishhnan/cxr-yolo12s-lung")
HF_FILENAME_YOLO   = st.secrets.get("HF_FILENAME_YOLO",   "best.pt")  # your actual filename

HF_MODEL_REPO_DPN  = st.secrets.get("HF_MODEL_REPO_DPN",  "sivaramakrishhnan/cxr-dpn68-tb-cls")
HF_FILENAME_DPN    = st.secrets.get("HF_FILENAME_DPN",    "dpn68_fold2.ckpt")  # your actual filename

# -------------------------
# Classifier preprocessing (match Biowulf test/val)
# -------------------------
def preprocess_cxr_rgb_to_tensor(rgb: np.ndarray, size: int = 224) -> torch.Tensor:
    """
    EXACTLY your test/val transform (no CLAHE):
      - Resize to 224x224
      - Normalize with ImageNet mean/std
      - CHW tensor
    """
    # Resize
    img = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    # Normalize (ImageNet)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    # HWC->CHW
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img)[None]  # (1,3,224,224)

# -------------------------
# Cached model loaders
# -------------------------
@st.cache_resource(show_spinner="Downloading YOLO (.pt) from Hugging Face…")
def get_yolo() -> YOLO:
    yolo_path = hf_download_robust(
        repo_id=HF_MODEL_REPO_YOLO,
        filename_or_list=[HF_FILENAME_YOLO, "best.pt"],
        repo_type="model",
        token=None,
    )
    model = YOLO(yolo_path)
    return model

@st.cache_resource(show_spinner="Downloading DPN-68 checkpoint from Hugging Face…")
def get_classifier() -> PneumoniaModel:
    ckpt_path = hf_download_robust(
        repo_id=HF_MODEL_REPO_DPN,
        filename_or_list=[HF_FILENAME_DPN, "dpn68_fold2.ckpt", "best.ckpt"],
        repo_type="model",
        token=None,
    )
    # Minimal h needed by your LightningModule
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

st.sidebar.header("Explainability")
cam_method = st.sidebar.selectbox("CAM method", ["gradcam", "gradcam++", "xgradcam", "layercam"], index=0)
cam_alpha = st.sidebar.slider("Heatmap alpha", 0.0, 1.0, 0.5, 0.05)

st.sidebar.header("Display")
pipe_width = st.sidebar.select_slider("Column width (px)", options=[300, 400, 500, 600], value=400)

# -------------------------
# Title + uploader
# -------------------------
st.title("🩺 Pediatric TB X-ray — Detection → Classification → Grad-CAM")

up = st.file_uploader("Upload a chest X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])
if not up:
    st.info("Upload an image to begin.")
    st.stop()

# Read original image (RGB)
orig = Image.open(io.BytesIO(up.read())).convert("RGB")
orig_rgb = np.array(orig)  # HxWx3 RGB
H0, W0 = orig_rgb.shape[:2]

# -------------------------
# Load models
# -------------------------
yolo = get_yolo()
clf  = get_classifier()

# -------------------------
# 1) YOLO detection on ORIGINAL-resolution image
#    Ultralytics returns xyxy in ORIGINAL coords.
# -------------------------
with st.spinner("Detecting lungs (Ultralytics)…"):
    # Ultralytics expects BGR np.ndarray; boxes returned in original coords
    res = yolo.predict(
        source=cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
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

# Visualizations precomputed
orig_with_box = orig_rgb.copy()
cv2.rectangle(orig_with_box, (x1i, y1i), (x2i, y2i), (0, 255, 0), 3)

crop_rgb = orig_rgb[y1i:y2i, x1i:x2i].copy()
if crop_rgb.size == 0:
    st.error("Crop is empty. Detector box was degenerate.")
    st.stop()

# -------------------------
# 2) Classify the crop (exact test preprocessing)
# -------------------------
inp = preprocess_cxr_rgb_to_tensor(crop_rgb, size=224).to(DEVICE)
with torch.no_grad():
    logits = clf(inp)
    probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred   = int(probs.argmax())

classes = ["normal", "normal_not"]
pred_prob = probs[pred]  # only the predicted class prob

if pred == 1:
    verdict = f"This chest X-ray **manifests TB-related manifestations** (probability **{pred_prob:.4f}**)."
else:
    verdict = f"This chest X-ray **shows normal lungs** (probability **{pred_prob:.4f}**)."

# -------------------------
# 3) Grad-CAM (target deepest conv = post3x3)
#    - Compute at 224x224 (classifier input)
#    - Resize CAM to the crop’s ORIGINAL resolution
#    - Overlay HOT colormap on the crop
# -------------------------
cam_mask_224 = compute_cam_mask(
    model=clf,
    input_tensor=inp,           # (1,3,224,224)
    class_index=1,              # abnormal class = 1
    method=cam_method,
    aug_smooth=True,
    eigen_smooth=True,
)  # float32 in [0,1], shape (224,224)

# Resize CAM to crop size
crop_h, crop_w = crop_rgb.shape[:2]
cam_mask_crop = cv2.resize(cam_mask_224, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

# Overlay on crop (in BGR for OpenCV colormap)
crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
cam_on_crop_bgr = overlay_heatmap_on_bgr(
    base_bgr=crop_bgr,
    cam_mask=cam_mask_crop,      # [0,1]
    alpha=cam_alpha,
    colormap=cv2.COLORMAP_HOT,
)
cam_on_crop_rgb = cv2.cvtColor(cam_on_crop_bgr, cv2.COLOR_BGR2RGB)

# -------------------------
# 4) Horizontal pipeline display
# -------------------------
st.markdown("### Inference pipeline")
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.image(orig_rgb, caption="Original", use_container_width=True)

with c2:
    st.image(orig_with_box, caption=f"YOLO box (conf {score:.2f})", use_container_width=True)

with c3:
    st.image(crop_rgb, caption="Cropped lungs (original resolution)", use_container_width=True)

with c4:
    st.markdown("#### Model verdict")
    st.markdown(verdict)

with c5:
    st.image(cam_on_crop_rgb, caption="Grad-CAM (HOT) on crop", use_container_width=True)

st.success("Done.")

