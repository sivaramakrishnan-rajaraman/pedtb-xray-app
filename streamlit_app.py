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
from src.tb_model import TBModel   # 
from src.cam_utils import compute_cam_mask, overlay_heatmap_on_bgr, METHOD_TO_CMAP_KIND

# =====================
# Page & global config
# =====================
st.set_page_config(page_title="PedTB X-ray Demo", layout="wide")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Inject a tiny CSS for bigger bold captions
st.markdown("""
<style>
.cap-big-bold {font-size:1.05rem; font-weight:800; margin-bottom:0.25rem;}
.verdict {font-size:1.1rem; font-weight:800;}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Hugging Face config (public by default)
# To secure later, set repos to PRIVATE and add HF_TOKEN in Streamlit secrets.
# -------------------------
HF_TOKEN = st.secrets.get("HF_TOKEN", None)  # None if public repos
HF_MODEL_REPO_YOLO = st.secrets.get("HF_MODEL_REPO_YOLO", "sivaramakrishhnan/cxr-yolo12s-lung")
HF_FILENAME_YOLO   = st.secrets.get("HF_FILENAME_YOLO",   "best.pt")

HF_MODEL_REPO_CLS  = st.secrets.get("HF_MODEL_REPO_CLS",  "sivaramakrishhnan/cxr-dpn68-tb-cls")
# HF_FILENAME_CLS    = st.secrets.get("HF_FILENAME_CLS",    "dpn68_fold2.ckpt")
HF_FILENAME_CLS    = st.secrets.get("HF_FILENAME_CLS",    "vgg13_fold2.ckpt")

# -------------------------
# Classifier preprocessing (exact Biowulf val/test: resize→normalize→CHW)
# -------------------------
def preprocess_cxr_rgb_to_tensor(rgb: np.ndarray, size: int = 224) -> torch.Tensor:
    img = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)  # HWC->CHW
    return torch.from_numpy(img)[None]  # (1,3,224,224)

# -------------------------
# Cached model loaders
# -------------------------
@st.cache_resource(show_spinner="Downloading YOLO checkpoint from Hugging Face…")
def get_yolo() -> YOLO:
    yolo_path = hf_download(
        repo_id=HF_MODEL_REPO_YOLO,
        filename=HF_FILENAME_YOLO,
        repo_type="model",
        token=HF_TOKEN,
        force_download=False,
    )
    return YOLO(yolo_path)

@st.cache_resource(show_spinner="Downloading classifier checkpoint from Hugging Face…")
def get_classifier() -> TBModel:
    ckpt_path = hf_download(
        repo_id=HF_MODEL_REPO_CLS,
        filename=HF_FILENAME_CLS,
        repo_type="model",
        token=HF_TOKEN,
        force_download=False,
    )
    
    h = {
        "model": "vgg13",  # change to "dpn68_new" when switching architectures (see §3)
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
    model = TBModel.load_from_checkpoint(ckpt_path, h=h, strict=False, map_location=DEVICE)
    model.to(DEVICE).eval()
    return model

# -------------------------
# Sidebar: ONLY explainability options (as requested)
# -------------------------
st.sidebar.header("Explainability")
cam_method = st.sidebar.selectbox(
    "CAM method",
    ["gradcam","gradcam++","scorecam","ablationcam","xgradcam","layercam","fullgrad","eigencam","eigengradcam","hirescam"],
    index=0,
)
cam_alpha = st.sidebar.slider("Heatmap alpha", 0.0, 1.0, 0.5, 0.05) #default is 0.5
# cam_threshold = st.sidebar.slider("Activation threshold", 0.0, 1.0, 0.3, 0.05) #default is 0.3
# cam_method = st.sidebar.selectbox(
#     "CAM method",
#     [
#         "gradcam", "gradcam++", "scorecam", "ablationcam", "xgradcam",
#         "layercam", "fullgrad", "eigencam", "eigengradcam", "hirescam",
#     ],
#     index=0,
# )
# cam_alpha = st.sidebar.slider("Heatmap alpha", 0.0, 1.0, 0.5, 0.05)

# Internals: detection defaults (do not expose in UI)
YOLO_CONF = 0.25
YOLO_IOU  = 0.75
YOLO_IMGSZ = 640

# -------------------------
# Title + uploader
# -------------------------
st.title("🩺 Pediatric TB frontal chest X-ray Classification — Lung Detection → Classification → Explanation")

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
# FAST PATH: show original immediately
# -------------------------
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown("<div class='cap-big-bold'>Original</div>", unsafe_allow_html=True)
    st.image(orig_rgb, use_container_width=True)

# -------------------------
# 1) YOLO detection on ORIGINAL-resolution image
# -------------------------
with st.spinner("Detecting lungs…"):
    res = yolo.predict(
        source=cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR),
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        imgsz=YOLO_IMGSZ,
        verbose=False,
        device=(0 if DEVICE == "cuda" else "cpu"),
    )[0]

if res.boxes is None or len(res.boxes) == 0:
    with col2:
        st.error("No lungs detected.")
    st.stop()

xyxy   = res.boxes.xyxy.cpu().numpy()
scores = res.boxes.conf.cpu().numpy()
k      = int(np.argmax(scores))
x1, y1, x2, y2 = xyxy[k]
score = float(scores[k])

# clamp & crop
x1i, y1i = int(max(0, round(x1))), int(max(0, round(y1)))
x2i, y2i = int(min(W0 - 1, round(x2))), int(min(H0 - 1, round(y2)))
if x2i <= x1i: x2i = min(W0 - 1, x1i + 1)
if y2i <= y1i: y2i = min(H0 - 1, y1i + 1)

orig_with_box = orig_rgb.copy()
cv2.rectangle(orig_with_box, (x1i, y1i), (x2i, y2i), (0, 255, 0), 3)

crop_rgb = orig_rgb[y1i:y2i, x1i:x2i].copy()

# Show YOLO & crop immediately
with col2:
    st.markdown(f"<div class='cap-big-bold'>Lung Detected (conf {score:.2f})</div>", unsafe_allow_html=True)
    st.image(orig_with_box, use_container_width=True)

with col3:
    st.markdown("<div class='cap-big-bold'>Cropped lung</div>", unsafe_allow_html=True)
    st.image(crop_rgb, use_container_width=True)

if crop_rgb.size == 0:
    with col3:
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
pred_prob = float(probs[pred])

with col4:
    st.markdown("<div class='cap-big-bold'>Model verdict</div>", unsafe_allow_html=True)
    if pred == 1:
        st.markdown(f"<div class='verdict'>This chest X-ray <b>shows manifestations consistent with Tuberculosis</b> (probability <b>{pred_prob:.4f}</b>).</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='verdict'>This chest X-ray <b>shows normal lungs</b> (probability <b>{pred_prob:.4f}</b>).</div>", unsafe_allow_html=True)

# -------------------------
# 3) Grad-CAM only if abnormal
#    Compute at 224×224 → resize CAM to crop size → overlay with per-method colormap
# -------------------------
if pred == 1:
    cam_mask_224 = compute_cam_mask(
        model=clf,
        input_tensor=inp,           # (1,3,224,224)
        class_index=1,              # abnormal class
        method=cam_method,
        aug_smooth=True,
        eigen_smooth=True,
    )  # float32 [0,1], (224,224)

    crop_h, crop_w = crop_rgb.shape[:2]
    cam_mask_crop = cv2.resize(cam_mask_224, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

    # Per-method colormap (mpl for first 5, cv2 for the rest)
    cmap_kind = METHOD_TO_CMAP_KIND[cam_method]  # ('mpl','Reds') or ('cv2', cv2.COLORMAP_*)
    cam_on_crop_bgr = overlay_heatmap_on_bgr(
    base_bgr=crop_bgr,
    cam_mask=cam_mask_crop,      # [0,1]
    alpha=cam_alpha,
    cmap_kind=cmap_kind,
    # threshold=cam_threshold,     
    )
    # cam_on_crop_bgr = overlay_heatmap_on_bgr(
    #     base_bgr=crop_bgr,
    #     cam_mask=cam_mask_crop,
    #     alpha=cam_alpha,
    #     cmap_kind=cmap_kind,
    # )
    cam_on_crop_rgb = cv2.cvtColor(cam_on_crop_bgr, cv2.COLOR_BGR2RGB)

    with col5:
        label = f"Explanation ({cmap_kind[0]}:{cmap_kind[1]})"
        st.markdown(f"<div class='cap-big-bold'>{label}</div>", unsafe_allow_html=True)
        st.image(cam_on_crop_rgb, use_container_width=True)
else:
    with col5:
        st.info("No explanation shown for normal prediction.")
