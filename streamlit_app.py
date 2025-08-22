# streamlit_app.py
from __future__ import annotations
import os
import io
from typing import Tuple, List

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageOps

import torch
import torch.nn.functional as F

from src.hf_utils import hf_download
from src.pneumonia_model import PneumoniaModel
from src.cam_utils import (
    compute_cam_map,
    heatmap_overlay,
    contours_and_bboxes,
    draw_bboxes,
)

from src.yolo_onnx import YOLOOnnx

# -----------------------------
# Configuration via secrets or defaults
# -----------------------------
HF_TOKEN = st.secrets.get("HF_TOKEN", None)   # not needed for public repos

# YOLO ONNX on HF (exported once offline)
HF_MODEL_REPO_YOLO = st.secrets.get("HF_MODEL_REPO_YOLO", "sivaramakrishhnan/cxr-yolo12s-lung")
HF_FILENAME_YOLO   = st.secrets.get("HF_FILENAME_YOLO",   "best.onnx")  # <-- upload this to your HF repo

# DPN-68 Lightning ckpt on HF (your best fold-2)
HF_MODEL_REPO_DPN  = st.secrets.get("HF_MODEL_REPO_DPN",  "sivaramakrishhnan/cxr-dpn68-tb-cls")
HF_FILENAME_DPN    = st.secrets.get("HF_FILENAME_DPN",    "dpn68_fold2.ckpt")

DEVICE = "cpu"  # Streamlit Cloud CPU

# Default hyperparams for classifier (must match training)
H_DEFAULT = {
    "model": "dpn68_new",
    "img_size": 224,
    "dropout": 0.3,
    "num_classes": 2,
}

# -----------------------------
# Cached loaders (download once & reuse)
# -----------------------------
@st.cache_resource(show_spinner="Downloading YOLO ONNX from Hugging Face…")
def load_yolo_onnx():
    onnx_path = hf_download(
        repo_id=HF_MODEL_REPO_YOLO,
        filename=HF_FILENAME_YOLO,
        repo_type="model",
        token=HF_TOKEN,
    )
    return YOLOOnnx(onnx_path)

@st.cache_resource(show_spinner="Downloading DPN-68 checkpoint from Hugging Face…")
def load_dpn_model(hdict: dict):
    ckpt_path = hf_download(
        repo_id=HF_MODEL_REPO_DPN,
        filename=HF_FILENAME_DPN,
        repo_type="model",
        token=HF_TOKEN,
    )
    model = PneumoniaModel.load_from_checkpoint(
        ckpt_path, h=hdict, strict=False, map_location=DEVICE
    )
    model.to(DEVICE)
    model.eval()
    return model

# -----------------------------
# Utils
# -----------------------------
def pil_from_upload(file) -> Image.Image:
    data = file.read()
    im = Image.open(io.BytesIO(data)).convert("RGB")
    return im

def crop_by_boxes(im: Image.Image, boxes: List[Tuple[float,float,float,float,float]]) -> Image.Image:
    """Pick the largest detected box and crop it with safe padding."""
    if not boxes:
        return im
    # choose largest area
    areas = [(x2-x1)*(y2-y1) for (x1,y1,x2,y2,_) in boxes]
    i = int(np.argmax(areas))
    x1, y1, x2, y2, _ = boxes[i]
    # add 2% padding
    pad_x = 0.02 * (x2 - x1)
    pad_y = 0.02 * (y2 - y1)
    x1 = max(0, int(round(x1 - pad_x)))
    y1 = max(0, int(round(y1 - pad_y)))
    x2 = min(im.width,  int(round(x2 + pad_x)))
    y2 = min(im.height, int(round(y2 + pad_y)))
    return im.crop((x1, y1, x2, y2))

def preprocess_for_classifier(im: Image.Image, img_size: int = 224) -> torch.Tensor:
    """Resize + normalize to ImageNet stats, CHW float32 tensor on CPU."""
    im_resized = im.resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(im_resized).astype(np.float32) / 255.0  # HWC
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)  # CHW
    return torch.from_numpy(arr).unsqueeze(0)  # 1x3xHxW

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Pediatric TB X-ray App", layout="wide")
st.title("Pediatric Chest X-ray — Lung Detection, Classification & Grad-CAM")

with st.sidebar:
    st.header("Detector (YOLO-ONNX)")
    conf_thres = st.slider("Confidence", 0.0, 1.0, 0.25, 0.01)
    iou_thres  = st.slider("IoU",        0.0, 1.0, 0.75, 0.01)
    det_imgsz  = st.selectbox("Detector input size", [640, 512, 416], index=0)

    st.header("Classifier (DPN-68)")
    cls_imgsz  = st.selectbox("Classifier input size", [224], index=0)

    st.header("Grad-CAM")
    cam_method = st.selectbox("Method", ["gradcam", "gradcam++"], index=0)
    cam_alpha  = st.slider("Overlay α", 0.0, 1.0, 0.5, 0.05)
    cam_thr    = st.slider("Contour Threshold", 0.0, 1.0, 0.4, 0.01)

st.write("Upload a frontal chest X-ray (PNG/JPG).")

uploaded = st.file_uploader("Image", type=["png","jpg","jpeg","bmp","tif","tiff"])

if uploaded is None:
    st.info("Waiting for an image…")
    st.stop()

# Load models
try:
    yolo = load_yolo_onnx()
except Exception as e:
    st.error(f"Failed to load YOLO ONNX. Make sure '{HF_FILENAME_YOLO}' exists in {HF_MODEL_REPO_YOLO}. Error: {e}")
    st.stop()

try:
    model = load_dpn_model(H_DEFAULT)
except Exception as e:
    st.error(f"Failed to load DPN-68 ckpt from {HF_MODEL_REPO_DPN}/{HF_FILENAME_DPN}: {e}")
    st.stop()

# Read image
im0 = pil_from_upload(uploaded)

# 1) Detection (ONNX) → crop lungs
dets = yolo.predict(im0, conf_thres=conf_thres, iou_thres=iou_thres, img_size=int(det_imgsz))
im_det = im0.copy()
draw = ImageDraw.Draw(im_det)
for (x1,y1,x2,y2,score) in dets:
    draw.rectangle([x1,y1,x2,y2], outline=(0,255,0), width=3)
    draw.text((x1+2, y1+2), f"{score:.2f}", fill=(0,255,0))

im_crop = crop_by_boxes(im0, dets)

# 2) Classification (DPN-68 LightningModule)
inp = preprocess_for_classifier(im_crop, img_size=int(cls_imgsz))
with torch.inference_mode():
    logits = model(inp.to(DEVICE))
    probs = torch.softmax(logits.float(), dim=1).cpu().numpy()[0]
    pred  = int(probs.argmax())
classes = ["normal", "normal_not"]
pred_text = f"Pred: {classes[pred]}  |  P(normal_not)={probs[1]:.4f}"

# 3) Grad-CAM on cropped lungs
cam_map = compute_cam_map(
    model=model,
    input_tensor=inp.to(DEVICE),
    method=cam_method,
    class_idx=1,  # abnormal class index
    upsample_to=(im_crop.height, im_crop.width),
)

# Visualizations
overlay = heatmap_overlay(im_crop, cam_map, alpha=float(cam_alpha), cmap_name="jet")
boxes = contours_and_bboxes(cam_map, thr=float(cam_thr))
contour_img = draw_bboxes(im_crop, boxes, color=(255,0,0), width=3)
bbox_img    = draw_bboxes(overlay, boxes, color=(255,255,255), width=2)

# Layout
c1, c2 = st.columns(2)
with c1:
    st.subheader("Original + Detected Lungs")
    st.image(im_det, use_column_width=True)
with c2:
    st.subheader("Cropped Lungs")
    st.image(im_crop, use_column_width=True)

c3, c4 = st.columns(2)
with c3:
    st.subheader("Grad-CAM Overlay")
    st.image(overlay, use_column_width=True)
with c4:
    st.subheader("Grad-CAM (BBoxes)")
    st.image(bbox_img, use_column_width=True)

st.success(pred_text)

