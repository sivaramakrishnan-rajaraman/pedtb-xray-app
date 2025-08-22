# streamlit_app.py
from __future__ import annotations
import os
import io
from typing import Tuple, Optional

import streamlit as st
import numpy as np
from PIL import Image
import cv2

import torch
from torchvision import transforms

from src.hf_utils import hf_download
from src.pneumonia_model import PneumoniaModel
from src.cam_utils import compute_cam_mask, overlay_heatmap_on_bgr, contours_and_boxes_on_bgr
from src.yolo_onnx import YOLOOnnx

# -------------------------------
# Config & secrets
# -------------------------------
st.set_page_config(page_title="Pediatric TB X-ray App", layout="wide")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# You made these public, so HF_TOKEN is optional
HF_TOKEN = st.secrets.get("HF_TOKEN", None)

# Repos and filenames (EDIT if you used other names)
REPO_YOLO = "sivaramakrishhnan/cxr-yolo12s-lung"
YOLO_PT   = "best.pt"           # .pt path in the repo (optional, used if ultralytics importable)
YOLO_ONNX = "yolo12s_lung_nms.onnx"  # <-- upload an ONNX with nms=True for the fallback

REPO_DPN  = "sivaramakrishhnan/cxr-dpn68-tb-cls"
DPN_CKPT  = "dpn68_fold2.ckpt"  # your best fold-2 or the one you prefer

# Classifier h
HCLS = {
    "model": "dpn68_new",
    "img_size": 224,
    "num_classes": 2,
    "dropout": 0.3,
}

# Preprocess for the classifier (match your train normalizations)
to_tensor = transforms.Compose([
    transforms.ToTensor(),  # converts PIL/np (H,W,3 uint8) -> float [0,1], C,H,W
    transforms.Resize((HCLS["img_size"], HCLS["img_size"])),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

# -------------------------------
# Try to import Ultralytics (optional)
# -------------------------------
_ULTRA_OK = False
try:
    from ultralytics import YOLO  # only if you installed locally with --no-deps
    _ULTRA_OK = True
except Exception:
    _ULTRA_OK = False


# -------------------------------
# Caching: models
# -------------------------------
@st.cache_resource(show_spinner="Loading DPN-68 classifier…")
def load_dpn() -> PneumoniaModel:
    ckpt_path = hf_download(REPO_DPN, DPN_CKPT, token=HF_TOKEN)
    model, info = PneumoniaModel.load_from_ckpt_auto_strict(ckpt_path, HCLS, map_location=DEVICE)
    model.to(DEVICE).eval()
    return model

@st.cache_resource(show_spinner="Loading YOLO detector…")
def load_yolo_detector(conf: float = 0.25, iou: float = 0.75):
    """
    Returns a tuple (mode, obj):
      - mode == "ultra": obj is an Ultralytics YOLO model (.pt)
      - mode == "onnx":  obj is YOLOOnnx instance
    """
    if _ULTRA_OK:
        try:
            yolo_pt_path = hf_download(REPO_YOLO, YOLO_PT, token=HF_TOKEN)
            mdl = YOLO(yolo_pt_path)
            return "ultra", mdl
        except Exception as e:
            st.warning(f"Ultralytics not available or failed to load .pt: {e}\nFalling back to ONNX.")
    # ONNX fallback
    onnx_path = hf_download(REPO_YOLO, YOLO_ONNX, token=HF_TOKEN)
    mdl = YOLOOnnx(onnx_path, img_size=640)
    return "onnx", mdl


# -------------------------------
# Detector wrapper
# -------------------------------
def detect_lung_bbox(
    det_kind: str,
    det_obj,
    bgr: np.ndarray,
    conf: float,
    iou: float
) -> Optional[Tuple[int,int,int,int]]:
    """
    Returns the largest bbox [x1,y1,x2,y2] in ORIGINAL coordinates (int),
    or None if no detection above threshold.
    """
    H, W = bgr.shape[:2]

    if det_kind == "ultra":
        # Ultralytics returns boxes in original coordinates already
        res = det_obj.predict(source=bgr[..., ::-1], imgsz=640, conf=conf, iou=iou, verbose=False)[0]
        if res.boxes is None or res.boxes.xyxy is None or len(res.boxes) == 0:
            return None
        xyxy = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        keep = scores >= conf
        if not keep.any():
            return None
        xyxy = xyxy[keep]
        # largest area
        areas = (xyxy[:,2]-xyxy[:,0]) * (xyxy[:,3]-xyxy[:,1])
        i = int(np.argmax(areas))
        x1,y1,x2,y2 = xyxy[i].astype(int)
        x1 = int(np.clip(x1, 0, W-1)); x2 = int(np.clip(x2, 0, W-1))
        y1 = int(np.clip(y1, 0, H-1)); y2 = int(np.clip(y2, 0, H-1))
        return x1,y1,x2,y2

    # ONNX fallback
    xyxy, scores, cls = det_obj.detect(bgr, conf_thres=conf, iou_thres=iou)
    if xyxy.shape[0] == 0:
        return None
    areas = (xyxy[:,2]-xyxy[:,0]) * (xyxy[:,3]-xyxy[:,1])
    i = int(np.argmax(areas))
    x1,y1,x2,y2 = xyxy[i].astype(int)
    return x1,y1,x2,y2


# -------------------------------
# UI
# -------------------------------
st.title("Pediatric TB X-ray — Detection ▸ Classification ▸ Grad-CAM")

with st.sidebar:
    st.header("Settings")
    det_conf = st.slider("YOLO confidence", 0.05, 0.95, 0.25, 0.05)
    det_iou  = st.slider("YOLO IoU",        0.10, 0.95, 0.75, 0.05)
    cam_method = st.selectbox(
        "CAM method",
        ["gradcam", "gradcam++", "xgradcam", "layercam", "eigencam", "eigengradcam", "hirescam", "scorecam", "ablationcam", "fullgrad"],
        index=0
    )
    cam_alpha = st.slider("CAM overlay alpha", 0.1, 0.9, 0.5, 0.05)
    cam_thr   = st.slider("CAM threshold (contours/boxes)", 0.05, 0.95, 0.40, 0.05)

upl = st.file_uploader("Upload a frontal chest X-ray (PNG/JPG)", type=["png","jpg","jpeg"])
if not upl:
    st.info("Upload an image to begin.")
    st.stop()

# Read original image as BGR uint8 for OpenCV ops
file_bytes = np.frombuffer(upl.read(), np.uint8)
orig_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if orig_bgr is None:
    st.error("Failed to read image.")
    st.stop()

st.subheader("Original")
st.image(orig_bgr[..., ::-1], caption="Uploaded image", use_container_width=True)

# Load models
det_kind, det_model = load_yolo_detector(det_conf, det_iou)
cls_model = load_dpn()

# Detect lung bbox in ORIGINAL coords
bbox = detect_lung_bbox(det_kind, det_model, orig_bgr, det_conf, det_iou)
if bbox is None:
    st.error("No lung region detected above the selected confidence.")
    st.stop()

x1,y1,x2,y2 = bbox
det_vis = orig_bgr.copy()
cv2.rectangle(det_vis, (x1,y1), (x2,y2), (0,255,0), 3, lineType=cv2.LINE_AA)
st.subheader("Detection")
st.image(det_vis[..., ::-1], caption="Lung bbox on original image", use_container_width=True)

# Crop lungs IN ORIGINAL PIXELS, then prepare classifier input (224x224)
crop_bgr = orig_bgr[y1:y2, x1:x2].copy()
if crop_bgr.size == 0:
    st.error("Empty crop from detection. Try a different image or thresholds.")
    st.stop()

crop_rgb = crop_bgr[..., ::-1]  # to RGB for torchvision
inp = to_tensor(Image.fromarray(crop_rgb)).unsqueeze(0).to(DEVICE)

# -------- Classification (no-grad OK) --------
with torch.no_grad():
    logits = cls_model(inp.float())
    probs  = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred   = int(probs.argmax())

classes = ["normal", "normal_not"]
pred_text = f"Prediction: **{classes[pred]}**    |    P(normal_not) = **{probs[1]:.4f}**"
st.markdown(pred_text)

# -------- Grad-CAM (must have grads) --------
# compute mask on the 224×224 input
cam_mask_224 = compute_cam_mask(
    model=cls_model,
    input_tensor=inp,          # (1,3,224,224)
    class_index=1,
    method=cam_method,
    use_aug_smooth=True,
    use_eigen_smooth=True,
)

# Map CAM to ORIGINAL CROP SIZE first, then overlay
crop_h, crop_w = crop_bgr.shape[:2]
cam_mask_crop = cv2.resize(cam_mask_224, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

heat_on_crop = overlay_heatmap_on_bgr(crop_bgr, cam_mask_crop, alpha=cam_alpha, colormap=cv2.COLORMAP_HOT)
cont_on_crop, box_on_crop = contours_and_boxes_on_bgr(crop_bgr, cam_mask_crop, threshold=cam_thr)

# Also paste the heat overlay back into the ORIGINAL full image for context
full_heat = orig_bgr.copy()
full_heat[y1:y2, x1:x2] = heat_on_crop
full_cont = orig_bgr.copy()
full_cont[y1:y2, x1:x2] = cont_on_crop
full_box  = orig_bgr.copy()
full_box[y1:y2, x1:x2]  = box_on_crop

# For presentation, downscale displays to 224×224 thumbnails (only for UI; processing was at native res)
def show_small(title: str, bgr_img: np.ndarray):
    small = cv2.resize(bgr_img, (224, 224), interpolation=cv2.INTER_AREA)
    st.image(small[..., ::-1], caption=title, use_container_width=False)

st.subheader("Explainability (on original crop; displayed downsized)")
col1, col2, col3 = st.columns(3)
with col1: show_small("Heatmap on crop", heat_on_crop)
with col2: show_small("Contours on crop", cont_on_crop)
with col3: show_small("BBox on crop", box_on_crop)

st.subheader("Explainability pasted in the full image (displayed downsized)")
col4, col5, col6 = st.columns(3)
with col4: show_small("Heatmap on full frame", full_heat)
with col5: show_small("Contours on full frame", full_cont)
with col6: show_small("BBox on full frame", full_box)

st.success("Done.")
