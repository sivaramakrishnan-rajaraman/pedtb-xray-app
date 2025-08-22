# streamlit_app.py
from __future__ import annotations
import os
from typing import Tuple, Optional

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw

import torch
from torchvision import transforms

from src.hf_utils import hf_download
from src.pneumonia_model import PneumoniaModel
from src.cam_utils import compute_cam_mask, overlay_heatmap_on_rgb, contours_and_boxes_on_rgb
from src.yolo_onnx import YOLOOnnx

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Pediatric TB X-ray App", layout="wide")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hugging Face
HF_TOKEN = st.secrets.get("HF_TOKEN", None)  # optional (public repos don't need it)

# Your repos / filenames (edit if you used different names)
REPO_YOLO = "sivaramakrishhnan/cxr-yolo12s-lung"
YOLO_ONNX = "yolo12s_lung_nms.onnx"  # must be ONNX with nms=True

REPO_DPN  = "sivaramakrishhnan/cxr-dpn68-tb-cls"
DPN_CKPT  = "dpn68_fold2.ckpt"

# Classifier config
HCLS = {
    "model": "dpn68_new",
    "img_size": 224,
    "num_classes": 2,
    "dropout": 0.3,
}

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((HCLS["img_size"], HCLS["img_size"])),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

# -------------------------------
# Cached model loaders
# -------------------------------
@st.cache_resource(show_spinner="Loading YOLO (ONNX)…")
def load_yolo() -> YOLOOnnx:
    onnx_path = hf_download(REPO_YOLO, YOLO_ONNX, token=HF_TOKEN)
    return YOLOOnnx(onnx_path, img_size=640)

@st.cache_resource(show_spinner="Loading DPN-68 classifier…")
def load_cls() -> PneumoniaModel:
    ckpt_path = hf_download(REPO_DPN, DPN_CKPT, token=HF_TOKEN)
    model, info = PneumoniaModel.load_from_ckpt_auto_strict(ckpt_path, HCLS, map_location=DEVICE)
    model.to(DEVICE).eval()
    return model

# -------------------------------
# Detector wrapper
# -------------------------------
def detect_lung_bbox(
    det_model: YOLOOnnx,
    rgb: np.ndarray,
    conf: float,
) -> Optional[Tuple[int,int,int,int]]:
    xyxy, scores, cls = det_model.detect(rgb, conf_thres=conf, iou_thres=0.75)
    if xyxy.shape[0] == 0:
        return None
    areas = (xyxy[:,2] - xyxy[:,0]) * (xyxy[:,3] - xyxy[:,1])
    i = int(np.argmax(areas))
    x1, y1, x2, y2 = xyxy[i].astype(int)
    return x1, y1, x2, y2

# -------------------------------
# UI
# -------------------------------
st.title("Pediatric TB X-ray — Detection ▸ Classification ▸ Grad-CAM")

with st.sidebar:
    st.header("Settings")
    det_conf = st.slider("YOLO confidence", 0.05, 0.95, 0.25, 0.05)
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

# Read image as RGB
im = Image.open(upl).convert("RGB")
orig_rgb = np.array(im, dtype=np.uint8)

st.subheader("Original")
st.image(im, caption="Uploaded image", use_container_width=True)

# Load models
det_model = load_yolo()
cls_model = load_cls()

# Detect lungs on ORIGINAL coords
bbox = detect_lung_bbox(det_model, orig_rgb, det_conf)
if bbox is None:
    st.error("No lung region detected above the selected confidence.")
    st.stop()

x1, y1, x2, y2 = bbox
draw_det = im.copy()
ImageDraw.Draw(draw_det).rectangle([(x1, y1), (x2, y2)], outline=(0,255,0), width=3)
st.subheader("Detection")
st.image(draw_det, caption="Lung bbox on original image", use_container_width=True)

# Crop lungs in ORIGINAL pixels
crop_rgb = orig_rgb[y1:y2, x1:x2].copy()
if crop_rgb.size == 0:
    st.error("Empty crop from detection. Try a different image/confidence threshold.")
    st.stop()

# Prepare classifier input (224×224)
inp = to_tensor(Image.fromarray(crop_rgb)).unsqueeze(0).to(DEVICE)

# Classification (no-grad OK)
with torch.no_grad():
    logits = cls_model(inp.float())
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred  = int(probs.argmax())

classes = ["normal", "normal_not"]
st.markdown(f"**Prediction**: {classes[pred]} &nbsp;&nbsp;|&nbsp;&nbsp; **P(normal_not)** = {probs[1]:.4f}")

# Grad-CAM (must allow grad)
cam_mask_224 = compute_cam_mask(
    model=cls_model,
    input_tensor=inp,          # (1,3,224,224)
    class_index=1,
    method=cam_method,
    use_aug_smooth=True,
    use_eigen_smooth=True,
)

# Map CAM to crop size (original resolution), then render overlays
crop_h, crop_w = crop_rgb.shape[:2]
cam_mask_crop = np.array(
    Image.fromarray((cam_mask_224 * 255).astype(np.uint8)).resize((crop_w, crop_h), Image.BILINEAR)
) / 255.0

heat_on_crop = overlay_heatmap_on_rgb(crop_rgb, cam_mask_crop, alpha=cam_alpha, cmap_name="hot")
cont_on_crop, box_on_crop = contours_and_boxes_on_rgb(
    crop_rgb, cam_mask_crop, threshold=cam_thr, line_color=(255,0,0), box_color=(255,0,0), thickness=3
)

# Paste these back to the full image (for context)
full_heat = orig_rgb.copy()
full_heat[y1:y2, x1:x2] = heat_on_crop
full_cont = orig_rgb.copy()
full_cont[y1:y2, x1:x2] = cont_on_crop
full_box  = orig_rgb.copy()
full_box[y1:y2, x1:x2]  = box_on_crop

# Show downsized (presentation only)
def show_small(title: str, rgb: np.ndarray):
    small = Image.fromarray(rgb).resize((224, 224), Image.BILINEAR)
    st.image(small, caption=title, use_container_width=False)

st.subheader("Grad-CAM on the crop (displayed downsized)")
c1, c2, c3 = st.columns(3)
with c1: show_small("Heatmap on crop", heat_on_crop)
with c2: show_small("Contours on crop", cont_on_crop)
with c3: show_small("BBox on crop", box_on_crop)

st.subheader("Grad-CAM pasted into full frame (displayed downsized)")
c4, c5, c6 = st.columns(3)
with c4: show_small("Heatmap on full frame", full_heat)
with c5: show_small("Contours on full frame", full_cont)
with c6: show_small("BBox on full frame", full_box)

st.success("Done.")
