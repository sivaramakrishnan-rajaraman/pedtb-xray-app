# streamlit_app.py
from __future__ import annotations
import os
import io
from typing import Tuple

import streamlit as st
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from ultralytics import YOLO
from huggingface_hub.utils import HfHubHTTPError

# Local modules
from src.hf_utils import hf_download
from src.pneumonia_model import PneumoniaModel
from src.cam_utils import compute_cam_map, discover_target_layer


# ------------------ App / page config ------------------
st.set_page_config(page_title="PedTB X-ray: Detection + Classification + CAM", layout="wide")
st.title("Pediatric TB X-ray: Lung Detection → TB Classification → Grad-CAM")

st.markdown("""
This app:
1. Runs a **YOLO12s** detector (single class "lung") to localize lungs.
2. Crops to the detected lung region.
3. Classifies the crop with a **DPN-68** model (best fold).
4. Shows **Grad-CAM** overlays (and contours/bboxes) for explainability.
""")

# ------------------ Defaults / constants ------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# HF model repos (PUBLIC)
HF_MODEL_REPO_YOLO = st.secrets.get("HF_MODEL_REPO_YOLO", "sivaramakrishhnan/cxr-yolo12s-lung")
HF_FILENAME_YOLO   = st.secrets.get("HF_FILENAME_YOLO",   "best.pt")

HF_MODEL_REPO_DPN  = st.secrets.get("HF_MODEL_REPO_DPN",  "sivaramakrishhnan/cxr-dpn68-tb-cls")
HF_FILENAME_DPN    = st.secrets.get("HF_FILENAME_DPN",    "dpn68_fold2.ckpt")

# Classification config
HDICT = {
    "model": "dpn68_new",   # accepts dpn68 or dpn68_new
    "img_size": 224,
    "dropout": 0.3,
    "num_classes": 2
}

# Preprocessing for classifier (match training)
def build_preprocess(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

PREPROC = build_preprocess(HDICT["img_size"])


# ------------------ Cached loaders ------------------
@st.cache_resource(show_spinner="Downloading YOLO12s weights from Hugging Face…")
def load_yolo_from_hf() -> YOLO:
    yolo_path = hf_download(
        repo_id=HF_MODEL_REPO_YOLO,
        filename=HF_FILENAME_YOLO,
        repo_type="model",
        token=None,
    )
    return YOLO(yolo_path)


@st.cache_resource(show_spinner="Downloading DPN-68 checkpoint from Hugging Face…")
def load_dpn_from_hf(hdict: dict) -> PneumoniaModel:
    ckpt_path = hf_download(
        repo_id=HF_MODEL_REPO_DPN,
        filename=HF_FILENAME_DPN,
        repo_type="model",
        token=None,
    )
    model = PneumoniaModel.load_from_checkpoint(
        ckpt_path, h=hdict, strict=False, map_location=DEVICE
    )
    model.to(DEVICE).eval()
    return model


# ------------------ Utility: run YOLO & crop ------------------
def detect_and_crop_lung(
    bgr: np.ndarray,
    yolo: YOLO,
    conf_thr: float = 0.25,
    iou_thr: float = 0.75,
) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    """
    Returns (cropped_bgr, (x1,y1,x2,y2)).
    If no detection, returns center crop (80%) and its box.
    """
    H, W = bgr.shape[:2]
    results = yolo.predict(
        bgr[..., ::-1],    # YOLO expects RGB
        conf=conf_thr,
        iou=iou_thr,
        verbose=False,
        device="cpu",      # Streamlit Cloud: CPU
        imgsz=max(H, W)
    )
    box = None
    if results and len(results[0].boxes) > 0:
        # Take highest-confidence box
        boxes = results[0].boxes
        scores = boxes.conf.cpu().numpy()
        best_i = int(np.argmax(scores))
        xyxy = boxes.xyxy[best_i].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy.tolist()
        # Clamp
        x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
        y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
        if x2 > x1 and y2 > y1:
            box = (x1, y1, x2, y2)

    if box is None:
        # Fallback: centered 80% crop
        cw, ch = int(0.8*W), int(0.8*H)
        x1 = (W - cw)//2; y1 = (H - ch)//2
        x2 = x1 + cw; y2 = y1 + ch
        box = (x1, y1, x2, y2)

    x1, y1, x2, y2 = box
    crop = bgr[y1:y2, x1:x2].copy()
    return crop, box


def draw_box(bgr: np.ndarray, box: Tuple[int,int,int,int], color=(0,255,0), thickness=2):
    x1, y1, x2, y2 = box
    out = bgr.copy()
    cv2.rectangle(out, (x1,y1), (x2,y2), color, thickness)
    return out


# ------------------ Sidebar controls ------------------
st.sidebar.header("Settings")

st.sidebar.subheader("YOLO lung detector")
conf_thr = st.sidebar.slider("Confidence", 0.0, 1.0, 0.25, 0.01)
iou_thr  = st.sidebar.slider("IoU",        0.0, 1.0, 0.75, 0.01)

st.sidebar.subheader("Grad-CAM")
cam_method = st.sidebar.selectbox("Method", ["gradcam", "gradcam++"], index=0)
heatmap_alpha = st.sidebar.slider("Heatmap α", 0.0, 1.0, 0.5, 0.05)
cam_threshold = st.sidebar.slider("Contour threshold", 0.0, 1.0, 0.4, 0.05)

st.sidebar.subheader("Display")
show_heatmap = st.sidebar.checkbox("Show heatmap overlay", True)
show_contours = st.sidebar.checkbox("Show contours", True)
show_bboxes = st.sidebar.checkbox("Show CAM bounding boxes", True)


# ------------------ Main UI: upload image ------------------
uploaded = st.file_uploader("Upload a chest X-ray (PNG/JPG).", type=["png","jpg","jpeg"])
colL, colR = st.columns([1,1])

if uploaded is None:
    st.info("Upload a CXR image to start.")
    st.stop()

# Read image as BGR np.array
file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if bgr is None:
    st.error("Failed to decode image.")
    st.stop()

with colL:
    st.subheader("Original")
    st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Input", use_container_width=True)

# ------------------ Load models ------------------
try:
    yolo = load_yolo_from_hf()
    dpn = load_dpn_from_hf(HDICT)
except HfHubHTTPError as e:
    st.error(f"Failed to download weights from Hugging Face: {e}")
    st.stop()

# ------------------ Detect lungs & crop ------------------
crop_bgr, box = detect_and_crop_lung(bgr, yolo, conf_thr=conf_thr, iou_thr=iou_thr)
boxed = draw_box(bgr, box, color=(0,255,0), thickness=2)

with colR:
    st.subheader("Lung detection")
    st.image(cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB), caption=f"YOLO box: {box}", use_container_width=True)

st.subheader("Cropped lung region")
st.image(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), caption="Crop", use_container_width=True)

# ------------------ Classify crop ------------------
# Preprocess to 224x224 + normalize
crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
inp = PREPROC(crop_rgb)            # (3,224,224)
inp = inp.unsqueeze(0).to(DEVICE)  # (1,3,224,224)

with torch.no_grad():
    logits = dpn(inp).float()
    probs = F.softmax(logits, dim=1)[0].cpu().numpy()

prob_normal = float(probs[0])
prob_abn    = float(probs[1])
pred_label = "normal_not (abnormal)" if prob_abn >= prob_normal else "normal"

st.markdown(f"**Prediction:** `{pred_label}`  —  P(abnormal)=**{prob_abn:.3f}**  |  P(normal)=**{prob_normal:.3f}**")

# ------------------ Grad-CAM on crop ------------------
# Compute CAM map on the input crop, then resize to crop size for overlay
cam_map = compute_cam_map(
    model=dpn,
    input_tensor=inp,
    method=cam_method,
    target_layer=discover_target_layer(dpn),
    class_idx=1,
    upsample_to=crop_bgr.shape[:2],   # (H,W) of the crop
).numpy()

# Build overlays
cam_u8 = (np.clip(cam_map, 0, 1) * 255).astype(np.uint8)
heat = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
overlay = cv2.addWeighted(heat, float(heatmap_alpha), crop_bgr, 1.0 - float(heatmap_alpha), 0.0)

# Contours / bboxes
contour_img = crop_bgr.copy()
bbox_img = crop_bgr.copy()
if show_contours or show_bboxes:
    thr = int(255 * float(cam_threshold))
    _, binary = cv2.threshold(cam_u8, thr, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if show_contours:
        cv2.drawContours(contour_img, contours, -1, (0,0,255), 2, lineType=cv2.LINE_AA)
    if show_bboxes:
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(bbox_img, (x,y), (x+w,y+h), (0,0,255), 2, lineType=cv2.LINE_AA)

# ------------------ Display CAM outputs ------------------
st.subheader("Explainability (Grad-CAM)")
cc1, cc2, cc3 = st.columns(3)
if show_heatmap:
    cc1.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption=f"Heatmap (α={heatmap_alpha:.2f})", use_container_width=True)
else:
    cc1.image(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), caption="Crop (no heatmap)", use_container_width=True)

cc2.image(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB), caption=f"Contours (thr={cam_threshold:.2f})", use_container_width=True)
cc3.image(cv2.cvtColor(bbox_img,    cv2.COLOR_BGR2RGB), caption=f"BBoxes (thr={cam_threshold:.2f})", use_container_width=True)

st.success("Done.")
