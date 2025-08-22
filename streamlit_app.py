# streamlit_app.py
# streamlit_app.py
from __future__ import annotations

import os
import io
import time
from typing import Tuple, Optional, List

import streamlit as st
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn.functional as F

# ---- Our project helpers (must exist in your repo under src/) ----
from src.hf_utils import hf_download
from src.pneumonia_model import PneumoniaModel
from src.cam_utils import compute_cam_map, heatmap_overlay, contours_and_boxes

# Optional ONNX path for detector
try:
    from src.yolo_onnx import YOLOOnnx  # preferred on Streamlit Cloud
    HAS_ONNX = True
except Exception:
    HAS_ONNX = False

# Optional Ultralytics fallback if you kept a .pt detector and want to use it
try:
    from ultralytics import YOLO as UltralyticsYOLO
    HAS_ULTRA = True
except Exception:
    HAS_ULTRA = False


# =========================
# App config
# =========================
st.set_page_config(page_title="Pediatric TB X-ray • Detection + Classification + Grad-CAM",
                   page_icon="🫁",
                   layout="wide")

st.title("🫁 Pediatric TB X-ray: Lungs Detection → DPN-68 Classification → Grad-CAM")
st.caption("Upload a chest X-ray → detect lungs → crop → classify with DPN-68 → visualize Grad-CAM (contours & boxes).")

# =========================
# Sidebar controls
# =========================
st.sidebar.header("Model Sources (Hugging Face Hub)")
# Your public repos & filenames (edit if different)
DEFAULT_REPO_YOLO = "sivaramakrishhnan/cxr-yolo12s-lung"
DEFAULT_FILE_YOLO = "best.onnx"  # ← Prefer ONNX on Streamlit Cloud. If you only have .pt, set "best.pt"

DEFAULT_REPO_DPN  = "sivaramakrishhnan/cxr-dpn68-tb-cls"
DEFAULT_FILE_DPN  = "dpn68_fold2.ckpt"

repo_yolo = st.sidebar.text_input("YOLO repo_id", value=DEFAULT_REPO_YOLO, help="owner/name on HF")
file_yolo = st.sidebar.text_input("YOLO filename", value=DEFAULT_FILE_YOLO, help="e.g., best.onnx or best.pt")

repo_dpn  = st.sidebar.text_input("Classifier repo_id", value=DEFAULT_REPO_DPN)
file_dpn  = st.sidebar.text_input("Classifier filename", value=DEFAULT_FILE_DPN)

st.sidebar.markdown("---")
st.sidebar.header("YOLO Detector Settings")
conf_thres = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
iou_thres  = st.sidebar.slider("IoU threshold",        0.0, 1.0, 0.75, 0.01)

st.sidebar.markdown("---")
st.sidebar.header("Grad-CAM Settings")
cam_method  = st.sidebar.selectbox(
    "CAM method",
    ["gradcam", "gradcam++", "layercam", "xgradcam", "smoothgradcampp"],
    index=0
)
cam_alpha   = st.sidebar.slider("Heatmap alpha", 0.0, 1.0, 0.5, 0.05)
cam_thresh  = st.sidebar.slider("Contour threshold", 0.0, 1.0, 0.4, 0.01)

st.sidebar.markdown("---")
st.sidebar.header("Compute")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"**Device:** {DEVICE}")

# Optional token (for private repos; not required for public)
HF_TOKEN = st.secrets.get("HF_TOKEN", None)


# =========================
# Small helpers
# =========================
@st.cache_resource(show_spinner="Downloading YOLO weights from Hugging Face…")
def load_yolo_detector(repo_id: str, filename: str, token: Optional[str]) -> dict:
    """
    Download detector weights. If it's .onnx => use YOLOOnnx (preferred).
    If it's .pt => use Ultralytics YOLO (fallback).
    Returns dict {kind: "onnx"|"ultra", "model": <obj>, "path": <path>}.
    """
    local_path = hf_download(repo_id=repo_id, filename=filename, repo_type="model", token=token)
    kind = "onnx" if filename.lower().endswith(".onnx") else "ultra"
    if kind == "onnx":
        if not HAS_ONNX:
            raise RuntimeError("src/yolo_onnx.py not available or failed to import, but ONNX file was provided.")
        model = YOLOOnnx(local_path)  # your wrapper class
    else:
        if not HAS_ULTRA:
            raise RuntimeError("Ultralytics YOLO not installed but a .pt file was provided.")
        model = UltralyticsYOLO(local_path)
    return {"kind": kind, "model": model, "path": local_path}


@st.cache_resource(show_spinner="Downloading DPN-68 checkpoint from Hugging Face…")
def load_dpn_classifier(repo_id: str, filename: str, token: Optional[str], hdict: dict):
    """
    Download and load your LightningModule from .ckpt.
    """
    ckpt_path = hf_download(repo_id=repo_id, filename=filename, repo_type="model", token=token)
    model = PneumoniaModel.load_from_checkpoint(
        ckpt_path,
        h=hdict,
        strict=False,
        map_location=DEVICE
    )
    model.to(DEVICE).eval()
    return model


def preprocess_cxr_for_classifier(img_bgr: np.ndarray, size: int = 224) -> torch.Tensor:
    """
    OpenCV BGR (H, W, 3) → normalized tensor (1, 3, size, size)
    """
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img  = (img - mean) / std
    img  = np.transpose(img, (2, 0, 1))  # (3, H, W)
    t = torch.from_numpy(img).unsqueeze(0)  # (1, 3, H, W)
    return t.to(DEVICE)


def draw_boxes(bgr: np.ndarray, boxes_xyxy: List[Tuple[int, int, int, int]], color=(0, 255, 0), thick=2) -> np.ndarray:
    out = bgr.copy()
    for (x1, y1, x2, y2) in boxes_xyxy:
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
    return out


def crop_to_box(bgr: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return bgr.copy()
    return bgr[y1:y2, x1:x2]


def run_yolo_detect(det: dict, bgr: np.ndarray, conf: float, iou: float) -> List[Tuple[int, int, int, int]]:
    """
    Return list of lung boxes [(x1,y1,x2,y2)] in absolute pixel coords.
    Tries to be robust to different wrapper APIs.
    """
    kind = det["kind"]
    model = det["model"]
    H, W = bgr.shape[:2]

    if kind == "onnx":
        # Your YOLOOnnx wrapper: try common method names
        if hasattr(model, "predict"):
            boxes = model.predict(bgr, conf_thres=conf, iou_thres=iou)
        elif hasattr(model, "infer"):
            boxes = model.infer(bgr, conf_thres=conf, iou_thres=iou)
        else:
            # last resort: call like a function
            boxes = model(bgr, conf_thres=conf, iou_thres=iou)

        # Expect list of xyxy abs coords; if normalized were returned, convert
        parsed = []
        for b in boxes:
            if isinstance(b, (list, tuple)) and len(b) >= 4:
                x1, y1, x2, y2 = b[:4]
                # Heuristic: if coords in [0,1], treat as normalized
                if 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0:
                    x1 = int(round(x1 * W))
                    y1 = int(round(y1 * H))
                    x2 = int(round(x2 * W))
                    y2 = int(round(y2 * H))
                parsed.append((int(x1), int(y1), int(x2), int(y2)))
        return parsed

    # Ultralytics YOLO (.pt) fallback
    results = model.predict(bgr, imgsz=max(H, W), conf=conf, iou=iou, device="cpu", verbose=False)
    parsed = []
    for r in results:
        if r.boxes is None:  # no dets
            continue
        xyxy = r.boxes.xyxy  # tensor Nx4
        for row in xyxy.cpu().numpy().astype(int):
            x1, y1, x2, y2 = map(int, row[:4])
            parsed.append((x1, y1, x2, y2))
    return parsed


# =========================
# Load models (cached)
# =========================
# Classifier h-dict (inference-time essentials)
HCLS = {
    "model": "dpn68_new",
    "img_size": 224,
    "dropout": 0.3,
    "num_classes": 2,
    # other keys are harmless; kept minimal for clarity
}
CLASSES = ["normal", "normal_not"]  # index 1 is "abnormal"

try:
    det_bundle = load_yolo_detector(repo_yolo, file_yolo, HF_TOKEN)
    st.success(f"Loaded detector: {det_bundle['path']}  ({det_bundle['kind']})")
except Exception as e:
    st.error(f"Failed to load detector from {repo_yolo}/{file_yolo}: {e}")
    det_bundle = None

try:
    dpn_model = load_dpn_classifier(repo_dpn, file_dpn, HF_TOKEN, HCLS)
    st.success(f"Loaded classifier: {repo_dpn}/{file_dpn} on {DEVICE}")
except Exception as e:
    st.error(f"Failed to load DPN-68 ckpt from {repo_dpn}/{file_dpn}: {e}")
    dpn_model = None


# =========================
# File uploader
# =========================
st.markdown("### 1) Upload a chest X-ray image")
up = st.file_uploader("Choose a PNG/JPG chest X-ray", type=["png", "jpg", "jpeg"])
if up is None:
    st.info("Awaiting image upload…")
    st.stop()

# Read as BGR for OpenCV processing
file_bytes = np.frombuffer(up.read(), np.uint8)
bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if bgr is None:
    st.error("Failed to decode image.")
    st.stop()

st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Original image", use_container_width=True)

if (det_bundle is None) or (dpn_model is None):
    st.warning("Models not ready. Fix the errors above and rerun.")
    st.stop()


# =========================
# 2) Lung detection & crop
# =========================
st.markdown("### 2) Lung detection (YOLO) and cropping")
with st.spinner("Running detector…"):
    boxes = run_yolo_detect(det_bundle, bgr, conf_thres, iou_thres)

if not boxes:
    st.warning("No lungs detected. Falling back to center crop.")
    H, W = bgr.shape[:2]
    # 90% center crop
    w2, h2 = int(W * 0.9), int(H * 0.9)
    x1 = (W - w2) // 2
    y1 = (H - h2) // 2
    x2 = x1 + w2
    y2 = y1 + h2
    boxes = [(x1, y1, x2, y2)]

det_vis = draw_boxes(bgr, boxes)
st.image(cv2.cvtColor(det_vis, cv2.COLOR_BGR2RGB), caption="Detections", use_container_width=True)

# For this pipeline, take the highest-area box (in case of multiple)
areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in boxes]
best_idx = int(np.argmax(areas))
crop_bgr = crop_to_box(bgr, boxes[best_idx])
st.image(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), caption="Cropped lungs", use_container_width=True)


# =========================
# 3) Classification (no_grad OK)
# =========================
st.markdown("### 3) DPN-68 classification")
inp = preprocess_cxr_for_classifier(crop_bgr, size=HCLS["img_size"])

with torch.no_grad():   # ✅ classification may run without grads
    logits = dpn_model(inp)
    probs  = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
pred_idx = int(np.argmax(probs))
pred_text = f"**Prediction:** {CLASSES[pred_idx]}  |  **P(normal_not)** = {probs[1]:.4f}"
st.success(pred_text)


# =========================
# 4) Grad-CAM (DO NOT wrap in no_grad/inference_mode)
# =========================
st.markdown("### 4) Grad-CAM visualization")
with st.spinner("Computing Grad-CAM…"):
    # ⚠️ DO NOT put this inside torch.no_grad() or torch.inference_mode()
    cam_map = compute_cam_map(
        model=dpn_model,
        input_tensor=inp,          # (1,3,224,224)
        method=cam_method,
        class_idx=1,               # abnormal class index
        upsample_to=(im_crop.height, im_crop.width),
        use_autocast=False         # set True only if CUDA & you want AMP
    )

# Render overlays at original crop resolution
heat = heatmap_overlay(cam_map, crop_bgr, alpha=cam_alpha)
cont, box = contours_and_boxes(cam_map, crop_bgr, threshold=cam_thresh,
                               color=(0, 0, 255), thickness=3)

c1, c2, c3 = st.columns(3)
with c1:
    st.image(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), caption="Grad-CAM Heatmap", use_container_width=True)
with c2:
    st.image(cv2.cvtColor(cont, cv2.COLOR_BGR2RGB), caption="Contours", use_container_width=True)
with c3:
    st.image(cv2.cvtColor(box,  cv2.COLOR_BGR2RGB), caption="Bounding Boxes", use_container_width=True)

st.info("CAM is computed with gradients enabled (classification used no_grad). Change CAM method & thresholds in the sidebar.")
