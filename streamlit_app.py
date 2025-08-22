# streamlit_app.py

from __future__ import annotations

import os
from typing import Optional, List, Tuple

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- lightweight colormap (no matplotlib needed) ----
def jet_colormap(x: np.ndarray) -> np.ndarray:
    """
    x: float32 array in [0,1] -> returns uint8 RGB array same HxW
    Minimal JET-like colormap implemented with piecewise linear ramps.
    """
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)

# ------------------------------
# Hugging Face download (no secrets needed for PUBLIC repos)
# ------------------------------
try:
    from huggingface_hub import hf_hub_download
except Exception as e:
    st.stop()

def hf_download(repo_id: str, filename: str, token: Optional[str] = None) -> str:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model", token=token)

# ------------------------------
# ONNX Runtime YOLO wrapper (no OpenCV)
# ------------------------------
try:
    import onnxruntime as ort
    HAS_ORT = True
except Exception:
    HAS_ORT = False

class YOLOOnnxLite:
    """
    Minimal YOLO (Ultralytics-style) ONNX inference without OpenCV.
    Assumptions:
      - Single-class detector (lungs).
      - ONNX output: (1, N, 6+) with [x, y, w, h, obj_conf, (class_probs...)]
    We do:
      1) PIL letterbox to 640
      2) run session
      3) decode, confidence filter, NMS
      4) map boxes back to original image coords
    """

    def __init__(self, onnx_path: str, providers: Optional[List[str]] = None, input_size: int = 640):
        if not HAS_ORT:
            raise RuntimeError("onnxruntime not installed. Add `onnxruntime` to requirements.")
        self.onnx_path = onnx_path
        self.session = ort.InferenceSession(
            onnx_path,
            providers=providers or ["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.out_names = [o.name for o in self.session.get_outputs()]
        self.input_size = input_size

    @staticmethod
    def letterbox_pil(im: Image.Image, new_shape=640, color=(114, 114, 114)) -> Tuple[Image.Image, float, Tuple[int, int]]:
        w, h = im.size
        r = min(new_shape / w, new_shape / h)
        nw, nh = int(round(w * r)), int(round(h * r))
        im_resized = im.resize((nw, nh), Image.BILINEAR)
        new_im = Image.new("RGB", (new_shape, new_shape), color)
        dw = (new_shape - nw) // 2
        dh = (new_shape - nh) // 2
        new_im.paste(im_resized, (dw, dh))
        return new_im, r, (dw, dh)

    @staticmethod
    def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
        # standard NMS in numpy
        x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.clip(xx2 - xx1, 0, None)
            h = np.clip(yy2 - yy1, 0, None)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou <= iou_thres]
        return keep

    def predict(self, pil_rgb: Image.Image, conf_thres: float = 0.25, iou_thres: float = 0.75) -> List[Tuple[int, int, int, int]]:
        H0, W0 = pil_rgb.height, pil_rgb.width
        img_lb, r, (dw, dh) = self.letterbox_pil(pil_rgb, new_shape=self.input_size)

        arr = np.asarray(img_lb).astype(np.float32) / 255.0  # (H,W,3)
        arr = arr.transpose(2, 0, 1)[None, ...]  # (1,3,H,W)

        out = self.session.run(self.out_names, {self.input_name: arr})
        pred = out[0]  # assume single output
        if pred.ndim == 2:
            pred = pred[None, ...]  # (1, N, 6+)
        pred = np.squeeze(pred, axis=0)

        # Expect [x, y, w, h, obj_conf, (cls...)]
        if pred.shape[1] < 6:
            return []

        boxes_xyxy = []
        scores = []
        for row in pred:
            x, y, w, h, obj = row[:5]
            cls_prob = row[5:].max() if row.shape[0] > 5 else 1.0
            score = float(obj * cls_prob)
            if score < conf_thres:
                continue
            # xywh (center) -> xyxy on letterboxed image
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            boxes_xyxy.append([x1, y1, x2, y2])
            scores.append(score)

        if not boxes_xyxy:
            return []
        boxes = np.array(boxes_xyxy, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)

        # NMS on letterboxed coords
        keep = self.nms(boxes, scores, iou_thres)
        boxes = boxes[keep]

        # map back to original
        boxes[:, [0, 2]] -= dw
        boxes[:, [1, 3]] -= dh
        boxes /= r

        # clamp & convert to ints
        boxes[:, 0] = np.clip(boxes[:, 0], 0, W0)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, H0)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, W0)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, H0)

        out_boxes: List[Tuple[int, int, int, int]] = []
        for b in boxes:
            x1, y1, x2, y2 = b.astype(int).tolist()
            if x2 > x1 and y2 > y1:
                out_boxes.append((x1, y1, x2, y2))
        return out_boxes

# ------------------------------
# Import your Lightning classifier
# ------------------------------
from src.pneumonia_model import PneumoniaModel  # must be present in your repo


# ------------------------------
# CAM implementation (torchcam) without OpenCV
# ------------------------------
try:
    from torchcam.methods import GradCAM, GradCAMpp, LayerCAM, XGradCAM, SmoothGradCAMpp
    HAS_TORCHCAM = True
except Exception:
    HAS_TORCHCAM = False

def pick_cam(method: str):
    method = (method or "gradcam").lower()
    return {
        "gradcam": GradCAM,
        "gradcam++": GradCAMpp,
        "layercam": LayerCAM,
        "xgradcam": XGradCAM,
        "smoothgradcampp": SmoothGradCAMpp,
    }.get(method, GradCAM)

def find_last_conv(m: nn.Module) -> nn.Module:
    last = None
    for _, mod in m.named_modules():
        if isinstance(mod, nn.Conv2d):
            last = mod
    if last is None:
        raise RuntimeError("No Conv2d found for CAM target.")
    return last

def compute_cam_map(model: nn.Module, inp: torch.Tensor, class_idx: int, method: str) -> np.ndarray:
    """
    Run CAM with gradients enabled. Resizes CAM to input spatial size.
    Returns float32 HxW in [0,1].
    """
    if not HAS_TORCHCAM:
        raise RuntimeError("torchcam is not installed. Add `torchcam==0.4.0` to requirements.")
    target_layer = find_last_conv(model)
    CAMClass = pick_cam(method)
    cam_extractor = CAMClass(model, target_layer)
    # DO NOT wrap in no_grad
    scores = model(inp.float())
    cams = cam_extractor(class_idx, scores)
    cam = cams[0] if isinstance(cams, (list, tuple)) else cams
    # cam is a torch tensor (Hc,Wc) - upscale to input (H,W)
    cam_t = cam.unsqueeze(0).unsqueeze(0)  # (1,1,h,w)
    cam_up = F.interpolate(cam_t, size=inp.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam_up = cam_up.detach().cpu().numpy().astype(np.float32)
    mmin, mmax = float(cam_up.min()), float(cam_up.max())
    if mmax > mmin:
        cam_up = (cam_up - mmin) / (mmax - mmin)
    else:
        cam_up[:] = 0.0
    return cam_up

# ------------------------------
# Simple overlays (no OpenCV)
# ------------------------------
def overlay_heatmap_on_pil(base_rgb: Image.Image, mask01: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """
    base_rgb: PIL RGB
    mask01: HxW float in [0,1]
    returns: PIL RGB
    """
    base = np.asarray(base_rgb).astype(np.float32)
    heat = jet_colormap(mask01)
    # alpha blend
    out = (alpha * heat + (1.0 - alpha) * base).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out)

def mask_edges(mask01: np.ndarray, thr: float = 0.4) -> np.ndarray:
    """
    Returns boolean array of edge pixels for the thresholded mask.
    Edge = positive pixel with at least one 8-neighbor negative.
    """
    m = (mask01 >= thr)
    if not m.any():
        return np.zeros_like(m, dtype=bool)
    # 8-neighborhood check via rolls
    nbrs = [
        np.roll(m, 1, 0), np.roll(m, -1, 0),
        np.roll(m, 1, 1), np.roll(m, -1, 1),
        np.roll(np.roll(m, 1, 0), 1, 1),
        np.roll(np.roll(m, 1, 0), -1, 1),
        np.roll(np.roll(m, -1, 0), 1, 1),
        np.roll(np.roll(m, -1, 0), -1, 1),
    ]
    all_neighbors_pos = nbrs[0]
    for k in range(1, len(nbrs)):
        all_neighbors_pos = all_neighbors_pos & nbrs[k]
    # edge: pixel is pos but not all neighbors are pos
    edges = m & (~all_neighbors_pos)
    return edges

def draw_contours_pil(base_rgb: Image.Image, edges: np.ndarray, color=(255, 0, 0)) -> Image.Image:
    """
    Draw a 1px colored overlay for edge pixels on PIL image (RGB).
    """
    arr = np.asarray(base_rgb).copy()
    yy, xx = np.where(edges)
    arr[yy, xx] = np.array(color, dtype=np.uint8)
    return Image.fromarray(arr)

def draw_bbox_from_mask_pil(base_rgb: Image.Image, mask01: np.ndarray, thr: float = 0.4, color=(255, 0, 0), width: int = 3) -> Image.Image:
    """
    Draw a single tight bounding box around all positive pixels.
    (If you need multiple boxes for disjoint blobs, we can extend later.)
    """
    m = (mask01 >= thr)
    if not m.any():
        return base_rgb.copy()
    ys, xs = np.where(m)
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    out = base_rgb.copy()
    drw = ImageDraw.Draw(out)
    # Draw multiple rectangles to emulate thickness
    for t in range(width):
        drw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color, width=1)
    return out

# ------------------------------
# App UI
# ------------------------------
st.set_page_config(layout="wide", page_title="Pediatric TB X-ray App")
st.title("🫁 Pediatric TB X-ray • Lungs Detection → DPN-68 Classification → Grad-CAM (no OpenCV)")

st.sidebar.header("Hugging Face repos / files (PUBLIC)")
repo_yolo = st.sidebar.text_input("YOLO repo_id", "sivaramakrishhnan/cxr-yolo12s-lung")
file_yolo = st.sidebar.text_input("YOLO filename",  "best.onnx")   # ONNX strongly recommended

repo_dpn  = st.sidebar.text_input("Classifier repo_id", "sivaramakrishhnan/cxr-dpn68-tb-cls")
file_dpn  = st.sidebar.text_input("Classifier filename", "dpn68_fold2.ckpt")

st.sidebar.markdown("---")
st.sidebar.header("Detector settings")
conf_thres = st.sidebar.slider("Confidence", 0.0, 1.0, 0.25, 0.01)
iou_thres  = st.sidebar.slider("IoU",        0.0, 1.0, 0.75, 0.01)

st.sidebar.markdown("---")
st.sidebar.header("Grad-CAM settings")
cam_method = st.sidebar.selectbox("Method", ["gradcam", "gradcam++", "layercam", "xgradcam", "smoothgradcampp"], index=0)
cam_alpha  = st.sidebar.slider("Heatmap alpha", 0.0, 1.0, 0.5, 0.05)
cam_thr    = st.sidebar.slider("Threshold (contours/box)", 0.0, 1.0, 0.4, 0.01)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"Device: **{DEVICE}**")

# Classifier h dict (minimal inference keys)
HCLS = {
    "model": "dpn68_new",
    "img_size": 224,
    "dropout": 0.3,
    "num_classes": 2,
}
CLASSES = ["normal", "normal_not"]

# ---- Download models (cached) ----
@st.cache_resource(show_spinner="Downloading YOLO ONNX from HF…")
def load_detector(repo_id: str, filename: str) -> YOLOOnnxLite:
    path = hf_download(repo_id, filename)
    return YOLOOnnxLite(path, providers=["CPUExecutionProvider"], input_size=640)

@st.cache_resource(show_spinner="Downloading DPN-68 checkpoint from HF…")
def load_classifier(repo_id: str, filename: str, h: dict) -> nn.Module:
    path = hf_download(repo_id, filename)
    model = PneumoniaModel.load_from_checkpoint(path, h=h, strict=False, map_location=DEVICE)
    model.to(DEVICE).eval()
    return model

# Try to load
try:
    det = load_detector(repo_yolo, file_yolo)
    st.success(f"Detector loaded: {file_yolo}")
except Exception as e:
    st.error(f"Failed to load YOLO ONNX: {e}")
    det = None

try:
    clf = load_classifier(repo_dpn, file_dpn, HCLS)
    st.success(f"Classifier loaded: {file_dpn}")
except Exception as e:
    st.error(f"Failed to load classifier: {e}")
    clf = None

st.markdown("### 1) Upload a chest X-ray")
up = st.file_uploader("PNG/JPG", type=["png", "jpg", "jpeg"])
if up is None:
    st.stop()

# Read image as PIL RGB
try:
    pil = Image.open(up).convert("RGB")
except Exception:
    st.error("Failed to read image.")
    st.stop()

st.image(pil, caption="Original", use_container_width=True)

if det is None or clf is None:
    st.warning("Models unavailable. Fix errors above.")
    st.stop()

# ---- Detection ----
st.markdown("### 2) Detect lungs and crop")
with st.spinner("Running detector…"):
    boxes = det.predict(pil, conf_thres=conf_thres, iou_thres=iou_thres)

draw = pil.copy()
drawD = ImageDraw.Draw(draw)
for (x1, y1, x2, y2) in boxes:
    drawD.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)

st.image(draw, caption="Detections", use_container_width=True)

if not boxes:
    # fallback: center crop (90%)
    W, H = pil.size
    cw, ch = int(W * 0.9), int(H * 0.9)
    x1 = (W - cw) // 2
    y1 = (H - ch) // 2
    x2 = x1 + cw
    y2 = y1 + ch
    boxes = [(x1, y1, x2, y2)]

# take the largest box
areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in boxes]
bi = int(np.argmax(areas))
x1, y1, x2, y2 = boxes[bi]
crop = pil.crop((x1, y1, x2, y2))
st.image(crop, caption="Cropped lungs", use_container_width=True)

# ---- Preprocess for classifier ----
def preprocess_pil_for_classifier(im: Image.Image, size: int = 224) -> torch.Tensor:
    imr = im.resize((size, size), Image.BILINEAR)
    arr = np.asarray(imr).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)  # (3,H,W)
    t = torch.from_numpy(arr)[None, ...]  # (1,3,H,W)
    return t.to(DEVICE)

inp = preprocess_pil_for_classifier(crop, size=HCLS["img_size"])

# ---- Classification (no grad is OK) ----
st.markdown("### 3) Classification")
with torch.no_grad():
    logits = clf(inp)
    prob = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
pred = int(np.argmax(prob))
st.success(f"Prediction: **{CLASSES[pred]}**   |   P(normal_not)={prob[1]:.4f}")

# ---- Grad-CAM (with grads enabled) ----
st.markdown("### 4) Grad-CAM")
with st.spinner("Computing Grad-CAM…"):
    cam_map = compute_cam_map(clf, inp, class_idx=1, method=cam_method)  # abnormal class=1
# Render overlays at crop resolution
heat = overlay_heatmap_on_pil(crop, cam_map, alpha=cam_alpha)
edges = mask_edges(cam_map, thr=cam_thr)
cont = draw_contours_pil(crop, edges, color=(255, 0, 0))
bbox = draw_bbox_from_mask_pil(crop, cam_map, thr=cam_thr, color=(255, 0, 0), width=3)

c1, c2, c3 = st.columns(3)
with c1:
    st.image(heat, caption="Heatmap overlay", use_container_width=True)
with c2:
    st.image(cont, caption="Contours", use_container_width=True)
with c3:
    st.image(bbox, caption="Bounding box", use_container_width=True)

st.info("Classification used torch.no_grad(); Grad-CAM ran with gradients enabled. No OpenCV used anywhere.")

