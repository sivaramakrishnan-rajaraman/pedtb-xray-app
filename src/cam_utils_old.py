# src/cam_utils.py
from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import cv2

from pytorch_grad_cam import (
    GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM,
    XGradCAM, LayerCAM, FullGrad, HiResCAM, EigenCAM, EigenGradCAM
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ---------------------------
# Target layer discovery
# ---------------------------
def _find_last_conv(module: nn.Module) -> Optional[nn.Conv2d]:
    last = None
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last

def discover_target_layer(model: nn.Module) -> nn.Module:
    if hasattr(model, "cam_target") and isinstance(model.cam_target, nn.Module):
        return model.cam_target
    if hasattr(model, "backbone"):
        last = _find_last_conv(model.backbone)
        if last is not None:
            return last
    last = _find_last_conv(model)
    if last is None:
        raise RuntimeError("No Conv2d layer found for CAM target.")
    return last

_CAM_MAP = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "scorecam": ScoreCAM,
    "ablationcam": AblationCAM,
    "xgradcam": XGradCAM,
    "layercam": LayerCAM,
    "fullgrad": FullGrad,
    "eigencam": EigenCAM,
    "eigengradcam": EigenGradCAM,
    "hirescam": HiResCAM,
}

def _build_cam(model: nn.Module, method: str = "gradcam"):
    CAMClass = _CAM_MAP.get((method or "gradcam").lower(), GradCAM)
    return CAMClass(model=model, target_layers=[discover_target_layer(model)], reshape_transform=None)

# ---------------------------
# CAM
# ---------------------------
def compute_cam_mask(
    model: nn.Module,
    input_tensor: torch.Tensor,  # (1,3,H,W), normalized, on model.device
    class_index: int = 1,
    method: str = "gradcam",
    aug_smooth: bool = True,
    eigen_smooth: bool = True,
) -> np.ndarray:
    """Returns float32 mask in [0,1] at input spatial resolution."""
    assert input_tensor.ndim == 4 and input_tensor.size(0) == 1, "input_tensor must be (1,C,H,W)"
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    cam = _build_cam(model, method)
    mask = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(int(class_index))],
        aug_smooth=bool(aug_smooth),
        eigen_smooth=bool(eigen_smooth),
    )[0]
    mask = np.asarray(mask, dtype=np.float32)
    mmin, mmax = float(mask.min()), float(mask.max())
    mask = (mask - mmin) / (mmax - mmin) if mmax > mmin else np.zeros_like(mask, dtype=np.float32)

    try:
        cam.activations_and_grads.release()
    except Exception:
        pass
    return mask

# ---------------------------
# Rendering (OpenCV BGR)
# ---------------------------
def overlay_heatmap_on_bgr(
    base_bgr: np.ndarray,     # uint8 (H,W,3)
    cam_mask: np.ndarray,     # float32 (Hc,Wc) in [0,1]
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_HOT,
) -> np.ndarray:
    H, W = base_bgr.shape[:2]
    cam_u8 = np.uint8(np.clip(cam_mask, 0, 1) * 255)
    if cam_u8.shape != (H, W):
        cam_u8 = cv2.resize(cam_u8, (W, H), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap(cam_u8, colormap)
    return cv2.addWeighted(heat, float(alpha), base_bgr, 1 - float(alpha), 0)

def contours_and_boxes_on_bgr(
    base_bgr: np.ndarray,     # uint8 (H,W,3)
    cam_mask: np.ndarray,     # float32 (Hc,Wc)
    threshold: float = 0.4,
    color: Tuple[int,int,int] = (0,0,255),
    thickness: int = 2,
    line_type: int = cv2.LINE_AA
):
    H, W = base_bgr.shape[:2]
    cam_u8 = np.uint8(np.clip(cam_mask, 0, 1) * 255)
    if cam_u8.shape != (H, W):
        cam_u8 = cv2.resize(cam_u8, (W, H), interpolation=cv2.INTER_NEAREST)

    thr = int(255 * float(threshold))
    _, binary = cv2.threshold(cam_u8, thr, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cont_img = base_bgr.copy()
    box_img  = base_bgr.copy()
    cv2.drawContours(cont_img, contours, -1, color, thickness, lineType=line_type)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(box_img, (x, y), (x + w, y + h), color, thickness, lineType=line_type)
    return cont_img, box_img

# Backward-compat names (if your app imported old ones)
def overlay_heatmap_on_rgb(base_rgb: np.ndarray, cam_mask: np.ndarray, alpha: float = 0.5):
    # assume RGB -> convert to BGR internally, then back
    bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
    out = overlay_heatmap_on_bgr(bgr, cam_mask, alpha=alpha)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

def contours_and_boxes_on_rgb(base_rgb: np.ndarray, cam_mask: np.ndarray, **kwargs):
    bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
    ci, bi = contours_and_boxes_on_bgr(bgr, cam_mask, **kwargs)
    return cv2.cvtColor(ci, cv2.COLOR_BGR2RGB), cv2.cvtColor(bi, cv2.COLOR_BGR2RGB)

# Legacy alias names to avoid import errors
def compute_cam_map(*a, **k): return compute_cam_mask(*a, **k)
def heatmap_overlay(cam_map, base_bgr, alpha=0.5): return overlay_heatmap_on_bgr(base_bgr, cam_map, alpha=alpha)
def contours_and_boxes(*a, **k): return contours_and_boxes_on_bgr(*a, **k)

__all__ = [
    "discover_target_layer",
    "compute_cam_mask",
    "overlay_heatmap_on_bgr",
    "contours_and_boxes_on_bgr",
    "overlay_heatmap_on_rgb",
    "contours_and_boxes_on_rgb",
    # aliases
    "compute_cam_map", "heatmap_overlay", "contours_and_boxes",
]

