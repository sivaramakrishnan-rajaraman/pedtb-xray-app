# src/cam_utils.py
from __future__ import annotations
from typing import Tuple, Optional, Dict
import numpy as np
import cv2
import torch
import torch.nn as nn

# pytorch-grad-cam
from pytorch_grad_cam import (
    GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM,
    XGradCAM, LayerCAM, FullGrad, HiResCAM, EigenCAM, EigenGradCAM
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ------------ target discovery ------------
def _find_last_conv(mod: nn.Module) -> Optional[nn.Conv2d]:
    last = None
    for _, m in mod.named_modules():
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

# ------------ CAM builder ------------
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

def build_cam(model: nn.Module, method: str = "gradcam") -> Tuple[nn.Module, nn.Module]:
    tl = discover_target_layer(model)
    CAMClass = _CAM_MAP.get(method.lower(), GradCAM)
    cam = CAMClass(model=model, target_layers=[tl], reshape_transform=None)
    cam.batch_size = 1
    return cam, tl

# ------------ main CAM call ------------
def compute_cam_mask(
    model: nn.Module,
    input_tensor: torch.Tensor,   # (1,3,H,W) normalized
    class_index: int = 1,
    method: str = "gradcam",
    use_aug_smooth: bool = True,
    use_eigen_smooth: bool = True,
) -> np.ndarray:
    """
    Return float32 CAM mask in [0,1] with spatial size == input HxW.
    DO NOT wrap this call in torch.no_grad().
    """
    cam, _ = build_cam(model, method=method)

    # pytorch-grad-cam returns input-sized mask already
    mask = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(int(class_index))],
        aug_smooth=bool(use_aug_smooth),
        eigen_smooth=bool(use_eigen_smooth),
    )[0]

    mask = np.asarray(mask, dtype=np.float32)
    # Normalize robustly
    mmin, mmax = float(mask.min()), float(mask.max())
    if mmax > mmin:
        mask = (mask - mmin) / (mmax - mmin)
    else:
        mask[:] = 0.0

    # free hooks
    try:
        cam.activations_and_grads.release()
    except Exception:
        pass

    return mask  # (H, W) in [0,1]

# ------------ rendering helpers ------------
def overlay_heatmap_on_bgr(
    base_bgr: np.ndarray,  # uint8 HxWx3 (original crop or full image)
    cam_mask: np.ndarray,  # float32 HxW (same spatial size as base_bgr when passed)
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_HOT,
) -> np.ndarray:
    cam_u8 = (np.clip(cam_mask, 0, 1) * 255).astype(np.uint8)
    if cam_u8.shape[:2] != base_bgr.shape[:2]:
        cam_u8 = cv2.resize(cam_u8, (base_bgr.shape[1], base_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap(cam_u8, colormap)
    out = cv2.addWeighted(heat, float(alpha), base_bgr, 1.0 - float(alpha), 0.0)
    return out

def contours_and_boxes_on_bgr(
    base_bgr: np.ndarray,
    cam_mask: np.ndarray,
    threshold: float = 0.4,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    cam_u8 = (np.clip(cam_mask, 0, 1) * 255).astype(np.uint8)
    if cam_u8.shape[:2] != base_bgr.shape[:2]:
        cam_u8 = cv2.resize(cam_u8, (base_bgr.shape[1], base_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    _, binary = cv2.threshold(cam_u8, int(255 * threshold), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cont_img = base_bgr.copy()
    box_img  = base_bgr.copy()

    for cnt in contours:
        cv2.drawContours(cont_img, [cnt], -1, color, thickness, lineType=cv2.LINE_AA)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(box_img, (x, y), (x + w, y + h), color, thickness, lineType=cv2.LINE_AA)

    return cont_img, box_img
