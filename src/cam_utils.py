# src/cam_utils.py
from __future__ import annotations
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import cv2

# pytorch-grad-cam (we use the same family you ran on Biowulf)
from pytorch_grad_cam import (
    GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM,
    XGradCAM, LayerCAM, FullGrad, HiResCAM,
    EigenCAM, EigenGradCAM
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ---------------------------
# CAM method resolver
# ---------------------------
_CAM_CLASSES = {
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

def _resolve_cam_class(method: str):
    key = (method or "gradcam").strip().lower()
    if key not in _CAM_CLASSES:
        raise ValueError(f"Unknown CAM method '{method}'. "
                         f"Choose one of: {', '.join(_CAM_CLASSES.keys())}")
    return _CAM_CLASSES[key]


# ---------------------------
# Target-layer discovery
# ---------------------------
def _last_conv_kgt1(module: nn.Module) -> Optional[nn.Conv2d]:
    """Return the last Conv2d with kernel > 1; else the last Conv2d; else None."""
    last_any = None
    last_kgt1 = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last_any = m
            k = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
            if max(k) > 1:
                last_kgt1 = m
    return last_kgt1 if last_kgt1 is not None else last_any

def discover_target_layer(model: nn.Module) -> nn.Module:
    """
    Matches your Biowulf pipeline:
      1) If the model exposes `cam_target`, use that (your PneumoniaModel sets it to post3x3[0]
         when added, or the last-stage conv otherwise).
      2) Else: last Conv2d with k>1 in `model.backbone`.
      3) Else: last Conv2d anywhere in the model.
    """
    # (1) Preferred: model sets cam_target (PneumoniaModel does this)
    if hasattr(model, "cam_target") and isinstance(model.cam_target, nn.Module):
        return model.cam_target

    # (2) Fallbacks
    if hasattr(model, "backbone"):
        tl = _last_conv_kgt1(model.backbone)
        if isinstance(tl, nn.Conv2d):
            return tl

    tl = _last_conv_kgt1(model)
    if isinstance(tl, nn.Conv2d):
        return tl

    raise RuntimeError("No suitable Conv2d layer found for Grad-CAM target.")


# ---------------------------
# Main: compute normalized CAM map
# ---------------------------
def compute_cam_mask(
    model: nn.Module,
    input_tensor: torch.Tensor,       # shape (1, 3, H, W), already normalized to training stats
    class_index: int = 1,             # abnormal class index
    method: str = "gradcam",
    aug_smooth: bool = True,
    eigen_smooth: bool = True,
) -> np.ndarray:
    """
    Returns: CAM mask as float32 in [0,1] with shape (H, W) == input spatial resolution.

    IMPORTANT:
    - Do NOT call this under torch.no_grad()/inference_mode(). This function enables grad internally.
    - The model is put in eval() here, but parameters are set requires_grad_(True) so CAM can backprop.
    """
    if input_tensor.ndim != 4 or input_tensor.size(0) != 1:
        raise ValueError("input_tensor must be batched (1, C, H, W)")

    device = input_tensor.device
    model.eval()  # eval is fine; we still need grads
    for p in model.parameters():
        p.requires_grad_(True)

    target_layer = discover_target_layer(model)
    CAMClass = _resolve_cam_class(method)
    cam = CAMClass(model=model, target_layers=[target_layer], reshape_transform=None)
    cam.batch_size = 1

    # Forward with grads enabled
    with torch.enable_grad():
        # class-agnostic EigenCAM doesn't use targets; others do
        if method.strip().lower() == "eigencam":
            mask = cam(
                input_tensor=input_tensor,
                aug_smooth=aug_smooth,
                eigen_smooth=eigen_smooth
            )[0]
        else:
            mask = cam(
                input_tensor=input_tensor,
                targets=[ClassifierOutputTarget(int(class_index))],
                aug_smooth=aug_smooth,
                eigen_smooth=eigen_smooth
            )[0]

    # Normalize to [0,1]
    mask = np.asarray(mask, dtype=np.float32)
    mmin, mmax = float(mask.min()), float(mask.max())
    if mmax > mmin:
        mask = (mask - mmin) / (mmax - mmin)
    else:
        mask = np.zeros_like(mask, dtype=np.float32)

    # Release hooks (defensive)
    if hasattr(cam, "activations_and_grads"):
        try:
            cam.activations_and_grads.release()
        except Exception:
            pass

    return mask.astype(np.float32)  # (H, W) in [0,1]


# ---------------------------
# Rendering: overlay on BGR image
# ---------------------------
def overlay_heatmap_on_bgr(
    base_bgr: np.ndarray,   # uint8 (H, W, 3) BGR
    cam_mask: np.ndarray,   # float32 (h, w) in [0,1]; typically 224×224 (classifier input size)
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_HOT,  # match Biowulf
) -> np.ndarray:
    """
    Resizes CAM to the base image size, colorizes, and alpha-blends.
    Returns a new uint8 BGR image of the same size as base.
    """
    if base_bgr.dtype != np.uint8:
        raise ValueError("base_bgr must be uint8 BGR image.")
    H, W = base_bgr.shape[:2]

    cam_u8 = np.uint8(np.clip(cam_mask, 0.0, 1.0) * 255.0)
    cam_u8 = cv2.resize(cam_u8, (W, H), interpolation=cv2.INTER_LINEAR)

    heat = cv2.applyColorMap(cam_u8, colormap)
    overlay = cv2.addWeighted(heat, float(alpha), base_bgr, 1.0 - float(alpha), 0.0)
    return overlay

