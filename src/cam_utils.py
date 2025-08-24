# src/cam_utils.py
from __future__ import annotations
from typing import Optional, Dict, Type, List
import numpy as np
import torch
import torch.nn as nn
import cv2

# Jacobgil's pytorch-grad-cam
from pytorch_grad_cam import (
    GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM,
    XGradCAM, LayerCAM, FullGrad, EigenCAM, EigenGradCAM, HiResCAM
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ---------------------------
# Choose CAM class by name
# ---------------------------
_CAM_CLASSES: Dict[str, Type] = {
    "gradcam":       GradCAM,
    "gradcam++":     GradCAMPlusPlus,
    "scorecam":      ScoreCAM,
    "ablationcam":   AblationCAM,
    "xgradcam":      XGradCAM,
    "layercam":      LayerCAM,
    "fullgrad":      FullGrad,
    "eigencam":      EigenCAM,
    "eigengradcam":  EigenGradCAM,
    "hirescam":      HiResCAM,
}

def _get_cam_class(method: str):
    m = (method or "gradcam").strip().lower()
    if m not in _CAM_CLASSES:
        raise ValueError(f"Unknown CAM method '{method}'. "
                         f"Supported: {', '.join(sorted(_CAM_CLASSES.keys()))}")
    return _CAM_CLASSES[m]

# ---------------------------
# Resolve a good target layer
# ---------------------------
def _find_last_conv(module: nn.Module) -> Optional[nn.Conv2d]:
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last

def _resolve_target_layer(model: nn.Module) -> nn.Module:
    """
    Priority:
      1) model.cam_target (your Pneumonia/TB model exposes this for post3x3 conv)
      2) any Conv2d inside model.post3x3
      3) last Conv2d in model.backbone
      4) last Conv2d anywhere in the model
    """
    if hasattr(model, "cam_target") and isinstance(model.cam_target, nn.Module):
        return model.cam_target
    p3 = getattr(model, "post3x3", None)
    if isinstance(p3, nn.Module):
        c = _find_last_conv(p3)
        if c is not None:
            return c
    if hasattr(model, "backbone") and isinstance(model.backbone, nn.Module):
        c = _find_last_conv(model.backbone)
        if c is not None:
            return c
    c = _find_last_conv(model)
    if c is None:
        raise RuntimeError("No Conv2d layer found to use for CAM.")
    return c

# ---------------------------
# Core: compute a [0,1] CAM mask
# ---------------------------
def compute_cam_mask(
    model: nn.Module,
    input_tensor: torch.Tensor,     # (1,3,H,W) on correct device
    class_index: int = 1,           # abnormal class
    method: str = "gradcam",
    aug_smooth: bool = True,
    eigen_smooth: bool = True,
) -> np.ndarray:
    """
    Returns a float32 CAM in [0,1], same spatial size as input_tensor (H, W).
    NOTE: must NOT be called under torch.no_grad().
    """
    assert input_tensor.ndim == 4 and input_tensor.size(0) == 1, \
        "input_tensor must be (1,3,H,W)"
    model.eval()

    # Make sure tensors can carry gradients for hooks/backprop
    for p in model.parameters():
        p.requires_grad_(True)

    target_layer = _resolve_target_layer(model)
    CAMClass = _get_cam_class(method)

    # Some methods ignore targets (e.g., EigenCAM), handle gracefully
    targets = None
    if CAMClass is not EigenCAM:
        targets = [ClassifierOutputTarget(int(class_index))]

    cam_obj = CAMClass(model=model, target_layers=[target_layer], use_cuda=input_tensor.device.type == "cuda")
    cam_obj.batch_size = 1

    with torch.enable_grad():
        masks: List[np.ndarray] = cam_obj(
            input_tensor=input_tensor.float(),
            targets=targets,
            aug_smooth=bool(aug_smooth),
            eigen_smooth=bool(eigen_smooth),
        )  # list[np.ndarray] of shape (H,W)
    # pytorch-grad-cam returns a list; use the first for batch size 1
    cam = masks[0].astype(np.float32)

    # Normalize robustly to [0,1]
    mn, mx = float(cam.min()), float(cam.max())
    if mx > mn:
        cam = (cam - mn) / (mx - mn)
    else:
        cam = np.zeros_like(cam, dtype=np.float32)

    # Cleanup hooks
    try:
        cam_obj.activations_and_grads.release()
    except Exception:
        pass

    return cam  # (H,W) float32 in [0,1]

# ---------------------------
# Thresholded, mask-aware overlay
# ---------------------------
def overlay_heatmap_on_bgr(
    base_bgr: np.ndarray,          # uint8 (H,W,3) BGR
    cam_mask: np.ndarray,          # float32 (h,w) in [0,1]
    alpha: float = 0.5,            # global max opacity in [0,1]
    colormap: int = cv2.COLORMAP_HOT,
    threshold: Optional[float] = 0.3, #show only activations >= threshold, otherwise None
) -> np.ndarray:
    """
    Apply colormap only where cam_mask >= threshold (if provided), and blend
    without dimming pixels outside the activation region.

    out = base*(1 - alpha*mask_bin) + heat*(alpha*mask_bin)
    where mask_bin ∈ {0,1} after thresholding.

    If threshold is None, we use mask_bin = (cam_mask > 0).
    """
    assert base_bgr.ndim == 3 and base_bgr.shape[2] == 3, "base_bgr must be HxWx3 BGR"
    H, W = base_bgr.shape[:2]

    # Ensure shape alignment
    cam = cam_mask.astype(np.float32)
    if cam.shape != (H, W):
        cam = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)

    # Build 0/1 mask from threshold
    if threshold is None:
        mask_bin = (cam > 0.0).astype(np.float32)
    else:
        thr = float(threshold)
        thr = min(max(thr, 0.0), 1.0)
        mask_bin = (cam >= thr).astype(np.float32)

    # Prepare heatmap from the (optionally thresholded) intensities
    # We keep original intensities for color, but we gate blending with mask_bin.
    cam_u8 = np.uint8(np.clip(cam, 0, 1) * 255)
    heat = cv2.applyColorMap(cam_u8, colormap).astype(np.float32)

    # Broadcast mask into 3 channels and compute alpha-per-pixel
    mask3 = mask_bin[..., None]  # (H,W,1)
    a = float(alpha)
    a = min(max(a, 0.0), 1.0)
    per_pixel_alpha = a * mask3   # (H,W,1) ∈ [0,a]

    base_f = base_bgr.astype(np.float32)
    out = base_f * (1.0 - per_pixel_alpha) + heat * per_pixel_alpha

    return np.clip(out, 0, 255).astype(np.uint8)
