# src/cam_utils.py
from __future__ import annotations
from typing import Optional, Dict, Type, List, Tuple
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

# ======================================================================
# Public mappings so the app can decide how to color each method
# ======================================================================

# Which family to use for each CAM method: 'mpl' (matplotlib) or 'cv2'
METHOD_TO_CMAP_KIND: Dict[str, str] = {
    # matplotlib palettes (red/blue/green/orange/purple)
    "gradcam":       "mpl",
    "gradcam++":     "mpl",
    "scorecam":      "mpl",
    "ablationcam":   "mpl",
    "xgradcam":      "mpl",

    # OpenCV colormaps (hot/bone/inferno/magma/pink)
    # (User asked for "afm_hot" and "pnk"—OpenCV does not have AFMHOT,
    # so we map to the closest available: HOT. "pnk" -> PINK.)
    "layercam":      "cv2",
    "fullgrad":      "cv2",
    "eigencam":      "cv2",
    "eigengradcam":  "cv2",
    "hirescam":      "cv2",
}

# Exact matplotlib palette names for 'mpl' methods
METHOD_TO_MPL_NAME: Dict[str, str] = {
    "gradcam":       "Reds",
    "gradcam++":     "Blues",
    "scorecam":      "Greens",
    "ablationcam":   "Oranges",
    "xgradcam":      "Purples",
}

# Exact OpenCV colormap constants for 'cv2' methods
METHOD_TO_CV2_CMAP: Dict[str, int] = {
    # "afm_hot" requested by user → fallback to HOT in OpenCV
    "layercam":      cv2.COLORMAP_HOT,
    "fullgrad":      cv2.COLORMAP_BONE,
    "eigencam":      cv2.COLORMAP_INFERNO,
    "eigengradcam":  cv2.COLORMAP_MAGMA,
    "hirescam":      cv2.COLORMAP_PINK,  # "pnk" → PINK
}

# ======================================================================
# Internal helpers
# ======================================================================

def _get_cam_class(method: str):
    """Return the CAM class for a given method string."""
    CAM_CLASSES: Dict[str, Type] = {
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
    key = (method or "gradcam").strip().lower()
    if key not in CAM_CLASSES:
        raise ValueError(f"Unknown CAM method '{method}'. "
                         f"Supported: {', '.join(sorted(CAM_CLASSES.keys()))}")
    return CAM_CLASSES[key]

def _find_last_conv(module: nn.Module) -> Optional[nn.Conv2d]:
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last

def _resolve_target_layer(model: nn.Module) -> nn.Module:
    """
    Priority:
      1) model.cam_target (your TB model exposes post3x3 conv here)
      2) last Conv2d inside model.post3x3
      3) last Conv2d in model.backbone
      4) last Conv2d anywhere
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

# ======================================================================
# Core CAM computation
# ======================================================================

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

    # Some methods ignore targets (e.g., EigenCAM); handle gracefully
    targets = None
    if CAMClass is not EigenCAM:
        targets = [ClassifierOutputTarget(int(class_index))]

    cam_obj = CAMClass(
        model=model,
        target_layers=[target_layer],
        use_cuda=(input_tensor.device.type == "cuda")
    )
    cam_obj.batch_size = 1

    with torch.enable_grad():
        masks: List[np.ndarray] = cam_obj(
            input_tensor=input_tensor.float(),
            targets=targets,
            aug_smooth=bool(aug_smooth),
            eigen_smooth=bool(eigen_smooth),
        )  # list[np.ndarray] of shape (H,W)
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

# ======================================================================
# Coloring (matplotlib or OpenCV) + thresholded, mask-aware overlay
# ======================================================================

def _apply_mpl_colormap(cam_u8: np.ndarray, cmap_name: str) -> np.ndarray:
    """
    Return BGR heatmap (uint8) using a matplotlib colormap name.
    Falls back to OpenCV JET if matplotlib is unavailable.
    """
    try:
        # Lazy import to avoid hard dependency if not needed
        import matplotlib.cm as cm
        import numpy as np
        cmap = cm.get_cmap(cmap_name)
        # cam_u8 -> [0,1] -> RGBA -> RGB
        cam01 = cam_u8.astype(np.float32) / 255.0
        rgb = (cmap(cam01)[..., :3] * 255.0).astype(np.uint8)  # HxWx3 RGB
        bgr = rgb[..., ::-1]  # RGB->BGR
        return bgr
    except Exception:
        # Fallback: OpenCV JET
        return cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)

def _apply_cv2_colormap(cam_u8: np.ndarray, cv2_cmap: int) -> np.ndarray:
    return cv2.applyColorMap(cam_u8, cv2_cmap)

def get_method_colormap(method: str) -> Tuple[str, str | int]:
    """
    Returns (kind, spec)
      - kind = 'mpl' or 'cv2'
      - spec = mpl name (str) or cv2 constant (int)
    """
    m = (method or "gradcam").strip().lower()
    kind = METHOD_TO_CMAP_KIND.get(m, "cv2")
    if kind == "mpl":
        name = METHOD_TO_MPL_NAME.get(m, "Reds")
        return ("mpl", name)
    else:
        cv2_cmap = METHOD_TO_CV2_CMAP.get(m, cv2.COLORMAP_HOT)
        return ("cv2", cv2_cmap)

def overlay_heatmap_on_bgr(
    base_bgr: np.ndarray,          # uint8 (H,W,3) BGR
    cam_mask: np.ndarray,          # float32 (h,w) in [0,1]
    alpha: float = 0.5,            # global max opacity in [0,1]
    colormap: Optional[int] = None,
    threshold: Optional[float] = 0.3, #activations>threshold are shown, otherwise, None
    method: Optional[str] = None,  # if given, overrides 'colormap' via per-method palette
) -> np.ndarray:
    """
    Thresholded, mask-aware overlay:
      out = base*(1 - alpha*mask_bin) + heat*(alpha*mask_bin),
      where mask_bin ∈ {0,1} after thresholding.
    If `method` is provided, we choose the palette by method name
    (using matplotlib or OpenCV as defined above). Otherwise, use the
    provided OpenCV `colormap` constant (default HOT).
    """
    assert base_bgr.ndim == 3 and base_bgr.shape[2] == 3, "base_bgr must be HxWx3 BGR"
    H, W = base_bgr.shape[:2]

    cam = cam_mask.astype(np.float32)
    if cam.shape != (H, W):
        cam = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)

    # Binary gate from threshold
    if threshold is None:
        mask_bin = (cam > 0.0).astype(np.float32)
    else:
        thr = float(min(max(threshold, 0.0), 1.0))
        mask_bin = (cam >= thr).astype(np.float32)

    cam_u8 = np.uint8(np.clip(cam, 0, 1) * 255)

    # Choose heatmap coloring
    if method is not None:
        kind, spec = get_method_colormap(method)
        if kind == "mpl":
            heat = _apply_mpl_colormap(cam_u8, str(spec)).astype(np.float32)
        else:
            heat = _apply_cv2_colormap(cam_u8, int(spec)).astype(np.float32)
    else:
        # Back-compat: explicit OpenCV colormap or HOT
        cv2_cmap = cv2.COLORMAP_HOT if colormap is None else int(colormap)
        heat = _apply_cv2_colormap(cam_u8, cv2_cmap).astype(np.float32)

    # Mask-aware alpha blend (no dimming outside activation)
    mask3 = mask_bin[..., None]     # (H,W,1)
    a = float(min(max(alpha, 0.0), 1.0))
    per_pixel_alpha = a * mask3

    base_f = base_bgr.astype(np.float32)
    out = base_f * (1.0 - per_pixel_alpha) + heat * per_pixel_alpha
    return np.clip(out, 0, 255).astype(np.uint8)
