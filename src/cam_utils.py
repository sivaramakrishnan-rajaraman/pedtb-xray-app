# src/cam_utils.py
from __future__ import annotations
from typing import Optional, Dict, Tuple, List, Type
import numpy as np
import torch
import torch.nn as nn
import cv2

# pytorch-grad-cam (Jacobgil)
from pytorch_grad_cam import (
    GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM,
    XGradCAM, LayerCAM, FullGrad, EigenCAM, EigenGradCAM, HiResCAM
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# =============================================================================
# Public mapping: method -> (kind, spec)
#   kind: 'mpl' or 'cv2'
#   spec: matplotlib colormap name (str) when kind=='mpl'
#         OpenCV colormap constant (int) when kind=='cv2'
# =============================================================================
METHOD_TO_CMAP_KIND: Dict[str, Tuple[str, object]] = {
    # Matplotlib (red, blue, green, orange, purple)
    "gradcam":       ("mpl", "Reds"),
    "gradcam++":     ("mpl", "Blues"),
    "scorecam":      ("mpl", "Greens"),
    "ablationcam":   ("mpl", "Oranges"),
    "xgradcam":      ("mpl", "Purples"),

    # OpenCV (hot, bone, inferno, magma, pink)
    "layercam":      ("cv2", cv2.COLORMAP_HOT),      # requested "afm_hot" → HOT
    "fullgrad":      ("cv2", cv2.COLORMAP_BONE),
    "eigencam":      ("cv2", cv2.COLORMAP_INFERNO),
    "eigengradcam":  ("cv2", cv2.COLORMAP_MAGMA),
    "hirescam":      ("cv2", cv2.COLORMAP_PINK),     # requested "pnk" → PINK
}


# =============================================================================
# Internals: model/targets and method dispatch
# =============================================================================
def _get_cam_class(method: str) -> Type:
    mapping: Dict[str, Type] = {
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
    if key not in mapping:
        raise ValueError(f"Unknown CAM method '{method}'. "
                         f"Supported: {', '.join(sorted(mapping.keys()))}")
    return mapping[key]


def _find_last_conv(module: nn.Module) -> Optional[nn.Conv2d]:
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


def _resolve_target_layer(model: nn.Module) -> nn.Module:
    """
    Priority:
      1) model.cam_target (your TB model exposes the post3x3 conv here)
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


# =============================================================================
# Core CAM computation (normalized [0,1] at input resolution)
# =============================================================================
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

    # ensure params carry grad for hooks/backprop
    for p in model.parameters():
        p.requires_grad_(True)

    target_layer = _resolve_target_layer(model)
    CAMClass = _get_cam_class(method)

    # EigenCAM ignores explicit targets
    targets = None if CAMClass is EigenCAM else [ClassifierOutputTarget(int(class_index))]

    cam = CAMClass(model=model, target_layers=[target_layer])
    cam.batch_size = 1

    with torch.enable_grad():
        masks: List[np.ndarray] = cam(
            input_tensor=input_tensor.float(),
            targets=targets,
            aug_smooth=bool(aug_smooth),
            eigen_smooth=bool(eigen_smooth),
        )
    m = masks[0].astype(np.float32)

    # normalize robustly to [0,1]
    mn, mx = float(m.min()), float(m.max())
    if mx > mn:
        m = (m - mn) / (mx - mn)
    else:
        m = np.zeros_like(m, dtype=np.float32)

    # cleanup
    try:
        cam.activations_and_grads.release()
    except Exception:
        pass

    return m  # (H,W) float32 in [0,1]


# =============================================================================
# Coloring + thresholded, mask-aware overlay
# =============================================================================
def _apply_mpl_colormap(cam_u8: np.ndarray, cmap_name: str) -> np.ndarray:
    """
    Returns BGR heatmap (uint8) using a matplotlib colormap name.
    Falls back to OpenCV JET if matplotlib isn't available.
    """
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(cmap_name)
        cam01 = cam_u8.astype(np.float32) / 255.0
        rgb = (cmap(cam01)[..., :3] * 255.0).astype(np.uint8)  # HxWx3 RGB
        return rgb[..., ::-1]  # RGB->BGR
    except Exception:
        return cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)


def _apply_cv2_colormap(cam_u8: np.ndarray, cv2_cmap: int) -> np.ndarray:
    return cv2.applyColorMap(cam_u8, cv2_cmap)


def overlay_heatmap_on_bgr(
    base_bgr: np.ndarray,          # uint8 (H,W,3) BGR
    cam_mask: np.ndarray,          # float32 (h,w) in [0,1]
    alpha: float = 0.5,            # max opacity in [0,1]
    threshold: Optional[float] = None,
    cmap_kind: Optional[Tuple[str, object]] = None,
) -> np.ndarray:
    """
    Thresholded, mask-aware overlay:
      out = base*(1 - a*mask_bin) + heat*(a*mask_bin),
      where mask_bin ∈ {0,1} after thresholding.

    cmap_kind:
      ('mpl', 'Reds'|'Blues'|...)  -> matplotlib colormap name
      ('cv2', cv2.COLORMAP_*)      -> OpenCV colormap constant
      None                         -> defaults to cv2 HOT
    """
    assert base_bgr.ndim == 3 and base_bgr.shape[2] == 3, "base_bgr must be HxWx3 BGR"

    H, W = base_bgr.shape[:2]
    cam = cam_mask.astype(np.float32)
    if cam.shape != (H, W):
        cam = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)

    # threshold gate
    if threshold is None:
        mask_bin = (cam > 0.0).astype(np.float32)
    else:
        thr = float(min(max(threshold, 0.0), 1.0))
        mask_bin = (cam >= thr).astype(np.float32)

    cam_u8 = np.uint8(np.clip(cam, 0, 1) * 255)

    # choose colormap family
    heat: np.ndarray
    if isinstance(cmap_kind, tuple) and len(cmap_kind) == 2:
        kind, spec = cmap_kind[0], cmap_kind[1]
        if kind == "mpl":
            heat = _apply_mpl_colormap(cam_u8, str(spec)).astype(np.float32)
        else:
            heat = _apply_cv2_colormap(cam_u8, int(spec)).astype(np.float32)
    else:
        # default: cv2 HOT
        heat = _apply_cv2_colormap(cam_u8, cv2.COLORMAP_HOT).astype(np.float32)

    # mask-aware alpha blend
    a = float(min(max(alpha, 0.0), 1.0))
    mask3 = mask_bin[..., None]  # (H,W,1)
    base_f = base_bgr.astype(np.float32)
    out = base_f * (1.0 - a * mask3) + heat * (a * mask3)
    return np.clip(out, 0, 255).astype(np.uint8)
