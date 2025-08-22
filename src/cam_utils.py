# src/cam_utils.py
from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.cm as cm
import torch
import torch.nn as nn

# pytorch-grad-cam (installed via git+ URL in requirements.txt)
from pytorch_grad_cam import (
    GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM,
    XGradCAM, LayerCAM, FullGrad, HiResCAM, EigenCAM, EigenGradCAM
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from skimage.measure import find_contours, label, regionprops


# ---------------------------
# Utilities: find a good target layer
# ---------------------------
def _find_last_conv(module: nn.Module) -> Optional[nn.Conv2d]:
    last = None
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last

def discover_target_layer(model: nn.Module) -> nn.Module:
    """Prefer model.cam_target; else fallback to last Conv2d in backbone or model."""
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


# ---------------------------
# Build the CAM object
# ---------------------------
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
    tl = discover_target_layer(model)
    CAMClass = _CAM_MAP.get((method or "gradcam").lower(), GradCAM)
    cam = CAMClass(model=model, target_layers=[tl], reshape_transform=None)
    cam.batch_size = 1
    return cam


# ---------------------------
# Main: compute normalized CAM mask
# ---------------------------
def compute_cam_mask(
    model: nn.Module,
    input_tensor: torch.Tensor,  # (1,3,H,W) normalized, on same device as model
    class_index: int = 1,
    method: str = "gradcam",
    use_aug_smooth: bool = True,
    use_eigen_smooth: bool = True,
) -> np.ndarray:
    """
    Returns a float32 CAM mask in [0,1] with the same spatial size as the model's input.
    IMPORTANT: Do NOT wrap the caller in torch.no_grad(); this function needs grad.
    """
    if input_tensor.ndim != 4 or input_tensor.size(0) != 1:
        raise ValueError("input_tensor must be shaped (1, C, H, W)")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    cam = _build_cam(model, method)
    mask = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(int(class_index))],
        aug_smooth=bool(use_aug_smooth),
        eigen_smooth=bool(use_eigen_smooth),
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
# Rendering helpers (RGB only; no OpenCV)
# ---------------------------
def overlay_heatmap_on_rgb(
    base_rgb: np.ndarray,   # uint8 HxWx3
    cam_mask: np.ndarray,   # float32 HxW in [0,1]
    alpha: float = 0.5,
    cmap_name: str = "hot",
) -> np.ndarray:
    """Resize CAM to image if needed, colorize via matplotlib, and alpha-blend."""
    if cam_mask.shape[:2] != base_rgb.shape[:2]:
        cam_mask = np.array(
            Image.fromarray((cam_mask * 255).astype(np.uint8)).resize(
                (base_rgb.shape[1], base_rgb.shape[0]), Image.BILINEAR
            )
        ) / 255.0

    cmap = cm.get_cmap(cmap_name)
    heat_rgba = cmap(cam_mask)            # HxWx4 float in [0,1]
    heat_rgb  = (heat_rgba[..., :3] * 255).astype(np.uint8)

    blended = (alpha * heat_rgb + (1 - alpha) * base_rgb).clip(0, 255).astype(np.uint8)
    return blended


def contours_and_boxes_on_rgb(
    base_rgb: np.ndarray,   # uint8 HxWx3
    cam_mask: np.ndarray,   # float32 HxW in [0,1]
    threshold: float = 0.4,
    line_color: Tuple[int,int,int] = (255, 0, 0),
    box_color: Tuple[int,int,int]  = (255, 0, 0),
    thickness: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draws contours (skimage.find_contours) and connected-component bounding boxes
    onto copies of base_rgb. Returns (contour_image, bbox_image) as uint8 arrays.
    """
    if cam_mask.shape[:2] != base_rgb.shape[:2]:
        cam_mask = np.array(
            Image.fromarray((cam_mask * 255).astype(np.uint8)).resize(
                (base_rgb.shape[1], base_rgb.shape[0]), Image.NEAREST
            )
        ) / 255.0

    binary = (cam_mask >= float(threshold)).astype(np.uint8)

    # Contours
    cont_img = Image.fromarray(base_rgb.copy())
    draw_c = ImageDraw.Draw(cont_img)
    for cnt in find_contours(binary, level=0.5):
        # cnt is (N, 2) array of (row, col) floats
        poly = [(float(xy[1]), float(xy[0])) for xy in cnt]
        if len(poly) >= 2:
            draw_c.line(poly, fill=tuple(line_color), width=int(thickness))

    # Bounding boxes via connected components
    box_img = Image.fromarray(base_rgb.copy())
    draw_b = ImageDraw.Draw(box_img)
    lab = label(binary, connectivity=1)
    for rp in regionprops(lab):
        minr, minc, maxr, maxc = rp.bbox  # rows, cols
        draw_b.rectangle([(minc, minr), (maxc, maxr)], outline=tuple(box_color), width=int(thickness))

    return np.array(cont_img), np.array(box_img)


# ---------------------------
# Backward-compat aliases (so old imports won't break)
# ---------------------------
# Older code sometimes imported these names:
def compute_cam_map(*args, **kwargs):
    """Alias for compute_cam_mask (backward compatibility)."""
    return compute_cam_mask(*args, **kwargs)

def heatmap_overlay(cam_map: np.ndarray, base_bgr_or_rgb: np.ndarray, alpha: float = 0.5):
    """
    Alias for overlay_heatmap_on_rgb; accepts either RGB or BGR but treats as RGB.
    (We dropped OpenCV, so everything is RGB now.)
    """
    # If someone passes BGR by mistake, colors will just be swapped visually;
    # we keep this alias to avoid import errors.
    return overlay_heatmap_on_rgb(base_bgr_or_rgb, cam_map, alpha=alpha, cmap_name="hot")

def contours_and_boxes(*args, **kwargs):
    """Alias for contours_and_boxes_on_rgb."""
    return contours_and_boxes_on_rgb(*args, **kwargs)


# What this module exports
__all__ = [
    "discover_target_layer",
    "compute_cam_mask",
    "overlay_heatmap_on_rgb",
    "contours_and_boxes_on_rgb",
    # aliases
    "compute_cam_map",
    "heatmap_overlay",
    "contours_and_boxes",
]
