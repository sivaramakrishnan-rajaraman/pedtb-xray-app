# src/cam_utils.py
from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.cm as cm
import torch
import torch.nn as nn

# pytorch-grad-cam
from pytorch_grad_cam import (
    GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM,
    XGradCAM, LayerCAM, FullGrad, HiResCAM, EigenCAM, EigenGradCAM
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from skimage.measure import find_contours, label, regionprops

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

def build_cam(model: nn.Module, method: str = "gradcam"):
    tl = discover_target_layer(model)
    CAMClass = _CAM_MAP.get((method or "gradcam").lower(), GradCAM)
    cam = CAMClass(model=model, target_layers=[tl], reshape_transform=None)
    cam.batch_size = 1
    return cam

# ---------------------------
# Compute CAM mask (returns HxW in [0,1], same as input spatial)
# ---------------------------
def compute_cam_mask(
    model: nn.Module,
    input_tensor: torch.Tensor,  # (1,3,H,W) normalized
    class_index: int = 1,
    method: str = "gradcam",
    use_aug_smooth: bool = True,
    use_eigen_smooth: bool = True,
) -> np.ndarray:
    cam = build_cam(model, method)
    mask = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(int(class_index))],
        aug_smooth=bool(use_aug_smooth),
        eigen_smooth=bool(use_eigen_smooth),
    )[0]  # (H,W)

    mask = np.asarray(mask, dtype=np.float32)
    mmin, mmax = float(mask.min()), float(mask.max())
    mask = (mask - mmin) / (mmax - mmin) if mmax > mmin else np.zeros_like(mask, dtype=np.float32)

    try:
        cam.activations_and_grads.release()
    except Exception:
        pass

    return mask

# ---------------------------
# Rendering helpers (RGB only)
# ---------------------------
def overlay_heatmap_on_rgb(
    base_rgb: np.ndarray,  # uint8 HxWx3
    cam_mask: np.ndarray,  # float32 HxW in [0,1]
    alpha: float = 0.5,
    cmap_name: str = "hot",
) -> np.ndarray:
    if cam_mask.shape[:2] != base_rgb.shape[:2]:
        cam_mask = np.array(Image.fromarray((cam_mask * 255).astype(np.uint8)).resize(
            (base_rgb.shape[1], base_rgb.shape[0]), Image.BILINEAR
        )) / 255.0

    cmap = cm.get_cmap(cmap_name)
    heat_rgba = cmap(cam_mask)  # HxWx4 float in [0,1]
    heat_rgb  = (heat_rgba[..., :3] * 255).astype(np.uint8)

    blended = (alpha * heat_rgb + (1 - alpha) * base_rgb).clip(0, 255).astype(np.uint8)
    return blended

def contours_and_boxes_on_rgb(
    base_rgb: np.ndarray,
    cam_mask: np.ndarray,
    threshold: float = 0.4,
    line_color: Tuple[int,int,int] = (255, 0, 0),
    box_color: Tuple[int,int,int]  = (255, 0, 0),
    thickness: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    if cam_mask.shape[:2] != base_rgb.shape[:2]:
        cam_mask = np.array(Image.fromarray((cam_mask * 255).astype(np.uint8)).resize(
            (base_rgb.shape[1], base_rgb.shape[0]), Image.NEAREST
        )) / 255.0

    H, W = base_rgb.shape[:2]
    binary = (cam_mask >= float(threshold)).astype(np.uint8)

    # Contours
    cont_img = Image.fromarray(base_rgb.copy())
    draw_c = ImageDraw.Draw(cont_img)
    for cnt in find_contours(binary, level=0.5):
        # cnt is (N, 2) with (row, col)
        poly = [(float(xy[1]), float(xy[0])) for xy in cnt]
        if len(poly) >= 2:
            draw_c.line(poly, fill=line_color, width=thickness)

    # Bounding boxes via connected components
    box_img = Image.fromarray(base_rgb.copy())
    draw_b = ImageDraw.Draw(box_img)
    lab = label(binary, connectivity=1)
    for rp in regionprops(lab):
        minr, minc, maxr, maxc = rp.bbox  # rows, cols
        draw_b.rectangle([(minc, minr), (maxc, maxr)], outline=box_color, width=thickness)

    return np.array(cont_img), np.array(box_img)
