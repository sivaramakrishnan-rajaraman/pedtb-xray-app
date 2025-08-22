# src/cam_utils.py

from __future__ import annotations
from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcam.methods import GradCAM, SmoothGradCAMpp
from skimage.measure import label, regionprops
from PIL import Image, ImageDraw
import matplotlib.cm as cm

__all__ = [
    "discover_target_layer",
    "build_cam_extractor",
    "compute_cam_map",
    "heatmap_overlay",
    "contours_and_bboxes",
    "draw_bboxes",
]

def discover_target_layer(model: nn.Module) -> nn.Module:
    # Prefer explicit tap
    if hasattr(model, "cam_target") and isinstance(model.cam_target, nn.Module):
        return model.cam_target
    # Then the 3x3 we added
    if hasattr(model, "post3x3") and isinstance(model.post3x3, nn.Conv2d):
        return model.post3x3
    # Else last conv in backbone
    if hasattr(model, "backbone"):
        last = None
        for _, m in model.backbone.named_modules():
            if isinstance(m, nn.Conv2d):
                last = m
        if last is not None:
            return last
    # Else last conv anywhere
    last = None
    for _, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d layer found for CAM.")
    return last

def build_cam_extractor(model: nn.Module, target_layer: nn.Module, method: str = "gradcam"):
    m = method.lower().strip()
    if m in ("gradcam", "grad-cam"):
        return GradCAM(model, target_layer=target_layer)
    elif m in ("gradcam++", "smoothgradcampp", "smooth-grad-cam++"):
        return SmoothGradCAMpp(model, target_layer=target_layer)
    else:
        raise ValueError(f"Unsupported CAM method '{method}'. Use 'gradcam' or 'gradcam++'.")

@torch.no_grad()
def compute_cam_map(
    model: nn.Module,
    input_tensor: torch.Tensor,     # (1,3,H,W) normalized
    method: str = "gradcam",
    class_idx: int = 1,
    target_layer: Optional[nn.Module] = None,
    upsample_to: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    model.eval()
    if target_layer is None:
        target_layer = discover_target_layer(model)
    cam_extractor = build_cam_extractor(model, target_layer, method=method)

    scores = model(input_tensor.float())
    cams = cam_extractor(class_idx, scores)   # returns list[tensor] or tensor
    cam = cams[0] if isinstance(cams, (list, tuple)) else cams
    cam = cam - cam.min()
    cam = cam / cam.max().clamp(min=1e-8)

    if upsample_to is not None:
        H0, W0 = int(upsample_to[0]), int(upsample_to[1])
        cam = cam.unsqueeze(0).unsqueeze(0)  # 1x1xhxw
        cam = F.interpolate(cam, size=(H0, W0), mode="bilinear", align_corners=False)
        cam = cam.squeeze(0).squeeze(0)
    return cam.cpu().float().numpy()

def heatmap_overlay(
    rgb: Image.Image,
    cam01: np.ndarray,
    alpha: float = 0.5,
    cmap_name: str = "jet"
) -> Image.Image:
    """Blend a [0,1] CAM with an RGB PIL image."""
    cmap = cm.get_cmap(cmap_name)
    heat = (cmap(cam01)[:, :, :3] * 255).astype(np.uint8)  # HxWx3
    heat_img = Image.fromarray(heat).resize(rgb.size, Image.BILINEAR)
    return Image.blend(rgb.convert("RGB"), heat_img, alpha=alpha)

def contours_and_bboxes(cam01: np.ndarray, thr: float = 0.4) -> List[Tuple[int,int,int,int]]:
    """Axis-aligned bboxes from thresholded CAM (uses skimage)."""
    mask = (cam01 >= float(thr)).astype(np.uint8)
    lab = label(mask, connectivity=1)
    boxes = []
    for r in regionprops(lab):
        minr, minc, maxr, maxc = r.bbox
        boxes.append((int(minc), int(minr), int(maxc), int(maxr)))  # x1,y1,x2,y2
    return boxes

def draw_bboxes(
    rgb: Image.Image,
    boxes: List[Tuple[int,int,int,int]],
    color: Tuple[int,int,int]=(255,0,0),
    width: int = 3
) -> Image.Image:
    out = rgb.copy()
    dr = ImageDraw.Draw(out)
    for (x1,y1,x2,y2) in boxes:
        dr.rectangle([x1,y1,x2,y2], outline=color, width=width)
    return out
