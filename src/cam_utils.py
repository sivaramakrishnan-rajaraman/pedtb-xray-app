# src/cam_utils.py
from __future__ import annotations
from typing import Tuple, Optional, List, Union

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F

# TorchCAM (CPU/GPU friendly, no system libs)
from torchcam.methods import GradCAM, GradCAMpp, LayerCAM, XGradCAM, SmoothGradCAMpp


# -----------------------------------------------------------------------------
# Utilities (no OpenCV)
# -----------------------------------------------------------------------------
ImgLike = Union[Image.Image, np.ndarray]


def _to_rgb_np(img: ImgLike) -> np.ndarray:
    """
    Convert a PIL image or numpy array to uint8 RGB numpy array (H, W, 3).
    - If np.ndarray and shape is (H,W), it will be repeated to 3 channels.
    - If np.ndarray and shape is (H,W,3/4), it will be converted to RGB.
    """
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("RGB"), dtype=np.uint8)

    arr = np.asarray(img)
    if arr.ndim == 2:  # grayscale -> RGB
        arr = np.repeat(arr[..., None], 3, axis=-1)
    elif arr.ndim == 3:
        if arr.shape[2] == 4:  # RGBA -> RGB
            arr = arr[..., :3]
        elif arr.shape[2] == 3:
            pass
        else:
            raise ValueError(f"Unsupported channel count: {arr.shape}")
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    if arr.dtype != np.uint8:
        # assume 0..1 float or arbitrary; clip to [0,255]
        arr = np.clip(arr, 0, 255)
        if arr.max() <= 1.0:
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return arr


def _resize_mask(mask01: np.ndarray, size_hw: Tuple[int, int], mode: str = "bilinear") -> np.ndarray:
    """
    Resize a float [0,1] mask (H, W) to (H', W') using PIL.
    mode: "bilinear" or "nearest"
    """
    Ht, Wt = size_hw
    m = np.clip(mask01, 0.0, 1.0)
    pil = Image.fromarray((m * 255.0).astype(np.uint8), mode="L")
    pil = pil.resize((Wt, Ht), Image.BILINEAR if mode == "bilinear" else Image.NEAREST)
    out = np.asarray(pil).astype(np.float32) / 255.0
    return out


def _jet_colormap(x: np.ndarray) -> np.ndarray:
    """
    Minimal JET colormap.
    x: float32 array in [0,1] (H,W) -> uint8 RGB (H,W,3)
    """
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def _edges_from_mask(mask01: np.ndarray, thr: float = 0.4) -> np.ndarray:
    """
    Return boolean edge map (H,W) from a thresholded mask (no OpenCV).
    A pixel is an edge if it's positive and at least one 8-neighbor is negative.
    """
    m = (mask01 >= thr)
    if not m.any():
        return np.zeros_like(m, dtype=bool)

    nbrs = [
        np.roll(m, 1, 0), np.roll(m, -1, 0),
        np.roll(m, 1, 1), np.roll(m, -1, 1),
        np.roll(np.roll(m, 1, 0), 1, 1),
        np.roll(np.roll(m, 1, 0), -1, 1),
        np.roll(np.roll(m, -1, 0), 1, 1),
        np.roll(np.roll(m, -1, 0), -1, 1),
    ]
    # A crude edge: any neighbor differs
    all_same = nbrs[0]
    for k in range(1, len(nbrs)):
        all_same = all_same & nbrs[k]
    edges = m & (~all_same)
    return edges


# -----------------------------------------------------------------------------
# Target layer discovery (matches PneumoniaModel behavior)
# -----------------------------------------------------------------------------
def _find_last_conv(module: nn.Module) -> Optional[nn.Conv2d]:
    last = None
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


def discover_target_layer(model: nn.Module) -> nn.Module:
    """
    Prefer model.cam_target if present, else last Conv2d in backbone, else last Conv2d in whole model.
    """
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


# -----------------------------------------------------------------------------
# CAM extractors
# -----------------------------------------------------------------------------
def build_cam_extractor(model: nn.Module, target_layer: nn.Module, method: str = "gradcam"):
    method = (method or "gradcam").lower()
    if method in ("gradcam", "gc"):
        return GradCAM(model, target_layer)
    if method in ("gradcam++", "gcpp", "++"):
        return GradCAMpp(model, target_layer)
    if method in ("layercam", "lc"):
        return LayerCAM(model, target_layer)
    if method in ("xgradcam", "xgc"):
        return XGradCAM(model, target_layer)
    if method in ("smoothgradcampp", "sgcpp", "smooth++"):
        return SmoothGradCAMpp(model, target_layer)
    # default
    return GradCAM(model, target_layer)


# -----------------------------------------------------------------------------
# Main: compute normalized CAM map (no OpenCV; gradients enabled)
# -----------------------------------------------------------------------------
def compute_cam_map(
    model: nn.Module,
    input_tensor: torch.Tensor,        # shape (1, 3, H, W) on the correct device
    method: str = "gradcam",
    class_idx: int = 1,
    use_autocast: bool = False,
) -> np.ndarray:
    """
    Returns a float32 CAM map in [0,1] at THE INPUT spatial resolution (H,W).
    NOTE: This function *enables grad* internally. Do NOT wrap it in torch.no_grad().
    """
    if input_tensor.ndim != 4 or input_tensor.size(0) != 1:
        raise ValueError("input_tensor must be batched (1, C, H, W)")

    device = input_tensor.device
    model.eval()

    # Ensure params can carry grads for hooks/backprop
    for p in model.parameters():
        p.requires_grad_(True)

    target_layer = discover_target_layer(model)
    cam_extractor = build_cam_extractor(model, target_layer, method=method)

    # Forward with grad enabled
    with torch.enable_grad():
        x = input_tensor.float()
        if use_autocast and device.type == "cuda":
            with torch.cuda.amp.autocast():
                scores = model(x)
        else:
            scores = model(x)

        cams = cam_extractor(class_idx, scores)  # torchcam API
        cam = cams[0] if isinstance(cams, (list, tuple)) else cams  # (h, w) tensor

    # Upsample to input spatial size
    cam = cam.unsqueeze(0).unsqueeze(0)  # (1,1,h,w)
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam_np = cam.detach().cpu().float().numpy()

    # Normalize to [0,1]
    cam_np = cam_np - cam_np.min()
    if cam_np.max() > 0:
        cam_np = cam_np / cam_np.max()
    else:
        cam_np[:] = 0.0

    # Clean hooks if available
    try:
        cam_extractor.remove_hooks()
    except Exception:
        pass

    return cam_np.astype(np.float32)


# -----------------------------------------------------------------------------
# Rendering helpers (no OpenCV) — return PIL images
# -----------------------------------------------------------------------------
def heatmap_overlay(
    cam_map: np.ndarray,        # float [0,1], (Hc, Wc) or already (H, W)
    base_image: ImgLike,        # PIL.Image or np.ndarray (uint8)
    alpha: float = 0.5,
) -> Image.Image:
    """
    Resize CAM to image size, colorize with a small JET, and alpha-blend over the RGB image.
    Returns: PIL.Image (RGB)
    """
    base_rgb = _to_rgb_np(base_image)
    H, W = base_rgb.shape[:2]
    if cam_map.shape != (H, W):
        cam_map = _resize_mask(cam_map, (H, W), mode="bilinear")

    heat = _jet_colormap(cam_map)           # (H,W,3) uint8
    out = (alpha * heat + (1.0 - alpha) * base_rgb).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out)


def contours_and_boxes(
    cam_map: np.ndarray,        # float [0,1], (Hc, Wc) or (H, W)
    base_image: ImgLike,        # PIL.Image or np.ndarray
    threshold: float = 0.4,
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> Tuple[Image.Image, Image.Image]:
    """
    Draw:
      - contour image: 1px (approx) edges overlaid
      - box image: a single tight bounding box around all positive pixels
    Both returned as PIL.Image (RGB).
    """
    base_rgb = _to_rgb_np(base_image)
    H, W = base_rgb.shape[:2]
    if cam_map.shape != (H, W):
        cam_map = _resize_mask(cam_map, (H, W), mode="nearest")

    # Contours via edge mask
    edges = _edges_from_mask(cam_map, thr=threshold)
    cont_arr = base_rgb.copy()
    ys, xs = np.where(edges)
    if thickness <= 1:
        cont_arr[ys, xs] = np.array(color, dtype=np.uint8)
    else:
        # thicken by painting neighbors within radius
        r = max(1, thickness // 2)
        for y, x in zip(ys.tolist(), xs.tolist()):
            y0, y1 = max(0, y - r), min(H, y + r + 1)
            x0, x1 = max(0, x - r), min(W, x + r + 1)
            cont_arr[y0:y1, x0:x1] = np.array(color, dtype=np.uint8)
    cont_img = Image.fromarray(cont_arr)

    # Bounding box from all positive pixels
    m = (cam_map >= threshold)
    box_img = Image.fromarray(base_rgb.copy())
    if m.any():
        yy, xx = np.where(m)
        y1, y2 = int(yy.min()), int(yy.max())
        x1, x2 = int(xx.min()), int(xx.max())
        draw = ImageDraw.Draw(box_img)
        # emulate thickness by drawing multiple rectangles
        for t in range(thickness):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color, width=1)

    return cont_img, box_img
