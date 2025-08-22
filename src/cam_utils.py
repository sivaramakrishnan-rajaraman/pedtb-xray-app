# src/cam_utils.py
from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

# TorchCAM (CPU-friendly; no system libs needed)
from torchcam.methods import GradCAM, GradCAMpp, LayerCAM, XGradCAM, SmoothGradCAMpp

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
    """
    Prefer model.cam_target if the model exposes one (our PneumoniaModel does).
    Fallback to last Conv2d in the backbone.
    """
    if hasattr(model, "cam_target") and isinstance(model.cam_target, nn.Module):
        return model.cam_target
    if hasattr(model, "backbone"):
        last = _find_last_conv(model.backbone)
        if last is not None:
            return last
    # last resort: scan the whole model
    last = _find_last_conv(model)
    if last is None:
        raise RuntimeError("No Conv2d layer found for CAM target.")
    return last


# ---------------------------
# Build the CAM extractor
# ---------------------------
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

# ---------------------------
# Main: compute normalized CAM map
# ---------------------------
def compute_cam_map(
    model: nn.Module,
    input_tensor: torch.Tensor,        # shape (1, 3, H, W) on the correct device
    method: str = "gradcam",
    class_idx: int = 1,
    use_autocast: bool = False,
) -> np.ndarray:
    """
    Returns a float32 CAM map in [0,1] at the network's feature resolution.
    NOTE: This function *enables grad* internally. Do NOT wrap it in torch.no_grad().
    """
    if input_tensor.ndim != 4 or input_tensor.size(0) != 1:
        raise ValueError("input_tensor must be batched (1, C, H, W)")

    device = input_tensor.device
    model.eval()

    # Ensure all params can carry grads for hooks/backprop (eval mode is fine)
    for p in model.parameters():
        p.requires_grad_(True)

    target_layer = discover_target_layer(model)
    cam_extractor = build_cam_extractor(model, target_layer, method=method)

    # Forward with grad enabled (this is the key fix)
    with torch.enable_grad():
        input_tensor = input_tensor.float()
        if use_autocast and device.type == "cuda":
            with torch.cuda.amp.autocast():
                scores = model(input_tensor)
        else:
            scores = model(input_tensor)

        # torchcam API: call extractor with (class_idx, scores)
        cams = cam_extractor(class_idx, scores)
        cam = cams[0] if isinstance(cams, (list, tuple)) else cams  # (Hc, Wc) torch.Tensor

    # Normalize to [0,1]
    cam = cam.detach().cpu().float().numpy()
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    else:
        cam = np.zeros_like(cam, dtype=np.float32)

    # Clean hooks if available (defensive)
    try:
        cam_extractor.remove_hooks()
    except Exception:
        pass

    return cam.astype(np.float32)

# ---------------------------
# Rendering helpers
# ---------------------------
def heatmap_overlay(
    cam_map: np.ndarray,      # float [0,1], (Hc, Wc)
    base_bgr: np.ndarray,     # uint8 (H, W, 3) in BGR
    alpha: float = 0.5,
) -> np.ndarray:
    """Resize CAM to image, colorize, and alpha-blend."""
    H, W = base_bgr.shape[:2]
    cam_u8 = np.uint8(np.clip(cam_map, 0, 1) * 255)
    cam_u8 = cv2.resize(cam_u8, (W, H), interpolation=cv2.INTER_LINEAR)

    heat = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heat, float(alpha), base_bgr, 1.0 - float(alpha), 0.0)
    return overlay


def contours_and_boxes(
    cam_map: np.ndarray,      # float [0,1], (Hc, Wc)
    base_bgr: np.ndarray,     # uint8 (H, W, 3)
    threshold: float = 0.4,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
    line_type: int = cv2.LINE_AA,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw contours and tight bounding boxes for CAM≥threshold."""
    H, W = base_bgr.shape[:2]
    cam_u8 = np.uint8(np.clip(cam_map, 0, 1) * 255)
    cam_u8 = cv2.resize(cam_u8, (W, H), interpolation=cv2.INTER_NEAREST)

    thr = int(255 * float(threshold))
    _, binary = cv2.threshold(cam_u8, thr, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cont_img = base_bgr.copy()
    box_img  = base_bgr.copy()

    for cnt in contours:
        cv2.drawContours(cont_img, [cnt], -1, color, thickness, lineType=line_type)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(box_img, (x, y), (x + w, y + h), color, thickness, lineType=line_type)

    return cont_img, box_img
