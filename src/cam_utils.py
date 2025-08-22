from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import cv2
import torch
import torch.nn as nn
from pytorch_grad_cam import (
    GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM,
    XGradCAM, LayerCAM, FullGrad, HiResCAM, EigenCAM, EigenGradCAM
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

CAM_METHODS = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "xgradcam": XGradCAM,
    "layercam": LayerCAM,
    "scorecam": ScoreCAM,
    "ablationcam": AblationCAM,
    "fullgrad": FullGrad,
    "eigencam": EigenCAM,
    "eigengradcam": EigenGradCAM,
    "hirescam": HiResCAM,
}

def get_target_layer(model: nn.Module) -> nn.Module:
    # Prefer model.cam_target if present.
    if hasattr(model, "cam_target") and isinstance(model.cam_target, nn.Module):
        return model.cam_target
    # Fallback: last Conv2d
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d layer found for CAM target.")
    return last

def cam_mask(
    model: nn.Module,
    image_tensor: torch.Tensor,            # 1x3xHxW (normalized)
    method: str = "gradcam",
    class_idx: int = 1,
    aug_smooth: bool = True,
    eigen_smooth: bool = True,
) -> np.ndarray:
    """
    Returns a float32 mask in [0,1] at the model's last feature map resolution.
    """
    target_layer = get_target_layer(model)
    CAMClass = CAM_METHODS[method.lower()]
    with CAMClass(model=model, target_layers=[target_layer], reshape_transform=None) as cam:
        if method.lower() == "eigencam":
            m = cam(input_tensor=image_tensor)[0]
        else:
            m = cam(
                input_tensor=image_tensor,
                targets=[ClassifierOutputTarget(class_idx)],
                aug_smooth=aug_smooth,
                eigen_smooth=eigen_smooth
            )[0]
    m = np.asarray(m, dtype=np.float32)
    # normalize
    vmin, vmax = float(m.min()), float(m.max())
    if vmax > vmin:
        m = (m - vmin) / (vmax - vmin)
    else:
        m[:] = 0.0
    return m

def overlay_and_shapes(
    orig_bgr: np.ndarray,
    mask: np.ndarray,         # [H',W'] or resized to [H,W] already
    alpha: float = 0.5,
    threshold: float = 0.4,
    contour_color=(0,0,255),
    contour_thickness=2,
    line_type=cv2.LINE_AA
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (heat_overlay_bgr, contours_bgr, bboxes_bgr) at original resolution.
    """
    H, W = orig_bgr.shape[:2]
    if mask.shape[:2] != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    mask_u8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(mask_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heat, float(alpha), orig_bgr, 1.0-float(alpha), 0.0)

    thr = int(255 * float(threshold))
    _, binary = cv2.threshold(mask_u8, thr, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cont_img = orig_bgr.copy()
    if contours:
        cv2.drawContours(cont_img, contours, -1, contour_color, contour_thickness, lineType=line_type)

    box_img = orig_bgr.copy()
    for cnt in contours or []:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(box_img, (x, y), (x+w, y+h), contour_color, contour_thickness, lineType=line_type)

    return overlay, cont_img, box_img
