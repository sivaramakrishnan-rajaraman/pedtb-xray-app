# src/cam_utils.py
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# TorchCAM backends
from torchcam.methods import GradCAM, SmoothGradCAMpp


def discover_target_layer(model: nn.Module) -> nn.Module:
    """
    Pick a good CAM target layer from our PneumoniaModel:
      1) model.cam_target if present
      2) model.post3x3 if it's a Conv2d
      3) the last Conv2d inside model.backbone
      4) last Conv2d anywhere in the model
    """
    if hasattr(model, "cam_target") and isinstance(model.cam_target, nn.Module):
        return model.cam_target

    if hasattr(model, "post3x3") and isinstance(model.post3x3, nn.Conv2d):
        return model.post3x3

    if hasattr(model, "backbone"):
        last = None
        for _, m in model.backbone.named_modules():
            if isinstance(m, nn.Conv2d):
                last = m
        if last is not None:
            return last

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
    target_layer: Optional[nn.Module] = None,
    class_idx: int = 1,
    upsample_to: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Runs forward pass and returns a 2D CAM heatmap in [0,1] as float32 tensor.
    Upsamples to 'upsample_to' if provided (H0, W0).
    """
    model.eval()

    if target_layer is None:
        target_layer = discover_target_layer(model)

    cam_extractor = build_cam_extractor(model, target_layer, method=method)
    scores = model(input_tensor.float())  # (1,C)

    # TorchCAM API: pass the index and the scores; returns list of CAMs for each target layer
    cams = cam_extractor(class_idx, scores)
    cam = cams[0] if isinstance(cams, (list, tuple)) else cams  # (h,w)

    # Normalize to [0,1]
    cam = cam - cam.min()
    cam = cam / cam.max().clamp(min=1e-8)

    if upsample_to is not None:
        H0, W0 = int(upsample_to[0]), int(upsample_to[1])
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(H0, W0), mode="bilinear", align_corners=False)
        cam = cam.squeeze(0).squeeze(0)

    return cam.cpu().float()
