# src/cam_utils.py
from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# torchcam provides GradCAM & SmoothGradCAMpp
from torchcam.methods import GradCAM, SmoothGradCAMpp


# --------- target-layer resolution (works with our PneumoniaModel) ---------
def discover_target_layer(model: nn.Module) -> nn.Module:
    """
    Priority order:
      1) model.cam_target if present
      2) model.post3x3 if it is a Conv2d (some backbones get an extra 3x3 conv)
      3) last Conv2d found in model.backbone
    """
    # 1) Explicit CAM tap
    if hasattr(model, "cam_target") and isinstance(model.cam_target, nn.Module):
        return model.cam_target

    # 2) Post 3x3 conv
    if hasattr(model, "post3x3") and isinstance(model.post3x3, nn.Conv2d):
        return model.post3x3

    # 3) Last Conv2d inside backbone
    if hasattr(model, "backbone"):
        last_conv = None
        for _, m in model.backbone.named_modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is not None:
            return last_conv

    # 4) Fallback: search whole model
    last_conv = None
    for _, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("No Conv2d found to use as a CAM target layer.")
    return last_conv


# --------- builders for torchcam extractors ---------
def build_cam_extractor(model: nn.Module, target_layer: nn.Module, method: str = "gradcam"):
    m = method.lower().strip()
    if m in ("gradcam", "grad-cam"):
        return GradCAM(model, target_layer=target_layer)
    elif m in ("gradcam++", "smoothgradcampp", "smooth-grad-cam++"):
        return SmoothGradCAMpp(model, target_layer=target_layer)
    else:
        raise ValueError(f"Unsupported CAM method for torchcam: {method}. "
                         f"Use 'gradcam' or 'gradcam++'.")


# --------- main API used by the app ---------
@torch.no_grad()
def compute_cam_map(
    model: nn.Module,
    input_tensor: torch.Tensor,     # (1,3,H,W) preprocessed
    method: str = "gradcam",
    target_layer: Optional[nn.Module] = None,
    class_idx: int = 1,             # abnormal class index
    upsample_to: Optional[Tuple[int,int]] = None,  # (H0,W0) original image size
) -> torch.Tensor:
    """
    Runs forward pass, extracts a CAM (H,W), returns float32 map in [0,1].
    - Uses torchcam internals; no manual backward/grad plumbing required.
    - Returns a single-channel heatmap matching 'upsample_to' if provided.
    """
    model.eval()

    if target_layer is None:
        target_layer = discover_target_layer(model)

    cam_extractor = build_cam_extractor(model, target_layer, method=method)

    # forward pass (no autocast here; CPU on Streamlit Cloud)
    scores = model(input_tensor.float())  # (1,C)

    # torchcam API: returns a list of CAMs (one per target layer); we choose first
    # It internally does backward on scores[:, class_idx]
    cams = cam_extractor(class_idx, scores)
    if isinstance(cams, (list, tuple)):
        cam = cams[0]
    else:
        cam = cams

    # cam is 2D (H,W) tensor; normalize to [0,1]
    cam = cam.detach()
    cam = cam - cam.min()
    denom = cam.max().clamp(min=1e-8)
    cam = cam / denom

    # upsample if needed
    if upsample_to is not None:
        H0, W0 = int(upsample_to[0]), int(upsample_to[1])
        cam = cam.unsqueeze(0).unsqueeze(0)           # (1,1,h,w)
        cam = F.interpolate(cam, size=(H0, W0), mode="bilinear", align_corners=False)
        cam = cam.squeeze(0).squeeze(0).contiguous()  # (H0,W0)

    return cam.cpu().float()
