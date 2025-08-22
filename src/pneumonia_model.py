# src/pneumonia_model.py
from __future__ import annotations
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm

# ---------- small helpers ----------

def _find_last_conv(module: nn.Module) -> Optional[nn.Conv2d]:
    last = None
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last

def _last_conv_kernel_size(module: nn.Module) -> Optional[Tuple[int, int]]:
    last = _find_last_conv(module)
    if last is None:
        return None
    # kernel_size is a tuple[int,int]
    return tuple(last.kernel_size) if hasattr(last, "kernel_size") else None

# ---------- Lightning model ----------

class PneumoniaModel(pl.LightningModule):
    """
    Minimal, inference-friendly model for the app:

    - Accepts h: dict with at least {model, img_size, dropout, num_classes, lr, max_epochs}
    - Builds a timm backbone with features_only=True
      * CNN / hybrid: DO NOT pass img_size
      * ViT-like:     pass img_size (some support it for patch embedding shapes)
    - Adds a 3x3 "post3x3" conv if the final conv kernel != 3x3 (helps CAM quality),
      otherwise identity.
    - Head: GAP -> Flatten -> Dropout -> Linear
    - Exposes `self.cam_target` for Grad-CAM (post3x3 if present, else last conv)
    """
    def __init__(self, h: Dict):
        super().__init__()
        self.h = h
        # Save only 'h' to checkpoint (Lightning expects this for load_from_checkpoint)
        self.save_hyperparameters("h")

        name: str        = str(h["model"])
        img_size: int    = int(h.get("img_size", 224))
        drop_rate: float = float(h.get("dropout", 0.3))
        num_classes: int = int(h.get("num_classes", 2))
        self.num_classes = num_classes

        # ---------- name mapping ----------
        # Map your custom names to proper timm model IDs
        cnn_map = {
            # examples; include others you use if needed
            "dpn68_new":     "dpn68.mx_in1k",
            "hrnet32_new":   "hrnet_w32.ms_in1k",
            "resnet18_new":  "resnet18",
            "resnet34_new":  "resnet34",
            "resnet50_new":  "resnet50",
            "densenet121_new": "densenet121",
            "convnext_nano_new":  "convnext_nano.in12k",
            "convnext_small_new": "convnext_small",
            "efficientnet_b0_new": "efficientnet_b0",
            "efficientnetv2s_new": "tf_efficientnetv2_s.in21k",
            "coatnet_0_rw_224.sw_in1k_new": "coatnet_0_rw_224.sw_in1k",
            "coatnet_nano_rw_224.sw_in1k_new": "coatnet_nano_rw_224.sw_in1k",
            # add others you trained under *_new if you need them in the app
        }

        vit_map = {
            # use timm IDs that support features_only and img_size
            "vit_base":    "vit_base_patch16_224.augreg2_in21k_ft_in1k",
            "deit3_small": "deit3_small_patch16_224.fb_in22k_ft_in1k",
            "beit3_base":  "beitv2_base_patch16_224.in1k_ft_in22k_in1k",
        }

        is_vit_like = name in vit_map

        # ---------- build backbone ----------
        if is_vit_like:
            timm_id = vit_map[name]
            # ViT-like: pass img_size, features_only=True
            base = timm.create_model(
                timm_id,
                pretrained=True,
                features_only=True,
                img_size=img_size,
                drop_rate=0.0,
                drop_path_rate=0.0,
                pretrained_strict=False,
            )
        else:
            timm_id = cnn_map.get(name, None)
            if timm_id is None:
                # Fallback: assume the user passed a valid timm id directly in h["model"]
                timm_id = name
            # CNN / hybrid: DO NOT pass img_size (DPN, HRNet, etc. reject it)
            base = timm.create_model(
                timm_id,
                pretrained=True,
                features_only=True,
                pretrained_strict=False,
            )

        self.backbone = base
        # features_only → get channels from feature_info
        feat_dim = int(self.backbone.feature_info[-1]["num_chs"])

        # ---------- add post-3x3 if needed (CNN/hybrid only) ----------
        self.post3x3: nn.Module
        if is_vit_like:
            # ViTs already produce token/patch features; usually a conv here is not meaningful.
            self.post3x3 = nn.Identity()
        else:
            k = _last_conv_kernel_size(self.backbone)
            need_3x3 = (k is None) or (tuple(k) != (3, 3))
            if need_3x3:
                self.post3x3 = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False)
            else:
                self.post3x3 = nn.Identity()

        # ---------- classification head ----------
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, num_classes),
        )

        # ---------- expose CAM target ----------
        # Prefer the 3x3 we added; otherwise use the last conv in the backbone.
        if isinstance(self.post3x3, nn.Conv2d):
            self.cam_target = self.post3x3
        else:
            self.cam_target = _find_last_conv(self.backbone)

    # ----- forward -----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        # Apply post-3x3 conv if it exists
        feats = self.post3x3(feats)
        logits = self.head(feats)
        return logits

    # ----- training bits (unused in app, but harmless) -----
    @staticmethod
    def _nll_from_onehot_logits(logits: torch.Tensor, labels_oh: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        return -(labels_oh * logp).sum(dim=1).mean()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.h.get("lr", 1e-3), weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.h.get("max_epochs", 10))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

