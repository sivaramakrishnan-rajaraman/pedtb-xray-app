# src/pneumonia_model.py
from __future__ import annotations
from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm

# ---------- small helpers ----------
def _find_last_conv(mod: nn.Module) -> Optional[nn.Conv2d]:
    last = None
    for _, m in mod.named_modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last

def _discover_laststage_conv_k(mod: nn.Module) -> Optional[Tuple[int, int]]:
    last = _find_last_conv(mod)
    if last is None:
        return None
    k = last.kernel_size if hasattr(last, "kernel_size") else None
    if isinstance(k, tuple) and len(k) == 2:
        return (int(k[0]), int(k[1]))
    return None

# ---------- the model ----------
class PneumoniaModel(pl.LightningModule):
    """
    Inference-only: builds the same backbone+head topology you trained.
    """
    def __init__(self, h: Dict):
        super().__init__()
        self.save_hyperparameters("h")
        self.h = h

        name         = str(h["model"])
        img_size     = int(h.get("img_size", 224))
        num_classes  = int(h.get("num_classes", 2))
        drop_rate    = float(h.get("dropout", 0.3))

        self.is_transformer = False
        self.post3x3 = nn.Identity()

        # ---- backbones of interest for the app: dpn68_new (timm DPN-68)
        if name in ("dpn68_new", "dpn68"):
            base = timm.create_model("dpn68.mx_in1k", pretrained=True, features_only=True)
            self.backbone = base
            feat_dim = base.feature_info[-1]["num_chs"]
            # ensure last stage ends with a 3x3 conv; if not, add one (you trained like this)
            k = _discover_laststage_conv_k(self.backbone)
            need_3x3 = (k is None) or (k != (3, 3))
            if need_3x3:
                self.post3x3 = nn.Sequential(
                    nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(feat_dim),
                    nn.SiLU(inplace=True),
                )
        else:
            # Generic timm feature extractor if you later switch the model
            base = timm.create_model(name, pretrained=True, features_only=True)
            self.backbone = base
            feat_dim = base.feature_info[-1]["num_chs"]
            k = _discover_laststage_conv_k(self.backbone)
            need_3x3 = (k is None) or (k != (3, 3))
            if need_3x3:
                self.post3x3 = nn.Sequential(
                    nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(feat_dim),
                    nn.SiLU(inplace=True),
                )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, num_classes),
        )

        # Expose a good CAM tap by default
        self.cam_target = _find_last_conv(self.backbone) or _find_last_conv(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        feats = self.post3x3(feats)
        logits = self.head(feats)
        return logits

    # -------- convenience loader with strict fallback --------
    @classmethod
    def load_from_ckpt_auto_strict(cls, ckpt_path: str, h: Dict, map_location="cpu"):
        try:
            m = cls.load_from_checkpoint(ckpt_path, h=h, strict=True, map_location=map_location)
            info = {"strict_used": True, "missing_keys": [], "unexpected_keys": []}
            return m, info
        except Exception as e:
            m = cls.load_from_checkpoint(ckpt_path, h=h, strict=False, map_location=map_location)
            info = {"strict_used": False, "missing_keys": [], "unexpected_keys": []}
            print(f"[load_from_ckpt_auto_strict] strict=True failed: {e}\nUsing strict=False.")
            return m, info
