# src/pneumonia_model.py
from __future__ import annotations
from typing import Dict, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models
from torchmetrics.classification import Accuracy, F1Score

h = {
    "model": "dpn68_new",
    "img_size": 224, 
    "batch_size": 64,
    "num_workers": 2,
    "dropout": 0.3,
    "num_classes": 2,
    "pin_memory": True,
    "lr": 5e-5,
    "max_epochs": 64,
    "patience": 10,
    "balance": True,
    }

def _map_name_timm(name: str) -> str:
    # Allow *_new aliases
    remap = {
        "resnet18": "resnet18",
        "inceptionv3": "inception_v3",
        "xception": "xception",
        "resnet34": "resnet34",
        "resnet50": "resnet50",
        "densenet121": "densenet121",
        "efficientnet_b0": "efficientnet_b0",
        "convnext_nano": "convnext_nano.in12k",
        "convnext_small": "convnext_small",
        "dpn68": "dpn68.mx_in1k",
        "dpn68_new": "dpn68.mx_in1k",
        "hrnet32": "hrnet_w32.ms_in1k",
        "hrnet32_new": "hrnet_w32.ms_in1k",
        "efficientnetv2s": "tf_efficientnetv2_s.in21k",

        # Hybrid / transformer-ish
        "coatnet_0_rw_224.sw_in1k": "coatnet_0_rw_224.sw_in1k",
        "coatnet_nano_rw_224.sw_in1k": "coatnet_nano_rw_224.sw_in1k",
        "maxvit_small_tf_224.in1k": "maxvit_small_tf_224.in1k",
        "maxvit_tiny_tf_224.in1k": "maxvit_tiny_tf_224.in1k",

        # ViTs (features_only produces spatial maps)
        "vit_base": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        "deit3_small": "deit3_small_patch16_224.fb_in22k_ft_in1k",
        "beit3_base": "beitv2_base_patch16_224.in1k_ft_in22k_in1k",
    }
    if name not in remap:
        raise ValueError(f"Unsupported model name '{name}'.")
    return remap[name]


def _discover_last_conv_k(module: nn.Module):
    last_conv = None
    k = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
            k = m.kernel_size
    return last_conv, k


class PneumoniaModel(pl.LightningModule):
    """
    Inference-friendly LightningModule:
      - timm backbones with features_only=True
      - optional post3x3 conv when backbone doesn't end with 3x3
      - unified head: GAP -> Dropout -> Linear
    """
    def __init__(self, h: Dict):
        super().__init__()
        self.h = h
        self.save_hyperparameters('h')

        name = str(h["model"])
        img_size = int(h.get("img_size", 224))
        drop_rate = float(h.get("dropout", 0.3))
        num_classes = int(h.get("num_classes", 2))

        # VGG (torchvision)
        if name in ["vgg11", "vgg13", "vgg16", "vgg19"]:
            weights = getattr(models, f"{name.upper()}_Weights").IMAGENET1K_V1
            vgg = getattr(models, name)(weights=weights)
            self.backbone = vgg.features
            feat_dim = 512
            self.post3x3 = nn.Identity()

        else:
            timm_id = _map_name_timm(name)
            base = timm.create_model(
                timm_id, pretrained=True, features_only=True,
                img_size=img_size, pretrained_strict=False
            )
            self.backbone = base
            feat_dim = base.feature_info[-1]["num_chs"]

            # Hybrid or conv nets where 3x3 end is not guaranteed
            last_conv, k = _discover_last_conv_k(self.backbone)
            need_3x3 = (k is None) or (k != (3, 3))
            self.post3x3 = (
                nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False)
                if need_3x3 else nn.Identity()
            )
            # A BN+act is optional; for CAM purity we keep it linear here.

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, num_classes),
        )

        # Optional attribute used by CAM utilities
        self.cam_target = self.post3x3 if isinstance(self.post3x3, nn.Conv2d) else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        feats = self.post3x3(feats)
        logits = self.head(feats)
        return logits

    # (No training/val logic needed for inference in the app.)
    # The following are used only if you also train in this app; harmless at inference time.
    def _nll_from_onehot_logits(self, logits, labels_oh):
        logp = F.log_softmax(logits, dim=1)
        return -(labels_oh * logp).sum(dim=1).mean()

    def training_step(self, batch, batch_idx):
        x, y = batch
        if y.ndim == 1:
            y_oh = F.one_hot(y, num_classes=self.num_classes).float()
        else:
            y_oh = y.float()
        logits = self(x)
        loss = self._nll_from_onehot_logits(logits, y_oh)
        preds = logits.argmax(dim=1)
        tgt = y_oh.argmax(dim=1)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc",  self.train_acc(preds, tgt), prog_bar=True)
        self.log("train_f1",   self.train_f1(preds, tgt),  prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if y.ndim == 1:
            y_oh = F.one_hot(y, num_classes=self.num_classes).float()
        else:
            y_oh = y.float()
        logits = self(x)
        loss = self._nll_from_onehot_logits(logits, y_oh)
        preds = logits.argmax(dim=1)
        tgt = y_oh.argmax(dim=1)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc",  self.val_acc(preds, tgt), prog_bar=True)
        self.log("val_f1",   self.val_f1(preds, tgt),  prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.h.get("lr", 5e-5), weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.h.get("max_epochs", 10))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}
