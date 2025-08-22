from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import pytorch_lightning as pl
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

# ---------- helpers for CAM layer discovery ----------
def _is_conv_k_gt1(m: nn.Module) -> bool:
    if not isinstance(m, nn.Conv2d):
        return False
    k = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
    return max(k) > 1

def discover_last_conv_kgt1(module: nn.Module) -> Optional[nn.Conv2d]:
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d) and _is_conv_k_gt1(m):
            last = m
    return last

def discover_any_last_conv(module: nn.Module) -> Optional[nn.Conv2d]:
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last

class PneumoniaModel(pl.LightningModule):
    """
    Minimal inference/training model compatible with your checkpoints.
    - Backbones from timm, features_only=True (returns list of feature maps)
    - Inserts a 3x3 conv 'post3x3' if last stage does not end with k>1 conv
    - Head: GAP -> Dropout -> Linear(num_classes)
    - Exposes self.cam_target (conv to use for Grad-CAM)
    """
    def __init__(self, h: Dict):
        super().__init__()
        self.h = h
        self.save_hyperparameters('h')

        name        = str(h['model'])
        img_size    = int(h.get('img_size', 224))
        drop_rate   = float(h.get('dropout', 0.3))
        num_classes = int(h.get('num_classes', 2))
        self.num_classes = num_classes

        # ---------------- backbone ----------------
        if name in ['dpn68', 'dpn68_new', 'dpn68.mx_in1k']:
            timm_name = 'dpn68.mx_in1k'
        elif name in ['hrnet32', 'hrnet_w32.ms_in1k', 'hrnet32_new']:
            timm_name = 'hrnet_w32.ms_in1k'
        elif name in ['coatnet_0_rw_224.sw_in1k', 'coatnet_nano_rw_224.sw_in1k',
                      'maxvit_small_tf_224.in1k', 'maxvit_tiny_tf_224.in1k']:
            timm_name = name
        else:
            # fallbacks (you can extend)
            timm_name = name

        base = timm.create_model(timm_name, pretrained=True, features_only=True)
        self.backbone = base
        feat_dim = base.feature_info[-1]['num_chs']

        # -------------- ensure 3x3 conv before GAP --------------
        # If no Conv2d with k>1 exists near the end, add a 3x3 conv.
        # This improves CAM spatial localization for HRNet/CoAtNet/DPN.
        candid = discover_last_conv_kgt1(self.backbone)
        needs_3x3 = candid is None
        self.post3x3 = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.SiLU(inplace=True),
        ) if needs_3x3 else nn.Identity()

        # -------------- head --------------
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, num_classes),
        )

        # -------------- metrics --------------
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.val_acc   = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.train_f1  = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1    = F1Score(task="multiclass", num_classes=num_classes, average="macro")

        # -------------- cam target --------------
        # Prefer the inserted 3x3 if present, else last conv with k>1, else any last conv.
        if isinstance(self.post3x3, nn.Sequential) and isinstance(self.post3x3[0], nn.Conv2d):
            self.cam_target = self.post3x3[0]
        else:
            self.cam_target = discover_last_conv_kgt1(self.backbone) or discover_any_last_conv(self.backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        feats = self.post3x3(feats)
        logits = self.head(feats)
        return logits

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
