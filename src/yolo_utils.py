from __future__ import annotations
from typing import Tuple, Optional, List
import numpy as np
import cv2
from ultralytics import YOLO

def detect_lungs(
    yolo: YOLO,
    img_bgr: np.ndarray,
    conf: float = 0.25,
    iou: float = 0.75,
) -> Optional[Tuple[int,int,int,int]]:
    """
    Run YOLO single-class detector on a BGR image and return the best bbox (x1,y1,x2,y2).
    Returns None if no boxes.
    """
    # YOLO expects RGB ndarray or path. We'll pass RGB directly.
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = yolo.predict(img_rgb, conf=conf, iou=iou, verbose=False)
    if not res or len(res[0].boxes) == 0:
        return None
    # take the highest-score box
    boxes = res[0].boxes
    scores = boxes.conf.cpu().numpy()
    best = int(scores.argmax())
    xyxy = boxes.xyxy[best].cpu().numpy().astype(int)
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    # clip to image bounds
    h, w = img_bgr.shape[:2]
    x1 = max(0, min(w-1, x1)); x2 = max(0, min(w-1, x2))
    y1 = max(0, min(h-1, y1)); y2 = max(0, min(h-1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

def crop_with_box(img_bgr: np.ndarray, box: Tuple[int,int,int,int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return img_bgr[y1:y2, x1:x2].copy()

def draw_box(img_bgr: np.ndarray, box: Tuple[int,int,int,int], color=(0,255,0), thickness=2) -> np.ndarray:
    x1, y1, x2, y2 = box
    out = img_bgr.copy()
    cv2.rectangle(out, (x1,y1), (x2,y2), color, thickness)
    return out
