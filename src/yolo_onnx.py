# src/yolo_onnx.py
from __future__ import annotations
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort

def letterbox_rgb(
    rgb: np.ndarray,
    new_shape: int = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize & pad an RGB image to a square new_shape, preserving aspect ratio.
    Returns:
      - padded RGB (H,W,3) uint8
      - scale r
      - pad (left, top)
    """
    H, W = rgb.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / H, new_shape[1] / W)
    new_unpad = (int(round(W * r)), int(round(H * r)))  # (w,h)

    im = Image.fromarray(rgb)
    if im.size != new_unpad:
        im = im.resize(new_unpad, Image.BILINEAR)

    pad_w = new_shape[1] - new_unpad[0]
    pad_h = new_shape[0] - new_unpad[1]
    left = pad_w // 2
    top  = pad_h // 2

    padded = ImageOps.expand(im, border=(left, top, pad_w - left, pad_h - top), fill=color)
    return np.array(padded, dtype=np.uint8), r, (left, top)

def scale_boxes_back(xyxy: np.ndarray, r: float, pad: Tuple[int,int], orig_shape: Tuple[int,int]):
    """Map boxes from letterboxed space back to original image space."""
    H, W = orig_shape
    x1 = (xyxy[:,0] - pad[0]) / r
    y1 = (xyxy[:,1] - pad[1]) / r
    x2 = (xyxy[:,2] - pad[0]) / r
    y2 = (xyxy[:,3] - pad[1]) / r
    boxes = np.stack([x1,y1,x2,y2], axis=1)
    boxes[:, [0,2]] = boxes[:, [0,2]].clip(0, W - 1)
    boxes[:, [1,3]] = boxes[:, [1,3]].clip(0, H - 1)
    return boxes

class YOLOOnnx:
    """
    Expects an Ultralytics-exported ONNX with NMS, output shape (1, N, 6):
    [x1, y1, x2, y2, score, cls]
    """
    def __init__(self, onnx_path: str, img_size: int = 640):
        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"]  # Streamlit Cloud CPU
        )
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.img_size    = int(img_size)

    def detect(self, rgb: np.ndarray, conf_thres: float = 0.25, iou_thres: float = 0.75):
        H0, W0 = rgb.shape[:2]
        img, r, pad = letterbox_rgb(rgb, new_shape=self.img_size)

        # prepare tensor: (1,3,H,W) float32 in [0,1]
        im = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        preds = self.session.run([self.output_name], {self.input_name: im})[0]
        if preds.ndim != 3 or preds.shape[-1] < 6:
            raise RuntimeError("Unexpected ONNX output. Export with nms=True.")

        det = preds[0]  # (N,6)
        xyxy, scores, cls = det[:, :4], det[:, 4], det[:, 5].astype(np.int32)
        keep = scores >= float(conf_thres)
        xyxy, scores, cls = xyxy[keep], scores[keep], cls[keep]
        if xyxy.shape[0] == 0:
            return xyxy, scores, cls

        xyxy = scale_boxes_back(xyxy, r, pad, (H0, W0))
        return xyxy, scores, cls
