# src/yolo_onnx.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import onnxruntime as ort
from PIL import Image

def _letterbox(im: Image.Image, new_size: int = 640, color=(114,114,114)) -> Tuple[np.ndarray, float, Tuple[int,int]]:
    """Resize + pad to square new_size while keeping aspect ratio. Returns CHW float32 [0,1]."""
    w, h = im.size
    scale = min(new_size / w, new_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    im_resized = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (new_size, new_size), color)
    pad_w, pad_h = (new_size - nw) // 2, (new_size - nh) // 2
    canvas.paste(im_resized, (pad_w, pad_h))
    arr = np.asarray(canvas).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # CHW
    return arr, scale, (pad_w, pad_h)

def _xywh2xyxy(xywh: np.ndarray) -> np.ndarray:
    x, y, w, h = xywh.T
    return np.vstack((x - w/2, y - h/2, x + w/2, y + h/2)).T

def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    """Greedy NMS."""
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = _iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_thres]
    return keep

def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:,0])
    y1 = np.maximum(box[1], boxes[:,1])
    x2 = np.minimum(box[2], boxes[:,2])
    y2 = np.minimum(box[3], boxes[:,3])
    inter = np.clip(x2-x1, a_min=0, a_max=None) * np.clip(y2-y1, a_min=0, a_max=None)
    a = (box[2]-box[0]) * (box[3]-box[1])
    b = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    return inter / (a + b - inter + 1e-6)

class YOLOOnnx:
    """Minimal YOLOv8-like ONNX runner for single-class detection."""
    def __init__(self, onnx_path: str, providers: List[str] | None = None):
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, pil_image: Image.Image, conf_thres: float = 0.25, iou_thres: float = 0.75, img_size: int = 640) -> List[Tuple[float,float,float,float,float]]:
        """
        Returns list of detections as (x1, y1, x2, y2, score) in ORIGINAL image coords.
        """
        w0, h0 = pil_image.size
        chw, scale, (padw, padh) = _letterbox(pil_image, new_size=img_size)
        inp = np.expand_dims(chw, 0)  # 1x3xHxW

        pred = self.session.run([self.output_name], {self.input_name: inp})[0]
        # Shape can be (1, N, C) or (1, C, N); make it (N, C)
        if pred.ndim != 3:
            raise RuntimeError(f"Unexpected ONNX output shape: {pred.shape}")
        if pred.shape[1] < pred.shape[2]:
            pred = pred[0]            # (N, C)
        else:
            pred = pred[0].T          # (N, C)

        # Expect [x, y, w, h, obj, cls...] ; single class → cls shape (N,1)
        xywh = pred[:, 0:4]
        obj  = pred[:, 4]
        if pred.shape[1] >= 6:
            cls_prob = pred[:, 5:]
            if cls_prob.shape[1] == 0:
                cls_conf = np.ones_like(obj)
            else:
                cls_conf = cls_prob.max(axis=1)
            score = obj * cls_conf
        else:
            score = obj

        mask = score >= conf_thres
        if not np.any(mask):
            return []

        xywh = xywh[mask]
        score = score[mask]

        # Convert to padded-canvas xyxy
        xyxy = _xywh2xyxy(xywh)
        # Undo padding/scale -> original image coords
        xyxy[:, [0,2]] -= padw
        xyxy[:, [1,3]] -= padh
        xyxy /= scale

        # Clip to image bounds
        xyxy[:, [0,2]] = np.clip(xyxy[:, [0,2]], 0, w0)
        xyxy[:, [1,3]] = np.clip(xyxy[:, [1,3]], 0, h0)

        # NMS
        keep = _nms(xyxy, score, iou_thres=iou_thres)
        dets = [(float(x1), float(y1), float(x2), float(y2), float(s)) for (x1,y1,x2,y2), s in zip(xyxy[keep], score[keep])]
        return dets
