# src/yolo_onnx.py
from __future__ import annotations
from typing import Tuple, List
import numpy as np
import onnxruntime as ort
import cv2

def letterbox(
    img: np.ndarray,
    new_shape: int | Tuple[int, int] = 640,
    color=(114, 114, 114),
    auto=False,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    """
    Resize + pad to meet stride-multiple constraints like YOLOv8.
    Returns: img, ratio, (dw, dh)
    """
    shape = img.shape[:2]  # current shape [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down
        r = min(r, 1.0)

    # Compute padding
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # width, height
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        r = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    ratio = (r, r)
    return img, ratio, (left, top)

def _nms(boxes, scores, iou_thr=0.75):
    """Basic NMS, expects boxes [N,4] in xyxy, scores [N]."""
    if boxes.size == 0:
        return np.empty((0,), dtype=int)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ov = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        inds = np.where(ov <= iou_thr)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=int)

class YOLOOnnx:
    def __init__(self, onnx_path: str, providers: List[str] | None = None, input_size: int = 640):
        self.session = ort.InferenceSession(
            onnx_path,
            providers=providers or ["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = int(input_size)

    def _preprocess(self, img_bgr: np.ndarray):
        lb, ratio, pad = letterbox(img_bgr, new_shape=self.input_size, stride=32, auto=False)
        img = lb.transpose(2, 0, 1)  # HWC->CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        if img.ndim == 3:
            img = img[None]  # add batch
        return img, ratio, pad, lb.shape[:2]  # (bh,bw)

    def _postprocess(self, preds: np.ndarray, origin_shape, ratio, pad, conf_thr=0.25, iou_thr=0.75):
        """
        Assumes ONNX outputs [batch, num, 6]: [x, y, w, h, conf, cls] OR [x1,y1,x2,y2, conf, cls].
        We’ll detect format by simple heuristic.
        """
        H0, W0 = origin_shape
        pred = preds[0]
        if pred.size == 0:
            return []

        if pred.shape[1] < 6:
            raise RuntimeError("Unexpected YOLO ONNX output shape")

        boxes = pred[:, :4].copy()
        scores = pred[:, 4].copy()
        # xywh or xyxy?
        # If x2<x1 anywhere, we assume xywh
        assume_xywh = np.any(boxes[:, 2] < boxes[:, 0])
        if assume_xywh:
            # xywh -> xyxy
            boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
            boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        # Filter by confidence
        m = scores >= float(conf_thr)
        boxes, scores = boxes[m], scores[m]
        if boxes.size == 0:
            return []

        # NMS
        keep = _nms(boxes, scores, iou_thr=iou_thr)
        boxes, scores = boxes[keep], scores[keep]

        # Map back to original image: undo padding, then divide by ratio
        # boxes are in letterboxed image coords
        (padw, padh) = pad
        boxes[:, [0, 2]] -= padw
        boxes[:, [1, 3]] -= padh
        boxes /= ratio[0]  # same for x and y

        # Clip to original
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W0 - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H0 - 1)

        # Return: list of (x1,y1,x2,y2,score)
        return [(float(x1), float(y1), float(x2), float(y2), float(s)) for (x1, y1, x2, y2), s in zip(boxes, scores)]

    def predict(self, img_bgr: np.ndarray, conf=0.25, iou=0.75):
        H0, W0 = img_bgr.shape[:2]
        inp, ratio, pad, _ = self._preprocess(img_bgr)
        ort_outs = self.session.run(None, {self.input_name: inp})
        # support either [outputs] or multiple; choose the first with 3 dims
        outs = None
        for o in ort_outs:
            if o.ndim == 3:
                outs = o
                break
        if outs is None:
            outs = ort_outs[0]
        return self._postprocess(outs, (H0, W0), ratio, pad, conf_thr=conf, iou_thr=iou)
