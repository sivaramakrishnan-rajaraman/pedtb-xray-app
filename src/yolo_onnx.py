# src/yolo_onnx.py
from __future__ import annotations
from typing import Tuple, List, Optional
import numpy as np
import onnxruntime as ort
import cv2

def letterbox(
    img: np.ndarray,
    new_shape: int = 640,
    color: Tuple[int,int,int] = (114,114,114),
    stride: int = 32
):
    """
    Resize + pad to meet stride-multiple constraints.
    Returns: image, scale (r), pad (pad_w, pad_h)
    """
    shape = img.shape[:2]  # H, W
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)

def scale_boxes_back(xyxy: np.ndarray, r: float, pad: Tuple[int,int], orig_shape: Tuple[int,int]):
    """
    Map boxes from letterboxed space back to original image space.
    xyxy: (N,4) in resized+pad space
    """
    x1 = (xyxy[:,0] - pad[0]) / r
    y1 = (xyxy[:,1] - pad[1]) / r
    x2 = (xyxy[:,2] - pad[0]) / r
    y2 = (xyxy[:,3] - pad[1]) / r
    boxes = np.stack([x1,y1,x2,y2], axis=1)
    # clip
    h, w = orig_shape
    boxes[:, [0,2]] = boxes[:, [0,2]].clip(0, w-1)
    boxes[:, [1,3]] = boxes[:, [1,3]].clip(0, h-1)
    return boxes

class YOLOOnnx:
    def __init__(self, onnx_path: str, providers: Optional[List[str]] = None, img_size: int = 640):
        self.session = ort.InferenceSession(
            onnx_path,
            providers=providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.img_size = int(img_size)

        # Find input/output names
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def detect(self, bgr: np.ndarray, conf_thres: float = 0.25, iou_thres: float = 0.75):
        """
        Returns detections in ORIGINAL pixel coords:
          boxes_xyxy (N,4), scores (N,), classes (N,)
        """
        H0, W0 = bgr.shape[:2]
        img, r, pad = letterbox(bgr, new_shape=self.img_size)
        img = img[:, :, ::-1]  # BGR->RGB
        img = img.transpose(2, 0, 1)  # HWC->CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        img = img[None]  # (1,3,h,w)

        preds = self.session.run([self.output_name], {self.input_name: img})[0]
        # Expect (1, N, 6) with N post-NMS rows
        if preds.ndim == 3 and preds.shape[-1] >= 6:
            det = preds[0]
            xyxy, scores, cls = det[:, :4], det[:, 4], det[:, 5].astype(np.int32)
            # filter conf
            keep = scores >= float(conf_thres)
            xyxy, scores, cls = xyxy[keep], scores[keep], cls[keep]
            # map back
            xyxy = scale_boxes_back(xyxy, r, pad, (H0, W0))
            return xyxy, scores, cls

        # If your ONNX is raw logits, you'd implement decode+NMS here.
        raise RuntimeError("Unexpected ONNX output shape. Export with nms=True.")
