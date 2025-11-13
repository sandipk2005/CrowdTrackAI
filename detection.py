# detection.py – YOLOv8 People Detection Module

import cv2
import numpy as np
from ultralytics import YOLO
from config import YOLO_MODELS

# Global YOLO model cache
_model = None
_current_model_name = None


def load_yolo_model(model_choice="small"):
    """
    Loads and caches YOLO model on CPU.
    """
    global _model, _current_model_name

    if _model is not None and _current_model_name == model_choice:
        return _model

    model_path = YOLO_MODELS.get(model_choice, "yolov8n.pt")
    _model = YOLO(model_path)   # CPU ONLY
    _current_model_name = model_choice

    print(f"✅ Loaded YOLO model: {model_path}")
    return _model


def detect_people(frame):
    """
    Detect people in a frame using YOLOv8.
    Returns:
        detections: [ [ [x,y,w,h], confidence ], ... ]
        count: number of people detected
        avg_conf: average confidence
    """

    global _model
    if _model is None:
        _model = load_yolo_model()

    results = _model(frame, verbose=False)[0]

    detections = []
    confs = []

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0]) * 100

        # Person class = 0
        if cls == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            detections.append([[x1, y1, w, h], conf])
            confs.append(conf)

    count = len(detections)
    avg_conf = round(sum(confs) / len(confs), 1) if confs else 0.0

    print(f"🧠 Raw: {len(results.boxes)} | Final: {count} | Confidence: {avg_conf}%")

    return detections, count, avg_conf
