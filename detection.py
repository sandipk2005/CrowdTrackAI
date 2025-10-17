# detection.py
from ultralytics import YOLO
import cv2

MODEL_PATH = "weights/yolov8n.pt"  # change if your model name differs
model = YOLO(MODEL_PATH)

def detect_people(frame, conf=0.3):
    if frame is None:
        return []
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img, imgsz=640, conf=conf, verbose=False)
    dets = []
    for r in results:
        for box, cls, score in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            if int(cls) != 0:  # keep only person class
                continue
            x1, y1, x2, y2 = map(int, box.tolist())
            w, h = x2 - x1, y2 - y1
            dets.append([x1, y1, w, h, float(score)])
    return dets
