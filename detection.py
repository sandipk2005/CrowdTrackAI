# detection.py
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect_people(frame):
    """Detect only persons (class 0)"""
    results = model(frame)
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id == 0:  # 0 = person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append([[x1, y1, x2 - x1, y2 - y1], conf, "person"])
    return detections
