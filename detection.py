# detection.py
from ultralytics import YOLO
from config import YOLO_MODEL

# Load YOLOv8 model
model = YOLO(YOLO_MODEL)

def detect_people(frame):
    results = model(frame)
    detections = []
    for r in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, r)
        detections.append([x1, y1, x2-x1, y2-y1, 1.0])
    return detections
