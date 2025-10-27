# detection.py
import cv2
from ultralytics import YOLO

# Load YOLOv8 model (keep the file in same folder or update path)
model = YOLO("yolov8n.pt")

def detect_people(frame):
    """Detect only persons (class 0)"""
    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model(frame_rgb)
    detections = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        # ✅ Only detect 'person' class with confidence > 0.3
        if cls_id == 0 and conf > 0.3:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            detections.append([[x1, y1, w, h], conf, "person"])

    print(f"🧠 Detected: {len(detections)} people")  # Debug log
    return detections
