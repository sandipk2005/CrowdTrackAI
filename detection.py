from ultralytics import YOLO

# Load local YOLOv8 model
model = YOLO("yolov8n.pt")  # make sure this file exists in your project folder

def detect_people(frame):
    results = model(frame)
    detections = []
    for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
        if int(cls) == 0:  # class 0 = person
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            detections.append([x1, y1, w, h, 1.0])
    return detections
