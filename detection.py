import cv2
import numpy as np
from ultralytics import YOLO
from config import YOLO_MODELS

# Global variables to verify model loading
_model = None

def load_yolo_model(model_name):
    """
    Loads the YOLO model globally.
    """
    global _model
    try:
        print(f"🔄 Loading Model: {model_name}...")
        _model = YOLO(model_name)
        print(f"✅ Model Loaded Successfully: {model_name}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        # Fallback
        _model = YOLO('yolov8n.pt')

def detect_people(frame, is_video=False, conf_threshold=0.25, iou_threshold=0.45, max_detections=1000):
    """
    Simplified detection logic guaranteed to return results if model works.
    """
    global _model
    
    # Safety Check: If model isn't loaded, load the default
    if _model is None:
        load_yolo_model('yolov8m-visdrone.pt')

    height, width = frame.shape[:2]
    
    # 1. Run Prediction on Full Frame
    # NOTE: We removed 'classes=0' so it detects EVERYTHING (People, Cars, etc.)
    # This ensures we see if the model is working at all.
    results = _model.predict(
        frame, 
        conf=conf_threshold, 
        iou=iou_threshold,
        verbose=False
    )
    
    detections = []
    
    # 2. Extract Boxes
    for result in results:
        for box in result.boxes:
            # Get coordinates [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get confidence and class
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # VISDRONE SPECIFIC CLASSES:
            # 0 = Pedestrian, 1 = People
            # We only keep class 0 and 1 for people counting
            if cls in [0, 1]: 
                detections.append([[x1, y1, x2 - x1, y2 - y1], conf, cls])

    # 3. Sort by confidence
    detections = sorted(detections, key=lambda x: x[1], reverse=True)[:max_detections]

    # 4. Format Output for App.py
    # Returns: detections_list, count, average_confidence
    
    formatted_detections = []
    total_conf = 0
    
    if len(detections) > 0:
        for det in detections:
            bbox = det[0] # [x, y, w, h]
            conf = det[1]
            cls = det[2]
            
            # App logic expects flat list for video sometimes, 
            # but to be safe we return consistent list format
            formatted_detections.append([bbox, conf, cls])
            total_conf += conf
            
        avg_conf = (total_conf / len(detections)) * 100
        return formatted_detections, len(formatted_detections), round(avg_conf, 1)
    
    return [], 0, 0.0