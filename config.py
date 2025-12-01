# config.py
# ✅ YOLO model paths
YOLO_MODELS = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt", 
    "yolov8m": "yolov8m.pt",
    "yolov8l": "yolov8l.pt",
    "yolov8x": "yolov8x.pt",
    "pose_small": "models/yolov8s-pose.pt",
    "pose_large": "models/yolov8x-pose.pt"
}


# 🚷 Safety limits
MAX_PEOPLE = 5000  # legacy/fallback crowd threshold before alert

# 👇 New: default range slider values (min, max)
RANGE_DEFAULTS = {
    "min": 10,
    "max": 1000
}

# ⚙️ Feature Toggles
ENABLE_FPS_DISPLAY = True
ENABLE_DENSITY_METER = True
ENABLE_HEATMAP = False
ENABLE_OVERCROWD_ALERT = True

# 🔊 Alert sound
ALERT_SOUND = "alert.mp3"  # Make sure alert.mp3 is placed in project root

# 📦 Default paths
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_LOG_DIR = "logs"