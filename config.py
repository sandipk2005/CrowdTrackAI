# ✅ YOLO model paths
YOLO_MODELS = {
    "small": "yolov8n.pt",    # ⚡ Fast (Speed Mode)
    "medium": "yolov8m.pt",   # 🟠 Medium (Balanced Mode)
    "large": "yolov8l.pt"     # 🎯 Accurate (Accuracy Mode)
}

# 🚷 Safety limits
MAX_PEOPLE = 5000  # crowd threshold before alert 

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
