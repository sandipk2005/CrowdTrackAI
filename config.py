"""
CrowdTrackAI – Central Configuration File
------------------------------------------
This file stores all important settings such as model paths, feature toggles,
alert parameters, and default folder locations.

Modify these values as per your project needs.
"""

# -----------------------------------------------------------
#  YOLO Model Options
# -----------------------------------------------------------

YOLO_MODELS = {
    "small": "yolov8n.pt",   # ⚡ Fast inference (recommended for CPU / Streamlit Cloud)
    "large": "yolov8l.pt"    # 🎯 High accuracy (use only on GPU machines)
}

# -----------------------------------------------------------
#  Alert / Safety Settings
# -----------------------------------------------------------

MAX_PEOPLE = 5000         # 🚷 Threshold for overcrowding alert

# Path to alert sound (place alert.mp3 in root directory)
ALERT_SOUND = "alert.mp3"


# -----------------------------------------------------------
#  Feature Toggles (Enable / Disable)
# -----------------------------------------------------------

ENABLE_FPS_DISPLAY = True        # Show real-time FPS
ENABLE_DENSITY_METER = True      # Show crowd density indicator
ENABLE_HEATMAP = False           # Show live heatmap (CPU heavy)
ENABLE_OVERCROWD_ALERT = True    # Enable audio / visual alerts


# -----------------------------------------------------------
#  Default Paths
# -----------------------------------------------------------

DEFAULT_OUTPUT_DIR = "output"    # Folder to save results (videos, images)
DEFAULT_LOG_DIR = "logs"         # Folder to save logs

