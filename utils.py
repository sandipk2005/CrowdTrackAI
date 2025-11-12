import cv2
import numpy as np
from config import MAX_PEOPLE
import os

def draw_tracks(frame, tracks):
    """Draw bounding boxes for detections."""
    for i, det in enumerate(tracks):
        bbox = det[0]
        if len(bbox) != 4:
            continue
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'ID:{i+1}', (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    return frame


def generate_heatmap(frame, detections):
    """Generate colored heatmap from detections."""
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
    for det in detections:
        x, y, w, h = map(int, det[0])
        cx, cy = x + w // 2, y + h // 2
        if 0 <= cx < frame.shape[1] and 0 <= cy < frame.shape[0]:
            heatmap[cy, cx] += 1

    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    if np.max(heatmap) > 0:
        heatmap = np.uint8(255 * heatmap / np.max(heatmap))
    return cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)


def overcrowding_alert(count):
    """Return alert message if crowd exceeds limit."""
    if count > MAX_PEOPLE:
        return f"⚠️ Overcrowding detected! ({count})"
    return None


def save_video(frames, output_path, fps=30):
    """Save list of frames to MP4."""
    if not frames:
        return
    h, w, _ = frames[0].shape
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()
