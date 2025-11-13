# utils.py – Helper utilities for CrowdTrackAI

import cv2
import numpy as np
import os
from config import MAX_PEOPLE


def draw_tracks(frame, detections):
    """Draw bounding boxes for detected people."""
    for idx, det in enumerate(detections):
        bbox = det[0]
        if len(bbox) != 4:
            continue

        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{idx+1}",
                    (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 2)

    return frame


def generate_heatmap(frame, detections):
    """Generate a visual heatmap from detection positions."""
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

    for det in detections:
        x, y, w, h = map(int, det[0])
        cx = x + w // 2
        cy = y + h // 2

        if 0 <= cx < frame.shape[1] and 0 <= cy < frame.shape[0]:
            heatmap[cy][cx] += 1

    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)

    if np.max(heatmap) > 0:
        heatmap = (heatmap / np.max(heatmap)) * 255

    heatmap = heatmap.astype(np.uint8)
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return colored


def overcrowding_alert(count):
    """Returns warning string if crowd exceeds limit."""
    if count > MAX_PEOPLE:
        return f"⚠️ Overcrowding detected! Currently: {count} people"
    return None


def save_video(frames, output_path, fps=30):
    """Save a list of frames to MP4 video."""
    if not frames:
        return

    h, w, _ = frames[0].shape
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    for frame in frames:
        writer.write(frame)

    writer.release()
    print(f"💾 Saved video: {output_path}")
